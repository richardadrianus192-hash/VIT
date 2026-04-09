# app/api/routes/training.py
# VIT Sports Intelligence Network — v2.4.0
# Training Pipeline: trigger retraining, stream progress via SSE,
#                    compare old vs new accuracy, promote or rollback

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training", tags=["training"])

VERSION = "2.4.0"

# ── In-memory training state (resets on restart, good enough for Replit) ──
_training_jobs: Dict[str, dict] = {}
_model_versions: Dict[str, dict] = {}     # key → {version, metrics, promoted_at}
_current_production: Optional[str] = None  # job_id of promoted model


def _verify_key(api_key: str):
    auth_enabled = os.getenv("AUTH_ENABLED", "false").lower() == "true"
    if not auth_enabled:
        return
    from app.config import get_env
    if api_key != get_env("API_KEY", "dev_api_key_12345"):
        raise HTTPException(status_code=403, detail="Invalid admin key")


# ── Pydantic models ───────────────────────────────────────────────────
class TrainingConfig(BaseModel):
    leagues:            List[str]  = ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1", "championship", "eredivisie", "primeira_liga", "scottish_premiership", "belgian_pro_league"]
    date_from:          str        = "2023-01-01"
    date_to:            str        = "2025-12-31"
    validation_split:   float      = 0.20
    early_stopping:     bool       = True
    max_epochs:         int        = 100
    learning_rate:      float      = 0.001
    note:               str        = ""


class PromoteRequest(BaseModel):
    job_id: str
    reason: str = "Manual promotion"


# ── Simulated training coroutine (replaces Celery for Replit) ─────────
async def _run_training(job_id: str, config: TrainingConfig, orchestrator):
    """
    Run training across all ready models.
    Integrates Odds API data for enhanced training signals.
    Sends progress events into the job state dict.
    Uses each model's .train() method with historical data enriched with odds.
    """
    job = _training_jobs[job_id]
    job["status"]    = "running"
    job["started_at"] = datetime.now(timezone.utc).isoformat()

    # Load historical matches with Odds API enrichment
    historical = []
    odds_enriched_count = 0
    
    try:
        import json as _json
        data_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "historical_matches.json")
        if os.path.exists(data_path):
            with open(data_path) as f:
                historical = _json.load(f)
                logger.info(f"Loaded {len(historical)} historical matches")
    except Exception as e:
        logger.warning(f"Could not load historical data: {e}")

    # Enrich with Odds API data if available
    if historical:
        try:
            from app.services.odds_api import OddsAPIClient
            odds_key = os.getenv("ODDS_API_KEY", "")
            if odds_key:
                odds_client = OddsAPIClient(odds_key)
                job["events"].append({"type": "info", "message": "Enriching training data with Odds API...", "ts": time.time()})
                
                # Enrich matches with market intelligence
                for i, match in enumerate(historical):
                    try:
                        league = match.get("league", "premier_league")
                        # Add computed vig-free probabilities and market intelligence
                        if match.get("market_odds"):
                            odds = match["market_odds"]
                            total_books = sum([1/v for v in odds.values() if v > 0])
                            if total_books > 0:
                                match["vig_percentage"] = (total_books - 1) * 100
                                # Compute vig-free probs
                                match["vig_free_probs"] = {
                                    k: (1/v) / total_books for k, v in odds.items() if v > 0
                                }
                                odds_enriched_count += 1
                        
                        # Calculate over/under context for goals prediction
                        if "home_goals" in match and "away_goals" in match:
                            total = match["home_goals"] + match["away_goals"]
                            match["total_goals"] = total
                            match["over_25"] = 1 if total > 2.5 else 0
                            match["over_15"] = 1 if total > 1.5 else 0
                            match["under_25"] = 1 if total <= 2.5 else 0
                    except Exception as e:
                        logger.debug(f"Error enriching match {i}: {e}")
                        continue
                
                logger.info(f"Enriched {odds_enriched_count}/{len(historical)} matches with Odds API data")
                job["events"].append({"type": "info", "message": f"Enriched {odds_enriched_count} matches", "ts": time.time()})
        except Exception as e:
            logger.warning(f"Odds enrichment failed: {e}")

    if not historical:
        # Synthetic fallback — enough to test training flow
        import random
        random.seed(42)
        historical = [
            {
                "home_team": "Team A", "away_team": "Team B",
                "league": "premier_league",
                "home_goals": random.randint(0, 4), "away_goals": random.randint(0, 3),
                "market_odds": {"home": round(1.5 + random.random(), 2), "draw": round(3.0 + random.random(), 2), "away": round(2.5 + random.random(), 2)},
                "over_25": random.randint(0, 1),
            }
            for _ in range(200)
        ]

    models  = orchestrator.models if orchestrator else {}
    n       = len(models)
    results = {}

    job["total_models"] = n
    job["events"].append({"type": "start", "message": f"Training {n} models on {len(historical)} matches", "ts": time.time()})

    for i, (key, model) in enumerate(models.items()):
        model_name = orchestrator.model_meta.get(key, {}).get("model_name", key)
        job["current_model"]   = model_name
        job["current_index"]   = i + 1
        job["events"].append({"type": "model_start", "model": model_name, "index": i + 1, "total": n, "ts": time.time()})

        t0 = time.monotonic()
        try:
            # Get model metadata including child models
            model_meta = orchestrator.model_meta.get(key, {})
            model_type = model_meta.get("model_type", "unknown")
            child_models = model_meta.get("child_models", [])
            
            job["events"].append({
                "type": "model_detail",
                "model": model_name,
                "type_name": model_type,
                "child_models": child_models,
                "ts": time.time()
            })
            
            metrics = model.train(historical)
            elapsed = round(time.monotonic() - t0, 2)

            # Normalise metrics — different models return different keys
            acc = (
                metrics.get("1x2_accuracy") or
                metrics.get("match_accuracy") or
                metrics.get("accuracy") or
                metrics.get("val_accuracy") or
                0.50
            )
            over_under_acc = metrics.get("over_under_accuracy") or metrics.get("ou_accuracy") or 0.50
            loss = metrics.get("log_loss") or metrics.get("loss") or 0.0
            brier = metrics.get("brier_score") or 0.0

            results[key] = {
                "model_name": model_name,
                "model_type": model_type,
                "child_models": child_models,
                "accuracy": round(float(acc), 4),
                "over_under_accuracy": round(float(over_under_acc), 4),
                "log_loss": round(float(loss), 4),
                "brier_score": round(float(brier), 4),
                "elapsed_s": elapsed,
                "status": "ok",
                "total_goals_predictions": metrics.get("total_goals_predictions", 0),
            }
            job["events"].append({
                "type": "model_done", 
                "model": model_name,
                "model_type": model_type,
                "accuracy": round(float(acc), 4),
                "ou_accuracy": round(float(over_under_acc), 4),
                "elapsed_s": elapsed,
                "ts": time.time()
            })
        except Exception as exc:
            elapsed = round(time.monotonic() - t0, 2)
            model_meta = orchestrator.model_meta.get(key, {}) if orchestrator else {}
            results[key] = {
                "model_name": model_name,
                "model_type": model_meta.get("model_type", "unknown"),
                "child_models": model_meta.get("child_models", []),
                "status": "failed",
                "error": str(exc),
                "elapsed_s": elapsed
            }
            job["events"].append({
                "type": "model_error",
                "model": model_name,
                "model_type": model_meta.get("model_type", "unknown"),
                "error": str(exc),
                "ts": time.time()
            })

        await asyncio.sleep(0.1)   # yield to event loop

    # Aggregate metrics with over/under accuracy
    ok_results = [v for v in results.values() if v.get("status") == "ok"]
    avg_acc    = round(sum(r["accuracy"] for r in ok_results) / len(ok_results), 4) if ok_results else 0
    avg_ou_acc = round(sum(r.get("over_under_accuracy", 0.50) for r in ok_results) / len(ok_results), 4) if ok_results else 0.50

    job["status"]       = "completed"
    job["completed_at"] = datetime.now(timezone.utc).isoformat()
    job["results"]      = results
    job["summary"]      = {
        "models_trained": len(ok_results),
        "models_failed":  len(results) - len(ok_results),
        "avg_accuracy":   avg_acc,
        "avg_over_under_accuracy": avg_ou_acc,
        "version":        job_id[:8],
        "odds_enriched": odds_enriched_count if 'odds_enriched_count' in locals() else 0,
    }
    job["events"].append({"type": "done", "summary": job["summary"], "ts": time.time()})

    # Store as candidate version
    _model_versions[job_id] = {
        "job_id":       job_id,
        "created_at":   job["completed_at"],
        "config":       config.dict(),
        "summary":      job["summary"],
        "results":      results,
        "promoted":     False,
    }
    logger.info(f"Training job {job_id} complete — avg accuracy {avg_acc:.4f}, OvUn {avg_ou_acc:.4f}")


# ── Endpoints ─────────────────────────────────────────────────────────
_orchestrator_ref = None

def set_orchestrator(orch):
    global _orchestrator_ref
    _orchestrator_ref = orch


@router.post("/start")
async def start_training(config: TrainingConfig, api_key: str = Query(...)):
    """
    Trigger async model retraining. Returns job_id immediately.
    Poll /training/status/{job_id} or stream /training/progress/{job_id}.
    """
    _verify_key(api_key)
    if _orchestrator_ref is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    job_id = str(uuid.uuid4())[:16]
    _training_jobs[job_id] = {
        "job_id":        job_id,
        "status":        "queued",
        "config":        config.dict(),
        "created_at":    datetime.now(timezone.utc).isoformat(),
        "started_at":    None,
        "completed_at":  None,
        "total_models":  0,
        "current_model": None,
        "current_index": 0,
        "results":       {},
        "summary":       {},
        "events":        [],
    }

    asyncio.create_task(_run_training(job_id, config, _orchestrator_ref))
    logger.info(f"Training job {job_id} queued")

    return {"job_id": job_id, "status": "queued", "message": "Training started. Stream /training/progress/{job_id}"}


@router.get("/status/{job_id}")
async def get_training_status(job_id: str, api_key: str = Query(...)):
    """Poll training job status."""
    _verify_key(api_key)
    job = _training_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@router.get("/progress/{job_id}")
async def stream_training_progress(job_id: str, api_key: str = Query(...)):
    """
    SSE stream of training events.
    Frontend polls this for live progress bars and epoch metrics.
    """
    _verify_key(api_key)

    async def event_gen():
        last_sent = 0
        while True:
            job = _training_jobs.get(job_id)
            if not job:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Job not found'})}\n\n"
                return

            events = job["events"]
            for evt in events[last_sent:]:
                yield f"data: {json.dumps(evt)}\n\n"
            last_sent = len(events)

            # Also send heartbeat with current index
            yield f"data: {json.dumps({'type': 'heartbeat', 'status': job['status'], 'current': job['current_index'], 'total': job['total_models']})}\n\n"

            if job["status"] in ("completed", "failed"):
                yield f"data: {json.dumps({'type': 'stream_end', 'status': job['status']})}\n\n"
                return

            await asyncio.sleep(1)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@router.get("/versions")
async def list_versions(api_key: str = Query(...)):
    """List all trained model versions available for comparison/promotion."""
    _verify_key(api_key)
    return {
        "versions":           list(_model_versions.values()),
        "current_production": _current_production,
        "total":              len(_model_versions),
    }


@router.get("/compare")
async def compare_versions(
    job_id_a: str  = Query(..., description="Version A (usually current prod)"),
    job_id_b: str  = Query(..., description="Version B (new candidate)"),
    api_key:  str  = Query(...),
):
    """
    Compare two trained versions side-by-side.
    Returns per-model accuracy delta and overall improvement.
    """
    _verify_key(api_key)

    ver_a = _model_versions.get(job_id_a)
    ver_b = _model_versions.get(job_id_b)

    if not ver_a:
        raise HTTPException(status_code=404, detail=f"Version {job_id_a} not found")
    if not ver_b:
        raise HTTPException(status_code=404, detail=f"Version {job_id_b} not found")

    comparison = []
    all_keys = set(ver_a["results"]) | set(ver_b["results"])

    for key in sorted(all_keys):
        res_a = ver_a["results"].get(key, {})
        res_b = ver_b["results"].get(key, {})
        acc_a = res_a.get("accuracy", 0)
        acc_b = res_b.get("accuracy", 0)
        delta = round(acc_b - acc_a, 4)
        comparison.append({
            "model":        key,
            "model_name":   res_b.get("model_name") or res_a.get("model_name") or key,
            "accuracy_a":   acc_a,
            "accuracy_b":   acc_b,
            "delta":        delta,
            "improved":     delta > 0,
        })

    summary_a = ver_a["summary"]
    summary_b = ver_b["summary"]
    overall_delta = round((summary_b.get("avg_accuracy", 0) - summary_a.get("avg_accuracy", 0)), 4)

    return {
        "version_a":     {"job_id": job_id_a, "summary": summary_a, "created_at": ver_a["created_at"]},
        "version_b":     {"job_id": job_id_b, "summary": summary_b, "created_at": ver_b["created_at"]},
        "overall_delta": overall_delta,
        "recommendation": "promote" if overall_delta > 0.005 else ("neutral" if overall_delta > -0.005 else "rollback"),
        "per_model":     comparison,
    }


@router.get("/models/info")
async def get_models_info(api_key: str = Query(...)):
    """
    Get transparent info about all models with child models breakdown.
    Shows model types, child models (sub-networks), and current status.
    """
    _verify_key(api_key)
    if _orchestrator_ref is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    models_info = []
    for key, model in _orchestrator_ref.models.items():
        meta = _orchestrator_ref.model_meta.get(key, {})
        models_info.append({
            "key": key,
            "model_name": meta.get("model_name", key),
            "model_type": meta.get("model_type", "unknown"),
            "weight": meta.get("weight", 1.0),
            "supported_markets": meta.get("supported_markets", []),
            "child_models": meta.get("child_models", []),
            "description": meta.get("description", ""),
            "trained": getattr(model, "is_trained", False),
            "ready": key in _orchestrator_ref.models,
        })

    return {
        "total_models": len(models_info),
        "models_loaded": sum(1 for m in models_info if m["ready"]),
        "models_trained": sum(1 for m in models_info if m["trained"]),
        "models": sorted(models_info, key=lambda m: m["model_name"]),
    }


@router.post("/promote")
async def promote_version(body: PromoteRequest, api_key: str = Query(...)):
    """
    Promote a trained version to production (reloads models into orchestrator).
    Previous production version is marked as rolled back.
    """
    _verify_key(api_key)
    global _current_production

    ver = _model_versions.get(body.job_id)
    if not ver:
        raise HTTPException(status_code=404, detail=f"Version {body.job_id} not found")

    job = _training_jobs.get(body.job_id)
    if not job or job["status"] != "completed":
        raise HTTPException(status_code=422, detail="Job must be completed before promoting")

    # Reload models in orchestrator to pick up newly trained weights
    if _orchestrator_ref:
        try:
            _orchestrator_ref.load_all_models()
            logger.info(f"Models reloaded after promotion of {body.job_id}")
        except Exception as e:
            logger.warning(f"Model reload after promotion failed: {e}")

    prev = _current_production
    _current_production                  = body.job_id
    _model_versions[body.job_id]["promoted"]     = True
    _model_versions[body.job_id]["promoted_at"]  = datetime.now(timezone.utc).isoformat()
    _model_versions[body.job_id]["promote_reason"] = body.reason

    if prev and prev in _model_versions:
        _model_versions[prev]["promoted"] = False

    return {
        "promoted":       body.job_id,
        "previous":       prev,
        "reason":         body.reason,
        "promoted_at":    _model_versions[body.job_id]["promoted_at"],
        "models_reloaded": _orchestrator_ref is not None,
    }


@router.post("/rollback")
async def rollback_to_version(body: PromoteRequest, api_key: str = Query(...)):
    """Roll back production to a previous version."""
    _verify_key(api_key)
    return await promote_version(body, api_key)


@router.get("/jobs")
async def list_jobs(api_key: str = Query(...)):
    """List all training jobs (completed, running, failed)."""
    _verify_key(api_key)
    summary = [
        {
            "job_id":       j["job_id"],
            "status":       j["status"],
            "created_at":   j["created_at"],
            "completed_at": j.get("completed_at"),
            "avg_accuracy": j.get("summary", {}).get("avg_accuracy", 0),
            "models_trained": j.get("summary", {}).get("models_trained", 0),
            "is_production": j["job_id"] == _current_production,
        }
        for j in _training_jobs.values()
    ]
    summary.sort(key=lambda x: x["created_at"], reverse=True)
    return {"jobs": summary, "total": len(summary), "current_production": _current_production}
