# main.py — VIT Sports Intelligence Network v3.0.0
# Updated: registers training, analytics, and odds_compare routers

import sys
import os
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

from app.config import get_env
from app.db.database import engine, Base, get_db, _is_sqlite
from app.api.routes import predict, result, history, admin
from app.api.routes import training as training_route
from app.api.routes import analytics as analytics_route
from app.api.routes import odds_compare as odds_route
from app.api.middleware.auth import APIKeyMiddleware
from app.api.middleware.logging import LoggingMiddleware
from app.schemas.schemas import HealthResponse
from app.pipelines.data_loader import DataLoader
from app.services.alerts import TelegramAlert, AlertPriority

# ML Orchestrator
from services.ml_service.models.model_orchestrator import ModelOrchestrator

orchestrator    = None
data_loader     = None
telegram_alerts = None


async def _check_db_connection() -> bool:
    try:
        async for session in get_db():
            await session.execute(select(1))
            return True
    except Exception as e:
        print(f"❌ DB Connection Check Failed: {e}")
        return False


async def fetch_and_predict(competition: str, days_ahead: int = 7):
    global data_loader, orchestrator, telegram_alerts
    if not data_loader or not orchestrator:
        print("❌ Data loader or orchestrator not initialized")
        return
    try:
        print(f"\n📡 Fetching data for {competition}...")
        context = await data_loader.fetch_all_context(
            competition=competition, days_ahead=days_ahead,
            include_recent_form=True, include_h2h=True, include_odds=True
        )
        print(f"   ✅ Fetched {len(context.fixtures)} fixtures")
        for fixture in context.fixtures:
            try:
                features = {
                    "home_team":   fixture["home_team"]["name"],
                    "away_team":   fixture["away_team"]["name"],
                    "league":      competition,
                    "market_odds": fixture.get("odds", {}),
                }
                await orchestrator.predict(features, str(fixture.get("external_id", "")))
            except Exception as e:
                print(f"   ⚠️ Prediction failed: {e}")
    except Exception as e:
        print(f"❌ Fetch failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator, data_loader, telegram_alerts

    print("\n🚀 VIT Sports Intelligence Network v3.0.0 — Starting...")

    # 1. MODELS
    print("\n🤖 Loading ML Models...")
    try:
        orchestrator = ModelOrchestrator()
        results = orchestrator.load_all_models()
        ready   = sum(1 for v in results.values() if v)
        print(f"   ✅ {ready}/{len(results)} models loaded")
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        orchestrator = ModelOrchestrator()
        orchestrator.initialized = True

    # 2. DATA LOADER
    print("\n📡 Initialising Data Loader...")
    try:
        football_api_key = os.getenv("FOOTBALL_DATA_API_KEY", "")
        odds_api_key     = os.getenv("ODDS_API_KEY", "") or os.getenv("THE_ODDS_API_KEY", "")
        data_loader = DataLoader(
            football_api_key=football_api_key,
            odds_api_key=odds_api_key,
            enable_scraping=get_env("ENABLE_SCRAPING", "true").lower() == "true",
            enable_odds=get_env("ENABLE_ODDS", "true").lower() == "true",
        )
        print(f"   ✅ Football API: {'ENABLED' if football_api_key else 'DISABLED'}")
        print(f"   ✅ Odds API: {'ENABLED' if odds_api_key else 'DISABLED'}")
    except Exception as e:
        print(f"   ❌ Data loader init failed: {e}")
        data_loader = None

    # 3. TELEGRAM
    print("\n📱 Setting Up Notifications...")
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id   = os.getenv("TELEGRAM_CHAT_ID", "")
    if bot_token and chat_id:
        try:
            telegram_alerts = TelegramAlert(bot_token, chat_id, enabled=True)
            await telegram_alerts.send_startup_message()
            print("   ✅ Telegram Alerts: ENABLED")
        except Exception as e:
            print(f"   ⚠️ Telegram init failed: {e}")
            telegram_alerts = TelegramAlert("", "", enabled=False)
    else:
        telegram_alerts = TelegramAlert("", "", enabled=False)
        print("   ⚠️ Telegram Alerts: DISABLED")

    # 4. WIRE ROUTES
    print("\n🔗 Linking Components to Routes...")
    try:
        predict.set_orchestrator(orchestrator)
        admin.set_orchestrator(orchestrator)
        training_route.set_orchestrator(orchestrator)   # v2.4.0
        if telegram_alerts and telegram_alerts.enabled:
            predict.set_telegram_alerts(telegram_alerts)
            admin.set_telegram_alerts(telegram_alerts)
        print("   ✅ All routes configured (v3.0.0)")
    except Exception as e:
        print(f"   ⚠️ Route config partial: {e}")

    # 5. DATABASE
    print("\n🗄️ Initialising Database...")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        db_ok = await _check_db_connection()
        print(f"   {'✅' if db_ok else '❌'} Database: {'CONNECTED' if db_ok else 'FAILED'}")
    except Exception as e:
        print(f"   ❌ Database init failed: {e}")

    print("\n✅ VIT Network v3.0.0 — Ready\n")
    yield

    # Shutdown
    print("\n🛑 Shutting down VIT Network...")
    if telegram_alerts and telegram_alerts.enabled:
        try:
            await telegram_alerts.send_shutdown_message()
        except Exception:
            pass


app = FastAPI(
    title="VIT Sports Intelligence Network",
    version="3.0.0",
    description="12-Model ML Ensemble for Football Match Predictions",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(APIKeyMiddleware)
app.add_middleware(LoggingMiddleware)

# ── Routes ────────────────────────────────────────────────────────────
app.include_router(predict.router)
app.include_router(result.router)
app.include_router(history.router)
app.include_router(admin.router)
app.include_router(training_route.router)   # v2.4.0
app.include_router(analytics_route.router)  # v2.5.0
app.include_router(odds_route.router)       # v3.0.0


# ── Health ────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    db_connected = await _check_db_connection()
    return HealthResponse(
        status="ok" if (orchestrator and orchestrator.initialized) else "degraded",
        models_loaded=orchestrator.num_models_ready() if orchestrator else 0,
        db_connected=db_connected,
        clv_tracking_enabled=True,
    )


@app.get("/system/status")
async def system_status():
    db_connected = await _check_db_connection()
    return {
        "version": "3.0.0",
        "status":  "operational",
        "components": {
            "orchestrator": {
                "initialized": orchestrator.initialized if orchestrator else False,
                "models_ready": orchestrator.num_models_ready() if orchestrator else 0,
                "total_models": 12,
            },
            "data_loader": {"status": "ready" if data_loader else "not_initialized"},
            "alerts": {
                "telegram_enabled": telegram_alerts.enabled if telegram_alerts else False,
                "status": "ready" if (telegram_alerts and telegram_alerts.enabled) else "disabled",
            },
            "database": {
                "connected": db_connected,
                "type": "sqlite" if _is_sqlite else "postgresql",
            },
        },
    }


@app.post("/test-predict")
async def test_predict(match: dict):
    if orchestrator is None:
        return {"error": "Orchestrator not initialized", "status": "unavailable"}
    features = {
        "home_team":   match.get("home_team"),
        "away_team":   match.get("away_team"),
        "league":      match.get("league", "premier_league"),
        "market_odds": match.get("market_odds", {}),
    }
    try:
        result = await orchestrator.predict(features, "test")
        return {"status": "success", "predictions": result.get("predictions", {})}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


@app.get("/api")
async def root():
    return {
        "name":    "VIT Sports Intelligence Network",
        "version": "3.0.0",
        "status":  "operational",
        "endpoints": {
            "core":      {"POST /predict": "Predict", "GET /history": "History", "GET /health": "Health"},
            "admin":     {"GET /admin/models/status": "Model status", "POST /admin/models/reload": "Reload",
                          "GET /admin/data-sources/status": "API health", "POST /admin/matches/manual": "Manual match",
                          "POST /admin/upload/csv": "Bulk upload",
                          "GET /admin/accumulator/candidates": "Acc candidates", "POST /admin/accumulator/generate": "Build acca"},
            "training":  {"POST /training/start": "Start training", "GET /training/progress/{id}": "Stream progress",
                          "GET /training/compare": "Compare versions", "POST /training/promote": "Promote",
                          "POST /training/rollback": "Rollback"},
            "analytics": {"GET /analytics/accuracy": "Accuracy", "GET /analytics/roi": "ROI",
                          "GET /analytics/clv": "CLV", "GET /analytics/model-contribution": "Models",
                          "GET /analytics/export/csv": "Export CSV", "GET /analytics/summary": "Summary"},
            "odds":      {"GET /odds/compare": "Multi-book odds", "GET /odds/arbitrage": "Arb scanner",
                          "POST /odds/injuries": "Add injury", "GET /odds/audit-log": "Audit log"},
        },
    }


# Static files (React frontend)
frontend_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
