# VIT Sports Intelligence — v2.4.0 + v2.5.0 + v3.0.0 Release

## Files in This Release (10 files)

```
v2.4_2.5_3.0_release/
├── main.py                              ← Updated — registers 3 new routers
├── app/api/routes/
│   ├── training.py                      ← v2.4.0  Training pipeline
│   ├── analytics.py                     ← v2.5.0  Analytics suite
│   └── odds_compare.py                  ← v3.0.0  Odds + arbitrage + audit
├── frontend/src/
│   ├── TrainingPanel.jsx                ← v2.4.0  Training UI
│   ├── AnalyticsPanel.jsx               ← v2.5.0  Analytics UI
│   ├── OddsPanel.jsx                    ← v3.0.0  Odds/arb/injuries UI
│   └── App.jsx                          ← Updated — all 7 tabs wired in
└── RELEASE_NOTES.md
```

---

## v2.4.0 — Training Pipeline

### Endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | `/training/start` | Trigger async training run |
| GET  | `/training/progress/{id}` | SSE stream of live events |
| GET  | `/training/status/{id}` | Poll job status |
| GET  | `/training/jobs` | List all jobs |
| GET  | `/training/compare?job_id_a=&job_id_b=` | Side-by-side accuracy delta |
| POST | `/training/promote` | Promote version to production |
| POST | `/training/rollback` | Rollback to previous version |

### UI Features
- Config form (date range, validation split, epochs, run note)
- Live progress bar per model + event log stream
- Model results grid (accuracy + elapsed time per model)
- Training history table with Promote button
- Side-by-side comparison with per-model Δ accuracy
- Auto-suggestion: promote / neutral / rollback

---

## v2.5.0 — Analytics Suite

### Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/analytics/accuracy` | Overall + by league + by confidence + weekly trend |
| GET | `/analytics/roi` | P&L, ROI, win rate, max drawdown, equity curve |
| GET | `/analytics/clv` | CLV summary + per-match series |
| GET | `/analytics/model-contribution` | Per-model accuracy, participation, confidence |
| GET | `/analytics/export/csv` | Download full history as CSV |
| GET | `/analytics/summary` | Single-call key metrics |

### UI Features (5 sub-tabs)
- **Overview** — summary stats cards
- **Accuracy** — league table, confidence buckets, weekly bar chart
- **ROI** — stat cards + SVG equity curve
- **CLV** — closing line value table
- **Models** — contribution breakdown with participation rate bars

---

## v3.0.0 — Production Release

### Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET  | `/odds/compare` | Multi-bookmaker odds per match |
| GET  | `/odds/arbitrage` | Guaranteed profit opportunities |
| POST | `/odds/injuries` | Add injury/team news note |
| GET  | `/odds/injuries` | List all injury notes |
| DELETE | `/odds/injuries/{id}` | Remove injury note |
| GET  | `/odds/audit-log` | Full admin action history |

### UI Features (4 sub-tabs)
- **Odds Comparison** — table of all bookmakers per event, best price highlighted
- **Arbitrage Scanner** — events where sum(1/odds) < 1, with stake calculator
- **Injury Notes** — add/delete player injury/news entries
- **Audit Log** — every admin action logged with timestamp

---

## Replit Agent Deploy Prompt

```
I have new files in v2.4_2.5_3.0_release/. Please:

1. Replace main.py with v2.4_2.5_3.0_release/main.py
2. ADD app/api/routes/training.py from v2.4_2.5_3.0_release/app/api/routes/training.py
3. ADD app/api/routes/analytics.py from v2.4_2.5_3.0_release/app/api/routes/analytics.py
4. ADD app/api/routes/odds_compare.py from v2.4_2.5_3.0_release/app/api/routes/odds_compare.py
5. ADD frontend/src/TrainingPanel.jsx from v2.4_2.5_3.0_release/frontend/src/TrainingPanel.jsx
6. ADD frontend/src/AnalyticsPanel.jsx from v2.4_2.5_3.0_release/frontend/src/AnalyticsPanel.jsx
7. ADD frontend/src/OddsPanel.jsx from v2.4_2.5_3.0_release/frontend/src/OddsPanel.jsx
8. Replace frontend/src/App.jsx with v2.4_2.5_3.0_release/frontend/src/App.jsx
9. Do not change any other files.
10. Run: cd frontend && npm run build
11. Restart the app.
```

---

## Version Summary: All Releases

| Version | Focus | Files Changed |
|---------|-------|---------------|
| v2.1.0 | Fix 36.3% bug, Telegram alerts | 4 files |
| v2.2.0 | Admin panel: model mgmt, manual entry, CSV | 4 files |
| v2.3.0 | Accumulator generator | 2 files added |
| v2.4.0 | Training pipeline | 2 files added |
| v2.5.0 | Analytics suite | 1 file added |
| v3.0.0 | Odds compare, arbitrage, audit log | 2 files added |

**Total new/changed files across all releases: 15 files**
