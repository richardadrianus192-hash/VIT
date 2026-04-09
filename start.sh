#!/bin/bash
# Start backend on port 8000 in background
export AUTH_ENABLED="false"
export LOG_LEVEL="INFO"
export PYTHONPATH="."
export ENABLE_SCRAPING="true"
export ENABLE_ODDS="true"

# Use DATABASE_URL from secrets if set, otherwise fall back to SQLite for local dev
if [ -z "$DATABASE_URL" ]; then
  export DATABASE_URL="sqlite+aiosqlite:///vit.db"
fi

echo "Starting FastAPI backend on port 8000..."
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "Starting Vite frontend on port 5000..."
cd frontend && npm run dev
