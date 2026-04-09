#!/usr/bin/env python3
"""
Clear all predictions and CLV entries from the database.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.db.database import engine, Base
from app.db.models import Prediction, CLVEntry
from sqlalchemy import delete
import asyncio

async def clear_predictions():
    async with engine.begin() as conn:
        # Delete CLV entries first (foreign key constraint)
        await conn.execute(delete(CLVEntry))
        print("✅ Cleared CLV entries")

        # Delete predictions
        await conn.execute(delete(Prediction))
        print("✅ Cleared predictions")

        print("🎯 Database predictions cleared successfully!")

if __name__ == "__main__":
    asyncio.run(clear_predictions())