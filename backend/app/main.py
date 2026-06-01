# ── main.py ──────────────────────────────────────────────
# Purpose: Entry point of the FastAPI application
# Creates the app, connects the database, registers routes
# ─────────────────────────────────────────────────────────

from fastapi import FastAPI
from app.database import engine, Base

# Import the Article model so SQLAlchemy knows about it
# Without this import, Base.metadata won't include the articles table
from app.models import article

# Create all tables in PostgreSQL that don't exist yet
# This reads every class that inherits from Base
# and creates the corresponding table automatically
# Safe to run multiple times — it skips tables that already exist
Base.metadata.create_all(bind=engine)

# Create the FastAPI application instance
app = FastAPI(
    title="Fake News Detector API",
    description="API for detecting fake news using ML",
    version="1.0.0"
)

# Basic health check route
# Always build this first — confirms your server is running
@app.get("/")
def health_check():
    return {
        "status": "running",
        "message": "Fake News Detector API is live!"
    }