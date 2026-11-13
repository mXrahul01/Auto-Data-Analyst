"""Ingestion bootstrap. Domain features are intentionally not implemented yet."""

from fastapi import FastAPI

from app.api.health import router
from app.config import Settings

settings = Settings()
app = FastAPI(title="Auto Data Analyst Ingestion", version="0.1.0")
app.state.settings = settings
app.include_router(router)
