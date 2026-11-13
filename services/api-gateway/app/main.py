"""API gateway bootstrap. Domain features are intentionally not implemented yet."""

from fastapi import FastAPI

from app.api.health import router
from app.api.services import router as services_router
from app.config import Settings

settings = Settings()
app = FastAPI(title="Auto Data Analyst API gateway", version="0.1.0")
app.state.settings = settings
app.include_router(router)
app.include_router(services_router, prefix="/api/v1")
