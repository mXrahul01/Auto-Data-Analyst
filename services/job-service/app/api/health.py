"""Bootstrap probes. Readiness currently measures application startup only."""

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict:
    return {
        "service": request.app.state.settings.service_name,
        "status": "ok",
        "stage": "scaffold",
    }


@router.get("/ready")
async def ready(request: Request) -> dict:
    return {
        "service": request.app.state.settings.service_name,
        "status": "ready",
        "scope": "application_startup_only",
    }
