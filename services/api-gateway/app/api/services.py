"""A fixed service catalog; upstream addresses cannot be supplied by callers."""

import asyncio

import httpx
from fastapi import APIRouter

router = APIRouter(tags=["services"])
SERVICES = {
    "ingestion-service": "ingestion",
    "profiling-service": "profiling",
    "transformation-service": "transformation",
    "ml-service": "ml",
    "job-service": "jobs",
}


async def inspect_service(name: str, prefix: str, client: httpx.AsyncClient) -> dict:
    try:
        response = await client.get(f"http://{name}:8000/health")
        response.raise_for_status()
        body = response.json()
        healthy = (
            isinstance(body, dict)
            and body.get("status") == "ok"
            and body.get("service") == name
        )
        state = "reachable" if healthy else "unavailable"
    except (httpx.HTTPError, ValueError):
        state = "unavailable"
    return {
        "name": name,
        "state": state,
        "stage": "scaffold",
        "health_path": f"/api/v1/{prefix}/health",
    }


@router.get("/services")
async def services() -> dict:
    async with httpx.AsyncClient(timeout=2.0, trust_env=False) as client:
        results = await asyncio.gather(
            *(inspect_service(name, prefix, client) for name, prefix in SERVICES.items())
        )
    return {"stage": "scaffold", "services": results}
