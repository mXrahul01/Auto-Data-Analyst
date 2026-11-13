import asyncio

import httpx

from app.api.services import inspect_service


def test_upstream_failure_is_reported_as_unavailable():
    async def run():
        transport = httpx.MockTransport(lambda request: httpx.Response(503))
        async with httpx.AsyncClient(transport=transport) as client:
            return await inspect_service("ingestion-service", "ingestion", client)

    result = asyncio.run(run())
    assert result["state"] == "unavailable"
    assert result["health_path"] == "/api/v1/ingestion/health"


def test_another_service_cannot_masquerade_as_the_expected_upstream():
    async def run():
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"service": "wrong", "status": "ok"})
        )
        async with httpx.AsyncClient(transport=transport) as client:
            return await inspect_service("ingestion-service", "ingestion", client)

    assert asyncio.run(run())["state"] == "unavailable"


def test_valid_probe_is_reported_as_reachable():
    async def run():
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"service": "ingestion-service", "status": "ok", "stage": "scaffold"}
            )
        )
        async with httpx.AsyncClient(transport=transport) as client:
            return await inspect_service("ingestion-service", "ingestion", client)

    assert asyncio.run(run())["state"] == "reachable"
