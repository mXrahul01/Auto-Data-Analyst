# Dashboard

Saved chart specifications and shared filter state.

**Status:** runnable HTTP bootstrap only. Domain endpoints are planned. This service is reserved for the optional `extended` Compose profile.

## Ownership

Own this service's tables, migrations, domain logic, and API. Exchange versioned contracts and artifact references; never import another service's application package or modify its database tables.

## Local commands

Run from this directory with Python 3.12 or 3.13 and uv installed:

```sh
uv sync --frozen
uv run uvicorn app.main:app --reload --port 8000
uv run pytest
uv run ruff check app tests
```

Start one service at a time on port 8000, or choose separate ports. Compose handles isolation and service DNS. Probes: `/health` and `/ready`; schema: `/openapi.json`; interactive docs: `/docs`.

Readiness currently reports application startup only. Add dependency checks when real integrations are implemented. The scaffold has no authentication, dataset access, or training API and is for local development only.
