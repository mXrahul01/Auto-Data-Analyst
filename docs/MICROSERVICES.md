# Microservices foundation

This is the entry point for the new architecture. It follows the user's request to use microservices, superseding the original blueprint's modular monolith decision. The previous backend and static frontend are preserved for later assessment; no migration of their behavior is implied.

## What runs today

Ten independently packaged FastAPI bootstraps expose `/health`, `/ready`, and OpenAPI documentation. The gateway additionally checks the five core upstream HTTP probes. The React workspace displays real reachability and clearly labels the scaffold stage. Optional Celery worker startup is configured; no business tasks or reliable orchestration are implemented.

Dataset upload, cleaning, analytical dashboards, authentication, persistence, training, and export workflows are planned. The health routes report application startup only. They are not proof that database, broker, data access, or ML functions are ready.

## Start locally

Prerequisites: Git, Docker Engine or Docker Desktop with Compose v2. For native development, use Python 3.12 or 3.13, uv 0.13.6, and Node 22.12 or newer.

```sh
git clone https://github.com/mXrahul01/Auto-Data-Analyst.git
cd Auto-Data-Analyst
git switch scaffold/microservices-foundation
cp .env.microservices.example .env.microservices
docker compose --env-file .env.microservices up --build -d
```

Open http://localhost:8080. The new page shows connection status. The gateway service list is at http://localhost:8080/api/v1/services. Example probe: http://localhost:8080/api/v1/ingestion/health. OpenAPI for that service is at http://localhost:8080/api/v1/ingestion/openapi.json; direct native startup also provides interactive `/docs`. Swagger UI accessed behind a stripped prefix needs root-path configuration before it can be advertised through the proxy.

```sh
docker compose --env-file .env.microservices logs -f --tail=100
docker compose --env-file .env.microservices down
```

Stopping preserves development volumes. No destructive volume-removal command is included. On Windows PowerShell use `Copy-Item .env.microservices.example .env.microservices`; the Docker commands are otherwise the same.

To start the four later-service HTTP shells, add `--profile extended`. To start the empty Celery worker, add `--profile workers`. Both profiles are for development. The browser does not call optional services yet.

## Service boundaries

| Service | Planned ownership | Default stack |
| --- | --- | --- |
| api-gateway | Entry point, cross-cutting HTTP policy | Yes |
| ingestion-service | Uploads, parsing, normalized originals | Yes |
| profiling-service | Quality evidence and statistics | Yes |
| transformation-service | Recipes and version lineage | Yes |
| ml-service | Split policy, pipelines, experiments, predictions | Yes |
| job-service | Job lifecycle, dispatch, cancellation | Yes |
| auth-service | Identity and session lifecycle | Extended |
| analytics-service | Validated metrics and aggregate queries | Extended |
| dashboard-service | Chart definitions and shared filters | Extended |
| export-service | Reports and export artifacts | Extended |

Each service has `app/api`, `app/domain`, `app/schemas`, `app/repositories`, `app/services`, `tests`, a Dockerfile, pyproject, and lockfile. Reserved modules contain a purpose README so Git preserves the directory structure without pretending that domain code exists.

## Repository map

| Location | Purpose |
| --- | --- |
| services/ | Independent Python service packages |
| frontend/microservices/ | New React and TypeScript workspace |
| shared/schemas/ | Error and dataset reference JSON schemas |
| shared/events/ | Planned event envelope and messaging conventions |
| infrastructure/nginx/ | Same-origin routing and SPA hosting |
| infrastructure/compose/ | Orchestration notes |
| docker-compose.yml | Local foundation stack |
| .env.microservices.example | Dedicated local configuration template |
| sample_data/ | Documented synthetic CSV fixture |
| tests/integration/ and tests/e2e/ | Future cross-service workflows |
| docs/microservices/ | Architecture decisions, roadmap, and verification |
| .github/workflows/microservices.yml | New scaffold checks only |

The root README, old dependency files, `backend/`, and the original frontend files describe earlier work. They are not a verified status report for the new architecture. The existing `.Makefile` is separate from the new functional `Makefile`.

## Local-only boundaries

Only the frontend binds a host port, on 127.0.0.1. Other containers use an internal Docker network with no published database or broker ports. The sample database password is for this isolated development environment only. Authentication, TLS, service-to-service identity, per-service database roles, storage authorization, and durable job lifecycle must be implemented before public hosting.

The database is reserved infrastructure; current APIs do not connect to it and no shared superuser credential is passed to application services. Replace this with service-owned roles, migrations, and minimal privileges when persistence is added.
