# Auto Data Analyst contributor instructions

## Current work
The microservices foundation lives under `services/`, `frontend/microservices/`, `shared/`, and `infrastructure/`. Start with [docs/MICROSERVICES.md](docs/MICROSERVICES.md). The original backend and static frontend remain in place; do not move or delete them without an explicit migration task.

## Scope and architecture
- Services own their domain logic, database tables, migrations, dependencies, and tests.
- One service must not import another service's application package or write its tables.
- Shared code is limited to versioned wire contracts and genuinely repeated technical utilities.
- Celery tasks and domain events are different concepts. Use an outbox and idempotent consumers when reliable domain messaging is implemented.
- The present services are health-only bootstraps. Do not claim analysis, training, authorization, or durable orchestration is implemented.

## Data and ML
Preserve raw uploads, create immutable output versions, and record recipes. Never execute user-supplied Python or SQL. Fit learned preprocessing inside training partitions and cross-validation folds. Keep final test data separate from model selection. Never invent model metrics.

## Verification
Run each service's tests in its own environment: `cd services/<name> && uv run --frozen pytest`. Run `uv run --frozen ruff check app tests` in the same directory. Build the new frontend with `cd frontend/microservices && npm ci && npm run build`. Validate Compose with `docker compose --env-file .env.microservices config --quiet` where Docker is available. Report missing tools and unrun checks honestly.

## Safety and repository hygiene
Keep secrets, uploads, logs, models, local databases, and generated build output out of Git. The Compose environment is local-only and has no application authentication yet. Do not deploy it publicly as a finished application. Use focused branches and pull requests; preserve unrelated user work.
