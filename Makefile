SERVICES := api-gateway ingestion-service profiling-service transformation-service ml-service job-service auth-service analytics-service dashboard-service export-service
COMPOSE := docker compose --env-file .env.microservices

.PHONY: setup up down logs sync test lint frontend-build
setup:
	cp -n .env.microservices.example .env.microservices
up:
	$(COMPOSE) up --build -d
down:
	$(COMPOSE) down
logs:
	$(COMPOSE) logs -f --tail=100
sync:
	@set -e; for service in $(SERVICES); do (cd services/$$service && uv sync --frozen); done
test:
	@set -e; for service in $(SERVICES); do (cd services/$$service && uv run --frozen pytest); done
lint:
	@set -e; for service in $(SERVICES); do (cd services/$$service && uv run --frozen ruff check app tests); done
frontend-build:
	cd frontend/microservices && npm ci && npm run build
