# Verification

## Reproduce scaffold checks

Run each Python package separately to avoid collisions between the independent `app` packages:

```sh
make sync
make lint
make test
make frontend-build
```

Or run `uv sync --frozen`, `uv run --frozen pytest`, and `uv run --frozen ruff check app tests` within one service directory.

Compose configuration: `docker compose --env-file .env.microservices config --quiet`. Full runtime: start Compose and visit http://localhost:8080. Dependency readiness and business workflows are not covered by bootstrap probes.

## Meaning of results

Unit checks verify that health routes disclose the scaffold stage and startup-only readiness. Gateway checks exercise unavailable and successful upstream responses without requiring containers. Frontend checks verify TypeScript compilation and production bundling. A passing build is not a browser usability check or a data science benchmark.

A validation report from this change appears in the pull request. Docker runtime validation must be performed on a machine with Docker before claiming the full stack was exercised.

## Scaffold verification results

- Resolved and installed all ten independent Python environments with committed uv lockfiles.
- All 23 service tests passed, including gateway failure and identity-mismatch cases.
- Ruff checks passed for all ten service packages after import-order formatting.
- React TypeScript compilation and Vite production build passed.
- Shared JSON schemas, Python manifests, Compose YAML syntax, and referenced build contexts were validated.
- The FastAPI test client emitted an upstream deprecation warning about its httpx integration; checks still passed. Revisit the test transport when updating dependencies.
- Docker is not installed in the authoring environment. Container builds, full Compose semantic validation, and a browser walkthrough of the assembled stack were not run.

These results cover the foundation only. They establish no data analysis, training accuracy, scale, or production reliability claims.
