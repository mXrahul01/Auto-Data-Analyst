# Local orchestration

The runnable file is [docker-compose.yml](../../docker-compose.yml) at the repository root. Run `docker compose --env-file .env.microservices up --build`. The default profile starts foundation services; `--profile extended` adds four reserved service bootstraps. These health-only services do not implement their business workflows.
