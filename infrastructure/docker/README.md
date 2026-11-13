# Containers

Each Python service owns its Dockerfile. The frontend builds from the repository root using `frontend/microservices/Dockerfile`, then serves compiled files and routes HTTP through Nginx. Images use non-root users. Keep runtime data in private stores rather than container image layers.
