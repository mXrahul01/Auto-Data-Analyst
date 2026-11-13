# Microservices

Ten service boundaries are reserved. Each service has its own environment, lockfile, container, health routes, tests, and domain folders. The default stack runs the gateway, ingestion, profiling, transformation, ML, and job service APIs. Auth, analytics, dashboard, and export bootstraps are optional under the `extended` profile. Only bootstrap health and a service-status catalog work today.

Business endpoints, persistent jobs, authentication, and data processing are not implemented in this scaffold. Avoid adding dummy successful responses. Use the milestones in [the build plan](../docs/microservices/ROADMAP.md).
