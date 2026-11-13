# Domain event contracts

The envelope defines planned business events, not an implemented event bus. Celery task messages and Celery's monitoring events are separate from domain events. PostgreSQL outbox publication and idempotent consumers must be implemented before claiming reliable event delivery. Payloads carry authorized IDs and artifact references, never full datasets or secrets. Add an event-specific payload schema with each feature.
