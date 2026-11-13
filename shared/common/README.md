# Common conventions

Request IDs, structured error envelopes, timestamps in UTC, and schema versioning belong here as specifications. Add reusable code only after two services need it and give it a versioned package. Never place repositories, ORM models, or data-cleaning algorithms in a shared catch-all module.
