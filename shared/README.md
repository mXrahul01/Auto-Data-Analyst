# Shared contracts

Share versioned wire formats, not domain implementation or database models. JSON schemas are the cross-language source of truth. Producers and consumers must validate them at the boundary when features are implemented. Breaking changes require a new schema version and compatibility plan. No service imports another service's app package.
