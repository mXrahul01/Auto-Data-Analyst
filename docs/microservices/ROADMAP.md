# Implementation roadmap

## M0 Scaffold
Independent packages, health routes, service-status UI, containers, shared contracts, CI, and local setup. This change implements M0 foundations only.

## M1 Ingestion vertical slice
Define owner/project/version contracts and storage boundaries. Implement bounded CSV upload, parsing, schema confirmation, leading-zero ID preservation, normalized Parquet, and a real frontend preview. Add service-owned migrations and identity before persistent private user data is accepted.

Acceptance: malformed rows are reported, limits are enforced, originals remain immutable, preview pagination works, and only the owner can read artifacts.

## M2 Profiling
Implement missingness, type conflicts, category variants, constant columns, and duplicate candidates with evidence and explicit denominators. Show full versus sampled results. Connect the quality screen to actual results.

## M3 Preparation
Add typed operation registry, previews, approved fixed rules, lineage, immutable versions, and reproducible recipes. Keep learned ML transforms out of generic full-data cleaning. Verify recipe replay and stale-version conflicts.

## M4 Analytics and dashboards
Implement validated metric and query specifications, shared filter semantics, useful charts, and backing tables. Confirm row grain and denominators. Add no arbitrary SQL endpoint.

## M5 Training
Classification and regression only. Require target, prediction moment, available features, metric, and split. Add dummy, linear, and forest candidates. Fit preprocessing inside validation folds, choose using development data, and evaluate the final choice on reserved test data. Record real results and failures.

## M6 Prediction and export
Validate new input against the saved feature schema, reuse the fitted pipeline without refitting, and export data, recipe, report, and trusted model bundle.

## M7 Operational readiness
Persistent jobs, outbox dispatch, leases, safe retries, cancellation, quotas, deletion, sessions, minimal service roles, TLS, backups, and measured resource limits. Extend health readiness to actual dependencies.

## M8 Optional AI
Explain verified findings using evidence IDs. Require explicit external-data processing choices and support an API-key-free core. Exclude arbitrary generated Python and SQL execution.
