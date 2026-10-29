# Retained local evidence producers

See the [experiment guide](../experiments/README.md) for the workflow entry points.
To record a new checkpoint evaluation as it runs, use
[evaluation capture](evaluation-capture.md).

`python -m jaxborg.experiment_producers --frozen-bundle /absolute/bundle --smoke-store /absolute/smoke --output /outside/checkout/records.json`
imports two local producers into the common version 1 record contract. The output
must be new and outside both the checkout and the original evidence directories.
The adapters read the original bytes without rewriting them.

The frozen bundle adapter checks the index hashes, producing run identity, and
retained output completion/hash fields. Original tracking-store identity qualifies
run identities. Scientific training cells come from the protocol training seeds;
evaluation cells come from the saved validation/test candidate and cohort
relationships. Exact run selection does not establish retry genealogy: the frozen
producer records a sourced unknown retry link where that genealogy is absent. Source-import and report executions remain provenance with unknown
scientific trial membership. They are not independent training replications.
Evaluation metrics include Blue mean return and sample episode SD. SD describes
episode variability, not an effect interval or independent training variability.

The SQLite adapter reads every retained run in a transaction, rejects an active
WAL or changing database, and preserves run fields, parameters, tags, metrics, and
source row references. Same-name runs stay separate executions. Without an
authoritative protocol/trial association their scientific membership and retry
relationships stay unknown. A recorded `env_steps` metric stays an observed axis
fact; completed scientific work remains unknown without verified producing-code
semantics. Adjacent recipes, metrics files, and model bytes
remain unowned artifacts. Screenshots and database lock files are not scientific
evidence. Importing two producers establishes common discovery, not permission to
pool their observations.

Use `experiment_records.collate` with explicit trial-to-attempt and evaluation
identities, plus an analysis source and resolved configuration. For the frozen report, set
`analysis.source` using the callable
`jaxborg.experiment_producers.source.analysis_source_identity(analyzer_path)` and
`analysis.configuration` to the complete resolved configuration. The identity is
SHA256 of canonical JSON mapping repository paths to SHA256 values for the analyzer
and its imported local record/producer helper modules, including package initializers.
`analysis_source_files(analyzer_path)` returns this canonical path-to-file mapping.
The report archives those exact imported helper files under `analysis-source/src/`. Keep all imported
records in the snapshot so excluded unresolved evidence stays visible. Required
known facts are declared by the frozen protocol. Saving a collation does not
change the original study.

The existing frozen report accepts `--selection /absolute/collation.json`.
It verifies the saved collation through the common contract, checks its records
against the current frozen bundle, requires the complete original scientific
selection, and matches the resolved analysis configuration. Its original scientific
checks still validate policy identities, game contracts, validation selection,
held-out cohorts, budgets, and source evidence. The report saves a copy of the
collation alongside its outputs. Unrelated discovery evidence is retained in the
snapshot but its bytes are not inputs to the frozen analysis.
