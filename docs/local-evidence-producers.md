# Retained local evidence producers

See the [experiment guide](../experiments/README.md) for workflow and selection
conventions, or [evaluation capture](evaluation-capture.md) to retain a new run.

```bash
python -m jaxborg.experiment_producers \
  --frozen-bundle /absolute/bundle --smoke-store /absolute/smoke \
  --output /outside/checkout/records.json
```

This imports both producers into version 1 records without rewriting their
bytes. Output must be new and outside the checkout and input directories.

The frozen adapter verifies index hashes, producing run identity, and retained
output completion/hash fields. Tracking-store identity qualifies run IDs.
Training cells use protocol training seeds; evaluation cells use saved candidate
and cohort relationships. Missing retry genealogy stays a sourced unknown.
Import/report executions have unknown scientific trial membership. Blue return
metrics include the mean and sample episode SD; SD measures episode variability,
not independent training variability or an effect interval.

The SQLite adapter reads every run in a transaction and rejects an active WAL or
changing database. It preserves fields, parameters, tags, metrics, and row
references. Same-name runs remain separate. Missing protocol/trial membership
and retry links stay unknown. `env_steps` remains an observed axis fact without
verified producing-code semantics for completed work. Adjacent recipes, metrics,
and models remain unowned artifacts; screenshots and lock files are excluded.
Common discovery does not establish permission to pool observations.

Use `experiment_records.collate` with explicit trial-to-attempt and evaluation
IDs, analysis source, and complete resolved analysis configuration. Keep all
imported records so excluded unresolved evidence remains visible. For the frozen
report, obtain `analysis.source` with
`jaxborg.experiment_producers.source.analysis_source_identity(analyzer_path)`.
It hashes the analyzer and imported local record/producer helpers;
`analysis_source_files(analyzer_path)` returns their canonical path-to-file map.
The report archives those helpers under `analysis-source/src/`.

The frozen report accepts `--selection /absolute/collation.json`, checks its
records against the bundle, and requires the complete original scientific
selection and matching analysis configuration. Existing policy, game, cohort,
budget, and source checks still apply, including required known protocol facts.
The saved collation accompanies the outputs without changing the original
study. Unrelated discovery records remain visible but their bytes are not
analysis inputs.
