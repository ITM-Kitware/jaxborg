# Study instructions

Before creating or changing a study, read the [study guide](README.md).
For work on ITM, including jobs controlled remotely, follow these requirements
and the commands in [Studies on ITM](../docs/itm-studies.md).

## Storage and tracking on ITM

- Keep study definitions, protocols, recipes, and analysis code in the source
  checkout. Put new personal checkouts and their environments under
  `/data/users/<user>/src/cyber/`, with large caches on the data drive too.
- Put generated study files under `/data/shared/jaxborg/studies/<study>/`.
  Use unique run directories and retain evidence outside the checkout. Keep
  reusable inputs in `/data/shared/jaxborg/datasets/` and standalone benchmarks
  in `/data/shared/jaxborg/benchmarks/`. Preserve collaborator access through
  group `itm` and shared output permissions.
- All studies use one dedicated Jaxborg MLflow installation and service, with
  metadata at `/data/shared/jaxborg/mlflow/mlflow.db` and logged artifacts at
  `/data/shared/jaxborg/mlflow/artifacts/`. Open World's installation and
  database under `/data/shared/mlflow/` are separate.
- Set `JAXBORG_EXP_DIR`, `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`, and
  the recorded execution's evidence root explicitly in the remote job. Use the
  dedicated service's documented HTTP endpoint and an experiment name such as
  `jaxborg/<study>`. If the service is unavailable, report it; do not create a
  per-study database or fall back to Open World.
- Populate tracking through the supported recorded execution, which creates
  the attempt's run, logs metrics, and uploads terminal evidence. Check a small
  run and verify a downloaded checkpoint before production. Importing historical
  stores is a separate operation following the [import procedure](../docs/itm-studies.md#bring-existing-runs-into-the-jaxborg-database).
  Changing the tracking URI does not ingest old runs.

## Checkpoint budget

- Default to a periodic recovery checkpoint every **four hours of elapsed
  training per run**, excluding queue time. Use **two hours** when interruption
  risk or recovery cost warrants it. More frequent saves require a specific
  scientific or recovery need recorded in the execution settings.
- Retain required initial state, the final checkpoint, and protocol-declared
  scientific milestones. A short run normally needs no periodic recovery save.
  Keep scalar metric logging frequent without saving weights at every log step.
- Set `jax.checkpoint_every_updates` explicitly using pilot throughput to
  approximate the chosen interval. It counts updates, not elapsed hours.
  Audit `mlflow.checkpoint_eval.every_steps`, `--checkpoint-update`, team
  overrides, and evaluation suites for additional checkpoint saves.
- Estimate storage across all conditions, seeds, attempts, policy weights,
  full `.npz` resume states, native/evidence copies, and MLflow exports before
  submission. Include repeated dependency bytes in the estimate and verify
  actual checkpoint sizes during the pilot.
- Preserve existing checkpoints, manifests, resume lineage, symlinks, and queued
  jobs. Apply sparse checkpoint settings to new work; existing-store cleanup
  requires a separate reviewed plan.
