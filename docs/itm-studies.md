# Studies on ITM

Use this guide when preparing Jaxborg or CAGE4 studies on ITM, including work
controlled from another machine over `ssh itm`. Follow the required
[study instructions](../studies/AGENTS.md). Start with the
[study guide](../studies/README.md) for the scientific comparison and supported
[recorded execution](joint-execution-records.md) for training and evaluation.
The paths below define the layout for new work; they do not migrate existing
results or provision a tracking service.

## Storage layout

All paths in this table are on ITM. Replace `<user>`, `<study>`, and `<run>` with
the relevant identities. Keep study definitions and analysis code in Git;
keep generated outputs on the data drive.

| Contents | Location |
| --- | --- |
| Personal source repositories and worktrees | `/data/users/<user>/src/cyber/` |
| Jaxborg repository and worktrees | `/data/users/<user>/src/cyber/jaxborg/{.bare,main,<branch>}/` |
| Shared study outputs | `/data/shared/jaxborg/studies/<study>/` |
| Native trainer outputs | `/data/shared/jaxborg/studies/<study>/runs/<run>/` |
| Retained evidence | `/data/shared/jaxborg/studies/<study>/evidence/` |
| Study evaluations, reports, and figures | `/data/shared/jaxborg/studies/<study>/{evaluations,reports,figures}/` |
| Job logs and submission records | `/data/shared/jaxborg/studies/<study>/{logs,submissions}/` |
| Dedicated Jaxborg MLflow database | `/data/shared/jaxborg/mlflow/mlflow.db` |
| Dedicated Jaxborg MLflow artifact store | `/data/shared/jaxborg/mlflow/artifacts/` |
| Historical import manifests and run mappings | `/data/shared/jaxborg/mlflow/imports/<batch>/` |
| Shared reusable inputs and standalone benchmarks | `/data/shared/jaxborg/{datasets,benchmarks}/` |
| Temporary shared work | `/data/shared/jaxborg/scratch/<user>/<run>/` |
| Personal uv and JAX caches | `/data/users/<user>/.cache/{uv,jaxborg/xla}/` |

Use a bare repository with sibling worktrees, name each worktree after its
branch, and develop in a non-main worktree. Keep its `.venv` on the data drive.
Use `umask 0002` for shared results and verify that new directories inherit
group `itm` with collaborator access. Inspect `df -h` and resolved paths before
launching. Home-directory links must resolve onto `/data/` for large outputs.
Consult `/data/shared/jaxborg/README.md` before extending older collections;
leave their paths and existing jobs intact unless a migration is requested.

## One Jaxborg tracking service

All Jaxborg studies use one dedicated Jaxborg MLflow installation, service,
database, and artifact store. Open World's installation on port `5000`, with
data under `/data/shared/mlflow/`, is a separate service. Do not use its database,
environment, or artifact destination for Jaxborg.

The service owner provisions a dedicated environment under
`/data/users/<owner>/src/cyber/jaxborg-mlflow/`, with an explicitly pinned MLflow
version compatible with the study clients. Record its owner, endpoint, startup
configuration, package version, and restart/backup procedure in
`/data/shared/jaxborg/mlflow/README.md`. Use a persistent service configuration
so a terminal disconnect does not interrupt tracking.

For the dedicated service, the MLflow server arguments are:

```sh
mlflow server --host 127.0.0.1 --port "$JAXBORG_MLFLOW_PORT" --workers 1 \
  --backend-store-uri sqlite:////data/shared/jaxborg/mlflow/mlflow.db \
  --artifacts-destination /data/shared/jaxborg/mlflow/artifacts \
  --default-artifact-root mlflow-artifacts:/ --serve-artifacts
```

The owner selects and records an available port other than `5000`, creates the
directories with the shared permissions above, and runs this command from the
dedicated environment. These arguments describe the configuration, not a
second server to start for each study. Clients use its HTTP URL; only the
service opens the database. Existing studies must not silently fall back to
per-checkout databases when the service is unavailable.

For access from another machine, forward an available local port to the recorded
ITM port:

```sh
ssh -N -L "127.0.0.1:${LOCAL_MLFLOW_PORT}:127.0.0.1:${JAXBORG_MLFLOW_PORT}" itm
```

ITM jobs use the ITM endpoint, not the controller's forwarded port. Loopback
access assumes the job runs on ITM itself; a different compute node needs a
documented route to this same service.

## Populate MLflow with new executions

After shell initialization, set the following in each remote job. The service
owner supplies `JAXBORG_MLFLOW_PORT`; choose `STUDY_ID` and a fresh `RUN_ID` before
submission. Use the actual ITM username for `ITM_USER`.

```sh
umask 0002
export STUDY_ROOT="/data/shared/jaxborg/studies/$STUDY_ID"
export JAXBORG_EXP_DIR="$STUDY_ROOT/runs/$RUN_ID"
export MLFLOW_TRACKING_URI="http://127.0.0.1:$JAXBORG_MLFLOW_PORT"
export MLFLOW_EXPERIMENT_NAME="jaxborg/$STUDY_ID"
export UV_CACHE_DIR="/data/users/$ITM_USER/.cache/uv"
export JAX_COMPILATION_CACHE_DIR="/data/users/$ITM_USER/.cache/jaxborg/xla"
mkdir -p "$JAXBORG_EXP_DIR" "$STUDY_ROOT/evidence" \
  "$STUDY_ROOT/logs" "$STUDY_ROOT/submissions"
```

For launchers with an existing study-specific directory contract, retain that
layout beneath `STUDY_ROOT` rather than adding a second set of output folders.
Create Slurm log directories before submission. Read the host instructions
for allocations.

Run the study's recorded entrypoint with its protocol, trial identity, exact
source revision, and `--evidence-root "$STUDY_ROOT/evidence"`; follow the
[execution guide](joint-execution-records.md) for the supported trainer arguments.
Setting environment variables alone does not ingest old runs.

Recorded execution creates an MLflow run for its attempt, logs parameters and
metrics, and uploads terminal evidence and retained dependencies to that run's
`evidence/` artifact directory. The database holds tracking metadata; the
artifact store holds checkpoint and evidence bytes. An experiment name groups
runs and does not replace the protocol's scientific study identity.

Before production, use the study's small recorded smoke run to confirm that:

- The run appears in the dedicated Jaxborg service with the expected study,
  trial, and attempt identities.
- Its artifact URI uses `mlflow-artifacts:/`, and a required checkpoint can be
  downloaded through the API and verified using
  [the evidence retrieval instructions](evidence-retrieval.md#download-and-verify).
- A collaborator using a separate client can find and retrieve those bytes.
- Terminal delivery status is checked separately from scientific completion.

Do not manually re-upload an entire study store after each run. The maintained
export includes the selected attempt and its dependencies; overlapping exports
can still repeat dependency bytes. Account for that duplication in storage
estimates. If a terminal run's delivery fails, use the existing-run
[tracking refresh](joint-execution-records.md#live-tracking-navigation).

## Bring existing runs into the Jaxborg database

Changing `MLFLOW_TRACKING_URI` affects future logging. It does not merge old
SQLite databases, rewrite old artifact URIs, or index existing `mlruns/`
directories. Treat historical consolidation as a separate, explicit import.

1. Inventory source stores and terminal runs. Identify each by its source store
   and original run ID, plus its retained attempt identity when available.
   Inventory metrics, artifact paths, hashes, sizes, and referenced dependencies.
   Keep active stores and queued jobs untouched.
2. Prepare an import manifest under `mlflow/imports/<batch>/` with proposed
   destination experiments, source identities, selected complete evidence
   bundles, and estimated metadata/artifact bytes. Identify repeated attempts
   across exports before uploading them again. Preserve all required source,
   policy, sidecar, resume-state, and input dependencies.
3. Import through the destination MLflow API. Preserve metric histories and
   original timestamps, parameters, scientific identities and source status;
   retain source-to-destination experiment and run ID mappings. Destination run
   IDs may differ. Make retries identify already imported source runs rather
   than creating duplicates. Do not concatenate databases or copy source
   `mlruns/` directories into the live artifact store as an ingestion method.
4. Verify counts, histories, artifact downloads and hashes through the new
   service, including a collaborator's client. Record exclusions and failures.
   Keep original stores until the verified import and any cleanup are reviewed.

The [existing-evidence adapters](local-evidence-producers.md) produce research
records; they do not perform this MLflow migration. Likewise,
`reconcile --sync-tracking` refreshes a uniquely matching run already in the
configured destination; it does not create historical runs in an empty database.
A reviewed importer is required for bulk consolidation. Until then, link the
existing stores from the study records and direct new runs to the shared service.

## Checkpoint budget

Before production, choose the required checkpoints and estimate retained bytes
across all conditions, seeds, attempts, and artifact exports. Checkpoint cadence
is a storage decision, independent of how frequently scalar metrics are logged.

- Default to one periodic checkpoint per **four hours of elapsed training per
  run**, excluding queue time. Use **two hours** when interruption risk or restart
  cost warrants it, and record the reason in the execution settings. Shorter
  intervals require a concrete scientific or recovery need.
- Retain the trainer's required initial state, the completed final checkpoint,
  and exact scientific milestones declared by the protocol. These are exceptions
  to the recovery cadence. A short run normally needs no periodic recovery save.
  Do not change milestone selection in an already running or sealed study.
- Keep logging scalar learning curves at the resolution needed for analysis;
  do not save model weights or full optimizer state merely to record a metric.
- For the current JAX IPPO trainer, set `jax.checkpoint_every_updates` explicitly.
  It counts rollout updates, not hours. Estimate steady-state seconds per update
  from a representative pilot and choose approximately
  `ceil(target_hours * 3600 / seconds_per_update)` updates per save. Record the
  estimate and resolved stride. This approximates elapsed time; the trainer
  does not enforce a wall-clock checkpoint timer.
- Audit `mlflow.checkpoint_eval.every_steps`, explicit `--checkpoint-update`
  arguments, team overrides, and checkpoint-history evaluation suites as well.
  Inline evaluation can trigger extra durable checkpoints, and recorded
  Blue/FSM and Red/frozen-Blue paths save full `.npz` resume state alongside
  policy weights at retained checkpoints. A sparse periodic stride alone does
  not bound all saves. Keep only scientifically required evaluation milestones.
- Budget for `.safetensors`, `.npz`, recipe sidecars, native outputs, retained
  evidence copies, and MLflow exports including repeated dependencies. Measure
  actual pilot sizes and extrapolate to the full study before submission.
- Verify the first production saves against the budget. Configure new runs to
  avoid unnecessary saves; do not delete checkpoints from immutable evidence
  manifests, break resume lineage, or prune existing stores without a separately
  reviewed cleanup plan.

Retain the cadence and storage estimate in execution settings beside the study's
protocol. Mention cadence in the research summary only when checkpoint selection
affects the comparison. Existing recipes, queued jobs, and historical stores are
not changed by this guide.
