# Run storage and reproducibility

Set `JAXBORG_EXP_DIR` explicitly to one persistent shared project directory outside
every checkout for production runs. Reuse that root across studies, methods,
seeds and launch checkouts; do not create a new database for each study or run.
Set `JAXBORG_MLFLOW_EXPERIMENT` to the study's name to group its runs in MLflow,
and use `JAXBORG_CAMPAIGN` for a particular campaign. Model, evaluation and log
files have distinct run IDs beneath the common artifact store.

Development smoke checks and tests use separate disposable roots. Existing
databases and active jobs retain their original root until a separately verified
import or migration preserves their run IDs and artifact references. Changing
`JAXBORG_EXP_DIR` does not merge databases or make another store's runs visible.
The SQLite database is `<root>/mlflow.db`. New MLflow experiments explicitly use
`<root>/artifacts`; actual run locations come from MLflow's artifact URI. Repeated
names/recipes/seeds create different run IDs. An incompatible old experiment's
artifact location produces an error: choose a new root or explicitly set
`JAXBORG_MLFLOW_EXPERIMENT`. Existing locations are never rewritten.

For this workspace, the persistent project store is
`/home/local/KHQ/paul.elliott/jaxborg-exp/shared/`. Load its reusable
`store.env` before future launches and set the study name:

```bash
source /home/local/KHQ/paul.elliott/jaxborg-exp/shared/store.env
export JAXBORG_MLFLOW_EXPERIMENT=study-name
export JAXBORG_CAMPAIGN=campaign-name
```

The already running Stage B pilot keeps `jaxborg-exp/oracle-stage-b/`; its database
has not been moved, merged or redirected. Its reports identify that original
store. The shared store is ready for future launches, without changing the
running pilot's pinned source or isolated environment.

Each run starts before outputs are written. Its `manifest.json` contains schema,
source, configuration, environment, inputs, status and verified output identities.
`recipe.yaml` is the full effective recipe; MLflow parameters are just a searchable
index and may be truncated. `effective_config` records projected backend defaults,
CLI overrides and derived settings. Full game variants, observation/action sizes,
contract implementation hashes, seed specifications and source/dependency archives
identify the executed settings. A descriptive branch is not a source identity.
`JAXBORG_CAMPAIGN` optionally identifies a study. Input training revisions are
separate from the evaluator revision and every input has an explicit link/hash.

Source capture retains project files tracked by Git and narrowly scoped untracked
source during a development override. It excludes dotfiles except `.github`,
credentials, secrets, datasets, caches, environments and model/database files.
Source hashes and `source/source.tar.gz` identify the captured bytes; dependency
versions, installed Git origins and `uv.lock` identify the environment. Full
command, working directory, node and selected nonsecret runtime settings are kept.
Generators can be replayed from the archived implementation, dependency revisions,
resolved variants and the recorded seed specifications. External inputs are hashed;
legacy inputs are retained under `inputs/`, while canonical inputs link to their
existing owner. Cynex V2 trajectories additionally retain the generated topology.
No simulator construction or random-key handling changes are introduced here.

## Prepare and retain a pinned launch

Commit the implementation first. Use a dedicated checkout and its own environment;
never update/rebase that checkout or sync its dependencies while a job uses it.
Resolve the complete revision now, before scheduling:

```bash
# From an implementation checkout. Choose unused destinations.
launch_sha=$(git rev-parse HEAD)
git worktree add --detach /absolute/launch-checkout "$launch_sha"
cd /absolute/launch-checkout
uv sync --frozen --extra cuda     # use --extra cpu for a CPU smoke
export JAXBORG_EXP_DIR=/absolute/shared-project-root
export JAXBORG_MLFLOW_EXPERIMENT=study-name
./scripts/train/run.sh jax default 42
# Or submit a batch job (recipe plus ordinary trainer options):
./scripts/sbatch/run_ippo.sh default --seed 42
# Or run CPU CybORG:
./scripts/train/run.sh cyborg default 42
# CPU JAX smoke:
JAXBORG_ALLOW_CPU=1 ./scripts/train/run.sh jax default 42 --num-envs 1 --total-timesteps 500
```

`run.sh` and the self-submitting batch wrapper resolve and retain HEAD under
`refs/jaxborg/launches/<full-sha>` and create a launch record under `<root>/launches/`.
Inside the allocation, `allocated.sh` verifies the expected full SHA, clean source,
checkout, exact interpreter, lockfile hash and installed dependency snapshot,
then checks the assigned JAX GPU backend. GPU discovery never runs on the
submitting host. GPU wrappers explicitly select community. Seed launches route
through `run.sh`. Runtime launch commands call the retained `.venv/bin/python`
directly; they cannot silently sync dependencies or modify a lockfile. Slurm's
startup log is under `launches/slurm/`; training console/metrics belong to the run.

Python entrypoints also reject dirty source and a mismatched
`JAXBORG_EXPECTED_SHA`. Direct JAX execution requires an allocation or explicit
`JAX_PLATFORMS=cpu`. For deliberately noncanonical development only,
`JAXBORG_ALLOW_DIRTY=1` records a patch, relevant untracked source, archived bytes
and the dirty label. Production wrappers always require clean source.

The workflow and snapshots support replay of inputs/configuration/code and
statistical replication. Main policy evaluation retains its existing `base_seed +
replica` expansion (overlapping base seeds can repeat episode seeds). Its torch
stochastic evaluator uses process RNG without a per-episode torch seed; this is
recorded as a replay limitation, not silently changed. Single-worker evaluation
retains its pre-model-load CPU RNG state; spawned worker sampling states remain
explicitly unknown. JAX policy sampling uses the first base seed times 100003
plus the flattened episode index, as in the existing runner. GPU training across different hardware/software is not
guaranteed bit identical. Portable policy weights do not save the optimizer,
PRNG, normalization, recurrent/environment state needed for full resume. The
CybORG final training-state artifact saves some state, but no full resume is
implemented. Re-running creates a new attempt; continuing training would need a
separate implementation. Do not pass an MLflow run ID to resume these trainers.

## Publish, inspect and export

Model files are staged temporarily outside the checkout. Native MLflow artifact
upload is followed by hash verification, local atomic rename and a manifest
completion record. Only fully written policy/sidecar pairs are advertised.
All completed checkpoints are retained. JAX preserves
`jax.checkpoint_every_updates` (default 50); CybORG adds
`cleanrl.checkpoint_every_updates` or `--checkpoint-every-updates` (default 50),
at completed PPO update boundaries. In-progress updates are never marked complete.
Metrics and console logs are retained during execution. A failed/canceled run can
own a complete, loadable partial-training checkpoint without being successful.
`KeyboardInterrupt`/SIGTERM mark a run KILLED; other exceptions mark it FAILED.

```bash
.venv/bin/python -m jaxborg.tracking inspect RUN_ID
.venv/bin/python -m jaxborg.tracking resolve runs:/RUN_ID/checkpoints/checkpoint_960000.pt
.venv/bin/python -m jaxborg.tracking search --filter "tags.\`git.commit\` = 'FULL_SHA'"
```

References use `runs:/<run-id>/<relative-artifact>`. Resolve checks completion,
bytes and matching policy sidecar. It also supports evaluation/trajectory/plot
artifacts. Legacy model paths and their adjacent sidecars still load without
rewriting the originals. Readable names are never resolved to an arbitrary run.
Paths under another experiment root require selecting that root first.

`--output`, `--output-dir`, `--output-json`, `--summary-json` and
`--per-episode-json` are explicit exports: publish and verify the canonical output,
then atomically copy it to the requested destination and record its reference/hash.
Cynex V2 JSON and scoring JSON formats are preserved. Exports may be overwritten
at the caller's chosen destination; authoritative run outputs remain independent.
Omitting the optional Cynex export directory creates only canonical artifacts.
Use canonical references as inputs to retain validated lineage rather than relying
on an exported filename. Legacy exported inputs with no provenance remain unknown
and are retained/hashes recorded. Export paths are never the basis for reuse.

## Independent evaluation, reuse and corrections

```bash
.venv/bin/python scripts/eval/eval_recipe.py \
  --model runs:/TRAIN_RUN/checkpoints/model_default_seed42.pt \
  --seeds 42-51 --episodes 10 --workers 1
# Explicit reuse; reports the actual original evaluation ID:
.venv/bin/python scripts/eval/eval_recipe.py \
  --model runs:/TRAIN_RUN/checkpoints/model_default_seed42.pt \
  --seeds 42-51 --episodes 10 --workers 1 --reuse
# Score exact trajectory inputs in a separate comparison run:
.venv/bin/python scripts/eval/score_trajectories.py runs:/TRAJECTORY_RUN/trajectories
```

Default invocations execute new evaluations. Only `eval_recipe.py --reuse`
requests reuse. The fingerprint covers model/sidecar contents, full effective
recipe/game contracts, topology identities/specification, seeds, episodes,
determinism, evaluator SHA/source bytes, installed dependencies and schema.
Every advertised output must still be complete and hash-valid. Changed bytes at
the same path, an evaluator revision, dependencies, seeds, topology or settings
invalidate reuse. Reuse returns the original evaluation ID and executes no new run.
Corrected evaluations always execute, even if `--reuse` is also supplied.

For an evaluation-only bug, find/annotate affected runs and evaluate the unchanged
canonical model using committed corrected code in a new clean checkout:

```bash
.venv/bin/python -m jaxborg.tracking search --filter "tags.\`git.commit\` = 'OLD_EVALUATOR_SHA'"
.venv/bin/python -m jaxborg.tracking annotate OLD_EVAL --bug issue-123 \
  --note 'Potential evaluator-only bug; unchanged checkpoint can be re-evaluated'
.venv/bin/python scripts/eval/eval_recipe.py \
  --model runs:/TRAIN_RUN/checkpoints/model_default_seed42.pt \
  --seeds 42-51 --episodes 10 --workers 1 \
  --supersedes-eval-run-id OLD_EVAL --bug-reference issue-123
```

The new evaluator SHA/fingerprint and `supersedes_eval_run_id` identify the
correction. The old metrics/artifacts remain intact. Training/environment bugs may
require retraining instead; unknown historical metadata stays unknown. This is a
manual investigation workflow, not automated impact analysis or archive migration.

A hard kill can leave MLflow RUNNING and a live console log without a terminal
callback. Inspect the job with `sacct`/`squeue`, check the launch/run identities and
verify each advertised checkpoint. After confirming the process has ended:

```bash
.venv/bin/python -m jaxborg.tracking reconcile RUN_ID --status KILLED \
  --note 'Slurm job JOB_ID ended by signal; completed checkpoints verified'
```

Reconciliation records the reason/status and preserves outputs. An unadvertised
`.pending` upload is incomplete. A staging directory left by a hard kill is
explicit disposable staging; do not advertise it as a completed model.

## Main coverage and research boundary

| Entrypoint | Coverage |
| --- | --- |
| `train/algorithms/ippo_jax.py` | Unique training owner, effective overrides, existing cadence, every policy sidecar, streamed metrics/logs |
| `train/algorithms/ippo_cyborg.py` | Same, configurable periodic policy saves, terminal training-state artifact, cancellation status |
| `eval/eval_recipe.py` | Independent evaluation, canonical/legacy inputs, explicit validated reuse, corrections, exports |
| `eval/baselines_jax.py`, `eval/baselines_cyborg.py` | Independent baseline evaluations with episode results/contracts |
| `eval/export_trajectory.py`, `eval/generate_cynex_trajectories.py` | Trajectory ownership, policy inputs, V2 export compatibility |
| `eval/cc4_trajectory_eval.py` | Trajectory ownership, exact model input, completed per-episode JSONL exports |
| `eval/score_trajectories.py` | Separate comparison owner with exact trajectory inputs and summary/per-episode outputs |
| `train/run.sh`, `train/run_seeds.sh`, `sbatch/run_ippo.sh` | Clean pinned source/environment checks; GPU verification inside allocation |
| `eval/benchmark_jax.py` | Performance developer benchmark, outside policy/scientific-result tracking scope |
| `scripts/dev/*` (including parity reporting/transfer) | Development diagnostics, outside experiment ownership scope |
| topology export CLI | Deterministic input generation utility; consumers should retain/hash the exported topology |

No required main adapter is deferred. This base has no separate scientific plot
consumer. Generic comparisons can use `Run(kind="comparison", inputs=[...])` and
publish under `plots/`; source evaluation links work identically for any artifact.
The core accepts multiple inputs with explicit roles; it does not assume Blue/Red
bundles or one parent. Research co-training/HMARL/joint IPPO/MAPPO and PR #26
adapters remain separate work under issue #28. Stage B/C require compatible
research adapters and their own smoke verification. The stock main smoke cannot
establish enhanced-observation compatibility. Stage A remains independent.
