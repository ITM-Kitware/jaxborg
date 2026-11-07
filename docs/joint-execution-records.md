# Recorded joint training and evaluation

The maintained joint trainer can retain a declared research execution before
initialization, then save completed rollout updates, per-team update counts,
owned checkpoints and terminal status. Use an exact source revision and an
explicit protocol/trial. Set `JAXBORG_EXP_DIR` outside the checkout:

```sh
python scripts/train/algorithms/ippo_jax.py --recipe /inputs/recipe.yaml \
  --evidence-root /results/evidence --protocol /inputs/protocol.json \
  --trial training-cell --expected-revision FULL_COMMIT_SHA
```

The CLI prints the evidence root and attempt ID on success or after a retained
failure. Use that attempt ID for an explicit retry.

Recorded training supports fresh zero-sum joint JAX training and fresh Blue-only
training against the native stock FSM Red. Fixed and adaptive update schedules
use the existing joint trainer; Blue-only training updates Blue every rollout,
uses global gradient clipping, and requires rollout length to equal the episode
horizon. Blue team architecture overrides apply to both training and checkpoint
metadata. The legacy `joint_rollout_updates` budget unit denotes outer rollout
batches for either path; Blue-only records retain zero Red policy updates.

Blue-only checkpoints retain an initial policy, periodic policies, numeric
initial state, and exact effective observation/architecture metadata. Its legacy
inline checkpoint evaluator uses CybORG; disable inline evaluation and run a
separately recorded JAX evaluator when native JAX measurements are required.
Explicit topology snapshots and recipe-generated banks are retained as exact
input bytes after the ordinary recipe projection materializes them. Generated
banks retain their generation settings and snapshot metadata; recording does
not substitute a different generator or sampling procedure. Frozen learned opponents,
joint-training resume and auxiliary mission banks remain unsupported by this recorded path.
`--development-no-recording` explicitly opts out of research records.

For recorded Blue/FSM training, every retained policy checkpoint also owns a
numeric full-state archive containing parameters, optimizer state, reward
normalizer, RNG, environment state and observations. Use `--checkpoint-update N`
to retain an additional cumulative rollout milestone. Resume into a new output
tag and attempt with `--resume-state PATH_TO_RETAINED_NPZ`, keeping the original
seed, complete recipe/horizon, source revision, dependencies and topology bytes.
The checkpoint and original zero-step policy artifacts must belong to the same
evidence store and pass ownership/hash checks; cross-store resume is unsupported.
The output tag/directory must be fresh. The loader rejects mismatched contracts
and restores the complete state rather than restarting the optimizer from policy weights. Continuation is limited to
aligned episode/rollout boundaries; recurrent carry resets there by design.

Continuation attempts count only newly executed updates/transitions in their
budgets and retain cumulative counts in each metrics row and checkpoint. Their
initial checkpoint truthfully records the restored step; `training_origin`
retains the original zero-step policy identity. The initial diagnostic NPZ
described below remains separate from these full-state continuation archives.

The configured `scripts/eval/eval_scripted_reds_jax.py` post-training job gets
its own recorded evaluation over the training-owned final checkpoint. It
retains the source and effective recipes, topology inputs, evaluator source,
per-opponent episode results, CIA observations, and policy producer identity.
Other configured suites keep their existing execution and outputs; they are
not automatically converted into this recorded evaluation format. Declare
which suites belong to a study in its recipe instead of assuming every native
output has the same retention contract.

Periodic checkpoint evaluation keeps its existing cadence and settings. Its
metrics use the training-owned MLflow run, while training capture owns the
checkpoint and sidecar bytes, including evaluation-only temporary checkpoints.
The evaluation still runs if tracking is unavailable; its returned means are
retained in the training record's `checkpoint_evaluation_<steps>` fact.

Training completion is sealed before post-training evaluation. A subsequent
evaluation failure does not relabel successful training as failed. Completed
opponent batches remain available when a later opponent fails; an interrupted
batch may have no retained observations. Requested evaluation horizon and
observed episode transitions are distinct: this scripted path records the
horizon and does not invent an actual transition count when unavailable.

Use `jaxborg.evaluation.joint_capture.run_matchup` for independent evaluation.
It wraps
the maintained `evaluate_matchup` executor with the same policy loader, seed
expansion and scalar/batched rollout. Its supported inputs are JAX policies,
zero-sum payoffs and an explicit fixed topology bank. A study callback checks the
final resolved variant and builder arguments before execution. A condition label
alone does not establish the treatment.

Each invocation gets a new attempt. Training checkpoint and sidecar artifacts
are owned by its training attempt. Evaluations reference those exact artifact
IDs; imported foreign policies and topology inputs have no invented training
parent. Topology hashes and available generator metadata are retained with the
copied snapshot bytes actually consumed. Terminal evidence is automatically
uploaded to the same execution run for native MLflow downloads.

A full source revision, entrypoint, declared local helper closure, dependency
versions, bound builder arguments, command, supplied recipe, overrides and
resolved settings accompany the execution. `--allow-dirty` is an explicit
local development mode that retains changed source bytes and a patch. Source
revision, imported local helper coverage and retained hashes are checked again
inside execution. An undeclared local import or changed source rejects completion.

Training retains initial policy weights and numeric optimizer, normalizer, RNG,
environment and observation diagnostics as NPZ arrays with a path/dtype/shape
manifest. These diagnostic leaves do not define a resume format. Later policy
bundles remain weights-only outputs. Blue/FSM continuation uses the separate
full-state archives above; joint checkpoints do not support continuation.
Environment transitions, joint rollout updates and each team's policy updates
have distinct budget units.

Team update counts describe PPO rounds, not optimizer minibatch steps. Use the
existing `updated` flag when interpreting zero losses on skipped updates. The
adaptive phase, gate state, window statistics and forced updates come from the
trainer's measurements. Retained timing facts describe their inclusion of
setup, compilation, recording and evaluation; fewer updates alone do not
establish reduced runtime.

Set `trajectories=True` on `run_matchup` to retain observations, actions, Blue
payoffs and terminal flags from the same executed JAX scans. Completed episode
records carry their actual transition count. These are JAX policy trajectories;
CybORG event exports have different semantics. No second rollout is presented
as the original trajectory. Failed or cancelled executions retain completed
observations and partial budgets.

For a corrected evaluation, pass the original evaluation ID, a reason and the
changed procedure/input description:

```python
correction = {
    "evaluation_id": original_evaluation_id,
    "reason": "Correct the explicitly identified procedure error",
    "changes": {"procedure": corrected_source_identity},
}
```

The versioned `evaluation_correction` extension must reference a terminal
evaluation of the same protocol and trial, without cycles. Originals remain
unchanged. Collation selects the new evaluation explicitly and retains the
correction edge; discovery indexes its original evaluation ID. Repeating an
evaluation always creates a new identity. Publication reuse is a storage action,
not evaluation reuse.

After a hard interruption, an attempt stays running until explicitly reconciled.
In its original host boot and PID namespace, inspect the execution and then run:

```sh
python -m jaxborg.evaluation.reconcile /results/evidence \
  --attempt STORE_QUALIFIED_ATTEMPT_ID --reason 'Observed writer termination'
```

The command requires the recorded local writer PID to be absent and records the
operator's reason and check before sealing the attempt as cancelled. A live
writer, a different host, boot or PID namespace, or an unavailable writer
identity rejects reconciliation.
A checkpoint file never establishes successful completion. A retry uses a new
attempt and `--retry-of` with the terminal attempt of that same trial.

Scientific training completion is sealed before final tracking delivery. Tracking
creation, metric or artifact delivery failures are logged and do not abort training
or change its retained scientific status. During-training scientific failures retain
partial work so an explicit retry can proceed.

## Live tracking navigation

Recorded training uses one owned MLflow run for its retained attempt, training
labels, parameters, curves and artifacts. Recorded native and joint evaluations
do so when `MLFLOW_TRACKING_URI` or
`JAXBORG_EXP_DIR` explicitly selects a destination. Without either setting,
evaluation still retains complete local evidence. `MLFLOW_EXPERIMENT_NAME`
optionally selects the experiment; an explicit tracking URI takes precedence over
the local store. These settings do not create or deploy a shared service.

In native MLflow views or `MlflowClient.search_runs`, use this filter:

```text
tags.`jaxborg.discovery.role` = 'execution' AND tags.`jaxborg.discovery.study_id` = '<study-id>'
```

`attempt_id`, `evaluation_id`, `protocol_id`, `trial_id`, `retry_of`, condition,
source and budget tags retain scientific identities and observed progress.
`configuration.json` and `execution.json` provide the resolved settings and
current records, including artifact identities. On completion, `evidence/` holds
a verifiable native store with the attempt and its retained dependencies.
Download it through MLflow and call `load_records(downloaded_directory)`; see
the [download guide](evidence-retrieval.md). No separate publication run is needed.

Execution runs always have `state=unverified`. `scientific_status` reflects the
retained execution; `delivery=current`, `pending` or `failed` describes tracking
delivery. `data_incomplete=true` retains a known gap in training
metadata, curves or artifacts. Terminal synchronization refreshes the execution
projection and terminal evidence without replaying training metrics; a known gap
keeps delivery and the
tracking run failed even when scientific completion is synchronized. Live
views are mutable and may be stale after interruption
or delivery failure. Verify downloaded evidence before using it; a tracking tag
does not establish file integrity. Evidence-upload failures remain visible and
can be retried on the existing run without changing scientific completion.

With the original destination configured, interruption reconciliation also
refreshes the uniquely matching execution run. If delivery fails after local
cancellation, retry tracking delivery from unchanged terminal evidence:

```bash
python -m jaxborg.evaluation.reconcile /evidence/study \
  --attempt '<store-qualified-attempt-id>' --sync-tracking
```

Projection or terminal evidence-upload failures can recover to `delivery=current`.
This operation
neither reseals nor resumes execution. Missing or ambiguous
tracking matches fail explicitly; it never creates a replacement run. Raw store
operations and reconciliation without a configured destination stay offline.
