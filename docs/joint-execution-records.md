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

The supported recorded path starts both policies fresh, uses zero-sum joint JAX
training, fixed update schedules and explicit topology snapshots. Prepare banks
before launching; generated banks, adaptive schedules, frozen opponents and
resume are rejected. Automatic checkpoint/post-training evaluation suites must
be disabled. Run independent recorded evaluations over the owned checkpoint
artifacts. `--development-no-recording` explicitly opts out of research records.

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
copied snapshot bytes actually consumed. The same store can be prepared with
`experiment_discovery prepare --kind evaluation-capture`.

A full source revision, entrypoint, declared local helper closure, dependency
versions, bound builder arguments, command, supplied recipe, overrides and
resolved settings accompany the execution. `--allow-dirty` is an explicit
local development mode that retains changed source bytes and a patch. Source
revision, imported local helper coverage and retained hashes are checked again
inside execution. An undeclared local import or changed source rejects completion.

Training retains initial policy weights and numeric optimizer, normalizer, RNG,
environment and observation diagnostics as NPZ arrays with a path/dtype/shape
manifest. These diagnostic leaves do not define a resume format. Later policy
checkpoints remain weights-only outputs; exact continuation is unsupported.
Environment transitions, joint rollout updates and each team's policy updates
have distinct budget units.

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
current protocol/attempt/evaluation projection. The full artifact graph remains
in retained evidence and verified packages.

Execution runs always have `state=unverified`. `scientific_status` reflects the
retained execution; `delivery=current`, `pending` or `failed` describes tracking
delivery. `data_incomplete=true` retains a known gap in training
metadata, curves or artifacts. Terminal synchronization refreshes the execution
projection without replaying those data; a known gap keeps delivery and the
tracking run failed even when scientific completion is synchronized. Live
views are mutable and may be stale after interruption
or delivery failure. Package search/fetch continues to require completed verified
evidence and never treats these runs as available packages.

With the original destination configured, interruption reconciliation also
refreshes the uniquely matching execution run. If delivery fails after local
cancellation, retry only the projection from unchanged terminal evidence:

```bash
python -m jaxborg.evaluation.reconcile /evidence/study \
  --attempt '<store-qualified-attempt-id>' --sync-tracking
```

Projection-only failures can recover to `delivery=current`. This operation
neither reseals nor resumes execution. Missing or ambiguous
tracking matches fail explicitly; it never creates a replacement run. Raw store
operations and reconciliation without a configured destination stay offline.
