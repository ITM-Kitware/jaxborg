# Capture a checkpoint evaluation

Start with the [experiment guide](../experiments/README.md) to choose a workflow.
For existing saved results, use [historical evidence import](local-evidence-producers.md).

`python -m jaxborg.evaluation.capture` retains evidence from a Torch `.pt`
checkpoint evaluated against scripted Red in CybORG. It uses the maintained
checkpoint loader, game variants, and rollout. The checkpoint must have its
`recipe_<tag>.yaml` or `.yml` sidecar beside `model_<tag>.pt`.

This path runs serially, with one explicit environment seed per episode and
a configured horizon between 1 and 500 steps. It
does not train a policy, attach to MLflow, resume an interrupted attempt, reuse
an evaluation, or support JAX checkpoints and custom environment builders.

## Declare the scientific cells first

Save a version 1 protocol outside the checkout. Trial IDs describe intended
observations, including trials that have not started. An attempt is one execution
of a trial. Episode seeds are subsamples within an evaluation, not independent
policy training replicates.

For example, this protocol compares action selection settings on a fixed policy
and matched episode cohort. Its replicate names describe evaluation cohorts.
They make no claim about independent training replication.

```json
{
  "schema_version": 1,
  "kind": "protocol",
  "id": "action-selection:protocol:1",
  "study_id": "action-selection",
  "revision": "1",
  "selection_rule": "explicit",
  "required_facts": {
    "attempt": ["resolved_config", "source_files", "source_identity"],
    "evaluation": ["resolved_config", "source_files", "source_identity"]
  },
  "trials": [
    {
      "id": "stochastic:cohort-1",
      "condition_id": "stochastic",
      "replicate_id": "cohort-1",
      "matched_unit_id": "cohort-1"
    },
    {
      "id": "deterministic:cohort-1",
      "condition_id": "deterministic",
      "replicate_id": "cohort-1",
      "matched_unit_id": "cohort-1"
    }
  ],
  "metric_definitions": [
    {
      "name": "blue_return",
      "unit": "score",
      "direction": "higher",
      "aggregation": "arithmetic mean",
      "sampling_unit": "episode conditional on fixed policy",
      "uncertainty": "none"
    }
  ]
}
```

Declare the protocol before launching any trials:

```bash
python -m jaxborg.evaluation.capture declare \
  --root /evidence/action-selection --protocol /inputs/protocol.json
```

The evidence root must be outside the source checkout. Each root holds one
immutable protocol. A changed protocol needs a new root and revision.

## Run and retain an attempt

```bash
python -m jaxborg.evaluation.capture run \
  --root /evidence/action-selection --protocol /inputs/protocol.json \
  --trial stochastic:cohort-1 --model /inputs/model_policy.pt \
  --seeds 17 18 --horizon 500 --policy-seed 100

python -m jaxborg.evaluation.capture run \
  --root /evidence/action-selection --protocol /inputs/protocol.json \
  --trial deterministic:cohort-1 --model /inputs/model_policy.pt \
  --seeds 17 18 --horizon 500 --policy-seed 100 --deterministic
```

Every valid invocation creates new store-qualified attempt and evaluation IDs.
Success and execution-failure responses include both IDs for subsequent selection
or an explicit retry.
Rerunning the same trial does not overwrite earlier evidence or create another
scientific cell. To declare a retry, pass `--retry-of` with the previous terminal
attempt's full ID. The predecessor must belong to the same protocol and trial.
There is no automatic choice among successful attempts.

A production execution requires a clean source checkout. `--allow-dirty`
permits a development execution and records that override. Each attempt retains
the package's Python sources, `pyproject.toml`, `uv.lock`, a source hash manifest,
Git revision, installed dependency versions and available VCS revisions, Python
and operating system observations, and evaluator settings. The retained lock is
the intended dependency resolution; the installed versions describe what ran.
Source changes during evaluation prevent a successful completion marker.

The evaluator consumes retained copies of the checkpoint and recipe. Imported
weights remain external artifacts with no invented training parent. Resolved
settings include the actual game variant and horizon, policy sampling mode,
environment seed for each episode, and separate policy RNG seeds. Each reset
retains the actual topology and any role assignment. Completed episode records
contain the observed Blue return and step count. A complete evaluation reports
the arithmetic mean only when the protocol declares the matching metric.

## Failures, cancellation, and missing work

Invalid launch settings or an unreadable recipe are rejected during preflight.
After preflight, the initial running records exist before checkpoint copying or
loading. Copy, model loading, and rollout failures retain failed records and
available artifacts. Keyboard interruption retains cancelled records. Completed
episodes remain inspectable, but neither failed nor cancelled evaluations gain
a completed aggregate metric.

Episode budgets count durable completed observations. Step counts from those
observations describe completed episodes; they do not include unobserved work
inside an interrupted episode. Terminal evidence is sealed against further
writes through the capture API. Integrity verification checks retained bytes
before success and export; it is not a cryptographic signature against someone
rewriting the entire store.

A hard kill can leave an attempt marked running. Keep it as incomplete evidence;
do not infer success from files or elapsed time. This slice has no automated
reconciliation or resume command. A fresh invocation can execute the same trial
with a new attempt ID. It cannot claim an explicit retry of an unresolved
running predecessor.

## Export and choose explicitly

```bash
python -m jaxborg.evaluation.capture export \
  --root /evidence/action-selection --output /evidence/records.json
```

Export verifies artifact sizes and hashes, then writes a new record file. A
preexisting export is never overwritten. Artifact URIs are relative to the
evidence root, not the exported file's directory. Copy the entire evidence root
when moving or sharing it. The source checkout and original checkpoint files
are not needed to inspect the retained bytes.

Use the record command with a separate selection JSON containing `protocol_id`,
`selected_attempts` (trial ID to attempt ID), `selected_evaluations`, and
`analysis` (source identity and resolved configuration):

```bash
python -m jaxborg.experiment_records \
  /evidence/records.json /inputs/selection.json /evidence/collation.json
```

Only explicitly selected successful attempts and evaluations contribute.
Unselected attempts remain in the saved evidence. A declared trial with no
execution is reported as `no_attempt`; an executed trial without a selection is
`no_explicit_selection`. Retry counts and per-episode observations do not change
the protocol's replicate identities. Metadata validation does not itself prove
that two scientific conditions are comparable.
