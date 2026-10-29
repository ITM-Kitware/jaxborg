# Capture a checkpoint evaluation

See the [experiment guide](../experiments/README.md) for workflow and recipe
conventions, or [historical import](local-evidence-producers.md) for saved results.

`scripts/eval/eval_recipe.py` records research evaluation by default. Supply an
evidence root outside the checkout and an explicit protocol/trial identity:

```bash
python scripts/eval/eval_recipe.py --model /inputs/model_policy.pt \
  --evidence-root /evidence/action-selection --protocol /inputs/protocol.json \
  --trial stochastic:cohort-1 --seeds 17,18 --episodes-per-seed 1 \
  --horizon 500 --policy-seed 100
```

The command requires a native Torch `.pt` checkpoint with adjacent
`recipe_<tag>.yaml` or `.yml`. Recording is serial; omitted workers means one,
and other worker counts fail before execution. The JSON receipt contains the
attempt/evaluation identities, including retained execution failures. Research
mode does not write legacy result rows or attach metrics to a training run.

Root seeds expand as `s * episodes_per_seed + episode_index`. Use
`--episode-seeds 17 18` for exact ordered episode seeds instead. Research policy
seeds start at `--policy-seed` (default zero) and advance by global episode index.
`--horizon` defaults to 500; `--eval-red` overrides the evaluation selector while
retaining the original recipe. The command retains its entrypoint source and pins
HEAD at invocation; use `--source-revision` to verify a revision chosen earlier
at submission. Capture stores the resolved settings and raw
observations. Condition labels do not validate these settings; study-specific
checks must establish treatment membership.

Use `--development-no-capture` explicitly for unrecorded development or the
currently unsupported JAX-to-CybORG recording path. That mode retains legacy
result/MLflow behavior and seeds native policy sampling from each expanded
environment seed unless `--policy-seed` is supplied. It cannot be combined with
protocol membership. Existing custom post-training entries must explicitly
supply research membership or choose the development opt-out; the runner does
not silently disable recording. Programmatic evaluation retains its existing
API and is not automatically a recorded research invocation.

The Python `run_evaluation` API takes unique explicit episode seeds. Capture
does not train, resume, reuse evaluations, or support custom builders on this
native Torch path.

## Declare scientific cells

Save a version 1 protocol outside the checkout. It declares intended trials,
including unstarted ones; runtime settings come from the retained recipe and
supported capture overrides. This example compares sampling modes on one fixed
policy and matched episode cohorts. Its replicates are evaluation cohorts,
not independently trained policies.

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

Declare the protocol before launching trials:

```bash
python -m jaxborg.evaluation.capture declare \
  --root /evidence/action-selection --protocol /inputs/protocol.json
```

Each evidence root must be outside the source checkout and holds one immutable
protocol. A changed protocol needs a new root and revision.

## Run and retain an attempt

```bash
python scripts/eval/eval_recipe.py \
  --evidence-root /evidence/action-selection --protocol /inputs/protocol.json \
  --trial stochastic:cohort-1 --model /inputs/model_policy.pt \
  --seeds 17,18 --episodes-per-seed 1 --horizon 500 --policy-seed 100

python scripts/eval/eval_recipe.py \
  --evidence-root /evidence/action-selection --protocol /inputs/protocol.json \
  --trial deterministic:cohort-1 --model /inputs/model_policy.pt \
  --seeds 17,18 --episodes-per-seed 1 --horizon 500 --policy-seed 100 --deterministic
```

Condition labels do not select or validate settings: `stochastic:cohort-1` does
not reject `--deterministic`. Study-specific checks must compare intended
conditions with retained settings. Required-fact checks establish presence and
knownness, not treatment equality or scientific comparability.

Each valid invocation creates fresh store-qualified attempt and evaluation IDs,
returned on success and execution failure. To declare a retry, pass `--retry-of`
with a terminal predecessor's full attempt ID from the same protocol and trial.
Reruns preserve earlier evidence; successful attempts are never selected
automatically.

Production requires a clean source checkout; `--allow-dirty` records a development
override. Each attempt retains the package's Python sources, `pyproject.toml`,
`uv.lock`, source hashes, Git revision, installed dependency/VCS versions,
Python/OS observations, and resolved settings. The lock records intended
dependencies; installed versions record what ran. Source changes during execution
prevent success.

Execution loads retained checkpoint and recipe copies. Settings include the
variant, horizon, sampling mode, and environment/policy seed schedule. Resets
retain actual topology and roles; completed episodes retain Blue return and step
count. Success reports the declared arithmetic mean. Imported weights remain
external inputs without an invented training parent.

## Failures and incomplete work

Invalid settings and unreadable recipes fail preflight. Running records exist
before input copying and model loading; subsequent failures retain failed
records and available artifacts. Keyboard interruption records cancellation.
Completed episodes remain inspectable, but failed/cancelled evaluations have no
completed aggregate. Budgets count durable completed episodes and their steps,
excluding partial episode work.

Terminal evidence rejects further API writes. Hash/size checks before success
and export detect changed bytes, but do not authenticate a rewritten whole store.
A hard kill may leave an attempt running. There is no reconciliation or resume
command; a new attempt may run the same trial but cannot claim an explicit retry
of an unresolved running predecessor.

## Export and select

```bash
python -m jaxborg.evaluation.capture export \
  --root /evidence/action-selection --output /evidence/records.json
```

Export verifies artifact sizes/hashes and requires a new output file. Artifact
URIs are relative to the evidence root, not the export directory. Move the whole
root to preserve inspection without the original checkpoint or source checkout.

Supply a separate selection JSON with `protocol_id`, `selected_attempts` (trial
to attempt IDs), `selected_evaluations`, and `analysis` (source identity and
resolved configuration):

```bash
python -m jaxborg.experiment_records \
  /evidence/records.json /inputs/selection.json /evidence/collation.json
```

Only explicitly selected successful attempts/evaluations contribute. Others stay
visible: an unexecuted trial is `no_attempt`; an executed trial without selection
is `no_explicit_selection`. Retries and episode observations do not change the
protocol's replicate identities.
