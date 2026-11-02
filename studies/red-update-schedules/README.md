# Red update schedules

## Question and evidence scope

How does changing Red's update schedule during co-training affect Blue's
performance against a fixed scripted Red suite? The comparison is intended to
identify which training configuration, if any, merits a larger study.

This directory defines the comparison and its analysis. No scientific results
are published here yet. The short smoke configuration checks the execution and
analysis path; it cannot establish learned performance or a preferred schedule.

## Compared training recipes

| Condition | Recipe | Scientific difference |
| --- | --- | --- |
| Baseline | [Baseline co-training](../../recipes/cotraining/separate/cotraining_separate.yaml) | Blue and Red update every rollout. |
| Quarter-rate Red | [Quarter-rate co-training](../../recipes/cotraining/separate/cotraining_red_quarter_rate_separate.yaml) | Blue updates every rollout; Red updates every fourth rollout. Original optimizer settings are retained. |
| Adaptive | [Adaptive co-training](../../recipes/cotraining/separate/cotraining_adaptive_updates_separate.yaml) | Adaptive update gating, Blue LR `1e-4`, Red LR `3e-4`, and learning-rate annealing disabled. |

Adaptive changes both scheduling and optimization, so this comparison cannot
isolate a gate effect. Its substantive settings retain the 200-Blue-update
warmup, 20-rollout return window, `-2500` threshold, and maximum 100 closed-gate
rollouts. The existing [scheduler](../../src/jaxborg/adaptive_updates.py) defines
the gate boundaries.

Each condition retains its preset architecture, observations, training rules,
reward normalization, and one generated training topology. Preparation replaces
the training budget and environment count with the declared study values.
Evaluate each trained policy at its final completed checkpoint.

## Evaluation and analysis

### Common evaluation procedure

Only the final JAX scripted `fsm`, `cia_c`, `cia_i`, and `cia_a` suite runs.
Preparation replaces `eval.after_training` and removes cross-play,
cross-seed-play, play-priors, checkpoint-scripted, and legacy scripted suites.
Checkpoint evaluation cadence remains disabled. This is a common evaluation
procedure, not a reproduction of every preset's original evaluation suite.

Evaluation uses stochastic policies, `cia_resilience`, fixed topology role
assignment, exhaustive held-out topology coverage, and the declared episode
seeds, episode count, and horizon. Evaluation topology generation starts at 100
with three operational-zone servers; training starts at 0. Episode seeds expand
through the [existing helper](../../src/jaxborg/evaluation/episode_seeds.py).
Prepared recipes, effective evaluator settings, and materialized topology bytes
are retained with execution records.

### Aggregation, missingness, and interpretation

- Episodes receive equal weight within a topology, topologies equal weight within
  an opponent, and independently trained policies equal weight within a condition.
  Opponents remain separate; there is no implicit panel average.
- A primary condition/opponent mean requires every declared training seed and
  its complete evaluation cases. Available seed values and ranges remain visible
  when the primary estimate is absent. Missing trials remain expected cells.
- Reward and CIA units follow the evaluator. More episodes do not create more
  independent training replicates. Matching seed labels across conditions do not
  establish paired initialization.
- Report each training-seed value and descriptive seed SD/range. With two training
  seeds there are two policies, regardless of episode count. Rankings are
  exploratory; the analyzer produces no confidence interval or superiority claim.
- Costs accompany performance. Update counts are completed team PPO rounds, not
  minibatch steps or instantaneous learning rates. Interpret losses with
  `team.*.updated`. Fewer Red updates alone do not establish a speedup.
  Retained timers have distinct scopes; evaluation wall seconds exclude process
  startup and artifact delivery. Unknown allocation or incomparable timing remains
  a limitation, and JAX device discovery does not establish scheduler allocation.

The analyzer verifies the declared source, recipes, trial/seed identity, final
checkpoint ownership, evaluation ancestry, evaluator settings, topology and role
identities, case expansion, and complete finite observations. Altered artifacts
fail verification. Changed evaluation inputs require a new protocol, including
a same-size replacement held-out bank. These checks are specific to this study.

## Run the comparison

Use [project setup](../../docs/setup.md), the shared
[training guide](../../docs/training.md#launch-training), and
[evaluation guide](../../docs/evaluation.md#post-training-evaluation) for the
environment, external storage, tracking, and launch procedure. Use a clean
committed source revision. Keep generated settings and evidence outside the
checkout. The [study guide](../README.md) explains protocols, attempts, retries,
and saved selections; [evidence retrieval](../../docs/evidence-retrieval.md)
covers tracking access and retained artifacts.

### Prepare and execute

Set the variables below to the intended source revision, seed lists, external
settings directory, and execution budget. Substantive preparation requires
explicit steps, environment count, topology count, horizon, resources, and
wall-time. Resource and wall-time descriptions record the plan; they do not
configure or submit a scheduler job.

```bash
uv run python studies/red-update-schedules/analyze.py prepare "$SETTINGS" \
  --mode substantive --expected-revision "$PINNED_REVISION" \
  --seeds "${TRAINING_SEEDS[@]}" --eval-seeds "${EVALUATION_SEEDS[@]}" \
  --steps "$TRANSITIONS" --num-envs "$ENVIRONMENTS" \
  --topologies "$EVALUATION_TOPOLOGIES" --horizon "$EVALUATION_HORIZON" \
  --episodes-per-seed "$EPISODES_PER_SEED" --resources "$RESOURCE_DESCRIPTION" \
  --wall-time "$WALL_TIME_LIMIT"
cat "$SETTINGS/commands.sh"
```

`prepare` writes ordinary recipes, a protocol, an empty selection template,
training commands, and its analyzer source. It launches nothing. Review the
generated commands and execute them through the site's supported launch path.
The existing trainer invokes the native post-training evaluator. Record actual
allocation and accounting when comparing compute cost.

For a pipeline check, replace the preparation options with `--mode smoke
--expected-revision "$PINNED_REVISION" --seeds 42 --eval-seeds 1000`, using a fresh
external settings directory. Smoke defaults to four 8-step rollouts, one
environment, one training topology, one held-out topology, one episode per
opponent, and an 8-step evaluation horizon. It reduces optimizer work and changes
adaptive warmup/window/frozen-rollout cap to `1/1/1`. Inspect adaptive diagnostics
to determine which paths executed. These settings do not reproduce the
substantive schedule.

### Select evidence and compare

After execution, inspect exact attempt, checkpoint, artifact, and evaluation IDs:

```bash
uv run python studies/red-update-schedules/analyze.py inventory \
  "$SETTINGS/evidence" > "$SETTINGS/inventory.json"
```

Copy `selection-template.json` to a new selection file. Selection is manual:
add one entry per chosen training replicate and one evaluation pick per
completed opponent. Each entry has this shape, using IDs from the inventory:

```json
{
  "condition": "baseline",
  "seed": 42,
  "attempt_id": "EXACT_TRAINING_ATTEMPT_ID",
  "checkpoint_id": "EXACT_FINAL_MODEL_ARTIFACT_ID",
  "evaluations": [
    {"artifact_id": "EXACT_RETAINED_JSONL_ARTIFACT_ID", "eval_id": "EXACT_FSM_ROW_ID"}
  ]
}
```

Add entries to the template's `selected` array and retain its protocol identity,
exploratory status, revision ancestry, and rationale. Omit absent or unsuccessful
cells; the report still lists them as missing. Select the declared final model
and its evaluations explicitly, never by latest timestamp or favorable score.
The JSONL artifact belongs to the evaluation record; its row identifies the
training attempt, model artifact, and model hash. The saved membership includes
selected attempts and their retry ancestry, so unrelated later attempts cannot
change the comparison.

```bash
uv run python studies/red-update-schedules/analyze.py compare \
  "$SETTINGS/evidence" "$SELECTION" "$COMPARISON"
```

`SELECTION` is the completed selection file and `COMPARISON` is a new external
output directory.

## Read and reconstruct results

The comparison bundle contains the report and the observations behind it:

| File | Meaning |
| --- | --- |
| `report.md` | Interpretation and limitations linked to retained tables |
| `episodes.csv` | Every selected reward/CIA observation and case identity |
| `cases.csv` | Per-topology and role-map mean within opponent/trained policy |
| `training-seeds.csv` | One score per trained policy/opponent; conditional episode SD |
| `comparison.csv` | Equal training-seed means, baseline differences, descriptive seed SD/range, completeness |
| `costs.csv` | Completed transitions, rollout/team PPO rounds, retained timers |
| `training-updates.csv` | Actual update flags/counts and adaptive diagnostics |
| `missing.csv` | Expected condition/seed/opponent cells without a complete selection |
| `analysis-inputs.json` | Exact selection, protocol, record facts, raw rows, and training updates |
| `analyze.py`, `selection.json`, `manifest.json` | Retained analyzer, membership, and byte identities |

Reconstruct a retrieved bundle into a fresh directory using Python 3.11+ and its
standard library:

```bash
python /retrieved/comparison-01/analyze.py rebuild \
  /retrieved/comparison-01 /fresh/reconstructed-01
```

`rebuild` verifies the manifest and regenerates byte-identical tables and report.
The bundle retains analysis inputs and provenance, not model weights or the
complete producing source archive. Retrieve those separately when auditing or
reevaluating models, and authenticate the manifest digest through the trusted
evidence destination.

For a changed metric or aggregation, maintain the revised study analyzer and
write a new bundle from the retained observations:

```bash
python /path/to/revised/analyze.py revise /retrieved/comparison-01 \
  /external/comparison-02 --rationale 'Explain the question and analysis change'
```

`revise` verifies the parent, retains revised source, marks the new result
exploratory, and binds the parent manifest digest. New evidence membership needs
a new selection and `compare` output, with its ancestry and rationale recorded.
The original bundle remains reconstructable with its original analyzer.
