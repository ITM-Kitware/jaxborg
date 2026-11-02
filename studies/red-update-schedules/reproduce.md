# Reproduce the red update comparison

The [study summary](README.md) defines the question, conditions, scientific
methods, and evidence status. This companion describes how to run its analyzer
and locate the generated results.

Use the shared [training guide](../../docs/training.md#launch-training),
[evaluation guide](../../docs/evaluation.md#post-training-evaluation), and
[study guide](../README.md) for environment, execution, and evidence conventions.

## Output locations

- `SETTINGS`: a new external directory receiving prepared recipes, protocol,
  commands, and the selection template. Execution records go in its `evidence`
  subdirectory.
- `SELECTION`: the completed selection JSON file that you create after inspecting
  execution records.
- `COMPARISON`: a new external directory written by `compare`. It contains
  `report.md` and the tables listed below. Names such as `comparison-01` in the
  reconstruction examples are illustrative output directories, not bundled
  datasets or existing results.

## Prepare and execute

Supply the study settings explicitly:

- Training: seed list, transition budget, and parallel environment count.
- Evaluation: held-out topology count, episode seed list, episodes per seed,
  and horizon.
- Preparation: settings directory, source revision, resource description, and
  wall-time. The last two record the plan; they do not submit a scheduler job.

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

`prepare` writes recipes, a protocol, `selection-template.json`, training commands,
and its analyzer source. It launches nothing. Review and execute the generated
commands using the shared training guide; the trainer runs the configured
post-training evaluation.

## Select evidence and compare

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

Add entries to the template's `selected` array and retain its other fields.
Omit absent or unsuccessful cells; the report still lists them as missing.
Select the declared final model and its evaluations explicitly, never by latest
timestamp or favorable score.

```bash
uv run python studies/red-update-schedules/analyze.py compare \
  "$SETTINGS/evidence" "$SELECTION" "$COMPARISON"
```

`SELECTION` is the completed selection file and `COMPARISON` is a new external
output directory.

## Read generated results

Start with `$COMPARISON/report.md`. The comparison bundle contains:

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

## Reconstruct or revise an analysis

Use the retrieved bundle's retained analyzer:

```bash
python /retrieved/comparison-01/analyze.py rebuild \
  /retrieved/comparison-01 /fresh/reconstructed-01
```

`rebuild` verifies the bundle and regenerates its tables and report. See
[evidence retrieval](../../docs/evidence-retrieval.md) for downloading and
verifying retained inputs; model weights are retrieved separately.

For a changed metric or aggregation, maintain the revised study analyzer and
write a new exploratory analysis of the retained observations:

```bash
python /path/to/revised/analyze.py revise /retrieved/comparison-01 \
  /external/comparison-02 --rationale 'Explain the question and analysis change'
```

New evidence membership needs a new selection and `compare` output. Record the
parent analysis and rationale when revising either the analysis or selection.
