# Red update schedules

Compare Blue performance against the same fixed scripted Red suite after fresh
co-training with [baseline](../../recipes/cotraining/separate/cotraining_separate.yaml),
[quarter-rate Red](../../recipes/cotraining/separate/cotraining_red_quarter_rate_separate.yaml),
and [adaptive](../../recipes/cotraining/separate/cotraining_adaptive_updates_separate.yaml)
recipes. The decision is which configuration, if any, deserves a larger study.
This directory supplies a prospective workflow. It contains no pilot performance
results, completed independent assessment, or authorized production workload.

Adaptive combines a schedule change with Blue LR `1e-4`, Red LR `3e-4`, and
learning-rate annealing disabled. It cannot isolate a gate effect. Baseline and
quarter keep their original optimizer settings. Substantive adaptive settings
retain the existing 200-Blue-update warmup, 20-rollout return window, `-2500`
threshold, and maximum 100 closed-gate rollouts. The existing
[scheduler](../../src/jaxborg/adaptive_updates.py) defines boundary semantics;
this study does not reimplement them.

## Declare fresh execution

Use a clean committed revision with the [recorded training path](../../docs/joint-execution-records.md)
and the repository environment. The existing trainer and native post-training
executor run the jobs. `prepare` only writes ordinary recipes, a protocol,
a selection template, commands, and the preparation analyzer source. Generated
evidence belongs outside the checkout.
Set `JAXBORG_EXP_DIR` outside the checkout and use a fresh tag/store for each pilot.
Follow the [tracking and retrieval guide](../../docs/evidence-retrieval.md) and
site launch rules before running commands on a shared machine.
Use a dedicated local MLflow instance with fresh storage for development,
following the tracking guide. Keep this pilot separate from other projects'
services. Remote compute can reach this local instance through the documented
private reverse tunnel; set its tracking URI to the forwarded endpoint.

A small pipeline check, separate from scientific evidence:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5062
export MLFLOW_EXPERIMENT_NAME=red-update-development
export JAXBORG_EXP_DIR=/absolute/external/smoke/runtime
uv run python studies/red-update-schedules/analyze.py prepare /absolute/external/smoke/settings \
  --mode smoke --expected-revision "$(git rev-parse HEAD)" --seeds 42 --eval-seeds 1000
cat /absolute/external/smoke/settings/commands.sh
# After reviewing the three short training/evaluation commands:
bash /absolute/external/smoke/settings/commands.sh
uv run python studies/red-update-schedules/analyze.py inventory \
  /absolute/external/smoke/settings/evidence > /absolute/external/smoke/inventory.json
```

Smoke uses four 8-step rollouts, one environment, one training topology, one
held-out evaluation topology, one episode per opponent and an 8-step evaluation
horizon. It changes optimizer minibatch/epoch counts for integration only.
Its adaptive warmup/window/frozen-rollout cap are explicitly changed to `1/1/1`
to reach the gate phase. Inspect retained adaptive diagnostics to establish
which paths actually executed. These settings do not reproduce the substantive
schedule and cannot support learned-performance conclusions. Test fixtures
also do not establish that an execution succeeded on a particular machine.

Substantive preparation has no step, environment-count, topology-count, horizon,
resource, or wall-time default. Supply the agreed values explicitly:

```bash
uv run python studies/red-update-schedules/analyze.py prepare "$SETTINGS" \
  --mode substantive --expected-revision "$PINNED_REVISION" \
  --seeds "${TRAINING_SEEDS[@]}" --eval-seeds "${EVALUATION_SEEDS[@]}" \
  --steps "$TRANSITIONS" --num-envs "$ENVIRONMENTS" \
  --topologies "$EVALUATION_TOPOLOGIES" --horizon "$EVALUATION_HORIZON" \
  --episodes-per-seed "$EPISODES_PER_SEED" --resources "$RESOURCE_DESCRIPTION" \
  --wall-time "$WALL_TIME_LIMIT"
```

These shell variables must be supplied from the reviewed execution budget.
Resource descriptions and wall-time limits are declarations, not scheduler
submission settings. Submit only the reviewed workload using the site's existing
launcher. Neither these instructions nor the old recipes authorize six 50M runs.
Record actual scheduler allocation, hardware, queue delay and accounting alongside
the evidence when evaluating cost. Keep compilation/startup/delivery timing scopes
explicit. Retain service access/retention details with the pilot handoff.

Every condition retains its original architecture, training rules, observations,
reward normalization, and one generated training topology. Steps round down to
complete `num_envs * episode_length` rollouts; the protocol retains requested
and completed budgets. The checkpoint rule is the final completed checkpoint,
not a favorable intermediate or a fixed historical step count. Retries use the
ordinary `--retry-of ATTEMPT_ID` flag and the same trial. A retry never adds an
independent training replicate. Inspect failed attempts and preserve partial output.

Only the final JAX scripted `fsm`, `cia_c`, `cia_i`, `cia_a` suite runs. Prepared
recipes explicitly replace `eval.after_training` and remove cross-play,
cross-seed-play, play-priors, checkpoint-scripted and legacy scripted suites.
This does not reproduce every original recipe's evaluations. Checkpoint evaluation
cadence remains disabled. Primary evaluation uses stochastic policies,
`cia_resilience`, fixed topology role assignment, exhaustive held-out topology
coverage, and the declared episode seeds/count/horizon. Default evaluation topology
generation starts at 100 with three operational-zone servers; training starts at 0.
The prepared recipe bytes, effective evaluator settings and materialized topology
bytes are retained by execution recording. The unmodified preset YAML files are
referenced above at the declared source revision; execution recording does not
copy those original preset files separately. Episode seed
expansion uses the [existing helper](../../src/jaxborg/evaluation/episode_seeds.py).

## Select and compare

Use `inventory` to inspect exact attempt/checkpoint/artifact/evaluation IDs and
statuses. Copy the template to a new selection file and fill each chosen entry:

```json
{
  "protocol_id": "red-update:EXACT_PROTOCOL_DIGEST",
  "exploratory": false,
  "revision_of": null,
  "rationale": "Initial declared analysis",
  "selected": [
    {
      "condition": "baseline",
      "seed": 42,
      "attempt_id": "EXACT_TRAINING_ATTEMPT_ID",
      "checkpoint_id": "EXACT_FINAL_MODEL_ARTIFACT_ID",
      "evaluations": [
        {"artifact_id": "EXACT_RETAINED_JSONL_ARTIFACT_ID", "eval_id": "EXACT_FSM_ROW_ID"}
      ]
    }
  ]
}
```

Add one evaluation pick per completed opponent and one entry per selected training
replicate. Omit absent or unsuccessful cells. The protocol still declares them and
the report shows missingness; partial evidence does not become a final policy.
Saved analysis membership includes only selected training/evaluation attempts and
their retry ancestry; later unrelated attempts do not change an existing selection.
No ID is selected by latest timestamp, directory name, favorable score or historical
70M-step assumptions. The JSONL artifact is owned by the evaluation record,
which consumes the selected training-owned model. Its row also identifies the
training attempt, checkpoint artifact, and exact model hash.

```bash
uv run python studies/red-update-schedules/analyze.py compare \
  "$SETTINGS/evidence" /absolute/external/selection.json /absolute/external/comparison-01
```

Compatibility checks cover the source pin in the hashed protocol, trial/seed,
declared condition recipe, final checkpoint
ownership, training policy/game/topology/source identity, evaluation input ancestry,
shared source bytes/revision, effective evaluator recipe/variant/seeds, horizon,
stochastic JAX resilience procedure, retained topology generator/seed/server metadata
and input-bound fingerprints, topology
fingerprints, role identities, case expansion and complete finite observations.
They are specific to this study, not a universal comparability checker. Altered
artifacts fail verification. A same-size alternate held-out bank cannot be labeled
as the original protocol. Declare changed evaluation inputs in a new protocol. There is no claim that equal seed numbers establish
paired initialization or cross-condition training replication.

The output is a self-contained analysis bundle:

| File | Meaning |
| --- | --- |
| `episodes.csv` | Every selected reward/CIA observation and case identity |
| `cases.csv` | Per-topology and role-map mean within opponent/trained policy |
| `training-seeds.csv` | One score per trained policy/opponent; conditional episode SD |
| `comparison.csv` | Equal training-seed means, baseline differences, descriptive seed SD/range, completeness |
| `costs.csv` | Completed transitions, rollout/team PPO rounds, actual retained timers |
| `training-updates.csv` | Actual update flags/counts and adaptive diagnostics |
| `missing.csv` | Every expected condition/seed/opponent lacking a complete selection |
| `analysis-inputs.json` | Exact selection, protocol, record facts, raw rows and training updates |
| `analyze.py`, `selection.json`, `manifest.json` | Analyzer bytes, membership and byte identities |
| `report.md` | Interpretation and limitations linked to these retained tables |

Episodes receive equal weight within topology, topologies equal weight within
opponent, and trained policies equal weight within condition. All expected seeds
must be present for a primary mean. Available seed values/ranges remain visible
when the primary estimate is absent. Opponents remain separate with no implicit
panel average. Reward/CIA units and scores follow the existing evaluator. Two
training seeds give two policies, regardless of episode count. Show both seed
values and describe ranking as exploratory; no confidence interval or superiority
claim is generated. A missingness gap cannot silently change the estimand.

Costs accompany performance rather than define an automatic ranking. Team update
counts are completed PPO rounds, not optimizer minibatch steps or instantaneous
learning rates. Use `team.*.updated` when interpreting loss diagnostics. Training
cost facts retain their timer scope; evaluation wall seconds cover selected rows
and exclude process startup/artifact delivery. JAX device discovery is not scheduler
accelerator allocation. Unknown/incomparable allocation or timing must remain a
limitation; fewer Red updates do not alone establish a speedup.

## Reconstruct and revise

Copy the complete comparison bundle to a fresh directory. Analysis reconstruction
needs Python 3.11+ and its standard library, without JAX, MLflow, a producer checkout,
or surviving branch names:

```bash
python /retrieved/comparison-01/analyze.py rebuild \
  /retrieved/comparison-01 /fresh/reconstructed-01
```

This verifies every manifest member and regenerates byte-identical tables/report
with the retained analyzer. The bundle retains analysis inputs and provenance;
it does not duplicate checkpoint weights or the entire producing source archive.
Retrieve and verify those separately through the execution store when auditing or
reevaluating a model. A manifest detects accidental change; authenticate its digest
through the trusted evidence destination when transferring a bundle.

A changed metric/aggregation belongs in a maintained study-local analyzer. Copy or
edit this analyzer, use the already retained `analysis-inputs.json` observations,
and run its `revise` command to a new output directory:

```bash
python /path/to/revised/analyze.py revise /retrieved/comparison-01 \
  /external/comparison-02 --rationale 'Explain the question and analysis change after inspecting outcomes'
```

`revise` verifies the parent bundle, retains the revised source, marks the result
exploratory and binds the parent manifest digest. It never replaces comparison-01.
Reconstruction of comparison-01 still uses its original source. New evidence
membership requires a new selection and `compare` output. Record parent identity,
exploratory status and rationale in that selection as well.

Complete the [handoff and assessment](handoff.md) before claiming portable
reconstruction, research usefulness or reduced effort. An inconclusive scientific
answer and an infrastructure assessment recommending simplification are valid.
