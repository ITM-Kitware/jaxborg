# Matched defender system screen

## Question and decision

Do the four archived final LSTM defenders remain far behind the released H-MARL
Expert and Meta systems when all six run under the same current JAX stock-game
rules, initial scenarios, horizon and dynamics-seed ledger? A large gap motivates
investigating competence and policy behavior before another diversity treatment.
A small gap would weaken the claim that historical unmatched scores demonstrate
a major deficit. Neither outcome independently validates the trainer.

This is a comparison of complete defender systems. H-MARL has hierarchical action
masks, separate defender actors and engineered memory/communication; Expert also
uses a hand-coded IOC-based controller. The LSTMs receive the same version-2
observed IOC evidence through a different representation. Their action choices,
training histories and policy abstractions differ. The adapter does not expose
latent Red compromise or sessions as observations. This is not an architecture
ablation or a reproduction of published H-MARL/SOTA scores. The original paper's
full environment/rule correspondence has not been established.

## Fixed design

[benchmark-protocol.json](benchmark-protocol.json) declares six systems: final
single/diverse separate-encoder LSTMs at training seeds 42/100, plus imported
H-MARL Expert/Meta iter_49. There is no training or checkpoint search.

The full screen uses the first 16 entries of the previously frozen 100-case native
stock-CybORG scenario ledger, selected by index rather than return. All systems
receive those exact exported snapshot bytes. Each runs one stochastic 500-step
episode per case: 96 episodes in total. Case 99 is a disjoint preflight for one
LSTM and both H-MARL systems; successful preflight is required before the screen.

Both systems use the same current FsmRedCC4Env, stock FSM Red, CC4 blocking rules,
Green behavior, native action durations and unchanged reward components. There
is no fixed server-count override and no resilience-role assignment. H-MARL's
initializer must leave every environment-state/constant leaf unchanged. Each
trajectory must end at step 500 and its return must equal the component sum.

Policy and dynamics randomness use separate roots and a per-transition fold-in
schedule. Corresponding scenarios share dynamics roots across systems, but
policies change subsequent trajectories. Matching actor seed numbers does not
make different architectures take equivalent random draws. This RNG contract is
separate from the earlier transfer evaluator; old scores are not pooled here.

## Evidence and interpretation

The prior native/JAX H-MARL comparison already completed in ITM jobs 4006/4007/4008.
Its 100-case Expert means were -144.05 native and -83.55 JAX; Meta means were
-180.60 and -118.67. The roughly 60-point engine/adapter gaps are a reason to use
the same current simulator for this screen, not scores to combine with it.
Retained prior evidence is under:

```text
~/jaxborg-exp/shared/hmarl-native-matched-20261023/attempt-02/
  summary.json
  summary.txt
  cases.json
  topologies/
```

The new run retains copied weights, recipes, snapshots, source revision/archive,
dependency versions, protocol, explicit episode ledger, captured attempt records,
per-step actions/reward components and an explicit successful-attempt selection.
[benchmark.py](benchmark.py) reconstructs descriptive policy means and matched
scenario gaps from checked captured artifacts. Generated outputs remain outside
Git; this protocol, runner, launcher and findings belong to the existing study.

Sixteen scenarios are a screening sample, not 16 independent training replicates.
Report each trained policy separately. There are only two archived training seeds
per LSTM condition. A gap can reflect policy engineering, training exposure or
learning quality; this experiment alone does not identify its cause. The exported
scenarios are new to this evaluation, but exact historical training-layout novelty
is not established. A diversity claim still requires matched opponent exposure
and genuinely independent held-out opponents.

## Completed result: job 4112

All six saved systems completed the same 16 scenarios. Mean stock team return
is shown below; larger (less negative) is better.

| System | Training seed | Scenarios | Mean return |
| --- | ---: | ---: | ---: |
| LSTM, single training layout | 42 | 16 | -408.5625 |
| LSTM, single training layout | 100 | 16 | -573.6250 |
| LSTM, diverse training layouts | 42 | 16 | -377.3125 |
| LSTM, diverse training layouts | 100 | 16 | -368.1250 |
| H-MARL Expert | Imported checkpoint | 16 | -89.6250 |
| H-MARL Meta | Imported checkpoint | 16 | -107.6250 |

Every archived LSTM is below both system references on this sample. The deficits
are 278.5-484 points versus Expert and 260.5-466 versus Meta. Together with the
[early-to-final learning check](checks/lstm-learning-20261107/README.md), this
supports substantial learning with a remaining performance gap. It does not
establish a training bug, verify the current trainer, or establish paper-level
SOTA performance. The system differences described above remain confounds.

Diverse minus single is +31.25 for seed 42 and +205.5 for seed 100. This is a
descriptive advantage on these stock-FSM scenarios and two saved seed pairs;
it is not a general diversity effect or evidence of robustness to independent
held-out learned opponents. No significance claim follows from these means.

The checked-in [result record](checks/matched-system-20261107/record.json) retains
these values, analysis hashes and the [exact selected evaluations](checks/matched-system-20261107/selection.json).
Seven downloaded analysis artifacts were independently hash-verified.

The producing analyzer verified the captured records and exact 96-episode
ledger before writing these results. Authoritative evidence on ITM:

```text
~/jaxborg-exp/shared/studies/defender-benchmark-20261107/
  inputs/manifest.json
  selection-full.json
  evidence/
  analysis-full/means.csv
  analysis-full/paired-scenario-gaps.csv
  analysis-full/paired-mean-gaps.csv
  analysis-full/provenance.json
  analysis-full/hashes.json
  logs/4112.out
  logs/4112.err
```

Reconstruct with the execution revision above and a new output directory:

```bash
study="$HOME/jaxborg-exp/shared/studies/defender-benchmark-20261107"
JAX_PLATFORMS=cpu .venv/bin/python studies/defender-transfer/benchmark.py analyze \
  --manifest "$study/inputs/manifest.json" \
  --root "$study/evidence" \
  --selection "$study/selection-full.json" \
  --stage full \
  --output "$study/reviews/new-reconstruction"
```

## Execute and reconstruct

Status: **completed, Slurm job 4112, exit 0:0**, elapsed 25 minutes 31 seconds.
It finished at 2026-11-07 13:00:22 America/New_York; completion was verified at 13:17:27. All 96 full
episodes and all three smoke episodes completed. All 23 focused tests passed
on ITM before submission. Execution is pinned to revision
`0bc41414c10858503cced002e477279de70567a3` in the separate checkout
`~/src/cyber/jaxborg/defender-benchmark-20261107`. The active transfer checkout
and job 4107 were preserved. Shared MLflow experiment:
`defender-matched-system-benchmark`.

On ITM, read the host and shared-store instructions, then prepare the existing
pinned CPU/CUDA runtime with `bash studies/defender-transfer/setup.sh`. Submit:

```bash
bash studies/defender-transfer/benchmark_launch.sh submit \
  "$HOME/jaxborg-exp/shared/studies/defender-benchmark-20261107" \
  "$HOME/jaxborg-exp/shared/studies/defender-transfer-20261107-v4/inputs/manifest.json" \
  "$HOME/jaxborg-exp/shared/hmarl-native-matched-20261023/attempt-02" \
  "$HOME/jaxborg-exp/shared/hmarl-native-matched-20261023/inputs/actors.safetensors"
```

The launcher requests one GPU, eight CPU cores, 32 GiB and at most two hours in
`community`, preserves Slurm's device assignment, and activates the shared MLflow
store without printing its environment. Initial compilation is part of that
budget. Failed checks stop the run; retries require a new output root and explicit
selection rather than silently overwriting an attempt.

The launcher writes analysis automatically after the full screen. For manual
reconstruction, use `python studies/defender-transfer/benchmark.py analyze --help`
with the retained manifest, capture root, selection and a new output directory.
The exact CLI invocation and source revision are retained in the run evidence.
