# Frozen defender transfer diagnostic

This study asks whether saved recurrent defenders retain useful performance
against a common attacker panel as the network and opponent change. It extends
the recorded evaluation workflow included in
[PR #55](https://github.com/ITM-Kitware/jaxborg/pull/55).

The saved policies use the 450-feature version 2 observation contract from
[Dena's H-MARL work](https://github.com/ITM-Kitware/jaxborg/pull/27).
The required IOC history and telemetry implementation is included with this study;
observations are not padded or truncated to load a checkpoint.

The first comparison uses saved policies. New training depends on the diagnostic:
if fixed-opponent performance is weak, investigate Blue learning; if competence
is present but opponent coverage is narrow, compare Blue-only learning against
one frozen Red with learning against a fixed Red bank. A later diversity test
must hold opponent exposure fixed. Fresh-Red searches require separate validation
and confirmation episodes and cross-testing every challenger against every Blue.
Those training interventions are not executed by this diagnostic launcher.

## Declared comparison

`prepare.py` selects the standard separate-encoder LSTM runs, single/diverse
training conditions and training seeds 42/100, at 960,000 steps and the final saved
model. These eight Blue policies face the same four final learned Reds. The
final-policy stage runs first; the early stage provides learning references.
Checkpoint ages and evaluation episodes are not additional training replicates.

Every cell uses 500 steps, stochastic policy sampling, and 16 explicit episode
roots 21000..21015 per topology. There is no score-based checkpoint or topology
selection. These are exploratory screening episodes, not a powered confirmation.

The network panels are:

- `anchor`: reconstructed stock-generator seed 0, declared in both training banks.
- `familiar`: reconstructed seeds 1..3, declared only in diverse training.
- `unseen`: reconstructed seeds 200..203, outside both declared training banks.
- `cia`: seeds 100..103 with three operational servers per zone, checked against
  archived canonical fingerprints and role assignments.

Original stock training snapshots and the exact training source revision were unavailable.
The published generator implementation matches the current generator semantics.
The first three panels therefore establish seed membership only. They cannot
establish a verified seen-versus-unseen memorization gap. The CIA panel changes
server-count constraints and is a separate benchmark.

Stock panels use FSM Red with fixed-role instrumentation on the unchanged
snapshots. Roles do not affect the FSM selector, Blue observations, Green behavior
or task rewards. A panel with fewer than three eligible servers is blocked for
this evaluator; its seeds are never silently replaced. The CIA panel reports FSM,
C, I and A targeting separately. Learned-Red matchups use the stock rules on the
anchor and unseen panels. Scripted and learned evaluators have different random
number contracts, so comparisons are paired within an evaluator family.

The retained measurements are team return and, for scripted evaluation, temporal
CIA role-host impact proxies. They are not measured exfiltration or corruption.
Green failure counts and action-level causes are not collected here. No inactive
Red or Sleep-Blue condition is claimed.

## Reproduce on ITM

Use a clean checkout of the recorded revision. Read the host's shared-machine
instructions and the shared experiment store's README first. All GPU execution
uses Slurm `community`, one GPU, eight CPU cores and 32 GiB memory. The launcher
preserves Slurm's device assignment and caps each submitted job at six hours.

```bash
UV_BIN="$HOME/.local/bin/uv" studies/defender-transfer/setup.sh
studies/defender-transfer/launch.sh submit \
  "$HOME/jaxborg-exp/shared/studies/defender-transfer-20261107" smoke
```

The setup uses `uv.lock` plus the explicit CUDA package pins in
`cuda-runtime.txt`. The PR's JAX extra does not install these required plugins
with this locked JAX version. Installed versions and device observations are
retained with execution; setup does not modify another study's environment.

The archive default is `/data/shared/jaxborg/jaxborg-separate`; supply an explicit
archive root as the fourth launcher argument for another installation. The archive
must contain the expected checkpoints, sidecars and benchmark evaluation JSONL.
`prepare.py --help` exposes the same deterministic preparation operation. Generated
inputs, evidence, caches, submissions and reports remain outside the checkout.

Inspect both smoke receipts and successful tracking delivery before production:

```bash
studies/defender-transfer/launch.sh submit \
  "$HOME/jaxborg-exp/shared/studies/defender-transfer-20261107" final
studies/defender-transfer/launch.sh submit \
  "$HOME/jaxborg-exp/shared/studies/defender-transfer-20261107" early
```

Run stages serially. A source change needs a new study root. Each cell receives
its own attempt and evaluation identity in the PR's evidence store and shared
MLflow experiment `defender-transfer-diagnostic`. Archived checkpoints are external
inputs with no invented training parent. Original recipes, effective recipes,
weights, topology bytes, resolved settings, sources and episodes are retained.

`selection.json` explicitly records successful receipts used by analysis. The
launcher stops on an execution error. Existing selected cells are not rerun;
corrected attempts and selection changes must be explicit. Smoke uses a separate
evidence root and selection and does not contribute production measurements.

Rebuild a report from saved evidence without running a rollout:

```bash
python studies/defender-transfer/analyze.py \
  --manifest /path/to/study/inputs/manifest.json \
  --root /path/to/study/evidence \
  --selection /path/to/study/selection.json \
  --output /path/to/new-report
```

The analyzer verifies selected artifact bytes and reports missing or blocked
cells. It reports per-topology outcomes and within-case descriptive contrasts,
including diverse-minus-single and final-minus-early comparisons. Two training
runs per condition do not support a precise population-level method claim.
A topology score drop can reflect intrinsic difficulty; the report does not
interpret that drop alone as a transfer failure.

## Recorded checks

- [Archived LSTM learning check, 2026-11-07](checks/lstm-learning-20261107/README.md):
  early/final learning and final Blue comparisons against common opposite-seed
  Red lineages. Includes the frozen 69-cell selection, evidence hashes, results
  and a CPU-only verification command. Keep later checks in separate named
  subdirectories while sharing the study's maintained analyzer.

## Matched reference screen

The [matched defender system screen](benchmark.md) compares the saved LSTMs and
released H-MARL systems under one current stock-game contract. Its protocol,
runner, evidence capture and interpretation limits are maintained here. It is
a separate named check within this study, with no new training.

The [CPU/GPU timing check](timing.md) measures the current scalar evaluator to
guide resource allocation; its repeated episodes are excluded from scientific
comparison counts.
