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

## Execute and reconstruct

Status: prepared; runtime validation and submission are pending. No new evaluation
result is claimed here. Use a separate clean checkout of the committed revision;
leave the checkout pinned by existing transfer jobs unchanged.

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
