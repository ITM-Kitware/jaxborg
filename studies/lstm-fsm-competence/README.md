# LSTM competence against fixed FSM Red

Can the current separate actor/critic LSTM learn effective defense in the normal
stock game against the standard, fixed FSM attacker? This provides a missing
competence control for the archived co-training results. It is not a new
environmental-diversity comparison.

## Method

Train two independently initialized Blues, seeds 42 and 100, against stock FSM
Red. Keep normal Green traffic, stock rewards, action rules and subnet scaffold.
The training bank contains 100 ordinary generated network instances, seeds 0-99;
the same exact snapshot bytes are used by both training seeds. This retains
ordinary network variation instead of artificially restricting the network.

The [recipe](recipe.yaml) preserves the archived diverse-LSTM Blue settings:
enhanced observations (current contract: 450 features), separate actor and critic
LSTM trunks, hidden width 256, learning rate 0.0003, gamma 0.99, GAE lambda 0.95,
reward normalization, 96 parallel environments, 500-step rollouts, 16 minibatches
and four PPO epochs. One update contains 48,000 environment transitions. The
50M requested budget completes 1,041 whole updates, or **49,968,000 transitions**
per seed. These are environment transitions, not agent transitions.

Keep initialization, checkpoints every 20 updates (960,000 transitions), and the
last planned checkpoint. Select that final checkpoint in advance, without
looking for the best test score. Inline checkpoint evaluation is disabled because
that existing single-team path invokes CybORG; native CPU evaluation is a separate
recorded operation. Training does not inspect held-out evaluation outcomes.

A separate seed-42 implementation smoke performs three updates (144,000
transitions) with **identical** parallelism, 500-step horizon, architecture,
optimizer and topology bytes. Its first update includes compilation; subsequent
synchronized updates support a runtime estimate. The smoke is excluded from
scientific replicate counts. Production starts fresh, and does not resume its
weights or optimizer state.

## Evaluation and interpretation

Use [evaluate.py](evaluate.py) and [eval_launch.sh](eval_launch.sh) on CPU. The
implementation smoke checks initialization and its last checkpoint on a separate
FSM case. The scientific panel evaluates initialization and the last planned
checkpoint on four stock
layouts (generator seeds 200-203) and episode roots 21000-21015. The evaluator
uses the retained defender-transfer inputs, checking topology identities before
claiming agreement with this study's reserved bank. Every policy receives the
same cases and 500-step horizon, with stochastic policy sampling.

Report fixed-FSM return as the primary competence outcome, with separate results
for both independent training seeds. Evaluate the same policies against the four
saved learned Reds as a secondary transfer panel. These Reds were excluded from
the new Blue training, but come from a related archive; they are not a verified
independent holdout population. The scripted path retains CIA impact proxies; learned-opponent matchups retain
team return. Higher return alone does not establish better mission alignment.
Intermediate training checkpoints remain available for a separately specified analysis.

Improvement limited to FSM indicates task-specific competence. Improvement on
other Reds is evidence of transfer to this fixed panel. Weak results motivate
targeted learner or observation investigations. Comparing these runs with old
co-trained LSTMs is descriptive: the exact historical training source is unknown.
A causal training-regime comparison requires fresh matched co-training runs.
FSM versus learned co-training also changes attacker behavior and exposure, so
the comparison does not isolate nonstationarity.

## Status and retained evidence

Implementation and validation are in progress; no scientific result is implied
by these files. [protocol.json](protocol.json) declares the smoke and two
production trials, budgets and checkpoint-selection rule. Failed or cancelled
attempts remain incomplete; a checkpoint does not establish completed training.

Generated inputs, checkpoints, logs and results live outside the checkout, under
the execution-specific study root. On ITM the planned location is:

```text
~/jaxborg-exp/shared/studies/lstm-fsm-competence-20261107/
```

The shared MLflow experiment is `lstm-fsm-competence`. Each training attempt
retains its exact source, supplied and effective recipe, topology bytes, initial
state diagnostics, checkpoint bytes and sidecars, metrics, actual budget and
terminal status. The runner additionally retains commands, full logs and the
training receipt in `runs/<trial>/`. Submit-time Slurm identities are under
`logs/`; `inputs/manifest.json` pins the prepared inputs and source revision.
See the [recorded training guide](../../docs/joint-execution-records.md).

## Commands and resource limits

Use a clean checkout at the committed revision and the project's frozen runtime.
Prepare the immutable banks once in a CPU allocation, with `JAX_PLATFORMS=cpu`
set before Python imports:

```bash
ROOT="$HOME/jaxborg-exp/shared/studies/lstm-fsm-competence-20261107"
SHA=$(git rev-parse HEAD)
srun --partition=community --cpus-per-task=4 --mem=16G --time=00:20:00 \
  env JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python studies/lstm-fsm-competence/prepare.py prepare \
    --study-root "$ROOT" --expected-revision "$SHA"
bash studies/lstm-fsm-competence/launch.sh submit "$ROOT" smoke 42
```

Validate the recorded smoke before either production submission:

```bash
JAX_PLATFORMS=cpu .venv/bin/python studies/lstm-fsm-competence/validate.py \
  --study-root "$ROOT" --expected-revision "$SHA"
```

This verifies complete budgets, fixed-FSM game rules, the 100 retained topology
hashes, finite checkpoints, training-owned artifacts, architecture dimensions,
and changed recurrent kernels in both actor and critic. It retains checkpoint
fingerprints and first/warm update timings in `runs/smoke-seed42/acceptance.json`.
Production submissions revalidate this gate and require two completed, finite
CPU smoke evaluations (initial and final FSM checkpoints), retained in
`runs/smoke-seed42/production-ready.json`. Three updates need not improve
reward; the gate verifies execution and learning mechanics, not performance.

Review the smoke's recording integrity, finite losses, policy updates,
checkpoint reload and native CPU evaluation before starting production. Use the
warm synchronized update times to check that the fixed budget fits the window:

```bash
bash studies/lstm-fsm-competence/launch.sh submit "$ROOT" full 42 36
bash studies/lstm-fsm-competence/launch.sh submit "$ROOT" full 100 36
```

Each training job requests community, one GPU, eight CPUs and 32 GiB. The launcher
preserves Slurm's `CUDA_VISIBLE_DEVICES`, verifies the pinned source and frozen
input hashes, and refuses duplicate submissions. The final argument is a maximum
walltime in hours; production defaults to 36 and smoke is limited to one hour.
Actual allocation and process limits are shortened to finish at least ten minutes
before **Monday November 9, 2026, 8 a.m. America/New_York** (13:00 UTC). A delayed
allocation rechecks this deadline before starting. Incomplete runs must not be
relabeled complete or automatically resumed.

Preparation refuses an existing study root. For an explicit retry, retain the
original attempt, use the trainer's recorded `--retry-of` mechanism and a new
output tag; do not remove job or attempt records to bypass duplicate protection.
