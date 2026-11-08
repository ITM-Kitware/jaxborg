# LSTM competence against fixed FSM Red

Can the current separate actor/critic LSTM learn effective defense in the normal
stock game against the fixed standard FSM attacker? This is a competence control
for the archived co-training results, with ordinary network variation, normal
Green traffic, stock rewards and unrestricted stock Blue actions.

## Method and prespecified checkpoints

Train two independently initialized Blues, seeds 42 and 100, on the same retained
100-network stock bank (generator seeds 0-99). Keep enhanced observations (450
features), separate actor and critic LSTM trunks of width 256, learning rate
0.0003, gamma 0.99, GAE lambda 0.95 and reward normalization. Each PPO round uses
96 environments, 500-step rollouts, 16 minibatches and four epochs: 48,000
environment transitions per update. Learning rate is constant.

The protocol-v2 overnight budget is **125M requested steps: 2,604 complete
updates, or
124,992,000 actual transitions per seed**. The original primary comparison stays
at **update 1,041, or 49,968,000 transitions**. Save that exact milestone even
though it falls between the regular 20-update checkpoints. Retain initialization,
every twentieth update (about eight minutes at the measured throughput), the
primary milestone and the final overnight checkpoint. Each retained checkpoint
includes policy weights and a complete numeric training
state that supports exact continuation with the same source and configuration.

The user requested useful overnight work toward Sunday November 8 at 9 a.m.
Eastern. The measured v1 smoke suggested about 16.3 training hours per seed at
this extended budget, before extra full-state retention overhead. This is an
estimate, not a finish guarantee. The fixed hard resource deadline remains
Monday November 9 at 8 a.m. Eastern. Budget extension was declared before
production and does not select checkpoints based on evaluation outcomes.

## Correctness and resume smoke

The original smoke runs three complete updates (144,000 transitions) using the
production architecture, batch size, horizon, optimizer and exact topology bytes.
It additionally saves full state after update two (96,000 transitions). A separate
`smoke-resume-seed42` attempt restores that state and executes update three only.
Its new-work budget is one update and 48,000 transitions; cumulative progress is
three updates and 144,000 transitions.

Production is gated on complete retained budgets, stock rules, unchanged topology
bytes, finite checkpoints, changed recurrent actor and critic weights, and exact
equality of the scientific state after uninterrupted versus resumed update three.
The state comparison includes optimizer, RNG, environment, observations and
normalization state; producer metadata is excluded from the equality digest.
CPU evaluation must also reload both initial and final smoke checkpoints and
complete their FSM cases with finite returns. Three smoke updates need not improve
reward. Both smoke trials are excluded from scientific replicate counts.

## Evaluation and interpretation

Native CPU evaluation is separate from training. Inline checkpoint evaluation is
disabled because that existing single-team path invokes CybORG.

- **Interim:** update 200, or 9.6M transitions, against FSM for the scheduled
  progress review. This is a development check; it does not select the primary
  checkpoint or change the fixed training budget.
- **Primary:** initialization and update 1,041 (49.968M transitions), against FSM
  and the four saved learned Reds.
- **Final extension:** update 2,604 (124.992M transitions), against the same panel.

The common panel uses four stock layouts (seeds 200-203), episode roots
21000-21015, 500-step episodes and stochastic policy sampling. Exact retained
transfer-study topology bytes are checked against the prepared reserved bank.
Shared initial references are explicitly reused rather than re-executed when the
panel and opponent match. See [evaluate.py](evaluate.py) and
[eval_launch.sh](eval_launch.sh) for the recorded CPU stages.

Report each training seed separately. FSM return at the primary milestone is the
main competence result. Improvement confined to FSM indicates specialization;
improvement against saved Reds supports transfer to that panel. The saved Reds
were excluded from new Blue training but are from a related archive, not a verified
independent holdout population. Scripted evaluation retains CIA impact proxies;
learned-opponent evaluation retains team return. Reward alone does not establish
better mission alignment.

Historical co-trained LSTMs remain descriptive references because their exact
training source is unknown. A causal training-regime comparison needs fresh
matched co-training. FSM versus learned co-training also changes attacker behavior
and exposure, so this baseline does not isolate nonstationarity.

## Recording and versioned inputs

[protocol.json](protocol.json) declares the v2 trials and budgets. Source,
recipes, topology bytes, checkpoint weights and full training state, commands,
Slurm identities, training metrics, evaluation selections and terminal statuses
are retained. Failed or interrupted attempts remain incomplete. Full-state resume
creates a new recorded attempt linked to the consumed state; it is not a new
independent training seed.

Protocol version 2 is the scientific design; execution-root suffixes identify
separate retained implementations. The v1 execution passed its basic smoke. The
v2 execution failed in the full-state serializer before acceptance. Both remain
preserved. The v3 execution uses source
`0aef148bbebb45818e71b20e9b5a52be031f93f1`, which fixes serialization of weakly
typed nonscalar JAX state, under the same protocol-v2 design:

```text
~/jaxborg-exp/shared/studies/lstm-fsm-competence-20261107-v3/
```

Preflight passed 60 focused ITM tests, GPU and CPU smoke checks, and exact
238-leaf full-state resume equivalence. The
[launch record](checks/20261107-overnight/execution-evidence.json) preserves those checks.

At **November 8, 2026, 10:59 a.m. Eastern**, training jobs **4128/4129** and final
evaluations **4132/4135** had completed. Both seeds reached **124.992M transitions**
with verified final checkpoint hashes. FSM return improved from the 49.968M
primary milestone to the final checkpoint for both seeds. Primary evaluation jobs
**4131/4134** failed after successful partial receipts; six learned-Red cells
remain missing, with CPU recovery pending. See the [results and limitations](results.md)
and [retained status snapshot](checks/20261108-overnight-results/status.json).

`--reuse-inputs` copies and rehashes the exact v1 topology bytes into the new
root, recording the previous source, manifest and topology hashes without
changing v1. MLflow uses `lstm-fsm-competence`; protocol and source identities
distinguish executions. See the [recorded training guide](../../docs/joint-execution-records.md).

## Commands

Use clean source at the pinned revision and the frozen runtime. These commands
show the execution workflow; preparation requires a new root, so use a separate
root when reproducing the already-created v3 execution. Reusing inputs copies
verified bytes without regenerating networks:

```bash
OLD="$HOME/jaxborg-exp/shared/studies/lstm-fsm-competence-20261107"
ROOT="$HOME/jaxborg-exp/shared/studies/lstm-fsm-competence-20261107-v3"
SHA=0aef148bbebb45818e71b20e9b5a52be031f93f1
JAX_PLATFORMS=cpu .venv/bin/python studies/lstm-fsm-competence/prepare.py prepare \
  --study-root "$ROOT" --expected-revision "$SHA" --reuse-inputs "$OLD"
bash studies/lstm-fsm-competence/launch.sh submit "$ROOT" smoke 42
```

After the original smoke completes, submit its exact continuation check:

```bash
bash studies/lstm-fsm-competence/launch.sh submit "$ROOT" smoke-resume 42
```

Run the CPU smoke once using the retained transfer manifest. Then validate all
gates and start the independent production seeds:

```bash
bash studies/lstm-fsm-competence/eval_launch.sh submit \
  "$ROOT" smoke-seed42 smoke "$TRANSFER_MANIFEST"
JAX_PLATFORMS=cpu .venv/bin/python studies/lstm-fsm-competence/validate.py \
  --study-root "$ROOT" --expected-revision "$SHA" --require-resume --require-cpu-eval
bash studies/lstm-fsm-competence/launch.sh submit "$ROOT" full 42 18
bash studies/lstm-fsm-competence/launch.sh submit "$ROOT" full 100 18
```

Each training allocation requests community, one GPU, eight CPUs and 32 GiB;
Slurm's `CUDA_VISIBLE_DEVICES` is preserved. Smoke allocations allow one hour.
These overnight production commands request an explicit **18-hour maximum**,
shortened if needed to finish at least ten minutes before the hard Monday
deadline. Launchers refuse
duplicate trial submissions. The two production seeds begin fresh, independently
of the smoke weights. Review warm update timing again after full-state capture is
enabled before claiming the overnight completion time.

For an explicit interrupted-production continuation, use the maintained trainer
with the same prepared full recipe, seed, pinned revision, protocol and trial;
provide a new output tag plus `--resume-state` pointing to a verified retained NPZ
and `--checkpoint-update 1041`. Retain the new attempt's receipt and its resumed
input provenance. Do not delete prior job records or pretend a weights-only file
contains enough state to resume.
