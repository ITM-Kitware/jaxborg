# jaxborg

JAX port of [CybORG CAGE Challenge 4 (CC4)](https://github.com/cage-challenge/cage-challenge-4) using [JaxMARL](https://github.com/FLAIROx/JaxMARL) for GPU-accelerated parallel RL training.

CybORG CC4 is a multi-agent cybersecurity simulation (9 subnets, ~80 hosts, 5 blue agents, 6 red agents, 3 mission phases). This project re-implements CC4's environment logic as JIT-compilable JAX arrays for massively parallel simulation on GPU. Parity is verified by:

- **Differential testing** — lockstep state comparison after every step
- **TOST equivalence testing** — statistical comparison of independent rollout rewards across both engines

## Results

[Trajectory Visualizations](https://cynex-trajectories.netlify.app/)

### Speed

| Engine  | Parallelism                  | Steps/sec | 20M wall time |
| ------- | ---------------------------- | --------- | ------------- |
| CybORG  | 48 CPU processes             | 332       | 16.7 h        |
| jaxborg | 1,024 vectorized envs on GPU | 2,512     | 2.2 h         |

**~7.5x throughput**, **~7.5x wall-time** on a single NVIDIA RTX A6000. jaxborg's entire training loop (rollout + GAE + PPO update) compiles to one XLA program; first-run compile takes ~7 min (cached thereafter).

### Parity

#### TOST Equivalence

We verify that jaxborg reproduces CybORG's behavior using the TOST (two one-sided t-test) equivalence procedure from [Karten et al. (2026)](https://arxiv.org/abs/2603.12145). Two independent claims:

| comparison                                                                | n   | gap         | TOST                            | verdict                            |
| ------------------------------------------------------------------------- | --: | ----------: | ------------------------------- | ---------------------------------- |
| Same trained policy, jaxborg vs CybORG env (3 seeds × n=100, pure mode)   | 300 |  +109 ± 30  | Δ=±284, p=2.8e-7                | **EQUIVALENT**                     |
| Cross-policy matched training: jaxborg-trained vs CybORG-trained, on CybORG env (3 seeds × n=100, paired) | 300 |   +5.4 ± 58 | Δ=±200, p=4e-4 / Δ=±284, p<1e-6 | **EQUIVALENT** at Δ=±200 and ±284  |

Δ=±84 = 2σ across same-backend seed means (noise floor); Δ=±284 = 5% of the sleep→trained learnable signal span. See [`docs/parity.md`](docs/parity.md) for the full per-test parity index and tolerances.

#### Training Comparison

Matched-hyperparameter training (NUM_ENVS=48, 3 seeds, same shared-trunk actor-critic, identical PPO hparams). Reward is the mean training-time episode reward at 3M steps; each policy is on the env it trained against (the ~145-pt gap between the two columns reflects independent green-RNG host-selection between engines, not a parity bug — see [`docs/parity.md`](docs/parity.md)).

| Run          |          Reward (mean ± σ across 3 seeds) | Steps |
| ------------ | ----------------------------------------: | ----- |
| CybORG PPO   | -1,854 ± 46                               | 3M    |
| jaxborg IPPO | -1,998 ± 118                              | 3M    |

When *both* policies are eval'd on the same env (CybORG) for 100 paired episodes per seed, the cross-policy gap is **+5.4 ± 58 pts** (n=300), TOST-equivalent at Δ=±200 — the two trained policies are statistically interchangeable.

#### Action Distribution

Both engines produce essentially the same learned defensive strategy. Pooled across 3 seeds × 5 blue agents × 100 eps each on CybORG env, decisions only (busy ticks filtered):

| Action       | jaxborg | CybORG | Delta |
| ------------ | ------: | -----: | ----: |
| Analyse      |   21.2% |  19.9% | +1.3% |
| Remove       |   20.0% |  18.9% | +1.2% |
| Decoy        |   22.3% |  23.8% | -1.5% |
| AllowTraffic |   15.1% |  13.9% | +1.2% |
| BlockTraffic |   10.1% |  11.8% | -1.7% |
| Restore      |    7.0% |   7.5% | -0.5% |
| Sleep        |    2.2% |   1.9% | +0.2% |
| Monitor      |    2.1% |   2.4% | -0.3% |

All buckets within ~1.7%. Pooled L1 distribution distance = 0.079 (max 2.0). Action entropy is also matched: jaxborg 1.852 nats / CybORG 1.862 nats (Hill diversity 6.37 / 6.44 effective action types out of 8).

## Setup

```bash
uv sync --extra cpu          # CPU-only (tests, eval, macOS dev, CI)
uv sync --extra cuda         # GPU support (training on NVIDIA Linux hosts)
# Plain `uv sync` (no extra) is also valid but installs jaxlib only via
# transitive resolution — pick one explicitly to avoid the silent
# "GPU was allocated but JAX fell back to CPU" trap.
```

## Usage

```bash
# Tests (fast suite ~7 min; slow L3 fuzz excluded by default)
uv run pytest            # default: -n auto -m 'not slow'
uv run pytest -m slow    # L3 full-episode differential fuzz + CybORG-trained policy rollouts
uv run pytest -m ""      # everything

# Train jaxborg IPPO (recipe-driven; see `recipes/<name>.yaml`)
./scripts/train/run.sh jax default 42

# Train CybORG PPO baseline (CPU-only, CleanRL — no slurm)
./scripts/train/run.sh cleanrl default 42

# Opt-in simultaneous Blue/Red self-play (fresh policy for each team).
# The recipe automatically evaluates final trained Blue vs FSM + CIA-C/I/A.
./scripts/train/run.sh jax cotraining 42
./scripts/train/run.sh cleanrl cotraining 42

# Multi-seed sweep (3 seeds, parallel for cleanrl, sequential under srun for jax)
./scripts/train/run_seeds.sh cleanrl default 3 0
./scripts/train/run_seeds.sh jax default 3 0

# Legacy contract evaluation: learned Blue vs scripted Red in CybORG
uv run python scripts/eval/eval_recipe.py \
    --model jaxborg-exp/ippo_cyborg/<tag>/model_<tag>.pt \
    --episodes-per-seed 1 --seeds 42-141

uv run python scripts/eval/eval_recipe.py \
    --model jaxborg-exp/ippo_jax/<tag>/model_<tag>.safetensors \
    --episodes-per-seed 1 --seeds 42-141

# Manual/re-run sweep: trained Blue vs FSM and all three CIA Red agents.
# Works for either final .safetensors or .pt bundle.
JAX_PLATFORMS=cpu uv run python scripts/eval/eval_scripted_reds.py \
    --model jaxborg-exp/ippo_jax/<tag>/model_<tag>.safetensors \
    --seeds 1000-1099 --episodes-per-seed 1 --workers 8

# Plot all standardized JSONL evaluation results (prints a table too).
JAXBORG_EXP_DIR=./jaxborg-exp uv run python scripts/eval/plot_results.py

# View training and evaluation metrics tracked in MLflow.
JAXBORG_EXP_DIR=./jaxborg-exp ./scripts/train/view_mlflow.sh

# Learned Blue from one run vs learned Red from another, in JAX CC4
uv run python scripts/eval/eval_matchup.py \
    --recipe my_matchup --policy-backend jax \
    --blue-experiment blue_seed42 --red-experiment red_seed42 \
    --episodes-per-seed 10 --seeds 42-51

# Dev parity transfer check: independent rollouts on both engines + TOST
JAX_PLATFORMS=cpu uv run python scripts/dev/transfer.py \
    --checkpoint jaxborg-exp/ippo_jax/<tag>/model_<tag>.safetensors \
    --episodes 100
```

Training output goes to `$JAXBORG_EXP_DIR` (the launcher defaults to `./jaxborg-exp/`).

## Recipes

A recipe is a single YAML under `recipes/` that drives both backends. The `algorithm:` key picks the trainer (`scripts/train/algorithms/<algorithm>_<backend>.py`); `arch.name` picks the policy from the `src/jaxborg/policies/` registry; `core` / `train` / `jax` / `cleanrl` sections project to backend-specific PPO config.

```yaml
# recipes/default.yaml — Matched-Training v2 baseline
algorithm: ippo
mlflow:
  checkpoint_eval:
    every_steps: 0              # 0 disables checkpoint evaluation
    episodes_per_seed: 10
    seed: null                  # defaults to the training seed + 100000
    deterministic: false
core:    {lr: 3.0e-4, gamma: 0.99, ent_coef: 0.01, ...}
arch:    {name: shared, hidden_dim: 256, hidden_layers: 2}
train:   {episode_length: 500, total_timesteps: 3000000}
jax:     {num_envs: 48, num_minibatches: 16, update_epochs: 4}
cleanrl: {num_envs: 48, rollout_length: 500, num_rollouts_per_update: 1}
```

MLflow checkpoint evaluation works in both trainers. At the first completed
update that crosses each `every_steps` boundary, and again at the final update,
the trainer freezes the exact portable checkpoint and recipe sidecar as MLflow
artifacts, evaluates that checkpoint for `episodes_per_seed` episodes generated
from its configured evaluation seed, and logs
`eval.checkpoint.blue.mean_reward` and/or
`eval.checkpoint.red.mean_reward` at the checkpoint's environment step. MLflow
plots those scalar histories automatically. Only trained teams are evaluated:
a joint run records both curves, while Blue-only or Red-only training records
the corresponding single curve. For co-training this periodic curve is the
learned Blue-vs-learned Red matchup; it is separate from the final
Blue-vs-scripted-Red sweep.

`recipes/cotraining.yaml` uses `eval.after_training` to run two final checks in
order: learned Blue versus its co-trained PPO Red in the JAX joint environment,
then learned Blue versus `fsm`, `cia_c`, `cia_i`, and `cia_a` in CybORG. Both
jobs use the exact final bundle. Increase their seed ranges to `1000-1099` for
a 100-episode study. A failed required job makes the overall command fail after
leaving the model and evaluation manifest safely on disk.

For more than one final-checkpoint evaluation, use the ordered
`eval.after_training` list. Every item launches a fresh Python process after
the durable model and sidecar have been saved; the exact model is supplied as
`--model` automatically. For example, this runs a stochastic comparison and
then an argmax comparison:

```yaml
eval:
  after_training:
    - name: stochastic-reds
      script: scripts/eval/eval_scripted_reds.py
      args: [--reds, fsm, cia_c, --seeds, 1000-1099, --workers, 8]
    - name: deterministic-reds
      script: scripts/eval/eval_scripted_reds.py
      args: [--reds, fsm, cia_c, --seeds, 1000-1099, --deterministic, --workers, 8]
```

Entries execute in list order and are required by default. Set
`required: false` to record a failure and continue to the next evaluation, or
`model_arg: --checkpoint` for a script with a different checkpoint flag.
Arguments may use `{model}`, `{recipe}`, `{backend}`, `{exp_dir}`, `{eval_dir}`, and
`{name}` placeholders. The legacy `eval.scripted_red.after_training` form is
still supported, but cannot be combined with the new list.

Re-run the list for an existing checkpoint (using its sidecar configuration):

```bash
uv run python scripts/eval/run_after_training.py \
  --model jaxborg-exp/ippo_jax/<tag>/model_<tag>.safetensors
```

Built-in evaluators write JSONL under `$JAXBORG_EXP_DIR/eval/`. Ordered runs
also write a command/status manifest under `eval/manifests/`, and named runs
attach distinct `eval.after_training.<name>.*` metrics to MLflow. Generate a
PNG comparison with `scripts/eval/plot_results.py`; plots go to `eval/plots/`.
Start `scripts/train/view_mlflow.sh` and open `http://127.0.0.1:5000` for
interactive metric curves. See [`docs/evaluation.md`](docs/evaluation.md).

For evaluators that accept a seed list, the rollout-count option is uniformly
named `--episodes-per-seed`. `eval_recipe.py` and `eval_matchup.py` still accept
the older `--episodes` spelling as a compatibility alias.

`train.teams` selects `blue`, `red`, or `both` and defaults to `blue`, so
existing recipes retain Blue-versus-scripted-Red behavior. In `both` mode the
teams have independent parameter-sharing policies, act from the same pre-step
state, and update independently. `train.team_overrides.<team>` deep-merges
team-specific `core` and `arch` values. A one-team run can name a frozen
learned opponent with `train.opponents`; Red-only training requires a Blue
opponent, while Blue-only training still defaults to scripted Red.

Model references accept either `path` or `experiment`. A relative path is
resolved from the recipe directory and takes precedence; an experiment is
resolved below `$JAXBORG_EXP_DIR/<algorithm>_<backend>/<experiment>/`, using
the enclosing recipe's `algorithm`. See
[`recipes/cotraining.yaml`](recipes/cotraining.yaml) and
[`docs/training.md`](docs/training.md) for complete examples.
For reproducible CAGE/JAX topology pools, disjoint train/test seed ranges, and
held-out evaluation, see [`docs/topologies.md`](docs/topologies.md).

Each training run writes the resolved recipe alongside the model as `recipe_<tag>.yaml`. This sidecar is required for `eval_recipe.py` and the dev transfer parity check — pre-sidecar checkpoints no longer load.

Add a new arch: drop a module under `src/jaxborg/policies/` exporting `JAX_FACTORY` + `TORCH_FACTORY`, register it in `policies/__init__.py`, then reference it as `arch.name: <new>` in any recipe.

## Network Architecture

Both engines use the same parameter-sharing architecture. A Blue policy is
shared by all five Blue agents; a distinct Red policy is shared by all six Red
agents. `both` does not share weights across teams.

| team | agents | observation | policy actions | sharing |
| --- | ---: | ---: | ---: | --- |
| Blue | 5 | 210 | 242 | one policy across Blue agents |
| Red | 6 | 706 | 1,106 | one separate policy across Red agents |

Agents 0-3 each observe one subnet; agent 4 observes three (the full 210-dim vector). Agents 0-3 are zero-padded to 210 obs / 242 actions, with action masking to prevent invalid actions.

Exports are versioned `.safetensors` (JAX/Flax) or `.pt` (CybORG/Torch)
bundles. A bundle contains every active learned policy, including a frozen
learned opponent, plus team, dimensions, architecture, trainable status, and
source provenance. Legacy unversioned files remain Blue-only.

## Resilience Metric and Alignment

The [Resilience metric](https://github.com/xcadet/CyberResilience) is implemented using a separate topology that labels an operational host in each subnet as CIA-tied assets. An example configuration can be found in the `recipe/resilience.yaml` configuration. Resilience metric training and evaluation can be done with the following:

```bash

# Training:
../scripts/traing/run.sh jax resilience 42

# Evaluation:
# 1. Record trajectories
uv run python scripts/eval/cc4_trajectory_eval.py \
    --model jaxborg-exp/ippo_cyborg/resilience_seed42/model_resilience_seed42.pt \
    --episodes 100 \
    --seed 42 \
    --output-dir trajs/resilience_seed42 \
    --recipe recipes/resilience.yaml

# 2. Score them (CIA + resilience)
uv run python scripts/eval/score_trajectories.py trajs/resilience_seed42 \
    --recipe recipes/resilience.yaml

```
