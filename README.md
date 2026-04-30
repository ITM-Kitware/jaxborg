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
uv sync                      # CPU-only (tests, eval)
uv sync --group cuda         # GPU support (training)
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

# Multi-seed sweep (3 seeds, parallel for cleanrl, sequential under srun for jax)
./scripts/train/run_seeds.sh cleanrl default 3 0
./scripts/train/run_seeds.sh jax default 3 0

# Evaluate any policy on CybORG via recipe sidecar
# (.pt → torch state_dict; .pkl → JAX Flax params with action translation)
uv run python scripts/eval/eval_recipe.py \
    --model jaxborg-exp/ippo_cyborg/<tag>/model_<tag>.pt \
    --episodes 10 --seeds 42-141

uv run python scripts/eval/eval_recipe.py \
    --model jaxborg-exp/ippo_jax/<tag>/model_<tag>.pkl \
    --episodes 10 --seeds 42-141

# Independent rollouts on both engines + TOST (JAX checkpoints only)
JAX_PLATFORMS=cpu uv run python scripts/eval/transfer.py \
    --checkpoint jaxborg-exp/ippo_jax/<tag>/model_<tag>.pkl \
    --episodes 100
```

Training output goes to `../jaxborg-exp/`.

## Network Architecture

Both engines use the same network architecture — a single shared-trunk actor-critic for all 5 blue agents:

|                  | jaxborg IPPO (Flax/JAX)      | CybORG CleanRL PPO (PyTorch) |
| ---------------- | ---------------------------- | ---------------------------- |
| **Policy**       | 1 shared across all 5 agents | 1 shared across all 5 agents |
| **Obs dim**      | 210                          | 210                          |
| **Action dim**   | 242                          | 242                          |
| **Architecture** | Shared trunk [256, 256] tanh | Shared trunk [256, 256] tanh |
| **Heads**        | Single-layer actor + critic  | Single-layer actor + critic  |
| **Params**       | ~182K                        | ~182K                        |

Agents 0-3 each observe one subnet; agent 4 observes three (the full 210-dim vector). Agents 0-3 are zero-padded to 210 obs / 242 actions, with action masking to prevent invalid actions.
