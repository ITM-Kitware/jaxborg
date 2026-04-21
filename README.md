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

We verify that jaxborg reproduces CybORG's behavior using the TOST (two one-sided t-test) equivalence procedure from [Karten et al. (2026)](https://arxiv.org/abs/2603.12145): run the same trained policy independently on both engines and test whether mean reward difference falls within ±Δ with 95% confidence.

Stochastic, 95% confidence, Δ=±200:

| Policy               | Episodes | jaxborg | CybORG |  Gap | 95% CI          | Verdict        |
| -------------------- | -------: | ------: | -----: | ---: | --------------- | -------------- |
| IPPO 20M (cyborg_bank) |       30 |  -3,696 | -2,976 | -720 | [-1142, -297]   | NOT EQUIVALENT |

#### Training Comparison

| Run          | Reward | Steps |
| ------------ | -----: | ----- |
| CybORG PPO   | -1,641 | 20M   |
| jaxborg IPPO | -1,302 | 20M   |

#### Action Distribution

Both engines produce the same learned defensive strategy (decision steps only, filtering out busy ticks; 30 episodes):

| Action       | jaxborg | CybORG | Delta |
| ------------ | ------: | -----: | ----: |
| Analyse      |   20.9% |  21.1% | -0.2% |
| Remove       |   22.1% |  22.5% | -0.4% |
| Decoy        |   28.4% |  28.7% | -0.3% |
| AllowTraffic |   19.1% |  18.4% | +0.7% |
| BlockTraffic |    3.3% |   3.2% | +0.1% |
| Restore      |    1.9% |   1.9% | +0.0% |
| Sleep        |    2.3% |   2.2% | +0.1% |
| Monitor      |    1.9% |   2.0% | -0.1% |

All action types within ~0.7% across backends.

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

# Train jaxborg IPPO
uv run python scripts/train/ippo_jax.py TOTAL_TIMESTEPS=50000000

# Evaluate a trained policy (independent rollouts on both engines + TOST)
JAX_PLATFORMS=cpu uv run python scripts/eval/transfer.py \
    --checkpoint jaxborg-exp/<run>/checkpoint_final.pkl \
    --episodes 100

# Train CybORG PPO baseline (CPU-only, CleanRL)
JAX_PLATFORMS=cpu uv run python scripts/train/ppo_cleanrl_cyborg.py \
    --total-timesteps 20000000 --num-epochs 4 --num-minibatches 16 \
    --no-anneal-lr
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
