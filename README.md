# jaxborg

JAX port of [CybORG CAGE Challenge 4 (CC4)](https://github.com/cage-challenge/cage-challenge-4) using [JaxMARL](https://github.com/FLAIROx/JaxMARL) for GPU-accelerated parallel RL training.

CybORG CC4 is a multi-agent cybersecurity simulation (9 subnets, ~80 hosts, 5 blue agents, 6 red agents, 3 mission phases). This project re-implements CC4's environment logic as JIT-compilable JAX arrays for massively parallel simulation on GPU. Parity is verified by:

- **Differential testing** — lockstep state comparison after every step
- **TOST equivalence testing** — statistical comparison of independent rollout rewards across both engines

## Results

### Speed

| Engine  | Parallelism                  | Steps/sec | 50M wall time |
| ------- | ---------------------------- | --------- | ------------- |
| CybORG  | 48 CPU processes             | 332       | 41.8 h        |
| jaxborg | 1,024 vectorized envs on GPU | 2,645     | 5.4 h         |

**8x throughput**, **~8x wall-time** on a single NVIDIA RTX A6000. jaxborg's entire training loop (rollout + GAE + PPO update) compiles to one XLA program; first-run compile takes ~7 min (cached thereafter).

### Parity

#### TOST Equivalence

We verify that jaxborg reproduces CybORG's behavior using the TOST (two one-sided t-test) equivalence procedure from [Karten et al. (2026)](https://arxiv.org/abs/2603.12145): run the same trained policy independently on both engines and test whether mean reward difference falls within ±Δ with 95% confidence.

100 episodes, stochastic, 95% confidence, Δ=±200:

| Policy            | jaxborg | CybORG | Gap | 95% CI     | Verdict        |
| ----------------- | ------: | -----: | --: | ---------- | -------------- |
| IPPO 50M (seed 0) |    -510 |   -595 | +85 | [+6, +165] | **EQUIVALENT** |
| IPPO 50M (seed 1) |    -549 |   -576 | +27 | [-16, +69] | **EQUIVALENT** |

#### Training Comparison

| Run                       |   Reward | Steps |
| ------------------------- | -------: | ----- |
| CybORG PPO                |   -1,658 | 20M   |
| jaxborg (shared trunk)    |   -1,373 | 25.6M |
| jaxborg (separate trunks) | **-225** | 50M   |

Matching jaxborg's training config to CybORG's (shared trunk, global grad clip, no busy masking) produces a similar reward plateau (-1,373 vs -1,658). The remaining gap is likely due to unmatched architecture and hyperparameter differences (see [Network Architecture](#network-architecture)). With separate actor and critic trunks and per-head grad clipping, jaxborg reaches -225.

#### Action Distribution

Both engines produce the same learned defensive strategy (decision steps only, filtering out busy ticks):

| Action       | jaxborg (shared trunk) | CybORG |
| ------------ | ---------------------: | -----: |
| Analyse      |                  24.9% |  21.0% |
| Remove       |                  28.9% |  23.3% |
| Decoy        |                  27.0% |  21.9% |
| Restore      |                  13.1% |   7.2% |
| BlockTraffic |                   1.5% |   8.5% |
| AllowTraffic |                   1.1% |  13.8% |
| Sleep        |                   2.0% |   2.3% |
| Monitor      |                   1.4% |   2.0% |

Both policies learned balanced Analyse/Remove/Decoy (~21–29%) with minimal Sleep. The main divergence is traffic control: CybORG's dedicated agent-4 network specializes on Block/AllowTraffic (22% combined vs 2.6% in jaxborg), where jaxborg's single shared network cannot.

## Setup

```bash
uv sync                      # CPU-only (tests, eval)
uv sync --group cuda         # GPU support (training)
```

## Usage

```bash
# Tests (use -n auto for parallel execution)
uv run pytest tests/ -v -n auto

# Train jaxborg IPPO
uv run python scripts/train/ippo_jax.py total_timesteps=50000000

# Evaluate a trained policy (independent rollouts on both engines + TOST)
JAX_PLATFORMS=cpu uv run python scripts/eval/transfer.py \
    --checkpoint jaxborg-exp/<run>/checkpoint_final.pkl \
    --episodes 100 --stochastic

# Train CybORG PPO baseline (CPU-only, CleanRL)
JAX_PLATFORMS=cpu uv run python scripts/train/ppo_cleanrl_cyborg.py \
    --num-envs 48 --total-timesteps 20000000 --lr 3e-4 --gamma 0.99 \
    --num-epochs 4 --ent-coef 0.01 --no-anneal-lr
```

Training output goes to `../jaxborg-exp/`.

## Network Architecture

jaxborg and CybORG use different network architectures for IPPO/PPO, which accounts for the training performance gap (not the environment):

|                  | jaxborg IPPO                  | CybORG CleanRL PPO                             |
| ---------------- | ----------------------------- | ---------------------------------------------- |
| **Networks**     | 1 shared across all 5 agents  | 2 (agents 0–3 share one, agent 4 gets its own) |
| **Obs dim**      | 210 for all agents            | 92 (agents 0–3), 210 (agent 4)                 |
| **Action dim**   | 260 for all agents            | 82 (agents 0–3), 242 (agent 4)                 |
| **Architecture** | Separate actor + critic heads | Shared trunk, split at final layer             |
| **Params**       | ~530K                         | ~320K                                          |
