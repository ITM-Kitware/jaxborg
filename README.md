# jaxborg

JAX port of [CybORG CAGE Challenge 4 (CC4)](https://github.com/cage-challenge/cage-challenge-4) using [JaxMARL](https://github.com/FLAIROx/JaxMARL) for GPU-accelerated parallel RL training.

CybORG CC4 is a multi-agent cybersecurity simulation (9 subnets, ~80 hosts, 5 blue agents, 6 red agents, 3 mission phases). This project re-implements CC4's environment logic as JIT-compilable JAX arrays for massively parallel simulation on GPU. Parity is verified by:
- **Differential testing** — lockstep state comparison after every step
- **TOST equivalence testing** — statistical comparison of independent rollout rewards across both engines

## Results

### Speed

| Engine | Parallelism | Steps/sec | 50M wall time |
|--------|-------------|-----------|---------------|
| CybORG | 48 CPU processes | 332 | 41.8 h |
| jaxborg | 1,024 vectorized envs on GPU | 2,645 | 5.4 h |

**8x throughput**, **~8x wall-time** on a single NVIDIA RTX A6000. jaxborg's entire training loop (rollout + GAE + PPO update) compiles to one XLA program; first-run compile takes ~7 min (cached thereafter).

### Training

Best policies trained for 50M steps, evaluated independently on both engines (10 episodes, stochastic):

| Agent | Eval Engine | Reward | Notes |
|-------|-------------|-------:|-------|
| jaxborg IPPO (50M, 1024 envs) | jaxborg | **-225** | Best policy |
| jaxborg IPPO (50M, 1024 envs) | CybORG | **-344** | Cross-engine transfer |
| CybORG PPO (20M, 48 envs) | CybORG | -1,658 | Matched hyperparams |

jaxborg-trained policies outperform CybORG-trained baselines by **~5x** on reward.

### Parity (TOST Equivalence)

We verify that jaxborg faithfully reproduces CybORG's dynamics using the TOST (two one-sided t-test) equivalence procedure from [Karten et al. (2026)](https://arxiv.org/abs/2603.12145): run the same trained policy independently on both engines and test whether mean reward difference falls within ±Δ with 95% confidence.

100 episodes, stochastic, α=0.05, Δ=±200 (~35% of reward magnitude):

| Policy | jaxborg | CybORG | Gap | 95% CI | Verdict |
|--------|--------:|-------:|----:|--------|---------|
| IPPO 50M (seed 0) | -510 | -595 | +85 | [+6, +165] | **EQUIVALENT** |
| IPPO 50M (seed 1) | -549 | -576 | +27 | [-16, +69] | **EQUIVALENT** |

Additional confirmation: hobbling jaxborg's training script to match CybORG's pipeline (shared trunk, global grad clip, no busy masking) moves jaxborg reward from -225 to -1,200 — the same plateau as CybORG's -1,600. The training pipeline, not the environment, explains the performance gap.

## Approach

- Implements JaxMARL's `MultiAgentEnv` interface
- Dynamic topology via pad-to-max with active-host masking (one JIT trace for all topologies)
- Differential testing against CybORG: lockstep state comparison after every step
- Behavior catalog drives parameterized test generation across all action/target/state combinations
- Cross-backend policy transfer validates sim-to-sim equivalence via TOST

Verification methodology inspired by [Karten et al. (2026)](https://arxiv.org/abs/2603.12145).

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

| | jaxborg IPPO | CybORG CleanRL PPO |
|-|--------------|---------------------|
| **Networks** | 1 shared across all 5 agents | 2 (agents 0–3 share one, agent 4 gets its own) |
| **Obs dim** | 210 for all agents | 92 (agents 0–3), 210 (agent 4) |
| **Action dim** | 260 for all agents | 82 (agents 0–3), 242 (agent 4) |
| **Architecture** | Separate actor + critic heads | Shared trunk, split at final layer |
| **Params** | ~530K | ~320K |

Each agent observes 1–3 subnets (59 dims per subnet: subnet ID, blocked/comms policy, per-host malicious processes and connections) plus a mission phase scalar and 32-dim inter-agent message vector. jaxborg pads all agents to 3 subnet slots for JIT-compatible shapes; CybORG sizes per agent.

