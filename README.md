# jaxborg

JAX port of CybORG CAGE Challenge 4 using [JaxMARL](https://github.com/FLAIROx/JaxMARL) for GPU-accelerated parallel RL training.

CybORG CC4 is a multi-agent cybersecurity simulation (9 subnets, ~80 hosts, 5 blue agents, 3 mission phases). This project re-implements CC4's environment logic as JIT-compilable JAX arrays for massively parallel simulation on GPU, validated step-by-step against the original CybORG.

## Results

### Simulation Speed

| Engine | Envs | Steps/sec |
|--------|------|-----------|
| CybORG (CPU, 48 envs) | 48 | 332 |
| JaxBorg (GPU, 1024 envs) | 1,024 | 2,312 |

**7x faster** than CybORG on GPU (NVIDIA RTX A6000).

### Training Performance

Best policies trained for 50M steps on JaxBorg, evaluated independently on both JaxBorg and CybORG (10 episodes, stochastic):

| Agent | Eval Engine | Stochastic Reward | Notes |
|-------|-------------|-------------------|-------|
| JaxBorg IPPO (50M, 1024 envs) | JaxBorg | **-225** | Best policy |
| JaxBorg IPPO (50M, 1024 envs) | CybORG | **-344** | Cross-engine transfer |
| CybORG PPO (20M, 48 envs) | CybORG | -1,658 | Matched hyperparams |

JaxBorg-trained policies outperform CybORG-trained baselines by **~5x** on reward, with verified parity between the two engines (TOST equivalence test passed, gap within +/-200).

### Parity (TOST Equivalence)

To verify that JaxBorg faithfully reproduces CybORG's dynamics, we run the same trained policy independently on both engines and compare total episode rewards using the TOST (two one-sided t-test) equivalence procedure from [Karten et al. (2026)](https://arxiv.org/abs/2603.12145). If the mean reward difference falls within ±Δ with 95% confidence, the engines are statistically equivalent. We use Δ=200 (~35% of reward magnitude); the observed gap is ~5%, so the result holds at much tighter margins.

100 episodes, stochastic, TOST α=0.05, Δ=±200:

| Policy | JaxBorg | CybORG | Gap | 95% CI | Verdict |
|--------|---------|--------|-----|--------|---------|
| IPPO 50M (seed 1) | -549.3 | -575.8 | +26.6 | [-15.7, +68.8] | **EQUIVALENT** |

## Approach

- Implements JaxMARL's `MultiAgentEnv` interface
- Dynamic topology handled via pad-to-max with active-host masking (one JIT trace for all topologies)
- Differential testing against CybORG: run both environments in lockstep, compare state after every step
- Behavior catalog drives parameterized test generation across all action/target/state combinations
- Cross-backend policy transfer validates zero sim-to-sim gap via TOST

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

# Train JaxBorg IPPO
uv run python scripts/train_ippo_cc4.py total_timesteps=50000000

# Evaluate a trained policy (independent rollouts on both engines + TOST)
JAX_PLATFORMS=cpu uv run python scripts/eval_transfer.py \
    --checkpoint jaxborg-exp/<run>/checkpoint_final.pkl \
    --episodes 100 --stochastic

# Train CybORG PPO baseline (CPU-only, CleanRL)
JAX_PLATFORMS=cpu uv run python scripts/baselines/train_cleanrl_ppo.py \
    --num-envs 48 --total-timesteps 20000000 --lr 3e-4 --gamma 0.99 \
    --num-epochs 4 --ent-coef 0.01 --no-anneal-lr
```

Training output goes to `../jaxborg-exp/`.

