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

## Approach

- Implements JaxMARL's `MultiAgentEnv` interface
- Dynamic topology handled via pad-to-max with active-host masking (one JIT trace for all topologies)
- Differential testing against CybORG: run both environments in lockstep, compare state after every step
- Behavior catalog drives parameterized test generation across all action/target/state combinations
- Final validation: train PPO on both environments, compare learning curves

## Setup

```bash
uv sync
```

## Usage

```bash
uv run pytest tests/ -v
uv run python scripts/baselines/eval_sleep.py --max-eps 100 --seed 42
uv run python scripts/baselines/eval_random.py --max-eps 100 --seed 42
uv run python scripts/baselines/train_ppo.py total_timesteps=250000
```

Training output (Hydra logs, tensorboard events, saved models) goes to `../jaxborg-exp/`.

## Guidelines

- **Pin external dependencies to git commits** for reproducibility. Never depend on local filesystem paths.
- The CybORG and JaxMARL commit hashes are pinned in `pyproject.toml`.
