# jaxborg

JAX port of [CybORG CAGE Challenge 4 (CC4)](https://github.com/cage-challenge/cage-challenge-4) using [JaxMARL](https://github.com/FLAIROx/JaxMARL) for GPU-accelerated parallel RL training.

CybORG CC4 is a multi-agent cybersecurity simulation (9 subnets, ~80 hosts, 5 blue agents, 6 red agents, 3 mission phases). This project re-implements CC4's environment logic as JIT-compilable JAX arrays for massively parallel simulation on GPU. Parity is verified by:

- **Differential testing**: lockstep state comparison after every step
- **TOST equivalence testing**: statistical comparison of independent rollout rewards across both engines

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

Matched-hyperparameter training (NUM_ENVS=48, 3 seeds, same shared-trunk actor-critic, identical PPO hparams). Reward is the mean training-time episode reward at 3M steps; each policy is on the env it trained against (the ~145-pt gap between the two columns reflects independent green-RNG host-selection between engines, not a parity bug; see [`docs/parity.md`](docs/parity.md)).

| Run          |          Reward (mean ± σ across 3 seeds) | Steps |
| ------------ | ----------------------------------------: | ----- |
| CybORG PPO   | -1,854 ± 46                               | 3M    |
| jaxborg IPPO | -1,998 ± 118                              | 3M    |

When *both* policies are eval'd on the same env (CybORG) for 100 paired episodes per seed, the cross-policy gap is **+5.4 ± 58 pts** (n=300), TOST-equivalent at Δ=±200; the two trained policies are statistically interchangeable.

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

## Start here

- [Project setup](docs/setup.md): Python environment, experiment storage, and MLflow.
- [Training guide](docs/training.md): recipes, launchers, teams, and checkpoint evaluation.
- [Evaluation guide](docs/evaluation.md): choose an evaluator, run post-training jobs, and inspect results.
- [Study guide](studies/README.md): compose a scientific comparison and retain its evidence.
  - [Individual studies and examples](studies/README.md#studies-and-examples) describe their questions, methods, status, and results.
- Reference topics:
  - [Environment parity](docs/parity.md).
  - [Topology generation and held-out banks](docs/topologies.md).
  - [Recorded joint executions](docs/joint-execution-records.md).
  - [Checkpoint evaluation capture](docs/evaluation-capture.md).
  - [Evidence retrieval](docs/evidence-retrieval.md).

## Development

- Run focused tests for the changed behavior, following [repository instructions](AGENTS.md).
- Add a policy architecture in `src/jaxborg/policies/`, export `JAX_FACTORY` and
  `TORCH_FACTORY`, register it in `policies/__init__.py`, and select it with
  `arch.name` in a recipe.

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

The [Resilience metric](https://github.com/xcadet/CyberResilience) uses AUTH,
DB, and WEB roles on operational-zone servers. For paired JAX evaluation,
enable the metric explicitly and generate a resilience-safe held-out bank:

```yaml
eval:
  variant: cia_resilience
  topology_sampling: exhaustive
  cia:
    enabled: true
    metric: resilience
    role_assignment: fixed_per_topology
  topology_generation:
    generator: jax
    seed_start: 100
    count: 100
    op_zone_servers: 3
    cache_dir: .bank_cache/topologies/my_eval_ops3
```

The snapshot contents determine a stable role-map ID, so copied or reordered
topologies keep the same roles across checkpoint, play-priors, learned-matchup,
and JAX scripted-Red evaluation. MLflow records signed C/I/A mean and sample
standard deviation alongside reward; zero is healthy and more-negative values
represent larger drops. The legacy `eval.cia_metric` key only selects the
metric and does not enable collection.

Offline CybORG trajectory scoring remains available separately:

```bash

# Training:
./scripts/train/run.sh jax resilience 42 --development-no-recording

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
