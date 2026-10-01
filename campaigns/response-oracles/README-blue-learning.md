# Bounded Blue traffic-learning diagnostics

The protocol investigates why a fresh Blue policy increases harmful traffic control. It changes no default training or environment behavior. The warm-start protocol is in `blue-learning-mechanism.yaml`; the captured-state credit replay is in `blue-learning-signal-forks.yaml`.

The historical checkpoint contains weights without Adam or reward-normalizer state. The first protocol resets Adam, calibrates normalization for four fixed-policy rollouts, and runs twenty training updates from the 1.92M checkpoint. It saves intermediate models, full numeric states, minibatches, normalization profiles, legal-action forks, and separate validation and confirmation episodes. A full PPO update is compared with the original execution revision. A resumed diagnostic must reproduce the captured update exactly; this does not make it an exact continuation of historical training.

Use a clean checkout and a compatible GPU environment. Submit through Slurm community; the launch script preserves its assigned devices and rejects a CPU backend. All runs use the existing shared MLflow root.

```bash
export JAXBORG_DIAGNOSTIC_PYTHON=/path/to/compatible/venv/bin/python
export JAXBORG_EXP_DIR=/path/to/shared/tracking-root
bash scripts/experiments/submit_blue_learning_mechanism.sh \
  campaigns/response-oracles/blue-learning-mechanism.yaml \
  /path/to/equilibrium-stage-c /path/to/new-output /path/to/original-execution-checkout
```

The original-execution checkout must contain `scripts/train/algorithms/ippo_jax_joint.py` at revision `0d577faab9ac3e0a06376e99e9592aadf128c130`. The data directory retains its original relative layout, including `eval/blue_diagnosis/checkpoints.csv`. Inspect configs before launching; do not reuse episode cohorts for a new confirmation.

The second protocol performs no training. It compares the naturally sampled traffic action, its legal opposite, Sleep, and a freshly sampled policy alternative. Other simultaneous actions remain fixed. It retains 32 common futures per saved state in each of two cohorts. It measures raw and discounted returns, normalized Monte Carlo credit, and GAE at lambda 0.95 and 1. Its normalization uses the frozen variance profile of the original 96-environment rollout; it does not replay counterfactual normalizer updates.

```bash
bash scripts/experiments/submit_blue_learning_mechanism.sh --signal-forks \
  campaigns/response-oracles/blue-learning-signal-forks.yaml \
  /path/to/finished-warm-output /path/to/new-credit-output
```

Before credit comparisons, 64 saved conditional simulations must reproduce their raw component returns exactly. Lambda-one GAE must telescope to the normalized Monte Carlo advantage. These controls complement the archived-evaluation and original-updater checks.

Recompute summaries and plots without GPU use:

```bash
python -m scripts.experiments.analyze_blue_learning_mechanism /path/to/finished-warm-output
JAX_PLATFORMS=cpu PYTHONPATH=src:. python -m scripts.experiments.probe_blue_learning_captures \
  /path/to/finished-warm-output --probe all
```

The summary script needs NumPy and Matplotlib. Capture probes additionally need the compatible repository/JAX environment and the trusted JAX tree structures accompanying the numeric NPZ files. Remap the topology path in a copied resolved recipe when transferring to another computer; preserve the archived recipe and record the remapping.

Action labels in raw exports are selection strata: `harmful_new_block` means a newly set own-direction bit on a mission-permitted pair; `useful_allow` means clearing that bit. Neither label establishes measured benefit. Reverse-direction blocking, other simultaneous actions, and future policy behavior can change the outcome. The fork intervals are conditional on twelve saved states from four rollout environments and must not be treated as independent topology or training-seed replications.

Actor, critic, and entropy comparisons use the same minibatch and optimizer state, with a zero-gradient Adam momentum control. Action-stratum attribution preserves the original global actor denominator and globally normalized advantages. Observation perturbations establish input aliasing; they do not establish a learning defect by themselves. Keep implementation defects, objective differences, credit-estimation errors, and representation limits separate when interpreting the results.

A single controlled training ablation is provided in `blue-learning-credit-lambda1.yaml`. It changes only core GAE lambda to 1, runs twenty updates, and loads the first protocol's calibrated policy, Adam, normalizers, environment and RNG state. The guard rejects unmatched settings, incomplete controls and a nonfresh optimizer. Reused validation baselines must share the original validation seeds. A new confirmation cohort compares the initial policy and both twenty-update endpoints, including the Block-exclusion control.

```bash
bash scripts/experiments/submit_blue_learning_mechanism.sh \
  campaigns/response-oracles/blue-learning-credit-lambda1.yaml \
  /path/to/equilibrium-stage-c /path/to/new-lambda1-output /path/to/original-execution-checkout \
  --controlled-from /path/to/finished-lambda095-warm-output
```

This ablation is an experiment-specific config, with no change to training defaults. Correct conditional credit rankings alone do not establish improved policy training or performance across other sources, seeds or topologies.
