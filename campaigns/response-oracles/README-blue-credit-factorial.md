# Bounded actor/critic credit experiment

Run only after the new seed-22001 matched lambda replication supports the
benefit. This opt-in experiment does not change default PPO settings. Its four
cells use actor GAE lambda / critic-target lambda:

| Arm | Actor | Critic | Training |
| --- | --- | --- | --- |
| a095-c095 | .95 | .95 | Reuse completed replication control |
| a100-c100 | 1 | 1 | Reuse completed replication lambda-one |
| a095-c100 | .95 | 1 | 960,000 new steps |
| a100-c095 | 1 | .95 | 960,000 new steps |

All four restore the same calibrated warm state from fresh seed 22001 at
historical 1.92M: parameters, fresh Adam, calibrated normalizers, environment,
and RNG. Frozen original Red is source 9.6M seed 42. Gamma .99, fixed topology
seed 0, stochastic 500-step episodes and other settings are unchanged.
Historical optimizer state was unavailable; this is not exact continuation.

Mixed cells compose advantages and targets separately from identical rewards,
old critic values, terminal flags and bootstrap values. Only actor advantages
are mask-normalized. Critic targets use their own unnormalized lambda-return;
lambda-one still bootstraps at nonterminal rollout boundaries. On captured
updates, all PPO epochs match the original execution-revision updater with its
GAE call replaced by independently composed original calls. Diagonal cases
remain exact legacy PPO. Focused tests verify separation, busy-row targets,
bootstrap handling, minibatch capture and all parameters/Adam/RNG/metrics.

Register confirmation **11200000–11200127**, bootstrap **11400001**, after
checking the local seed ledger and before mixed training. Endpoints are fixed
in advance: do not select on this cohort. Re-evaluate all four final policies,
starting policy and original Blue on this same cohort with unrestricted and
Block-excluded sampling. The earlier replication cohort stays distinct.

From a clean pinned checkout containing this protocol:

```bash
export JAXBORG_DIAGNOSTIC_PYTHON=/path/to/compatible/gpu/python
export JAXBORG_EXP_DIR=/path/to/existing/shared/mlflow-root
study=/path/to/blue-credit-replication
reference=/path/to/checkout/pinned-at-0d577faab9ac3e0a06376e99e9592aadf128c130

actor095_job=$(bash scripts/experiments/submit_blue_learning_mechanism.sh \
  campaigns/response-oracles/blue-credit-factorial-a095-c100.yaml \
  "$study/input-data" "$study/a095-c100" "$reference" \
  --controlled-from "$study/control")
actor095_job=${actor095_job%%;*}
actor100_job=$(bash scripts/experiments/submit_blue_learning_mechanism.sh \
  campaigns/response-oracles/blue-credit-factorial-a100-c095.yaml \
  "$study/input-data" "$study/a100-c095" "$reference" \
  --controlled-from "$study/control")
actor100_job=${actor100_job%%;*}
JAXBORG_SLURM_AFTEROK="$actor095_job:$actor100_job" \
  bash scripts/experiments/submit_blue_learning_mechanism.sh --factorial-eval \
  campaigns/response-oracles/blue-credit-factorial-evaluation.yaml \
  "$study" "$study/input-data" "$study/factorial-evaluation"
```

Two mixed training jobs may run concurrently with one Slurm community GPU each;
the confirmation job depends on both completing. Use the existing shared MLflow
store. Preserve failed attempts and use fresh output directories on a retry.
Never substitute CPU training. Archived scores must match before training.

CPU statistical reproduction:

```bash
PYTHONPATH=src:. MPLCONFIGDIR=/tmp/blue-credit-mpl \
  python -m scripts.experiments.analyze_blue_credit_factorial \
  --study-root "$study" --evaluation "$study/factorial-evaluation" \
  --output "$study/factorial-analysis"
```

Reports contain paired score/component/action means, factorial contrasts,
phase/traffic learning signals and plots. The four policies come from ONE
warm-start training seed. Episode bootstrap intervals quantify conditional
policy performance, not variation over independent training seeds. An actor
main effect does not prove critic values or shared representation are irrelevant.
No permanent change is justified until this bounded evidence is assessed.
