# Joint payoff and blocking study

This bounded example composes a local zero-sum payoff offset with the existing
Blue blocking rule through the shared trainer, evaluator and record store.
[`recipe.yaml`](recipe.yaml) uses the existing recipe schema. No separate
configuration system is required.

| Condition | Blue offset per step | Blocking |
| --- | ---: | --- |
| baseline | 0 | cc4 |
| offset | 1 | cc4 |
| blocking | 0 | mission_safe |
| combined | 1 | mission_safe |

## Run

```sh
python experiments/reward-offset/study.py run \
  --output /evidence/payoff-blocking --expected-revision FULL_COMMIT_SHA
```

The destination must be new and outside the checkout. The command prepares
separate training/evaluation topology snapshots (seeds 0 and 1), shared across
conditions, then executes each condition in a fresh process and retains training,
owned checkpoints, independent evaluations, collation and report. Children verify
the supplied recipe and topology hashes before each execution and after consuming
the inputs, then return exact attempt/evaluation IDs for selection.
Capture retains source, settings and consumed inputs. A child failure or
interruption stops aggregation while preserving completed and interrupted
attempts. Each condition uses the same initialization seed and
evaluation cohort. Source is pinned; `--allow-dirty` explicitly retains a
development patch.

The default budget is two training transitions and two evaluation episodes per
condition. This demonstrates construction and recording, not learned effects or
independent training variability. The protocol declares one unlaunched trial so
missing work remains visible. Retries require new attempts and explicit selection.

## Construction contract

[`rules.py`](rules.py) adds the offset to Blue rewards and `reward_ria`, and
subtracts it from Red rewards. Both training and evaluation receive the same
bound `build_env` callable via `env_factory`. It receives the variant plus
`topology_mode`, `training_mode` and `topology_path`; construction happens outside
compiled rollout. `bind_condition(name)` supplies the recipe, builder and a check
of the final resolved settings/source before execution.

The hook supports canonical joint CC4 state, observations, actions, masks,
pytrees, shapes and dtypes with shared zero-sum team rewards. Reset, step and
rules must be pure, JAX-traceable functions of explicit state and PRNG keys.
Override `step_env` so scalar, batched and evaluation paths share the behavior.
Structural checks cannot prove scientific equivalence. This example rejects
resilience-role variants; other layouts, backends and non-zero-sum games require
separate support. See [joint execution records](../../docs/joint-execution-records.md).

## Rebuild a saved comparison

```sh
python experiments/reward-offset/study.py analyze \
  --collation /evidence/payoff-blocking/collation.json \
  --evidence-root /evidence/payoff-blocking/evidence --output /reports/payoff-blocking
```

The study checks contracts, budgets, seed roles, topology, source, installed
dependencies, runtime and exact checkpoint ancestry. Only declared treatments and resulting weights may differ.
Payoff and original simulator score are separate columns; the latter subtracts
the offset times actual completed steps under that condition's blocking rule.

The shared plot consumer retains a figure, joined CSV/JSON, selection,
missing/excluded cells, correction links, analyzer source and input/output hashes.
Error bars show sample episode SD conditional on fixed policies, not confidence
intervals. With one episode, SD is unavailable and no error bar is drawn.
Conditions are not pooled.

For a revised selection, call common `collate` with explicit attempt/evaluation
IDs and `analysis={"source": analysis_identity(sources()),
"configuration": report_configuration()}` using this module and
`scripts.eval.plot_results.analysis_identity`. A new selection gets a new identity.
Other episode studies can use `plot_results.py --collation --evidence-root --output`
with that consumer's own analysis identity and `REPORT_CONFIGURATION`.

## Exercise failure and recovery

```sh
python experiments/reward-offset/recovery_example.py \
  --source-store /evidence/payoff-blocking/evidence \
  --source-evaluation BASELINE_EVALUATION_ID \
  --output /evidence/recovery-example --expected-revision FULL_COMMIT_SHA
```

This separate fixture deliberately supplies invalid checkpoint bytes, retains
the failure, then retries with the selected baseline policy and records its
trajectory. Its correction link describes that deliberate input replacement;
it does not assert a historical evaluator defect. The new store imports the
verified baseline inputs without assigning a training parent. It retains the
snapshot before retry, successful retry selection and a generic report;
the original study evidence and report remain unchanged.
