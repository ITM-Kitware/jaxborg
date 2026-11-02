# Red update schedules

## Question

How does changing Red's update schedule during co-training affect Blue's
performance against a fixed scripted Red suite? The comparison is intended to
identify which training configuration, if any, merits a larger study.

## Method

### Training recipes

| Condition | Recipe | Scientific difference |
| --- | --- | --- |
| Baseline | [Baseline co-training](../../recipes/cotraining/separate/cotraining_separate.yaml) | Blue and Red update every rollout. |
| Quarter-rate Red | [Quarter-rate co-training](../../recipes/cotraining/separate/cotraining_red_quarter_rate_separate.yaml) | Blue updates every rollout; Red updates every fourth rollout. Original optimizer settings are retained. |
| Adaptive | [Adaptive co-training](../../recipes/cotraining/separate/cotraining_adaptive_updates_separate.yaml) | Adaptive update gating, Blue LR `1e-4`, Red LR `3e-4`, and learning-rate annealing disabled. |

Adaptive changes both scheduling and optimization, so this comparison cannot
isolate a gate effect. Its substantive settings retain the 200-Blue-update
warmup, 20-rollout return window, `-2500` threshold, and maximum 100 closed-gate
rollouts. The existing [scheduler](../../src/jaxborg/adaptive_updates.py) defines
the gate boundaries.

Each condition retains its preset architecture, observations, training rules,
reward normalization, and one generated training topology. Preparation replaces
the training budget and environment count with the declared study values.
Evaluate each trained policy at its final completed checkpoint.

### Common evaluation procedure

- **Opponents:** final JAX scripted `fsm`, `cia_c`, `cia_i`, and `cia_a`.
  No cross-play, cross-seed-play, play-priors, or intermediate-checkpoint
  evaluation contributes to this comparison.
- **Policy and score:** stochastic policies with `cia_resilience` and fixed
  topology role assignment.
- **Test bank:** exhaustive held-out topology coverage, starting at topology
  seed 100 with three operational-zone servers; training starts at 0.
- **Episodes:** declared seeds, count, and horizon. Seeds expand through the
  [existing helper](../../src/jaxborg/evaluation/episode_seeds.py).

### How we compare results

- Episodes receive equal weight within a topology, topologies equal weight within
  an opponent, and independently trained policies equal weight within a condition.
  Opponents remain separate; there is no implicit panel average.
- A primary condition/opponent mean requires every declared training seed and
  its complete evaluation cases. Available seed values and ranges remain visible
  when the primary estimate is absent. Missing trials remain expected cells.
- Reward and CIA units follow the evaluator. More episodes do not create more
  independent training replicates. Matching seed labels across conditions do not
  establish paired initialization.
- Report each training-seed value and descriptive seed SD/range. With two training
  seeds there are two policies, regardless of episode count. Rankings are
  exploratory; the analyzer produces no confidence interval or superiority claim.
- Costs accompany performance. Update counts are completed team PPO rounds, not
  minibatch steps or instantaneous learning rates. Interpret losses with
  `team.*.updated`. Fewer Red updates alone do not establish a speedup.
  Evaluation wall seconds exclude process startup and artifact delivery.
  Unknown allocation or incomparable timing remains a limitation.

Changed evaluation inputs require a new protocol, including a same-size
replacement held-out bank. The analyzer checks selected inputs against the
study's declared recipes, evaluation cases, and final-checkpoint rule.

## Status and findings

No substantive scientific comparison has been completed. Its training budget
and seed set remain to be chosen. Smoke configurations exercise the pipeline
and cannot establish learned performance or a preferred schedule.

## Evidence

- **Methods and analysis:** the recipes above and [the study analyzer](analyze.py).
- **Results:** no public result bundle is available yet. A completed comparison's
  report and comparison data live in its external artifacts. Its results link
  belongs here once available to readers.

## Commands

See [Reproduce the comparison](reproduce.md) for preparation, execution,
evidence selection, and rebuilding or revising saved results.
