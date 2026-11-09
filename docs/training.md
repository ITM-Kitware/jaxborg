# Training

- First configure [the environment, storage, and MLflow](setup.md).
- Use [the study guide](../studies/README.md) to define a comparison before running
  it, and [evaluation](evaluation.md) for test procedures and outputs.
- Commands run from the repository root. Choose the host's required allocation
  procedure before launching compute.

## Launch training

- The [recipe launcher](../scripts/train/run.sh) accepts a backend, recipe, seed,
  then trainer arguments:

  ```bash
  ./scripts/train/run.sh jax default 42 --development-no-recording
  ./scripts/train/run.sh cleanrl default 42
  ```

  - `jax` selects the JAX trainer and requests `srun --gres=gpu:1 --mem=64G`.
    Its preflight rejects a missing GPU unless `JAXBORG_ALLOW_CPU=1` is set.
  - `cleanrl` selects the CybORG/Torch trainer and runs directly on CPU.
  - The JAX development opt-out above deliberately omits research records.
    Research training instead supplies `--evidence-root`, `--protocol`, `--trial`,
    and `--expected-revision`; see [recorded joint executions](joint-execution-records.md)
    for supported fresh joint training and Blue-only training against stock FSM
    Red, including their restrictions.
  - Logs go under `$JAXBORG_EXP_DIR/logs`; model bundles and resolved recipe
    sidecars go under `<algorithm>_<backend>/<tag>/` in that output root.
- [The multi-seed launcher](../scripts/train/run_seeds.sh) accepts backend,
  recipe, seed count, and starting seed. It runs consecutive seeds sequentially
  through Slurm for JAX and concurrently on CPU for CleanRL.
  - It does not forward per-trial evidence arguments. For recorded studies,
    invoke each trial separately with its declared identity.
  - Seed count describes executions; the study defines which executions count
    as independent replicates and how failed attempts are selected or retried.

## Recipes and teams

- A recipe is YAML loaded by [the shared recipe loader](../src/jaxborg/recipe.py).
  Use a recipe name, a path under `recipes/`, or an explicit YAML path.
- Main settings:
  - `algorithm` selects `scripts/train/algorithms/<algorithm>_<backend>.py`.
  - `arch.name` selects the registered policy architecture.
  - `core`, `train`, `jax`, and `cleanrl` project settings into the selected trainer.
  - `train.teams` is `blue` (default), `red`, or `both`.
    - In `both`, the teams have separate policies and update independently.
    - `train.team_overrides.<team>` deep-merges team-specific `core` and `arch` settings.
    - `train.opponents` supplies a frozen learned opponent. Red-only training
      requires a Blue opponent; Blue-only defaults to scripted Red.
- Model references accept `path` or `experiment`:
  - A relative path resolves from the recipe directory and takes precedence.
  - An experiment resolves below `$JAXBORG_EXP_DIR/<algorithm>_<backend>/<experiment>/`.
- Examples of settings, rather than study methods:
  - [Default recipe](../recipes/default.yaml).
  - [Co-training recipe](../recipes/cotraining/cotraining.yaml).
  - [Topology generation and train/test banks](topologies.md).
- Each model has an adjacent resolved `recipe_<tag>.yaml`. Preserve the bundle
  and sidecar together; recipe evaluation requires the sidecar.

### Opponent control and learning

- Scripted opponents choose actions and update their memory inside the environment.
- Learned opponents act through a policy network. A frozen network still acts,
  but its parameters do not update during training.
- The joint trainer supports training both teams or one team against a frozen
  policy. The [recorded execution guide](joint-execution-records.md) describes
  which of these paths support retained research records.
- Custom environment builders can be used by joint training and evaluation.
  The [payoff and blocking example](../studies/reward-offset/README.md#construction-contract)
  documents the builder contract and its supported environment settings.

## Training budget

- JAX IPPO runs complete rollout batches. Each batch contains the environment
  count multiplied by the rollout length in transitions.
- When the requested transition budget is not a multiple of that batch size,
  the remaining partial batch is not run. Retained execution records distinguish
  the requested budget from the completed transitions and updates.

## Checkpoint evaluation

- On ITM, use the [checkpoint budget](itm-studies.md#checkpoint-budget) when
  choosing periodic saves and evaluation milestones. The default recovery
  cadence is approximately four hours; metric logging can remain frequent.
- `mlflow.checkpoint_eval` configures evaluation during training:
  - `every_steps: 0` disables it.
  - `episodes_per_seed`, `seed`, and `deterministic` specify its sampling.
  - An omitted seed defaults to training seed plus 100000.
- At the first completed update crossing each enabled boundary, and at the
  final update, the trainer freezes a portable checkpoint with its recipe and
  evaluates the trained teams.
  - Metrics include `eval.checkpoint.blue.mean_reward` and/or
    `eval.checkpoint.red.mean_reward`, logged at the checkpoint environment step.
  - Joint training evaluates learned Blue against learned Red. This curve is
    separate from the final Blue-versus-scripted-Red evaluation.
- See [post-training evaluation](evaluation.md#post-training-evaluation) for
  ordered final-checkpoint jobs and controls for running them separately.
