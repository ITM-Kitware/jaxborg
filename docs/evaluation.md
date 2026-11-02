# Evaluation

- First configure [the environment, output storage, and MLflow](setup.md).
- [Training](training.md) produces checkpoint bundles and recipe sidecars.
- [The study guide](../studies/README.md) defines comparison methods, replicate
  identities, and saved selections. The commands here are shared tool usage;
  study READMEs specify which tests answer their scientific questions.

## Choose an evaluation path

- **Recorded research evaluation:**
  - [Checkpoint capture](evaluation-capture.md) evaluates native Torch `.pt`
    policies in CybORG, retaining inputs, source, episodes, failures, and retries.
    `eval_recipe.py` uses this mode by default and requires protocol membership.
  - [Joint execution records](joint-execution-records.md) describe supported
    JAX checkpoint matchups and scripted-Red capture, including topology and
    policy ownership requirements.
  - Follow each path's seed contract. Matching numeric seed labels across
    different executors does not establish matched episodes or topologies.
- **Development evaluation:**
  - `eval_recipe.py --development-no-capture` explicitly opts out of retained
    research records. This also supports the JAX-to-CybORG path that capture
    does not currently support.
  - Example, using a checkpoint with an adjacent recipe sidecar:

    ```bash
    uv run python scripts/eval/eval_recipe.py --development-no-capture \
      --model /absolute/path/model_policy.pt \
      --episodes-per-seed 1 --seeds 42-51
    ```
- **Other shared evaluators:**
  - [Learned-policy matchup](../scripts/eval/eval_matchup.py): select a recipe,
    Blue and Red model paths or experiment names, seed list, and episodes per seed.
  - [CybORG scripted-Red sweep](../scripts/eval/eval_scripted_reds.py): an unpaired
    CybORG contract sweep; it does not replay JAX topology snapshots or fixed CIA roles.
  - [JAX scripted-Red sweep](../scripts/eval/eval_scripted_reds_jax.py): configured
    JAX environment and held-out topology evaluation.
  - Use each script's `--help` for its supported arguments. Native outputs from
    these tools do not all have the same research retention contract.

## Post-training evaluation

- Recipes can declare an ordered `eval.after_training` list. Each job runs in a
  fresh Python process after the durable final bundle and sidecar are saved:

  ```yaml
  eval:
    after_training:
      - name: stochastic-reds
        script: scripts/eval/eval_scripted_reds.py
        args: [--reds, fsm, cia_c, --seeds, 1000-1099, --workers, 8]
  ```

  - The runner supplies `--model` automatically. `model_arg` selects another flag.
  - Jobs are required by default; `required: false` records a failure and continues.
  - Argument placeholders include `{model}`, `{recipe}`, `{backend}`, `{exp_dir}`,
    `{eval_dir}`, and `{name}`.
  - A required failure makes the command fail while preserving the saved model
    and evaluation manifest.
- Re-run the configured list from an existing sidecar:

  ```bash
  JAX_PLATFORMS=cuda uv run python scripts/eval/run_after_training.py \
    --model /absolute/path/model_policy.safetensors
  ```

- Environment controls:
  - `JAXBORG_SKIP_POST_TRAINING_EVAL=1` skips play-priors and after-training jobs.
    Scope it to the trainer command so a later evaluator does not inherit it.
    It preserves training metrics, model saving, and separately configured
    in-training checkpoint evaluation.
  - `JAX_PLATFORMS` selects the child platform; default is CUDA for a JAX final
    bundle and CPU for a CybORG final bundle. Use `cpu` for CPU-only execution.
  - `JAXBORG_EVAL_BATCH_SIZE` defaults to 64 episodes per batched JAX matchup
    call. Reduce it to lower peak device memory.
- A later post-training invocation starts the list again and writes a new
  manifest; it does not resume the earlier list.
- Co-training recipes may also configure:
  - `eval.cross_play`: historical Blue/Red checkpoint matchups.
  - [Cross-seed evaluator](../scripts/eval/eval_cross_seed_only.py): final
    policies from different training seeds.
  - Inspect the selected recipe to establish the exact matchups and test bank.

## Inspect results

- Evaluators using native JSONL outputs write them under `$JAXBORG_EXP_DIR/eval`.
  Ordered post-training runs also write command/status manifests in
  `eval/manifests` and named MLflow metrics under `eval.after_training.<name>.*`.
- Recorded research evaluation retains its measurements in the declared evidence
  store. It may replace native JSONL output; check the selected evaluator's
  recording guide for its output contract.
- Plot native evaluation files or discover MLflow metrics:

  ```bash
  uv run python scripts/eval/plot_results.py
  uv run python plots/plot_mlflow.py ippo-cc4 --list-metrics
  uv run python plots/plot_mlflow.py ippo-cc4 --name returns --metric 'team.*.return'
  ```

  - Native comparison plots go under `eval/plots` in the output root.
  - MLflow figures default to `plots/<experiment>` in that root, with PNG and PDF
    output. Use `--recipe`, `--backend`, `--seed`, or `--run-id` to select runs.
  - The MLflow plotter groups by recipe and aggregates repeated seeds. A study
    analysis must still state its exact selected runs and uncertainty method.
- [Evidence retrieval](evidence-retrieval.md) explains browsing, downloads,
  byte verification, and reconstruction of saved selections.
