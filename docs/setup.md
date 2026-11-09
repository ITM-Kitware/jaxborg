# Project setup

- Next: [training](training.md), [evaluation](evaluation.md), or [composing a study](../studies/README.md).
- Commands below run from the repository root. Paths such as `/absolute/path/...`
  are placeholders to replace with your chosen locations.
- On ITM, follow [Studies on ITM](itm-studies.md) for shared storage, the
  dedicated Jaxborg tracking destination, and checkpoint budgets.

## Python environment

- Install dependencies with one explicit hardware extra:

  ```bash
  uv sync --extra cpu
  # Or, on an NVIDIA Linux training host:
  uv sync --extra cuda
  ```

- CPU dependencies support tests and CPU evaluation. CUDA dependencies support
  GPU training; confirm that JAX sees the intended device before a GPU workload.
- The [JAX environment script](../scripts/jax_env.sh) configures persistent
  compilation caching. The JAX launcher sources it; source it before direct
  Python invocations when you want the same cache settings.
- Cluster allocation and shared-machine policy are host-specific. Read the
  execution host's instructions before launching work.

## Experiment storage

- Set the native output root before training, evaluation, or plotting:

  ```bash
  export JAXBORG_EXP_DIR=/absolute/path/outside-the-checkout/experiments
  ```

- The launcher defaults to `./jaxborg-exp` when this is unset. For studies,
  keep generated evidence outside the source checkout.
- Keep these locations distinct:
  - Source: maintained recipes, study methods, and analyzer code.
  - Native outputs: logs, checkpoint bundles, recipe sidecars, and evaluation files
    below `JAXBORG_EXP_DIR`.
  - Retained research evidence: the explicit `--evidence-root` used by supported
    recorded execution paths.
  - MLflow: the tracking destination and its artifact store. An experiment name
    groups tracked runs; it is independent of scientific study identity.
- Use storage and tracking destinations designated for this project. On a shared
  installation, read its documented configuration before production launches.
  For isolated development, follow the dedicated local-instance setup in the
  [tracking guide](evidence-retrieval.md).

## MLflow

- For an existing file-backed experiment store, browse its database with:

  ```bash
  ./scripts/train/view_mlflow.sh
  ```

  - The script reads `$JAXBORG_EXP_DIR/mlflow.db` and defaults to
    `http://127.0.0.1:5001`.
  - `--port PORT` changes the UI port; `--remote` serves the mirror under
    `remote/jaxborg-exp`.
- An explicit `MLFLOW_TRACKING_URI` takes precedence over the native output
  directory's tracking database. `MLFLOW_EXPERIMENT_NAME` selects the destination.
- Use [MLflow browsing and evidence retrieval](evidence-retrieval.md) for
  service configuration, isolated development setup, artifact downloads, and
  reconstructing selected reports.
