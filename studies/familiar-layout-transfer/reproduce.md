# Replay the defender comparison

The protocol uses four saved joint IPPO-LSTM bundles at 49.92M steps. Obtain the bundles and their recipe YAML sidecars from the study archive. Arrange them below a checkpoint root using the relative directories and names in [the protocol](protocol.json). The evaluation checks each checkpoint hash and its training seed, source revision and step. The [comparison tables](results.md) are available in this study. The saved weights are an input dependency.

Use JAXborg source revision [45957e2e](https://github.com/ITM-Kitware/jaxborg/tree/45957e2e18fea380bb81d2a62cb2c7ddb5c2034e), Python 3.11 and CPU JAX/JAXlib 0.10.2. The study code imports that revision via `PYTHONPATH`. Install its locked CPU environment with `uv sync --locked --python 3.11 --extra cpu --no-dev`, then use that environment's Python for the commands below. Use a clean output directory on the shared data drive and configure the documented JAXborg MLflow service before executing. On ITM, follow the shared study setup for cache and temporary paths.

Run the preparation once, then run task indices 0 through 9 for each panel. The first six tasks evaluate the six scripted Reds. The last four evaluate the two learned-Red conditions for each Blue training seed.

```sh
export JAX_PLATFORMS=cpu
export JAXBORG_EVAL_BATCH_SIZE=10
export PYTHONPATH="$PINNED_JAXBORG/src"

python "$STUDY/replay.py" "$OUTPUT" --prepare

python "$STUDY/replay.py" "$OUTPUT" \
  --panel held-out --task "$SLURM_ARRAY_TASK_ID" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --experiment "$MLFLOW_EXPERIMENT_NAME"
```

Use the same array indices with `--panel familiar` to replay map 0. Tasks write distinct directories and log episode records to the study's tracking experiment. Use a fresh output directory when repeating a completed task.

Generate the report after the two panels finish.

```sh
python "$STUDY/analyze.py" "$OUTPUT" "$REPORT"
```

To reuse the original map 0 measurements, supply their learned and scripted result directories with `--archived-learned` and `--archived-scripted`. The analyzer requires the complete expected matrix and checks the matched episode seeds. It summarizes by layout, opponent and Blue training seed, then compares equally weighted layout means across panels.

The familiar layout was regenerated from its original generator revision and seed. Its fingerprint matches the map 0 evaluation artifact. The original training cache was unavailable for direct comparison.
