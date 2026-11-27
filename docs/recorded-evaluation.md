# Record and replay a Blue evaluation

Use a serial Torch Blue checkpoint evaluation when a reported mean needs a link back to the policy and individual episodes. The recorder retains the checkpoint and adjacent recipe, resolved settings, source revision and files, reset topologies, and each completed episode's reward.

## Record

Copy the [example protocol](examples/blue-evaluation-protocol.json) and declare the expected training replicates and conditions for your study. Choose a clean checkout, an evidence directory outside it, an MLflow tracking URI and an experiment name. Seeds below are explicit ordered environment seeds. Stochastic sampling uses consecutive policy seeds starting at 700. Add `--deterministic` for argmax.

```sh
export MLFLOW_TRACKING_URI="$TRACKING_SERVICE"
export MLFLOW_EXPERIMENT_NAME="jaxborg/$STUDY"
uv run python -m jaxborg.evaluation.capture run \
  --model "$MODEL" --protocol "$PROTOCOL" --trial blue-1 \
  --root "$EVIDENCE_ROOT" --revision "$(git rev-parse HEAD)" \
  --seeds 42 43 --policy-seed 700 --horizon 500
```

The command returns the attempt and evaluation IDs and the MLflow run ID when tracking starts. Save the run ID and service beside the study result. One run owns the attempt's artifacts under `evidence/`. Check the `jaxborg.discovery.evidence_path` tag for successful delivery. A local attempt can finish while tracking fails. If a run exists and delivery failed, use the same tracking URI and run `uv run python -m jaxborg.evaluation.capture sync --root "$EVIDENCE_ROOT" --attempt "$ATTEMPT_ID"`. A retry takes `--retry-of` with the previous attempt ID.

## Rebuild a mean

From a fresh checkout with access to the service, use the run ID selected by the study and a new download directory.

```sh
uv run python -m jaxborg.evaluation.replay \
  --tracking-uri "$MLFLOW_TRACKING_URI" --run-id "$RUN_ID" \
  --destination "$NEW_DOWNLOAD_DIRECTORY"
```

Replay verifies retained record relationships, file sizes and hashes, including checkpoint dependencies. It then reads the selected evaluation's episode rewards, computes their arithmetic mean and checks it against the recorded value. It works with compatible historical Jaxborg evaluation stores, including stores with training parents. Record the source revision of the replay alongside the result. This command rebuilds a single evaluation mean. Cross-run contrasts and uncertainty estimates use the study's own saved selection and analyzer.
