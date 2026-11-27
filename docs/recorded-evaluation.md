# Record and replay a Blue evaluation

Use the recorder to link a serial Torch Blue evaluation to its policy, resolved settings and individual episode rewards.

## Record

Copy the [example protocol](examples/blue-evaluation-protocol.json) and declare the conditions and training replicates. Start from a clean checkout and choose an evidence directory outside it.

```sh
export MLFLOW_TRACKING_URI="$TRACKING_SERVICE"
export MLFLOW_EXPERIMENT_NAME="jaxborg/$STUDY"
uv run python -m jaxborg.evaluation.capture run \
  --model "$MODEL" --protocol "$PROTOCOL" --root "$EVIDENCE_ROOT" \
  --seeds 42 43 --policy-seed 700 --horizon 500
```

- **Trial and revision.** A single-trial protocol supplies the trial ID. Pass `--trial` for multiple trials. The recorder captures the checkout commit. Pass `--revision "$PLANNED_COMMIT"` to enforce a pin.
- **Episode choices.** Supply ordered environment seeds. Sampling uses consecutive policy seeds starting at 700 in this example. Add `--deterministic` for argmax.
- **Training lineage.** To verify the checkpoint and recipe hashes against a training run in this service, pass `--producer-run-id "$TRAIN_RUN_ID"`.

## Find the evidence

Link the evaluation run from the study result. In MLflow, the `jaxborg.discovery.evidence_path` tag points to delivered artifacts. Under Tags, `trained_backend`, `checkpoint_format` and `eval_env` share the `jaxborg.discovery.` prefix. The training simulator comes from the saved recipe when available. The evaluator identifies the format it loads and the simulator it runs.

| IPPO filename | Contents |
| --- | --- |
| `model_*.pt` | Torch policy weights for evaluation |
| `checkpoint_*.pt` | Torch training state for resuming, including policy and optimizer |
| `model_*.safetensors` | Flax policy parameters. For a multi-team bundle, consult its embedded policy metadata. |

The recipe evaluator also writes these fields to its result JSON. For a linked training run, open **Artifacts** and select the JSON under `evaluations/`.

## Rebuild a mean

With the tracking service selected, pass the recorded run ID and a new download directory.

```sh
uv run python -m jaxborg.evaluation.replay \
  --run-id "$RUN_ID" --destination "$NEW_DOWNLOAD_DIRECTORY"
```

Replay verifies the retained files and hashes, rebuilds the arithmetic mean from episode rewards, and compares it with the saved metric. Cross-run comparisons use the study's selection and analyzer.

**Repair delivery.** If a run exists but evidence delivery failed, run `uv run python -m jaxborg.evaluation.capture sync --root "$EVIDENCE_ROOT" --attempt "$ATTEMPT_ID"` against the same service. For a new execution, pass `--retry-of "$ATTEMPT_ID"` to the recorder.
