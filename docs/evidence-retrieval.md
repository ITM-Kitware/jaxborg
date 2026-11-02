# Browse runs and download evidence with MLflow

Use the MLflow UI or `MlflowClient` to browse experiments and runs, filter tags,
inspect settings and metrics, and follow recorded input identities. Configure
`MLFLOW_TRACKING_URI` for the authorized service and `MLFLOW_EXPERIMENT_NAME`
for the destination. Scientific `study_id` is independent of the experiment name.
These settings do not deploy a shared service.

For development against an existing MLflow server on the execution machine,
select it explicitly and keep the pilot in its own experiment:

```sh
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_EXPERIMENT_NAME=red-update-development
export JAXBORG_EXP_DIR=/absolute/path/outside-the-checkout/development-runs
```

`JAXBORG_EXP_DIR` holds native training outputs; the training command's
`--evidence-root` selects retained evidence. The explicit tracking URI takes
precedence over a database beneath `JAXBORG_EXP_DIR`. Keep both output locations
outside the source checkout. These commands use the existing service without
starting, stopping, migrating or reconfiguring it.

When the server runs on another machine, a local SSH forward makes its UI and
artifact API available to the client:

```sh
ssh -N -L 5001:127.0.0.1:5000 <execution-host>
```

Open `http://127.0.0.1:5001` and use that URI for client-side downloads. The
training process on the execution host continues to use port 5000. Verify a
download through the API below; a browser connection alone does not establish
that required artifact bytes are available to the client.

Recorded training and evaluation use their existing execution run. At completion,
they automatically upload a native evidence store under `evidence/`, including
that attempt's exact inputs, source, observations and checkpoints, and the retained
attempts it depends on. Unrelated attempts are excluded, so other running work
does not need to finish first. Failed and cancelled attempts retain useful partial
work. No separate package publication or custom search index is required.

## Browse and follow inputs

For example, use the same tag filter in the UI or Python:

```python
import os
from mlflow import MlflowClient

client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
experiment = client.get_experiment_by_name(os.environ["MLFLOW_EXPERIMENT_NAME"])
runs = client.search_runs(
    [experiment.experiment_id],
    filter_string="tags.`jaxborg.discovery.study_id` = 'my-study'",
)
```

Select an explicit run ID after inspecting its settings and status. An attempt
has its own `jaxborg.discovery.attempt_id`; evaluation runs also expose an
`evaluation_id`. `execution.json` holds the protocol, records, exact input artifact
identities and available producing relationships. An attempt ID can be looked up
with another native tag filter. Names and recency do not establish comparability.

`jaxborg.discovery.scientific_status` reflects retained execution status.
`delivery` and `data_incomplete` expose tracking delivery problems. Runs remain
`state=unverified`: the UI is navigation, not verification of downloaded bytes.
The `evidence_path` tag is set after terminal evidence upload succeeds. A terminal
run with failed delivery can require an explicit tracking refresh; see the
[recording guide](joint-execution-records.md#live-tracking-navigation).

## Download and verify

An MLflow artifact is addressed by run ID and artifact path; MLflow resolves its
storage location. Use native downloads to a fresh directory outside the checkout:

```python
from pathlib import Path
from jaxborg.evaluation.capture_store import load_records

run_id = "<selected-run-id>"
destination = Path("/results/downloads/selected-run")
destination.mkdir(parents=True, exist_ok=False)
evidence = client.download_artifacts(run_id, "evidence", str(destination))
records = load_records(evidence, verify=True)
```

Verification checks the retained record relationships, file hashes/sizes,
completion, budgets and observations. A failed execution can still have valid
partial evidence; it does not become a successful execution when downloaded.
Native downloads may leave partial files on failure. Treat the directory as
unverified until verification succeeds; retry into a fresh directory.

Use retained instructions and source to reconstruct the analysis in a separate
output directory. Downloading and verifying do not execute archived source.
A report should retain its explicit selection and analyzer identity and point to
its input runs/artifacts. A checkpoint's presence alone does not establish
successful training or an independently replicated result.

## Optional report archives

A finished report sometimes needs one portable snapshot of its selected evidence,
outputs and exact analyzer source. `jaxborg.evidence_archive.prepare_package`
and `verify_package` support the maintained frozen-defender and evaluation-capture
formats. This is an optional report-sharing operation, not a prerequisite for
browsing training runs. The capture report option supports the collation plot,
not arbitrary report directories.

For a supported collation report, attach the archive to the existing analysis run
using ordinary MLflow artifact logging:

```python
from jaxborg.evidence_archive import prepare_package, verify_package

manifest = prepare_package(
    "evaluation-capture", evidence_root, archive_directory,
    selection=selection_file, report=report_directory,
)
client.log_artifacts(report_run_id, str(archive_directory), artifact_path="report-evidence")
```

Download `report-evidence` through the same native API, then call
`verify_package(downloaded_directory, expected_digest)` using the saved manifest
digest when available. `package_paths` exposes input, selection, report and
analyzer paths. Preserve the downloaded archive and rebuild into a different
output directory. Its archived `pyproject.toml` and `uv.lock` specify dependencies;
the partial source snapshot does not need to be installed as a project.

For a capture collation, explicitly invoke its saved analyzer after verification:

```bash
ARCHIVE="$DOWNLOADED/report/analysis-source"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ARCHIVE/src:$ARCHIVE" \
  python "$ARCHIVE/scripts/eval/plot_results.py" \
  --collation "$DOWNLOADED/selection.json" \
  --evidence-root "$DOWNLOADED/input" --output "$REBUILT_REPORT"
```

Use the retained study entrypoint for study-specific checks. The
[frozen-defender guide](../studies/frozen-defender-response/README.md) documents
that report's analyzer and materialization inputs. Retained inputs do not make
unretained checkpoints reevaluable.

Durable service ownership, authorized colleague access and a real second-client
demonstration remain acceptance work in
[#32](https://github.com/ITM-Kitware/jaxborg/issues/32).
