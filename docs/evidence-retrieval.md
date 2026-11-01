# Browse runs and download evidence with MLflow

Use the MLflow UI or `MlflowClient` to browse experiments and runs, filter tags,
inspect settings and metrics, and download artifacts. Scientific `study_id` is
independent of the MLflow experiment used for storage. Choose explicit run IDs
and artifact paths; readable names and recency do not establish comparability.
MLflow resolves the storage location behind its artifact API.

```python
import os
from pathlib import Path
from mlflow import MlflowClient

client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
experiment = client.get_experiment_by_name(os.environ["MLFLOW_EXPERIMENT_NAME"])
runs = client.search_runs([experiment.experiment_id])
run_id = "<selected-run-id>"
artifact_path = "<selected-artifact-path>"
destination = Path("/results/downloads/selected-run")
destination.mkdir(parents=True, exist_ok=False)
downloaded = client.download_artifacts(run_id, artifact_path, str(destination))
```

Use the existing run's settings and recorded input identities to understand what
an artifact represents. A successful download alone does not establish scientific
completeness or comparability. Native downloads may leave partial files after a
failure; verify evidence before using it and retry into a fresh directory.
Automatic evidence uploads on recorded execution runs are integrated in
[#51](https://github.com/ITM-Kitware/jaxborg/pull/51).

## Optional evidence archives

When a finished analysis needs one portable snapshot, use
`jaxborg.evidence_archive.prepare_package` and `verify_package`. This supports
terminal evaluation-capture stores and indexed frozen-defender bundles. Running
attempts must finish first; failures and cancellations remain valid partial
evidence. Preparation preserves original identities, input relationships,
unknowns and missing cells. It is optional and does not create another MLflow run.
Attach the result to the existing execution or analysis run with native logging:

```python
from jaxborg.evidence_archive import prepare_package, verify_package

manifest = prepare_package("evaluation-capture", evidence_root, archive_directory)
client.log_artifacts(run_id, str(archive_directory), artifact_path="report-evidence")
```

Download `report-evidence` through the same native API, then call
`verify_package(downloaded_directory, expected_digest)` using the saved manifest
digest when available. Verification checks the expected file set, hashes/sizes,
record relationships and completion. `package_paths` exposes the retained inputs,
selection and materialization manifest. Downloading and verification do not
execute archived code. Rebuild the analysis into a separate output directory.

A frozen-defender selection can be supplied with `selection=selection_file`.
The [frozen-defender guide](../studies/frozen-defender-response/README.md)
documents its analyzer and original-provenance materialization inputs. Retained
inputs do not make unretained checkpoints reevaluable.

Durable service ownership, authorized access and a real second-client
demonstration remain acceptance work in
[#32](https://github.com/ITM-Kitware/jaxborg/issues/32).
