# Retrieve retained evidence

`python -m jaxborg.experiment_discovery` prepares a verified immutable package,
indexes it in an explicitly selected MLflow experiment, and retrieves an exact
publication. Supported inputs are the [indexed frozen-defender bundle](local-evidence-producers.md)
and a terminal [evaluation-capture store](evaluation-capture.md). Running attempts
must finish before preparation. A failed or cancelled attempt remains valid
evidence of that execution.

Run these commands with the repository's Python environment. Keep generated
packages, downloads and reports outside the checkout. Set `URI` to an authorized
MLflow tracking service with HTTP artifact proxying, and `EXPERIMENT` to its
destination experiment. There is no implicit tracking destination.

`--experiment` names the MLflow storage destination. `--study` filters the
scientific `study_id` already carried by protocol records. These names may
differ; neither selects a recipe, loads a study directory, or launches work.

```bash
python -m jaxborg.experiment_discovery prepare \
  --kind frozen-defender --input "$INPUT" --selection "$COLLATION" \
  --output "$PACKAGE"
python -m jaxborg.experiment_discovery publish \
  --package "$PACKAGE" --tracking-uri "$URI" --experiment "$EXPERIMENT"
python -m jaxborg.experiment_discovery search \
  --tracking-uri "$URI" --experiment "$EXPERIMENT" --study "$STUDY_ID"
```

For capture input, use `prepare --kind evaluation-capture --input "$STORE"
--output "$PACKAGE"`. Preparation retains authoritative terminal records and
their verified file closure. Later capture attempts require another package and
produce another digest. Original record IDs, artifact locators, unknown-fact
reasons and unretained checkpoint references remain unchanged.

Every successful command writes one JSON object to stdout. Save `run_id` and
`sha256` from the publication receipt; readable names do not identify a revision.
For example, the receipt has this shape (identifiers shown as placeholders):

```json
{"run_id":"RUN_ID","sha256":"PACKAGE_SHA256","state":"verified","reused":false}
```

Search returns `{"publications": [...]}` and supports exact `--protocol`,
`--trial` and `--record` constraints. Results retain every matching package
revision. An `identity_conflicts` entry identifies record IDs whose retained
content differs across matching publications, with their candidate run IDs.
Inspect verifies a selected publication and exposes its digest, kind, records,
execution relationships and declared missing trial cells:

```bash
python -m jaxborg.experiment_discovery inspect \
  --tracking-uri "$URI" --run-id "$RUN_ID" --sha256 "$PACKAGE_SHA256"
python -m jaxborg.experiment_discovery fetch \
  --tracking-uri "$URI" --run-id "$RUN_ID" --sha256 "$PACKAGE_SHA256" \
  --output "$DOWNLOADED"
```

Fetch checks publication completion, record projection, file sizes and hashes,
safe paths and native input validity before exposing a fresh output directory.
An existing output is rejected; failed retrieval leaves no completed destination.
Use one local writer per output destination. Its result supplies absolute paths:

```json
{
  "run_id":"RUN_ID",
  "sha256":"PACKAGE_SHA256",
  "state":"verified",
  "paths":{
    "input":"/downloads/evidence/input",
    "records":"/downloads/evidence/records.json",
    "selection":"/downloads/evidence/selection.json",
    "materialization":"/downloads/evidence/manifest.json"
  }
}
```

`selection` is null when no collation was supplied. Fetch never executes bundled
source. To [rebuild the maintained frozen report](../experiments/frozen-defender-response/README.md),
use the returned paths explicitly with the same analyzer/helper source identity
recorded by the saved collation:

```bash
python experiments/frozen-defender-response/analyze.py \
  --input "$DOWNLOADED/input" --selection "$DOWNLOADED/selection.json" \
  --materialization "$DOWNLOADED/manifest.json" --output "$REPORT"
```

The materialization manifest verifies the relocated bytes while retaining the
original provenance locators and unchanged collation. Source, configuration and
scientific selection checks still apply. Retained report inputs do not imply
that unretained policy checkpoints can be reevaluated.

Publication uses a single writer per package/experiment in this pilot. Repeating
an identical publication verifies and reuses the existing content. Incomplete,
conflicting or ambiguous candidates fail explicitly and list candidate run IDs;
there is no automatic repair or choice by recency. A nonzero command returns
`{"error":{"type":"PublicationConflict","message":"...","candidate_run_ids":["..."]}}`
for these conflicts. Other failures return `error.type` and `error.message`.
Transport completion remains separate from scientific execution status: a fully
retrieved failed evaluation still has a failed status and partial episode budget.

The disposable local HTTP acceptance service demonstrates software retrieval.
A shared deployment, its access/ownership decision and an independent retrieval
of an authorized real reference bundle remain separate acceptance work for
[#32](https://github.com/ITM-Kitware/jaxborg/issues/32).
