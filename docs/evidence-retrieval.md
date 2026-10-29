# Retrieve retained evidence

`python -m jaxborg.experiment_discovery` prepares an immutable package, publishes
it to an explicitly selected MLflow experiment, and retrieves an exact revision.
Inputs are an [indexed frozen-defender bundle](local-evidence-producers.md) or a
terminal [evaluation-capture store](evaluation-capture.md). Running attempts must
finish first; failed/cancelled attempts remain valid evidence. Publication and
retrieval preserve scientific status and partial budgets.

Use the repository's Python environment and keep generated outputs outside the
checkout. Set `URI` to an authorized MLflow service with HTTP artifact proxying
and `EXPERIMENT` to its storage destination. There is no default destination.
`--study` filters protocol `study_id`; it need not match the experiment name.
Neither selects a recipe or launches work.

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
--output "$PACKAGE"`. Preparation retains verified terminal records and every
referenced file. Later attempts require a new package/digest. Record IDs,
original artifact locators, unknown-fact reasons, and unretained checkpoint
references are preserved.

Successful commands write one JSON object to stdout. Save the publication's
`run_id` and `sha256`; readable names do not identify revisions. Search returns
`{"publications": [...]}` and accepts exact `--protocol`, `--trial`, and `--record`
filters. Search reads a small advisory index, never evidence payloads. Rows have
`verified: false` and `available: null`; a published `state` tag is only a
navigation hint. `match: candidate` means the index matches the filters;
`match: unknown` with `index_error` means missing metadata prevents a decision.
Search retains matching revisions and reports duplicate digest candidates.
`identity_conflicts` reports differing candidate record hashes, not verified
scientific contradictions. Inspect or fetch an exact run and digest to verify.

```bash
python -m jaxborg.experiment_discovery inspect \
  --tracking-uri "$URI" --run-id "$RUN_ID" --sha256 "$PACKAGE_SHA256"
python -m jaxborg.experiment_discovery fetch \
  --tracking-uri "$URI" --run-id "$RUN_ID" --sha256 "$PACKAGE_SHA256" \
  --output "$DOWNLOADED"
```

Inspect verifies the publication and exposes its digest, kind, records,
execution relationships, and missing trial cells. Fetch checks completion,
the retained record graph, file sizes/hashes, safe paths, and native input validity
before exposing a new output directory. Each package has one native publication
run with its study/protocol tags, search index, completion marker and package
artifacts. Search filters trial and record identities from that index; inspect
and fetched records expose the complete relationships, facts and unknowns.
Existing outputs are rejected and
failures leave no completed destination. Use one local writer per destination.
Returned paths are absolute:

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

`selection` is null if none was supplied. Fetch does not execute bundled source.
To [rebuild the frozen report](../experiments/frozen-defender-response/README.md),
use the returned paths and the analyzer/helper source identity from its saved
collation:

```bash
python experiments/frozen-defender-response/analyze.py \
  --input "$DOWNLOADED/input" --selection "$DOWNLOADED/selection.json" \
  --materialization "$DOWNLOADED/manifest.json" --output "$REPORT"
```

The manifest verifies relocated bytes while preserving original provenance and
collation. Source, configuration, and scientific selection checks still apply.
Retained report inputs do not make unretained checkpoints reevaluable.

Use one publisher per package/experiment. Repeating an identical publication
verifies and reuses it. Incomplete, conflicting, or ambiguous candidates fail
without automatic repair or selection by recency. These failures return a
nonzero exit and
`{"error":{"type":"PublicationConflict","message":"...","candidate_run_ids":["..."]}}`;
other failures return `error.type` and `error.message`.

A shared deployment, access/ownership decision, and independent retrieval of an
authorized real reference bundle remain acceptance work for
[#32](https://github.com/ITM-Kitware/jaxborg/issues/32).
