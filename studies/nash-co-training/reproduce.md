# Replay and browse the Nash retrospective

Start with the [study summary](README.md) and [archive index](archive-studies.json).
Use the unpacked `reproducibility-20261102` bundle and Python 3.11 or newer.
Use a Python environment with MLflow installed for the import and browser
steps. Replay itself uses the standard library.

Run the commands from the repository root. Set `BUNDLE` and `DEMO` to absolute
paths, with `DEMO` outside the source checkout and the input bundle.

```bash
BUNDLE=/path/to/reproducibility-20261102
DEMO=/path/to/local/nash-review
PYTHON=python3
mkdir -p "$DEMO"

"$PYTHON" studies/nash-co-training/analyze.py compare \
  --bundle "$BUNDLE" --output "$DEMO/analysis"
```

The analysis directory must be new. The command reads retained outcomes and
runs the published reproduction and provenance checks. It recomputes 60
contrasts in the four registered families and retains the source and input
bytes used for the comparison. It also records expected but missing GPU
cells, inherited GPU references, and the unstarted budget protocol. A
reconstruction uses the original correction families, aligned roots, and
frozen response selection. No training or rollouts are needed.

## Read the result

| File | Use |
| --- | --- |
| `report.md` | Study status, findings and interpretation |
| `contrasts.csv` | Recomputed effects and corrected intervals, with original family labels |
| `matchups.csv` | Cell membership, cohort, policy ages, run IDs, root coverage and missing cells |
| `response-selection.json` | Frozen validation selection and checkpoint identities |
| `manifest.json`, `source/`, `inputs/` | File hashes, exact analyzer, producer code and input evidence for replay |

For example, find the own-return gain of each selected fresh Red against
its frozen Blue, using Red150 as the common baseline. The rows use the same
four-contrast correction family as the defender comparisons.

```bash
"$PYTHON" - "$DEMO/analysis/contrasts.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    for row in csv.DictReader(stream):
        if row["study"] == "matched-red-response" and row["team"] == "red":
            gain = float(row["own_return_first_minus_second"])
            low = float(row["simultaneous_95_low"])
            high = float(row["simultaneous_95_high"])
            print(f'{row["contrast"]}: {gain:+.2f} [{low:.2f}, {high:.2f}]')
PY
```

To reconstruct a retrieved analysis in a new directory, run the retained
analyzer. The original bundle and MLflow are optional for this command.

```bash
"$PYTHON" "$DEMO/analysis/source/analyze.py" rebuild "$DEMO/analysis" \
  --output "$DEMO/rebuilt-analysis"
```

Read the original study reports for the earlier screening and behavior
analyses. Their raw data and scripts remain in the archive. Keep their
comparison memberships and statistical families separate from these four.

## Render the study figures

Use the project Python environment with the plotting dependencies installed.
The [figure script](plot.py) reads the replay tables and frozen selection.
It reuses the [cross-play heatmap](../../plots/plot_cross_play_fsm.py) and
draws the retained simultaneous intervals.

```bash
"$PYTHON" studies/nash-co-training/plot.py \
  --analysis "$DEMO/analysis" --output "$DEMO/figures"
```

The figures directory must be new. The script writes PNG and SVG versions of
the fresh Red response contrasts and CPU cross-play comparison.

## Browse in a separate local MLflow store

Import the original run histories, archived artifact bytes, question groups,
and replay artifacts into a new store. Allow at least 8 GB of free disk space
for the deduplicated artifact bytes and database.

```bash
"$PYTHON" studies/nash-co-training/import_mlflow.py \
  --bundle "$BUNDLE" --store "$DEMO/tracking" --analysis "$DEMO/analysis"

"$PYTHON" -m mlflow server --host 127.0.0.1 --port 5076 --workers 1 \
  --backend-store-uri "sqlite:///$DEMO/tracking/mlflow.db" \
  --default-artifact-root "$DEMO/tracking/artifacts" --no-serve-artifacts
```

Open [local MLflow](http://127.0.0.1:5076/), choose **Model training**, and
select a Nash experiment.
The import summary in `$DEMO/tracking/import-summary.json` maps original IDs
to local MLflow IDs. Use the archive runs for protocols, status, and
historical reports. In the historical-context experiment, open **Study archive and reports**
and find the reconstructed analysis under **Artifacts → replay**.
Retrieve that directory to replay it offline.

Each original run has one local identity. Search the
`nash.archive.original_run_id` tag to locate a producer by its original ID.
The original status and producer records remain available as provenance.
Latest-only metric values appear under `snapshot.*`, separate from recorded
time series. Shared inputs can be
referenced by more than one study. A missing evaluation or cancelled
protocol remains visible in the study index. Original MLflow tracking
destinations are provenance and the import uses the new SQLite store.
