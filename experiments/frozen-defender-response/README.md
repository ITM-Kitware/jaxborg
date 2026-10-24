# Frozen-defender response analysis

This study-local command regenerates the completed MAPPO frozen-Blue response
report from saved measurements. It uses NumPy and Matplotlib from the main
project dependencies and lockfile. The analysis does not import JAX, the
simulator, training code, or MLflow.

From the repository root, run:

```bash
uv run --frozen --extra cpu python experiments/frozen-defender-response/analyze.py \
  --input /path/to/input-bundle --output /path/to/new-report
```

The output must be a new directory outside the source checkout and input bundle.
The input directory is read only. Once the project environment has been installed,
`uv run --offline --frozen --extra cpu ...` runs without network access. Use the
repository's `cuda` extra instead if working in an existing GPU environment.

## File ownership

```text
Source checkout:
  pyproject.toml          shared project dependencies, including NumPy and Matplotlib
  uv.lock                 exact resolved dependency versions
  .python-version         shared Python version
  experiments/frozen-defender-response/
    analyze.py           validation, statistics, tables, and figures
    index_bundle.py      index the copied records for this one reference study
    test_analyze.py        synthetic command-line checks
    README.md

Artifact directory outside the checkout:
  input/
    index.json           relative paths, hashes, and original run references
    analysis.json        captured report settings
    analysis-source/     snapshot of the committed analysis and dependency lock
    analysis-source.json snapshot revision and file hashes
    producer/
      <run-id>/artifacts/ original manifests, evaluations, logs, and source evidence
  report/
    manifest.json        analysis source, actual settings, inputs, output hashes
    resolved-config.json
    summary.json
    *.csv                 episode, validation, learning-curve, and tail evidence
    duration.json         measured training, evaluation, and pipeline durations
    *.png, *.svg          previews and vector figures
    analysis-source/     the generating script, lockfile, and instructions
    README.md
```

Important maintained presets may be committed, but individual parameter trials do
not require commits. The command accepts an external JSON configuration or
bootstrap overrides and captures the supplied values, defaults, overrides, and
fully resolved configuration in the output:

```bash
uv run --frozen --extra cpu python experiments/frozen-defender-response/analyze.py \
  --input /path/to/input-bundle \
  --output /path/to/another-report --bootstrap-seed 3000002
```

The original interval is checked exactly when the bootstrap settings match the
reference. With changed bootstrap settings, the output explicitly records that
the original interval was not compared. Raw means still must match the reference.
Inputs and recorded selection are not modified by an analysis override.

## Reference input bundle

The producing store identity is
`itm (sqlite:////home/local/KHQ/paul.elliott/jaxborg-exp/shared/mlflow.db)`.
This string identifies the original evidence; the analyzer never connects to it.
The report is `fd73b4b6b6ed400aab12dc6a96dfe56b`, protocol owner
`af6bc98321dd4e9e892d76b0369e0d8e`, and producing revision
`c4f1b32c99faacb5ab229cbef056fa0e66367112`.

The input bundle contains the original bytes of:

- Source and report manifests, source archives, lockfiles, and installed-environment records.
- Protocol, campaign, resolved training/evaluation recipes, and source/template sidecars.
- The report's selection/state record and reference summary/tables.
- All four validation evaluation manifests and raw results, 100 episodes each.
- Both final evaluation manifests and raw results, 600 paired cases each.
- All three training manifests, raw `logs/metrics.jsonl` files, training result records,
  and final checkpoint sidecars. Model weights and topology bytes are not required
  for these report-only calculations.

The three training attempts are identified in the retained selection record. Their
logs supply learning curves, tail comparisons, and training duration; exported
learning-curve tables are retained only as reference outputs.

To assemble this particular fixture, copy the selected producer files into a new
`input/producer/<run-id>/artifacts/` tree, preserving their bytes. The original
report's `results/state.json` identifies all training, validation, and final runs.
Retain the analysis scripts and instructions at their repository-relative paths,
plus the root `pyproject.toml`, `uv.lock`, `.python-version`, and `README.md`, in
`analysis-source/` beside the producer records. Exclude `.venv` and caches. Record
the snapshot's commit and file hashes in `analysis-source.json`.
Then index that copied tree once:

```bash
uv run --frozen --extra cpu python experiments/frozen-defender-response/index_bundle.py \
  --input /path/to/input-bundle \
  --source-store 'itm (sqlite:////home/local/KHQ/paul.elliott/jaxborg-exp/shared/mlflow.db)'
```

The indexer is specific to this reference study. It writes the input hash index
and initial analysis settings, and refuses to overwrite an indexed bundle. The
analyzer additionally verifies the producer manifests' completion and hashes;
indexing alone does not establish scientific validity. A trusted copy of the index
or its checksum is needed to detect replacement of the entire bundle.

## Scientific checks and limits

The command checks the prescribed seed cohorts, raw return lengths and means,
zero-sum rewards, stochastic evaluation, horizon, observation/action dimensions,
policy and topology identities, evaluator/source records, training completion,
and validation-only selection with Original Red fallback and the recorded tie
order. Every validation candidate is checked against its own completed training
checkpoint identity. Missing files, changed bytes, duplicate cases, incompatible
contracts, or validation/test overlap reject the report before final output appears.

The retained 600 cases reproduce baseline Blue mean `-2837.945`, selected-opponent
mean `-2949.9533333333334`, and additional Red return `112.00833333333334`. The
10,000-resample bootstrap with seed `3000001` reproduces the interval
`[86.68933333333334, 137.38225]` under the pinned numerical dependencies.

The uncertainty is conditional on these selected policies from one original
training run. It is not training-replication uncertainty or evidence of equilibrium.
The single topology was reconstructed, not an original training snapshot. Freeze
assertions and policy hashes are checked as recorded evidence; the command does
not independently load or rehash model tensors. Full policy reevaluation needs
weights, sidecars, topology bytes, and the producing execution environment.

The original source snapshot is separate from the new analysis source. Output
metadata records the analysis commit and any dirty state, source file hashes,
selected Python and dependencies, exact command, and actual settings. Final
outputs should come from the clean committed analysis revision. Files in the
source archive and producer manifests are retained evidence and are not executed.
Running the standalone analysis snapshot without Git records unknown Git revision
and cleanliness; the input snapshot identity and exact source hashes remain available.
The copied snapshot preserves the shared project lock and repository-relative
analysis paths. To run it without installing the absent trainer source, use these
commands from its `analysis-source/` directory:

```bash
uv sync --frozen --extra cpu --no-install-project
uv run --no-sync python experiments/frozen-defender-response/analyze.py \
  --input /path/to/input-bundle --output /path/to/new-report
```

## Focused verification

From the repository root, use the shared environment:

```bash
uv run --frozen --extra cpu \
  python -m unittest discover -s experiments/frozen-defender-response -p test_analyze.py
```

The tests use synthetic files and observe command exit status and saved outputs.
The real research bundle stays outside Git and is checked separately from a
second checkout and unrelated input/output directories. Numerical values and
links are compared; image metadata and path strings are not scientific identities.
