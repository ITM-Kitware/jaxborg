# Reconstruct the frozen-defender response report

These commands rebuild the [reported result](README.md) from saved measurements.
They use NumPy and Matplotlib from the project lockfile and do not run training,
simulator evaluation, or MLflow. The analyzer validates this particular historical
fixture against its reference summary.

## Required inputs

Obtain the reference input bundle from the study author; no shared download
location has been established. The bundle is outside Git, and the indexer does
not download it. Its report is `fd73b4b6b6ed400aab12dc6a96dfe56b`, protocol owner
`af6bc98321dd4e9e892d76b0369e0d8e`, and producing revision
`c4f1b32c99faacb5ab229cbef056fa0e66367112`.

The bundle retains original producer files beneath
`producer/<run-id>/artifacts/`, with `index.json`, `analysis.json`, and a separate
`analysis-source/` snapshot. Required evidence includes:

- Source/report manifests, protocol, campaign, resolved recipes, selection/state
  record, source archives, dependency records, and reference summary/tables.
- Four validation evaluations with 100 episodes each, and two final evaluations
  with 600 paired cases each, including manifests and raw results.
- Three training manifests, raw `logs/metrics.jsonl` files, result records, and
  final checkpoint sidecars. Training logs supply curves and durations.

Model weights and topology bytes are unnecessary for these report calculations.
The original tracking-store identity is retained as provenance; the analyzer
does not connect to that store.

## Run and inspect

From the repository root:

```bash
uv run --frozen --extra cpu python studies/frozen-defender-response/analyze.py \
  --input /path/to/input-bundle --output /path/to/new-report
```

The output must be a new directory outside the checkout and input bundle.
Inputs are read only. After installing the project environment, add `--offline`
to run without network access, or use the `cuda` extra in an existing GPU
environment.

Read the generated `README.md` and `comparison.png` for the result,
`validation.csv` for selection, and `learning-curves.png` for challenger training.
`manifest.json` and `resolved-config.json` retain source, input identities,
settings, and output hashes. Numerical tables remain beside the figures.

The reference baseline Blue mean is `-2837.945`; the selected-opponent mean is
`-2949.9533333333334`, giving additional Red return `112.00833333333334`.
The 10,000-resample bootstrap with seed `3000001` reproduces the paired interval
`[86.68933333333334, 137.38225]` under the pinned numerical dependencies.

The analyzer checks seed cohorts, case counts and means, zero-sum rewards,
stochastic evaluation, horizon, observation/action dimensions, policy/topology
identities, source records, training completion, and validation-only selection
with Original Red fallback and recorded tie order. Validation/test overlap,
duplicate cases, incompatible contracts, missing files, or changed bytes reject
the report. Freeze assertions and policy hashes are checked as recorded evidence;
model tensors are not independently loaded or rehashed.

## Analysis variations and source snapshots

An external JSON configuration or bootstrap override creates a separate analysis:

```bash
uv run --frozen --extra cpu python studies/frozen-defender-response/analyze.py \
  --input /path/to/input-bundle \
  --output /path/to/another-report --bootstrap-seed 3000002
```

Outputs retain supplied settings and their resolved values. Changed bootstrap
settings explicitly skip comparison with the reference interval; raw means
still must match. Overrides do not change inputs or the recorded selection.

The original producing source and new analysis source have separate identities.
Outputs retain the analysis revision, dirty state, file hashes, runtime,
dependencies, command, and settings. Produce final outputs from a clean committed
analysis revision. Retained producer source is evidence and is not executed.
Without Git, revision and cleanliness are unknown but snapshot hashes remain.

A retained snapshot preserves the shared lockfile and repository-relative paths.
From its `analysis-source/` directory, run without installing absent trainer code:

```bash
uv sync --frozen --extra cpu --no-install-project
uv run --no-sync python studies/frozen-defender-response/analyze.py \
  --input /path/to/input-bundle --output /path/to/new-report
```

## Assemble this reference bundle

Use the original report's `results/state.json` to identify training, validation,
and final runs. Copy selected producer files without changing their bytes into
`producer/<run-id>/artifacts/`. Preserve analysis scripts and instructions at their
repository-relative paths in `analysis-source/`, together with `studies/README.md`
and root `pyproject.toml`, `uv.lock`, `.python-version`, and `README.md`. Exclude
environments and caches. Record the snapshot commit and hashes in
`analysis-source.json`, then index once:

```bash
uv run --frozen --extra cpu python studies/frozen-defender-response/index_bundle.py \
  --input /path/to/input-bundle \
  --source-store '<original tracking-store identity>'
```

The indexer writes hashes and initial settings and refuses to overwrite an
indexed bundle. Its source/report IDs are specific to this fixture. The analyzer
also verifies producer manifests; indexing alone does not establish scientific
validity. Retain a trusted index or checksum to detect replacement of the entire
bundle. For a new study, define its own input contract and analysis.
