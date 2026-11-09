# Reconstruct the observation-value comparison

The retained `inputs/manifest.json` and `inputs/protocol.json` define the exact
policy hashes, layout hashes, calibration roots, held-out roots and expected
cells. Obtain those inputs from the shared study directory or the dedicated
MLflow experiment. Do not replace policies with newer checkpoints.

Use the retained source revision and its resolved environment. Each Blue policy
requires the following recorded stages, in order:

```sh
python studies/observation-value/compare.py replay --root "$STUDY_ROOT" \
  --blue "$BLUE_ID" --revision "$SOURCE_REVISION"
python studies/observation-value/compare.py calibrate --root "$STUDY_ROOT" \
  --blue "$BLUE_ID" --revision "$SOURCE_REVISION"
python studies/observation-value/compare.py fit --root "$STUDY_ROOT" \
  --blue "$BLUE_ID" --revision "$SOURCE_REVISION"
python studies/observation-value/compare.py run --root "$STUDY_ROOT" \
  --blue "$BLUE_ID" --revision "$SOURCE_REVISION"
```

The historical replay must reproduce both selected returns exactly for each
Blue. The study worker also downloads and verifies a pilot run's evidence
through MLflow before allowing calibration or evaluation.

Fitting uses every genuine calibration decision, with legal masks and teacher
probabilities. Sparse observation/carry records are solely for the descriptive
input interventions. The frozen baseline receipt records its hash and every
calibration artifact. Evaluation never refits the baseline.

To reconstruct a completed report without rerunning episodes, copy the selected
results, frozen baselines and sensitivity files to a fresh output root with the
same `inputs/manifest.json`, then run:

```sh
python studies/observation-value/compare.py analyze --root "$REPORT_ROOT" \
  --revision "$SOURCE_REVISION"
```

`report/comparison.json` retains the matched differences, confidence intervals,
selected evaluation identities and sensitivity summary. `report/episodes.csv`
contains all scored test cases. Native evidence and MLflow delivery validation
are checked separately with `verify.py`.
