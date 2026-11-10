# Reconstruct the observation-value comparison

The README's figure can be rebuilt from the accompanying compact measurement
snapshot, without training or simulator evaluation:

```sh
python studies/observation-value/plot.py \
  --report studies/observation-value/reported-results.json \
  --output-dir /tmp/observation-value-figure
```

Use a new output directory outside the checkout. `reported-results.json` contains
the per-policy means, paired root differences and confidence intervals, plus
the earlier action-selection comparison and sensitivity summaries. It is a
reporting snapshot; full inputs, checkpoints and native evidence remain in the
external study store described below.

The concise briefing with charts is retained under `report/brief-20261110/` in
`/data/shared/jaxborg/studies/observation-value-20261109/` on ITM. It includes
the selected reports, exact plotting values and a standalone `render.py`.
The earlier sampling-versus-argmax chart is descriptive; its episode roots differ
from the follow-up comparison and the gaps must not be combined.

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

Export the completed comparison and its plotting inputs with:

```sh
python studies/observation-value/plot.py \
  --report "$REPORT_ROOT/report/comparison.json" --output-dir "$FIGURE_ROOT"
```

The figure exports retain the report hash, plotting-source hash, exact estimates
and intervals. Keep the output directory outside the source checkout.

## Interpretation and related evidence

Each baseline imitates the original policy's full action probabilities at genuine
calibration decisions, respecting legal masks. It does not optimize return, and
the comparison does not rule out a stronger controller without live evidence.
All baselines were fitted before their test evaluations. The retained manifest
defines the independent calibration and test roots.

Controlled swaps of complete host evidence changed the top action in 59–65% of
eligible calibration decisions with mask and incoming carry held fixed. Both
LSTMs also depended on incoming carry. These checks establish input dependence;
the return comparison removes phase, static context, evidence and history
together and does not isolate useful threat response or memory.

Intervals describe fresh-root variation conditional on these opponents, familiar
layouts and fitted baselines. They exclude uncertainty from refitting and new
training runs. Two LSTM training seeds and one MAPPO trajectory do not establish
a population-wide or architecture-level result.

The earlier [host-level behavior check](../lstm-fsm-competence/checks/behavior-20261108/results.md)
contains exploratory sensing and recovery examples from the same final LSTM 42
checkpoint. These illustrate individual episodes, rather than the cause of the
sampling-versus-argmax gap.

Evaluation source revision: `a102881a791eb96f71da542725aa3f5b908bb51d`.
Tracking experiment: `jaxborg/observation-value-20261109` (ID 5).
Report run: `4ca41b21d74f4ca980d24b9e29a7cd05`.
