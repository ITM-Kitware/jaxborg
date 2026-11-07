# Archived LSTM learning check — 2026-11-07

**Status:** completed CPU analysis of existing evaluations; no new training or
rollouts. Frozen at 2026-11-07 17:02:33 UTC: **69/96 cells, 4,480 episodes**.
This is one named check within the existing defender-transfer study.

**Question:** do the archived separate-encoder LSTM defenders improve from early
to final checkpoints, and does diversity help against common opposite-seed Red
lineages?

## Method and findings

Compare Blue at 0.96M and 49.968M training steps with the same retained layouts,
opponents and episode cases. Use the original stock game reward and Green
traffic. Each episode has 500 steps and stochastic policy sampling. The following
FSM Red results average four reserved stock-generator layouts, with 16 episode
roots per layout. Larger return is better.

| Blue training condition | Seed | Early return | Final return | Final minus early |
| --- | ---: | ---: | ---: | ---: |
| Single network | 42 | -1208.00 | -454.95 | +753.05 |
| Single network | 100 | -1312.92 | -485.38 | +827.55 |
| Diverse networks | 42 | -1114.19 | -467.36 | +646.83 |
| Diverse networks | 100 | -1233.19 | -432.53 | +800.66 |

All 28 completed aggregate scripted bank/opponent/Blue comparisons improve
return (range +236.55 to +956.38); five worsen at least one CIA proxy. These are
overlapping descriptive comparisons, not 28 independent training runs.

For each training seed, compare final single/diverse Blues against both Reds
from the **opposite seed**. Thus each contrast uses a common opponent and excludes
both Blues' own co-trained Red lineages. Positive differences favor diverse Blue.

| Blue seed | Layout panel | Against single-trained Red | Against diverse-trained Red |
| --- | --- | ---: | ---: |
| 42 | Anchor | -638.69 | -357.88 |
| 100 | Anchor | -609.00 | -353.19 |
| 42 | Reserved stock layouts | +5.11 | -51.11 |
| 100 | Reserved stock layouts | +65.75 | -207.14 |

The archived LSTMs demonstrably learned under this evaluation protocol.
There is **no consistent diversity advantage** on this related learned-Red panel.

## Interpretation limits

There are two Blue training seeds per condition. The opposite-seed Reds remain
related archive material, not an independent final holdout population. Original
exact training-source revision and stock training-layout bytes are unavailable;
reserved seeds do not establish genuinely novel layout bytes. This check does
not establish SOTA performance or validate the current trainer from scratch.

Final-stage coverage is complete (48 cells). Early coverage has all 16 scripted
cells and only five learned cells; 27 early learned cells are absent. The final
learned-Red comparison above does not depend on those missing cells. Scripted and
learned evaluations use different RNG contracts. CIA values are temporal
role-host impact proxies, not measured exfiltration or corruption.

## Retained evidence

[record.json](record.json) specifies comparison settings, exact source/protocol
identities, all 12 saved analysis hashes, coverage, resource requirements and
external evidence locators. [selection.json](selection.json) retains the exact
69 selected evaluation receipts. Its SHA-256 is
`34b3232840b970e9203c1666e20c0a14312f15d9e089ce773f901b557e9a5926`.

The producing evaluator revision is
`1545af8cb381ab2027c79814353b4b4963c7754c`. The original summary was
`LSTM-CHECK-20261107.md`; this tracked record makes that check durable in Git.
Weights, raw episodes, receipts and generated reports remain in the external
experiment store:

```text
ITM:
/home/local/KHQ/paul.elliott/jaxborg-exp/shared/studies/defender-transfer-20261107-v4/
  inputs/manifest.json
  evidence/
  reviews/lstm-check-20261107T170233Z/analysis/

Local WSL analysis mirror:
/home/paulhax/src/itm/cyber/experiments/defender-transfer-20261107/lstm-check-20261107T170233Z/analysis
```

The saved report's `policy-means.csv` supports the learning table;
`paired-contrasts.csv` supports the reward/CIA and opponent comparisons.
`coverage.csv`, `gaps.json`, `analysis-inputs.json` and `provenance.json` identify
coverage and retained inputs. Scheduler observations in the record describe the
snapshot time only.

## Reproduce without rollouts

Python 3.11+ and its standard library suffice. From the repository root, verify
all 12 files of the retained analysis and regenerate the check summary:

```bash
python studies/defender-transfer/check_learning.py \
  --analysis /path/to/lstm-check-20261107T170233Z/analysis \
  --exact-snapshot \
  --output /path/outside/checkout/new-learning-check
```

The new output contains `check.json`, the check code, record and exact selection.
It refuses to overwrite an existing output or to write inside this checkout.
No GPU, simulator rollout or MLflow evaluation is created.

To reconstruct from raw evidence, use the **retained analyzer** (or
`studies/defender-transfer/analyze.py` from the producing revision) and its
original Python 3.11.13 runtime. Keep this checked-in frozen selection:

```bash
study=/home/local/KHQ/paul.elliott/jaxborg-exp/shared/studies/defender-transfer-20261107-v4
selection=studies/defender-transfer/checks/lstm-learning-20261107/selection.json
python "$study/reviews/lstm-check-20261107T170233Z/analysis/analyze.py" \
  --manifest "$study/inputs/manifest.json" \
  --root "$study/evidence" \
  --selection "$selection" \
  --output /path/outside/checkout/new-analysis
python studies/defender-transfer/check_learning.py \
  --analysis /path/outside/checkout/new-analysis \
  --output /path/outside/checkout/new-learning-check
```

The maintained analyzer verifies selected artifact bytes and input/case
identities; this check consumes its outputs rather than implementing a second
episode analyzer. Reconstruction must match the retained analyzer, manifest,
selection and four numerical CSV hashes. Paths, Python metadata and inventory of
later attempts can differ, so only `--exact-snapshot` requires all saved report
metadata hashes to match. Changing the evidence selection creates a new check;
it must not silently revise this frozen result.
