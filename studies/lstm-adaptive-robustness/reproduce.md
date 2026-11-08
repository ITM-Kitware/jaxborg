# Reproduction

Use the committed source revision recorded in `inputs/manifest.json` and the
shared Python environment described in [setup](../../docs/setup.md). Inputs and
results stay outside the checkout. Preparation copies and verifies the exact
training layouts and owned Blue checkpoints from the completed competence
study; it does not regenerate a substitute bank.

```sh
python studies/lstm-adaptive-robustness/prepare.py prepare \
  --study-root "$STUDY_ROOT" --baseline-study "$BASELINE_ROOT" \
  --expected-revision "$REVISION"
python studies/lstm-adaptive-robustness/submit_jobs.py \
  --study-root "$STUDY_ROOT" --expected-revision "$REVISION" --phase validation
```

The resumed smoke restores update 2 of the four-update smoke and must reproduce
its final numeric state exactly. Run the recorded CPU evaluation smoke, then
`validate.py --study-root "$STUDY_ROOT" --expected-revision "$REVISION"
--require-cpu-eval`. Its acceptance record gates production. The separate
ten-update timing trial estimates throughput without entering scientific
comparisons. Then queue the complete paired blocks that fit the measured rate:

```sh
python studies/lstm-adaptive-robustness/submit_jobs.py \
  --study-root "$STUDY_ROOT" --expected-revision "$REVISION" --phase production
```

Each response is followed by recorded CPU cross-play against all four Blue
checkpoints. Submission records identify which four- or eight-run design was
actually queued; the protocol continues to list unstarted trials.

Run a declared trial with `run.py --study-root "$STUDY_ROOT"
--expected-revision "$REVISION" --trial br-blue42-primary-red17`. Use the
corresponding trial IDs in the protocol for other cells. To continue an
interrupted trial, preserve its original recipe and pass the exact retained
full-state archive with `--resume-state`, a fresh `--run-id`, and, when retrying
that trial, `--retry-of`. Resume is continuation within the original budget;
changing the budget or scientific settings requires a new declared experiment.

`validate.py --pairs` verifies initial Red parameter identity across completed
age pairs. Keep explicit attempt selections when a trial has retries. Production
and evaluation receipts identify the actual attempt, checkpoint artifacts,
source, and consumed inputs. Checkpoint selection follows the protocol; neither
a later retry nor a stronger intermediate attacker silently replaces a result.
