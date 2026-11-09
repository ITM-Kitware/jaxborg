# Reproduction

Use the committed source revision recorded in `inputs/manifest.json` and the
shared Python environment described in [setup](../../docs/setup.md). Inputs and
results stay outside the checkout. Preparation copies and verifies the exact
training layouts and owned Blue checkpoints from the completed competence
study; it does not regenerate a substitute bank. Choose `--production-updates`
from 208, 416, 520 or 624 before preparation, based only on timing and the deadline.
The frozen manifest and materialized protocol retain the chosen common horizon.
Every production run retains update 208 as the primary comparison; longer final
checkpoints are secondary. A changed horizon requires a fresh prepared root and
validation; do not alter an existing run or its resume contract.

```sh
python studies/lstm-adaptive-robustness/prepare.py prepare \
  --study-root "$STUDY_ROOT" --baseline-study "$BASELINE_ROOT" \
  --expected-revision "$REVISION" --production-updates 624
python studies/lstm-adaptive-robustness/submit_jobs.py \
  --study-root "$STUDY_ROOT" --expected-revision "$REVISION" --phase validation
```

The resumed smoke restores update 2 of the four-update smoke and must reproduce
its final numeric state exactly. Run the recorded CPU evaluation smoke, then
`validate.py --study-root "$STUDY_ROOT" --expected-revision "$REVISION"
--require-cpu-eval`. Its acceptance record gates production, including exact agreement with the independent timing run at update 2. The launcher sets deterministic GPU operations; preserve that setting when resuming. The separate
ten-update timing trial estimates throughput without entering scientific
comparisons. Then queue the complete paired blocks that fit the measured rate:

```sh
python studies/lstm-adaptive-robustness/submit_jobs.py \
  --study-root "$STUDY_ROOT" --expected-revision "$REVISION" --phase production
```

Each response is followed by recorded CPU cross-play against all four Blue
checkpoints. Submission records identify which four- or eight-run design was
actually queued; the protocol continues to list unstarted trials. Optional Red 149
trials always stop at 208 updates; `manifest.trial_updates` records each trial
horizon. They follow all eight main searches and contribute only an additional
primary-budget replication, separately reported from the two-seed result. The
8 a.m. Eastern cutoff applies to GPU training; the full CPU cross-play matrix
may finish later.

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

## Complete interrupted cross-play

The original evaluator jobs cannot finish the full panel within their launcher
deadline. Use `recover_evaluations.py` from the current study branch to retain a
recovery plan outside the pinned runtime checkout, then submit the retained copy:

```sh
python recover_evaluations.py plan --study-root "$STUDY_ROOT" \
  --source-root "$PINNED_SOURCE" --expected-revision "$REVISION" \
  --recovery-root "$RECOVERY_ROOT"
python "$RECOVERY_ROOT/recover_evaluations.py" submit \
  --recovery-root "$RECOVERY_ROOT" --dry-run
python "$RECOVERY_ROOT/recover_evaluations.py" submit \
  --recovery-root "$RECOVERY_ROOT"
```

Recovery waits for each original evaluator to finish and rechecks its retained
results before evaluating a missing cell. A receipt alias selects the exact
original training attempt; it does not represent new training. Each incomplete
cell repeats its full original panel in a fresh destination, preserving partial
evidence. Primary and final attacker budgets remain separate. The recovery plan,
submission ledger, per-cell selections and results identify all reused and new
evaluations. A failed submission with an uncertain outcome requires reconciliation
with Slurm before another submission.
