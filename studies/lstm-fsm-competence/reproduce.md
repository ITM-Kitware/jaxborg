# Reproduce the LSTM/FSM study

These are the recorded execution commands. Use a clean checkout of
`0aef148bbebb45818e71b20e9b5a52be031f93f1` with the
[project environment](../../docs/setup.md). The retained input roots below must
be available. Choose a fresh output root. For a later rerun, prepare a new
protocol revision: the pinned launcher uses the original experiment dates.

```bash
OLD="$HOME/jaxborg-exp/shared/studies/lstm-fsm-competence-20261107"
ROOT="$HOME/jaxborg-exp/shared/studies/lstm-fsm-competence-reproduction"
TRANSFER_MANIFEST="$HOME/jaxborg-exp/shared/studies/defender-transfer-20261107-v4/inputs/manifest.json"
SHA=0aef148bbebb45818e71b20e9b5a52be031f93f1
JAX_PLATFORMS=cpu .venv/bin/python studies/lstm-fsm-competence/prepare.py prepare \
  --study-root "$ROOT" --expected-revision "$SHA" --reuse-inputs "$OLD"
bash studies/lstm-fsm-competence/launch.sh submit "$ROOT" smoke 42
```

After the smoke completes, check exact continuation and checkpoint evaluation:

```bash
bash studies/lstm-fsm-competence/launch.sh submit "$ROOT" smoke-resume 42
bash studies/lstm-fsm-competence/eval_launch.sh submit \
  "$ROOT" smoke-seed42 smoke "$TRANSFER_MANIFEST"
```

After both checks complete:

```bash
JAX_PLATFORMS=cpu .venv/bin/python studies/lstm-fsm-competence/validate.py \
  --study-root "$ROOT" --expected-revision "$SHA" --require-resume --require-cpu-eval
bash studies/lstm-fsm-competence/launch.sh submit "$ROOT" full 42 18
bash studies/lstm-fsm-competence/launch.sh submit "$ROOT" full 100 18
```

Evaluate a stage after its required checkpoint is retained:

```bash
bash studies/lstm-fsm-competence/eval_launch.sh submit \
  "$ROOT" fsm-seed42 primary "$TRANSFER_MANIFEST"
```

Repeat for seed 100 and stages `interim` and `final`. The stages select 9.6M,
49.968M and 124.992M transitions. Identical initial reference evaluations are
explicitly reused. Select completed attempts and checkpoint identities from the
saved receipts, without choosing by score.

Full-state continuation uses the maintained trainer's `--resume-state` with the
same recipe, seed and source, a new output tag, and `--checkpoint-update 1041`.
See the [recording interface](../../docs/joint-execution-records.md). The
[result note](results.md) links the retained selections and partial-evaluation
recovery evidence for the original execution.
