# LSTM robustness to adaptive attackers

Does the extra fixed-FSM training from 49.968M to 124.992M transitions make a
Blue policy harder for a newly trained attacker to exploit? Training and
validation are pending; better FSM reward alone does not answer this question.

Train fresh Red policies against frozen early and late checkpoints from the
[competence study](../lstm-fsm-competence/README.md). Each attacker gets
**9.984M transitions**, the same 100 stock training layouts, and the existing
shared feedforward architecture (256 × 2, tanh). Blue observations, normal Green
traffic, actions and rewards stay unchanged. Red seeds **17 and 83** are paired
across checkpoint ages, with initial parameter identity verified from the
retained bundles.

The first four runs target Blue seed **42**, chosen after it showed the clearest
FSM improvement. A second complete four-run block targets Blue seed **100** if
measured throughput permits. Within each block, run checkpoint ages in opposite
orders for the two Red seeds. Evaluate every final Red against all four Blue
checkpoints on the same reserved 64 cases. Select checkpoints by budget, never
by evaluation score.

Compare own-target returns and the full cross-play matrix. Stronger resistance
by the later Blue would support improved competence under this attack-search
budget. Transferring attacks would reveal shared weaknesses. These are adaptive
stress tests, not certified exploitability or an independent-opponent holdout;
two Blue lineages and two attacker seeds limit generality. Incomplete pairs do
not replace the primary comparison.

[Protocol](protocol.json), [recipe](recipe.yaml), and [reproduction](reproduce.md).
ITM results: `~/jaxborg-exp/shared/studies/lstm-adaptive-robustness-20261108/`.
