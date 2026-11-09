# LSTM robustness to adaptive attackers

Does the extra fixed-FSM training from 49.968M to 124.992M transitions make Blue
harder for a newly trained attacker to exploit? The first two attacker searches
completed; the cross-play matrix is still incomplete.
[CPU recovery](checks/20261109-cpu-recovery/four-plus-three.json) is queued to finish
missing cells while preserving completed results and partial evidence. Direct
matchups take priority within each completed training batch.

Train fresh Red policies against both checkpoint ages from Blue seeds **42 and
100** in the [competence study](../lstm-fsm-competence/README.md). Red seeds **17
and 83** receive equal budgets: **9.984M transitions** for the primary comparison
and **29.952M** for a separately reported extension. Verify identical initial Red
weights within each checkpoint-age pair. Keep the original 100 stock layouts,
450-feature Blue observations, normal Green, actions and rewards. Red uses the
existing shared feedforward architecture, 256 × 2 with tanh.

Blue 42 runs first because it showed the larger FSM improvement; this selection
was informed by the earlier results. Counterbalance checkpoint-age order across
Red seeds. Evaluate each selected Red against all four Blues on the same 64
reserved cases. Four optional Red-149 searches at 9.984M add primary-budget
replication if time remains after the main runs.

Compare own-target returns and the full cross-play matrix. Improved resistance
would support better defense under this attacker-search budget; transferring
attacks would expose shared weaknesses. This measures finite-budget adaptive
vulnerability, not certified exploitability or independent-opponent generalization.

GPU validation confirmed unchanged Blue weights, Red learning, and exact numeric
agreement across independent fresh runs and checkpoint continuation.
[Launch evidence](checks/20261108-launch/launch-status.json),
[protocol](protocol.json), and [reproduction](reproduce.md).

ITM results:
`~/jaxborg-exp/shared/studies/lstm-adaptive-robustness-20261108/production/`.

Recovery selections, submissions and outputs:
`production/recoveries/20261109-cpu-v2/` beneath the dated study directory.
