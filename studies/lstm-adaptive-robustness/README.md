# LSTM robustness to adaptive attackers

Does the extra fixed-FSM training from 49.968M to 124.992M transitions make Blue
harder for a newly trained attacker to exploit?

The Blue LSTM policies were trained against **fixed FSM Red** at both checkpoint
ages. We then freeze each Blue checkpoint and train fresh learned Red attackers
against it. **Blue does not update during attacker training.** This tests whether
more training against the scripted attacker transfers to resistance against
learned attackers.

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

Keep two comparisons separate:

- **Own-target responses:** compare each Blue checkpoint against the Red trained
  specifically to attack it. Changing checkpoint age changes both Blue and its
  attacker. This asks what weaknesses a fresh attacker can discover in each
  defender under the declared search budget.
- **Fixed-attacker cross-play:** compare the Blue checkpoints against the same
  saved Red policies. Within each attacker budget, Blue changes while the
  evaluated attackers stay fixed. This asks which defender handles that common
  set of attacks better.

These comparisons can give different answers. Report each separately when judging
whether extra FSM training helped. They measure finite-budget adaptive
vulnerability, not certified exploitability or independent-opponent generalization.

GPU validation confirmed unchanged Blue weights, Red learning, and exact numeric
agreement across independent fresh runs and checkpoint continuation.
[Launch evidence](checks/20261108-launch/launch-status.json),
[protocol](protocol.json), and [reproduction](reproduce.md).

ITM results:
`~/jaxborg-exp/shared/studies/lstm-adaptive-robustness-20261108/production/`.

The checked-in [CPU recovery snapshot](checks/20261109-cpu-recovery/four-plus-three.json)
records an earlier partial matrix and missing-cell recovery. It is a dated
execution record, not current job status.

Recovery selections, submissions and outputs:
`production/recoveries/20261109-cpu-v2/` beneath the dated study directory.
