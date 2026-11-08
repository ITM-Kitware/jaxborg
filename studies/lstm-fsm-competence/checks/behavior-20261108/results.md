# What changed in Blue behavior

The later checkpoint uses **more Analyse and fewer Restores**, but host selection
and timing matter more than the totals. It repairs some compromises much earlier
and leaves others persistently infected.

All **16 traces reproduce their original returns exactly**: one Blue training
seed, two checkpoints, four layouts and two opponents. Analyse increases in all
eight paired cases; Restore decreases in seven. Among genuine decisions, excluding
forced waits, Restore falls from 3.09% to 1.55% against FSM and from 3.47% to 2.54%
against learned Red.

| Case | 49.968M return | 124.992M return | Behavior |
| --- | ---: | ---: | --- |
| FSM, layout 201 | -545 | -331 | Later Blue restores host 75 early; earlier Blue repeatedly Removes and never restores it. Earlier restoration of hosts 75 and 64 accompanies 136 points less local-work penalty on those hosts. |
| FSM, layout 200 | -217 | -925 | Earlier Blue restores host 36. Later Blue never Analyses or restores it; repeated Removes leave a privileged session. That host accounts for 477 points of local-work penalty. |
| Learned Red, layout 200 | -3155 | -2758 | Later Blue Analyses host 73, obtains file/IOC evidence, then restores it at episode ticks 49-53; earlier Blue waits until 212-216. Despite fewer team Restores, some targets receive earlier recovery. |

The improved learned-Red case still contains a failure: later Blue repeatedly
Removes host 31 without clearing its privileged session. Its eventual disappearance
comes from Red Withdraw, so session disappearance alone does not prove successful
Blue remediation.

## What the policy can observe

On FSM host 36, remembered process evidence disappears after Remove even though
the privileged session remains. Later Blue never Analyses that host to refresh
evidence. Blue cannot directly see the hidden session state; Remove does not reset
its LSTM memory. A [two-case regression](../../../../tests/evaluation/test_remove_observation_parity.py)
executes real CybORG Remove on root/user sessions and confirms exact agreement
with JAX across all 450 observation values. This behavior is not a newly identified
port mismatch.

Repeated Impact penalties also need care: [stock CC4 reward handling](../../../../src/jaxborg/actions/red_impact.py) can penalize failed attempts
against an already-stopped service. Counts of penalized attempts are not counts
of new outages. Across the learned-Red cases, later Blue improves mean return by
84.75 through local-work gains and lower restoration costs; Impact penalties
worsen. Traffic-access penalties contribute only 20 points across all 16 episodes.

## Interpretation and evidence

These are small, exploratory cases with diverging state histories. They do not
isolate action preferences on identical observations or diagnose GAE. The four
FSM cases worsen on average even though the full 64-case comparison improves.
A useful next probe is a same-state Remove-versus-Restore intervention at an
eligible decision, followed by common future randomness.

[Panel numbers](panel-summary.json), [FSM event evidence](paired-fsm-compact-evidence.json),
[learned-Red event evidence](paired-learned-layout200-compact-evidence.json) and
[exact sources and validation](analysis-record.json). Full traces remain in the
ITM output directory linked from the [check README](README.md).
