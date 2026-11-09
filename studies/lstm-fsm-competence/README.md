# LSTM competence against fixed FSM Red

Can a separate actor/critic LSTM learn effective defense against the standard
scripted attacker in the ordinary stock game? At the matched 49.968M-step budget,
the two fixed-FSM runs scored better against FSM than the archived co-trained
LSTMs. Longer training improved FSM scores and some learned-opponent scores,
but the defenders still struggled against diverse-trained Reds.

## Comparison

Train Blue against stock FSM Red with independent seeds **42 and 100**, using the
same 100 stock-network snapshots (generator seeds 0-99). Keep normal Green
traffic, rewards, Blue actions and enhanced observations. The
[recipe](recipe.yaml) specifies separate actor/critic LSTM trunks of width 256.

The prespecified primary checkpoint is **49.968M environment transitions**. The
extension ends at **124.992M**, about 2.5 times that budget; it is reported
separately and does not replace the primary result. Checkpoints were selected by
training step rather than evaluation score.

The evaluation panel uses four reserved stock layouts (seeds 200-203), with 16
episode roots per layout, a 500-step horizon and stochastic policy sampling. It
specifies the same 64 cases against FSM and each of four saved learned Reds.

| Training seed | Primary FSM return | Extended FSM return |
| --- | ---: | ---: |
| 42 | -425.78 | -323.19 |
| 100 | -365.88 | -355.52 |

Higher return is better. See [results](results.md) and the
[archived-LSTM comparison](checks/20261108-overnight-results/archived-lstm-comparison.md)
for exact values and opponent-specific outcomes.

## Limits

Training and final evaluations are complete. At **November 8, 12:45 p.m.
Eastern**, four primary learned-Red cells still needed recovery after evaluation
failures. Final results cannot substitute for those cells.

Two training seeds and a related archived opponent panel do not establish general
robustness. The archived training source and original training-layout bytes are
unavailable; opponent exposure also differs. Comparisons with co-training are
descriptive, not causal.

## Evidence and reproduction

ITM output root:

```text
~/jaxborg-exp/shared/studies/lstm-fsm-competence-20261107-v3/
```

[Behavior findings](checks/behavior-20261108/results.md), [protocol](protocol.json), [recorded snapshot](checks/20261108-overnight-results/status.json),
and [reproduction commands](reproduce.md). Producing source:
`0aef148bbebb45818e71b20e9b5a52be031f93f1`.

Checked-in evidence is a public summary with personal paths and scheduler
inventories removed. Each record retains the original snapshot hash; exact
operational records remain outside the public repository.
