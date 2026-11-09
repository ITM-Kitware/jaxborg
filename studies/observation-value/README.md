# Do observations improve Blue decisions?

Do the final Blue policies gain useful competence from observations and memory,
or does a fixed randomized preference over legal actions perform comparably?
This study compares each sampled policy with one observation-blind categorical
preference vector per agent, fitted on separate calibration episodes and frozen
before held-out evaluation. Observation-blindness remains a hypothesis.

Use the frozen learned-Red panel: two independently trained LSTM seeds at
124.992M steps and one MAPPO trajectory at 150M, against seven 29.952M adaptive
Reds and the co-trained MAPPO Red. Fit across all eight opponents, then compare
on 32 fresh episode roots across four reserved layouts, for 500 steps per case.
Both conditions sample; Red remains unchanged. Masks supply information to both
conditions. The baseline has no phase or opponent input. This tests value beyond
that particular distilled baseline, rather than the best possible blind policy.

The practical margin is 100 return points per episode. Report full-policy minus
blind return separately for each checkpoint, with a 95% interval over matched
root means conditional on this opponent/layout panel. An interval entirely above
100 supports useful responsiveness; one entirely within ±100 supports comparable
performance on this panel. Other results remain inconclusive about equivalence.
Repeated episodes and related Reds do not add independent Blue training runs.

Sparse actual decision inputs support controlled spatial evidence swaps and
separate observation/carry sensitivity measurements. These measure dependence;
closed-loop returns determine its usefulness. There is no established architecture
cause. Historical cases must replay exactly before new comparisons.

The frozen protocol, selections, calibration fit, measurements and report live at
`/data/shared/jaxborg/studies/observation-value-20261109/`. The dedicated tracking
experiment is `jaxborg/observation-value-20261109`. [prepare.py](prepare.py)
freezes inputs; [compare.py](compare.py) runs and analyzes the comparison.
