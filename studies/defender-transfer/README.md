# Defender transfer across networks and opponents

Do archived co-trained LSTM defenders retain useful performance when the network
or attacker changes?

The saved LSTMs learned: FSM return improved by about **647-828 points** between
0.96M and 49.968M training steps across the four policies. There was **no consistent
diversity advantage** against common opposite-seed learned Reds. A separate
matched screen placed all four LSTMs below the imported H-MARL systems. These
findings motivate checking current LSTM competence before another diversity test.

## Comparison

Compare early and final checkpoints from single-network and diverse-network
co-training, with two training seeds per condition. Policies receive their
original 450-feature version-2 observations. Evaluate common FSM and saved
learned opponents across reconstructed stock layouts, using stochastic 500-step
episodes and 16 episode roots per layout. Stock rewards and Green traffic remain
unchanged. A separate CIA panel constrains server counts and measures role-host
impact proxies, not exfiltration or corruption.

The [learning check](checks/lstm-learning-20261107/README.md) contains the selected
69 evaluation cells and per-policy results. The [matched system screen](benchmark.md)
compares the four final LSTMs with H-MARL Expert and Meta on 16 common scenarios.

## What this establishes

Original training-source revisions and stock layout bytes are unavailable, so
reserved generator seeds do not prove genuinely unseen networks. The learned
Reds are related archive policies, not an independent holdout population. The
learning check lacks 27 early learned-opponent cells; its final-stage comparison
is complete. Two training seeds per condition support descriptive comparisons.
H-MARL differs in policy design and training history, so its advantage does not
isolate an algorithm effect or establish published SOTA performance.

## Results and reproduction

On ITM, outputs are under
`/home/local/KHQ/paul.elliott/jaxborg-exp/shared/studies/`:

- `defender-transfer-20261107-v4/`: inputs, captured evaluations and
  `reviews/lstm-check-20261107T170233Z/analysis/`.
- `defender-benchmark-20261107/analysis-full/`: matched system results.

The linked checks retain exact selections and source identities. See
[reproduction commands](reproduce.md) to reconstruct results or run new cases.
The evaluation workflow builds on [PR #55](https://github.com/ITM-Kitware/jaxborg/pull/55).
