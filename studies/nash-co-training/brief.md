# How do challenger Red and co-trained Red compare?

**Can challenger Red outscore co-trained Red against the same Blue?** Against frozen Blue at 150M, challenger Red scored about **530 more points per episode**. In a separate test, which co-trained Red scored higher depended on the Blue checkpoint.

## Questions

1. Can challenger Red outscore co-trained Red against the same frozen Blue?
2. Does the same co-trained Red score higher across different Blue checkpoints?

## Method

- **Train challenger Reds.** Freeze co-trained Blue at the 150M checkpoint and train challenger Reds from scratch against it. Each challenger completed **9,984,000 simulation steps** of a planned 10 million. Select a Red from saved co-trained and freshly trained candidates on separate episodes. Compare the selected challenger Red with co-trained Red at 150M on 4,096 matched test episodes.
- **Swap saved opponents.** Take both Red and Blue from the same three checkpoints in one co-training run, at 99.84M, 124.8M, and 150M simulation steps. Play all nine Red–Blue pairings and run each pairing through the same 4,096 simulation setups.

## Results

### Challenger Red scored 530 points more against frozen Blue

Both Reds faced Blue frozen at the 150M checkpoint. Co-trained Red from that checkpoint averaged **2,531** points per episode. Challenger Red averaged **3,061**.

![Against frozen Blue at 150M, the co-trained Red scored 2,531 points per episode and the challenger Red trained against that Blue scored 3,061.](brief-figures/fresh-red-gain.png)

### Which co-trained Red scored higher depended on Blue

We compared co-trained Red at 99.84M and 150M against each of the three saved Blues. Red 99.84M scored lower against the first two Blue checkpoints and higher against Blue 150M.

![Red 99.84M scored about 29 and 14 points less than Red 150M against the first two Blues, and about 39 points more against Blue 150M.](brief-figures/saved-red-ranking.png)

### Full nine-matchup matrix

Rows show co-trained Blue checkpoints. Columns show co-trained Red checkpoints. The colored cells show average **Blue** reward. Higher values favor Blue. The right panel shows the same Red comparison as the simpler bar chart above.

![The complete three-by-three CPU payoff matrix and paired Red comparison across three fixed Blue opponents.](brief-figures/cpu-saved-policy-crossplay.png)

## Discussion

At 150M, Red could improve by switching policies while Blue stayed fixed. The cross-play results show why judging a Red policy across opponents takes more than one Blue checkpoint.

The 150M study also tested earlier saved Blues against frozen Red150. Training a fresh Blue challenger against Red150 remains open for this pair.

These results come from one co-training run and one simulated network. How much more a challenger Red could gain, and whether the pattern holds across other runs and networks, remain open questions.

<details>
<summary>Exact values and study evidence</summary>

| Comparison | Red reward difference | Approximate simultaneous 95% interval |
| --- | ---: | ---: |
| Challenger minus co-trained Red against frozen Blue 150M | +530.16 | [517.54, 542.79] |
| Red 99.84M minus Red 150M against Blue at 99.84M steps | −28.70 | [−42.75, −14.64] |
| Red 99.84M minus Red 150M against Blue at 124.8M steps | −14.09 | [−27.89, −0.29] |
| Red 99.84M minus Red 150M against Blue at 150M steps | +39.29 | [26.28, 52.30] |

The intervals describe variation across matched test episodes for these policies. The challenger Red comparison uses its four-contrast correction family. The saved-policy comparisons use their separate 24-contrast family.

See the [study](README.md) and [replay instructions](reproduce.md) for the full design, saved outcomes, and plot commands.

</details>
