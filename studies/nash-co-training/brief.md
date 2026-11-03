# How well do co-trained cyber policies hold up against new opponents?

**Can a new attacker still improve against Jaxborg’s co-trained defender?** Yes. A new attacker scored about **530 more reward points per simulated episode** against the same defender, and the better of two saved attackers changed when we switched defenders.

## Questions

1. Can an attacker trained against a fixed defender outscore that defender’s original attacker?
2. Does the stronger saved attacker stay stronger when it faces a different defender?

## Method

- **Train a new attacker.** Freeze Blue at the 150M checkpoint and train challenger Reds from scratch against it. Select an attacker from saved and newly trained candidates on separate episodes, then compare it with the original Red on 4,096 matched test episodes.
- **Swap saved opponents.** Take both Red and Blue from the same three checkpoints in one co-training run, at 99.84M, 124.8M, and 150M simulation steps. Play all nine Red–Blue pairings and run each pairing through the same 4,096 simulation setups.

## Results

### A new attacker scored 530 points more against the same defender

Both attackers faced Blue frozen at the 150M checkpoint. Its original co-trained Red averaged **2,531** points per episode. The challenger Red, trained from scratch against that frozen Blue, averaged **3,061**.

![Against frozen Blue at 150M, the co-trained Red scored 2,531 points per episode and the challenger Red trained against that Blue scored 3,061.](brief-figures/fresh-red-gain.png)

### The better saved attacker changed with the defender

We compared the same two saved attackers against each defender. The earlier attacker scored lower against the first two defenders and higher against the final defender.

![The earlier attacker lost by about 29 and 14 points against the first two defenders and won by about 39 points against the final defender.](brief-figures/saved-red-ranking.png)

## Discussion

The final co-trained pair still gave Red room to improve by switching strategies. The saved matchups show why a score against one opponent tells us about that matchup. Robustness across opponents takes a wider test.

These results come from one co-training run and one simulated network. Each new attacker trained for about 10M simulation steps. How much more an attacker could gain, and whether the pattern holds across other runs and networks, remain open questions.

<details>
<summary>Exact values and study evidence</summary>

| Comparison | Red reward difference | Approximate simultaneous 95% interval |
| --- | ---: | ---: |
| Adapted minus original Red against final Blue | +530.16 | [517.54, 542.79] |
| Earlier minus later Red against Blue at 99.84M steps | −28.70 | [−42.75, −14.64] |
| Earlier minus later Red against Blue at 124.8M steps | −14.09 | [−27.89, −0.29] |
| Earlier minus later Red against Blue at 150M steps | +39.29 | [26.28, 52.30] |

The intervals describe variation across matched test episodes for these policies. The adapted attacker comparison uses its four-contrast correction family. The saved-policy comparisons use their separate 24-contrast family.

See the [study](README.md) and [replay instructions](reproduce.md) for the full design, saved outcomes, and plot commands.

</details>
