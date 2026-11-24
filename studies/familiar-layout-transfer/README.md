# Defender transfer from a familiar CAGE 4 layout

Does the one-layout defender's advantage on its training map persist on unseen stock layouts when both defenders face the same opponents? Moving to unseen layouts shifted the reward gap toward the 100-layout Blue in all 16 opponent-by-training-seed comparisons. The result against learned Reds still depended on the Blue training seed.

Each matchup compares **the two Blue policies' average cumulative reward against the same Red**, on the same layouts with matched episode seeds. Higher reward means a better score for Blue. Where a row summarizes several opponents, it gives each opponent equal weight.

| Opponents | On map 0 | Across the ten unseen maps |
|---|---|---|
| **Learned Reds** | The one-map Blue scored higher against both Reds at both training seeds. | The one-map Blue scored higher against both Reds at seed 42. The 100-map Blue scored higher against both at seed 100. |
| **Confidentiality, Integrity, and Availability Reds** | The 100-map Blue scored higher on average across the three opponents. | The 100-map Blue scored higher against each of the three opponents at both seeds. |
| **Aggressive, Stealthy, and Impact Reds** | The one-map Blue scored higher on average across the three opponents. | The 100-map Blue scored higher against each of the three opponents at both seeds. |

We used four IPPO-LSTM checkpoints at 49.92M training steps, regenerated map 0, and ten stock layouts generated from seeds 100–109. For each Blue training seed, both policies faced the same two frozen Reds from the other seed and the same six scripted Reds, with episode seeds 1000–1009 on every layout. The 100-layout defender also scored higher on each targeted CIA metric at both training seeds in both panels. [Numerical results by opponent and training seed](results.md)

Both training pools came from the stock generator. The results show why a single fixed map is a weak baseline for a broader study of generalization. To test whether variation beyond stock helps, compare training with fresh stock layouts against training with one added source of variation. Use the same training budget and frozen training opponents, then evaluate both on stock and modified layouts to measure any gain and its cost on the stock game.

The current result is descriptive and covers one familiar map and two training seeds. Blue and Red co-trained, so changing the layout pool changed both Blue's layout exposure and the Red it learned against.

[Protocol and checkpoint hashes](protocol.json) · [Replay and analysis](reproduce.md)
