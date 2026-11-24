# Defender transfer from a familiar CAGE 4 layout

Does the one-layout defender's advantage on its own training map persist on unseen stock layouts when both defenders face the same opponents? Across the 16 opponent-by-training-seed comparisons, the reward gap shifted toward the 100-layout defender on held-out layouts. The learned-Red result still depended on the Blue training seed. The 100-layout defender led against each of the six scripted Reds on the held-out panel at both training seeds.

The table shows the reward difference, calculated as 100-layout Blue minus one-layout Blue. Opponents are equally weighted within each family.

| Opponent family | Blue training seed | Map 0 | Held-out layouts |
|---|---:|---:|---:|
| Learned | 42 | −386.50 | −72.71 |
| Learned | 100 | −402.85 | +121.16 |
| CIA-focused | 42 | +47.33 | +122.58 |
| CIA-focused | 100 | +55.73 | +225.02 |
| H-MARL scripted | 42 | −36.13 | +53.45 |
| H-MARL scripted | 100 | −39.97 | +181.50 |

We used the same four IPPO-LSTM checkpoints at 49.92M training steps on regenerated map 0 and ten stock layouts generated from seeds 100–109. Each checkpoint faced the same two other-seed frozen Reds and six scripted variants in both panels, with episode seeds 1000–1009 on each layout. The 100-layout defender also led on each targeted CIA score at both training seeds in both panels. [Per-opponent results](results.md)

This is a descriptive comparison with one familiar map and two training seeds. Both co-training conditions encountered map 0 at different frequencies. Changing the layout pool also changed the learned training opponents, so the comparison cannot isolate Blue's layout exposure. Both training pools came from the stock generator.

[Protocol and checkpoint hashes](protocol.json) · [Replay and analysis](reproduce.md)
