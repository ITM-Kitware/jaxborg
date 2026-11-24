# Matched evaluation results

Positive gaps favor the 100-layout Blue. Each panel weights its layouts equally.

| Blue training seed | Frozen opponent | Map 0 reward gap | Held-out reward gap | Gap change |
|---|---|---:|---:|---:|
| 42 | learned-one | -646.40 | -58.61 | +587.79 |
| 42 | learned-hundred | -126.60 | -86.81 | +39.79 |
| 42 | cia_c | +74.60 | +128.37 | +53.77 |
| 42 | cia_i | +14.10 | +117.54 | +103.44 |
| 42 | cia_a | +53.30 | +121.82 | +68.52 |
| 42 | aggressive | -66.00 | +18.24 | +84.24 |
| 42 | stealthy | -53.00 | +75.55 | +128.55 |
| 42 | impact | +10.60 | +66.55 | +55.95 |
| 100 | learned-one | -506.00 | +150.18 | +656.18 |
| 100 | learned-hundred | -299.70 | +92.13 | +391.83 |
| 100 | cia_c | +88.90 | +226.20 | +137.30 |
| 100 | cia_i | -20.80 | +228.85 | +249.65 |
| 100 | cia_a | +99.10 | +220.00 | +120.90 |
| 100 | aggressive | +9.20 | +192.94 | +183.74 |
| 100 | stealthy | -79.50 | +147.17 | +226.67 |
| 100 | impact | -49.60 | +204.38 | +253.98 |

| Blue training seed | Opponent family | Map 0 reward gap | Held-out reward gap | Gap change |
|---|---|---:|---:|---:|
| 42 | Learned | -386.50 | -72.71 | +313.79 |
| 42 | CIA-focused | +47.33 | +122.58 | +75.24 |
| 42 | H-MARL scripted | -36.13 | +53.45 | +89.58 |
| 100 | Learned | -402.85 | +121.16 | +524.00 |
| 100 | CIA-focused | +55.73 | +225.02 | +169.28 |
| 100 | H-MARL scripted | -39.97 | +181.50 | +221.46 |

| Blue training seed | Opponent | Map 0 targeted CIA gap | Held-out targeted CIA gap |
|---|---|---:|---:|
| 42 | cia_c | +5.886 | +4.143 |
| 42 | cia_i | +4.392 | +3.203 |
| 42 | cia_a | +4.382 | +4.869 |
| 100 | cia_c | +2.796 | +3.948 |
| 100 | cia_i | +3.036 | +3.613 |
| 100 | cia_a | +4.776 | +4.197 |

Held-out layout seeds: 100, 101, 102, 103, 104, 105, 106, 107, 108, 109.
