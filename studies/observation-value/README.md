# Do observations improve Blue decisions?

Full sampled Blue policies were compared with mask-aware, observation-blind
categorical preferences fitted on separate calibration episodes. All three full
policies gained meaningful return beyond their frozen baselines: every 95%
interval exceeded the prespecified 100-point practical margin.

| Final Blue checkpoint | Full mean | Blind mean | Full minus blind, 95% interval |
| --- | ---: | ---: | ---: |
| LSTM 42, 124.992M steps | -3,491.0 | -4,561.4 | +1,070.3 [984.7, 1,156.0] |
| LSTM 100, 124.992M steps | -3,501.1 | -4,719.2 | +1,218.1 [1,119.2, 1,316.9] |
| MAPPO 43, 150M steps | -2,352.8 | -4,819.8 | +2,467.0 [2,311.9, 2,622.1] |

The 6,144 held-out episodes used the frozen learned-Red panel: seven 29.952M
adaptive Reds and the co-trained MAPPO Red, four reserved layouts, 32 fresh
episode roots and 500 steps per case. Both Blue conditions sampled, and Red
always sampled. One canonical preference vector per agent was distilled across
all eight opponents using 64 calibration episodes per Blue. Masks supply
information to both conditions; the baseline receives no phase or opponent input.
The six historical replay returns matched exactly.

Controlled swaps of complete host evidence changed the top action in 59–65% of
eligible calibration decisions while masks and incoming carries were held fixed.
Both LSTMs also depended strongly on incoming carry. The return comparison
combines phase, static context, evidence and history, so it does not isolate
threat-specific responsiveness or useful memory. No architecture cause has been
established.

Intervals describe fresh-root variation conditional on these opponents, layouts
and fitted baselines. They exclude uncertainty from refitting or new training
runs. Two LSTM training seeds and one MAPPO trajectory do not establish a
population-wide result. This tests one distilled blind policy per checkpoint,
rather than the best return-optimized blind controller.

The frozen inputs, `report/comparison.json`, episode tables, sensitivity records
and validation checks live at
`/data/shared/jaxborg/studies/observation-value-20261109/` on ITM. Tracking uses
`jaxborg/observation-value-20261109` (experiment 5). Evaluation source revision:
`a102881a791eb96f71da542725aa3f5b908bb51d`.
See [reproduction instructions](reproduce.md) and [the analyzer](compare.py).
