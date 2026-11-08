# Fixed-FSM versus archived LSTM runs

Comparison uses the retained new-run snapshot from 2026-11-08T10:59:49.582272-05:00. Higher team return is better.

## FSM competence at matched and extended budgets

| Training | Steps | Seed 42 | Seed 100 | Two-seed mean |
| --- | ---: | ---: | ---: | ---: |
| Archived co-training: single network | 49,968,000 | -454.95 | -485.38 | -470.16 |
| Archived co-training: diverse networks | 49,968,000 | -467.36 | -432.53 | -449.95 |
| New fixed-FSM training: primary | 49,968,000 | -425.78 | -365.88 | -395.83 |
| New fixed-FSM training: extension | 124,992,000 | -323.19 | -355.52 | -339.35 |

At 49.968M steps, the new fixed-FSM runs improve on the corresponding archived diverse-network means by **41.58 and 66.66 points**, averaging **54.12**. The 124.992M extension uses about 2.5 times the budget and must be reported separately.

## Transfer to other-lineage learned Reds

Each Blue seed is compared against Reds from the opposite training-seed lineage. This excludes the archived Blue’s own co-trained Red lineage.

| Blue seed | Red opponent | Archived single, 49.968M | Archived diverse, 49.968M | New FSM, 49.968M | New FSM, 124.992M |
| --- | --- | ---: | ---: | ---: | ---: |
| 42 | red-single-100-final | -719.39 | -714.28 | Pending | -688.42 |
| 42 | red-diverse-100-final | -2543.27 | -2594.38 | -2920.72 | -2899.25 |
| 100 | red-single-42-final | -991.73 | -925.98 | Pending | -1029.78 |
| 100 | red-diverse-42-final | -2488.64 | -2695.78 | Pending | -2997.69 |

Despite its extra budget, the new extended policy scores about 305 points worse for seed 42 and 302 points worse for seed 100 against the opposite-seed diverse-trained Reds than the corresponding old diverse-trained Blues. Results against single-trained Reds are mixed. This supports better FSM competence but does not establish broader robustness.

## Matching and provenance

- Recomputed all 12 selected archived comparison-cell means from their episode rows. Each cell has 64 unique layout/seed pairs.
- Verified the same four topology hashes and episode roots 21000–21015; 500-step episodes with stochastic sampling. Available primary/final new cases match apart from checkpoint identity.
- Archived evaluations used GPU at source 1545af8cb381ab2027c79814353b4b4963c7754c; new evaluations used CPU at 0aef148bbebb45818e71b20e9b5a52be031f93f1. A separate source-diff review found no environment, LSTM rollout, or scripts/eval implementation changes. Differences concern recording, training/resume, and H-MARL import paths; the H-MARL paths are not used by these LSTM comparisons. Hardware/source differences still preclude a claim of bit-identical trajectories.
- Exact archived training source and original stock training-layout bytes are unavailable. Training-opponent exposure differs. These are descriptive two-seed comparisons, not a causal or statistical superiority claim.
- The other primary learned-Red cells were pending recovery at the retained snapshot; completed 125M cases do not substitute for them.

## Exact inputs

The machine-readable companion [archived-lstm-comparison.json](archived-lstm-comparison.json) retains exact values, case checks and source qualifications. Input files:

- `archived_policy_means`: `C:\Users\paul.elliott\Documents\Codex\2026-11-07\task\lstm-check-20261107T170233Z\analysis\policy-means.csv`
  SHA-256: `6777a073589b5376ca078ba8e842d7afa4af7595365cc8db150d8d9f93ddf7f0`
- `archived_episode_rows`: `C:\Users\paul.elliott\Documents\Codex\2026-11-07\task\lstm-check-20261107T170233Z\analysis\episodes.csv`
  SHA-256: `a75cd2b32c84d35b453a82b5cf019e7fb760c90287e12a6bdb929401addc3cf8`
- `new_run_status`: `C:\Users\paul.elliott\Documents\Codex\2026-11-07\task\fsm-result-status-20261108T105949.json`
  SHA-256: `b253ff54f779076f979f691e20b831f164898de964c8f382a09ef99a16fd5212`
