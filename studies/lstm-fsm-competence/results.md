# Overnight LSTM/FSM results

Status at **November 8, 2026, 10:59 a.m. Eastern**, from the
[retained snapshot](checks/20261108-overnight-results/status.json). Both training
runs completed the declared **124,992,000 transitions**. Final model, sidecar and
full-state checkpoint hashes were verified for both seeds.

| Seed | Training job | Completed, Eastern | Final evaluation job | Completed, Eastern |
| --- | --- | --- | --- | --- |
| 42 | 4128 | Nov 8, 07:33:35 | 4132 | Nov 8, 09:10:17 |
| 100 | 4129 | Nov 8, 07:16:42 | 4135 | Nov 8, 08:52:44 |

The fixed-FSM means below use the same four reserved stock layouts, 16 episode
roots per layout, 500-step horizon and stochastic policy sampling: **64 matched
episodes per checkpoint and training seed**. Higher stock team return is better.

| Training seed | Primary: 49.968M steps | Final: 124.992M steps | Final minus primary |
| --- | ---: | ---: | ---: |
| 42 | -425.78125 | -323.1875 | +102.59375 |
| 100 | -365.875 | -355.515625 | +10.359375 |

Both recorded FSM means improved during the extension, with a larger change for
seed 42. The prespecified primary checkpoint remains 49.968M; the final result
does not replace it. Two training seeds and these fixed evaluation cases do not
establish a population-level improvement or a state-of-the-art result.

## Incomplete primary opponent panel

The primary opponent panel remains incomplete after evaluation memory failures.
At **November 8, 12:45 p.m. Eastern**, recovery had completed two of the six missing
cells; four remained. The newly recovered seed-42 comparisons use the same
64 cases at both checkpoints:

| Opponent | Primary: 49.968M | Final: 124.992M | Final minus primary |
| --- | ---: | ---: | ---: |
| Diverse-trained Red, seed 42 | -2949.34 | -2842.72 | +106.63 |
| Single-network Red, seed 100 | -786.14 | -688.42 | +97.72 |

Extra FSM training improved these learned-opponent scores too. A complete
two-seed primary comparison still needs the remaining cells; final evaluations
cannot substitute for them. See the [verified recovery snapshot](checks/20261108-overnight-results/recovery-status-20261108T174549Z.json)
and [submission record](checks/20261108-overnight-results/recovery-submission.json).

The saved learned Reds come from a related archive, not a verified independent
holdout population. This FSM-trained baseline is a competence diagnostic;
comparison with historical co-training remains descriptive because its exact
training source is unknown. Higher return alone does not establish mission
alignment, and this comparison does not isolate the effect of nonstationarity.

## Evidence

- Execution root: `~/jaxborg-exp/shared/studies/lstm-fsm-competence-20261107-v3/`.
- Source: `0aef148bbebb45818e71b20e9b5a52be031f93f1`; scientific protocol version 2.
- Snapshot: [status.json](checks/20261108-overnight-results/status.json), copied
  byte-for-byte (110,162 bytes), SHA-256
  `b253ff54f779076f979f691e20b831f164898de964c8f382a09ef99a16fd5212`.
- Primary means: snapshot `primary_receipts`, paths
  `evaluations/fsm-seed{42,100}/primary/receipt-49968000-fsm.json`.
- Final means: snapshot `evaluations`, paths
  `evaluations/fsm-seed{42,100}/final/summary.json`, selecting checkpoint
  `124992000` and opponent `fsm`.

The snapshot retains scheduler status, training budgets and checkpoint identities,
hash-verification results, completed evaluation selections and partial receipts.
Full episode evidence and checkpoints remain in the ITM execution root.

The recovery helper passed nine focused tests and an independent review. The
submission record retains the exact deployed helper bytes and SHA-256
`2577c187f31b6f1687d1f8e4222865c451c324482d330b4db7f5772419af6789`.
The adjacent Python copy has only repository-lint whitespace normalization; use
the retained deployed bytes when reproducing this exact hash-pinned recovery.

See the [comparison with archived LSTMs](checks/20261108-overnight-results/archived-lstm-comparison.md) for matched-budget FSM results and transfer to other-lineage learned Reds.
