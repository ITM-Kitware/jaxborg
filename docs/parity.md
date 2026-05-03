# Parity tests

Index of the differential tests that verify jaxborg matches CybORG. Two axes:

1. **Env-mechanics parity** — given the same actions and the same RNG, do both backends produce the same per-step state and reward?
2. **Training-loop parity** — given the same rollout data and the same initial params, do the JAX IPPO and CybORG CleanRL PPO update bodies compute the same gradient and the same post-Adam params?

All tests live under `tests/differential/`. Run with `uv run pytest tests/differential/` (fast suite) or `uv run pytest -m slow tests/differential/` (slow regressions). Per-test wall time and tolerance are listed below.

## Env-mechanics parity

| test | what it asserts | tolerance | mark |
|---|---|---|---|
| `test_matched_rng_reward_parity.py` | Under `sync_green_rng=True`, JAX consumes CybORG's captured `np_random` stream → byte-equivalent per-step reward | exact | slow (5 seeds × 200 steps) |
| `test_red_policy_parity.py` | `FiniteStateRedAgent.get_action` byte-equivalent under matched RNG | exact (35,869 picks, 0 mismatches at last run) | slow |
| `test_fsm_parity.py` | Multi-step FSM red transitions match between backends | state hash | fast |
| `test_blue_policy_fuzz.py` | Per-action blue parity (Decoy, Restore, Block, …) under fixed traces | state hash | fast |
| `test_random_sync.py` | Strict-sync mode handles tricky CybORG action traces (decoy detection, exploit pending, etc.) | trace replay | fast |
| `test_reward_cc4_contract.py` | Full CC4 reward contract (BlueRewardMachine + action_cost) per-step parity | exact | fast |

These pass cleanly under matched RNG. **Under independent RNG**, an A1 LWF asymmetry persists (~+196 pts JAX-side on sleep blue, ~+430 pts on trained blue, due to divergent green host-selection); see `plans/jax/cc4/parity.md` for the 12-follow-up investigation. This is not a parity defect — it's two independent PRNG streams cascading through 500 steps of CC4's one-sided LWF reward, and **cancels in cross-policy comparisons on the same env**.

## Training-loop parity

| test | what it asserts | tolerance | mark |
|---|---|---|---|
| `test_adam_step_parity.py` | `torch.optim.Adam` ≡ `optax.adam` on identical params + grads + (m, v, step) state | ≤1e-6 per step / ≤1e-5 over 200 steps | fast (17 cases × {fresh, primed t∈{0,1,10,100}}) |
| `test_ppo_update_parity.py` | JAX `_loss_fn` + `optax.adam` ≡ torch loss + `torch.optim.Adam` on a synthetic IID Gaussian minibatch (N=128) | grads ≤1e-5 / post-Adam params ≤1e-4 | slow (1 case) + fast (forward + ddof audit) |
| `test_replay_trajectory_parity.py` | Same as above but the minibatch is drawn from a real `FsmRedCC4Env` rollout | grads ≤1e-4 / post-Adam params ≤1e-4 | slow |

These collectively prove the PPO update math is bit-equivalent at single-update granularity, both for synthetic IID and for real-distribution data — so any cross-backend gap at training time localizes to **rollout compounding** (different RNG streams driving different per-step trajectories), not loss math, optimizer math, or normalization scope.

One concrete `src/` fix landed via this work: `mb_adv.std()` in `scripts/train/ppo_cleanrl_cyborg.py` defaulted to torch's `unbiased=True` (ddof=1, sample variance) while JAX's `jnp.std` defaults to ddof=0 (population). The per-minibatch bias is `sqrt(N/(N-1))` ≈ 0.007% at N=7,500, same direction every minibatch, compounded ~13,300× per 5M-step training run. Fixed at commit `86cb561` to `mb_adv.std(unbiased=False)`; regression-tested by the audit in `test_advantage_normalization_ddof_audit`.

## TOST equivalence anchors

The headline equivalence claims in the project README:

| layer | gap | n | TOST verdict | source |
|---|---:|---:|---|---|
| Env parity (same trained policy, two backends, pure-mode topology, n=100/seed × 3 seeds) | +109 ± 30 | 300 | EQUIVALENT at Δ=±284 (p=2.8e-7) | 2026-04-21 final TOST |
| Matched training (3 JAX-trained vs 3 CybORG-trained policies, paired CybORG-env eval, n=100/seed × 3 seeds) | +5.4 ± 58 | 300 | EQUIVALENT at Δ=±200 (p=4×10⁻⁴) and Δ=±284 (p<10⁻⁶) | 2026-04-25 v2 |

Δ anchors:
- **Δ=±84** = 2σ across same-backend seed means (noise floor)
- **Δ=±200** = the prompt-default pragmatic anchor (Karten 2026 Table 3 typical)
- **Δ=±284** = 5% of the learnable signal span (sleep → trained)

## Where the longer narrative lives

The dated investigation log — 12 follow-ups of A1, the matched-training v1/v2 entries, the per-component breakdowns, and the Final Synthesis — is in (outside this repo):

- `plans/jax/cc4/parity.md` — env-mechanics parity, A1 LWF investigation, training-loop parity Tier 1 + 2.5
- `plans/jax/cc4/training-results.md` — matched-training comparisons, v1/v2 retest, final policy audit
