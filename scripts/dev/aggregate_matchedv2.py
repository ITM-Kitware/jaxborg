"""Aggregate Matched-Training v2 replication eval rows + training metrics.

Inputs: per-seed eval JSONLs from `eval_recipe.py` and per-seed training
metrics.jsonl files. Outputs a markdown table matching the layout of
`training-results.md` §2026-04-25 Matched-Training v2, plus a PASS/FAIL
verdict against the locked-in target band.

Usage:
    uv run python scripts/dev/aggregate_matchedv2.py \
        --jax-evals jaxborg-exp/eval/matchedv2_jax_s{0,1,2}.jsonl \
        --cyborg-evals jaxborg-exp/eval/matchedv2_cyborg_s{0,1,2}.jsonl \
        --jax-metrics jaxborg-exp/ippo_jax/default_seed{0,1,2}/metrics.jsonl \
        --cyborg-metrics jaxborg-exp/ippo_cyborg/default_seed{0,1,2}/metrics.jsonl

Targets (from training-results.md §2026-04-25):
    JAX train ep_rew @ 3M:      -1998 ± 118 (mean ± σ across 3 seeds)
    CybORG train ep_rew @ 3M:   -1854 ± 46
    JAX-trained → CybORG eval (pooled n=300):   -1853
    CybORG-trained → CybORG eval (pooled n=300): -1858
    Cross-policy gap:           +5.4 ± 58, EQUIVALENT @ Δ=200/284
    Final-50-update entropy:    ~3.84
    Final-50-update approx_kl:  ~0.005
    Final-50-update exp_var:    ~0.61
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

# Locked-in targets from training-results.md §2026-04-25 Matched-Training v2
TARGETS = {
    "jax_train_ep_rew_at_3M": (-1998, 118),  # (mean, σ across seeds)
    "cyborg_train_ep_rew_at_3M": (-1854, 46),
    "jax_eval_pooled": -1853,
    "cyborg_eval_pooled": -1858,
    "cross_policy_gap": (5.4, 58),  # (gap, paired SE on n=300)
    "entropy": 3.84,
    "approx_kl": 0.005,
    "explained_var": 0.61,
}

# Acceptable bands (2σ for noisy quantities; relative for diagnostics)
BANDS = {
    "jax_train_ep_rew_at_3M": ("|x − target| ≤ 2σ", lambda x: abs(x - TARGETS["jax_train_ep_rew_at_3M"][0]) <= 2 * TARGETS["jax_train_ep_rew_at_3M"][1]),
    "cyborg_train_ep_rew_at_3M": ("|x − target| ≤ 2σ", lambda x: abs(x - TARGETS["cyborg_train_ep_rew_at_3M"][0]) <= 2 * TARGETS["cyborg_train_ep_rew_at_3M"][1]),
    "jax_eval_pooled": ("|x − target| ≤ 116", lambda x: abs(x - TARGETS["jax_eval_pooled"]) <= 116),
    "cyborg_eval_pooled": ("|x − target| ≤ 116", lambda x: abs(x - TARGETS["cyborg_eval_pooled"]) <= 116),
    "cross_policy_gap": ("|x − 5.4| ≤ 116", lambda x: abs(x - 5.4) <= 116),
    "entropy": ("within ±2%", lambda x: abs(x - 3.84) / 3.84 <= 0.02),
    "approx_kl": ("within ±20%", lambda x: abs(x - 0.005) / 0.005 <= 0.20),
    "explained_var": ("within ±10%", lambda x: abs(x - 0.61) / 0.61 <= 0.10),
}


def _load_eval(path: Path) -> dict:
    """`eval_recipe.py` writes one JSON object (pretty-printed) per file."""
    return json.loads(Path(path).read_text())


def _load_metrics(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _final_n(rows: list[dict], n: int) -> list[dict]:
    return rows[-n:] if len(rows) > n else rows


def _se_paired(jax_rewards: list[float], cyborg_rewards: list[float]) -> float:
    """Paired SE assuming the two reward lists are aligned (same ep-index)."""
    if len(jax_rewards) != len(cyborg_rewards) or len(jax_rewards) < 2:
        return float("nan")
    diffs = [j - c for j, c in zip(jax_rewards, cyborg_rewards)]
    return stdev(diffs) / math.sqrt(len(diffs))


def _verdict(metric: str, value: float) -> str:
    desc, fn = BANDS[metric]
    return "PASS" if fn(value) else "FAIL"


def _stat(label: str, value: float, target_repr: str, verdict: str) -> str:
    return f"| {label} | {value:>10.2f} | {target_repr:>14} | {verdict:>4} |"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jax-evals", nargs="+", required=True)
    parser.add_argument("--cyborg-evals", nargs="+", required=True)
    parser.add_argument("--jax-metrics", nargs="+", required=True)
    parser.add_argument("--cyborg-metrics", nargs="+", required=True)
    parser.add_argument("--final-updates", type=int, default=50, help="N final updates to average diagnostics over")
    args = parser.parse_args()

    # Per-seed eval: load and pool
    jax_per_seed = [_load_eval(Path(p)) for p in args.jax_evals]
    cyborg_per_seed = [_load_eval(Path(p)) for p in args.cyborg_evals]

    jax_pool = [r for ev in jax_per_seed for r in ev["per_episode"]]
    cyborg_pool = [r for ev in cyborg_per_seed for r in ev["per_episode"]]
    jax_pool_mean = mean(jax_pool)
    cyborg_pool_mean = mean(cyborg_pool)

    # Per-seed paired gap (assumes both sides used same per-ep seeds)
    per_seed_gap = []
    for j_ev, c_ev in zip(jax_per_seed, cyborg_per_seed):
        g = mean(j_ev["per_episode"]) - mean(c_ev["per_episode"])
        per_seed_gap.append(g)

    # Pooled paired gap + paired SE on full n=N×seeds×ep
    cross_policy_gap_pooled = jax_pool_mean - cyborg_pool_mean
    paired_se_pooled = _se_paired(jax_pool, cyborg_pool)

    # σ across seed gaps
    sigma_seed_gap = stdev(per_seed_gap) if len(per_seed_gap) > 1 else float("nan")

    # Training curves: mean across seeds at last sample point + final-50 diagnostics
    jax_metrics_per_seed = [_load_metrics(Path(p)) for p in args.jax_metrics]
    cyborg_metrics_per_seed = [_load_metrics(Path(p)) for p in args.cyborg_metrics]

    def _final_ep_rew(metrics_per_seed: list[list[dict]], key: str) -> tuple[float, float]:
        finals = [_final_n(rows, args.final_updates) for rows in metrics_per_seed]
        per_seed_means = [mean(float(r[key]) for r in rows if key in r) for rows in finals if rows]
        return mean(per_seed_means), stdev(per_seed_means) if len(per_seed_means) > 1 else 0.0

    def _final_diag(metrics_per_seed: list[list[dict]], key: str) -> float:
        finals = [_final_n(rows, args.final_updates) for rows in metrics_per_seed]
        per_seed = [mean(float(r[key]) for r in rows if key in r) for rows in finals if rows]
        return mean(per_seed) if per_seed else float("nan")

    jax_train_mean, jax_train_sigma = _final_ep_rew(jax_metrics_per_seed, "train_episode_reward_mean")
    cyborg_train_mean, cyborg_train_sigma = _final_ep_rew(cyborg_metrics_per_seed, "train_episode_reward_mean")

    jax_entropy = _final_diag(jax_metrics_per_seed, "loss_entropy")
    cyborg_entropy = _final_diag(cyborg_metrics_per_seed, "loss_entropy")
    jax_kl = _final_diag(jax_metrics_per_seed, "ppo_kl_divergence")
    cyborg_kl = _final_diag(cyborg_metrics_per_seed, "ppo_kl_divergence")
    jax_ev = _final_diag(jax_metrics_per_seed, "ppo_explained_variance")
    cyborg_ev = _final_diag(cyborg_metrics_per_seed, "ppo_explained_variance")

    print("=" * 72)
    print("Matched-Training v2 Replication — Aggregated Report")
    print("=" * 72)
    print()
    print(f"## Per-seed eval ({len(jax_per_seed)} seeds × n={len(jax_per_seed[0]['per_episode'])} ep)")
    print()
    print("| seed | JAX-trained → CybORG | CybORG-trained → CybORG | gap (J−C) |")
    print("|---:|---:|---:|---:|")
    for i, (j_ev, c_ev) in enumerate(zip(jax_per_seed, cyborg_per_seed)):
        j_mean = mean(j_ev["per_episode"])
        c_mean = mean(c_ev["per_episode"])
        j_sd = stdev(j_ev["per_episode"]) if len(j_ev["per_episode"]) > 1 else 0.0
        c_sd = stdev(c_ev["per_episode"]) if len(c_ev["per_episode"]) > 1 else 0.0
        g = j_mean - c_mean
        print(f"| {i} | {j_mean:>+8.1f} ± {j_sd:>5.0f} | {c_mean:>+8.1f} ± {c_sd:>5.0f} | {g:>+7.1f} |")
    print(f"| **pooled** | **{jax_pool_mean:+.1f}** | **{cyborg_pool_mean:+.1f}** | **{cross_policy_gap_pooled:+.1f}** |")
    print(f"\nσ across seed gaps: **{sigma_seed_gap:.1f}**, paired SE pooled: {paired_se_pooled:.1f}")
    print()
    print("## Training-time reward (mean ± σ across seeds, final 50 updates)")
    print()
    print("| backend | mean | σ_seed |")
    print("|---|---:|---:|")
    print(f"| JAX     | {jax_train_mean:+.1f} | {jax_train_sigma:.1f} |")
    print(f"| CybORG  | {cyborg_train_mean:+.1f} | {cyborg_train_sigma:.1f} |")
    print()
    print("## PPO diagnostics (mean across seeds, final 50 updates)")
    print()
    print("| metric | JAX | CybORG |")
    print("|---|---:|---:|")
    print(f"| entropy | {jax_entropy:.3f} | {cyborg_entropy:.3f} |")
    print(f"| approx_kl | {jax_kl:.4f} | {cyborg_kl:.4f} |")
    print(f"| explained_var | {jax_ev:.3f} | {cyborg_ev:.3f} |")
    print()
    print("## Verdict — vs locked-in target")
    print()
    print("| metric | observed | target | verdict |")
    print("|---|---:|---:|:--|")
    print(_stat("JAX train ep_rew", jax_train_mean, "-1998 ± 118", _verdict("jax_train_ep_rew_at_3M", jax_train_mean)))
    print(_stat("CybORG train ep_rew", cyborg_train_mean, "-1854 ± 46", _verdict("cyborg_train_ep_rew_at_3M", cyborg_train_mean)))
    print(_stat("JAX eval pooled", jax_pool_mean, "-1853", _verdict("jax_eval_pooled", jax_pool_mean)))
    print(_stat("CybORG eval pooled", cyborg_pool_mean, "-1858", _verdict("cyborg_eval_pooled", cyborg_pool_mean)))
    print(_stat("cross-policy gap", cross_policy_gap_pooled, "+5.4 ± 58", _verdict("cross_policy_gap", cross_policy_gap_pooled)))
    print(_stat("JAX entropy", jax_entropy, "~3.84", _verdict("entropy", jax_entropy)))
    print(_stat("CybORG entropy", cyborg_entropy, "~3.84", _verdict("entropy", cyborg_entropy)))
    print(_stat("JAX kl", jax_kl, "~0.005", _verdict("approx_kl", jax_kl)))
    print(_stat("CybORG kl", cyborg_kl, "~0.005", _verdict("approx_kl", cyborg_kl)))
    print(_stat("JAX exp_var", jax_ev, "~0.61", _verdict("explained_var", jax_ev)))
    print(_stat("CybORG exp_var", cyborg_ev, "~0.61", _verdict("explained_var", cyborg_ev)))


if __name__ == "__main__":
    main()
