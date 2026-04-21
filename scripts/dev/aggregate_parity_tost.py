"""Aggregate N-seed TOST evaluation JSONs into a final parity verdict.

Consumes per-seed `tost_result.json` files written by `scripts/eval/transfer.py`
(one per trained seed, each containing per-episode `jax_rewards` and
`cyborg_rewards` arrays from an independent 100-ep stochastic eval on both
backends). Derives:

  - Noise anchor:  Δ_noise  = 2 * σ(per-seed JAXborg means)
  - Signal anchor: Δ_signal = 0.05 * (mean_trained_on_cyborg - sleep_mean)
  - Δ = max(Δ_noise, Δ_signal)

Runs TOST per checkpoint and pooled (all episodes concatenated) at the chosen
Δ and the paper's Δ=200 for reference. Emits a final JSON report.

Example:
    uv run python scripts/dev/aggregate_parity_tost.py \\
        --tost-jsons eval_seed0/tost_result.json \\
                     eval_seed1/tost_result.json \\
                     eval_seed2/tost_result.json \\
        --sleep-mean -6559 \\
        --output parity_final_report.json
"""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from scipy import stats


def tost_independent(perf, ref, margin, alpha=0.05):
    """Welch two one-sided t-tests for independent samples. Mirrors
    `tost_equivalence(..., paired=False)` in scripts/eval/transfer.py."""
    perf = np.asarray(perf, dtype=float)
    ref = np.asarray(ref, dtype=float)
    n_perf, n_ref = len(perf), len(ref)
    mean_diff = float(perf.mean() - ref.mean())
    s1, s2 = float(perf.var(ddof=1)), float(ref.var(ddof=1))
    se = float(np.sqrt(s1 / n_perf + s2 / n_ref))
    nu_num = (s1 / n_perf + s2 / n_ref) ** 2
    nu_den = (s1 / n_perf) ** 2 / (n_perf - 1) + (s2 / n_ref) ** 2 / (n_ref - 1)
    df = nu_num / nu_den if nu_den > 0 else min(n_perf, n_ref) - 1

    if se < 1e-12:
        return {
            "equivalent": True,
            "mean_diff": mean_diff,
            "margin": margin,
            "p_upper": 0.0,
            "p_lower": 0.0,
            "ci_lower": mean_diff,
            "ci_upper": mean_diff,
            "n_perf": n_perf,
            "n_ref": n_ref,
        }

    t_upper = (mean_diff - margin) / se
    p_upper = float(stats.t.cdf(t_upper, df))
    t_lower = (mean_diff + margin) / se
    p_lower = float(1.0 - stats.t.cdf(t_lower, df))
    t_crit = float(stats.t.ppf(1 - alpha, df))
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    return {
        "equivalent": p_upper < alpha and p_lower < alpha,
        "mean_diff": mean_diff,
        "margin": margin,
        "p_upper": p_upper,
        "p_lower": p_lower,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_perf": n_perf,
        "n_ref": n_ref,
    }


def load_seed_jsons(paths):
    """Load per-seed TOST JSONs and return [(label, jax_rewards, cyborg_rewards), ...]."""
    results = []
    for p in paths:
        data = json.loads(Path(p).read_text())
        jx = data.get("jax_rewards")
        cb = data.get("cyborg_rewards")
        if jx is None or cb is None:
            raise ValueError(f"{p} missing jax_rewards or cyborg_rewards (was --episodes < 2?)")
        results.append((Path(p).parent.name, np.asarray(jx, dtype=float), np.asarray(cb, dtype=float)))
    return results


def main():
    parser = argparse.ArgumentParser(description="Final parity TOST aggregator (Karten L4, multi-seed)")
    parser.add_argument(
        "--tost-jsons",
        nargs="+",
        required=True,
        help="Per-seed tost_result.json files written by scripts/eval/transfer.py",
    )
    parser.add_argument(
        "--sleep-mean",
        type=float,
        required=True,
        help="Mean episode reward of the CybORG sleep-blue policy (signal anchor floor)",
    )
    parser.add_argument(
        "--random-mean",
        type=float,
        default=None,
        help="Optional: mean episode reward of random-blue on CybORG (reported for context)",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--paper-margin", type=float, default=200.0, help="Δ from Karten paper, for context")
    parser.add_argument("--signal-ratio", type=float, default=0.05, help="Fraction of learnable span for Δ_signal")
    parser.add_argument("--noise-multiplier", type=float, default=2.0, help="σ multiplier for Δ_noise")
    parser.add_argument("--output", type=Path, default=None, help="Write final report to this JSON path")
    args = parser.parse_args()

    seeds = load_seed_jsons(args.tost_jsons)
    if len(seeds) < 2:
        raise SystemExit("Need >=2 seed JSONs to compute across-seed σ")

    print("=" * 70)
    print(f"Loaded {len(seeds)} seed(s):")
    for label, jx, cb in seeds:
        print(f"  {label:40s}  n_jax={len(jx):4d}  jax_mean={jx.mean():+8.1f}  cyborg_mean={cb.mean():+8.1f}")
    print("=" * 70)

    jax_means = [float(jx.mean()) for _, jx, _ in seeds]
    cyb_means = [float(cb.mean()) for _, _, cb in seeds]

    seed_sigma = stdev(jax_means) if len(jax_means) > 1 else 0.0
    delta_noise = args.noise_multiplier * seed_sigma

    trained_mean_on_cyborg = mean(cyb_means)
    signal_span = trained_mean_on_cyborg - args.sleep_mean
    delta_signal = args.signal_ratio * abs(signal_span)

    delta = max(delta_noise, delta_signal)

    print("\n-- Δ derivation --")
    print(f"  Per-seed JAX means:        {[f'{m:+.1f}' for m in jax_means]}")
    print(f"  σ across seed JAX means:   {seed_sigma:.2f}")
    print(f"  Δ_noise  = {args.noise_multiplier}σ         = {delta_noise:.1f}")
    print(f"  Sleep mean (CybORG):       {args.sleep_mean:+.1f}")
    print(f"  Trained mean (CybORG):     {trained_mean_on_cyborg:+.1f}")
    print(f"  Learnable span:            {signal_span:+.1f}")
    print(f"  Δ_signal = {args.signal_ratio:.0%} of span        = {delta_signal:.1f}")
    print(f"  Δ        = max(noise, signal) = {delta:.1f}")
    if args.random_mean is not None:
        print(f"  (Random-blue mean: {args.random_mean:+.1f})")

    per_ckpt = []
    print("\n-- Per-checkpoint TOST (independent) --")
    for label, jx, cb in seeds:
        r = tost_independent(jx, cb, delta, args.alpha)
        per_ckpt.append({"label": label, **r})
        verdict = "EQUIVALENT" if r["equivalent"] else "NOT EQUIVALENT"
        print(
            f"  {label:40s}  gap={r['mean_diff']:+7.1f}  "
            f"CI=[{r['ci_lower']:+7.1f},{r['ci_upper']:+7.1f}]  "
            f"p={max(r['p_upper'], r['p_lower']):.4f}  {verdict}"
        )

    pooled_jax = np.concatenate([jx for _, jx, _ in seeds])
    pooled_cyb = np.concatenate([cb for _, _, cb in seeds])
    pooled = tost_independent(pooled_jax, pooled_cyb, delta, args.alpha)
    paper = tost_independent(pooled_jax, pooled_cyb, args.paper_margin, args.alpha)

    print("\n-- Pooled TOST (all episodes concatenated) --")
    for label, r in [("Δ (derived)", pooled), (f"Δ={args.paper_margin:.0f} (paper)", paper)]:
        verdict = "EQUIVALENT" if r["equivalent"] else "NOT EQUIVALENT"
        print(
            f"  {label:20s}  margin=±{r['margin']:7.1f}  gap={r['mean_diff']:+7.1f}  "
            f"CI=[{r['ci_lower']:+7.1f},{r['ci_upper']:+7.1f}]  "
            f"p={max(r['p_upper'], r['p_lower']):.4f}  {verdict}"
        )

    delta_vs_sigma = delta / seed_sigma if seed_sigma > 1e-9 else float("inf")
    delta_vs_span = delta / abs(signal_span) if abs(signal_span) > 1e-9 else float("inf")
    print(
        f"\nOne-line verdict: Δ={delta:.0f} = {delta_vs_sigma:.1f}×σ_seed = "
        f"{delta_vs_span * 100:.1f}% of learnable signal, pooled p="
        f"{max(pooled['p_upper'], pooled['p_lower']):.3f} → "
        f"{'EQUIVALENT' if pooled['equivalent'] else 'NOT EQUIVALENT'}"
    )

    report = {
        "n_seeds": len(seeds),
        "seed_jax_means": jax_means,
        "seed_cyborg_means": cyb_means,
        "seed_sigma_jax_means": seed_sigma,
        "sleep_mean": args.sleep_mean,
        "random_mean": args.random_mean,
        "trained_mean_on_cyborg": trained_mean_on_cyborg,
        "signal_span": signal_span,
        "delta_noise": delta_noise,
        "delta_signal": delta_signal,
        "delta_chosen": delta,
        "delta_vs_sigma": delta_vs_sigma,
        "delta_vs_span": delta_vs_span,
        "alpha": args.alpha,
        "per_checkpoint_tost": per_ckpt,
        "pooled_tost_delta": pooled,
        "pooled_tost_paper_delta": paper,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
