"""Aggregate the matched-training v2 final audit.

Combines:
  - JAX-trained policy stats parsed from `evals/seed{S}/transfer.log` and
    `tost_result.json` (already produced by scripts/eval/transfer.py).
  - CybORG-trained policy stats from `evals/seed{S}/cyborg_audit.json`
    (produced by scripts/dev/audit_cyborg_policy.py — same rollout
    convention, same per-component monkeypatch).

Outputs a single JSON + markdown report under
`jaxborg-exp/matched_training_v2/evals/v2_audit_report.{json,md}`
covering:

  - Per-seed paired reward (J−C)  + paired SE + ρ + TOST verdicts
  - Per-component reward breakdown (RIA / LWF / ASF / implied action_cost),
    both policies, all 3 seeds
  - Per-agent action distribution (8 coarse buckets), both policies,
    pooled across seeds
  - L1 distance between policies' action distributions per agent
  - Action entropy + Hill diversity per agent per policy
  - Busy fraction per policy
  - Sanity: episode-length distribution
"""

# ruff: noqa: E402

import json
import math
import os
import re
import statistics
from collections import Counter
from pathlib import Path

# Honor JAXBORG_EXP_DIR (project convention). Default to the canonical
# sibling-of-repo location used in this experiment's artifacts.
_DEFAULT_EXP = Path(__file__).resolve().parents[2].parent / "jaxborg-exp"
EXP_DIR = Path(os.environ.get("JAXBORG_EXP_DIR", str(_DEFAULT_EXP))).resolve()
EVALS = EXP_DIR / "matched_training_v2/evals"
OUT_JSON = EVALS / "v2_audit_report.json"
OUT_MD = EVALS / "v2_audit_report.md"

ACTION_BUCKETS = ["Sleep", "Monitor", "Analyse", "Remove", "Restore",
                  "Decoy", "BlockTraffic", "AllowTraffic", "Other"]


def _label_to_bucket(label: str) -> str:
    # Strip the "[Invalid] " prefix CybORG attaches to actions whose target
    # host doesn't exist for this agent — the underlying action class is what
    # we want to bucket on (and these are masked out at rollout time anyway,
    # but the labels still appear in action_labels(agent)).
    if label.startswith("[Invalid] "):
        label = label[len("[Invalid] "):]
    if label.startswith("Sleep"):
        return "Sleep"
    if label.startswith("Monitor"):
        return "Monitor"
    if label.startswith("Analyse"):
        return "Analyse"
    if label.startswith("Remove"):
        return "Remove"
    if label.startswith("Restore"):
        return "Restore"
    if "Decoy" in label or label.startswith("Deploy"):
        return "Decoy"
    if label.startswith("BlockTraffic"):
        return "BlockTraffic"
    if label.startswith("AllowTraffic"):
        return "AllowTraffic"
    return "Other"


def _shannon_entropy_nats(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    h = 0.0
    for v in counter.values():
        if v == 0:
            continue
        p = v / total
        h -= p * math.log(p)
    return h


def _hill_diversity(counter: Counter) -> float:
    """Hill q=1 (= exp(Shannon entropy)) — effective number of action types."""
    return math.exp(_shannon_entropy_nats(counter))


def _l1(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    return sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


# ── JAX-trained: parse transfer.log per seed ─────────────────────────────


def parse_transfer_log(path: Path) -> dict:
    """Extract the CybORG-side per-agent action distribution and per-component."""
    text = path.read_text()
    out = {
        "per_agent_dist_cyborg_env_decisions": {},
        "per_agent_dist_cyborg_env_all": {},
        "per_component": {},
        "busy_fraction_cyborg_env": None,
    }

    # "Per-Agent Action Distribution (CybORG (decisions only)):" table
    m = re.search(
        r"Per-Agent Action Distribution \(CybORG \(decisions only\)\):\n"
        r"Agent.*\n-+\n((?:blue_\d.*\n)+)",
        text,
    )
    if m:
        for line in m.group(1).strip().splitlines():
            parts = line.split()
            if not parts:
                continue
            agent = parts[0]
            # transfer.log uses "blue_0" labels; align with audit JSON's "blue_agent_0".
            if agent.startswith("blue_") and not agent.startswith("blue_agent_"):
                agent = "blue_agent_" + agent[len("blue_"):]
            pcts = [float(x.rstrip("%")) / 100.0 for x in parts[1:]]
            keys = ["Sleep", "Monitor", "Analyse", "Remove", "Restore",
                    "Decoy", "BlockTraffic", "AllowTraffic"]
            out["per_agent_dist_cyborg_env_decisions"][agent] = dict(zip(keys, pcts))

    m = re.search(
        r"Per-Agent Action Distribution \(CybORG \(all steps\)\):\n"
        r"Agent.*\n-+\n((?:blue_\d.*\n)+)",
        text,
    )
    if m:
        for line in m.group(1).strip().splitlines():
            parts = line.split()
            if not parts:
                continue
            agent = parts[0]
            # transfer.log uses "blue_0" labels; align with audit JSON's "blue_agent_0".
            if agent.startswith("blue_") and not agent.startswith("blue_agent_"):
                agent = "blue_agent_" + agent[len("blue_"):]
            pcts = [float(x.rstrip("%")) / 100.0 for x in parts[1:]]
            keys = ["Sleep", "Monitor", "Analyse", "Remove", "Restore",
                    "Decoy", "BlockTraffic", "AllowTraffic"]
            out["per_agent_dist_cyborg_env_all"][agent] = dict(zip(keys, pcts))

    # busy fraction line
    m = re.search(r"Busy fraction:\s*JAXborg\s*([\d.]+)%,\s*CybORG\s*([\d.]+)%", text)
    if m:
        out["busy_fraction_cyborg_env"] = float(m.group(2)) / 100.0

    # per-component breakdown table
    m = re.search(
        r"Component\s+JAXborg\s+CybORG\s+Gap.*?\n-+\n"
        r"RIA \(Red Impact\)\s+(-?[\d.]+)\s+(-?[\d.]+).*\n"
        r"LWF \(LocalWork\)\s+(-?[\d.]+)\s+(-?[\d.]+).*\n"
        r"ASF \(AccessSvc\)\s+(-?[\d.]+)\s+(-?[\d.]+)",
        text,
    )
    if m:
        # JAXborg column = JAX policy on JAX env; CybORG column = JAX policy on CybORG env
        out["per_component"] = {
            "ria_jax_env": float(m.group(1)), "ria_cyborg_env": float(m.group(2)),
            "lwf_jax_env": float(m.group(3)), "lwf_cyborg_env": float(m.group(4)),
            "asf_jax_env": float(m.group(5)), "asf_cyborg_env": float(m.group(6)),
        }
    return out


# ── CybORG-trained: load audit JSON, compute distributions ────────────────


def cyborg_policy_stats(audit_path: Path) -> dict:
    d = json.load(audit_path.open())
    labels = d["action_labels_by_agent"]

    per_agent_dist_all = {}
    per_agent_dist_decisions = {}
    per_agent_entropy = {}
    per_agent_hill = {}
    per_agent_busy_frac = {}
    busy_total = 0
    step_total = 0

    for i, aid in enumerate(["blue_agent_0", "blue_agent_1", "blue_agent_2",
                             "blue_agent_3", "blue_agent_4"]):
        actions = d["per_step_actions_by_agent"][i]
        busy = d["per_step_busy_by_agent"][i]
        agent_labels = labels[aid]
        n_steps = len(actions)
        busy_total += sum(busy)
        step_total += n_steps

        cnt_all = Counter()
        cnt_dec = Counter()
        for a, b in zip(actions, busy):
            bucket = _label_to_bucket(agent_labels[a])
            cnt_all[bucket] += 1
            if b == 0:
                cnt_dec[bucket] += 1

        n_all = sum(cnt_all.values())
        n_dec = sum(cnt_dec.values())
        per_agent_dist_all[aid] = {k: cnt_all.get(k, 0) / n_all for k in ACTION_BUCKETS if cnt_all.get(k, 0)}
        per_agent_dist_decisions[aid] = {k: cnt_dec.get(k, 0) / n_dec for k in ACTION_BUCKETS if cnt_dec.get(k, 0)} if n_dec else {}
        per_agent_entropy[aid] = _shannon_entropy_nats(cnt_dec)
        per_agent_hill[aid] = _hill_diversity(cnt_dec)
        per_agent_busy_frac[aid] = sum(busy) / n_steps if n_steps else 0.0

    return {
        "per_agent_dist_all": per_agent_dist_all,
        "per_agent_dist_decisions": per_agent_dist_decisions,
        "per_agent_entropy_decisions": per_agent_entropy,
        "per_agent_hill_diversity_decisions": per_agent_hill,
        "per_agent_busy_fraction": per_agent_busy_frac,
        "busy_fraction_overall": busy_total / step_total,
        "mean_reward": d["mean_reward"],
        "stdev_reward": d["stdev_reward"],
        "mean_ria": d["mean_ria"],
        "mean_lwf": d["mean_lwf"],
        "mean_asf": d["mean_asf"],
        "per_episode_reward": d["per_episode_reward"],
    }


def jax_policy_dist_to_entropy_hill(per_agent_dist: dict) -> dict:
    out_e, out_h = {}, {}
    for agent, dist in per_agent_dist.items():
        cnt = Counter()
        for k, v in dist.items():
            cnt[k] = int(round(v * 1_000_000))  # scale to int counter
        out_e[agent] = _shannon_entropy_nats(cnt)
        out_h[agent] = _hill_diversity(cnt)
    return {"entropy": out_e, "hill": out_h}


# ── TOST helper ────────────────────────────────────────────────────────────


def tost(diff_mean: float, diff_se: float, delta: float, alpha: float = 0.05) -> dict:
    """Two one-sided test for equivalence within ±delta. Large-n z-approx."""
    z_lower = (diff_mean - (-delta)) / diff_se
    z_upper = (diff_mean - delta) / diff_se
    # p_lower = P(Z < z_lower); we want z_lower > z_crit for the lower test.
    # We compute the *upper-tail* p for the lower-side and *lower-tail* p for the upper-side.
    from math import erf, sqrt
    def _ndist_cdf(z): return 0.5 * (1 + erf(z / sqrt(2)))
    p_lower = 1 - _ndist_cdf(z_lower)
    p_upper = _ndist_cdf(z_upper)
    p_max = max(p_lower, p_upper)
    return {"delta": delta, "diff_mean": diff_mean, "diff_se": diff_se,
            "z_lower": z_lower, "z_upper": z_upper, "p_max": p_max,
            "equivalent": p_max < alpha}


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    seeds = (0, 1, 2)
    per_seed = {}
    pooled_diffs = []

    for s in seeds:
        # Paired rewards
        tost_path = EVALS / f"seed{s}/tost_result.json"
        audit_path = EVALS / f"seed{s}/cyborg_audit.json"
        cyb_eval_path = EVALS / f"seed{s}/cyborg_eval.json"
        log_path = EVALS / f"seed{s}/transfer.log"

        jax_eval = json.load(tost_path.open())
        jax_on_cyb = jax_eval["cyborg_rewards"]
        jax_on_jax = jax_eval["jax_rewards"]
        cyb_on_cyb = json.load(cyb_eval_path.open())["per_episode"]
        cyb_audit = cyborg_policy_stats(audit_path)
        jax_log = parse_transfer_log(log_path)

        diffs = [j - c for j, c in zip(jax_on_cyb, cyb_on_cyb)]
        pooled_diffs.extend(diffs)

        # Add JAX-policy entropy/hill from parsed dists
        jax_eh = jax_policy_dist_to_entropy_hill(jax_log["per_agent_dist_cyborg_env_decisions"])

        per_seed[s] = {
            "rewards": {
                "jax_on_jax_mean": statistics.mean(jax_on_jax),
                "jax_on_cyb_mean": statistics.mean(jax_on_cyb),
                "cyb_on_cyb_mean": statistics.mean(cyb_on_cyb),
                "paired_diff_mean": statistics.mean(diffs),
                "paired_diff_se": statistics.stdev(diffs) / (len(diffs) ** 0.5),
                "n": len(diffs),
            },
            "components": {
                "jax_policy": {
                    "ria_on_cyb": jax_log["per_component"].get("ria_cyborg_env"),
                    "lwf_on_cyb": jax_log["per_component"].get("lwf_cyborg_env"),
                    "asf_on_cyb": jax_log["per_component"].get("asf_cyborg_env"),
                },
                "cyborg_policy": {
                    "ria_on_cyb": cyb_audit["mean_ria"],
                    "lwf_on_cyb": cyb_audit["mean_lwf"],
                    "asf_on_cyb": cyb_audit["mean_asf"],
                },
            },
            "action_dist_cyborg_env_decisions": {
                "jax_policy": jax_log["per_agent_dist_cyborg_env_decisions"],
                "cyborg_policy": cyb_audit["per_agent_dist_decisions"],
            },
            "entropy_cyborg_env_decisions": {
                "jax_policy": jax_eh["entropy"],
                "cyborg_policy": cyb_audit["per_agent_entropy_decisions"],
            },
            "hill_diversity_cyborg_env_decisions": {
                "jax_policy": jax_eh["hill"],
                "cyborg_policy": cyb_audit["per_agent_hill_diversity_decisions"],
            },
            "busy_fraction": {
                "jax_policy_cyborg_env": jax_log["busy_fraction_cyborg_env"],
                "cyborg_policy_cyborg_env": cyb_audit["busy_fraction_overall"],
            },
        }

    pooled_mean = statistics.mean(pooled_diffs)
    pooled_se = statistics.stdev(pooled_diffs) / (len(pooled_diffs) ** 0.5)
    tost_results = {f"delta_{d}": tost(pooled_mean, pooled_se, d) for d in (84, 200, 284)}

    # Per-agent L1 distance pooled across seeds
    pooled_l1 = {}
    for agent in ("blue_agent_0", "blue_agent_1", "blue_agent_2", "blue_agent_3", "blue_agent_4"):
        l1s = []
        for s in seeds:
            jdist = per_seed[s]["action_dist_cyborg_env_decisions"]["jax_policy"].get(agent, {})
            cdist = per_seed[s]["action_dist_cyborg_env_decisions"]["cyborg_policy"].get(agent, {})
            if jdist and cdist:
                l1s.append(_l1(jdist, cdist))
        if l1s:
            pooled_l1[agent] = {"mean": statistics.mean(l1s), "per_seed": l1s}

    report = {
        "summary": {
            "pooled_paired_diff_mean": pooled_mean,
            "pooled_paired_se": pooled_se,
            "pooled_n": len(pooled_diffs),
            "tost": tost_results,
        },
        "per_seed": per_seed,
        "cross_policy_action_l1_per_agent_pooled": pooled_l1,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {OUT_JSON}")

    # ── Markdown summary ─────────────────────────────────────────────────
    md = []
    md.append("# Matched-Training v2 Final Audit\n")
    md.append(f"Generated by `scripts/dev/aggregate_matched_v2_audit.py` from "
              f"`jaxborg-exp/matched_training_v2/evals/`. Both policies eval'd on CybORG env "
              f"(n=100 each, seeds 42–141, paired).\n")

    md.append("\n## Headline reward gap\n")
    md.append("| seed | JAX→CybORG | CybORG→CybORG | gap (J−C) | paired SE |")
    md.append("|---:|---:|---:|---:|---:|")
    for s in seeds:
        r = per_seed[s]["rewards"]
        md.append(f"| {s} | {r['jax_on_cyb_mean']:.1f} | {r['cyb_on_cyb_mean']:.1f} | "
                  f"{r['paired_diff_mean']:+.1f} | {r['paired_diff_se']:.1f} |")
    md.append(f"| **pooled n={len(pooled_diffs)}** | | | **{pooled_mean:+.1f}** | **{pooled_se:.1f}** |")
    md.append("")
    md.append("**TOST verdicts**:")
    for d in (84, 200, 284):
        t = tost_results[f"delta_{d}"]
        verdict = "EQUIVALENT" if t["equivalent"] else "NOT EQUIVALENT"
        md.append(f"- Δ={d}: p_max={t['p_max']:.4f} → **{verdict}**")

    md.append("\n## Per-component reward (mean across 3 seeds, 100 eps each, on CybORG env)\n")
    md.append("| component | JAX-trained | CybORG-trained | gap (J−C) |")
    md.append("|---|---:|---:|---:|")
    for comp in ("ria", "lwf", "asf"):
        jax_v = statistics.mean(per_seed[s]["components"]["jax_policy"][f"{comp}_on_cyb"] for s in seeds)
        cyb_v = statistics.mean(per_seed[s]["components"]["cyborg_policy"][f"{comp}_on_cyb"] for s in seeds)
        md.append(f"| {comp.upper()} | {jax_v:.1f} | {cyb_v:.1f} | {jax_v - cyb_v:+.1f} |")

    md.append("\n## Per-agent action distribution (mean across 3 seeds, decisions only, CybORG env)\n")
    for agent in ("blue_agent_0", "blue_agent_1", "blue_agent_2", "blue_agent_3", "blue_agent_4"):
        md.append(f"### {agent}\n")
        md.append("| bucket | JAX-trained | CybORG-trained | Δ |")
        md.append("|---|---:|---:|---:|")
        all_buckets = set()
        for s in seeds:
            all_buckets.update(per_seed[s]["action_dist_cyborg_env_decisions"]["jax_policy"].get(agent, {}).keys())
            all_buckets.update(per_seed[s]["action_dist_cyborg_env_decisions"]["cyborg_policy"].get(agent, {}).keys())
        for b in ACTION_BUCKETS:
            if b not in all_buckets:
                continue
            j_avg = statistics.mean(per_seed[s]["action_dist_cyborg_env_decisions"]["jax_policy"].get(agent, {}).get(b, 0.0) for s in seeds)
            c_avg = statistics.mean(per_seed[s]["action_dist_cyborg_env_decisions"]["cyborg_policy"].get(agent, {}).get(b, 0.0) for s in seeds)
            md.append(f"| {b} | {100 * j_avg:.1f}% | {100 * c_avg:.1f}% | {100 * (j_avg - c_avg):+.1f}% |")
        l1 = pooled_l1.get(agent, {}).get("mean")
        if l1 is not None:
            md.append(f"\n**L1 distance (avg across 3 seeds): {l1:.3f}**")
        md.append("")

    md.append("\n## Action diversity (decisions only, mean across 3 seeds)\n")
    md.append("| agent | JAX entropy (nats) | CybORG entropy (nats) | JAX Hill | CybORG Hill |")
    md.append("|---|---:|---:|---:|---:|")
    for agent in ("blue_agent_0", "blue_agent_1", "blue_agent_2", "blue_agent_3", "blue_agent_4"):
        je = statistics.mean(per_seed[s]["entropy_cyborg_env_decisions"]["jax_policy"].get(agent, 0.0) for s in seeds)
        ce = statistics.mean(per_seed[s]["entropy_cyborg_env_decisions"]["cyborg_policy"].get(agent, 0.0) for s in seeds)
        jh = statistics.mean(per_seed[s]["hill_diversity_cyborg_env_decisions"]["jax_policy"].get(agent, 0.0) for s in seeds)
        ch = statistics.mean(per_seed[s]["hill_diversity_cyborg_env_decisions"]["cyborg_policy"].get(agent, 0.0) for s in seeds)
        md.append(f"| {agent} | {je:.3f} | {ce:.3f} | {jh:.2f} | {ch:.2f} |")

    md.append("\n## Busy fraction (CybORG env)\n")
    md.append("| seed | JAX-trained | CybORG-trained |")
    md.append("|---:|---:|---:|")
    for s in seeds:
        bj = per_seed[s]["busy_fraction"]["jax_policy_cyborg_env"]
        bc = per_seed[s]["busy_fraction"]["cyborg_policy_cyborg_env"]
        md.append(f"| {s} | {100*bj:.1f}% | {100*bc:.1f}% |")

    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
