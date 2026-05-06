"""CLI orchestration for dev transfer parity checks."""

# ruff: noqa: E402,I001

from __future__ import annotations

from scripts.dev.parity.bootstrap import EXP_DIR, ROOT, configure_runtime

configure_runtime()

import argparse
import json
from pathlib import Path
from statistics import stdev

import numpy as np

from jaxborg.constants import NUM_BLUE_AGENTS
from scripts.dev.parity.cyborg_rollout import rollout_cyborg
from scripts.dev.parity.diagnostics import (
    print_mask_summary,
    run_random_baseline,
    run_sleep_baseline,
    run_verbose_trace,
)
from scripts.dev.parity.jax_rollout import rollout_jaxborg, rollout_jaxborg_scan
from scripts.dev.parity.matched_rollout import rollout_matched_transfer
from scripts.dev.parity.policy import load_checkpoint
from scripts.dev.parity.reporting import (
    plot_action_distribution,
    plot_training_curves,
    print_comparison_report,
    print_per_agent_action_dist,
    print_tost_report,
    print_trajectory_summary,
    save_reward_plot,
)
from scripts.dev.parity.rollout_types import TransferComparison
from scripts.dev.parity.stats import ACTION_TYPE_NAMES, action_distribution, l1_distribution_distance


def main():
    parser = argparse.ArgumentParser(description="Evaluate JAXborg-trained policy: rollout, transfer, baselines")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_final.pkl")
    parser.add_argument("--episodes", type=int, default=10, help="Rollout episodes (default 10)")
    parser.add_argument("--deterministic", action="store_true", help="Use argmax instead of sampling from policy")
    parser.add_argument(
        "--matched",
        action="store_true",
        help="Run matched-state transfer diagnostics (lockstep JAX/CybORG, for debugging parity)",
    )
    parser.add_argument(
        "--jax-only",
        action="store_true",
        help="Run JAXborg-only evaluation (no CybORG, much faster)",
    )
    parser.add_argument(
        "--no-scan",
        action="store_true",
        help="Disable jax.lax.scan for JAX-only eval (use Python loop instead, for debugging)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    parser.add_argument("--baselines", action="store_true", help="Run sleep + random baselines")
    parser.add_argument("--verbose", type=int, default=0, help="Step-by-step CybORG trace for N steps")
    parser.add_argument("--plot", action="store_true", help="Save action dist + training curve PNGs")
    parser.add_argument("--mask-summary", action="store_true", help="Print per-agent mask breakdown")
    parser.add_argument("--tost-margin", type=float, default=200.0, help="Equivalence margin for TOST")
    parser.add_argument("--tost-alpha", type=float, default=0.05, help="Significance level for TOST")
    parser.add_argument(
        "--tost-output",
        type=str,
        default=None,
        help="Path for TOST JSON output (default: $JAXBORG_EXP_DIR/tost_result.json)",
    )
    args = parser.parse_args()

    deterministic = args.deterministic

    print(f"Loading checkpoint: {args.checkpoint}")
    policy, params = load_checkpoint(args.checkpoint)

    if args.mask_summary:
        print_mask_summary()

    if args.jax_only:
        print("\n" + "=" * 70)
        print("JAX-ONLY EVALUATION (no CybORG)")
        print("=" * 70)
        use_scan = not args.no_scan
        rollout_fn = rollout_jaxborg_scan if use_scan else rollout_jaxborg
        if use_scan:
            print("Using jax.lax.scan (compiled rollout, cached to disk)")
        jax_rollout = rollout_fn(
            policy,
            params,
            args.episodes,
            deterministic,
            seed=args.seed,
        )
        jax_actions = jax_rollout.actions
        jax_rewards = jax_rollout.rewards
        jax_results = jax_rollout.episodes
        jax_pooled_by_agent = [
            [a for ep in jax_results for a in ep.actions_by_agent[i]] for i in range(NUM_BLUE_AGENTS)
        ]
        print_per_agent_action_dist(jax_pooled_by_agent, label="JAXborg (all steps)")

        # Decision-only distribution (non-busy steps)
        if jax_results[0].blue_busy_by_agent:
            decision_by_agent = [
                [a for ep in jax_results for a, b in zip(ep.actions_by_agent[i], ep.blue_busy_by_agent[i]) if b == 0]
                for i in range(NUM_BLUE_AGENTS)
            ]
            total_steps = sum(len(ep.actions_by_agent[0]) for ep in jax_results) * NUM_BLUE_AGENTS
            decision_steps = sum(len(agent) for agent in decision_by_agent)
            busy_pct = 100.0 * (1.0 - decision_steps / total_steps) if total_steps > 0 else 0.0
            print(f"\n  (busy fraction: {busy_pct:.1f}% — filtered out below)")
            print_per_agent_action_dist(decision_by_agent, label="JAXborg (decisions only)")

        # Per-phase distribution
        if jax_results[0].phase_per_step:
            phase_names = {0: "Phase0", 1: "MissionA", 2: "MissionB"}
            phase_actions = {0: [], 1: [], 2: []}
            for ep in jax_results:
                for step_idx, phase in enumerate(ep.phase_per_step):
                    for i in range(NUM_BLUE_AGENTS):
                        busy = ep.blue_busy_by_agent[i][step_idx] if ep.blue_busy_by_agent else 0
                        if busy == 0:
                            phase_actions[phase].append(ep.actions_by_agent[i][step_idx])
            header = f"{'Phase':<10}"
            for name in ACTION_TYPE_NAMES:
                header += f" {name:>8}"
            header += f" {'N':>8}"
            print("\nPer-Phase Action Distribution (decisions only):")
            print(header)
            print("-" * len(header))
            for phase in [0, 1, 2]:
                acts = phase_actions[phase]
                if not acts:
                    continue
                dist = action_distribution(acts)
                row = f"{phase_names[phase]:<10}"
                for pct in dist:
                    row += f" {pct * 100:7.1f}%"
                row += f" {len(acts):>8}"
                print(row)

        print_trajectory_summary(jax_results[-1].trajectory, label=f"JAXborg ep {len(jax_results)}")

        print(f"\nMean reward ({args.episodes} episodes): {jax_rewards.mean():.1f}")
        if len(jax_rewards) > 1:
            print(f"Stdev: {stdev(jax_rewards.tolist()):.1f}")

        if args.plot:
            output_dir = EXP_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_action_distribution(jax_actions, "JAXborg Action Distribution", output_dir / "jax_action_dist.png")
        return

    is_matched = args.matched

    if not is_matched:
        print("\n" + "=" * 70)
        print("FULLY INDEPENDENT ROLLOUTS")
        print("=" * 70)
        print("Each backend runs completely on its own — no sync of any kind.")
        print("Same policy weights, independent generated/CybORG topologies and everything else.")
        print("Compare population means via TOST.\n")

        use_scan = not args.no_scan

        # Run JAXborg (scan/vmap) and CybORG (sequential) concurrently
        from concurrent.futures import ThreadPoolExecutor

        def _run_jaxborg():
            rollout_fn = rollout_jaxborg_scan if use_scan else rollout_jaxborg
            if use_scan:
                print("JAXborg: using jax.lax.scan (all episodes in parallel)", flush=True)
            else:
                print("JAXborg:", flush=True)
            return rollout_fn(
                policy,
                params,
                args.episodes,
                deterministic,
                seed=args.seed,
            )

        def _run_cyborg():
            print("CybORG:", flush=True)
            return rollout_cyborg(
                policy,
                params,
                args.episodes,
                deterministic,
                seed=args.seed,
                checkpoint_path=args.checkpoint,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            jax_future = pool.submit(_run_jaxborg)
            cyborg_future = pool.submit(_run_cyborg)
            jax_rollout = jax_future.result()
            cyborg_rollout = cyborg_future.result()

        comparison = TransferComparison(
            jax_actions=jax_rollout.actions,
            jax_rewards=jax_rollout.rewards,
            jax_episodes=jax_rollout.episodes,
            cyborg_actions=cyborg_rollout.actions,
            cyborg_rewards=cyborg_rollout.rewards,
            cyborg_actions_by_agent=cyborg_rollout.actions_by_agent,
            cyborg_ria=cyborg_rollout.ria,
            cyborg_lwf=cyborg_rollout.lwf,
            cyborg_asf=cyborg_rollout.asf,
            cyborg_busy_by_agent=cyborg_rollout.busy_by_agent,
            cyborg_phase_per_step=cyborg_rollout.phase_per_step,
        )
        jax_actions = comparison.jax_actions
        jax_rewards = comparison.jax_rewards
        jax_results = comparison.jax_episodes
        cyborg_actions = comparison.cyborg_actions
        cyborg_rewards = comparison.cyborg_rewards
        cyborg_actions_by_agent = comparison.cyborg_actions_by_agent
        cyborg_ria = comparison.cyborg_ria
        cyborg_lwf = comparison.cyborg_lwf
        cyborg_asf = comparison.cyborg_asf
        cyborg_busy_by_agent = comparison.cyborg_busy_by_agent
        cyborg_phase_per_step = comparison.cyborg_phase_per_step

        jax_pooled_by_agent = [
            [a for ep in jax_results for a in ep.actions_by_agent[i]] for i in range(NUM_BLUE_AGENTS)
        ]
        print_per_agent_action_dist(jax_pooled_by_agent, label="JAXborg (all steps)")
        print_trajectory_summary(jax_results[-1].trajectory, label=f"JAXborg ep {len(jax_results)}")
        print_per_agent_action_dist(cyborg_actions_by_agent, label="CybORG (all steps)")

        # Decision-only distributions (filter out busy ticks)
        jax_decision_by_agent = [
            [a for ep in jax_results for a, b in zip(ep.actions_by_agent[i], ep.blue_busy_by_agent[i]) if b == 0]
            for i in range(NUM_BLUE_AGENTS)
        ]
        cyborg_decision_by_agent = [
            [a for a, b in zip(cyborg_actions_by_agent[i], cyborg_busy_by_agent[i]) if b == 0]
            for i in range(NUM_BLUE_AGENTS)
        ]
        jax_total = sum(len(ep.actions_by_agent[0]) for ep in jax_results) * NUM_BLUE_AGENTS
        jax_decisions = sum(len(agent) for agent in jax_decision_by_agent)
        cyborg_total = sum(len(a) for a in cyborg_actions_by_agent)
        cyborg_decisions = sum(len(a) for a in cyborg_decision_by_agent)
        jax_busy_pct = 100.0 * (1.0 - jax_decisions / jax_total) if jax_total > 0 else 0.0
        cyborg_busy_pct = 100.0 * (1.0 - cyborg_decisions / cyborg_total) if cyborg_total > 0 else 0.0
        print(f"\n  Busy fraction: JAXborg {jax_busy_pct:.1f}%, CybORG {cyborg_busy_pct:.1f}% — filtered out below")
        print_per_agent_action_dist(jax_decision_by_agent, label="JAXborg (decisions only)")
        print_per_agent_action_dist(cyborg_decision_by_agent, label="CybORG (decisions only)")

        # Per-agent L1 distribution distance (decisions only)
        print("\nPer-Agent L1 Distribution Distance (decisions only, JAXborg vs CybORG):")
        print(f"{'Agent':<10} {'L1':>8}")
        print("-" * 20)
        for agent_idx in range(NUM_BLUE_AGENTS):
            j_dist = action_distribution(jax_decision_by_agent[agent_idx])
            c_dist = action_distribution(cyborg_decision_by_agent[agent_idx])
            l1 = l1_distribution_distance(j_dist, c_dist)
            print(f"{'blue_' + str(agent_idx):<10} {l1:8.3f}")
        j_pooled = [a for agent in jax_decision_by_agent for a in agent]
        c_pooled = [a for agent in cyborg_decision_by_agent for a in agent]
        l1_pooled = l1_distribution_distance(action_distribution(j_pooled), action_distribution(c_pooled))
        print("-" * 20)
        print(f"{'POOLED':<10} {l1_pooled:8.3f}")

        # Per-phase × per-agent action distribution (decisions only, both backends)
        phase_names = {0: "Phase0", 1: "MissionA", 2: "MissionB"}
        jax_phase_actions = {
            0: [[] for _ in range(NUM_BLUE_AGENTS)],
            1: [[] for _ in range(NUM_BLUE_AGENTS)],
            2: [[] for _ in range(NUM_BLUE_AGENTS)],
        }
        for ep in jax_results:
            if not ep.phase_per_step:
                continue
            for step_idx, phase in enumerate(ep.phase_per_step):
                for i in range(NUM_BLUE_AGENTS):
                    busy = ep.blue_busy_by_agent[i][step_idx] if ep.blue_busy_by_agent else 0
                    if busy == 0:
                        jax_phase_actions[phase][i].append(ep.actions_by_agent[i][step_idx])

        # CybORG: phase_per_step and actions_by_agent are both stored per-step × agent
        # cyborg_phase_per_step: [ep][step] -> phase; cyborg_actions_by_agent: [agent][ep*500 step]
        cy_phase_actions = {
            0: [[] for _ in range(NUM_BLUE_AGENTS)],
            1: [[] for _ in range(NUM_BLUE_AGENTS)],
            2: [[] for _ in range(NUM_BLUE_AGENTS)],
        }
        steps_per_ep = 500
        for ep_idx, ep_phase in enumerate(cyborg_phase_per_step):
            if not ep_phase:
                continue
            for step_idx, phase in enumerate(ep_phase):
                abs_idx = ep_idx * steps_per_ep + step_idx
                for i in range(NUM_BLUE_AGENTS):
                    if abs_idx >= len(cyborg_actions_by_agent[i]):
                        continue
                    busy = cyborg_busy_by_agent[i][abs_idx] if cyborg_busy_by_agent[i] else 0
                    if busy == 0:
                        cy_phase_actions[phase][i].append(cyborg_actions_by_agent[i][abs_idx])

        print("\nPer-Phase × Per-Agent L1 Distribution Distance (decisions only):")
        print(f"{'Phase':<10} {'Agent':<8} {'L1':>8} {'N_jax':>8} {'N_cy':>8}")
        print("-" * 46)
        for phase in [0, 1, 2]:
            for i in range(NUM_BLUE_AGENTS):
                ja = jax_phase_actions[phase][i]
                ca = cy_phase_actions[phase][i]
                if not ja and not ca:
                    continue
                l1 = l1_distribution_distance(action_distribution(ja), action_distribution(ca))
                print(f"{phase_names[phase]:<10} {'blue_' + str(i):<8} {l1:8.3f} {len(ja):>8} {len(ca):>8}")

        # Per-component reward breakdown
        if jax_results and hasattr(jax_results[0], "ria_total"):
            jax_ria = np.array([r.ria_total for r in jax_results])
            jax_lwf = np.array([r.lwf_total for r in jax_results])
            jax_asf = np.array([r.asf_total for r in jax_results])
            print("\n" + "=" * 70)
            print("PER-COMPONENT REWARD BREAKDOWN")
            print("=" * 70)
            print(f"{'Component':<20} {'JAXborg':>12} {'CybORG':>12} {'Gap (J-C)':>12}")
            print("-" * 56)
            for label, j_arr, c_arr in [
                ("RIA (Red Impact)", jax_ria, cyborg_ria),
                ("LWF (LocalWork)", jax_lwf, cyborg_lwf),
                ("ASF (AccessSvc)", jax_asf, cyborg_asf),
            ]:
                j_mean = float(j_arr.mean()) if len(j_arr) > 0 else 0.0
                c_mean = float(c_arr.mean()) if len(c_arr) > 0 else 0.0
                print(f"{label:<20} {j_mean:>12.1f} {c_mean:>12.1f} {j_mean - c_mean:>+12.1f}")
            j_total = float(jax_rewards.mean())
            c_total = float(cyborg_rewards.mean())
            print("-" * 56)
            print(f"{'Total':<20} {j_total:>12.1f} {c_total:>12.1f} {j_total - c_total:>+12.1f}")
            # Sanity: check sum of components matches total
            j_comp_sum = float(jax_ria.mean() + jax_lwf.mean() + jax_asf.mean())
            c_comp_sum = float(cyborg_ria.mean() + cyborg_lwf.mean() + cyborg_asf.mean())
            if abs(j_comp_sum - j_total) > 1.0:
                print(f"  WARNING: JAXborg component sum ({j_comp_sum:.1f}) != total ({j_total:.1f})")
            if abs(c_comp_sum - c_total) > 1.0:
                print(f"  WARNING: CybORG component sum ({c_comp_sum:.1f}) != total ({c_total:.1f})")
    else:
        mode_label = "STOCHASTIC" if not deterministic else "DETERMINISTIC"
        print("\n" + "=" * 70)
        print(f"MATCHED TRANSFER ROLLOUT ({mode_label})")
        print("=" * 70)
        print("Stepping synced episodes with JAX-selected actions.")
        print("CybORG actions below are the policy outputs on matched CybORG observations, not applied actions.")
        comparison = rollout_matched_transfer(
            policy,
            params,
            args.episodes,
            deterministic,
            seed=args.seed,
        )
        jax_actions = comparison.jax_actions
        jax_rewards = comparison.jax_rewards
        jax_results = comparison.jax_episodes
        cyborg_actions = comparison.cyborg_actions
        cyborg_rewards = comparison.cyborg_rewards
        cyborg_actions_by_agent = comparison.cyborg_actions_by_agent
        jax_pooled_by_agent = [
            [a for ep in jax_results for a in ep.actions_by_agent[i]] for i in range(NUM_BLUE_AGENTS)
        ]
        print_per_agent_action_dist(jax_pooled_by_agent, label="JAX Policy on JAX Obs")
        print_per_agent_action_dist(cyborg_actions_by_agent, label="JAX Policy on CybORG Obs")
        print_trajectory_summary(jax_results[-1].trajectory, label=f"Matched ep {len(jax_results)}")

    # Comparison report
    print_comparison_report(jax_actions, jax_rewards, cyborg_actions, cyborg_rewards)

    if len(jax_rewards) >= 2 and len(cyborg_rewards) >= 2:
        tost_result = print_tost_report(
            jax_rewards,
            cyborg_rewards,
            margin=args.tost_margin,
            alpha=args.tost_alpha,
            paired=is_matched,
        )

        # Save TOST result alongside other outputs
        tost_path = Path(args.tost_output) if args.tost_output else EXP_DIR / "tost_result.json"
        tost_path.parent.mkdir(parents=True, exist_ok=True)
        tost_result["jax_rewards"] = jax_rewards.tolist()
        tost_result["cyborg_rewards"] = cyborg_rewards.tolist()
        tost_result["checkpoint"] = str(Path(args.checkpoint).resolve())
        tost_result["episodes"] = args.episodes
        tost_result["seed"] = args.seed
        tost_result["deterministic"] = deterministic
        tost_result["matched"] = is_matched
        tost_path.write_text(json.dumps(tost_result, indent=2) + "\n")
        print(f"Saved TOST result: {tost_path}")
        try:
            import sys

            sys.path.insert(0, str(ROOT / "scripts" / "dev"))
            from catalog import update_l4_tost

            update_l4_tost(
                equivalent=tost_result["equivalent"],
                margin=tost_result["margin"],
                mean_diff=tost_result["mean_diff"],
                episodes=len(jax_rewards),
            )
        except Exception as e:
            print(f"WARNING: Failed to update catalog with L4 TOST result: {e}")

    # Optional: baselines
    if args.baselines:
        print("\n" + "=" * 70)
        print("SLEEP BASELINE")
        print("=" * 70)
        sleep_score = run_sleep_baseline(args.episodes)
        print(f"Sleep baseline ({args.episodes} episodes): {sleep_score:.1f}")

        print("\n" + "=" * 70)
        print("RANDOM POLICY (with JAXborg action mask)")
        print("=" * 70)
        random_score = run_random_baseline(args.episodes, seed=args.seed)
        print(f"Random policy ({args.episodes} episodes): {random_score:.1f}")

    # Optional: verbose trace
    if args.verbose > 0:
        print("\n" + "=" * 70)
        print(f"VERBOSE CYBORG TRACE ({args.verbose} steps, deterministic)")
        print("=" * 70)
        run_verbose_trace(policy, params, steps=args.verbose, seed=args.seed)

    # Optional: plots
    if args.plot:
        output_dir = EXP_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_action_distribution(jax_actions, "JAXborg Action Distribution", output_dir / "jax_action_dist.png")
        plot_action_distribution(cyborg_actions, "CybORG Action Distribution", output_dir / "cyborg_action_dist.png")

        try:
            save_reward_plot(jax_rewards, cyborg_rewards)
        except Exception as e:
            print(f"(Skipped reward plot: {e})")

        # New layout: metrics live under $EXP_DIR/ippo_jax/<tag>/metrics.jsonl
        candidates = sorted(EXP_DIR.glob("ippo_jax/*/metrics.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            plot_training_curves(candidates[0], output_dir / "training_curves.png")
        else:
            print(f"No metrics file under {EXP_DIR}/ippo_jax/*/, skipping training curves")
