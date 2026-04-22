"""Paired sleep baseline: same topology on CybORG and JAXborg, per-seed diff.

Independent-rollout baselines mix two variances: per-episode noise within a
backend, and topology variance across episodes. The latter dominates SE at
n=100 and inflates CI width. This script eliminates topology variance by
running both backends on the *same* topology each iteration:

  for seed s:
      cyborg_env = CybORG(EnterpriseScenarioGenerator, seed=s)
      cyborg_env.reset()
      r_cy(s) = sleep_episode(cyborg_env)
      const = build_const_from_cyborg(cyborg_env.env)        # captures topology
      jax_env_state = build_jax_env_state_from_const(const)  # fresh state, same topo
      r_jx(s) = sleep_episode(jax_env, jax_env_state)
      d(s) = r_jx(s) - r_cy(s)

Reports mean(d), stdev(d), 90% CI, and paired TOST at Δ=±200.

Interpretation (per parity-supplementary-followup A1):
- mean(d) ≈ 0  → independent +154 was topology-sampling noise. Close A1.
- mean(d) > ~50 → real per-episode env asymmetry. Proceed to A2.
"""

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

import numpy as np

EPISODE_LENGTH = 500


def _run_one_pair_synced(args: tuple) -> dict:
    """Paired sleep episode under MATCHED green RNG via the differential harness.

    CybORG steps first; the GreenRecorder captures each green agent's np_random
    calls and converts them to the precomputed (steps, hosts, 8) buffer that
    JAX consumes the same step. Both backends therefore sample the same service,
    same reliability roll, same FP/phishing rolls on every host on every step.
    Any residual paired reward gap with sync enabled MUST come from a real
    env-mechanism divergence (not RNG-stream bias).
    """
    seed, track_lwf = args

    import types as _types

    import jax.numpy as jnp
    from CybORG.Simulator.Actions import Impact as _Impact
    from CybORG.Simulator.Actions.GreenActions import GreenAccessService as _GAS
    from CybORG.Simulator.Actions.GreenActions import GreenLocalWork as _GLW

    from jaxborg.actions.encoding import BLUE_SLEEP
    from jaxborg.constants import NUM_BLUE_AGENTS
    from tests.differential.harness import CC4DifferentialHarness

    harness = CC4DifferentialHarness(
        seed=seed,
        max_steps=EPISODE_LENGTH,
        sync_green_rng=True,
        strict_random_sync=False,
        check_rewards=True,
        check_obs=False,
        check_masks=False,
    )
    harness.reset()

    # Per-step CybORG LWF failure counter, installed on BRM (after reset so
    # that green_recorder's execute_action wrapper is already in place).
    cy_lwf_per_step: list[int] = []
    if track_lwf:
        ec = harness.cyborg_env.environment_controller
        brm = ec.team_reward_calculators["Blue"]["BlueRewardMachine"]
        cy_lwf_counter = {"n": 0}

        def _counting_calculate(self, current_state, action_dict, agent_observations, done, state):
            self.phase_rewards = self.get_phase_rewards(state.mission_phase)
            total = 0.0
            lwf_n = 0
            for agent_name, action in action_dict.items():
                if not action:
                    continue
                act = action[0]
                if isinstance(act, _Impact):
                    hostname = act.hostname
                elif isinstance(act, (_GAS, _GLW)):
                    hostname = state.ip_addresses[act.ip_address]
                else:
                    continue
                subnet_name = state.hostname_subnet_map[hostname].value
                sessions = state.sessions[agent_name].values()
                if len([s.ident for s in sessions if s.active]) > 0:
                    success = agent_observations[agent_name].observations[0].data["success"]
                    rz = self.phase_rewards[subnet_name]
                    if "green" in agent_name and success == False:  # noqa: E712
                        if isinstance(act, _GLW):
                            total += rz["LWF"]
                            lwf_n += 1
                        elif isinstance(act, _GAS):
                            total += rz["ASF"]
                    elif "red" in agent_name and success and isinstance(act, _Impact):
                        total += rz["RIA"]
            cy_lwf_counter["n"] = lwf_n
            return total

        brm.calculate_reward = _types.MethodType(_counting_calculate, brm)
    else:
        cy_lwf_counter = None

    sleep_actions = {b: BLUE_SLEEP for b in range(NUM_BLUE_AGENTS)}
    per_step_jax_reward: list[float] = []
    per_step_cy_reward: list[float] = []
    jax_lwf_per_step: list[int] = []
    n_reward_diffs = 0
    worst_reward_diff = 0.0
    for _ in range(EPISODE_LENGTH):
        result = harness.full_step(sleep_actions)
        jr = float(result.jax_rewards["total"])
        cr = float(result.cyborg_rewards["total"])
        per_step_jax_reward.append(jr)
        per_step_cy_reward.append(cr)
        if abs(jr - cr) > 1e-6:
            n_reward_diffs += 1
            worst_reward_diff = max(worst_reward_diff, abs(jr - cr))
        if track_lwf:
            jax_lwf_per_step.append(int(harness.jax_state.green_lwf_this_step.sum()))
            cy_lwf_per_step.append(int(cy_lwf_counter["n"]))

    jx_total = float(sum(per_step_jax_reward))
    cy_total = float(sum(per_step_cy_reward))
    result_dict = {
        "seed": int(seed),
        "cyborg_total": cy_total,
        "jax_total": jx_total,
        "diff": jx_total - cy_total,
        "n_step_reward_diffs": n_reward_diffs,
        "worst_step_reward_diff": worst_reward_diff,
    }
    if track_lwf:
        jax_arr = jnp.array(jax_lwf_per_step)
        cy_arr = jnp.array(cy_lwf_per_step)
        diff_arr = jax_arr - cy_arr
        result_dict["lwf_total_jax"] = int(jax_arr.sum())
        result_dict["lwf_total_cyborg"] = int(cy_arr.sum())
        result_dict["lwf_steps_diff"] = int(jnp.sum(diff_arr != 0))
        result_dict["lwf_max_abs_step_diff"] = int(jnp.max(jnp.abs(diff_arr))) if diff_arr.size else 0
    return result_dict


def _run_one_pair(args: tuple) -> dict:
    """Worker: run paired sleep episode for one seed. Returns per-seed result.

    Each worker reimports JAX/CybORG to keep ProcessPoolExecutor (spawn) clean.
    """
    seed, track_components = args

    import types as _types

    import jax
    import jax.numpy as jnp
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Agents.Wrappers import BlueFlatWrapper
    from CybORG.Simulator.Actions.AbstractActions.Impact import Impact as _Impact
    from CybORG.Simulator.Actions.GreenActions import GreenAccessService as _GAS
    from CybORG.Simulator.Actions.GreenActions import GreenLocalWork as _GLW
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    from jaxborg.constants import NUM_BLUE_AGENTS
    from jaxborg.env import CC4EnvState, _init_red_state
    from jaxborg.fsm_red_env import FsmRedCC4Env
    from jaxborg.state import create_initial_state
    from jaxborg.topology import build_const_from_cyborg

    # --- CybORG side ---
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=EPISODE_LENGTH,
    )
    cyborg = CybORG(sg, "sim", seed=seed)
    cy_env = BlueFlatWrapper(env=cyborg)
    cy_env.reset()

    # Extract topology BEFORE running the CybORG episode — const is static, but
    # extracting from a clean post-reset env is the cleanest pattern.
    const = build_const_from_cyborg(cyborg)

    # Optional per-component tracker (mirrors transfer.py:761-794 monkeypatch).
    cy_components = {"ria": 0.0, "lwf": 0.0, "asf": 0.0}
    if track_components:
        ec = cyborg.environment_controller
        brm = ec.team_reward_calculators["Blue"]["BlueRewardMachine"]

        def _tracked_calculate(self, current_state, action_dict, agent_observations, done, state):
            self.phase_rewards = self.get_phase_rewards(state.mission_phase)
            reward_list = []
            for agent_name, action in action_dict.items():
                if not action:
                    continue
                act = action[0]
                if isinstance(act, _Impact):
                    hostname = act.hostname
                elif isinstance(act, (_GAS, _GLW)):
                    hostname = state.ip_addresses[act.ip_address]
                else:
                    continue
                subnet_name = state.hostname_subnet_map[hostname].value
                sessions = state.sessions[agent_name].values()
                if len([s.ident for s in sessions if s.active]) > 0:
                    success = agent_observations[agent_name].observations[0].data["success"]
                    rz = self.phase_rewards[subnet_name]
                    if "green" in agent_name and success == False:  # noqa: E712 -- TernaryEnum compat
                        if isinstance(act, _GLW):
                            r = rz["LWF"]
                            reward_list.append(r)
                            cy_components["lwf"] += r
                        elif isinstance(act, _GAS):
                            r = rz["ASF"]
                            reward_list.append(r)
                            cy_components["asf"] += r
                    elif "red" in agent_name and success and isinstance(act, _Impact):
                        r = rz["RIA"]
                        reward_list.append(r)
                        cy_components["ria"] += r
            return sum(reward_list)

        brm.calculate_reward = _types.MethodType(_tracked_calculate, brm)

    cy_actions = {agent: 0 for agent in cy_env.agents}
    cy_total = 0.0
    for _ in range(EPISODE_LENGTH):
        _, rewards, _, _, _ = cy_env.step(cy_actions)
        cy_total += float(mean(rewards.values()))

    # --- JAX side: build env_state from extracted const, run sleep ---
    state = create_initial_state()
    state = state.replace(
        host_services=jnp.array(const.initial_services),
        host_max_pid=const.host_initial_max_pid,
    )
    state = _init_red_state(const, state)
    env_state = CC4EnvState(state=state, const=const)

    jx_env = FsmRedCC4Env(num_steps=EPISODE_LENGTH)
    sleep_actions = {f"blue_{b}": jnp.int32(0) for b in range(NUM_BLUE_AGENTS)}
    key = jax.random.PRNGKey(seed)
    jx_total = 0.0
    jx_components = {"ria": 0.0, "lwf": 0.0, "asf": 0.0, "action_cost": 0.0}
    for _ in range(EPISODE_LENGTH):
        key, subkey = jax.random.split(key)
        _, env_state, rewards, _, info = jx_env.step_env(subkey, env_state, sleep_actions)
        jx_total += float(rewards["blue_0"])
        if track_components:
            jx_components["ria"] += float(info["reward_ria"])
            jx_components["lwf"] += float(info["reward_lwf"])
            jx_components["asf"] += float(info["reward_asf"])
            jx_components["action_cost"] += float(info["action_cost"])

    result = {
        "seed": int(seed),
        "cyborg_total": cy_total,
        "jax_total": jx_total,
        "diff": jx_total - cy_total,
    }
    if track_components:
        result["cyborg_components"] = cy_components
        result["jax_components"] = jx_components
    return result


def _tost_paired(diffs: np.ndarray, margin: float, alpha: float = 0.05) -> dict:
    from scipy import stats

    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    se = float(np.std(diffs, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    df = n - 1

    if se < 1e-12:
        return {
            "equivalent": True,
            "p_upper": 0.0,
            "p_lower": 0.0,
            "mean_diff": mean_diff,
            "margin": margin,
            "ci_lower": mean_diff,
            "ci_upper": mean_diff,
            "n": n,
        }

    t_upper = (mean_diff - margin) / se
    p_upper = float(stats.t.cdf(t_upper, df))
    t_lower = (mean_diff + margin) / se
    p_lower = float(1.0 - stats.t.cdf(t_lower, df))
    t_crit = float(stats.t.ppf(1 - alpha, df))
    return {
        "equivalent": p_upper < alpha and p_lower < alpha,
        "p_upper": p_upper,
        "p_lower": p_lower,
        "mean_diff": mean_diff,
        "margin": margin,
        "ci_lower": mean_diff - t_crit * se,
        "ci_upper": mean_diff + t_crit * se,
        "n": n,
    }


def main():
    parser = argparse.ArgumentParser(description="Paired sleep baseline (CybORG vs JAXborg, same topology)")
    parser.add_argument("--seed-start", type=int, default=42, help="First seed (default 42)")
    parser.add_argument("--seeds", type=int, default=100, help="Number of seeds (default 100)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers (default 8)")
    parser.add_argument("--margin", type=float, default=200.0, help="TOST margin Δ (default 200)")
    parser.add_argument("--output-json", default=None, help="Write per-seed + summary to JSON")
    parser.add_argument(
        "--track-components",
        action="store_true",
        help="Track per-component (RIA/LWF/ASF, JAX action_cost) totals to localize the gap",
    )
    parser.add_argument(
        "--sync-green-rng",
        action="store_true",
        help=(
            "Use the differential harness to drive both backends with matched green RNG "
            "(green_recorder replays CybORG's per-step np_random calls into JAX's "
            "precomputed green_randoms buffer). Any residual paired gap under this "
            "flag is a real env-mechanism divergence, not RNG-stream bias."
        ),
    )
    parser.add_argument(
        "--track-lwf",
        action="store_true",
        help="With --sync-green-rng, additionally log per-step LWF failure counts on both backends.",
    )
    args = parser.parse_args()

    if args.track_components and args.sync_green_rng:
        raise SystemExit("--track-components is incompatible with --sync-green-rng (harness path)")
    if args.track_lwf and not args.sync_green_rng:
        raise SystemExit("--track-lwf requires --sync-green-rng")

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    print(f"Running paired sleep baseline: {len(seeds)} seeds × {EPISODE_LENGTH} steps × 2 backends")
    print(
        f"Seeds: [{seeds[0]}..{seeds[-1]}], workers: {args.workers}, "
        f"track_components={args.track_components}, sync_green_rng={args.sync_green_rng}, "
        f"track_lwf={args.track_lwf}"
    )

    if args.sync_green_rng:
        runner = _run_one_pair_synced
        worker_args = [(s, args.track_lwf) for s in seeds]
    else:
        runner = _run_one_pair
        worker_args = [(s, args.track_components) for s in seeds]

    results: list[dict] = []
    if args.workers <= 1:
        for wa in worker_args:
            r = runner(wa)
            results.append(r)
            print(
                f"  seed={r['seed']:3d}  cyborg={r['cyborg_total']:9.1f}  "
                f"jax={r['jax_total']:9.1f}  diff={r['diff']:+8.1f}"
            )
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            futures = {pool.submit(runner, wa): wa[0] for wa in worker_args}
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                print(
                    f"  seed={r['seed']:3d}  cyborg={r['cyborg_total']:9.1f}  "
                    f"jax={r['jax_total']:9.1f}  diff={r['diff']:+8.1f}"
                )

    results.sort(key=lambda r: r["seed"])
    diffs = np.array([r["diff"] for r in results], dtype=float)
    cy_arr = np.array([r["cyborg_total"] for r in results], dtype=float)
    jx_arr = np.array([r["jax_total"] for r in results], dtype=float)

    tost = _tost_paired(diffs, args.margin)

    print()
    print("=" * 70)
    print(f"PAIRED SLEEP BASELINE — n={len(diffs)}")
    print("=" * 70)
    print(f"CybORG  mean: {cy_arr.mean():9.2f} ± {cy_arr.std(ddof=1):8.2f}")
    print(f"JAXborg mean: {jx_arr.mean():9.2f} ± {jx_arr.std(ddof=1):8.2f}")
    print(f"diff (J-C):  mean={tost['mean_diff']:+8.2f}  stdev={diffs.std(ddof=1):.2f}")
    print(f"             90% CI=[{tost['ci_lower']:+.2f}, {tost['ci_upper']:+.2f}]  (alpha=0.05, 1-sided)")
    if args.sync_green_rng:
        total_step_reward_diffs = int(sum(r.get("n_step_reward_diffs", 0) for r in results))
        worst_step = max((r.get("worst_step_reward_diff", 0.0) for r in results), default=0.0)
        print(
            f"Per-step reward diffs across {len(diffs)} eps × {EPISODE_LENGTH} steps: "
            f"{total_step_reward_diffs} (worst |Δ| = {worst_step:.2e})"
        )
        if args.track_lwf:
            lwf_jax = int(sum(r.get("lwf_total_jax", 0) for r in results))
            lwf_cy = int(sum(r.get("lwf_total_cyborg", 0) for r in results))
            lwf_steps_diff = int(sum(r.get("lwf_steps_diff", 0) for r in results))
            lwf_max_abs = max((r.get("lwf_max_abs_step_diff", 0) for r in results), default=0)
            print(
                f"LWF event totals (across all eps): JAX={lwf_jax}  CybORG={lwf_cy}  "
                f"Δ={lwf_jax - lwf_cy}    per-step diff count={lwf_steps_diff}  max |Δ_step|={lwf_max_abs}"
            )
    print(
        f"TOST Δ=±{tost['margin']:.0f}: {'PASS' if tost['equivalent'] else 'FAIL'}  "
        f"p_upper={tost['p_upper']:.4f}  p_lower={tost['p_lower']:.4f}"
    )

    component_summary = None
    if args.track_components and results and "cyborg_components" in results[0]:
        components = ("ria", "lwf", "asf")
        cy_means = {c: float(np.mean([r["cyborg_components"][c] for r in results])) for c in components}
        jx_means = {c: float(np.mean([r["jax_components"][c] for r in results])) for c in components}
        diff_arrays = {
            c: np.array([r["jax_components"][c] - r["cyborg_components"][c] for r in results], dtype=float)
            for c in components
        }
        jx_ac_mean = float(np.mean([r["jax_components"]["action_cost"] for r in results]))

        print()
        print("Per-component breakdown (paired diffs, mean ± stdev across n eps):")
        print(f"  {'comp':<6}  {'CybORG':>10}  {'JAXborg':>10}  {'diff (J-C)':>12}  {'stdev':>9}  {'%total':>7}")
        total_gap = tost["mean_diff"]
        component_summary = {}
        for c in components:
            d = diff_arrays[c]
            mean_d = float(d.mean())
            std_d = float(d.std(ddof=1))
            pct = 100.0 * mean_d / total_gap if abs(total_gap) > 1e-9 else 0.0
            print(
                f"  {c.upper():<6}  {cy_means[c]:>10.2f}  {jx_means[c]:>10.2f}  "
                f"{mean_d:>+12.2f}  {std_d:>9.2f}  {pct:>6.1f}%"
            )
            component_summary[c] = {
                "cyborg_mean": cy_means[c],
                "jax_mean": jx_means[c],
                "diff_mean": mean_d,
                "diff_std": std_d,
                "pct_of_total_gap": pct,
            }
        # JAX action_cost has no CybORG counterpart in sleep (no Restore submitted)
        print(f"  AC    (JAX): {jx_ac_mean:>10.2f}  (CybORG action_cost not tracked here; sleep submits no Restore)")
        component_summary["jax_action_cost_mean"] = jx_ac_mean

    payload = {
        "n": len(diffs),
        "seed_start": args.seed_start,
        "cyborg_mean": float(cy_arr.mean()),
        "cyborg_std": float(cy_arr.std(ddof=1)),
        "jax_mean": float(jx_arr.mean()),
        "jax_std": float(jx_arr.std(ddof=1)),
        "diff_mean": tost["mean_diff"],
        "diff_std": float(diffs.std(ddof=1)),
        "diff_ci_lower": tost["ci_lower"],
        "diff_ci_upper": tost["ci_upper"],
        "tost_margin": tost["margin"],
        "tost_equivalent": tost["equivalent"],
        "tost_p_upper": tost["p_upper"],
        "tost_p_lower": tost["p_lower"],
        "per_seed": results,
        "components": component_summary,
    }
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote: {out}")


if __name__ == "__main__":
    main()
