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
    seed, track_lwf, random_blue = (
        args if len(args) == 3 else (args[0], args[1], False)
    )

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

    # Per-step CybORG counters for all three event components + reward breakdown.
    # Installed on BRM after reset so green_recorder's execute_action wrapper is
    # already in place.
    cy_lwf_per_step: list[int] = []
    cy_asf_per_step: list[int] = []
    cy_ria_per_step: list[int] = []
    cy_lwf_reward_per_step: list[float] = []
    cy_asf_reward_per_step: list[float] = []
    cy_ria_reward_per_step: list[float] = []
    if track_lwf:
        ec = harness.cyborg_env.environment_controller
        brm = ec.team_reward_calculators["Blue"]["BlueRewardMachine"]
        cy_cnt = {"lwf_n": 0, "asf_n": 0, "ria_n": 0, "lwf_r": 0.0, "asf_r": 0.0, "ria_r": 0.0}

        def _counting_calculate(self, current_state, action_dict, agent_observations, done, state):
            self.phase_rewards = self.get_phase_rewards(state.mission_phase)
            total = 0.0
            lwf_n = 0
            asf_n = 0
            ria_n = 0
            lwf_r = 0.0
            asf_r = 0.0
            ria_r = 0.0
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
                            r = rz["LWF"]
                            total += r
                            lwf_n += 1
                            lwf_r += r
                        elif isinstance(act, _GAS):
                            r = rz["ASF"]
                            total += r
                            asf_n += 1
                            asf_r += r
                    elif "red" in agent_name and success and isinstance(act, _Impact):
                        r = rz["RIA"]
                        total += r
                        ria_n += 1
                        ria_r += r
            cy_cnt["lwf_n"] = lwf_n
            cy_cnt["asf_n"] = asf_n
            cy_cnt["ria_n"] = ria_n
            cy_cnt["lwf_r"] = lwf_r
            cy_cnt["asf_r"] = asf_r
            cy_cnt["ria_r"] = ria_r
            return total

        brm.calculate_reward = _types.MethodType(_counting_calculate, brm)
    else:
        cy_cnt = None

    from jaxborg.actions.action_defs import BLUE_ALLOW_TRAFFIC_END
    from jaxborg.actions.masking import compute_blue_action_mask

    sleep_actions = {b: BLUE_SLEEP for b in range(NUM_BLUE_AGENTS)}
    import numpy as _np_for_blue

    rng_blue = _np_for_blue.random.default_rng(int(seed) * 1_000_003 + 17)
    per_step_jax_reward: list[float] = []
    per_step_cy_reward: list[float] = []
    jax_lwf_per_step: list[int] = []
    jax_asf_per_step: list[int] = []
    jax_ria_per_step: list[int] = []
    n_reward_diffs = 0
    worst_reward_diff = 0.0
    for _ in range(EPISODE_LENGTH):
        if random_blue:
            # Mask-aware uniform blue: compute JAX mask from current (matched)
            # state, pick uniformly over the True entries per blue agent. Since
            # state is matched between backends under sync_green_rng, the same
            # mask is valid on CybORG; actions the mask allows will not be
            # replaced with Sleep on either side.
            step_actions = {}
            for b in range(NUM_BLUE_AGENTS):
                mask_b = compute_blue_action_mask(harness.jax_const, b, harness.jax_state)
                mask_np = _np_for_blue.asarray(mask_b, dtype=bool)
                valid_indices = _np_for_blue.flatnonzero(mask_np)
                if valid_indices.size == 0:
                    step_actions[b] = int(BLUE_SLEEP)
                else:
                    step_actions[b] = int(rng_blue.choice(valid_indices))
            result = harness.full_step(step_actions)
        else:
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
            jax_asf_per_step.append(int(harness.jax_state.green_asf_this_step.sum()))
            jax_ria_per_step.append(int(harness.jax_state.red_impact_attempted.sum()))
            cy_lwf_per_step.append(int(cy_cnt["lwf_n"]))
            cy_asf_per_step.append(int(cy_cnt["asf_n"]))
            cy_ria_per_step.append(int(cy_cnt["ria_n"]))
            cy_lwf_reward_per_step.append(float(cy_cnt["lwf_r"]))
            cy_asf_reward_per_step.append(float(cy_cnt["asf_r"]))
            cy_ria_reward_per_step.append(float(cy_cnt["ria_r"]))

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
        import numpy as _np

        def _step_stats(jx, cy):
            jx_a = _np.array(jx, dtype=_np.int64)
            cy_a = _np.array(cy, dtype=_np.int64)
            d = jx_a - cy_a
            return {
                "total_jax": int(jx_a.sum()),
                "total_cyborg": int(cy_a.sum()),
                "steps_diff": int((d != 0).sum()),
                "max_abs_step_diff": int(_np.max(_np.abs(d))) if d.size else 0,
            }

        result_dict["lwf"] = _step_stats(jax_lwf_per_step, cy_lwf_per_step)
        result_dict["asf"] = _step_stats(jax_asf_per_step, cy_asf_per_step)
        # RIA: JAX counts red impact *attempts*; CybORG counts successful impacts
        # that hit the reward. We still compare per-step totals and call out
        # any divergence for manual inspection.
        result_dict["ria"] = _step_stats(jax_ria_per_step, cy_ria_per_step)
        result_dict["cy_component_reward_totals"] = {
            "lwf": float(sum(cy_lwf_reward_per_step)),
            "asf": float(sum(cy_asf_reward_per_step)),
            "ria": float(sum(cy_ria_reward_per_step)),
        }
        # Legacy flat keys (kept for backward-compat with A1-fup² checkpoints).
        result_dict["lwf_total_jax"] = result_dict["lwf"]["total_jax"]
        result_dict["lwf_total_cyborg"] = result_dict["lwf"]["total_cyborg"]
        result_dict["lwf_steps_diff"] = result_dict["lwf"]["steps_diff"]
        result_dict["lwf_max_abs_step_diff"] = result_dict["lwf"]["max_abs_step_diff"]
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


ALT_SEED_OFFSET = 1_000_000


def _jax_sleep_total(const, seed: int) -> tuple[float, dict]:
    """One JAX sleep episode over a fixed `const`, driven by jax.random.PRNGKey(seed).

    Returns (total_reward, components) where components has ria/lwf/asf/action_cost totals.
    """
    import jax
    import jax.numpy as jnp

    from jaxborg.constants import NUM_BLUE_AGENTS
    from jaxborg.env import CC4EnvState, _init_red_state
    from jaxborg.fsm_red_env import FsmRedCC4Env
    from jaxborg.state import create_initial_state

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
    total = 0.0
    comps = {"ria": 0.0, "lwf": 0.0, "asf": 0.0, "action_cost": 0.0}
    for _ in range(EPISODE_LENGTH):
        key, subkey = jax.random.split(key)
        _, env_state, rewards, _, info = jx_env.step_env(subkey, env_state, sleep_actions)
        total += float(rewards["blue_0"])
        comps["ria"] += float(info["reward_ria"])
        comps["lwf"] += float(info["reward_lwf"])
        comps["asf"] += float(info["reward_asf"])
        comps["action_cost"] += float(info["action_cost"])
    return total, comps


def _build_cyborg_sleep_env(topology_seed: int):
    """Return (cyborg, wrapped_env) built with the given topology seed. Reset-complete."""
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Agents.Wrappers import BlueFlatWrapper
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=EPISODE_LENGTH,
    )
    cyborg = CybORG(sg, "sim", seed=topology_seed)
    cy_env = BlueFlatWrapper(env=cyborg)
    cy_env.reset()
    return cyborg, cy_env


def _cyborg_sleep_total(cy_env, cy_components: dict | None = None) -> float:
    """Run a sleep episode on an already-reset CybORG env. Returns total reward.

    If `cy_components` is not None, it should be the mutable dict returned by
    `_install_brm_component_tracker(cyborg)`. Components accumulate in place.
    """
    from statistics import mean as _mean

    cy_actions = {agent: 0 for agent in cy_env.agents}
    total = 0.0
    for _ in range(EPISODE_LENGTH):
        _, rewards, _, _, _ = cy_env.step(cy_actions)
        total += float(_mean(rewards.values()))
    return total


def _reseed_cyborg_agents_and_state(cyborg, new_seed: int) -> None:
    """Swap controller+state np_random (via set_seed) AND cascade to agent np_randoms.

    `set_seed` reseats `controller.np_random` and `state.np_random` but does NOT touch
    the agents' own `np_random` refs — those were bound at `_create_agents` time to
    the original generator. Cascade manually so the entire CybORG side is driven by
    the new stream for this episode.
    """
    cyborg.set_seed(new_seed)
    for ai in cyborg.environment_controller.agent_interfaces.values():
        ai.agent.np_random = cyborg.np_random


def _run_pair_jax_jax(args: tuple) -> dict:
    """Same topology (from CybORG seed=s); two JAX episodes at PRNGKey(s) and PRNGKey(s+1M).

    Both episodes track per-component rewards; we report diff_B - diff_A per component.
    """
    (seed,) = args
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    from jaxborg.topology import build_const_from_cyborg

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=EPISODE_LENGTH,
    )
    cyborg = CybORG(sg, "sim", seed=seed)
    cyborg.reset()
    const = build_const_from_cyborg(cyborg)

    r_a, comps_a = _jax_sleep_total(const, seed)
    r_b, comps_b = _jax_sleep_total(const, seed + ALT_SEED_OFFSET)
    # Use "r_a" = first episode (CybORG-equivalent), "r_b" = second (JAX-equivalent).
    # For jax-jax the labelling is cosmetic; what matters is diff = b - a.
    return {
        "seed": int(seed),
        "r_a": r_a,
        "r_b": r_b,
        "diff": r_b - r_a,
        # Cross-stream schema: "cyborg_components" (A) / "jax_components" (B).
        "cyborg_components": {k: comps_a[k] for k in ("ria", "lwf", "asf")},
        "jax_components": comps_b,
    }


def _run_pair_cyborg_cyborg(args: tuple) -> dict:
    """Same topology (CybORG seed=s, built twice); episode B reseeded to s+1M.

    Both runs install the BRM component tracker for per-component RIA/LWF/ASF totals.
    """
    (seed,) = args

    cyborg_a, cy_a = _build_cyborg_sleep_env(seed)
    hosts_a = tuple(sorted(cyborg_a.environment_controller.state.hosts.keys()))
    comps_a = _install_brm_component_tracker(cyborg_a)
    r_a = _cyborg_sleep_total(cy_a)

    cyborg_b, cy_b = _build_cyborg_sleep_env(seed)
    hosts_b = tuple(sorted(cyborg_b.environment_controller.state.hosts.keys()))
    if hosts_a != hosts_b:
        raise RuntimeError(
            f"cyborg-cyborg: topology mismatch at seed={seed} (|A|={len(hosts_a)}, |B|={len(hosts_b)})"
        )
    _reseed_cyborg_agents_and_state(cyborg_b, seed + ALT_SEED_OFFSET)
    comps_b = _install_brm_component_tracker(cyborg_b)
    r_b = _cyborg_sleep_total(cy_b)

    return {
        "seed": int(seed),
        "r_a": r_a,
        "r_b": r_b,
        "diff": r_b - r_a,
        # Cross-stream schema: A = "cyborg", B = "jax" (labels are cosmetic here).
        "cyborg_components": dict(comps_a),
        "jax_components": {**dict(comps_b), "action_cost": 0.0},
    }


def _run_pair_cross(args: tuple) -> dict:
    """Same topology; JAX (PRNGKey(s)) vs CybORG (seed=s). Diff = JAX - CybORG."""
    r = _run_one_pair((args[0], True))
    out = {
        "seed": r["seed"],
        "r_a": r["cyborg_total"],
        "r_b": r["jax_total"],
        "diff": r["diff"],
    }
    if "cyborg_components" in r:
        out["cyborg_components"] = r["cyborg_components"]
        out["jax_components"] = r["jax_components"]
    return out


def _install_brm_component_tracker(cyborg):
    """Install a BRM monkeypatch that accumulates per-episode RIA/LWF/ASF totals.

    Returns the mutable dict with keys 'ria','lwf','asf' updated by the step hook.
    """
    import types as _types

    from CybORG.Simulator.Actions.AbstractActions.Impact import Impact as _Impact
    from CybORG.Simulator.Actions.GreenActions import GreenAccessService as _GAS
    from CybORG.Simulator.Actions.GreenActions import GreenLocalWork as _GLW

    ec = cyborg.environment_controller
    brm = ec.team_reward_calculators["Blue"]["BlueRewardMachine"]
    comp = {"ria": 0.0, "lwf": 0.0, "asf": 0.0}

    def _tracked_calculate(self, current_state, action_dict, agent_observations, done, state):
        self.phase_rewards = self.get_phase_rewards(state.mission_phase)
        total = 0.0
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
                        r = rz["LWF"]; total += r; comp["lwf"] += r
                    elif isinstance(act, _GAS):
                        r = rz["ASF"]; total += r; comp["asf"] += r
                elif "red" in agent_name and success and isinstance(act, _Impact):
                    r = rz["RIA"]; total += r; comp["ria"] += r
        return total

    brm.calculate_reward = _types.MethodType(_tracked_calculate, brm)
    return comp


def _run_pair_cross_random_blue(args: tuple) -> dict:
    """Same topology, UNSYNCED RNG, mask-aware RANDOM blue on both backends.

    CybORG side: pick uniformly from the BlueFlatWrapper `action_mask` at each step.
    JAX side:    pick uniformly from `compute_blue_action_mask(const, agent, state)`.

    Both backends use the same seeded np.random generator for blue choice, but
    since states diverge under cross-stream, the masks (and thus selected
    actions) diverge too — realistic cross-stream behavior. Exercises Restore/
    Remove/BlockTraffic to stress-test phantom accumulation and PID-bound paths.

    Returns per-component reward breakdown (RIA/LWF/ASF) on both backends plus
    JAX action_cost (CybORG action_cost is folded into the caller-submission
    accounting and tracked separately by reward_list; we report totals only).
    """
    (seed,) = args
    from statistics import mean as _mean

    import jax
    import jax.numpy as jnp
    import numpy as _np
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Agents.Wrappers import BlueFlatWrapper
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    from jaxborg.actions.action_defs import BLUE_SLEEP
    from jaxborg.actions.masking import compute_blue_action_mask
    from jaxborg.constants import NUM_BLUE_AGENTS
    from jaxborg.env import CC4EnvState, _init_red_state
    from jaxborg.fsm_red_env import FsmRedCC4Env
    from jaxborg.state import create_initial_state
    from jaxborg.topology import build_const_from_cyborg

    # --- CybORG side: seed=s, mask-aware random blue ---
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=EPISODE_LENGTH,
    )
    cyborg = CybORG(sg, "sim", seed=seed)
    cy_env = BlueFlatWrapper(env=cyborg)
    cy_env.reset()
    const = build_const_from_cyborg(cyborg)
    cy_components = _install_brm_component_tracker(cyborg)

    rng_cy = _np.random.default_rng(int(seed) * 1_000_003 + 17)
    cy_total = 0.0
    for _ in range(EPISODE_LENGTH):
        actions = {}
        for agent in cy_env.agents:
            mask = cy_env.action_mask(agent)
            valid = _np.flatnonzero(_np.asarray(mask, dtype=bool))
            if valid.size == 0:
                actions[agent] = 0
            else:
                actions[agent] = int(rng_cy.choice(valid))
        _, rewards, _, _, _ = cy_env.step(actions)
        cy_total += float(_mean(rewards.values()))

    # --- JAX side: PRNGKey(s), mask-aware random blue (independent rng_jx) ---
    state = create_initial_state()
    state = state.replace(
        host_services=jnp.array(const.initial_services),
        host_max_pid=const.host_initial_max_pid,
    )
    state = _init_red_state(const, state)
    env_state = CC4EnvState(state=state, const=const)
    jx_env = FsmRedCC4Env(num_steps=EPISODE_LENGTH)

    rng_jx = _np.random.default_rng(int(seed) * 1_000_003 + 17)
    key = jax.random.PRNGKey(seed)
    jx_total = 0.0
    jx_components = {"ria": 0.0, "lwf": 0.0, "asf": 0.0, "action_cost": 0.0}
    for _ in range(EPISODE_LENGTH):
        step_actions = {}
        for b in range(NUM_BLUE_AGENTS):
            mask_b = compute_blue_action_mask(env_state.const, b, env_state.state)
            mask_np = _np.asarray(mask_b, dtype=bool)
            valid = _np.flatnonzero(mask_np)
            if valid.size == 0:
                step_actions[f"blue_{b}"] = jnp.int32(BLUE_SLEEP)
            else:
                step_actions[f"blue_{b}"] = jnp.int32(int(rng_jx.choice(valid)))
        key, subkey = jax.random.split(key)
        _, env_state, rewards, _, info = jx_env.step_env(subkey, env_state, step_actions)
        jx_total += float(rewards["blue_0"])
        jx_components["ria"] += float(info["reward_ria"])
        jx_components["lwf"] += float(info["reward_lwf"])
        jx_components["asf"] += float(info["reward_asf"])
        jx_components["action_cost"] += float(info["action_cost"])

    return {
        "seed": int(seed),
        "r_a": cy_total,
        "r_b": jx_total,
        "diff": jx_total - cy_total,
        "cyborg_components": dict(cy_components),
        "jax_components": jx_components,
    }


def _summarize_mode(label: str, results: list[dict]) -> dict:
    """Per-mode stats: mean, stdev, 90% CI, sign-test, Wilcoxon, one-sample t."""
    from scipy import stats

    diffs = np.array([r["diff"] for r in results], dtype=float)
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    se = std_d / np.sqrt(n) if n > 1 else 0.0
    t_crit_90 = float(stats.t.ppf(0.95, n - 1)) if n > 1 else 0.0
    ci_lo = mean_d - t_crit_90 * se
    ci_hi = mean_d + t_crit_90 * se

    pos = int((diffs > 0).sum())
    neg = int((diffs < 0).sum())
    sign_p = float(stats.binomtest(pos, pos + neg, 0.5).pvalue) if pos + neg > 0 else 1.0

    if std_d > 0 and n > 1:
        wilcoxon = stats.wilcoxon(diffs, zero_method="zsplit")
        wilcoxon_p = float(wilcoxon.pvalue)
        t_res = stats.ttest_1samp(diffs, 0.0)
        t_p = float(t_res.pvalue)
    else:
        wilcoxon_p = 1.0
        t_p = 1.0

    return {
        "label": label,
        "n": n,
        "diffs": diffs,
        "mean": mean_d,
        "std": std_d,
        "se": se,
        "ci90": (ci_lo, ci_hi),
        "pos": pos,
        "neg": neg,
        "sign_p": sign_p,
        "wilcoxon_p": wilcoxon_p,
        "t_p": t_p,
    }


def _write_mode_checkpoint(
    path: Path, label: str, seeds: list[int], results: list[dict]
) -> None:
    """Write a per-mode checkpoint with current accumulated results + summary."""
    summ = _summarize_mode(label, results)
    lo, hi = summ["ci90"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "mode": label,
                "seed_start": seeds[0],
                "n": len(seeds),
                "n_completed": len(results),
                "mean": summ["mean"],
                "std": summ["std"],
                "se": summ["se"],
                "ci90_lower": lo,
                "ci90_upper": hi,
                "pos": summ["pos"],
                "neg": summ["neg"],
                "sign_p": summ["sign_p"],
                "wilcoxon_p": summ["wilcoxon_p"],
                "t_p": summ["t_p"],
                "per_seed": results,
            },
            indent=2,
        )
        + "\n"
    )


def _run_rng_comparison_mode(
    label: str,
    runner,
    seeds: list[int],
    workers: int,
    chunk_tasks_per_worker: int = 25,
    checkpoint_path: Path | None = None,
    resume: bool = False,
) -> list[dict]:
    """Run one comparison mode with a fresh ProcessPoolExecutor per chunk.

    Each chunk processes `workers * chunk_tasks_per_worker` seeds, then the pool
    is torn down (freeing all JAX/CybORG state held in workers) before the next
    chunk's pool is created. This bounds memory and — unlike Python 3.11's
    `max_tasks_per_child` — does not deadlock after the first recycle.

    If `checkpoint_path` is set, the current accumulated results are written to
    that file after every chunk. With `resume=True`, the function picks up from
    whatever seeds the checkpoint already contains (within the same seed range).
    """
    done_seeds: set[int] = set()
    results: list[dict] = []
    if resume and checkpoint_path is not None and checkpoint_path.exists():
        try:
            data = json.loads(checkpoint_path.read_text())
            if data.get("mode") == label and data.get("seed_start") == seeds[0]:
                for r in data.get("per_seed", []):
                    if r["seed"] in seeds:
                        results.append(r)
                        done_seeds.add(r["seed"])
                if results:
                    print(
                        f"  [{label}] resumed from {checkpoint_path.name} "
                        f"({len(results)}/{len(seeds)} already done)",
                        flush=True,
                    )
        except (json.JSONDecodeError, KeyError):
            print(f"  [{label}] ignoring unreadable checkpoint {checkpoint_path}")

    remaining = [s for s in seeds if s not in done_seeds]

    if workers <= 1:
        for s in remaining:
            r = runner((s,))
            results.append(r)
            print(
                f"  [{label}] seed={r['seed']:4d}  r_a={r['r_a']:9.1f}  "
                f"r_b={r['r_b']:9.1f}  diff={r['diff']:+8.1f}",
                flush=True,
            )
            if checkpoint_path is not None and len(results) % 10 == 0:
                results.sort(key=lambda x: x["seed"])
                _write_mode_checkpoint(checkpoint_path, label, seeds, results)
    else:
        chunk_size = max(1, workers * chunk_tasks_per_worker)
        ctx = mp.get_context("spawn")
        for chunk_start in range(0, len(remaining), chunk_size):
            chunk_seeds = remaining[chunk_start : chunk_start + chunk_size]
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
                futures = {pool.submit(runner, (s,)): s for s in chunk_seeds}
                for fut in as_completed(futures):
                    r = fut.result()
                    results.append(r)
                    print(
                        f"  [{label}] seed={r['seed']:4d}  r_a={r['r_a']:9.1f}  "
                        f"r_b={r['r_b']:9.1f}  diff={r['diff']:+8.1f}",
                        flush=True,
                    )
            if checkpoint_path is not None:
                results.sort(key=lambda x: x["seed"])
                _write_mode_checkpoint(checkpoint_path, label, seeds, results)
                print(
                    f"  [{label}] chunk done; partial checkpoint: "
                    f"{len(results)}/{len(seeds)} → {checkpoint_path.name}",
                    flush=True,
                )

    results.sort(key=lambda r: r["seed"])
    return results


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


def _main_rng_comparison(args) -> None:
    """Three-mode stream-noise decomposition.

    - jax-jax:      same topo, two JAX PRNGKeys (within-JAX noise)
    - cyborg-cyborg: same topo, two np.random seeds (within-CybORG noise)
    - cross:        existing unsynced JAX↔CybORG paired path
    """
    from scipy import stats

    mode = args.rng_comparison
    modes = ("jax-jax", "cyborg-cyborg", "cross") if mode == "all" else (mode,)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    print(
        f"RNG-comparison modes: {list(modes)} | seeds: [{seeds[0]}..{seeds[-1]}] "
        f"(n={len(seeds)}) × {EPISODE_LENGTH} steps × 2 episodes | workers: {args.workers}"
    )

    cross_runner = _run_pair_cross_random_blue if args.random_blue else _run_pair_cross
    runners = {
        "jax-jax": _run_pair_jax_jax,
        "cyborg-cyborg": _run_pair_cyborg_cyborg,
        "cross": cross_runner,
    }
    mode_results: dict[str, list[dict]] = {}
    mode_summaries: dict[str, dict] = {}
    base_out = Path(args.output_json) if args.output_json else None
    for m in modes:
        print()
        print(f"--- mode={m} ---")

        per_mode_ckpt = (
            base_out.with_name(f"{base_out.stem}_{m}{base_out.suffix}") if base_out else None
        )
        results = _run_rng_comparison_mode(
            m,
            runners[m],
            seeds,
            args.workers,
            checkpoint_path=per_mode_ckpt,
            resume=args.resume,
        )
        mode_results[m] = results
        mode_summaries[m] = _summarize_mode(m, results)
        if per_mode_ckpt is not None:
            print(f"  [{m}] mode done; final checkpoint: {per_mode_ckpt.name}")

    print()
    print("=" * 78)
    print(f"RNG-COMPARISON SUMMARY — n={len(seeds)} per mode")
    print("=" * 78)
    print(
        f"  {'mode':<15}  {'mean':>9}  {'std':>9}  {'90% CI':>23}  "
        f"{'sign(+/-)':>11}  {'sign_p':>8}  {'wilcx_p':>8}  {'t_p':>8}"
    )
    for m in modes:
        s = mode_summaries[m]
        lo, hi = s["ci90"]
        print(
            f"  {m:<15}  {s['mean']:>+9.2f}  {s['std']:>9.2f}  "
            f"[{lo:>+8.2f}, {hi:>+8.2f}]  "
            f"{s['pos']:>4}/{s['neg']:<5}  {s['sign_p']:>8.4f}  "
            f"{s['wilcoxon_p']:>8.4f}  {s['t_p']:>8.4f}"
        )

    cross_stats: dict = {}
    verdict = None
    if len(modes) == 3:
        diffs_jj = mode_summaries["jax-jax"]["diffs"]
        diffs_cc = mode_summaries["cyborg-cyborg"]["diffs"]
        diffs_xs = mode_summaries["cross"]["diffs"]

        levene_stat, levene_p = stats.levene(diffs_jj, diffs_cc, diffs_xs, center="median")
        welch_cross_vs_jj = stats.ttest_ind(diffs_xs, diffs_jj, equal_var=False)
        welch_cross_vs_cc = stats.ttest_ind(diffs_xs, diffs_cc, equal_var=False)

        cross_stats = {
            "levene_stat": float(levene_stat),
            "levene_p": float(levene_p),
            "welch_cross_vs_jj_t": float(welch_cross_vs_jj.statistic),
            "welch_cross_vs_jj_p": float(welch_cross_vs_jj.pvalue),
            "welch_cross_vs_cc_t": float(welch_cross_vs_cc.statistic),
            "welch_cross_vs_cc_p": float(welch_cross_vs_cc.pvalue),
        }

        print()
        print("Cross-mode comparison:")
        print(f"  Levene (equal variances, 3 modes):           stat={levene_stat:.3f}  p={levene_p:.4f}")
        print(
            f"  Welch t: cross vs jax-jax         mean Δ={float(np.mean(diffs_xs) - np.mean(diffs_jj)):+.2f}  "
            f"t={welch_cross_vs_jj.statistic:+.3f}  p={welch_cross_vs_jj.pvalue:.4f}"
        )
        print(
            f"  Welch t: cross vs cyborg-cyborg   mean Δ={float(np.mean(diffs_xs) - np.mean(diffs_cc)):+.2f}  "
            f"t={welch_cross_vs_cc.statistic:+.3f}  p={welch_cross_vs_cc.pvalue:.4f}"
        )

        # Decision rule (File 2): cross within 1 SE of both within-stream means → coincidence;
        # cross ≥ 2 SE off both at matched n → real stream bias; else inconclusive.
        cross_mean = mode_summaries["cross"]["mean"]
        cross_se = mode_summaries["cross"]["se"]
        dist_jj = abs(cross_mean - mode_summaries["jax-jax"]["mean"])
        dist_cc = abs(cross_mean - mode_summaries["cyborg-cyborg"]["mean"])
        within_1_se = (dist_jj <= cross_se) and (dist_cc <= cross_se) and levene_p > 0.05
        beyond_2_se = (dist_jj >= 2 * cross_se) and (dist_cc >= 2 * cross_se)
        if within_1_se:
            verdict = "COINCIDENCE (cross mean is within 1 cross-SE of both within-stream means; variances compatible)"
        elif beyond_2_se:
            verdict = "REAL STREAM BIAS (cross mean ≥ 2 cross-SE off both within-stream means)"
        else:
            verdict = "INCONCLUSIVE (cross offset between 1 and 2 cross-SE, or variances incompatible)"
        print()
        print(f"VERDICT: {verdict}")

    # Per-component breakdown for cross-mode runs that captured it.
    for m in modes:
        res = mode_results[m]
        if res and "cyborg_components" in res[0]:
            comps = ("ria", "lwf", "asf")
            print()
            print(f"Per-component breakdown [{m}] (mean JAX - CybORG, across n={len(res)}):")
            print(
                f"  {'comp':<6}  {'CybORG_mean':>12}  {'JAX_mean':>12}  "
                f"{'diff (J-C)':>12}  {'stdev':>9}"
            )
            for c in comps:
                diffs_c = np.array(
                    [r["jax_components"][c] - r["cyborg_components"][c] for r in res],
                    dtype=float,
                )
                cy_mean = float(np.mean([r["cyborg_components"][c] for r in res]))
                jx_mean = float(np.mean([r["jax_components"][c] for r in res]))
                print(
                    f"  {c.upper():<6}  {cy_mean:>12.2f}  {jx_mean:>12.2f}  "
                    f"{float(diffs_c.mean()):>+12.2f}  {float(diffs_c.std(ddof=1)):>9.2f}"
                )
            jx_ac = float(np.mean([r["jax_components"]["action_cost"] for r in res]))
            print(f"  AC    (JAX action_cost mean): {jx_ac:+.2f}  (sleep blue: 0 by default; random blue: Restore penalties)")

    payload = {
        "mode": mode,
        "seed_start": args.seed_start,
        "n": len(seeds),
        "episode_length": EPISODE_LENGTH,
        "alt_seed_offset": ALT_SEED_OFFSET,
        "modes": {
            m: {
                "per_seed": mode_results[m],
                "mean": mode_summaries[m]["mean"],
                "std": mode_summaries[m]["std"],
                "se": mode_summaries[m]["se"],
                "ci90_lower": mode_summaries[m]["ci90"][0],
                "ci90_upper": mode_summaries[m]["ci90"][1],
                "pos": mode_summaries[m]["pos"],
                "neg": mode_summaries[m]["neg"],
                "sign_p": mode_summaries[m]["sign_p"],
                "wilcoxon_p": mode_summaries[m]["wilcoxon_p"],
                "t_p": mode_summaries[m]["t_p"],
            }
            for m in modes
        },
        "cross_stats": cross_stats,
        "verdict": verdict,
    }
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote: {out}")


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
        help="With --sync-green-rng, additionally log per-step LWF/ASF/RIA counts on both backends.",
    )
    parser.add_argument(
        "--random-blue",
        action="store_true",
        help=(
            "With --sync-green-rng, use mask-aware uniform random blue actions "
            "(seeded per-seed) instead of sleep. Exercises Restore/Remove/"
            "BlockTraffic/etc. so ASF and more code paths fire."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "With --rng-comparison, skip modes whose per-mode checkpoint "
            "(<output-json>_<mode>.json) already exists for the requested seed range."
        ),
    )
    parser.add_argument(
        "--rng-comparison",
        choices=("jax-jax", "cyborg-cyborg", "cross", "all"),
        default=None,
        help=(
            "Stream-noise decomposition: run within-stream (jax-jax, cyborg-cyborg) and/or "
            "cross-stream (cross) paired diffs on the same seed range. Compares the three "
            "distributions to decide whether the cross-backend lean is real stream bias or "
            "small-n coincidence. Incompatible with --track-components/--sync-green-rng."
        ),
    )
    args = parser.parse_args()

    if args.track_components and args.sync_green_rng:
        raise SystemExit("--track-components is incompatible with --sync-green-rng (harness path)")
    if args.track_lwf and not args.sync_green_rng:
        raise SystemExit("--track-lwf requires --sync-green-rng")
    if args.rng_comparison and (args.track_components or args.sync_green_rng or args.track_lwf):
        raise SystemExit(
            "--rng-comparison is incompatible with --track-components/--sync-green-rng/--track-lwf"
        )
    if args.random_blue and not (args.sync_green_rng or args.rng_comparison == "cross"):
        raise SystemExit(
            "--random-blue requires --sync-green-rng or --rng-comparison=cross"
        )

    if args.rng_comparison:
        return _main_rng_comparison(args)

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    print(f"Running paired sleep baseline: {len(seeds)} seeds × {EPISODE_LENGTH} steps × 2 backends")
    print(
        f"Seeds: [{seeds[0]}..{seeds[-1]}], workers: {args.workers}, "
        f"track_components={args.track_components}, sync_green_rng={args.sync_green_rng}, "
        f"track_lwf={args.track_lwf}"
    )

    if args.sync_green_rng:
        runner = _run_one_pair_synced
        worker_args = [(s, args.track_lwf, args.random_blue) for s in seeds]
    else:
        if args.random_blue:
            raise SystemExit("--random-blue requires --sync-green-rng")
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
            for comp in ("lwf", "asf", "ria"):
                if not any(comp in r for r in results):
                    continue
                jx = sum(r[comp]["total_jax"] for r in results if comp in r)
                cy = sum(r[comp]["total_cyborg"] for r in results if comp in r)
                steps_diff = sum(r[comp]["steps_diff"] for r in results if comp in r)
                max_abs = max((r[comp]["max_abs_step_diff"] for r in results if comp in r), default=0)
                print(
                    f"{comp.upper()} event totals (across all eps): JAX={jx}  CybORG={cy}  "
                    f"Δ={jx - cy}    per-step diff count={steps_diff}  max |Δ_step|={max_abs}"
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
