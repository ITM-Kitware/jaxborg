"""Compare CybORG vs JAXborg observations mid-episode after red activity.

Runs CybORG for N steps with Sleep blue, then compares the CybORG
BlueFlatWrapper observation against what JAXborg's get_blue_obs would
produce from a JAXborg state built from CybORG's live state.
"""

import jax.numpy as jnp
import numpy as np
from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Simulator.Actions import Sleep
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.constants import (
    GLOBAL_MAX_HOSTS,
    NUM_BLUE_AGENTS,
    NUM_SUBNETS,
)
from jaxborg.observations import get_blue_obs
from jaxborg.state import create_initial_state
from jaxborg.topology import CYBORG_SUFFIX_TO_ID, build_const_from_cyborg

SEED = 42
CHECK_STEPS = [0, 5, 10, 20, 50]


def make_env():
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500,
    )
    cyborg = CybORG(sg, "sim", seed=SEED)
    return BlueFlatWrapper(env=cyborg, pad_spaces=True)


def extract_jax_state_from_cyborg(cyborg_env, const):
    """Build a JAXborg CC4State from CybORG's live state for obs comparison."""
    state_obj = cyborg_env.environment_controller.state

    sorted_hostnames = sorted(state_obj.hosts.keys())
    hostname_to_idx = {h: i for i, h in enumerate(sorted_hostnames)}

    host_has_malware = np.zeros(GLOBAL_MAX_HOSTS, dtype=bool)
    host_activity_detected = np.zeros(GLOBAL_MAX_HOSTS, dtype=bool)

    for hostname, idx in hostname_to_idx.items():
        host = state_obj.hosts[hostname]
        proc_events = host.events.old_process_creation + host.events.process_creation
        conn_events = host.events.old_network_connections + host.events.network_connections
        host_has_malware[idx] = len(proc_events) > 0
        host_activity_detected[idx] = len(conn_events) > 0

    blocked_zones = np.zeros((NUM_SUBNETS, NUM_SUBNETS), dtype=bool)
    for dst_name, src_list in state_obj.blocks.items():
        dst_sid = CYBORG_SUFFIX_TO_ID.get(dst_name)
        if dst_sid is None:
            continue
        for src_name in src_list:
            src_sid = CYBORG_SUFFIX_TO_ID.get(src_name)
            if src_sid is not None:
                blocked_zones[dst_sid, src_sid] = True

    mission_phase = state_obj.mission_phase

    jax_state = create_initial_state()
    jax_state = jax_state.replace(
        mission_phase=jnp.array(mission_phase, dtype=jnp.int32),
        host_has_malware=jnp.array(host_has_malware),
        host_activity_detected=jnp.array(host_activity_detected),
        blocked_zones=jnp.array(blocked_zones),
    )
    return jax_state


def main():
    env = make_env()
    observations, _ = env.reset()
    inner = env.env
    const = build_const_from_cyborg(inner)

    print("=" * 70)
    print("MID-EPISODE OBSERVATION PARITY: CybORG vs JAXborg")
    print("=" * 70)

    for step in range(max(CHECK_STEPS) + 1):
        if step in CHECK_STEPS:
            jax_state = extract_jax_state_from_cyborg(inner, const)

            print(f"\n--- Step {step} ---")

            # Count non-zero malware/activity signals
            n_malware = int(jax_state.host_has_malware.sum())
            n_activity = int(jax_state.host_activity_detected.sum())
            n_blocked = int(jax_state.blocked_zones.sum())
            print(f"  CybORG state: malware={n_malware} hosts, activity={n_activity} hosts, blocked_pairs={n_blocked}")

            total_diffs = 0
            for agent_idx in range(NUM_BLUE_AGENTS):
                agent_name = f"blue_agent_{agent_idx}"
                cyborg_obs = np.array(observations[agent_name], dtype=np.float32)
                jax_obs = np.array(get_blue_obs(jax_state, const, agent_idx))

                diff_mask = np.abs(cyborg_obs - jax_obs) > 1e-6
                n_diff = diff_mask.sum()
                total_diffs += n_diff

                if n_diff > 0:
                    diff_indices = np.where(diff_mask)[0]
                    print(f"  {agent_name}: {n_diff} diffs at indices {diff_indices[:15].tolist()}")
                    for idx in diff_indices[:5]:
                        print(f"    [{idx:3d}] CybORG={cyborg_obs[idx]:.4f}  JAXborg={jax_obs[idx]:.4f}")

            if total_diffs == 0:
                print("  ALL AGENTS: observations match perfectly")

        actions = {a: Sleep() for a in env.agents}
        observations, rewards, _, _, _ = env.step(actions=actions)


if __name__ == "__main__":
    main()
