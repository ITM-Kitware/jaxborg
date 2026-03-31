"""Measure why CybORG server_session is smaller than sessions on allowed-subnet hosts.

For each red agent, at several milestones, computes:
- server_session IDs (what the FSM picks from)
- active session IDs on hosts in the agent's allowed subnets
- all active session IDs
This reveals which sessions are "missing" from server_session and why.

Usage:
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
    uv run python scripts/diagnose_ss_gap.py
"""

# ruff: noqa: E402

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from eval_transfer import DEFAULT_BANK_SIZE, NUM_BLUE_AGENTS, make_cyborg_env


def main():
    seed = 42

    wrapper = make_cyborg_env(seed=seed, bank_match_size=DEFAULT_BANK_SIZE)
    cyborg = wrapper.env
    wrapper.reset()

    ctrl = cyborg.environment_controller
    red_agents = sorted(n for n in ctrl.agent_interfaces if n.startswith("red_agent_"))
    blue_agents = [f"blue_agent_{i}" for i in range(NUM_BLUE_AGENTS)]

    # Get allowed subnets per red agent
    from CybORG.Simulator.Scenarios.EnterpriseScenarioGenerator import SUBNET
    allowed_map = {
        "red_agent_0": {SUBNET.CONTRACTOR_NETWORK.value},
        "red_agent_1": {SUBNET.RESTRICTED_ZONE_A.value},
        "red_agent_2": {SUBNET.OPERATIONAL_ZONE_A.value},
        "red_agent_3": {SUBNET.RESTRICTED_ZONE_B.value},
        "red_agent_4": {SUBNET.OPERATIONAL_ZONE_B.value},
        "red_agent_5": {SUBNET.PUBLIC_ACCESS_ZONE.value, SUBNET.ADMIN_NETWORK.value, SUBNET.OFFICE_NETWORK.value},
    }

    from CybORG.Simulator.Actions import Sleep

    milestones = [49, 99, 199, 299, 499]

    for step in range(500):
        actions = {name: Sleep() for name in blue_agents}
        ctrl.step(actions)

        if step not in milestones:
            continue

        print(f"\n{'='*80}")
        print(f"Step {step+1}")
        print(f"{'='*80}")

        for name in red_agents:
            ai = ctrl.agent_interfaces[name]
            ss = dict(ai.action_space.server_session)
            sessions = ctrl.state.sessions.get(name, {})
            allowed_subs = allowed_map[name]

            # Classify sessions by subnet
            on_allowed = {}
            on_other = {}
            for sid, sess in sessions.items():
                hostname = sess.hostname
                subnet_name = ctrl.state.hostname_subnet_map.get(hostname, "unknown")
                if subnet_name in allowed_subs:
                    on_allowed[sid] = (hostname, subnet_name, type(sess).__name__)
                else:
                    on_other[sid] = (hostname, subnet_name, type(sess).__name__)

            ss_ids = sorted(ss.keys())
            allowed_ids = sorted(on_allowed.keys())
            other_ids = sorted(on_other.keys())
            missing_from_ss = sorted(set(allowed_ids) - set(ss_ids))

            if not sessions:
                continue

            print(f"\n  {name} (allowed: {sorted(allowed_subs)})")
            print(f"    server_session IDs:  {ss_ids} ({len(ss_ids)})")
            print(f"    on-allowed-subnet:   {allowed_ids} ({len(allowed_ids)})")
            print(f"    on-other-subnet:     {other_ids} ({len(other_ids)})")
            print(f"    missing from SS:     {missing_from_ss} ({len(missing_from_ss)})")

            if missing_from_ss:
                print("    Details of missing sessions:")
                for sid in missing_from_ss[:5]:
                    hostname, subnet, stype = on_allowed[sid]
                    # Check if session has session_type attribute
                    sess = sessions[sid]
                    st = getattr(sess, 'session_type', 'unknown')
                    print(f"      sess {sid}: host={hostname} subnet={subnet} type={stype} session_type={st}")

            # Check if any SS entries don't exist in active sessions
            stale_ss = sorted(set(ss_ids) - set(sessions.keys()))
            if stale_ss:
                print(f"    stale in SS (dead): {stale_ss}")


if __name__ == "__main__":
    main()
