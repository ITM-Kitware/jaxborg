"""Instrument CybORG to measure red session-selection dynamics.

Monkey-patches ExploitRemoteService.execute and DiscoverNetworkServices.execute
to log exactly which session was selected, what ports it has, and why it
succeeded or failed.

Usage:
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
    uv run python scripts/diagnose_session_selection.py --episodes 3
"""

# ruff: noqa: E402

import os

os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.expanduser("~/.cache/jaxborg/xla"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from eval_transfer import DEFAULT_BANK_SIZE, NUM_BLUE_AGENTS, make_cyborg_env

from jaxborg.actions.encoding import BLUE_SLEEP
from jaxborg.actions.session_counts import effective_session_counts
from jaxborg.constants import NUM_RED_AGENTS
from jaxborg.fsm_red_env import FsmRedCC4Env

# ---------------------------------------------------------------------------
# Global log filled by monkey-patched CybORG methods
# ---------------------------------------------------------------------------
ACTION_LOG = []


def _install_patches():
    """Monkey-patch CybORG exploit & scan execute() to log session details."""
    from CybORG.Shared.Session import RedAbstractSession
    from CybORG.Simulator.Actions.AbstractActions.DiscoverNetworkServices import (
        DiscoverNetworkServices,
    )
    from CybORG.Simulator.Actions.AbstractActions.ExploitRemoteService import (
        ExploitRemoteService,
    )

    _orig_exploit = ExploitRemoteService.execute
    _orig_scan = DiscoverNetworkServices.execute

    def _patched_exploit(self, state):
        session_obj = state.sessions.get(self.agent, {}).get(self.session, None)
        entry = {
            "kind": "exploit",
            "agent": self.agent,
            "session_id": self.session,
            "target_ip": str(self.ip_address),
            "session_exists": session_obj is not None,
            "session_host": getattr(session_obj, "hostname", None),
            "is_abstract": isinstance(session_obj, RedAbstractSession) if session_obj else False,
            "has_ports": (
                (self.ip_address in session_obj.ports)
                if session_obj and isinstance(session_obj, RedAbstractSession)
                else False
            ),
            "num_ports_keys": len(session_obj.ports) if session_obj and hasattr(session_obj, "ports") else 0,
            "total_agent_sessions": len(state.sessions.get(self.agent, {})),
        }
        obs = _orig_exploit(self, state)
        success = obs.data.get("success", None)
        entry["success"] = str(success) if success is not None else None
        # Check TernaryEnum properly
        from CybORG.Shared.Enums import TernaryEnum
        entry["success_bool"] = (success == TernaryEnum.TRUE) if success is not None else None
        ACTION_LOG.append(entry)
        return obs

    def _patched_scan(self, state):
        session_obj = state.sessions.get(self.agent, {}).get(self.session, None)
        entry = {
            "kind": "scan",
            "agent": self.agent,
            "session_id": self.session,
            "target_ip": str(self.ip_address),
            "session_exists": session_obj is not None,
            "session_host": getattr(session_obj, "hostname", None),
            "is_abstract": isinstance(session_obj, RedAbstractSession) if session_obj else False,
            "total_agent_sessions": len(state.sessions.get(self.agent, {})),
        }
        obs = _orig_scan(self, state)
        success = obs.data.get("success", None)
        from CybORG.Shared.Enums import TernaryEnum
        entry["success_bool"] = (success == TernaryEnum.TRUE) if success is not None else None
        # Check if ports were stored on this session after scan
        session_after = state.sessions.get(self.agent, {}).get(self.session, None)
        if session_after and hasattr(session_after, "ports"):
            entry["ports_after"] = len(session_after.ports)
        else:
            entry["ports_after"] = 0
        ACTION_LOG.append(entry)
        return obs

    ExploitRemoteService.execute = _patched_exploit
    DiscoverNetworkServices.execute = _patched_scan


def run_cyborg_episode(seed, bank_size):
    """Run one CybORG episode with monkey-patched logging."""
    from CybORG.Simulator.Actions import Sleep

    ACTION_LOG.clear()
    wrapper = make_cyborg_env(seed=seed, bank_match_size=bank_size)
    cyborg = wrapper.env
    wrapper.reset()

    ctrl = cyborg.environment_controller
    red_agents = [n for n in ctrl.agent_interfaces if n.startswith("red_agent_")]
    blue_agents = [f"blue_agent_{i}" for i in range(NUM_BLUE_AGENTS)]

    # Track server_session sizes at milestones
    ss_snapshots = {}
    active_snapshots = {}

    for step in range(500):
        actions = {name: Sleep() for name in blue_agents}
        ctrl.step(actions)

        if step in (99, 249, 499):
            ss = {}
            act = {}
            for name in red_agents:
                ai = ctrl.agent_interfaces[name]
                ss[name] = dict(ai.action_space.server_session)
                sessions = ctrl.state.sessions.get(name, {})
                act[name] = {sid: s.hostname for sid, s in sessions.items()}
            ss_snapshots[step] = ss
            active_snapshots[step] = act

    return list(ACTION_LOG), ss_snapshots, active_snapshots


def run_jax_episode(seed, bank_size):
    """Run one JAX episode, collecting session counts at milestones."""
    env = FsmRedCC4Env(
        num_steps=500,
        topology_mode="cyborg_bank",
        topology_bank_size=bank_size,
    )
    key = jax.random.PRNGKey(seed)
    _, state = env.reset(key)
    sleep_actions = {f"blue_{i}": jnp.int32(BLUE_SLEEP) for i in range(NUM_BLUE_AGENTS)}

    snapshots = {}
    for step in range(500):
        key, step_key = jax.random.split(key)
        _, state, _, _, _ = env.step(step_key, state, sleep_actions)
        if step in (99, 249, 499):
            counts = np.asarray(effective_session_counts(state.state))
            snapshots[step] = {
                f"red_agent_{r}": int(counts[r].sum()) for r in range(NUM_RED_AGENTS)
            }
    return snapshots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topology-bank-size", type=int, default=DEFAULT_BANK_SIZE)
    args = parser.parse_args()

    _install_patches()

    agg_exploit = defaultdict(int)  # success_bool -> count
    agg_scan = defaultdict(int)
    agg_exploit_fail_reasons = defaultdict(int)
    session_id_picks = defaultdict(int)  # session_id -> count (for exploits)

    for ep in range(args.episodes):
        ep_seed = args.seed + ep * 100
        print(f"\n{'='*70}")
        print(f"Episode {ep+1} (seed={ep_seed})")
        print(f"{'='*70}")

        log, ss_snap, active_snap = run_cyborg_episode(ep_seed, args.topology_bank_size)
        jax_snap = run_jax_episode(ep_seed, args.topology_bank_size)

        # Classify log entries
        exploits = [e for e in log if e["kind"] == "exploit"]
        scans = [e for e in log if e["kind"] == "scan"]

        exploit_ok = sum(1 for e in exploits if e["success_bool"])
        exploit_fail = sum(1 for e in exploits if not e["success_bool"])
        scan_ok = sum(1 for e in scans if e["success_bool"])
        scan_fail = sum(1 for e in scans if not e["success_bool"])

        agg_exploit["ok"] += exploit_ok
        agg_exploit["fail"] += exploit_fail
        agg_scan["ok"] += scan_ok
        agg_scan["fail"] += scan_fail

        print(f"\n  Scans:    {scan_ok} ok, {scan_fail} fail (total {len(scans)})")
        print(f"  Exploits: {exploit_ok} ok, {exploit_fail} fail (total {len(exploits)})")

        # Exploit failure breakdown
        for e in exploits:
            if not e["success_bool"]:
                if not e["session_exists"]:
                    reason = "session_not_found"
                elif not e["is_abstract"]:
                    reason = "not_abstract"
                elif not e["has_ports"]:
                    reason = "no_ports_for_target"
                else:
                    reason = "other (blocked/sub_action)"
                agg_exploit_fail_reasons[reason] += 1

        # Session ID distribution for exploits
        for e in exploits:
            session_id_picks[e["session_id"]] += 1

        # Show failed exploit details
        failed_exploits = [e for e in exploits if not e["success_bool"]]
        if failed_exploits:
            print("\n  Failed exploit details (first 10):")
            for e in failed_exploits[:10]:
                print(f"    agent={e['agent']} sess={e['session_id']} "
                      f"host={e['session_host']} target={e['target_ip']} "
                      f"exists={e['session_exists']} abstract={e['is_abstract']} "
                      f"has_ports={e['has_ports']} total_sess={e['total_agent_sessions']}")

        # Scan failure details
        failed_scans = [e for e in scans if not e["success_bool"]]
        if failed_scans:
            print("\n  Failed scan details (first 10):")
            for e in failed_scans[:10]:
                print(f"    agent={e['agent']} sess={e['session_id']} "
                      f"host={e['session_host']} target={e['target_ip']} "
                      f"exists={e['session_exists']} abstract={e['is_abstract']}")

        # Session snapshots
        for step_idx in (99, 249, 499):
            if step_idx not in ss_snap:
                continue
            print(f"\n  Step {step_idx+1} session snapshot:")
            print(f"    {'Agent':<14} {'SS IDs':<30} {'Active IDs':<40} {'JAX total':>9}")
            for name in sorted(ss_snap[step_idx]):
                ss_ids = sorted(ss_snap[step_idx][name].keys())
                active_ids = sorted(active_snap[step_idx][name].keys())
                jax_total = jax_snap.get(step_idx, {}).get(name, "?")
                print(f"    {name:<14} {str(ss_ids):<30} {str(active_ids):<40} {jax_total:>9}")

    # Aggregate
    print(f"\n{'='*70}")
    print("AGGREGATE")
    print(f"{'='*70}")
    print(f"  Scans:    {agg_scan['ok']} ok, {agg_scan['fail']} fail")
    print(f"  Exploits: {agg_exploit['ok']} ok, {agg_exploit['fail']} fail")
    if agg_exploit_fail_reasons:
        print("  Exploit failure reasons:")
        for reason, count in sorted(agg_exploit_fail_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")
    print("\n  Exploit session ID picks (how often each session ID was selected):")
    for sid, count in sorted(session_id_picks.items()):
        print(f"    session {sid}: {count} times")


if __name__ == "__main__":
    main()
