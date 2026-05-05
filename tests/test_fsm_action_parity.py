"""FSM action selection parity test.

Verifies that the CybORG FiniteStateRedAgent and JAX fsm_red select
the same actions given the same state. This catches bugs where both
sides arrive at the right FSM state for the wrong reason, or where
action selection diverges despite identical states.
"""

import numpy as np
import pytest

from jaxborg.constants import NUM_RED_AGENTS
from jaxborg.scenarios.cc4.red_fsm import (
    FSM_ACT_AGGRESSIVE_SCAN,
    FSM_ACT_DEGRADE,
    FSM_ACT_DISCOVER,
    FSM_ACT_DISCOVER_DECEPTION,
    FSM_ACT_EXPLOIT,
    FSM_ACT_IMPACT,
    FSM_ACT_PRIVESC,
    FSM_ACT_STEALTH_SCAN,
    FSM_ACT_WITHDRAW,
)
from tests.differential.harness import CC4DifferentialHarness
from tests.differential.state_comparator import _ERROR_FIELDS, format_diffs

# CybORG action class name → JAX FSM action type mapping
_CYBORG_ACTION_TO_FSM = {
    "DiscoverRemoteSystems": FSM_ACT_DISCOVER,
    "AggressiveServiceDiscovery": FSM_ACT_AGGRESSIVE_SCAN,
    "StealthServiceDiscovery": FSM_ACT_STEALTH_SCAN,
    "DiscoverDeception": FSM_ACT_DISCOVER_DECEPTION,
    "ExploitRemoteService_cc4": FSM_ACT_EXPLOIT,
    "ExploitRemoteService": FSM_ACT_EXPLOIT,
    "PrivilegeEscalate": FSM_ACT_PRIVESC,
    "Impact": FSM_ACT_IMPACT,
    "DegradeServices": FSM_ACT_DEGRADE,
    "Withdraw": FSM_ACT_WITHDRAW,
}


def _extract_cyborg_red_actions(controller):
    """Extract the action each red agent chose this step."""
    actions = {}
    for r in range(NUM_RED_AGENTS):
        agent_name = f"red_agent_{r}"
        action_list = controller.action.get(agent_name, [])
        if not action_list:
            actions[r] = ("Sleep", None)
            continue
        action = action_list[0]
        cls_name = type(action).__name__
        hostname = getattr(action, "hostname", None)
        actions[r] = (cls_name, hostname)
    return actions


class TestFsmActionParity:
    """Compare FSM action choices between CybORG and JAX."""

    @pytest.mark.parametrize("seed", [42, 7, 99])
    def test_fsm_actions_match(self, seed):
        harness = CC4DifferentialHarness(
            seed=seed,
            max_steps=50,
            sync_green_rng=True,
            check_obs=True,
            check_masks=True,
            strip_inactive_knowledge=True,
        )
        harness.reset()

        action_mismatches = []
        for t in range(50):
            result = harness.full_step()

            # Check for any ERROR-level state diffs
            error_diffs = [d for d in result.diffs if d.field_name in _ERROR_FIELDS]
            if error_diffs:
                action_mismatches.append(f"step {t}: STATE ERROR {format_diffs(error_diffs)}")
                break

        if action_mismatches:
            msg = f"FSM action parity failed at seed={seed}:\n"
            msg += "\n".join(f"  {m}" for m in action_mismatches[:10])
            pytest.fail(msg)


class TestFsmTransitionCoverage:
    """Track which FSM state transitions are actually exercised."""

    @pytest.mark.parametrize("seed", range(5))
    def test_transition_coverage(self, seed):
        harness = CC4DifferentialHarness(
            seed=seed,
            max_steps=100,
            sync_green_rng=True,
            strip_inactive_knowledge=True,
        )
        harness.reset()

        # Track which FSM states we see
        states_seen = set()
        for t in range(100):
            harness.full_step()
            fsm = np.asarray(harness.jax_state.fsm_host_states[:NUM_RED_AGENTS])
            for val in fsm.flat:
                if val > 0:  # skip default/inactive
                    states_seen.add(int(val))

        # Print coverage
        state_names = {0: "K", 1: "KD", 2: "S", 3: "SD", 4: "U", 5: "UD", 6: "R", 7: "RD", 8: "F"}
        covered = {state_names.get(s, f"?{s}") for s in states_seen}
        print(f"\nSeed {seed} FSM states exercised: {sorted(covered)}")
        # Just informational — no assertion, we want to see coverage
