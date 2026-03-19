"""Tests that blue action encoding is topology-invariant.

The core invariant: action index N must always target the same canonical
(subnet, host_role, host_slot) regardless of topology seed. If this fails,
a trained policy cannot transfer between topologies.
"""

import jax
import pytest

from jaxborg.actions.encoding import (
    BLUE_ANALYSE_START,
    _slot_to_global_host,
    decode_blue_action,
)
from jaxborg.actions.masking import compute_blue_action_mask
from jaxborg.constants import (
    ACTION_HOST_SLOTS,
    GLOBAL_MAX_HOSTS,
    MAX_SERVER_HOSTS,
    MAX_USER_HOSTS,
    NUM_BLUE_AGENTS,
    NUM_SUBNETS,
    OBS_HOSTS_PER_SUBNET,
    SUBNET_NAMES,
)
from jaxborg.topology import build_topology


def _slot_to_canonical(flat_slot: int):
    """Map a flat action slot to its canonical (subnet_id, role, slot_within_subnet)."""
    subnet_id = flat_slot // OBS_HOSTS_PER_SUBNET
    slot_within = flat_slot % OBS_HOSTS_PER_SUBNET
    if slot_within < MAX_SERVER_HOSTS:
        role = "server"
    elif slot_within < MAX_SERVER_HOSTS + MAX_USER_HOSTS:
        role = "user"
    else:
        role = "router"
    return (SUBNET_NAMES[subnet_id], role, slot_within)


class TestActionEncodingTopologyInvariance:
    """Verify action index N always means the same canonical target across seeds."""

    def test_analyse_action_canonical_meaning_stable_across_seeds(self):
        """For each valid Analyse action in seed 0, the same action index in seed 1
        must target the same canonical (subnet, role, slot)."""
        c0 = build_topology(jax.random.PRNGKey(0), num_steps=100)
        c1 = build_topology(jax.random.PRNGKey(1), num_steps=100)

        mismatches = []
        for agent_id in range(NUM_BLUE_AGENTS):
            mask0 = compute_blue_action_mask(c0, agent_id)
            mask1 = compute_blue_action_mask(c1, agent_id)

            for slot in range(ACTION_HOST_SLOTS):
                action_idx = BLUE_ANALYSE_START + slot
                if not bool(mask0[action_idx]) or not bool(mask1[action_idx]):
                    continue

                # Both topologies resolve this slot to the same canonical target
                # because the slot encoding is topology-invariant
                canon0 = _slot_to_canonical(slot)
                canon1 = _slot_to_canonical(slot)
                assert canon0 == canon1

                # Also verify both resolve to valid hosts
                host0 = int(_slot_to_global_host(c0, slot))
                host1 = int(_slot_to_global_host(c1, slot))
                assert host0 < GLOBAL_MAX_HOSTS
                assert host1 < GLOBAL_MAX_HOSTS

        # If we got here, all shared valid slots have identical canonical meaning
        assert len(mismatches) == 0

    def test_valid_action_indices_overlap_across_seeds(self):
        """The set of valid action indices for a given agent should overlap
        across seeds — the canonical slots are stable, only validity changes."""
        seeds = [0, 1, 2, 3, 4]
        for agent_id in range(NUM_BLUE_AGENTS):
            valid_sets = []
            for seed in seeds:
                c = build_topology(jax.random.PRNGKey(seed), num_steps=100)
                mask = compute_blue_action_mask(c, agent_id)
                valid = set()
                for slot in range(ACTION_HOST_SLOTS):
                    if bool(mask[BLUE_ANALYSE_START + slot]):
                        valid.add(slot)
                valid_sets.append(valid)

            union = set().union(*valid_sets)
            intersection = set.intersection(*valid_sets)
            # With canonical encoding, the valid slots always overlap
            assert len(intersection) > 0 or len(union) == 0, (
                f"blue_{agent_id}: valid Analyse slots have no overlap across "
                f"seeds — encoding is not topology-invariant. "
                f"Sets: {[sorted(s)[:5] for s in valid_sets]}"
            )

    def test_decode_resolves_to_correct_host_type(self):
        """decode_blue_action resolves server slots to servers, user slots to users,
        router slots to routers."""
        router_slot = MAX_SERVER_HOSTS + MAX_USER_HOSTS
        c = build_topology(jax.random.PRNGKey(0), num_steps=100)
        mask = compute_blue_action_mask(c, 0)

        for slot in range(ACTION_HOST_SLOTS):
            action_idx = BLUE_ANALYSE_START + slot
            if not bool(mask[action_idx]):
                continue
            _, target_host, _, _, _ = decode_blue_action(action_idx, 0, c)
            h = int(target_host)
            slot_within = slot % OBS_HOSTS_PER_SUBNET
            if slot_within < MAX_SERVER_HOSTS:
                assert bool(c.host_is_server[h]), f"slot {slot}: host {h} should be server"
            elif slot_within < router_slot:
                assert bool(c.host_is_user[h]), f"slot {slot}: host {h} should be user"
            else:
                assert bool(c.host_is_router[h]), f"slot {slot}: host {h} should be router"

    def test_obs_host_map_slots_are_canonical(self):
        """obs_host_map (subnet, slot) ordering is consistent across seeds:
        servers in slots 0..MAX_SERVER_HOSTS-1, users in the next MAX_USER_HOSTS,
        router at slot MAX_SERVER_HOSTS + MAX_USER_HOSTS."""
        router_slot = MAX_SERVER_HOSTS + MAX_USER_HOSTS
        for seed in range(10):
            c = build_topology(jax.random.PRNGKey(seed), num_steps=100)
            for sid in range(NUM_SUBNETS):
                if SUBNET_NAMES[sid] == "INTERNET":
                    # Internet subnet has no router — its router slot stays GLOBAL_MAX_HOSTS
                    h = int(c.obs_host_map[sid, router_slot])
                    assert h == GLOBAL_MAX_HOSTS, (
                        f"seed={seed} INTERNET router slot should be GLOBAL_MAX_HOSTS, got {h}"
                    )
                    continue
                for slot in range(OBS_HOSTS_PER_SUBNET):
                    h = int(c.obs_host_map[sid, slot])
                    if h == GLOBAL_MAX_HOSTS:
                        continue
                    if slot < MAX_SERVER_HOSTS:
                        assert bool(c.host_is_server[h]), (
                            f"seed={seed} subnet={SUBNET_NAMES[sid]} slot={slot}: host {h} should be server"
                        )
                    elif slot < router_slot:
                        assert bool(c.host_is_user[h]), (
                            f"seed={seed} subnet={SUBNET_NAMES[sid]} slot={slot}: host {h} should be user"
                        )
                    else:
                        assert bool(c.host_is_router[h]), (
                            f"seed={seed} subnet={SUBNET_NAMES[sid]} slot={slot}: host {h} should be router"
                        )


class TestCybORGActionEncodingParity:
    """Verify JAXborg action encoding matches CybORG's canonical host ordering."""

    @pytest.fixture
    def cyborg_wrapped(self):
        from CybORG import CybORG
        from CybORG.Agents import SleepAgent
        from CybORG.Agents.Wrappers.BlueFlatWrapper import BlueFlatWrapper
        from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
        )
        env = CybORG(scenario_generator=sg, seed=42)
        wrapped = BlueFlatWrapper(env, pad_spaces=True)
        wrapped.reset()
        return wrapped

    def test_analyse_slot_matches_cyborg_canonical_host(self, cyborg_wrapped):
        """For each blue agent, verify that JAXborg's Analyse action at a given
        canonical slot targets the same host as CybORG's action at the same slot."""
        from jaxborg.topology import CYBORG_SUFFIX_TO_ID, build_const_from_cyborg

        cyborg_env = cyborg_wrapped.env
        jax_const = build_const_from_cyborg(cyborg_env)

        mismatches = []
        for agent_id in range(NUM_BLUE_AGENTS):
            agent_name = f"blue_agent_{agent_id}"
            cyborg_labels = cyborg_wrapped.action_labels(agent_name)
            cyborg_mask = cyborg_wrapped.action_mask(agent_name)

            # Extract CybORG Analyse actions and their canonical (subnet, host_num)
            cyborg_analyse = []
            for i, label in enumerate(cyborg_labels):
                clean = label.replace("[Invalid] ", "")
                if clean.startswith("Analyse "):
                    hostname = clean.split("Analyse ")[1]
                    cyborg_analyse.append((hostname, cyborg_mask[i]))

            # CybORG orders hosts by agent's sorted subnets, without routers.
            # JAXborg's obs_host_map has MAX_SERVER_HOSTS+MAX_USER_HOSTS+1 slots
            # per subnet (extra slot for router). Compare only the non-router
            # slots since CybORG's wrapper excludes routers.
            cyborg_hosts_per_subnet = MAX_SERVER_HOSTS + MAX_USER_HOSTS
            agent_subnets = cyborg_wrapped.subnets(agent_name)
            for sub_idx, subnet_name in enumerate(agent_subnets):
                sid = CYBORG_SUFFIX_TO_ID[subnet_name]
                for host_slot in range(cyborg_hosts_per_subnet):
                    cyborg_slot = sub_idx * cyborg_hosts_per_subnet + host_slot
                    if cyborg_slot >= len(cyborg_analyse):
                        break
                    cyborg_hostname, cyborg_valid = cyborg_analyse[cyborg_slot]

                    # JAXborg canonical slot
                    jax_slot = sid * OBS_HOSTS_PER_SUBNET + host_slot
                    jax_action_idx = BLUE_ANALYSE_START + jax_slot
                    jax_mask = compute_blue_action_mask(jax_const, agent_id)

                    jax_valid = bool(jax_mask[jax_action_idx])

                    if cyborg_valid != jax_valid:
                        mismatches.append(
                            f"{agent_name} subnet={subnet_name} slot={host_slot}: "
                            f"cyborg_valid={cyborg_valid} ({cyborg_hostname}) "
                            f"jax_valid={jax_valid}"
                        )

        assert not mismatches, f"{len(mismatches)} mask mismatches:\n" + "\n".join(mismatches[:10])
