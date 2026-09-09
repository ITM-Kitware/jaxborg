"""Opt-in co-training rule knobs: Red's payoff and Blue's BlockTraffic mask.

Both default to stock CC4.  The first test in each group is the parity guard:
if it fails, an opt-in knob has leaked into the default contract.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxborg.actions.encoding import BLUE_BLOCK_TRAFFIC_END, BLUE_BLOCK_TRAFFIC_START, decode_blue_action
from jaxborg.actions.masking import compute_blue_action_mask, mission_permitted_block_mask
from jaxborg.constants import NUM_BLUE_AGENTS
from jaxborg.joint_env import JointPolicyCC4Env
from jaxborg.recipe import eval_variant, train_variant
from jaxborg.scenarios.cc4.game_variant import GameVariant
from jaxborg.scenarios.cc4.game_variants import CC4_STOCK, variant_for_red
from jaxborg.scenarios.cc4.topology import build_topology

RED_POLICY_SLEEP = 0


def _joint_env(**kwargs):
    return JointPolicyCC4Env(num_steps=30, training_mode=True, **kwargs)


def _sleep_actions(env):
    return {a: jnp.int32(0) for a in env.blue_agents} | {a: jnp.int32(RED_POLICY_SLEEP) for a in env.red_agents}


class TestDefaultsAreStockCC4:
    def test_variant_defaults(self):
        assert CC4_STOCK.red_reward == "zero_sum"
        assert CC4_STOCK.blue_block_policy == "cc4"
        assert CC4_STOCK.is_stock_contract

    def test_mask_default_matches_pre_flag_behaviour(self):
        const = build_topology(jax.random.PRNGKey(0), num_steps=100)
        for agent_id in range(NUM_BLUE_AGENTS):
            explicit = compute_blue_action_mask(const, agent_id, blue_block_policy="cc4")
            assert jnp.array_equal(compute_blue_action_mask(const, agent_id), explicit)

    def test_zero_sum_is_the_default_red_payoff(self):
        env = _joint_env()
        obs, state = env.reset(jax.random.PRNGKey(0))
        _, _, rewards, _, _ = env.step_env(jax.random.PRNGKey(1), state, _sleep_actions(env))
        assert float(rewards["red_0"]) == pytest.approx(-float(rewards["blue_0"]))


class TestRedRewardModes:
    def test_damage_drops_asf_and_action_cost_from_red(self):
        """Red's payoff loses exactly the terms Red cannot cause."""
        actions = None
        for mode in ("zero_sum", "damage"):
            env = _joint_env(red_reward=mode)
            obs, state = env.reset(jax.random.PRNGKey(0))
            actions = actions or _sleep_actions(env)
            _, _, rewards, _, info = env.step_env(jax.random.PRNGKey(1), state, actions)
            blue = float(rewards["blue_0"])
            red = float(rewards["red_0"])
            damage = float(info["reward_ria"]) + float(info["reward_lwf"])
            if mode == "zero_sum":
                assert red == pytest.approx(-blue)
            else:
                assert red == pytest.approx(-damage)
                # The dropped terms are Blue-only: ASF needs a Blue block and
                # action_cost needs a Blue Restore.
                assert -blue - damage == pytest.approx(-float(info["reward_asf"]) - float(info["action_cost"]))

    def test_blue_payoff_is_identical_across_modes(self):
        rewards = {}
        for mode in ("zero_sum", "damage"):
            env = _joint_env(red_reward=mode)
            _, state = env.reset(jax.random.PRNGKey(3))
            _, _, r, _, _ = env.step_env(jax.random.PRNGKey(4), state, _sleep_actions(env))
            rewards[mode] = float(r["blue_0"])
        assert rewards["zero_sum"] == pytest.approx(rewards["damage"])

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match="red_reward"):
            _joint_env(red_reward="mirror")


class TestMissionSafeBlockMask:
    def test_masked_slots_are_exactly_the_comms_permitted_pairs(self):
        """Cross-check the packed slot mask against decode_blue_action."""
        const = build_topology(jax.random.PRNGKey(0), num_steps=100)
        for phase in range(const.allowed_subnet_pairs.shape[0]):
            allowed = const.allowed_subnet_pairs[phase]
            for agent_id in range(NUM_BLUE_AGENTS):
                permitted = mission_permitted_block_mask(const, agent_id, jnp.int32(phase))
                for slot in range(BLUE_BLOCK_TRAFFIC_END - BLUE_BLOCK_TRAFFIC_START):
                    _, _, _, src, dst = decode_blue_action(BLUE_BLOCK_TRAFFIC_START + slot, agent_id, const)
                    src, dst = int(src), int(dst)
                    expected = dst >= 0 and bool(allowed[src, dst] or allowed[dst, src])
                    assert bool(permitted[slot]) == expected, (phase, agent_id, slot, src, dst)

    def test_mission_safe_only_removes_block_actions(self):
        env = _joint_env(blue_block_policy="mission_safe")
        stock = _joint_env()
        _, state = env.reset(jax.random.PRNGKey(0))
        _, stock_state = stock.reset(jax.random.PRNGKey(0))
        safe_masks = env.get_avail_actions(state)
        stock_masks = stock.get_avail_actions(stock_state)
        for agent in env.blue_agents:
            safe, base = safe_masks[agent], stock_masks[agent]
            assert jnp.all(safe <= base), "mission_safe must never add an action"
            differing = jnp.where(safe != base)[0]
            assert jnp.all(differing >= BLUE_BLOCK_TRAFFIC_START)
            assert jnp.all(differing < BLUE_BLOCK_TRAFFIC_END)

    def test_mission_safe_leaves_some_block_available(self):
        """The free-lunch blocks Red still cares about must survive."""
        const = build_topology(jax.random.PRNGKey(0), num_steps=100)
        for agent_id in range(NUM_BLUE_AGENTS):
            mask = compute_blue_action_mask(
                const,
                agent_id,
                _joint_env().reset(jax.random.PRNGKey(0))[1].state,
                blue_block_policy="mission_safe",
            )
            assert bool(jnp.any(mask[BLUE_BLOCK_TRAFFIC_START:BLUE_BLOCK_TRAFFIC_END]))

    def test_mission_safe_requires_state(self):
        const = build_topology(jax.random.PRNGKey(0), num_steps=100)
        with pytest.raises(ValueError, match="mission phase"):
            compute_blue_action_mask(const, 0, None, blue_block_policy="mission_safe")

    def test_unknown_policy_rejected(self):
        with pytest.raises(ValueError, match="blue_block_policy"):
            GameVariant(name="x", blue_block_policy="off")


class TestRecipePlumbing:
    def test_overrides_reach_train_and_eval_variants(self):
        recipe = {
            "train": {
                "variant": "cc4_stock",
                "variant_overrides": {"red_reward": "damage", "blue_block_policy": "mission_safe"},
            },
            "eval": {"variant": "cia_resilience"},
        }
        train = train_variant(recipe)
        assert (train.red_reward, train.blue_block_policy) == ("damage", "mission_safe")
        ev = eval_variant(recipe)
        assert ev.name == "cia_resilience"
        assert (ev.red_reward, ev.blue_block_policy) == ("damage", "mission_safe")

    def test_overrides_survive_the_scripted_red_hop(self):
        """A Blue trained under a narrower mask must be scored under it too."""
        recipe = {
            "train": {"variant": "cc4_stock", "variant_overrides": {"blue_block_policy": "mission_safe"}},
            "eval": {"variant": "cia_resilience", "red": "cia_a"},
        }
        ev = eval_variant(recipe)
        assert ev.name == "cia_a"
        assert ev.blue_block_policy == "mission_safe"

    def test_variant_for_red_defaults_stay_stock(self):
        assert variant_for_red("fsm").blue_block_policy == "cc4"
        assert variant_for_red("cia_a").red_reward == "zero_sum"

    def test_no_overrides_is_the_registered_variant(self):
        assert train_variant({"train": {"variant": "cc4_stock"}}) is CC4_STOCK

    def test_unknown_override_key_rejected(self):
        with pytest.raises(ValueError, match="unknown train.variant_overrides"):
            train_variant({"train": {"variant": "cc4_stock", "variant_overrides": {"gamma": 0.5}}})
