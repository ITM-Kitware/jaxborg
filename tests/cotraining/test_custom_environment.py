"""Bounded CPU exercise of an experiment-local payoff through both loops."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.evaluation import matchup_runner
from jaxborg.evaluation.environment_contract import construct_joint_env
from jaxborg.evaluation.jax_env_factory import make_joint_jax_env
from jaxborg.evaluation.matchup_runner import LoadedMatchupPolicy, MatchupEvaluationContext, evaluate_matchup
from jaxborg.policies import make_jax_policy
from jaxborg.scenarios.cc4.game_variants import CC4_STOCK
from scripts.train.algorithms import ippo_jax_joint as joint
from tests.cotraining.test_jax_joint_trainer import _config

_spec = importlib.util.spec_from_file_location(
    "reward_offset_rules", Path(__file__).parents[2] / "experiments/reward-offset/rules.py"
)
rules = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rules)


def test_custom_payoff_reaches_training_and_evaluation(monkeypatch):
    monkeypatch.setenv("JAXBORG_EVAL_BATCH_SIZE", "1")
    variant = replace(CC4_STOCK, num_steps=1)
    configs = {team: {**_config(), "TRAIN_VARIANT": variant, "NUM_STEPS": 1} for team in joint.TEAMS}
    reference = make_joint_jax_env(variant)
    networks = {
        team: make_jax_policy(
            "separate",
            action_dim=reference.action_space(getattr(reference, f"{team}_agents")[0]).n,
            hidden_dim=4,
            hidden_layers=1,
        )
        for team in joint.TEAMS
    }
    key = jax.random.PRNGKey(0)
    reset_shape = jax.eval_shape(reference.reset, key)
    actions = {name: jnp.int32(0) for name in (*reference.blue_agents, *reference.red_agents)}
    step_shape = jax.eval_shape(reference.step_env, key, reset_shape[1], actions)

    def zeros(tree):
        return jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), tree)

    class SyntheticEnv(rules.RewardOffsetEnv):
        """One deterministic terminal transition with the canonical CC4 ABI."""

        def reset(self, key):
            del key
            return zeros(reset_shape)

        def get_avail_actions(self, state):
            del state
            return {name: jnp.ones(self.action_space(name).n, dtype=jnp.bool_) for name in actions}

        def step_env(self, key, state, actions):
            del key, actions
            obs, _, rewards, dones, info = zeros(step_shape)
            state = state.replace(state=state.state.replace(time=state.state.time + 1))
            rewards, info = rules.offset_payoff(rewards, info, self.blue_agents, self.offset)
            return obs, state, rewards, jax.tree.map(lambda x: jnp.ones_like(x), dones), info

        def step(self, key, state, actions):
            return self.step_env(key, state, actions)

        def step_batch(self, keys, states, actions, topology_key=None):
            del topology_key
            return jax.vmap(self.step)(keys, states, actions)

        def reset_batch(self, keys, topology_key):
            del topology_key
            return jax.vmap(self.reset)(keys)

    # The real example is traced against the canonical simulator as well.
    construct_joint_env(variant, env_factory=rules.build_env, default_factory=make_joint_jax_env)
    supports_batched = matchup_runner._supports_batched_eval
    results = []
    for batched, offset in ((False, 3.0), (True, 7.0)):

        def builder(variant, **settings):
            return SyntheticEnv(offset=offset, num_steps=variant.num_steps, **settings)

        env, obs, state, init, update = joint.make_joint_train(
            configs, networks, trainable_teams=("blue", "red"), env_factory=builder, use_batched_reset=batched
        )
        states = init(jax.random.PRNGKey(4))
        norms = {team: joint.initial_reward_norm_state(1) for team in joint.TEAMS}
        *_, metrics = update(states, state, obs, jax.random.PRNGKey(7), norms)
        context = MatchupEvaluationContext()
        policies = {
            team: LoadedMatchupPolicy(team, "jax", networks[team], states[team].params, {}) for team in joint.TEAMS
        }
        monkeypatch.setattr(context, "load_policy", lambda path, *, team, backend: policies[team])
        monkeypatch.setattr(
            matchup_runner, "_supports_batched_eval", supports_batched if batched else lambda env: False
        )
        evaluation = evaluate_matchup(
            "unused-blue",
            "unused-red",
            backend="jax",
            variant=variant,
            seeds=[42],
            progress=False,
            deterministic=True,
            context=context,
            env_factory=builder,
        )
        results.append(
            (float(metrics["blue"]["mean_rollout_return"]), evaluation.blue_returns[0], evaluation.red_returns[0])
        )
    np.testing.assert_allclose(results, [[3, 3, -3], [7, 7, -7]], atol=1e-5)


@pytest.mark.parametrize("failure", ["state", "dtype"])
def test_invalid_contract_rejected_before_rollout(failure):
    class Invalid(rules.RewardOffsetEnv):
        def reset(self, key):
            obs, state = super().reset(key)
            if failure == "state":
                return obs, (state, jnp.int32(0))
            return {name: value.astype(jnp.int32) for name, value in obs.items()}, state

    def builder(variant, **settings):
        return Invalid(offset=0, num_steps=variant.num_steps, **settings)

    with pytest.raises(ValueError, match="unsupported custom joint JAX environment contract.*reset output"):
        construct_joint_env(replace(CC4_STOCK, num_steps=1), env_factory=builder, default_factory=make_joint_jax_env)


def test_non_zero_sum_contract_rejected_before_construction():
    def builder(*args, **kwargs):
        raise AssertionError("unsupported payoff must fail before construction")

    with pytest.raises(ValueError, match="red_reward=zero_sum"):
        construct_joint_env(
            replace(CC4_STOCK, red_reward="damage"), env_factory=builder, default_factory=make_joint_jax_env
        )
