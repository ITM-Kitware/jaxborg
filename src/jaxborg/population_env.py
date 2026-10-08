"""Episode-stable learned/scripted Red dispatch with the joint policy API."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from jaxborg.env import ScenarioEnvState
from jaxborg.evaluation.jax_env_factory import make_jax_env, make_joint_jax_env
from jaxborg.opponent_population import OPPONENT_NAMES, OpponentPopulationSettings
from jaxborg.parity.fsm_red_env import FsmRedEnvState, _empty_extras_factory
from jaxborg.scenarios.cc4.topology_roles import assign_resilience_roles_from_const


@struct.dataclass
class PopulationEnvState:
    state: Any
    const: Any
    extras: dict
    opponent_id: jax.Array
    supplemental: jax.Array


class PopulationCC4Env:
    """Compose the existing joint and scripted environments without changing rules.

    With preservation enabled, a batch contains N primary games followed by
    N supplemental learned-Red games. Topologies are sampled without replacement
    separately within each batch, so the original bank-size requirement holds.
    """

    def __init__(self, variant, settings: OpponentPopulationSettings, **kwargs):
        self.settings = settings
        self.joint = make_joint_jax_env(variant, **kwargs)
        self.scripted = {
            name: make_jax_env(
                replace(variant, red_agent=name, resilience_roles=name != "fsm", target_weight=10.0), **kwargs
            )
            for name, percentage in zip(OPPONENT_NAMES[1:], settings.percentages[1:], strict=True)
            if percentage > 0
        }
        self.has_roles = any(name.startswith("cia_") for name in self.scripted)
        self._reset_fsm = next(iter(self.scripted.values()))

    def __getattr__(self, name):
        # Observations, action masks, spaces, and critic inputs retain their
        # existing joint interfaces and never expose population identity.
        return getattr(self.joint, name)

    def _wrap_reset(self, inner, key, supplemental=False):
        opponent_key, extras_key = jax.random.split(key)
        opponent_id = jnp.where(supplemental, jnp.int32(0), self.settings.sample(opponent_key))
        inner = jax.lax.cond(opponent_id != 0, self._reset_fsm._strip_inactive_red_reset_knowledge, lambda s: s, inner)
        extras = _empty_extras_factory(extras_key, inner.const)
        if self.has_roles:
            extras = {"host_resilience_role": assign_resilience_roles_from_const(inner.const, extras_key)}
        return PopulationEnvState(inner.state, inner.const, extras, opponent_id, jnp.asarray(supplemental))

    def reset(self, key):
        key_env, key_population = jax.random.split(key)
        _, inner = self.joint.reset(key_env)
        state = self._wrap_reset(inner, key_population)
        return self.get_obs(state), state

    def reset_at_topology(self, key, topology_index):
        key_env, key_population = jax.random.split(key)
        _, inner = self.joint.reset_at_topology(key_env, topology_index)
        state = self._wrap_reset(inner, key_population)
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=0)
    def reset_batch(self, keys, topology_key):
        split_keys = jax.vmap(jax.random.split)(keys)
        groups = 2 if self.settings.preserve_red_batch_size else 1
        size = keys.shape[0] // groups
        states = []
        for group in range(groups):
            part = slice(group * size, (group + 1) * size)
            _, inner = self.joint.reset_batch(split_keys[part, 0], jax.random.fold_in(topology_key, group))
            states.append(jax.vmap(lambda s, k: self._wrap_reset(s, k, group == 1))(inner, split_keys[part, 1]))
        state = jax.tree.map(lambda *xs: jnp.concatenate(xs), *states)
        return jax.vmap(self.get_obs)(state), state

    @partial(jax.jit, static_argnums=0)
    def step_env(self, key, state, actions):
        def learned(_):
            _, inner, rewards, dones, info = self.joint.step_env(
                key, ScenarioEnvState(state.state, state.const), actions
            )
            return inner, rewards, dones, info

        def scripted_branch(env):
            def step(_):
                _, inner, blue_rewards, blue_dones, info = env.step_env(
                    key,
                    FsmRedEnvState(state.state, state.const, state.extras),
                    {name: actions[name] for name in self.blue_agents},
                )
                # These synthetic Red rewards are masked out of PPO; expose a
                # homogeneous joint API while reusing the complete FSM step.
                rewards = {**blue_rewards, **dict.fromkeys(self.red_agents, -blue_rewards[self.blue_agents[0]])}
                dones = {**blue_dones, **dict.fromkeys(self.red_agents, blue_dones["__all__"])}
                return ScenarioEnvState(inner.state, inner.const), rewards, dones, info

            return step

        branches = [learned] + [
            scripted_branch(self.scripted[name]) if name in self.scripted else learned for name in OPPONENT_NAMES[1:]
        ]
        inner, rewards, dones, info = jax.lax.switch(state.opponent_id, branches, None)
        next_state = state.replace(state=inner.state, const=inner.const)
        return self.get_obs(next_state), next_state, rewards, dones, info

    def _reset_state(self, state, key):
        key_env, key_population = jax.random.split(key)
        inner = self.joint._env._reset_state(ScenarioEnvState(state.state, state.const), key_env)
        return self._wrap_reset(inner, key_population, state.supplemental)

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, actions):
        key_step, key_reset = jax.random.split(key)
        obs, next_state, rewards, dones, info = self.step_env(key_step, state, actions)
        obs, next_state = jax.lax.cond(
            dones["__all__"],
            lambda _: self._reset_observation(next_state, key_reset),
            lambda _: (obs, next_state),
            None,
        )
        return obs, next_state, rewards, dones, info

    def _reset_observation(self, state, key):
        reset = self._reset_state(state, key)
        return self.get_obs(reset), reset

    @partial(jax.jit, static_argnums=0)
    def step_batch(self, keys, states, actions, topology_key=None):
        split_keys = jax.vmap(jax.random.split)(keys)
        obs, next_states, rewards, dones, info = jax.vmap(self.step_env)(split_keys[:, 0], states, actions)
        done = dones["__all__"]

        def reset_finished(_):
            if topology_key is None:
                reset_states = jax.vmap(self._reset_state)(next_states, split_keys[:, 1])
                reset_obs = jax.vmap(self.get_obs)(reset_states)
            else:
                reset_obs, reset_states = self.reset_batch(split_keys[:, 1], topology_key)

            def select(reset_value, step_value):
                mask = done.reshape(done.shape + (1,) * (step_value.ndim - done.ndim))
                return jnp.where(mask, reset_value, step_value)

            return jax.tree.map(select, (reset_obs, reset_states), (obs, next_states))

        obs, next_states = jax.lax.cond(jnp.any(done), reset_finished, lambda _: (obs, next_states), None)
        return obs, next_states, rewards, dones, info
