"""Joint learned-Blue/learned-Red JAX environment.

The wrapper exposes all eleven policy agents while delegating game mechanics to
one :class:`jaxborg.env.ScenarioEnv` step.  Red compact actions are translated
from the same pre-step simulator state, then submitted together with Blue's
actions.  The underlying raw Red action ABI and FSM wrapper remain untouched.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State
from jaxmarl.environments.spaces import Box, Discrete

from jaxborg.actions.action_defs import BLUE_ALLOW_TRAFFIC_END, BLUE_SLEEP
from jaxborg.actions.masking import compute_blue_action_mask
from jaxborg.actions.red_policy import RED_POLICY_ACTION_DIM, RED_POLICY_SLEEP
from jaxborg.constants import BLUE_OBS_SIZE, CC4_CONFIG, RED_OBS_SIZE
from jaxborg.env import ScenarioEnv, ScenarioEnvState
from jaxborg.learned_red import (
    compact_red_action_to_raw,
    compute_red_policy_action_mask,
    get_red_policy_obs,
)
from jaxborg.observations import get_blue_obs
from jaxborg.scenarios.config import ScenarioConfig


def force_sleep_for_unavailable_actions(
    state: ScenarioEnvState,
    actions: Dict[str, chex.Array],
) -> Dict[str, chex.Array]:
    """Force policy submissions to Sleep while their agent cannot choose.

    Pending actions still progress inside the simulator.  Replacing a newly
    submitted action with Sleep prevents it from being charged or attributed to
    the actor while preserving the pending action's value-learning transition.
    """

    sim = state.state
    out: Dict[str, chex.Array] = {}
    for b in range(sim.blue_pending_ticks.shape[0]):
        submitted = jnp.asarray(actions[f"blue_{b}"], dtype=jnp.int32)
        valid = (submitted >= 0) & (submitted < BLUE_ALLOW_TRAFFIC_END)
        out[f"blue_{b}"] = jnp.where(
            (sim.blue_pending_ticks[b] <= 0) & valid,
            submitted,
            jnp.int32(BLUE_SLEEP),
        )
    for r in range(sim.red_pending_ticks.shape[0]):
        submitted = jnp.asarray(actions[f"red_{r}"], dtype=jnp.int32)
        valid = (submitted >= 0) & (submitted < RED_POLICY_ACTION_DIM)
        available = sim.red_agent_active[r] & (sim.red_pending_ticks[r] <= 0) & valid
        out[f"red_{r}"] = jnp.where(available, submitted, jnp.int32(RED_POLICY_SLEEP))
    return out


class JointPolicyCC4Env(MultiAgentEnv):
    """Policy-facing CC4 env for simultaneous learned Blue and Red play."""

    def __init__(
        self,
        num_steps: Optional[int] = None,
        *,
        topology_mode: str = "generative",
        training_mode: bool = False,
        topology_path: str | Path | Sequence[str | Path] | None = None,
        scenario_config: ScenarioConfig = CC4_CONFIG,
        op_zone_min_servers: int | None = None,
        name: str | None = None,
    ):
        self._env = ScenarioEnv(
            num_steps=num_steps,
            topology_mode=topology_mode,
            training_mode=training_mode,
            topology_path=topology_path,
            scenario_config=scenario_config,
            op_zone_min_servers=op_zone_min_servers,
        )
        self.cfg = scenario_config
        self.num_steps = self._env.num_steps
        self.training_mode = training_mode
        self.topology_mode = self._env.topology_mode
        self.blue_agents = list(self._env.blue_agents)
        self.red_agents = list(self._env.red_agents)
        self._name = name or "JointPolicyCC4"

        super().__init__(num_agents=len(self.blue_agents) + len(self.red_agents))
        # JaxMARL's base constructor installs generic ``agent_N`` names.
        # Preserve the policy-facing Blue/Red identifiers used by this env.
        self.agents = self.blue_agents + self.red_agents
        for agent in self.blue_agents:
            self.action_spaces[agent] = Discrete(BLUE_ALLOW_TRAFFIC_END)
            self.observation_spaces[agent] = Box(
                low=0.0,
                high=1.0,
                shape=(BLUE_OBS_SIZE,),
                dtype=jnp.float32,
            )
        for agent in self.red_agents:
            self.action_spaces[agent] = Discrete(RED_POLICY_ACTION_DIM)
            self.observation_spaces[agent] = Box(
                low=0.0,
                high=1.0,
                shape=(RED_OBS_SIZE,),
                dtype=jnp.float32,
            )

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], ScenarioEnvState]:
        _, env_state = self._env.reset(key)
        return self.get_obs(env_state), env_state

    @partial(jax.jit, static_argnums=[0])
    def _translate_actions(
        self,
        key: chex.PRNGKey,
        env_state: ScenarioEnvState,
        actions: Dict[str, chex.Array],
    ) -> Dict[str, chex.Array]:
        submitted = force_sleep_for_unavailable_actions(env_state, actions)
        red_keys = jax.random.split(key, self.cfg.num_red_agents)
        raw_actions: Dict[str, chex.Array] = {agent: submitted[agent] for agent in self.blue_agents}
        for r, agent in enumerate(self.red_agents):
            raw_actions[agent] = compact_red_action_to_raw(
                env_state.state,
                env_state.const,
                r,
                submitted[agent],
                red_keys[r],
            )
        return raw_actions

    @partial(jax.jit, static_argnums=[0])
    def step(
        self,
        key: chex.PRNGKey,
        state: ScenarioEnvState,
        actions: Dict[str, chex.Array],
        reset_state: Optional[State] = None,
    ) -> Tuple[Dict[str, chex.Array], ScenarioEnvState, Dict[str, float], Dict[str, bool], Dict]:
        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, info = self.step_env(key, state, actions)

        if reset_state is not None:
            states_re = reset_state
        else:
            states_re = self._env._reset_state(states_st, key_reset)
        obs_re = self.get_obs(states_re)
        next_state = jax.tree.map(
            lambda reset_value, step_value: jax.lax.select(dones["__all__"], reset_value, step_value),
            states_re,
            states_st,
        )
        obs = jax.tree.map(
            lambda reset_value, step_value: jax.lax.select(dones["__all__"], reset_value, step_value),
            obs_re,
            obs_st,
        )
        return obs, next_state, rewards, dones, info

    @partial(jax.jit, static_argnums=[0])
    def step_env(
        self,
        key: chex.PRNGKey,
        env_state: ScenarioEnvState,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], ScenarioEnvState, Dict[str, float], Dict[str, bool], Dict]:
        key_translate, key_env = jax.random.split(key)
        raw_actions = self._translate_actions(key_translate, env_state, actions)
        # Learned Red explicitly binds source/session actions to session 0.
        # The raw simulator's FSM path normally models a random choice from N
        # visible abstract sessions via a 1/N roll; overriding N to one removes
        # that FSM-only randomness without changing ScenarioEnv's default.
        primary_session_only = jnp.ones(self.cfg.num_red_agents, dtype=jnp.int32)
        _, next_state, rewards, dones, info = self._env.step_env(
            key_env,
            env_state,
            raw_actions,
            red_creation_visible_sessions_override=primary_session_only,
        )
        return self.get_obs(next_state), next_state, rewards, dones, info

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, env_state: ScenarioEnvState) -> Dict[str, chex.Array]:
        obs: Dict[str, chex.Array] = {}
        for b, agent in enumerate(self.blue_agents):
            obs[agent] = get_blue_obs(env_state.state, env_state.const, b)
        for r, agent in enumerate(self.red_agents):
            obs[agent] = get_red_policy_obs(env_state.state, env_state.const, r)
        return obs

    @partial(jax.jit, static_argnums=[0])
    def get_avail_actions(self, env_state: ScenarioEnvState) -> Dict[str, chex.Array]:
        masks: Dict[str, chex.Array] = {}
        blue_sleep_only = jnp.zeros(BLUE_ALLOW_TRAFFIC_END, dtype=jnp.bool_).at[BLUE_SLEEP].set(True)
        for b, agent in enumerate(self.blue_agents):
            base = compute_blue_action_mask(env_state.const, b, env_state.state)
            masks[agent] = jnp.where(env_state.state.blue_pending_ticks[b] <= 0, base, blue_sleep_only)
        for r, agent in enumerate(self.red_agents):
            masks[agent] = compute_red_policy_action_mask(env_state.state, env_state.const, r)
        return masks

    @property
    def name(self) -> str:
        return self._name

    @property
    def agent_classes(self) -> dict:
        return {
            "blue_agents": self.blue_agents,
            "red_agents": self.red_agents,
        }


__all__ = ["JointPolicyCC4Env", "force_sleep_for_unavailable_actions"]
