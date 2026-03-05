from functools import partial
from typing import Dict, Tuple

import chex
import jax
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Box, Discrete

from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END
from jaxborg.actions.masking import compute_blue_action_mask
from jaxborg.agents.fsm_red import (
    fsm_red_post_step_update,
    fsm_red_select_actions,
)
from jaxborg.constants import BLUE_OBS_SIZE, NUM_BLUE_AGENTS, NUM_RED_AGENTS
from jaxborg.env import CC4Env, CC4EnvState


class FsmRedCC4Env(MultiAgentEnv):
    """Blue-only CC4 environment with internal FSM red agents.

    Wraps CC4Env, computes red actions from FSM policy inside step_env,
    and exposes only the 5 blue agents for training.
    """

    def __init__(self, num_steps: int = 500):
        self._env = CC4Env(num_steps=num_steps)
        self.agents = list(self._env.blue_agents)

        super().__init__(num_agents=NUM_BLUE_AGENTS)

        for agent in self.agents:
            self.action_spaces[agent] = Discrete(BLUE_ALLOW_TRAFFIC_END)
            self.observation_spaces[agent] = Box(low=0.0, high=1.0, shape=(BLUE_OBS_SIZE,), dtype=jnp.float32)

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], CC4EnvState]:
        obs, env_state = self._env.reset(key)
        blue_obs = {a: obs[a] for a in self.agents}
        return blue_obs, env_state

    @partial(jax.jit, static_argnums=[0])
    def step(
        self,
        key: chex.PRNGKey,
        state: CC4EnvState,
        actions: Dict[str, chex.Array],
        reset_state=None,
    ) -> Tuple[Dict[str, chex.Array], CC4EnvState, Dict[str, float], Dict[str, bool], Dict]:
        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env(key, state, actions)

        if reset_state is not None:
            states_re = reset_state
        else:
            states_re = self._env._reset_state(states_st, key_reset)
        obs_re = self._get_blue_obs(states_re)

        states = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y),
            states_re,
            states_st,
        )
        obs = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y),
            obs_re,
            obs_st,
        )
        return obs, states, rewards, dones, infos

    @partial(jax.jit, static_argnums=[0])
    def step_env(
        self,
        key: chex.PRNGKey,
        env_state: CC4EnvState,
        blue_actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], CC4EnvState, Dict[str, float], Dict[str, bool], Dict]:
        key, key_red = jax.random.split(key)
        red_keys = jax.random.split(key_red, NUM_RED_AGENTS)

        state_before = env_state.state
        red_action_arr, target_hosts_arr, fsm_actions_arr, eligible_arr, state = fsm_red_select_actions(
            state_before, env_state.const, red_keys
        )
        red_actions = {f"red_{r}": red_action_arr[r] for r in range(NUM_RED_AGENTS)}
        target_hosts = [target_hosts_arr[r] for r in range(NUM_RED_AGENTS)]
        fsm_actions = [fsm_actions_arr[r] for r in range(NUM_RED_AGENTS)]
        eligible_flags = [eligible_arr[r] for r in range(NUM_RED_AGENTS)]
        env_state = CC4EnvState(state=state, const=env_state.const)

        all_actions = {**blue_actions, **red_actions}

        obs, env_state, rewards, dones, info = self._env.step_env(key, env_state, all_actions)

        executed_flags = [env_state.state.red_pending_ticks[r] == 0 for r in range(NUM_RED_AGENTS)]

        new_state = fsm_red_post_step_update(
            state_before,
            env_state.state,
            env_state.const,
            target_hosts,
            fsm_actions,
            eligible_flags,
            executed_flags,
        )
        env_state = CC4EnvState(state=new_state, const=env_state.const)

        blue_obs = {a: obs[a] for a in self.agents}
        blue_rewards = {a: rewards[a] for a in self.agents}
        blue_dones = {a: dones[a] for a in self.agents}
        blue_dones["__all__"] = dones["__all__"]

        return blue_obs, env_state, blue_rewards, blue_dones, info

    @partial(jax.jit, static_argnums=[0])
    def _get_blue_obs(self, env_state: CC4EnvState) -> Dict[str, chex.Array]:
        obs = self._env.get_obs(env_state)
        return {a: obs[a] for a in self.agents}

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, env_state: CC4EnvState) -> Dict[str, chex.Array]:
        return self._get_blue_obs(env_state)

    @partial(jax.jit, static_argnums=[0])
    def get_avail_actions(self, env_state: CC4EnvState) -> Dict[str, chex.Array]:
        return {f"blue_{i}": compute_blue_action_mask(env_state.const, i) for i in range(NUM_BLUE_AGENTS)}

    @property
    def name(self) -> str:
        return "FsmRedCC4"

    @property
    def agent_classes(self) -> dict:
        return {"blue_agents": self.agents}
