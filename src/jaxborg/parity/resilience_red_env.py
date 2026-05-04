"""Blue-only CC4 environment with resilience topology and biased FSM red agents.

Drop-in replacement for FsmRedCC4Env: same MultiAgentEnv interface, but uses
build_resilience_topology to assign auth/db/web roles to three Operational Zone
servers each episode, and routes red action selection through
resilience_red_select_actions so those servers are targeted preferentially.

The extra per-episode data (host_resilience_role) is carried in the extended
state type ResilienceEnvState alongside the inner ScenarioEnvState.
"""

from __future__ import annotations

from functools import partial
from typing import Dict, NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Box, Discrete

from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END
from jaxborg.actions.masking import compute_blue_action_mask
from jaxborg.constants import BLUE_OBS_SIZE, GLOBAL_MAX_HOSTS, NUM_BLUE_AGENTS, NUM_RED_AGENTS
from jaxborg.env import ScenarioEnv, ScenarioEnvState
from jaxborg.scenarios.cc4.red_fsm import (
    fsm_red_apply_delayed_update,
    fsm_red_schedule_post_step_update,
)
from jaxborg.scenarios.cc4.resilience_red_fsm import resilience_red_select_actions
from jaxborg.scenarios.cc4.resilience_topology import _assign_resilience_roles


class ResilienceEnvState(NamedTuple):
    env_state: ScenarioEnvState
    host_resilience_role: chex.Array  # (GLOBAL_MAX_HOSTS,) int32


class ResilienceRedCC4Env(MultiAgentEnv):
    """Blue-only CC4 env with resilience topology and resilience-biased FSM red agents.

    Each episode the three Operational Zone servers are assigned auth/db/web roles
    via _assign_resilience_roles.  The red FSM then targets those servers with
    RESILIENCE_TARGET_WEIGHT times higher probability than ordinary hosts.
    """

    def __init__(
        self,
        num_steps: int = 500,
        *,
        topology_mode: str = "generative",
        topology_bank_size: int = 0,
        sync_red_policy_bank: bool = False,
        training_mode: bool = False,
    ):
        self._env = ScenarioEnv(
            num_steps=num_steps,
            topology_mode=topology_mode,
            topology_bank_size=topology_bank_size,
            sync_red_policy_bank=sync_red_policy_bank,
            training_mode=training_mode,
        )
        self.agents = list(self._env.blue_agents)
        super().__init__(num_agents=NUM_BLUE_AGENTS)
        for agent in self.agents:
            self.action_spaces[agent] = Discrete(BLUE_ALLOW_TRAFFIC_END)
            self.observation_spaces[agent] = Box(low=0.0, high=1.0, shape=(BLUE_OBS_SIZE,), dtype=jnp.float32)

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], ResilienceEnvState]:
        key, key_roles = jax.random.split(key)
        obs, env_state = self._env.reset(key)
        env_state = self._strip_inactive_red_reset_knowledge(env_state)
        host_resilience_role = _assign_resilience_roles(key_roles, env_state.const)
        blue_obs = {a: obs[a] for a in self.agents}
        return blue_obs, ResilienceEnvState(env_state=env_state, host_resilience_role=host_resilience_role)

    def _strip_inactive_red_reset_knowledge(self, env_state: ScenarioEnvState) -> ScenarioEnvState:
        state = env_state.state
        inactive = ~state.red_agent_active
        state = state.replace(
            red_discovered_hosts=jnp.where(inactive[:, None], False, state.red_discovered_hosts),
            red_scanned_hosts=jnp.where(inactive[:, None], False, state.red_scanned_hosts),
            red_scanned_source_hosts=jnp.where(inactive[:, None, None], False, state.red_scanned_source_hosts),
            red_scan_source_pid=jnp.where(inactive[:, None], jnp.int32(-1), state.red_scan_source_pid),
            red_scan_anchor_host=jnp.where(inactive, jnp.int32(-1), state.red_scan_anchor_host),
            red_primary_is_abstract=jnp.where(inactive, True, state.red_primary_is_abstract),
            red_primary_pid=jnp.where(inactive, jnp.int32(-1), state.red_primary_pid),
            fsm_host_entered=jnp.where(inactive[:, None], False, state.fsm_host_entered),
        )
        return ScenarioEnvState(state=state, const=env_state.const)

    @partial(jax.jit, static_argnums=[0])
    def step(
        self,
        key: chex.PRNGKey,
        state: ResilienceEnvState,
        actions: Dict[str, chex.Array],
        reset_state=None,
    ) -> Tuple[Dict[str, chex.Array], ResilienceEnvState, Dict[str, float], Dict[str, bool], Dict]:
        key, key_reset, key_roles = jax.random.split(key, 3)
        obs_st, states_st, rewards, dones, infos = self.step_env(key, state, actions)

        if reset_state is not None:
            inner_re = reset_state.env_state
            roles_re = reset_state.host_resilience_role
        else:
            inner_re = self._env._reset_state(states_st.env_state, key_reset)
            inner_re = self._strip_inactive_red_reset_knowledge(inner_re)
            roles_re = _assign_resilience_roles(key_roles, inner_re.const)

        states_re = ResilienceEnvState(env_state=inner_re, host_resilience_role=roles_re)
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
        resilience_state: ResilienceEnvState,
        blue_actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], ResilienceEnvState, Dict[str, float], Dict[str, bool], Dict]:
        env_state = resilience_state.env_state
        host_resilience_role = resilience_state.host_resilience_role

        key, key_red = jax.random.split(key)
        red_keys = jax.random.split(key_red, NUM_RED_AGENTS)

        state_before = fsm_red_apply_delayed_update(env_state.state)
        active_before = state_before.red_agent_active
        (
            red_action_arr,
            target_hosts_arr,
            target_subnets_arr,
            fsm_actions_arr,
            eligible_arr,
            state,
        ) = resilience_red_select_actions(state_before, env_state.const, host_resilience_role, red_keys)

        red_actions = {f"red_{r}": red_action_arr[r] for r in range(NUM_RED_AGENTS)}
        target_hosts = [target_hosts_arr[r] for r in range(NUM_RED_AGENTS)]
        target_subnets = [target_subnets_arr[r] for r in range(NUM_RED_AGENTS)]
        fsm_actions = [fsm_actions_arr[r] for r in range(NUM_RED_AGENTS)]
        eligible_flags = [eligible_arr[r] for r in range(NUM_RED_AGENTS)]
        env_state = ScenarioEnvState(state=state, const=env_state.const)

        all_actions = {**blue_actions, **red_actions}
        obs, env_state, rewards, dones, info = self._env.step_env(key, env_state, all_actions)

        newly_active = ~active_before & env_state.state.red_agent_active
        discovered = env_state.state.red_discovered_hosts
        for r in range(NUM_RED_AGENTS):
            start_h = env_state.const.red_start_hosts[r]
            has_session_at_start = env_state.state.red_sessions[r, start_h]
            discovered = jnp.where(
                newly_active[r] & ~has_session_at_start,
                discovered.at[r, start_h].set(False),
                discovered,
            )
        env_state = ScenarioEnvState(
            state=env_state.state.replace(red_discovered_hosts=discovered),
            const=env_state.const,
        )

        executed_flags = [env_state.state.red_pending_ticks[r] == 0 for r in range(NUM_RED_AGENTS)]
        new_state = fsm_red_schedule_post_step_update(
            state_before,
            env_state.state,
            env_state.const,
            target_hosts,
            target_subnets,
            fsm_actions,
            eligible_flags,
            executed_flags,
        )
        env_state = ScenarioEnvState(state=new_state, const=env_state.const)
        out_state = ResilienceEnvState(env_state=env_state, host_resilience_role=host_resilience_role)

        blue_obs = {a: obs[a] for a in self.agents}
        blue_rewards = {a: rewards[a] for a in self.agents}
        blue_dones = {a: dones[a] for a in self.agents}
        blue_dones["__all__"] = dones["__all__"]
        return blue_obs, out_state, blue_rewards, blue_dones, info

    @partial(jax.jit, static_argnums=[0])
    def _get_blue_obs(self, resilience_state: ResilienceEnvState) -> Dict[str, chex.Array]:
        obs = self._env.get_obs(resilience_state.env_state)
        return {a: obs[a] for a in self.agents}

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, resilience_state: ResilienceEnvState) -> Dict[str, chex.Array]:
        return self._get_blue_obs(resilience_state)

    @partial(jax.jit, static_argnums=[0])
    def get_avail_actions(self, resilience_state: ResilienceEnvState) -> Dict[str, chex.Array]:
        env_state = resilience_state.env_state
        return {
            f"blue_{i}": compute_blue_action_mask(env_state.const, i, env_state.state)
            for i in range(NUM_BLUE_AGENTS)
        }

    @property
    def name(self) -> str:
        return "ResilienceRedCC4"

    @property
    def agent_classes(self) -> dict:
        return {"blue_agents": self.agents}
