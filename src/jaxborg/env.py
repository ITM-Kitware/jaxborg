from functools import partial
from typing import Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State
from jaxmarl.environments.spaces import Box, Discrete

from jaxborg.actions.duration import process_blue_with_duration, process_red_with_duration
from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END, RED_WITHDRAW_END
from jaxborg.actions.green import apply_green_agents
from jaxborg.actions.masking import compute_blue_action_mask
from jaxborg.actions.pids import append_pid_to_row
from jaxborg.agents.fsm_red import fsm_red_init_states
from jaxborg.constants import (
    BLUE_OBS_SIZE,
    COMPROMISE_USER,
    GLOBAL_MAX_HOSTS,
    NUM_BLUE_AGENTS,
    NUM_RED_AGENTS,
)
from jaxborg.observations import get_blue_obs, get_red_obs
from jaxborg.reassignment import reassign_cross_subnet_sessions
from jaxborg.rewards import advance_mission_phase, compute_rewards
from jaxborg.state import CC4Const, CC4State, create_initial_state
from jaxborg.topology import build_topology


@struct.dataclass
class CC4EnvState:
    state: CC4State
    const: CC4Const


def _init_red_state(const: CC4Const, state: CC4State) -> CC4State:
    red_sessions = state.red_sessions
    red_session_count = state.red_session_count
    red_privilege = state.red_privilege
    red_discovered = state.red_discovered_hosts | const.red_initial_discovered_hosts
    red_scanned = state.red_scanned_hosts | const.red_initial_scanned_hosts
    fsm_states = state.fsm_host_states
    host_compromised = state.host_compromised
    red_scan_anchor_host = state.red_scan_anchor_host
    red_session_is_abstract = state.red_session_is_abstract
    red_abstract_host_rank = state.red_abstract_host_rank
    red_next_abstract_rank = state.red_next_abstract_rank
    red_scanned_via = state.red_scanned_via
    red_scanned_source_hosts = state.red_scanned_source_hosts
    red_session_pids = state.red_session_pids
    red_next_pid = state.red_next_pid

    for r in range(NUM_RED_AGENTS):
        start_host = const.red_start_hosts[r]
        is_active = const.red_agent_active[r]
        red_sessions = jnp.where(
            is_active,
            red_sessions.at[r, start_host].set(True),
            red_sessions,
        )
        red_session_count = jnp.where(
            is_active,
            red_session_count.at[r, start_host].set(1),
            red_session_count,
        )
        red_session_is_abstract = jnp.where(
            is_active,
            red_session_is_abstract.at[r, start_host].set(True),
            red_session_is_abstract,
        )
        red_abstract_host_rank = jnp.where(
            is_active,
            red_abstract_host_rank.at[r, start_host].set(0),
            red_abstract_host_rank,
        )
        red_next_abstract_rank = jnp.where(
            is_active,
            red_next_abstract_rank.at[r].set(1),
            red_next_abstract_rank,
        )
        pid_row = red_session_pids[r, start_host]
        red_session_pids = jnp.where(
            is_active,
            red_session_pids.at[r, start_host].set(append_pid_to_row(pid_row, red_next_pid)),
            red_session_pids,
        )
        red_next_pid = jnp.where(is_active, red_next_pid + 1, red_next_pid)
        red_privilege = jnp.where(
            is_active,
            red_privilege.at[r, start_host].set(COMPROMISE_USER),
            red_privilege,
        )
        red_discovered = jnp.where(
            is_active,
            red_discovered.at[r, start_host].set(True),
            red_discovered,
        )
        host_compromised = jnp.where(
            is_active,
            host_compromised.at[start_host].set(jnp.maximum(host_compromised[start_host], COMPROMISE_USER)),
            host_compromised,
        )
        fsm_states = jnp.where(
            is_active,
            fsm_states.at[r].set(fsm_red_init_states(const, r)),
            fsm_states,
        )
        red_scan_anchor_host = jnp.where(
            is_active,
            red_scan_anchor_host.at[r].set(start_host),
            red_scan_anchor_host,
        )
        initially_scanned = const.red_initial_scanned_hosts[r]
        new_via_row = jnp.where(initially_scanned, start_host, red_scanned_via[r])
        red_scanned_via = jnp.where(
            is_active,
            red_scanned_via.at[r].set(new_via_row),
            red_scanned_via,
        )
        red_scanned_source_hosts = jnp.where(
            is_active,
            red_scanned_source_hosts.at[r, :, start_host].set(initially_scanned),
            red_scanned_source_hosts,
        )

    return state.replace(
        red_sessions=red_sessions,
        red_session_count=red_session_count,
        red_privilege=red_privilege,
        red_discovered_hosts=red_discovered,
        red_scanned_hosts=red_scanned,
        red_scanned_via=red_scanned_via,
        red_scanned_source_hosts=red_scanned_source_hosts,
        red_scan_anchor_host=red_scan_anchor_host,
        host_compromised=host_compromised,
        fsm_host_states=fsm_states,
        red_session_is_abstract=red_session_is_abstract,
        red_abstract_host_rank=red_abstract_host_rank,
        red_next_abstract_rank=red_next_abstract_rank,
        red_session_pids=red_session_pids,
        red_next_pid=red_next_pid,
    )


class CC4Env(MultiAgentEnv):
    def __init__(self, num_steps: int = 500):
        self.num_steps = num_steps

        self.blue_agents = [f"blue_{i}" for i in range(NUM_BLUE_AGENTS)]
        self.red_agents = [f"red_{i}" for i in range(NUM_RED_AGENTS)]
        self.agents = self.blue_agents + self.red_agents

        super().__init__(num_agents=NUM_BLUE_AGENTS + NUM_RED_AGENTS)

        for agent in self.blue_agents:
            self.action_spaces[agent] = Discrete(BLUE_ALLOW_TRAFFIC_END)
            self.observation_spaces[agent] = Box(low=0.0, high=1.0, shape=(BLUE_OBS_SIZE,), dtype=jnp.float32)
        for agent in self.red_agents:
            self.action_spaces[agent] = Discrete(RED_WITHDRAW_END)
            self.observation_spaces[agent] = Box(low=0.0, high=1.0, shape=(BLUE_OBS_SIZE,), dtype=jnp.float32)

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], CC4EnvState]:
        const = build_topology(key, num_steps=self.num_steps)
        state = create_initial_state()
        state = state.replace(host_services=jnp.array(const.initial_services))
        state = _init_red_state(const, state)

        env_state = CC4EnvState(state=state, const=const)
        obs = self.get_obs(env_state)
        return obs, env_state

    @partial(jax.jit, static_argnums=[0])
    def _reset_state(self, env_state: CC4EnvState) -> CC4EnvState:
        """Reset dynamic state while keeping the same topology (for auto-reset)."""
        const = env_state.const
        state = create_initial_state()
        state = state.replace(host_services=jnp.array(const.initial_services))
        state = _init_red_state(const, state)
        return CC4EnvState(state=state, const=const)

    @partial(jax.jit, static_argnums=[0])
    def step(
        self,
        key: chex.PRNGKey,
        state: CC4EnvState,
        actions: Dict[str, chex.Array],
        reset_state: Optional[State] = None,
    ) -> Tuple[Dict[str, chex.Array], CC4EnvState, Dict[str, float], Dict[str, bool], Dict]:
        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env(key, state, actions)

        if reset_state is not None:
            states_re = reset_state
        else:
            states_re = self._reset_state(states_st)
        obs_re = self.get_obs(states_re)

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
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], CC4EnvState, Dict[str, float], Dict[str, bool], Dict]:
        state = env_state.state
        const = env_state.const

        key, key_green, *red_keys = jax.random.split(key, 2 + NUM_RED_AGENTS)

        state = advance_mission_phase(state, const)

        state = state.replace(
            red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32),
            green_lwf_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_),
            green_asf_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_),
        )

        ot_before = state.ot_service_stopped

        for r in range(NUM_RED_AGENTS):
            state = process_red_with_duration(state, const, r, actions[f"red_{r}"], red_keys[r])

        impact_hosts = state.ot_service_stopped & ~ot_before

        state = apply_green_agents(state, const, key_green)

        for b in range(NUM_BLUE_AGENTS):
            state = process_blue_with_duration(state, const, b, actions[f"blue_{b}"])

        state = reassign_cross_subnet_sessions(state, const)

        reward = compute_rewards(
            state,
            const,
            impact_hosts,
            state.green_lwf_this_step,
            state.green_asf_this_step,
        )

        state = state.replace(time=state.time + 1)
        done = state.time >= const.max_steps
        state = state.replace(done=jnp.array(done))

        env_state = CC4EnvState(state=state, const=const)
        obs = self.get_obs(env_state)

        rewards = {}
        for agent in self.blue_agents:
            rewards[agent] = reward
        neg_reward = -reward
        for agent in self.red_agents:
            rewards[agent] = neg_reward

        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done

        info = {}

        return obs, env_state, rewards, dones, info

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, env_state: CC4EnvState) -> Dict[str, chex.Array]:
        state = env_state.state
        const = env_state.const
        obs = {}
        for b in range(NUM_BLUE_AGENTS):
            obs[f"blue_{b}"] = get_blue_obs(state, const, b)
        for r in range(NUM_RED_AGENTS):
            obs[f"red_{r}"] = get_red_obs(state, const, r)
        return obs

    @partial(jax.jit, static_argnums=[0])
    def get_avail_actions(self, env_state: CC4EnvState) -> Dict[str, chex.Array]:
        masks = {}
        for i in range(NUM_BLUE_AGENTS):
            masks[f"blue_{i}"] = compute_blue_action_mask(env_state.const, i)
        for agent in self.red_agents:
            masks[agent] = jnp.ones(RED_WITHDRAW_END, dtype=jnp.bool_)
        return masks

    @property
    def name(self) -> str:
        return "CC4"

    @property
    def agent_classes(self) -> dict:
        return {
            "blue_agents": self.blue_agents,
            "red_agents": self.red_agents,
        }
