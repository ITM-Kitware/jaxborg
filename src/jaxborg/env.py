from functools import partial
from typing import Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State
from jaxmarl.environments.spaces import Box, Discrete

from jaxborg.actions.blue_monitor import apply_blue_monitor
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
from jaxborg.rewards import advance_mission_phase, compute_reward_breakdown
from jaxborg.state import CC4Const, CC4State, create_initial_state
from jaxborg.topology import (
    build_topology,
    cyborg_bank_index_from_key,
    get_cyborg_green_random_bank,
    get_cyborg_red_policy_random_bank,
    get_cyborg_topology_bank,
)


def apply_all_actions(
    state: CC4State,
    const: CC4Const,
    blue_actions: jnp.ndarray,
    red_actions: jnp.ndarray,
    key_green: chex.PRNGKey,
    red_keys: jnp.ndarray,
    forced_primary_hosts: jnp.ndarray,
) -> CC4State:
    """Apply all agent actions in CybORG-correct order: blue → green → red → reassign → monitors.

    Shared by CC4Env.step_env and the differential harness.

    Args:
        blue_actions: (NUM_BLUE_AGENTS,) int32
        red_actions: (NUM_RED_AGENTS,) int32
        red_keys: (NUM_RED_AGENTS, 2) PRNGKey per red agent
        forced_primary_hosts: (NUM_RED_AGENTS,) int32, -1 for no override
    """
    for b in range(NUM_BLUE_AGENTS):
        state = process_blue_with_duration(state, const, b, blue_actions[b])

    state = apply_green_agents(state, const, key_green)

    for r in range(NUM_RED_AGENTS):
        state = process_red_with_duration(
            state, const, r, red_actions[r], red_keys[r], forced_primary_host=forced_primary_hosts[r]
        )

    state = reassign_cross_subnet_sessions(state, const)
    for b in range(NUM_BLUE_AGENTS):
        state = apply_blue_monitor(state, const, b)

    return state


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
    red_scanned_source_hosts = state.red_scanned_source_hosts
    red_session_pids = state.red_session_pids
    red_session_abstract_pids = state.red_session_abstract_pids
    red_next_pid = state.red_next_pid

    # Only red_agent_0 is active at reset; others activate via session reassignment
    red_agent_active = state.red_agent_active.at[0].set(True)

    for r in range(NUM_RED_AGENTS):
        start_host = const.red_start_hosts[r]
        is_active = red_agent_active[r]
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
        abstract_pid_row = red_session_abstract_pids[r, start_host]
        red_session_abstract_pids = jnp.where(
            is_active,
            red_session_abstract_pids.at[r, start_host].set(append_pid_to_row(abstract_pid_row, red_next_pid)),
            red_session_abstract_pids,
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
        red_scanned_source_hosts=red_scanned_source_hosts,
        red_scan_anchor_host=red_scan_anchor_host,
        host_compromised=host_compromised,
        fsm_host_states=fsm_states,
        red_session_is_abstract=red_session_is_abstract,
        red_abstract_host_rank=red_abstract_host_rank,
        red_next_abstract_rank=red_next_abstract_rank,
        red_session_pids=red_session_pids,
        red_session_abstract_pids=red_session_abstract_pids,
        red_next_pid=red_next_pid,
        red_agent_active=red_agent_active,
    )


class CC4Env(MultiAgentEnv):
    def __init__(
        self,
        num_steps: int = 500,
        *,
        topology_mode: str = "pure",
        topology_bank_size: int = 0,
        sync_red_policy_bank: bool = False,
    ):
        self.num_steps = num_steps
        self.topology_mode = topology_mode
        self.topology_bank_size = topology_bank_size
        self.sync_red_policy_bank = sync_red_policy_bank
        self._const_bank = None
        self._green_random_bank = None
        self._red_policy_random_bank = None
        if topology_mode == "cyborg_bank":
            self._const_bank = get_cyborg_topology_bank(num_steps, topology_bank_size)
            self._green_random_bank = get_cyborg_green_random_bank(num_steps, topology_bank_size)
            if sync_red_policy_bank:
                self._red_policy_random_bank = get_cyborg_red_policy_random_bank(num_steps, topology_bank_size)
        elif topology_mode != "pure":
            raise ValueError(f"Unknown topology_mode={topology_mode!r}")

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

    def _select_const(self, key: chex.PRNGKey) -> CC4Const:
        if self._const_bank is None:
            return build_topology(key, num_steps=self.num_steps)

        bank_idx = cyborg_bank_index_from_key(key, self.topology_bank_size)
        return jax.tree.map(lambda x: x[bank_idx], self._const_bank)

    def _select_green_randoms(self, key: chex.PRNGKey) -> chex.Array | None:
        if self._green_random_bank is None:
            return None

        bank_idx = cyborg_bank_index_from_key(key, self.topology_bank_size)
        return self._green_random_bank[bank_idx]

    def _select_red_policy_randoms(self, key: chex.PRNGKey) -> chex.Array | None:
        if self._red_policy_random_bank is None:
            return None

        bank_idx = cyborg_bank_index_from_key(key, self.topology_bank_size)
        return self._red_policy_random_bank[bank_idx]

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], CC4EnvState]:
        const = self._select_const(key)
        green_randoms = self._select_green_randoms(key)
        red_policy_randoms = self._select_red_policy_randoms(key)
        if green_randoms is not None:
            const = const.replace(green_randoms=green_randoms, use_green_randoms=jnp.array(True))
        if red_policy_randoms is not None:
            const = const.replace(red_policy_randoms=red_policy_randoms, use_red_policy_randoms=jnp.array(True))
        state = create_initial_state()
        state = state.replace(host_services=jnp.array(const.initial_services))
        state = _init_red_state(const, state)

        env_state = CC4EnvState(state=state, const=const)
        obs = self.get_obs(env_state)
        return obs, env_state

    @partial(jax.jit, static_argnums=[0])
    def _reset_state(self, env_state: CC4EnvState, key: chex.PRNGKey) -> CC4EnvState:
        """Reset with a new random topology (for auto-reset)."""
        const = self._select_const(key)
        green_randoms = self._select_green_randoms(key)
        red_policy_randoms = self._select_red_policy_randoms(key)
        if green_randoms is not None:
            const = const.replace(green_randoms=green_randoms, use_green_randoms=jnp.array(True))
        if red_policy_randoms is not None:
            const = const.replace(red_policy_randoms=red_policy_randoms, use_red_policy_randoms=jnp.array(True))
        state = create_initial_state()
        state = state.replace(host_services=const.initial_services)
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
            states_re = self._reset_state(states_st, key_reset)
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

        key, key_green, key_red = jax.random.split(key, 3)
        red_keys = jax.random.split(key_red, NUM_RED_AGENTS)

        state = advance_mission_phase(state, const)

        state = state.replace(
            red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32),
            green_lwf_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_),
            green_asf_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_),
            red_impact_attempted=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_),
        )

        blue_action_arr = jnp.array([actions[f"blue_{b}"] for b in range(NUM_BLUE_AGENTS)], dtype=jnp.int32)
        red_action_arr = jnp.array([actions[f"red_{r}"] for r in range(NUM_RED_AGENTS)], dtype=jnp.int32)
        no_forced = jnp.full(NUM_RED_AGENTS, -1, dtype=jnp.int32)

        state = apply_all_actions(state, const, blue_action_arr, red_action_arr, key_green, red_keys, no_forced)

        reward_breakdown = compute_reward_breakdown(
            state,
            const,
            state.red_impact_attempted,
            state.green_lwf_this_step,
            state.green_asf_this_step,
        )
        reward = reward_breakdown.total

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

        info = {
            "reward_ria": reward_breakdown.ria_reward,
            "reward_lwf": reward_breakdown.lwf_reward,
            "reward_asf": reward_breakdown.asf_reward,
            "impact_count": reward_breakdown.ria_count,
            "green_lwf_count": reward_breakdown.lwf_count,
            "green_asf_count": reward_breakdown.asf_count,
        }

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
            masks[f"blue_{i}"] = compute_blue_action_mask(env_state.const, i, env_state.state)
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
