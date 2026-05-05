from functools import partial
from typing import Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State
from jaxmarl.environments.spaces import Box, Discrete

from jaxborg.actions.blue_monitor import apply_blue_monitor
from jaxborg.actions.duration import (
    UNKNOWN_PRIMARY_HOST,
    UNKNOWN_PRIMARY_PID,
    process_blue_with_duration,
    process_red_with_duration,
)
from jaxborg.actions.encoding import (
    BLUE_ALLOW_TRAFFIC_END,
    BLUE_BLOCK_TRAFFIC_START,
    RED_WITHDRAW_END,
)
from jaxborg.actions.masking import compute_blue_action_mask
from jaxborg.actions.pids import append_pid_to_row
from jaxborg.actions.red_common import apply_red_session_check
from jaxborg.constants import CC4_CONFIG, COMPROMISE_USER
from jaxborg.observations import get_blue_obs, get_red_obs
from jaxborg.reassignment import reassign_cross_subnet_sessions
from jaxborg.rewards import advance_mission_phase, compute_reward_breakdown
from jaxborg.scenarios.cc4.red_fsm import fsm_red_init_states
from jaxborg.scenarios.cc4.topology import (
    build_topology,
    cyborg_bank_index_from_key,
    get_cyborg_green_random_bank,
    get_cyborg_red_policy_random_bank,
    get_cyborg_topology_bank,
)
from jaxborg.scenarios.config import ScenarioConfig
from jaxborg.state import SimulatorConst, SimulatorState, create_initial_state


def apply_all_actions(
    state: SimulatorState,
    const: SimulatorConst,
    blue_actions: jnp.ndarray,
    red_actions: jnp.ndarray,
    key_green: chex.PRNGKey,
    red_keys: jnp.ndarray,
    forced_primary_hosts: jnp.ndarray,
    forced_primary_pids: jnp.ndarray,
    blue_keys: jnp.ndarray = None,
) -> SimulatorState:
    """Apply one CybORG step in CybORG's deterministic priority order.

    Split into 3 typed phases matching CybORG's execution order:
    traffic-control blue → other blue → green → red. Each loop body handles
    only its own action type, eliminating the 3-level nested jax.lax.cond
    that would force XLA to evaluate all 3 branches per iteration.

    Order correctness: CybORG sorts by priority (ControlTraffic=1, else=99)
    via a stable sort, then executes in agent-interface dict-insertion order
    within each tier (blue 0..4 → green hosts → red 0..5). The bandwidth
    shuffle in CybORG's sort_action_order is a no-op for CC4 (every action
    has bandwidth_usage=0 → no drops, returned list is the un-shuffled
    priority sort). The JAX derivation reproduces this exactly; verified
    against real CybORG traces by the differential test suite.
    """
    n_blue = blue_actions.shape[0]
    n_red = red_actions.shape[0]

    # CybORG sorts actions by priority (ControlTraffic=1, else=99) then
    # executes in deterministic agent-interface insertion order within each
    # tier.  The shuffle in sort_action_order() is only for bandwidth
    # checking (all bandwidth_usage=0 in CC4, so effectively a no-op).
    # Match this by using a fixed order within each phase.

    # --- Phase 1: Blue agents (traffic-control first, then others) ---
    is_traffic = (blue_actions >= BLUE_BLOCK_TRAFFIC_START) & (blue_actions < BLUE_ALLOW_TRAFFIC_END)
    blue_order = jnp.arange(n_blue, dtype=jnp.int32)
    blue_priority = jnp.where(is_traffic, 0, 1)
    # Deterministic sort by priority; stable sort preserves agent index order
    # within the same tier, matching CybORG's dict insertion order.
    blue_order = blue_order[jnp.argsort(blue_priority, stable=True)]

    if blue_keys is None:
        blue_keys = jax.random.split(jax.random.PRNGKey(0), n_blue)

    # n_blue is small (5) and static; unroll for stronger XLA fusion / no while carrier.
    for i in range(n_blue):
        b = blue_order[i]
        state = process_blue_with_duration(state, const, b, blue_actions[b], blue_keys[b])

    # --- Phase 2: Green agents (vmap+scatter) ---
    # CybORG's FSM calls get_action() for ALL agents BEFORE any execute().
    # server_session (used for exploit 1/N roll) therefore reflects the
    # previous step's observation — it does NOT include phishing sessions
    # created in the current step.  Snapshot the pre-green count and pass
    # it to the red phase so exploit creation-time N matches CybORG's
    # get_action() timing.
    pre_green_visible_sessions = state.red_server_session_count

    from jaxborg.actions.green_vmap import apply_green_agents_vmapped

    state = apply_green_agents_vmapped(state, const, key_green)

    # --- Phase 3: Red agents (deterministic order matching CybORG) ---
    # red_order = arange so r == i; n_red is small (6) and static — unroll directly.
    for r in range(n_red):
        state = process_red_with_duration(
            state,
            const,
            r,
            red_actions[r],
            red_keys[r],
            forced_primary_host=forced_primary_hosts[r],
            forced_primary_pid=forced_primary_pids[r],
            run_session_check=False,
            creation_visible_sessions_override=pre_green_visible_sessions[r],
        )

    # --- Post-step processing ---

    # CybORG's server_session dict accumulates session IDs monotonically —
    # entries are never removed even after Blue Restore or cross-subnet
    # reassignment.  CybORG processes the exploit action's observation
    # red_server_session_count is now maintained as a cumulative counter:
    # incremented in green_vmap (phishing) and reassignment (session transfer).
    # No HWM update needed here.

    state = reassign_cross_subnet_sessions(state, const)

    for b in range(n_blue):
        state = apply_blue_monitor(state, const, b)

    for r in range(n_red):
        session_check_key = jax.random.fold_in(jnp.asarray(red_keys[r], dtype=jnp.uint32), jnp.int32(931))
        state = apply_red_session_check(state, const, r, session_check_key)

    # CybORG's _process_new_observations adds ALL hosts from the observation
    # to host_states.  The observation includes every host where the agent has
    # a session.  Mark these so JAX's FSM knowledge matches CybORG's.
    fsm_host_entered = state.fsm_host_entered | state.red_sessions

    return state.replace(
        fsm_host_entered=fsm_host_entered,
    )


@struct.dataclass
class ScenarioEnvState:
    state: SimulatorState
    const: SimulatorConst


def _init_red_state(const: SimulatorConst, state: SimulatorState) -> SimulatorState:
    red_sessions = state.red_sessions
    red_session_count = state.red_session_count
    red_privilege = state.red_privilege
    red_discovered = state.red_discovered_hosts | const.red_initial_discovered_hosts
    red_scanned = state.red_scanned_hosts | const.red_initial_scanned_hosts
    fsm_states = state.fsm_host_states
    host_compromised = state.host_compromised
    red_scan_anchor_host = state.red_scan_anchor_host
    red_session_is_abstract = state.red_session_is_abstract
    red_primary_pid = state.red_primary_pid
    red_abstract_host_rank = state.red_abstract_host_rank
    red_next_abstract_rank = state.red_next_abstract_rank
    red_scanned_source_hosts = state.red_scanned_source_hosts
    red_scan_source_pid = state.red_scan_source_pid
    red_session_pids = state.red_session_pids
    red_session_abstract_pids = state.red_session_abstract_pids
    red_next_pid = state.red_next_pid
    fsm_host_entered = state.fsm_host_entered

    # Only red_agent_0 is active at reset; others activate via session reassignment
    red_agent_active = state.red_agent_active.at[0].set(True)

    n_red = state.red_agent_active.shape[0]
    for r in range(n_red):
        start_host = const.red_start_hosts[r]
        is_active = red_agent_active[r]
        # Push is_active into scalar/row/column-scoped scatters: when False, the
        # scatter writes the prior value back (no-op).  Avoids materializing full
        # (NUM_RED, GLOBAL_MAX_HOSTS, ...) arrays under both branches of jnp.where
        # for each of the ~15 fields below.  Behavior is identical.
        red_sessions = red_sessions.at[r, start_host].set(jnp.where(is_active, True, red_sessions[r, start_host]))
        red_session_count = red_session_count.at[r, start_host].set(
            jnp.where(is_active, jnp.int32(1), red_session_count[r, start_host])
        )
        red_session_is_abstract = red_session_is_abstract.at[r, start_host].set(
            jnp.where(is_active, True, red_session_is_abstract[r, start_host])
        )
        red_abstract_host_rank = red_abstract_host_rank.at[r, start_host].set(
            jnp.where(is_active, jnp.int32(0), red_abstract_host_rank[r, start_host])
        )
        red_next_abstract_rank = red_next_abstract_rank.at[r].set(
            jnp.where(is_active, jnp.int32(1), red_next_abstract_rank[r])
        )
        pid_row = red_session_pids[r, start_host]
        red_session_pids = red_session_pids.at[r, start_host].set(
            jnp.where(is_active, append_pid_to_row(pid_row, red_next_pid), pid_row)
        )
        abstract_pid_row = red_session_abstract_pids[r, start_host]
        red_session_abstract_pids = red_session_abstract_pids.at[r, start_host].set(
            jnp.where(is_active, append_pid_to_row(abstract_pid_row, red_next_pid), abstract_pid_row)
        )
        red_primary_pid = red_primary_pid.at[r].set(jnp.where(is_active, red_next_pid, red_primary_pid[r]))
        red_next_pid = jnp.where(is_active, red_next_pid + 1, red_next_pid)
        red_privilege = red_privilege.at[r, start_host].set(
            jnp.where(is_active, jnp.int32(COMPROMISE_USER), red_privilege[r, start_host])
        )
        # CybORG pre-seeds aspace.ip_address with the start host at reset
        # for ALL agents (including initially inactive ones).  Always mark
        # start host as discovered so ScenarioEnv action replay has the correct
        # action space.  FsmRedCC4Env._strip_inactive_red_reset_knowledge
        # will clear this for inactive agents to match the FSM's host_states.
        red_discovered = red_discovered.at[r, start_host].set(True)
        host_compromised = host_compromised.at[start_host].set(
            jnp.where(
                is_active,
                jnp.maximum(host_compromised[start_host], COMPROMISE_USER),
                host_compromised[start_host],
            )
        )
        fsm_states = fsm_states.at[r].set(jnp.where(is_active, fsm_red_init_states(const, r), fsm_states[r]))
        fsm_host_entered = fsm_host_entered.at[r, start_host].set(
            jnp.where(is_active, True, fsm_host_entered[r, start_host])
        )
        red_scan_anchor_host = red_scan_anchor_host.at[r].set(jnp.where(is_active, start_host, red_scan_anchor_host[r]))
        initially_scanned = const.red_initial_scanned_hosts[r]
        prior_scan_col = red_scanned_source_hosts[r, :, start_host]
        red_scanned_source_hosts = red_scanned_source_hosts.at[r, :, start_host].set(
            jnp.where(is_active, initially_scanned, prior_scan_col)
        )
        # Record scan-owning PID for initial knowledge sourced from start_host.
        has_initial_scan = jnp.any(initially_scanned)
        red_scan_source_pid = red_scan_source_pid.at[r, start_host].set(
            jnp.where(
                is_active & has_initial_scan,
                red_primary_pid[r],
                red_scan_source_pid[r, start_host],
            )
        )

    # CybORG's server_session dict gets one entry per active agent at reset
    # (the initial RedAbstractSession).  Set the cumulative counter to 1 for
    # initially active agents, 0 for inactive.
    red_server_session_count = red_agent_active.astype(jnp.int32)

    return state.replace(
        red_sessions=red_sessions,
        red_session_count=red_session_count,
        red_abstract_session_count=red_session_count,  # at reset, all sessions are abstract (primary)
        red_server_session_count=red_server_session_count,
        red_privilege=red_privilege,
        red_discovered_hosts=red_discovered,
        red_scanned_hosts=red_scanned,
        red_scanned_source_hosts=red_scanned_source_hosts,
        red_scan_source_pid=red_scan_source_pid,
        red_scan_anchor_host=red_scan_anchor_host,
        red_primary_pid=red_primary_pid,
        host_compromised=host_compromised,
        fsm_host_states=fsm_states,
        fsm_host_entered=fsm_host_entered,
        red_session_is_abstract=red_session_is_abstract,
        red_abstract_host_rank=red_abstract_host_rank,
        red_next_abstract_rank=red_next_abstract_rank,
        red_session_pids=red_session_pids,
        red_session_abstract_pids=red_session_abstract_pids,
        red_next_pid=red_next_pid,
        red_agent_active=red_agent_active,
    )


class ScenarioEnv(MultiAgentEnv):
    def __init__(
        self,
        num_steps: Optional[int] = None,
        *,
        topology_mode: str = "generative",
        topology_bank_size: int = 0,
        sync_red_policy_bank: bool = False,
        training_mode: bool = False,
        scenario_config: ScenarioConfig = CC4_CONFIG,
    ):
        self.cfg = scenario_config
        self.num_steps = num_steps if num_steps is not None else scenario_config.max_steps
        self.topology_mode = topology_mode
        self.topology_bank_size = topology_bank_size
        self.sync_red_policy_bank = sync_red_policy_bank
        self.training_mode = training_mode
        self._const_bank = None
        self._green_random_bank = None
        self._red_policy_random_bank = None
        if topology_mode == "cyborg_bank":
            self._const_bank = get_cyborg_topology_bank(self.num_steps, topology_bank_size)
            # Green random bank encodes CybORG's specific green agent decisions
            # for a reference trajectory.  Always load in non-training mode so
            # green phishing/LWF decisions match CybORG.  In training mode tokens
            # become stale when blue actions change services, so only load when
            # explicitly syncing the full policy bank.
            if not training_mode or sync_red_policy_bank:
                self._green_random_bank = get_cyborg_green_random_bank(self.num_steps, topology_bank_size)
            if sync_red_policy_bank:
                self._red_policy_random_bank = get_cyborg_red_policy_random_bank(self.num_steps, topology_bank_size)
        elif topology_mode != "generative":
            raise ValueError(f"Unknown topology_mode={topology_mode!r}")

        self.blue_agents = [f"blue_{i}" for i in range(self.cfg.num_blue_agents)]
        self.red_agents = [f"red_{i}" for i in range(self.cfg.num_red_agents)]
        self.agents = self.blue_agents + self.red_agents

        super().__init__(num_agents=self.cfg.num_blue_agents + self.cfg.num_red_agents)

        for agent in self.blue_agents:
            self.action_spaces[agent] = Discrete(BLUE_ALLOW_TRAFFIC_END)
            self.observation_spaces[agent] = Box(low=0.0, high=1.0, shape=(self.cfg.blue_obs_size,), dtype=jnp.float32)
        for agent in self.red_agents:
            self.action_spaces[agent] = Discrete(RED_WITHDRAW_END)
            self.observation_spaces[agent] = Box(low=0.0, high=1.0, shape=(self.cfg.blue_obs_size,), dtype=jnp.float32)

    def _select_const(self, key: chex.PRNGKey) -> SimulatorConst:
        if self._const_bank is None:
            return build_topology(key, num_steps=self.num_steps, training_mode=self.training_mode)

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

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], ScenarioEnvState]:
        const = self._select_const(key)
        green_randoms = self._select_green_randoms(key)
        red_policy_randoms = self._select_red_policy_randoms(key)
        if green_randoms is not None:
            const = const.replace(green_randoms=green_randoms, use_green_randoms=jnp.array(True))
        if red_policy_randoms is not None:
            const = const.replace(red_policy_randoms=red_policy_randoms, use_red_policy_randoms=jnp.array(True))
        state = create_initial_state(self.cfg)
        state = state.replace(
            host_services=jnp.array(const.initial_services),
            host_max_pid=const.host_initial_max_pid,
        )
        state = _init_red_state(const, state)

        env_state = ScenarioEnvState(state=state, const=const)
        obs = self.get_obs(env_state)
        return obs, env_state

    @partial(jax.jit, static_argnums=[0])
    def _reset_state(self, env_state: ScenarioEnvState, key: chex.PRNGKey) -> ScenarioEnvState:
        """Reset with a new random topology (for auto-reset)."""
        const = self._select_const(key)
        green_randoms = self._select_green_randoms(key)
        red_policy_randoms = self._select_red_policy_randoms(key)
        if green_randoms is not None:
            const = const.replace(green_randoms=green_randoms, use_green_randoms=jnp.array(True))
        if red_policy_randoms is not None:
            const = const.replace(red_policy_randoms=red_policy_randoms, use_red_policy_randoms=jnp.array(True))
        state = create_initial_state(self.cfg)
        state = state.replace(
            host_services=const.initial_services,
            host_max_pid=const.host_initial_max_pid,
        )
        state = _init_red_state(const, state)
        return ScenarioEnvState(state=state, const=const)

    @partial(jax.jit, static_argnums=[0])
    def step(
        self,
        key: chex.PRNGKey,
        state: ScenarioEnvState,
        actions: Dict[str, chex.Array],
        reset_state: Optional[State] = None,
    ) -> Tuple[Dict[str, chex.Array], ScenarioEnvState, Dict[str, float], Dict[str, bool], Dict]:
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
        env_state: ScenarioEnvState,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], ScenarioEnvState, Dict[str, float], Dict[str, bool], Dict]:
        state = env_state.state
        const = env_state.const
        n_blue = self.cfg.num_blue_agents
        n_red = self.cfg.num_red_agents
        n_hosts = self.cfg.num_hosts

        key, key_green, key_red, key_blue = jax.random.split(key, 4)
        red_keys = jax.random.split(key_red, n_red)
        blue_keys = jax.random.split(key_blue, n_blue)

        state = advance_mission_phase(state, const)

        state = state.replace(
            red_scan_success=jnp.zeros(n_red, dtype=jnp.bool_),
            red_exploit_success=jnp.zeros(n_red, dtype=jnp.bool_),
            red_discover_success=jnp.zeros(n_red, dtype=jnp.bool_),
            red_activity_this_step=jnp.zeros(n_hosts, dtype=jnp.int32),
            green_lwf_this_step=jnp.zeros(n_hosts, dtype=jnp.bool_),
            green_asf_this_step=jnp.zeros(n_hosts, dtype=jnp.bool_),
            red_impact_attempted=jnp.zeros(n_hosts, dtype=jnp.bool_),
        )

        blue_action_arr = jnp.array([actions[f"blue_{b}"] for b in range(n_blue)], dtype=jnp.int32)
        red_action_arr = jnp.array([actions[f"red_{r}"] for r in range(n_red)], dtype=jnp.int32)
        no_forced = jnp.full(n_red, UNKNOWN_PRIMARY_HOST, dtype=jnp.int32)
        no_forced_pids = jnp.full(n_red, UNKNOWN_PRIMARY_PID, dtype=jnp.int32)

        state = apply_all_actions(
            state,
            const,
            blue_action_arr,
            red_action_arr,
            key_green,
            red_keys,
            no_forced,
            no_forced_pids,
            blue_keys,
        )

        reward_breakdown = compute_reward_breakdown(
            state,
            const,
            state.red_impact_attempted,
            state.green_lwf_this_step,
            state.green_asf_this_step,
            blue_actions=blue_action_arr,
        )
        reward = reward_breakdown.total

        state = state.replace(time=state.time + 1)
        done = state.time >= const.max_steps
        state = state.replace(done=jnp.array(done))

        env_state = ScenarioEnvState(state=state, const=const)
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
            "action_cost": reward_breakdown.action_cost,
            "impact_count": reward_breakdown.ria_count,
            "green_lwf_count": reward_breakdown.lwf_count,
            "green_asf_count": reward_breakdown.asf_count,
        }

        return obs, env_state, rewards, dones, info

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, env_state: ScenarioEnvState) -> Dict[str, chex.Array]:
        state = env_state.state
        const = env_state.const
        obs = {}
        for b in range(self.cfg.num_blue_agents):
            obs[f"blue_{b}"] = get_blue_obs(state, const, b)
        for r in range(self.cfg.num_red_agents):
            obs[f"red_{r}"] = get_red_obs(state, const, r)
        return obs

    @partial(jax.jit, static_argnums=[0])
    def get_avail_actions(self, env_state: ScenarioEnvState) -> Dict[str, chex.Array]:
        masks = {}
        for i in range(self.cfg.num_blue_agents):
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
