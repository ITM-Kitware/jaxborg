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
from jaxborg.actions.green import apply_green_agent_action
from jaxborg.actions.masking import compute_blue_action_mask
from jaxborg.actions.pids import append_pid_to_row
from jaxborg.actions.red_common import apply_red_session_check
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

TOTAL_ACTION_ACTOR_SLOTS = NUM_BLUE_AGENTS + GLOBAL_MAX_HOSTS + NUM_RED_AGENTS


def _frontload_order(front_slots: tuple[int, ...]) -> jnp.ndarray:
    slots = [slot for slot in range(TOTAL_ACTION_ACTOR_SLOTS) if slot not in front_slots]
    return jnp.array([*front_slots, *slots], dtype=jnp.int32)


def _cyborg_priority_execution_order(blue_actions: jnp.ndarray) -> jnp.ndarray:
    """Build execution order matching CybORG's deterministic priority sort.

    CybORG sorts actions by priority (stable sort):
      - ControlTraffic (BlockTraffic/AllowTraffic): priority 1 → execute first
      - Everything else: priority 99 → execute after

    Within the same priority tier the original agent-interface insertion order
    is preserved (blue 0-4, then green hosts, then red 0-5).
    """
    all_slots = jnp.arange(TOTAL_ACTION_ACTOR_SLOTS, dtype=jnp.int32)
    # Blue agents doing BlockTraffic or AllowTraffic have priority 1.
    is_traffic = (blue_actions >= BLUE_BLOCK_TRAFFIC_START) & (blue_actions < BLUE_ALLOW_TRAFFIC_END)
    # Build priority: 1 for traffic-control blue slots, 99 for everything else.
    priorities = jnp.full(TOTAL_ACTION_ACTOR_SLOTS, 99, dtype=jnp.int32)
    priorities = priorities.at[:NUM_BLUE_AGENTS].set(jnp.where(is_traffic, 1, 99))
    return all_slots[jnp.argsort(priorities, stable=True)]


def apply_all_actions_in_order(
    state: CC4State,
    const: CC4Const,
    blue_actions: jnp.ndarray,
    red_actions: jnp.ndarray,
    key_green: chex.PRNGKey,
    red_keys: jnp.ndarray,
    forced_primary_hosts: jnp.ndarray,
    forced_primary_pids: jnp.ndarray,
    execution_order: jnp.ndarray,
    blue_keys: jnp.ndarray = None,
) -> CC4State:
    """Apply one CybORG step using an explicit chosen-action execution order."""
    if blue_keys is None:
        blue_keys = jax.random.split(jax.random.PRNGKey(0), NUM_BLUE_AGENTS)
    green_keys = jax.random.split(key_green, GLOBAL_MAX_HOSTS)

    def step_actor(step_idx, carry_state):
        actor_slot = execution_order[step_idx]
        blue_id = jnp.clip(actor_slot, 0, NUM_BLUE_AGENTS - 1)
        green_host = jnp.clip(actor_slot - NUM_BLUE_AGENTS, 0, GLOBAL_MAX_HOSTS - 1)
        red_id = jnp.clip(actor_slot - NUM_BLUE_AGENTS - GLOBAL_MAX_HOSTS, 0, NUM_RED_AGENTS - 1)

        is_blue = actor_slot < NUM_BLUE_AGENTS
        is_green = (actor_slot >= NUM_BLUE_AGENTS) & (actor_slot < NUM_BLUE_AGENTS + GLOBAL_MAX_HOSTS)

        return jax.lax.cond(
            is_blue,
            lambda s: process_blue_with_duration(s, const, blue_id, blue_actions[blue_id], blue_keys[blue_id]),
            lambda s: jax.lax.cond(
                is_green,
                lambda gs: apply_green_agent_action(gs, const, green_host, green_keys[green_host]),
                lambda rs: process_red_with_duration(
                    rs,
                    const,
                    red_id,
                    red_actions[red_id],
                    red_keys[red_id],
                    forced_primary_host=forced_primary_hosts[red_id],
                    forced_primary_pid=forced_primary_pids[red_id],
                    run_session_check=False,
                ),
                s,
            ),
            carry_state,
        )

    state = jax.lax.fori_loop(0, TOTAL_ACTION_ACTOR_SLOTS, step_actor, state)
    state = reassign_cross_subnet_sessions(state, const)
    for b in range(NUM_BLUE_AGENTS):
        state = apply_blue_monitor(state, const, b)
    for r in range(NUM_RED_AGENTS):
        session_check_key = jax.random.fold_in(jnp.asarray(red_keys[r], dtype=jnp.uint32), jnp.int32(931))
        state = apply_red_session_check(state, const, r, session_check_key)

    from jaxborg.actions.red_common import compute_visible_sessions

    def _update_server_session_hwm(r, ss_counts):
        live = compute_visible_sessions(state, const, r)
        return ss_counts.at[r].set(jnp.maximum(ss_counts[r], live))

    server_session_count = jax.lax.fori_loop(
        0, NUM_RED_AGENTS, _update_server_session_hwm, state.red_server_session_count
    )

    # CybORG's _process_new_observations adds ALL hosts from the observation
    # to host_states.  The observation includes every host where the agent has
    # a session.  Mark these so JAX's FSM knowledge matches CybORG's.
    fsm_host_entered = state.fsm_host_entered | state.red_sessions

    return state.replace(
        fsm_host_entered=fsm_host_entered,
        red_server_session_count=server_session_count,
    )


def apply_all_actions_typed(
    state: CC4State,
    const: CC4Const,
    blue_actions: jnp.ndarray,
    red_actions: jnp.ndarray,
    key_green: chex.PRNGKey,
    red_keys: jnp.ndarray,
    forced_primary_hosts: jnp.ndarray,
    forced_primary_pids: jnp.ndarray,
    execution_order: jnp.ndarray,
    blue_keys: jnp.ndarray = None,
    *,
    use_green_vmap: bool = True,
) -> CC4State:
    """Apply one CybORG step using typed loops (no cond dispatch).

    Splits the monolithic fori_loop into 3 typed phases matching CybORG's
    execution order: traffic-control blue → other blue → green → red.
    Each loop body handles only its own action type, eliminating the 3-level
    nested jax.lax.cond that forces XLA to evaluate all 3 branches per iteration.

    Execution order correctness: CybORG sorts by priority (traffic=1, else=99)
    then by insertion order (blue 0-4, green hosts, red 0-5). After traffic-control
    blue agents, the order is always: remaining blue → green → red.
    """
    green_keys = jax.random.split(key_green, GLOBAL_MAX_HOSTS)

    # CybORG shuffles same-priority actions randomly each step.  Replicate
    # this by shuffling within each phase using a key derived from key_green.
    shuffle_key = jax.random.fold_in(key_green, 7919)  # distinct fold-in constant

    # --- Phase 1: Blue agents (traffic-control first, then others) ---
    is_traffic = (blue_actions >= BLUE_BLOCK_TRAFFIC_START) & (blue_actions < BLUE_ALLOW_TRAFFIC_END)
    blue_order = jnp.arange(NUM_BLUE_AGENTS, dtype=jnp.int32)
    blue_priority = jnp.where(is_traffic, 0, 1)
    # Shuffle within each priority group: combine priority (major) + random (minor)
    blue_shuffle_key = jax.random.fold_in(shuffle_key, 0)
    blue_rand = jax.random.uniform(blue_shuffle_key, (NUM_BLUE_AGENTS,))
    blue_sort_key = blue_priority.astype(jnp.float32) + blue_rand * 0.5
    blue_order = blue_order[jnp.argsort(blue_sort_key, stable=True)]

    if blue_keys is None:
        blue_keys = jax.random.split(jax.random.PRNGKey(0), NUM_BLUE_AGENTS)

    def blue_step(i, carry_state):
        b = blue_order[i]
        return process_blue_with_duration(carry_state, const, b, blue_actions[b], blue_keys[b])

    state = jax.lax.fori_loop(0, NUM_BLUE_AGENTS, blue_step, state)

    # --- Phase 2: Green agents ---
    # Use vmapped green in pure/training mode (faster, training-correct).
    # Fall back to sequential fori_loop for cyborg_bank mode (exact parity).
    if use_green_vmap:
        from jaxborg.actions.green_vmap import apply_green_agents_vmapped

        state = apply_green_agents_vmapped(state, const, key_green)
    else:
        from jaxborg.actions.green import _apply_single_green, _ordered_green_hosts

        green_host_order = _ordered_green_hosts(const)
        # Shuffle green agent execution order (CybORG shuffles all priority-99
        # actions randomly; green agents are the most numerous).
        # IMPORTANT: only shuffle the first num_green_agents entries (active).
        # The remaining entries are inactive padding and must stay beyond the
        # loop's iteration range to avoid skipping active agents.
        green_shuffle_key = jax.random.fold_in(shuffle_key, 1)
        rand_keys = jax.random.uniform(green_shuffle_key, (GLOBAL_MAX_HOSTS,))
        is_active_pos = jnp.arange(GLOBAL_MAX_HOSTS) < const.num_green_agents
        shuffle_sort = jnp.where(is_active_pos, rand_keys, 2.0)
        green_host_order = green_host_order[jnp.argsort(shuffle_sort)]

        def green_step(i, carry_state):
            host_idx = green_host_order[i]
            return _apply_single_green(carry_state, const, host_idx, green_keys[host_idx])

        state = jax.lax.fori_loop(0, const.num_green_agents, green_step, state)

    # --- Phase 3: Red agents (shuffled to match CybORG's random order) ---
    red_shuffle_key = jax.random.fold_in(shuffle_key, 2)
    red_order = jax.random.permutation(red_shuffle_key, NUM_RED_AGENTS)

    def red_step(i, carry_state):
        r = red_order[i]
        return process_red_with_duration(
            carry_state,
            const,
            r,
            red_actions[r],
            red_keys[r],
            forced_primary_host=forced_primary_hosts[r],
            forced_primary_pid=forced_primary_pids[r],
            run_session_check=False,
        )

    state = jax.lax.fori_loop(0, NUM_RED_AGENTS, red_step, state)

    # --- Post-step processing ---
    state = reassign_cross_subnet_sessions(state, const)

    def monitor_step(b, carry_state):
        return apply_blue_monitor(carry_state, const, b)

    state = jax.lax.fori_loop(0, NUM_BLUE_AGENTS, monitor_step, state)

    def session_check_step(r, carry_state):
        session_check_key = jax.random.fold_in(jnp.asarray(red_keys[r], dtype=jnp.uint32), jnp.int32(931))
        return apply_red_session_check(carry_state, const, r, session_check_key)

    state = jax.lax.fori_loop(0, NUM_RED_AGENTS, session_check_step, state)

    # CybORG's server_session dict accumulates session IDs monotonically —
    # entries are never removed even after Blue Restore destroys the session.
    # Update the high-water mark so the exploit 1/N roll matches CybORG.
    from jaxborg.actions.red_common import compute_visible_sessions

    def _update_server_session_hwm(r, ss_counts):
        live = compute_visible_sessions(state, const, r)
        return ss_counts.at[r].set(jnp.maximum(ss_counts[r], live))

    server_session_count = jax.lax.fori_loop(
        0, NUM_RED_AGENTS, _update_server_session_hwm, state.red_server_session_count
    )

    # CybORG's _process_new_observations adds ALL hosts from the observation
    # to host_states.  The observation includes every host where the agent has
    # a session.  Mark these so JAX's FSM knowledge matches CybORG's.
    fsm_host_entered = state.fsm_host_entered | state.red_sessions

    return state.replace(
        fsm_host_entered=fsm_host_entered,
        red_server_session_count=server_session_count,
    )


def apply_all_actions(
    state: CC4State,
    const: CC4Const,
    blue_actions: jnp.ndarray,
    red_actions: jnp.ndarray,
    key_green: chex.PRNGKey,
    red_keys: jnp.ndarray,
    forced_primary_hosts: jnp.ndarray,
    forced_primary_pids: jnp.ndarray,
    blue_keys: jnp.ndarray = None,
) -> CC4State:
    """Apply all agent actions in CybORG's deterministic priority order.

    CybORG sorts by action priority (ControlTraffic=1, else=99) then
    executes in agent-interface insertion order within each tier.

    Shared by CC4Env.step_env and the differential harness.

    Args:
        blue_actions: (NUM_BLUE_AGENTS,) int32
        red_actions: (NUM_RED_AGENTS,) int32
        red_keys: (NUM_RED_AGENTS, 2) PRNGKey per red agent
        forced_primary_hosts: (NUM_RED_AGENTS,) int32, `UNKNOWN_PRIMARY_HOST` for no override
        forced_primary_pids: (NUM_RED_AGENTS,) int32, `UNKNOWN_PRIMARY_PID` for no override
    """
    execution_order = _cyborg_priority_execution_order(blue_actions)
    # When CybORG execution order is synced (differential testing), use the
    # full action order captured from CybORG's priority-based shuffle.  This
    # ensures session creation order within a single source agent matches
    # CybORG's shuffled order.
    execution_order = jnp.where(
        const.use_green_host_order,
        const.green_host_order[state.time],
        execution_order,
    )
    return apply_all_actions_in_order(
        state,
        const,
        blue_actions,
        red_actions,
        key_green,
        red_keys,
        forced_primary_hosts,
        forced_primary_pids,
        execution_order,
        blue_keys,
    )


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
        red_primary_pid = jnp.where(
            is_active,
            red_primary_pid.at[r].set(red_next_pid),
            red_primary_pid,
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
        fsm_host_entered = jnp.where(
            is_active,
            fsm_host_entered.at[r, start_host].set(True),
            fsm_host_entered,
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
        # Record scan-owning PID for initial knowledge sourced from start_host.
        has_initial_scan = jnp.any(initially_scanned)
        red_scan_source_pid = jnp.where(
            is_active & has_initial_scan,
            red_scan_source_pid.at[r, start_host].set(red_primary_pid[r]),
            red_scan_source_pid,
        )

    return state.replace(
        red_sessions=red_sessions,
        red_session_count=red_session_count,
        red_abstract_session_count=red_session_count,  # at reset, all sessions are abstract (primary)
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


class CC4Env(MultiAgentEnv):
    def __init__(
        self,
        num_steps: int = 500,
        *,
        topology_mode: str = "pure",
        topology_bank_size: int = 0,
        sync_red_policy_bank: bool = False,
        training_mode: bool = False,
    ):
        self.num_steps = num_steps
        self.topology_mode = topology_mode
        self.topology_bank_size = topology_bank_size
        self.sync_red_policy_bank = sync_red_policy_bank
        self.training_mode = training_mode
        self._const_bank = None
        self._green_random_bank = None
        self._red_policy_random_bank = None
        if topology_mode == "cyborg_bank":
            self._const_bank = get_cyborg_topology_bank(num_steps, topology_bank_size)
            # Green random bank encodes CybORG's specific green agent decisions
            # for a reference trajectory.  These tokens become stale (selecting
            # inactive services) when blue/red actions diverge from the recording,
            # producing spurious LWF failures.  Only load for differential sync.
            if sync_red_policy_bank:
                self._green_random_bank = get_cyborg_green_random_bank(num_steps, topology_bank_size)
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

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], CC4EnvState]:
        const = self._select_const(key)
        green_randoms = self._select_green_randoms(key)
        red_policy_randoms = self._select_red_policy_randoms(key)
        if green_randoms is not None:
            const = const.replace(green_randoms=green_randoms, use_green_randoms=jnp.array(True))
        if red_policy_randoms is not None:
            const = const.replace(red_policy_randoms=red_policy_randoms, use_red_policy_randoms=jnp.array(True))
        state = create_initial_state()
        state = state.replace(
            host_services=jnp.array(const.initial_services),
            host_max_pid=const.host_initial_max_pid,
        )
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
        state = state.replace(
            host_services=const.initial_services,
            host_max_pid=const.host_initial_max_pid,
        )
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

        key, key_green, key_red, key_blue = jax.random.split(key, 4)
        red_keys = jax.random.split(key_red, NUM_RED_AGENTS)
        blue_keys = jax.random.split(key_blue, NUM_BLUE_AGENTS)

        state = advance_mission_phase(state, const)

        state = state.replace(
            red_scan_success=jnp.zeros(NUM_RED_AGENTS, dtype=jnp.bool_),
            red_exploit_success=jnp.zeros(NUM_RED_AGENTS, dtype=jnp.bool_),
            red_discover_success=jnp.zeros(NUM_RED_AGENTS, dtype=jnp.bool_),
            red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32),
            green_lwf_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_),
            green_asf_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_),
            red_impact_attempted=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_),
        )

        blue_action_arr = jnp.array([actions[f"blue_{b}"] for b in range(NUM_BLUE_AGENTS)], dtype=jnp.int32)
        red_action_arr = jnp.array([actions[f"red_{r}"] for r in range(NUM_RED_AGENTS)], dtype=jnp.int32)
        no_forced = jnp.full(NUM_RED_AGENTS, UNKNOWN_PRIMARY_HOST, dtype=jnp.int32)
        no_forced_pids = jnp.full(NUM_RED_AGENTS, UNKNOWN_PRIMARY_PID, dtype=jnp.int32)

        execution_order = _cyborg_priority_execution_order(blue_action_arr)
        state = apply_all_actions_typed(
            state,
            const,
            blue_action_arr,
            red_action_arr,
            key_green,
            red_keys,
            no_forced,
            no_forced_pids,
            execution_order,
            blue_keys,
            use_green_vmap=(self.topology_mode == "pure"),
        )

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
