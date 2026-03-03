from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.actions import apply_blue_action, apply_red_action
from jaxborg.actions.blue_monitor import apply_blue_monitor
from jaxborg.actions.duration import (
    process_blue_with_duration,
    process_red_with_duration,
)
from jaxborg.actions.encoding import (
    ACTION_TYPE_AGGRESSIVE_SCAN,
    ACTION_TYPE_SCAN,
    ACTION_TYPE_STEALTH_SCAN,
    BLUE_DECOY_END,
    BLUE_DECOY_START,
    BLUE_SLEEP,
    RED_SLEEP,
    decode_red_action,
)
from jaxborg.actions.green import apply_green_agents
from jaxborg.actions.pending_source import (
    PENDING_SOURCE_KIND_NONE,
    PENDING_SOURCE_KIND_SESSION_BINDING,
)
from jaxborg.actions.pids import append_pid_to_row
from jaxborg.actions.red_common import select_bound_source_host
from jaxborg.agents.fsm_red import (
    fsm_red_get_action_and_info,
    fsm_red_init_states,
    fsm_red_post_step_update,
)
from jaxborg.constants import (
    ABSTRACT_RANK_NONE,
    GLOBAL_MAX_HOSTS,
    MAX_TRACKED_SESSION_PIDS,
    MAX_TRACKED_SUSPICIOUS_PIDS,
    NUM_BLUE_AGENTS,
    NUM_RED_AGENTS,
)
from jaxborg.reassignment import reassign_cross_subnet_sessions
from jaxborg.rewards import advance_mission_phase
from jaxborg.state import create_initial_state
from jaxborg.topology import build_const_from_cyborg
from jaxborg.translate import (
    build_mappings_from_cyborg,
    cyborg_blue_to_jax,
    jax_blue_to_cyborg,
    jax_red_to_cyborg,
)


@dataclass
class StateSnapshot:
    time: int = 0
    mission_phase: int = 0
    host_compromised: dict = field(default_factory=dict)
    red_privilege: dict = field(default_factory=dict)
    red_sessions: dict = field(default_factory=dict)
    host_services: dict = field(default_factory=dict)
    host_decoys: dict = field(default_factory=dict)
    ot_service_stopped: dict = field(default_factory=dict)
    blocked_zones: set = field(default_factory=set)
    rewards: dict = field(default_factory=dict)


@dataclass
class StateDiff:
    field_name: str
    cyborg_value: object
    jax_value: object
    host_or_agent: str = ""


@dataclass
class StepResult:
    step: int
    diffs: list[StateDiff] = field(default_factory=list)
    cyborg_rewards: dict = field(default_factory=dict)
    jax_rewards: dict = field(default_factory=dict)


@dataclass
class TestResult:
    steps_run: int = 0
    step_results: list[StepResult] = field(default_factory=list)
    error_diffs: int = 0


_ZERO_INT_HOSTS = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32)
_ZERO_BOOL_HOSTS = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_)


def _cy_action_succeeded(controller, agent_name: str) -> bool | None:
    obs_set = controller.observation.get(agent_name)
    if obs_set is None:
        return None
    try:
        obs = obs_set.get_combined_observation()
    except Exception:
        return None
    success = getattr(obs, "success", None)
    if success is None:
        return None
    return str(success).upper() == "TRUE"


@jax.jit
def _jit_fsm_red_get_action_and_info(state, const, agent_id, key):
    return fsm_red_get_action_and_info(state, const, jnp.int32(agent_id), key)


@jax.jit
def _jit_apply_red_action(state, const, agent_id, action_idx, key):
    return apply_red_action(state, const, jnp.int32(agent_id), jnp.int32(action_idx), key)


@jax.jit
def _jit_apply_blue_action(state, const, agent_id, action_idx):
    return apply_blue_action(state, const, jnp.int32(agent_id), jnp.int32(action_idx))


@jax.jit
def _jit_advance_and_clear(state, const):
    state = advance_mission_phase(state, const)
    return state.replace(
        red_activity_this_step=_ZERO_INT_HOSTS,
        green_lwf_this_step=_ZERO_BOOL_HOSTS,
        green_asf_this_step=_ZERO_BOOL_HOSTS,
    )


@jax.jit
def _jit_fsm_red_post_step_update(
    state_before, state_after, const, target_hosts, fsm_actions, eligible_flags, executed_flags=None
):
    return fsm_red_post_step_update(
        state_before,
        state_after,
        const,
        target_hosts,
        fsm_actions,
        eligible_flags,
        executed_flags,
    )


@jax.jit
def _jit_apply_green(state, const, key):
    return apply_green_agents(state, const, key)


@jax.jit
def _jit_apply_end_turn_monitors(state, const):
    for b in range(NUM_BLUE_AGENTS):
        state = apply_blue_monitor(state, const, b)
    return state


@jax.jit
def _jit_reassign(state, const):
    return reassign_cross_subnet_sessions(state, const)


@jax.jit
def _jit_process_red_with_duration(state, const, agent_id, action_idx, key, forced_primary_host):
    return process_red_with_duration(
        state,
        const,
        jnp.int32(agent_id),
        jnp.int32(action_idx),
        key,
        forced_primary_host=jnp.int32(forced_primary_host),
    )


@jax.jit
def _jit_process_blue_with_duration(state, const, agent_id, action_idx):
    return process_blue_with_duration(state, const, jnp.int32(agent_id), jnp.int32(action_idx))


class CC4DifferentialHarness:
    def __init__(
        self,
        seed=42,
        max_steps=500,
        blue_cls=SleepAgent,
        green_cls=EnterpriseGreenAgent,
        red_cls=FiniteStateRedAgent,
        check_rewards=True,
        check_obs=False,
        sync_green_rng=False,
        use_cyborg_blue_policy=False,
    ):
        self.seed = seed
        self.max_steps = max_steps
        self.blue_cls = blue_cls
        self.green_cls = green_cls
        self.red_cls = red_cls
        self.check_rewards = check_rewards
        self.check_obs = check_obs
        self.sync_green_rng = sync_green_rng
        self.use_cyborg_blue_policy = use_cyborg_blue_policy
        self.cyborg_env = None
        self.jax_state = None
        self.jax_const = None
        self.mappings = None
        self.rng_key = None
        self.green_recorder = None

    def _assert_pid_capacity(self, stage: str):
        max_session_tracked = int(MAX_TRACKED_SESSION_PIDS)
        max_session_count = int(jnp.max(self.jax_state.red_session_count))
        if max_session_count > max_session_tracked:
            raise RuntimeError(
                f"[{stage}] red_session_count overflow: observed {max_session_count} "
                f"> MAX_TRACKED_SESSION_PIDS={max_session_tracked}. "
                "CybORG session PID tracking is effectively unbounded; increase JAX PID capacity."
            )

    def reset(self):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=self.blue_cls,
            green_agent_class=self.green_cls,
            red_agent_class=self.red_cls,
            steps=self.max_steps,
        )
        self.cyborg_env = CybORG(scenario_generator=sg, seed=self.seed)
        self.cyborg_env.reset()

        self.jax_const = build_const_from_cyborg(self.cyborg_env)
        self.mappings = build_mappings_from_cyborg(self.cyborg_env)
        cyborg_state = self.cyborg_env.environment_controller.state
        controller = self.cyborg_env.environment_controller

        # CybORG action spaces seed red knowledge (known IPs/processes) even for
        # agents without active sessions. Mirror that into JAX init state.
        known_hosts_by_red = [set() for _ in range(NUM_RED_AGENTS)]
        scanned_hosts_by_red = [set() for _ in range(NUM_RED_AGENTS)]
        red_start_hosts = self.jax_const.red_start_hosts
        red_agent_active = self.jax_const.red_agent_active
        red_initial_discovered = self.jax_const.red_initial_discovered_hosts
        red_initial_scanned = self.jax_const.red_initial_scanned_hosts

        for r in range(NUM_RED_AGENTS):
            iface = controller.agent_interfaces.get(f"red_agent_{r}")
            if iface is None:
                continue
            aspace = iface.action_space

            for ip, known in getattr(aspace, "ip_address", {}).items():
                if not known:
                    continue
                hostname = cyborg_state.ip_addresses.get(ip)
                if hostname in self.mappings.hostname_to_idx:
                    known_hosts_by_red[r].add(self.mappings.hostname_to_idx[hostname])

            for sess in cyborg_state.sessions.get(f"red_agent_{r}", {}).values():
                for ip in getattr(sess, "ports", {}).keys():
                    hostname = cyborg_state.ip_addresses.get(ip)
                    if hostname in self.mappings.hostname_to_idx:
                        scanned_hosts_by_red[r].add(self.mappings.hostname_to_idx[hostname])

            if known_hosts_by_red[r]:
                red_agent_active = red_agent_active.at[r].set(True)
                red_start_hosts = red_start_hosts.at[r].set(min(known_hosts_by_red[r]))

            for hidx in known_hosts_by_red[r]:
                red_initial_discovered = red_initial_discovered.at[r, hidx].set(True)
            for hidx in scanned_hosts_by_red[r]:
                red_initial_scanned = red_initial_scanned.at[r, hidx].set(True)

        self.jax_const = self.jax_const.replace(
            red_start_hosts=red_start_hosts,
            red_agent_active=red_agent_active,
            red_initial_discovered_hosts=red_initial_discovered,
            red_initial_scanned_hosts=red_initial_scanned,
        )

        self.jax_state = create_initial_state()
        self.jax_state = self.jax_state.replace(
            host_services=jnp.array(self.jax_const.initial_services),
        )

        from CybORG.Shared.Session import RedAbstractSession

        start_sessions = jnp.zeros_like(self.jax_state.red_sessions)
        start_session_count = jnp.zeros((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS), dtype=jnp.int32)
        start_priv = jnp.zeros_like(self.jax_state.red_privilege)
        start_discovered = jnp.array(self.jax_const.red_initial_discovered_hosts)
        start_scanned = jnp.array(self.jax_const.red_initial_scanned_hosts)
        start_scanned_source_hosts = jnp.zeros((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS, GLOBAL_MAX_HOSTS), dtype=jnp.bool_)
        start_scan_anchor = jnp.full((NUM_RED_AGENTS,), -1, dtype=jnp.int32)
        start_abstract = jnp.zeros_like(self.jax_state.red_session_is_abstract)
        start_abstract_rank = jnp.full(
            (NUM_RED_AGENTS, GLOBAL_MAX_HOSTS),
            jnp.int32(ABSTRACT_RANK_NONE),
            dtype=jnp.int32,
        )
        start_next_abstract_rank = jnp.zeros((NUM_RED_AGENTS,), dtype=jnp.int32)
        start_session_pids = jnp.full((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS, MAX_TRACKED_SESSION_PIDS), -1, dtype=jnp.int32)
        start_abstract_session_pids = jnp.full(
            (NUM_RED_AGENTS, GLOBAL_MAX_HOSTS, MAX_TRACKED_SESSION_PIDS), -1, dtype=jnp.int32
        )
        start_privileged_session_pids = jnp.full(
            (NUM_RED_AGENTS, GLOBAL_MAX_HOSTS, MAX_TRACKED_SESSION_PIDS), -1, dtype=jnp.int32
        )
        max_pid_seen = 4999
        start_blue_suspicious_pids = jnp.full(
            (NUM_BLUE_AGENTS, GLOBAL_MAX_HOSTS, MAX_TRACKED_SUSPICIOUS_PIDS), -1, dtype=jnp.int32
        )
        host_compromised = self.jax_state.host_compromised
        fsm_states = self.jax_state.fsm_host_states
        for agent_name, sessions in cyborg_state.sessions.items():
            if not agent_name.startswith("red_agent_"):
                continue
            red_idx = int(agent_name.split("_")[-1])
            if red_idx >= NUM_RED_AGENTS:
                continue
            for sess in sessions.values():
                if sess.hostname in self.mappings.hostname_to_idx:
                    hidx = self.mappings.hostname_to_idx[sess.hostname]
                    start_sessions = start_sessions.at[red_idx, hidx].set(True)
                    start_session_count = start_session_count.at[red_idx, hidx].add(1)
                    start_discovered = start_discovered.at[red_idx, hidx].set(True)
                    if isinstance(sess, RedAbstractSession):
                        sess_ident = int(getattr(sess, "ident", -1))
                        start_abstract = start_abstract.at[red_idx, hidx].set(True)
                        if sess_ident >= 0:
                            start_abstract_rank = start_abstract_rank.at[red_idx, hidx].set(
                                jnp.minimum(start_abstract_rank[red_idx, hidx], jnp.int32(sess_ident))
                            )
                            start_next_abstract_rank = start_next_abstract_rank.at[red_idx].set(
                                jnp.maximum(start_next_abstract_rank[red_idx], jnp.int32(sess_ident + 1))
                            )
                    sess_pid = int(getattr(sess, "pid", -1))
                    if sess_pid >= 0:
                        max_pid_seen = max(max_pid_seen, sess_pid)
                        pid_row = start_session_pids[red_idx, hidx]
                        row_has_empty = bool(jnp.any(pid_row < 0))
                        row_has_pid = bool(jnp.any(pid_row == sess_pid))
                        if not row_has_empty and not row_has_pid:
                            raise RuntimeError(
                                "Reset overflow while syncing red session pids for "
                                f"red_agent_{red_idx} host={sess.hostname}: "
                                f"MAX_TRACKED_SESSION_PIDS={MAX_TRACKED_SESSION_PIDS}"
                            )
                        start_session_pids = start_session_pids.at[red_idx, hidx].set(
                            append_pid_to_row(pid_row, sess_pid)
                        )
                        if isinstance(sess, RedAbstractSession):
                            abs_row = start_abstract_session_pids[red_idx, hidx]
                            start_abstract_session_pids = start_abstract_session_pids.at[red_idx, hidx].set(
                                append_pid_to_row(abs_row, sess_pid)
                            )
                    level = 1
                    if hasattr(sess, "username") and sess.username in ("root", "SYSTEM"):
                        level = 2
                        if sess_pid >= 0:
                            priv_row = start_privileged_session_pids[red_idx, hidx]
                            start_privileged_session_pids = start_privileged_session_pids.at[red_idx, hidx].set(
                                append_pid_to_row(priv_row, sess_pid)
                            )
                    start_priv = start_priv.at[red_idx, hidx].set(jnp.maximum(start_priv[red_idx, hidx], level))
                    host_compromised = host_compromised.at[hidx].set(jnp.maximum(host_compromised[hidx], level))
                    for ip in getattr(sess, "ports", {}).keys():
                        scanned_host = cyborg_state.ip_addresses.get(ip)
                        if scanned_host in self.mappings.hostname_to_idx:
                            scanned_hidx = self.mappings.hostname_to_idx[scanned_host]
                            start_scanned_source_hosts = start_scanned_source_hosts.at[red_idx, scanned_hidx, hidx].set(
                                True
                            )
            if sessions:
                primary = sessions.get(0)
                if primary is None:
                    primary = next(iter(sessions.values()))
                if primary.hostname in self.mappings.hostname_to_idx:
                    anchor_host = self.mappings.hostname_to_idx[primary.hostname]
                    start_scan_anchor = start_scan_anchor.at[red_idx].set(anchor_host)
            if self.jax_const.red_agent_active[red_idx]:
                fsm_states = fsm_states.at[red_idx].set(fsm_red_init_states(self.jax_const, red_idx))
                start_host = int(self.jax_const.red_start_hosts[red_idx])
                start_scanned_source_hosts = start_scanned_source_hosts.at[red_idx, :, start_host].set(
                    start_scanned[red_idx]
                )

        for b in range(NUM_BLUE_AGENTS):
            blue_sessions = cyborg_state.sessions.get(f"blue_agent_{b}", {})
            for blue_sess in blue_sessions.values():
                sus_pids = getattr(blue_sess, "sus_pids", {})
                for hostname, pid_list in sus_pids.items():
                    if hostname not in self.mappings.hostname_to_idx:
                        continue
                    hidx = self.mappings.hostname_to_idx[hostname]
                    if len(pid_list) > MAX_TRACKED_SUSPICIOUS_PIDS:
                        raise RuntimeError(
                            "Reset overflow while syncing blue suspicious pids for "
                            f"blue_agent_{b} host={hostname}: observed {len(pid_list)} "
                            f"> MAX_TRACKED_SUSPICIOUS_PIDS={MAX_TRACKED_SUSPICIOUS_PIDS}"
                        )
                    slot = 0
                    for pid in pid_list:
                        start_blue_suspicious_pids = start_blue_suspicious_pids.at[b, hidx, slot].set(int(pid))
                        slot += 1

        self.jax_state = self.jax_state.replace(
            red_sessions=start_sessions,
            red_session_count=start_session_count,
            red_privilege=start_priv,
            red_session_pids=start_session_pids,
            red_session_abstract_pids=start_abstract_session_pids,
            red_session_privileged_pids=start_privileged_session_pids,
            red_next_pid=jnp.array(max_pid_seen + 1, dtype=jnp.int32),
            blue_suspicious_pids=start_blue_suspicious_pids,
            red_discovered_hosts=start_discovered,
            red_scanned_hosts=start_scanned,
            red_scanned_source_hosts=start_scanned_source_hosts,
            red_scan_anchor_host=start_scan_anchor,
            host_compromised=host_compromised,
            fsm_host_states=fsm_states,
            red_session_is_abstract=start_abstract,
            red_abstract_host_rank=start_abstract_rank,
            red_next_abstract_rank=start_next_abstract_rank,
        )
        self._assert_pid_capacity("reset")

        self.rng_key = jax.random.PRNGKey(self.seed)

        if self.sync_green_rng:
            from tests.differential.green_recorder import GreenRecorder

            self.green_recorder = GreenRecorder()
            self.green_recorder.install(self.cyborg_env, self.mappings)
            self.jax_state = self.jax_state.replace(
                use_green_randoms=jnp.array(True),
                use_red_pid_deltas=jnp.array(True),
                use_blue_decoy_pid_deltas=jnp.array(True),
            )

        from tests.differential.state_comparator import (
            extract_cyborg_snapshot,
            extract_jax_snapshot,
        )

        return (
            extract_cyborg_snapshot(self.cyborg_env, self.mappings),
            extract_jax_snapshot(self.jax_state, self.jax_const, self.mappings),
        )

    def step_red_only(self, agent_id: int, action_idx: int) -> StepResult:
        self.rng_key, subkey = jax.random.split(self.rng_key)

        cyborg_action = jax_red_to_cyborg(action_idx, agent_id, self.mappings)
        agent_name = f"red_agent_{agent_id}"
        self.cyborg_env.step(agent=agent_name, action=cyborg_action)

        self.jax_state = _jit_apply_red_action(self.jax_state, self.jax_const, agent_id, action_idx, subkey)

        from tests.differential.state_comparator import (
            compare_snapshots,
            extract_cyborg_snapshot,
            extract_jax_snapshot,
        )

        cyborg_snap = extract_cyborg_snapshot(self.cyborg_env, self.mappings)
        jax_snap = extract_jax_snapshot(self.jax_state, self.jax_const, self.mappings)
        diffs = compare_snapshots(cyborg_snap, jax_snap)

        return StepResult(step=int(self.jax_state.time), diffs=diffs)

    def step_blue_only(self, agent_id: int, action_idx: int) -> StepResult:
        cyborg_action = jax_blue_to_cyborg(action_idx, agent_id, self.mappings)
        agent_name = f"blue_agent_{agent_id}"
        self.cyborg_env.step(agent=agent_name, action=cyborg_action)

        self.jax_state = _jit_apply_blue_action(self.jax_state, self.jax_const, agent_id, action_idx)

        from tests.differential.state_comparator import (
            compare_snapshots,
            extract_cyborg_snapshot,
            extract_jax_snapshot,
        )

        cyborg_snap = extract_cyborg_snapshot(self.cyborg_env, self.mappings)
        jax_snap = extract_jax_snapshot(self.jax_state, self.jax_const, self.mappings)
        diffs = compare_snapshots(cyborg_snap, jax_snap)

        return StepResult(step=int(self.jax_state.time), diffs=diffs)

    def step(self, actions: dict) -> StepResult:
        self.rng_key, *subkeys = jax.random.split(self.rng_key, NUM_RED_AGENTS + 1)

        for agent_name, action_idx in actions.items():
            if agent_name.startswith("red_agent_"):
                cyborg_action = jax_red_to_cyborg(action_idx, _agent_idx(agent_name), self.mappings)
            else:
                cyborg_action = jax_blue_to_cyborg(action_idx, _agent_idx(agent_name), self.mappings)
            self.cyborg_env.step(agent=agent_name, action=cyborg_action)

        for agent_name, action_idx in actions.items():
            if agent_name.startswith("red_agent_"):
                aid = _agent_idx(agent_name)
                self.jax_state = _jit_apply_red_action(self.jax_state, self.jax_const, aid, action_idx, subkeys[aid])
            else:
                aid = _agent_idx(agent_name)
                self.jax_state = _jit_apply_blue_action(self.jax_state, self.jax_const, aid, action_idx)

        from tests.differential.state_comparator import (
            compare_snapshots,
            extract_cyborg_snapshot,
            extract_jax_snapshot,
        )

        cyborg_snap = extract_cyborg_snapshot(self.cyborg_env, self.mappings)
        jax_snap = extract_jax_snapshot(self.jax_state, self.jax_const, self.mappings)
        diffs = compare_snapshots(cyborg_snap, jax_snap)

        return StepResult(step=int(self.jax_state.time), diffs=diffs)

    def full_step(self, blue_actions=None) -> StepResult:
        """E2E step mirroring FsmRedCC4Env.step_env(): FSM red + green + blue + reassign + FSM update.

        Uses process_red_with_duration / process_blue_with_duration (same as training path)
        so JAX-native duration tracking is exercised.
        """
        self.rng_key, key_green, key_red, *subkeys = jax.random.split(self.rng_key, NUM_RED_AGENTS + 3)
        red_keys = jax.random.split(key_red, NUM_RED_AGENTS)

        if blue_actions is None and not self.use_cyborg_blue_policy:
            blue_actions = {b: BLUE_SLEEP for b in range(NUM_BLUE_AGENTS)}

        use_fsm = self.red_cls is FiniteStateRedAgent

        # --- Mirror step_env: advance phase + clear per-step fields ---
        self.jax_state = _jit_advance_and_clear(self.jax_state, self.jax_const)

        state_before = self.jax_state

        # --- FSM red action selection (matches FsmRedCC4Env.step_env) ---
        red_actions = {}
        target_hosts = []
        fsm_actions = []
        eligible_flags = []
        for r in range(NUM_RED_AGENTS):
            if use_fsm:
                is_busy = bool(self.jax_state.red_pending_ticks[r] > 0)
                if is_busy:
                    action = RED_SLEEP
                    host = jnp.int32(0)
                    fsm_act = jnp.int32(0)
                    eligible = jnp.bool_(False)
                else:
                    action, host, fsm_act, eligible = _jit_fsm_red_get_action_and_info(
                        self.jax_state, self.jax_const, r, red_keys[r]
                    )
                eff_host = jnp.where(is_busy, self.jax_state.red_pending_target_host[r], host)
                eff_fsm_act = jnp.where(is_busy, self.jax_state.red_pending_fsm_action[r], fsm_act)
                eff_eligible = jnp.where(is_busy, jnp.bool_(True), eligible)
                red_actions[r] = int(action)
                target_hosts.append(eff_host)
                fsm_actions.append(eff_fsm_act)
                eligible_flags.append(eff_eligible)
                prebound_source_kind = self.jax_state.red_pending_source_kind[r]
                prebound_source_host = self.jax_state.red_pending_source_host[r]
                if not is_busy:
                    action_type, _, target_host = decode_red_action(action, r, self.jax_const)
                    is_scan_action = (
                        (action_type == ACTION_TYPE_SCAN)
                        | (action_type == ACTION_TYPE_AGGRESSIVE_SCAN)
                        | (action_type == ACTION_TYPE_STEALTH_SCAN)
                    )
                    bound_anchor_source = select_bound_source_host(self.jax_state, self.jax_const, r)
                    prebound_source_kind = jnp.where(
                        is_scan_action,
                        jnp.where(
                            bound_anchor_source >= 0,
                            PENDING_SOURCE_KIND_SESSION_BINDING,
                            PENDING_SOURCE_KIND_NONE,
                        ),
                        PENDING_SOURCE_KIND_NONE,
                    )
                    prebound_source_host = jnp.int32(-1)

                self.jax_state = self.jax_state.replace(
                    red_pending_fsm_action=self.jax_state.red_pending_fsm_action.at[r].set(eff_fsm_act),
                    red_pending_target_host=self.jax_state.red_pending_target_host.at[r].set(eff_host),
                    red_pending_source_kind=self.jax_state.red_pending_source_kind.at[r].set(prebound_source_kind),
                    red_pending_source_host=self.jax_state.red_pending_source_host.at[r].set(prebound_source_host),
                )
            else:
                red_actions[r] = RED_SLEEP
                target_hosts.append(jnp.int32(0))
                fsm_actions.append(jnp.int32(0))
                eligible_flags.append(jnp.bool_(False))

        # --- CybORG side ---
        controller = self.cyborg_env.environment_controller
        cyborg_actions = {}
        for r, action_idx in red_actions.items():
            cyborg_actions[f"red_agent_{r}"] = jax_red_to_cyborg(action_idx, r, self.mappings)
        if blue_actions is not None:
            for b, action_idx in blue_actions.items():
                cyborg_actions[f"blue_agent_{b}"] = jax_blue_to_cyborg(action_idx, b, self.mappings)

        pre_services = {}
        cy_state = controller.state
        forced_primary_hosts_pre = _extract_primary_hosts(cy_state, self.mappings)
        for hostname in cy_state.hosts:
            pre_services[hostname] = set(cy_state.hosts[hostname].services.keys())

        controller.step(cyborg_actions)

        # Sync CybORG end-turn RedSessionCheck primary-session host choices for next-step anchor parity.
        forced_primary_hosts_post = _extract_primary_hosts(cy_state, self.mappings)
        abstract_host_rank = jnp.full(
            (NUM_RED_AGENTS, GLOBAL_MAX_HOSTS),
            jnp.int32(ABSTRACT_RANK_NONE),
            dtype=jnp.int32,
        )
        next_abstract_rank = jnp.zeros((NUM_RED_AGENTS,), dtype=jnp.int32)
        for r in range(NUM_RED_AGENTS):
            sessions = cy_state.sessions.get(f"red_agent_{r}", {})
            for sid, sess in sessions.items():
                if type(sess).__name__ != "RedAbstractSession":
                    continue
                host_idx = self.mappings.hostname_to_idx.get(sess.hostname)
                if host_idx is None:
                    continue
                abstract_host_rank = abstract_host_rank.at[r, host_idx].set(
                    jnp.minimum(abstract_host_rank[r, host_idx], jnp.int32(sid))
                )
                next_abstract_rank = next_abstract_rank.at[r].set(
                    jnp.maximum(next_abstract_rank[r], jnp.int32(sid + 1))
                )
        self.jax_state = self.jax_state.replace(
            red_abstract_host_rank=abstract_host_rank,
            red_next_abstract_rank=next_abstract_rank,
        )

        new_services_by_host = {}
        for hostname in cy_state.hosts:
            added = set(cy_state.hosts[hostname].services.keys()) - pre_services.get(hostname, set())
            if added:
                new_services_by_host[hostname] = added

        self._correct_pending_decoys(new_services_by_host)

        # --- Green RNG sync ---
        if self.green_recorder:
            step_fields, red_pid_deltas, blue_decoy_pid_deltas = self.green_recorder.extract_step(
                int(self.jax_state.time)
            )
            green_randoms = self.jax_state.green_randoms.at[self.jax_state.time].set(jnp.array(step_fields))
            red_pid_delta_row = self.jax_state.red_pid_deltas.at[self.jax_state.time].set(
                jnp.array(red_pid_deltas, dtype=jnp.int32)
            )
            blue_decoy_pid_delta_row = self.jax_state.blue_decoy_pid_deltas.at[self.jax_state.time].set(
                jnp.array(blue_decoy_pid_deltas, dtype=jnp.int32)
            )
            self.jax_state = self.jax_state.replace(
                green_randoms=green_randoms,
                red_pid_deltas=red_pid_delta_row,
                blue_decoy_pid_deltas=blue_decoy_pid_delta_row,
            )

        # --- JAX action application via duration functions (training code path) ---

        # Blue actions
        if blue_actions is not None:
            for b in range(NUM_BLUE_AGENTS):
                action_idx = self._resolve_blue_action(controller, b, blue_actions.get(b, BLUE_SLEEP))
                self.jax_state = _jit_process_blue_with_duration(self.jax_state, self.jax_const, b, action_idx)
        else:
            for b in range(NUM_BLUE_AGENTS):
                action_idx = self._resolve_blue_action(controller, b)
                self.jax_state = _jit_process_blue_with_duration(self.jax_state, self.jax_const, b, action_idx)

        # Green
        self.jax_state = _jit_apply_green(self.jax_state, self.jax_const, key_green)

        # Red actions
        for r in range(NUM_RED_AGENTS):
            action_idx = self._resolve_red_action(controller, r, red_actions.get(r, RED_SLEEP))
            self.jax_state = _jit_process_red_with_duration(
                self.jax_state,
                self.jax_const,
                r,
                action_idx,
                subkeys[r],
                forced_primary_hosts_pre[r],
            )

        self.jax_state = _jit_reassign(self.jax_state, self.jax_const)
        self.jax_state = _jit_apply_end_turn_monitors(self.jax_state, self.jax_const)
        self.jax_state = self.jax_state.replace(
            red_scan_anchor_host=forced_primary_hosts_post,
        )

        # --- FSM state updates (shared with FsmRedCC4Env) ---
        if use_fsm:
            executed_flags = jnp.array(
                [self.jax_state.red_pending_ticks[r] == 0 for r in range(NUM_RED_AGENTS)],
                dtype=jnp.bool_,
            )
            self.jax_state = _jit_fsm_red_post_step_update(
                state_before,
                self.jax_state,
                self.jax_const,
                jnp.asarray(target_hosts, dtype=jnp.int32),
                jnp.asarray(fsm_actions, dtype=jnp.int32),
                jnp.asarray(eligible_flags, dtype=jnp.bool_),
                executed_flags,
            )

        # --- Time increment ---
        self.jax_state = self.jax_state.replace(time=self.jax_state.time + 1)
        self._assert_pid_capacity("full_step")

        # --- Compare ---
        from tests.differential.state_comparator import compare_fast

        diffs = compare_fast(
            self.cyborg_env,
            self.jax_state,
            self.jax_const,
            self.mappings,
        )

        return StepResult(step=int(self.jax_state.time), diffs=diffs)

    def _resolve_red_action(self, controller, agent_idx, proposed_action):
        """Return JAX action matching what CybORG actually processed for this red agent.

        If CybORG invalidated the action (InvalidAction, duration=1), returns RED_SLEEP
        so JAX's duration system sees duration=1 too.
        If JAX agent is busy, returns RED_SLEEP (duration system uses stored pending action).
        """
        from CybORG.Simulator.Actions.Action import InvalidAction

        if bool(self.jax_state.red_pending_ticks[agent_idx] > 0):
            return RED_SLEEP

        agent_name = f"red_agent_{agent_idx}"
        executed = controller.action.get(agent_name, [])
        if any(isinstance(act, InvalidAction) for act in executed):
            return RED_SLEEP

        return proposed_action

    def _resolve_blue_action(self, controller, agent_idx, explicit_action=None):
        """Return JAX action matching what CybORG actually processed for this blue agent.

        With explicit_action: checks if CybORG invalidated the externally-provided action.
        Without explicit_action (CybORG-policy mode): reads CybORG's actual blue action
        and translates it to JAX.
        """
        from CybORG.Simulator.Actions.Action import InvalidAction

        if bool(self.jax_state.blue_pending_ticks[agent_idx] > 0):
            return BLUE_SLEEP

        agent_name = f"blue_agent_{agent_idx}"

        if explicit_action is not None:
            executed = controller.action.get(agent_name, [])
            if any(isinstance(act, InvalidAction) for act in executed):
                return BLUE_SLEEP
            return explicit_action

        pending = controller.actions_in_progress.get(agent_name)
        if pending is not None:
            action = pending["action"]
        else:
            executed = controller.action.get(agent_name, [])
            if not executed:
                return BLUE_SLEEP
            action = executed[0]

        if isinstance(action, InvalidAction) or type(action).__name__ == "Sleep":
            return BLUE_SLEEP

        return cyborg_blue_to_jax(action, agent_name, self.mappings)

    _SERVICE_TO_DECOY = {"haraka": 0, "apache2": 1, "tomcat": 2, "vsftpd": 3}

    def _correct_pending_decoys(self, new_services_by_host):
        for b in range(NUM_BLUE_AGENTS):
            if int(self.jax_state.blue_pending_ticks[b]) != 1:
                continue
            pending_action = int(self.jax_state.blue_pending_action[b])
            if not (BLUE_DECOY_START <= pending_action < BLUE_DECOY_END):
                continue
            offset = pending_action - BLUE_DECOY_START
            target_host = offset % GLOBAL_MAX_HOSTS
            hostname = self.mappings.idx_to_hostname.get(target_host)
            new_svcs = new_services_by_host.get(hostname, set())
            resolved_type = None
            for svc_name in new_svcs:
                if svc_name in self._SERVICE_TO_DECOY:
                    resolved_type = self._SERVICE_TO_DECOY[svc_name]
                    break
            if resolved_type is not None:
                correct_action = BLUE_DECOY_START + resolved_type * GLOBAL_MAX_HOSTS + target_host
                self.jax_state = self.jax_state.replace(
                    blue_pending_action=self.jax_state.blue_pending_action.at[b].set(correct_action)
                )
            else:
                self.jax_state = self.jax_state.replace(
                    blue_pending_ticks=self.jax_state.blue_pending_ticks.at[b].set(0),
                    blue_pending_action=self.jax_state.blue_pending_action.at[b].set(BLUE_SLEEP),
                )

    def run_episode(self, blue_policies=None, red_policy=None, max_steps=None) -> TestResult:
        max_steps = max_steps or self.max_steps
        self.reset()

        step_results = []
        error_count = 0

        for t in range(max_steps):
            actions = {}

            if red_policy:
                for r in range(NUM_RED_AGENTS):
                    actions[f"red_agent_{r}"] = red_policy(self.jax_state, self.jax_const, r)
            else:
                for r in range(NUM_RED_AGENTS):
                    actions[f"red_agent_{r}"] = RED_SLEEP

            if blue_policies:
                for b in range(NUM_BLUE_AGENTS):
                    actions[f"blue_agent_{b}"] = blue_policies(self.jax_state, self.jax_const, b)
            else:
                for b in range(NUM_BLUE_AGENTS):
                    actions[f"blue_agent_{b}"] = BLUE_SLEEP

            result = self.step(actions)
            step_results.append(result)

            from tests.differential.state_comparator import _ERROR_FIELDS

            error_count += sum(1 for d in result.diffs if d.field_name in _ERROR_FIELDS)

            self.jax_state = self.jax_state.replace(time=t + 1)

        return TestResult(
            steps_run=max_steps,
            step_results=step_results,
            error_diffs=error_count,
        )

    def get_cyborg_snapshot(self) -> StateSnapshot:
        from tests.differential.state_comparator import extract_cyborg_snapshot

        return extract_cyborg_snapshot(self.cyborg_env, self.mappings)

    def get_jax_snapshot(self) -> StateSnapshot:
        from tests.differential.state_comparator import extract_jax_snapshot

        return extract_jax_snapshot(self.jax_state, self.jax_const, self.mappings)


def _agent_idx(agent_name: str) -> int:
    return int(agent_name.split("_")[-1])


def _extract_primary_hosts(state, mappings) -> jax.Array:
    """Map CybORG session-0 host per red agent to JAX host indices."""
    primary_hosts = jnp.full((NUM_RED_AGENTS,), -1, dtype=jnp.int32)
    for r in range(NUM_RED_AGENTS):
        sessions = state.sessions.get(f"red_agent_{r}", {})
        primary = sessions.get(0)
        if primary is not None and primary.hostname in mappings.hostname_to_idx:
            primary_hosts = primary_hosts.at[r].set(mappings.hostname_to_idx[primary.hostname])
    return primary_hosts
