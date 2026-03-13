from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.actions import apply_blue_action, apply_red_action
from jaxborg.actions.blue_analyse import apply_blue_analyse
from jaxborg.actions.blue_decoys import apply_blue_decoy
from jaxborg.actions.blue_remove import apply_blue_remove
from jaxborg.actions.blue_restore import apply_blue_restore
from jaxborg.actions.encoding import (
    ACTION_TYPE_DISCOVER_DECEPTION,
    ACTION_TYPE_EXPLOIT_HTTP,
    ACTION_TYPE_EXPLOIT_SQL,
    ACTION_TYPE_EXPLOIT_SSH,
    ACTION_TYPE_PRIVESC,
    BLUE_DECOY_END,
    BLUE_DECOY_START,
    BLUE_SLEEP,
    RED_SLEEP,
    decode_red_action,
    encode_blue_action,
)
from jaxborg.actions.pids import append_pid_to_row
from jaxborg.agents.fsm_red import (
    fsm_red_apply_delayed_update,
    fsm_red_init_states,
    fsm_red_schedule_post_step_update,
    fsm_red_select_actions,
)
from jaxborg.constants import (
    ABSTRACT_RANK_NONE,
    GLOBAL_MAX_HOSTS,
    MAX_TRACKED_SESSION_PIDS,
    MAX_TRACKED_SUSPICIOUS_PIDS,
    NUM_BLUE_AGENTS,
    NUM_RED_AGENTS,
)
from jaxborg.env import apply_all_actions
from jaxborg.rewards import advance_mission_phase, compute_rewards
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
    host_service_reliability: dict = field(default_factory=dict)
    host_has_malware: dict = field(default_factory=dict)
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


def _capture_service_process_map(cy_state):
    service_map = {}
    for hostname, host in cy_state.hosts.items():
        host_services = {}
        for svc_name, svc in host.services.items():
            svc_str = str(svc_name).split(".")[-1].lower()
            pid = svc.process if hasattr(svc, "process") else svc.get("process")
            host_services[svc_str] = int(pid) if pid is not None else -1
        service_map[hostname] = host_services
    return service_map


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
def _jit_fsm_red_select_actions(state, const, red_keys):
    return fsm_red_select_actions(state, const, red_keys)


@jax.jit
def _jit_apply_red_action(state, const, agent_id, action_idx, key):
    return apply_red_action(state, const, jnp.int32(agent_id), jnp.int32(action_idx), key)


@jax.jit
def _jit_apply_blue_action(state, const, agent_id, action_idx):
    return apply_blue_action(state, const, jnp.int32(agent_id), jnp.int32(action_idx))


@jax.jit
def _jit_compute_rewards(state, const, impact_hosts, green_lwf, green_asf):
    return compute_rewards(state, const, impact_hosts, green_lwf, green_asf)


@jax.jit
def _jit_advance_and_clear(state, const):
    state = advance_mission_phase(state, const)
    any_covered = jnp.any(const.blue_agent_hosts, axis=0)
    return state.replace(
        red_activity_this_step=_ZERO_INT_HOSTS,
        green_lwf_this_step=_ZERO_BOOL_HOSTS,
        green_asf_this_step=_ZERO_BOOL_HOSTS,
        red_impact_attempted=_ZERO_BOOL_HOSTS,
        host_activity_detected=jnp.where(any_covered, False, state.host_activity_detected),
        host_exploit_detected=jnp.where(any_covered, False, state.host_exploit_detected),
    )


@jax.jit
def _jit_fsm_red_apply_delayed_update(state):
    return fsm_red_apply_delayed_update(state)


@jax.jit
def _jit_fsm_red_schedule_post_step_update(
    state_before, state_after, const, target_hosts, target_subnets, fsm_actions, eligible_flags, executed_flags=None
):
    return fsm_red_schedule_post_step_update(
        state_before,
        state_after,
        const,
        target_hosts,
        target_subnets,
        fsm_actions,
        eligible_flags,
        executed_flags,
    )


@jax.jit
def _jit_apply_all_actions(state, const, blue_actions, red_actions, key_green, red_keys, forced_primary_hosts):
    return apply_all_actions(state, const, blue_actions, red_actions, key_green, red_keys, forced_primary_hosts)


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
        strict_random_sync=False,
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
        self.strict_random_sync = strict_random_sync
        self.use_cyborg_blue_policy = use_cyborg_blue_policy
        self.cyborg_env = None
        self.jax_state = None
        self.jax_const = None
        self.mappings = None
        self.rng_key = None
        self.green_recorder = None
        self._blue_wrapper = None
        self._blue_unsupported_pending = {}
        self.last_random_sync_report = None

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
        self._blue_unsupported_pending = {}
        self.last_random_sync_report = None
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=self.blue_cls,
            green_agent_class=self.green_cls,
            red_agent_class=self.red_cls,
            steps=self.max_steps,
        )
        self.cyborg_env = CybORG(scenario_generator=sg, seed=self.seed)
        self.cyborg_env.reset()

        if self.check_obs:
            from CybORG.Agents.Wrappers.BlueFlatWrapper import BlueFlatWrapper

            self._blue_wrapper = BlueFlatWrapper(env=self.cyborg_env, pad_spaces=True)

        self.jax_const = build_const_from_cyborg(self.cyborg_env)
        self.mappings = build_mappings_from_cyborg(self.cyborg_env)
        cyborg_state = self.cyborg_env.environment_controller.state
        controller = self.cyborg_env.environment_controller

        for name, interface in controller.agent_interfaces.items():
            if not name.startswith("blue_agent_"):
                continue
            agent = getattr(interface, "agent", None)
            if agent is None or not hasattr(agent, "np_random"):
                continue
            agent_idx = int(name.split("_")[-1])
            agent.np_random = np.random.default_rng(self.seed * 100 + agent_idx)

        # CybORG action spaces seed red knowledge (known IPs/processes) even for
        # agents without active sessions. Mirror that into JAX init state.
        known_hosts_by_red = [set() for _ in range(NUM_RED_AGENTS)]
        scanned_hosts_by_red = [set() for _ in range(NUM_RED_AGENTS)]
        red_start_hosts = self.jax_const.red_start_hosts
        red_agent_active = jnp.zeros(NUM_RED_AGENTS, dtype=jnp.bool_)
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

            if iface is not None and iface.active:
                red_agent_active = red_agent_active.at[r].set(True)
            if known_hosts_by_red[r]:
                red_start_hosts = red_start_hosts.at[r].set(min(known_hosts_by_red[r]))

            for hidx in known_hosts_by_red[r]:
                red_initial_discovered = red_initial_discovered.at[r, hidx].set(True)
            for hidx in scanned_hosts_by_red[r]:
                red_initial_scanned = red_initial_scanned.at[r, hidx].set(True)

        self.jax_const = self.jax_const.replace(
            red_start_hosts=red_start_hosts,
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
            if red_agent_active[red_idx]:
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
            red_agent_active=red_agent_active,
        )
        self._assert_pid_capacity("reset")

        self.rng_key = jax.random.PRNGKey(self.seed)

        if self.sync_green_rng:
            from tests.differential.green_recorder import GreenRecorder

            self.green_recorder = GreenRecorder()
            self.green_recorder.install(self.cyborg_env, self.mappings)
            self.jax_const = self.jax_const.replace(
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
        self.cyborg_env.step(agent=agent_name, action=cyborg_action, skip_valid_action_check=True)

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
        cyborg_action = jax_blue_to_cyborg(action_idx, agent_id, self.mappings, const=self.jax_const)
        agent_name = f"blue_agent_{agent_id}"
        self.cyborg_env.step(agent=agent_name, action=cyborg_action, skip_valid_action_check=True)

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
                cyborg_action = jax_blue_to_cyborg(
                    action_idx, _agent_idx(agent_name), self.mappings, const=self.jax_const
                )
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
        detection_count = 0
        using_detection_sync = False

        if blue_actions is None and not self.use_cyborg_blue_policy:
            blue_actions = {b: BLUE_SLEEP for b in range(NUM_BLUE_AGENTS)}

        use_fsm = self.red_cls is FiniteStateRedAgent

        # --- Mirror step_env: advance phase + clear per-step fields ---
        self.jax_state = _jit_advance_and_clear(self.jax_state, self.jax_const)
        if use_fsm:
            self.jax_state = _jit_fsm_red_apply_delayed_update(self.jax_state)

        state_before = self.jax_state

        # --- Duration parity check: JAX and CybORG must agree on busy state ---
        controller = self.cyborg_env.environment_controller
        self._assert_duration_parity(controller)

        # --- FSM red action selection (shared with FsmRedCC4Env.step_env) ---
        if use_fsm:
            red_action_arr, target_hosts_arr, target_subnets_arr, fsm_actions_arr, eligible_arr, self.jax_state = (
                _jit_fsm_red_select_actions(self.jax_state, self.jax_const, red_keys)
            )
            red_actions = {r: int(red_action_arr[r]) for r in range(NUM_RED_AGENTS)}
            target_hosts = [target_hosts_arr[r] for r in range(NUM_RED_AGENTS)]
            target_subnets = [target_subnets_arr[r] for r in range(NUM_RED_AGENTS)]
            fsm_actions = [fsm_actions_arr[r] for r in range(NUM_RED_AGENTS)]
            eligible_flags = [eligible_arr[r] for r in range(NUM_RED_AGENTS)]
        else:
            red_actions = {r: RED_SLEEP for r in range(NUM_RED_AGENTS)}
            target_hosts = [jnp.int32(0) for _ in range(NUM_RED_AGENTS)]
            target_subnets = [jnp.int32(0) for _ in range(NUM_RED_AGENTS)]
            fsm_actions = [jnp.int32(0) for _ in range(NUM_RED_AGENTS)]
            eligible_flags = [jnp.bool_(False) for _ in range(NUM_RED_AGENTS)]

        # --- CybORG side ---
        cyborg_actions = {}
        for r, action_idx in red_actions.items():
            cyborg_actions[f"red_agent_{r}"] = jax_red_to_cyborg(action_idx, r, self.mappings)
        if blue_actions is not None:
            for b, action_idx in blue_actions.items():
                cyborg_actions[f"blue_agent_{b}"] = jax_blue_to_cyborg(
                    action_idx, b, self.mappings, const=self.jax_const
                )

        cy_state = controller.state
        forced_primary_hosts_pre = _extract_primary_hosts(cy_state, self.mappings)
        pre_service_map = _capture_service_process_map(cy_state)

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

        changed_services_by_host = {}
        post_service_map = _capture_service_process_map(cy_state)
        for hostname, post_services in post_service_map.items():
            before_services = pre_service_map.get(hostname, {})
            changed = {svc_name for svc_name, pid in post_services.items() if before_services.get(svc_name) != pid}
            if changed:
                changed_services_by_host[hostname] = changed

        self._correct_pending_decoys(changed_services_by_host)

        # --- Green RNG sync ---
        if self.green_recorder:
            step_fields, red_pid_deltas, blue_decoy_pid_deltas, random_sync_report = self.green_recorder.extract_step(
                int(self.jax_state.time)
            )
            self._sync_red_action_randoms(random_sync_report, red_actions)
            self._validate_blue_action_randoms(random_sync_report, controller)
            self.last_random_sync_report = random_sync_report
            green_randoms = self.jax_const.green_randoms.at[self.jax_state.time].set(jnp.array(step_fields))
            red_pid_delta_row = self.jax_const.red_pid_deltas.at[self.jax_state.time].set(
                jnp.array(red_pid_deltas, dtype=jnp.int32)
            )
            blue_decoy_pid_delta_row = self.jax_const.blue_decoy_pid_deltas.at[self.jax_state.time].set(
                jnp.array(blue_decoy_pid_deltas, dtype=jnp.int32)
            )
            detection_count = len(random_sync_report.detection_randoms)
            detection_randoms = self.jax_const.detection_randoms
            # Detection replay is step-local. Reusing the same buffer each step
            # avoids leaking stale prior-step consumption into current-step sync.
            if detection_count:
                if detection_count > detection_randoms.shape[0]:
                    raise RuntimeError(
                        "Detection random overflow while syncing CybORG RNG at "
                        f"step {int(self.jax_state.time)}: count={detection_count} "
                        f"> step_capacity={int(detection_randoms.shape[0])}"
                    )
                detection_randoms = detection_randoms.at[:detection_count].set(
                    jnp.array(random_sync_report.detection_randoms, dtype=jnp.float32)
                )
            using_detection_sync = bool(detection_count) and random_sync_report.detection_sync_supported
            self.jax_const = self.jax_const.replace(
                green_randoms=green_randoms,
                red_pid_deltas=red_pid_delta_row,
                blue_decoy_pid_deltas=blue_decoy_pid_delta_row,
                detection_randoms=detection_randoms,
                use_detection_randoms=jnp.array(using_detection_sync),
            )
            self.jax_state = self.jax_state.replace(
                detection_random_index=jnp.array(0, dtype=jnp.int32),
            )
            if self.strict_random_sync and random_sync_report.has_issues:
                raise AssertionError(random_sync_report.format(int(self.jax_state.time)))

        # --- JAX action application via shared apply_all_actions (same as training code path) ---
        blue_action_arr = jnp.array(
            [
                self._resolve_blue_action(controller, b, blue_actions.get(b, BLUE_SLEEP))
                if blue_actions is not None
                else self._resolve_blue_action(controller, b)
                for b in range(NUM_BLUE_AGENTS)
            ],
            dtype=jnp.int32,
        )
        red_action_arr = jnp.array(
            [self._resolve_red_action(controller, r, red_actions.get(r, RED_SLEEP)) for r in range(NUM_RED_AGENTS)],
            dtype=jnp.int32,
        )

        self.jax_state = _jit_apply_all_actions(
            self.jax_state,
            self.jax_const,
            blue_action_arr,
            red_action_arr,
            key_green,
            jnp.stack(subkeys[:NUM_RED_AGENTS]),
            forced_primary_hosts_pre,
        )
        if self.green_recorder:
            detection_consumed = int(self.jax_state.detection_random_index)
            expected_consumed = detection_count if using_detection_sync else 0
            if detection_consumed != expected_consumed:
                raise AssertionError(
                    "Detection RNG sync mismatch at step "
                    f"{int(self.jax_state.time)}: expected JAX to consume {expected_consumed} "
                    f"precomputed draw(s), consumed {detection_consumed}"
                )
        self._schedule_pending_generic_decoys(controller)
        self._sync_pending_unsupported_blue_actions(controller, changed_services_by_host)
        primary_abstract_flags = _extract_primary_is_abstract(cy_state)
        self.jax_state = self.jax_state.replace(
            red_scan_anchor_host=forced_primary_hosts_post,
            red_primary_is_abstract=primary_abstract_flags,
        )

        # --- FSM state updates (shared with FsmRedCC4Env) ---
        if use_fsm:
            executed_flags = jnp.array(
                [self.jax_state.red_pending_ticks[r] == 0 for r in range(NUM_RED_AGENTS)],
                dtype=jnp.bool_,
            )
            self.jax_state = _jit_fsm_red_schedule_post_step_update(
                state_before,
                self.jax_state,
                self.jax_const,
                jnp.asarray(target_hosts, dtype=jnp.int32),
                jnp.asarray(target_subnets, dtype=jnp.int32),
                jnp.asarray(fsm_actions, dtype=jnp.int32),
                jnp.asarray(eligible_flags, dtype=jnp.bool_),
                executed_flags,
            )

        # --- Time increment ---
        self.jax_state = self.jax_state.replace(time=self.jax_state.time + 1)
        self._assert_pid_capacity("full_step")

        # --- Reward comparison ---
        impact_hosts = self.jax_state.red_impact_attempted
        jax_reward = float(
            _jit_compute_rewards(
                self.jax_state,
                self.jax_const,
                impact_hosts,
                self.jax_state.green_lwf_this_step,
                self.jax_state.green_asf_this_step,
            )
        )
        cyborg_reward = controller.reward.get("Blue", {}).get("BlueRewardMachine", 0.0)

        # --- Compare ---
        from tests.differential.state_comparator import StateDiff, compare_fast

        diffs = compare_fast(
            self.cyborg_env,
            self.jax_state,
            self.jax_const,
            self.mappings,
        )

        if abs(jax_reward - cyborg_reward) > 1e-6:
            diffs.append(StateDiff("rewards", cyborg_reward, jax_reward))

        # --- Observation comparison ---
        if self.check_obs and self._blue_wrapper is not None:
            import numpy as np

            from jaxborg.observations import get_blue_obs

            for b in range(NUM_BLUE_AGENTS):
                agent_name = f"blue_agent_{b}"
                cyborg_obs_dict = self.cyborg_env.get_observation(agent_name)
                cyborg_obs = self._blue_wrapper.observation_change(agent_name, cyborg_obs_dict)
                cyborg_obs = np.asarray(cyborg_obs, dtype=np.float32)

                jax_obs = np.asarray(get_blue_obs(self.jax_state, self.jax_const, b), dtype=np.float32)

                # Trim to same length (padded wrapper may be longer)
                min_len = min(len(cyborg_obs), len(jax_obs))
                cyborg_trimmed = cyborg_obs[:min_len]
                jax_trimmed = jax_obs[:min_len]

                if not np.allclose(cyborg_trimmed, jax_trimmed, atol=1e-5):
                    mismatched_indices = np.where(np.abs(cyborg_trimmed - jax_trimmed) > 1e-5)[0]
                    detail = ", ".join(
                        f"idx={i} cy={cyborg_trimmed[i]:.0f} jax={jax_trimmed[i]:.4f}" for i in mismatched_indices[:10]
                    )
                    diffs.append(StateDiff("observation", cyborg_trimmed, jax_trimmed, f"{agent_name}: {detail}"))

        return StepResult(step=int(self.jax_state.time), diffs=diffs)

    def _sync_red_action_randoms(self, random_sync_report, red_actions):
        synced_randoms = list(random_sync_report.detection_randoms)

        for usage in random_sync_report.red_action_rng_usage:
            action_idx = self._effective_red_action_for_sync(
                usage.agent_idx,
                red_actions.get(usage.agent_idx, RED_SLEEP),
            )
            action_type, _, _ = decode_red_action(action_idx, usage.agent_idx, self.jax_const)
            action_type = int(action_type)
            if usage.action_type in {"AggressiveServiceDiscovery", "StealthServiceDiscovery"}:
                if len(usage.random_calls) == 1 and not usage.choice_sizes and not usage.integer_ranges:
                    synced_randoms.extend(usage.random_calls)
                    continue
                random_sync_report.unsupported_detection_actions.append(
                    f"red_agent_{usage.agent_idx}:{usage.action_type} used {usage.summary()}"
                )
                continue
            if usage.action_type == "DiscoverDeception":
                synced = self._sync_discover_deception_randoms(usage, action_type, action_idx)
                if synced is not None:
                    synced_randoms.extend(synced)
                    continue
                random_sync_report.unsupported_detection_actions.append(
                    f"red_agent_{usage.agent_idx}:{usage.action_type} used {usage.summary()}"
                )
                continue
            if usage.action_type == "RedSessionCheck":
                if not usage.random_calls and not usage.integer_ranges and len(usage.choice_sizes) == 1:
                    continue
                random_sync_report.unsupported_random_actions.append(
                    f"red_agent_{usage.agent_idx}:{usage.action_type} used {usage.summary()}"
                )
                continue
            if action_type in {ACTION_TYPE_EXPLOIT_SSH, ACTION_TYPE_EXPLOIT_HTTP, ACTION_TYPE_EXPLOIT_SQL}:
                if usage.random_calls and not usage.choice_sizes and not usage.integer_ranges:
                    synced_randoms.extend(usage.random_calls)
                    continue
                random_sync_report.unsupported_random_actions.append(
                    f"red_agent_{usage.agent_idx}:{usage.action_type} used {usage.summary()}"
                )
                continue
            if action_type == ACTION_TYPE_PRIVESC:
                if not usage.random_calls and not usage.integer_ranges and len(usage.choice_sizes) == 1:
                    continue
                random_sync_report.unsupported_random_actions.append(
                    f"red_agent_{usage.agent_idx}:{usage.action_type} used {usage.summary()}"
                )
                continue
            random_sync_report.unsupported_random_actions.append(
                f"red_agent_{usage.agent_idx}:{usage.action_type} used {usage.summary()}"
            )

        random_sync_report.detection_randoms = synced_randoms

    def _effective_red_action_for_sync(self, agent_idx: int, proposed_action: int) -> int:
        if bool(self.jax_state.red_pending_ticks[agent_idx] > 0):
            return int(self.jax_state.red_pending_action[agent_idx])
        return int(proposed_action)

    def _validate_blue_action_randoms(self, random_sync_report, controller) -> None:
        for usage in random_sync_report.blue_action_rng_usage:
            if usage.action_type != "DeployDecoy":
                random_sync_report.unsupported_random_actions.append(
                    f"blue_agent_{usage.agent_idx}:{usage.action_type} used {usage.summary()}"
                )
                continue
            if usage.random_calls or usage.integer_ranges or len(usage.choice_sizes) != 1:
                random_sync_report.unsupported_random_actions.append(
                    f"blue_agent_{usage.agent_idx}:{usage.action_type} used {usage.summary()}"
                )
                continue
            pending = controller.actions_in_progress.get(f"blue_agent_{usage.agent_idx}")
            executed = controller.action.get(f"blue_agent_{usage.agent_idx}", [])
            executed_is_generic_decoy = bool(executed) and type(executed[0]).__name__ == "DeployDecoy"
            pending_is_generic_decoy = pending is not None and type(pending["action"]).__name__ == "DeployDecoy"
            if not pending_is_generic_decoy and not executed_is_generic_decoy:
                random_sync_report.unsupported_random_actions.append(
                    f"blue_agent_{usage.agent_idx}:{usage.action_type} used {usage.summary()}"
                )

    def _sync_discover_deception_randoms(self, usage, action_type: int, action_idx: int) -> list[float] | None:
        if action_type != ACTION_TYPE_DISCOVER_DECEPTION:
            return None
        if usage.choice_sizes or usage.integer_ranges or not usage.random_calls:
            return None

        controller = self.cyborg_env.environment_controller
        observations = controller.observation.get(f"red_agent_{usage.agent_idx}")
        detected = False
        if observations is not None:
            for obs in observations.observations:
                for host_data in obs.data.values():
                    if not isinstance(host_data, dict):
                        continue
                    for process in host_data.get("Processes", []):
                        properties = process.get("Properties", [])
                        if "decoy" in properties:
                            detected = True
                            break
                    if detected:
                        break
                if detected:
                    break

        _, _, target_host = decode_red_action(action_idx, usage.agent_idx, self.jax_const)
        target_host = int(target_host)
        hostname = self.mappings.idx_to_hostname.get(target_host)
        if hostname is None:
            return None

        has_decoys = bool(np.any(np.asarray(self.jax_state.host_decoys[target_host])))
        if detected:
            return [0.0, 1.0] if has_decoys else [1.0, 0.0]
        return [1.0, 1.0]

    @staticmethod
    def _cyborg_discover_deception_detected(processes, random_calls: list[float]) -> bool | None:
        call_idx = 0
        detected = False

        for process in processes:
            if call_idx >= len(random_calls):
                return None
            decoy_draw = random_calls[call_idx]
            call_idx += 1
            is_exploit_decoy = getattr(getattr(process, "decoy_type", None), "name", "") == "EXPLOIT"
            if decoy_draw <= 0.5 and is_exploit_decoy:
                detected = True
                continue

            if call_idx >= len(random_calls):
                return None
            fp_draw = random_calls[call_idx]
            call_idx += 1
            if fp_draw <= 0.1 and not is_exploit_decoy:
                detected = True

        if call_idx != len(random_calls):
            return None
        return detected

    def _assert_duration_parity(self, controller):
        """Assert JAX and CybORG agree on which agents are busy (action in progress).

        A mismatch means JAX and CybORG use different durations for the same action type,
        causing the two systems to silently execute different action sequences.
        """
        step = int(self.jax_state.time)
        for r in range(NUM_RED_AGENTS):
            jax_busy = bool(self.jax_state.red_pending_ticks[r] > 0)
            cy_entry = controller.actions_in_progress.get(f"red_agent_{r}")
            cy_busy = cy_entry is not None
            if jax_busy != cy_busy:
                cy_detail = f"remaining_ticks={cy_entry['remaining_ticks']}" if cy_busy else "idle"
                raise AssertionError(
                    f"Duration mismatch for red_agent_{r} at step {step}: "
                    f"JAX red_pending_ticks={int(self.jax_state.red_pending_ticks[r])}, "
                    f"CybORG {cy_detail}"
                )
        for b in range(NUM_BLUE_AGENTS):
            jax_busy = self._blue_agent_is_busy(b)
            cy_entry = controller.actions_in_progress.get(f"blue_agent_{b}")
            cy_busy = cy_entry is not None
            if jax_busy != cy_busy:
                cy_detail = f"remaining_ticks={cy_entry['remaining_ticks']}" if cy_busy else "idle"
                raise AssertionError(
                    f"Duration mismatch for blue_agent_{b} at step {step}: "
                    f"JAX blue_pending_ticks={int(self.jax_state.blue_pending_ticks[b])}, "
                    f"CybORG {cy_detail}"
                )

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

        if self._blue_agent_is_busy(agent_idx):
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
            if type(action).__name__ == "DeployDecoy":
                return BLUE_SLEEP
            if self._is_unsupported_blue_host_action(action):
                return BLUE_SLEEP
        else:
            executed = controller.action.get(agent_name, [])
            if not executed:
                return BLUE_SLEEP
            action = executed[0]

        if isinstance(action, InvalidAction) or type(action).__name__ == "Sleep":
            return BLUE_SLEEP

        return cyborg_blue_to_jax(action, agent_name, self.mappings, const=self.jax_const)

    _SERVICE_TO_DECOY = {"haraka": 0, "apache2": 1, "tomcat": 2, "vsftpd": 3}
    _GENERIC_DECOY_PLACEHOLDER = "DeployDecoy_HarakaSMPT"
    _UNSUPPORTED_BLUE_HOST_ACTIONS = {
        "Analyse": apply_blue_analyse,
        "Remove": apply_blue_remove,
        "Restore": apply_blue_restore,
    }

    def _blue_agent_is_busy(self, agent_idx: int) -> bool:
        return bool(self.jax_state.blue_pending_ticks[agent_idx] > 0) or agent_idx in self._blue_unsupported_pending

    def _is_unsupported_blue_host_action(self, action) -> bool:
        action_name = type(action).__name__
        if action_name not in self._UNSUPPORTED_BLUE_HOST_ACTIONS:
            return False
        host_idx = self.mappings.hostname_to_idx.get(getattr(action, "hostname", None))
        if host_idx is None:
            return False
        encoded = encode_blue_action(action_name, host_idx, 0, const=self.jax_const)
        return encoded == BLUE_SLEEP

    def _apply_unsupported_blue_host_action(self, agent_idx: int, action_name: str, host_idx: int):
        apply_fn = self._UNSUPPORTED_BLUE_HOST_ACTIONS[action_name]
        self.jax_state = apply_fn(self.jax_state, self.jax_const, agent_idx, host_idx)

    def _correct_pending_decoys(self, new_services_by_host):
        for b in range(NUM_BLUE_AGENTS):
            if int(self.jax_state.blue_pending_ticks[b]) != 1:
                continue
            pending_action = int(self.jax_state.blue_pending_action[b])
            if not (BLUE_DECOY_START <= pending_action < BLUE_DECOY_END):
                continue
            from jaxborg.constants import ACTION_HOST_SLOTS, OBS_HOSTS_PER_SUBNET

            offset = pending_action - BLUE_DECOY_START
            flat_slot = offset % ACTION_HOST_SLOTS
            sid = flat_slot // OBS_HOSTS_PER_SUBNET
            slot_within = flat_slot % OBS_HOSTS_PER_SUBNET
            target_host = int(self.jax_const.obs_host_map[sid, slot_within])
            hostname = self.mappings.idx_to_hostname.get(target_host)
            new_svcs = new_services_by_host.get(hostname, set())
            resolved_type = None
            for svc_name in new_svcs:
                if svc_name in self._SERVICE_TO_DECOY:
                    resolved_type = self._SERVICE_TO_DECOY[svc_name]
                    break
            if resolved_type is not None:
                correct_action = BLUE_DECOY_START + resolved_type * ACTION_HOST_SLOTS + flat_slot
                self.jax_state = self.jax_state.replace(
                    blue_pending_action=self.jax_state.blue_pending_action.at[b].set(correct_action)
                )
            else:
                self.jax_state = self.jax_state.replace(
                    blue_pending_ticks=self.jax_state.blue_pending_ticks.at[b].set(0),
                    blue_pending_action=self.jax_state.blue_pending_action.at[b].set(BLUE_SLEEP),
                )

    def _schedule_pending_generic_decoys(self, controller):
        blue_pending_ticks = self.jax_state.blue_pending_ticks
        blue_pending_action = self.jax_state.blue_pending_action
        changed = False

        for b in range(NUM_BLUE_AGENTS):
            if int(blue_pending_ticks[b]) != 0:
                continue
            pending = controller.actions_in_progress.get(f"blue_agent_{b}")
            if pending is None:
                continue
            action = pending["action"]
            if type(action).__name__ != "DeployDecoy":
                continue
            host_idx = self.mappings.hostname_to_idx.get(action.hostname)
            if host_idx is None:
                continue
            placeholder = encode_blue_action(
                self._GENERIC_DECOY_PLACEHOLDER,
                host_idx,
                b,
                const=self.jax_const,
            )
            if placeholder == BLUE_SLEEP:
                continue
            blue_pending_ticks = blue_pending_ticks.at[b].set(int(pending["remaining_ticks"]))
            blue_pending_action = blue_pending_action.at[b].set(placeholder)
            changed = True

        if changed:
            self.jax_state = self.jax_state.replace(
                blue_pending_ticks=blue_pending_ticks,
                blue_pending_action=blue_pending_action,
            )

    def _sync_pending_unsupported_blue_actions(self, controller, new_services_by_host):
        from CybORG.Simulator.Actions.Action import InvalidAction

        next_pending = {}

        for b in range(NUM_BLUE_AGENTS):
            agent_name = f"blue_agent_{b}"
            pending = controller.actions_in_progress.get(agent_name)
            if pending is not None and type(pending["action"]).__name__ == "DeployDecoy":
                hostname = pending["action"].hostname
                host_idx = self.mappings.hostname_to_idx.get(hostname)
                if host_idx is not None:
                    placeholder = encode_blue_action(
                        self._GENERIC_DECOY_PLACEHOLDER,
                        host_idx,
                        b,
                        const=self.jax_const,
                    )
                    if placeholder == BLUE_SLEEP:
                        next_pending[b] = ("DeployDecoy", host_idx, int(pending["remaining_ticks"]))
                        continue
            if pending is not None and self._is_unsupported_blue_host_action(pending["action"]):
                action = pending["action"]
                host_idx = self.mappings.hostname_to_idx[action.hostname]
                next_pending[b] = (type(action).__name__, host_idx, int(pending["remaining_ticks"]))
                continue

            prior = self._blue_unsupported_pending.get(b)
            if prior is None:
                continue

            action_name, host_idx, _ = prior
            hostname = self.mappings.idx_to_hostname[host_idx]
            executed = controller.action.get(agent_name, [])
            failed = any(isinstance(act, InvalidAction) for act in executed)
            completed = any(
                type(act).__name__ == action_name and getattr(act, "hostname", None) == hostname for act in executed
            )
            if completed and not failed:
                if action_name == "DeployDecoy":
                    resolved = None
                    for service_name in new_services_by_host.get(hostname, set()):
                        if service_name in self._SERVICE_TO_DECOY:
                            resolved = self._SERVICE_TO_DECOY[service_name]
                            break
                    if resolved is not None:
                        self.jax_state = apply_blue_decoy(self.jax_state, self.jax_const, b, host_idx, resolved)
                    continue
                self._apply_unsupported_blue_host_action(b, action_name, host_idx)

        self._blue_unsupported_pending = next_pending

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


def _extract_primary_is_abstract(state) -> jax.Array:
    """Return per-agent bool: True when CybORG session 0 is RedAbstractSession."""
    from CybORG.Shared.Session import RedAbstractSession

    flags = jnp.ones((NUM_RED_AGENTS,), dtype=jnp.bool_)
    for r in range(NUM_RED_AGENTS):
        sessions = state.sessions.get(f"red_agent_{r}", {})
        primary = sessions.get(0)
        if primary is not None:
            flags = flags.at[r].set(isinstance(primary, RedAbstractSession))
        else:
            # No session 0 -> no valid primary; keep True as default (benign;
            # privesc will fail on other checks when there's no session).
            pass
    return flags
