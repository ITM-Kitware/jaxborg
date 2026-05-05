import chex
import jax.numpy as jnp
from flax import struct

from jaxborg.constants import ABSTRACT_RANK_NONE, CC4_CONFIG
from jaxborg.scenarios.config import ScenarioConfig


@struct.dataclass
class SimulatorConst:
    host_active: chex.Array  # (num_hosts,) bool
    host_subnet: chex.Array  # (num_hosts,) int
    host_is_router: chex.Array  # (num_hosts,) bool
    host_is_server: chex.Array  # (num_hosts,) bool
    host_is_user: chex.Array  # (num_hosts,) bool
    subnet_adjacency: chex.Array  # (num_subnets, num_subnets) bool
    data_links: chex.Array  # (num_hosts, num_hosts) bool

    initial_services: chex.Array  # (num_hosts, num_services) bool
    host_has_bruteforceable_user: chex.Array  # (num_hosts,) bool
    host_has_rfi: chex.Array  # (num_hosts,) bool
    host_respond_to_ping: chex.Array  # (num_hosts,) bool
    host_initial_max_pid: chex.Array  # (num_hosts,) int32 — max host process pid at reset

    blue_agent_subnets: chex.Array  # (num_blue_agents, num_subnets) bool
    blue_agent_hosts: chex.Array  # (num_blue_agents, num_hosts) bool
    red_start_hosts: chex.Array  # (num_red_agents,) int
    red_agent_subnets: chex.Array  # (num_red_agents, num_subnets) bool — allowed subnets per red agent
    red_initial_discovered_hosts: chex.Array  # (num_red_agents, num_hosts) bool
    red_initial_scanned_hosts: chex.Array  # (num_red_agents, num_hosts) bool
    host_info_links: chex.Array  # (num_hosts, num_hosts) bool

    green_agent_host: chex.Array  # (num_hosts,) int — green agent index per host, -1 if none
    green_agent_active: chex.Array  # (num_hosts,) bool
    num_green_agents: chex.Array  # scalar int32

    phase_rewards: chex.Array  # (mission_phases, num_subnets, 3) float — LWF/ASF/RIA per subnet per phase
    phase_boundaries: chex.Array  # (mission_phases,) int — step at which each phase starts
    allowed_subnet_pairs: chex.Array  # (mission_phases, num_subnets, num_subnets) bool

    obs_host_map: chex.Array  # (num_subnets, obs_hosts_per_subnet) int — global host idx per subnet in obs order
    blue_obs_subnets: chex.Array  # (num_blue_agents, blue_max_observed_subnets) int
    comms_policy: chex.Array  # (mission_phases, num_subnets, num_subnets) bool — True = not connected

    max_steps: int
    num_hosts: chex.Array  # scalar int32

    green_agents_active: chex.Array  # scalar bool — False = skip green actions (SleepAgent parity)


@struct.dataclass
class SimulatorState:
    time: int
    done: chex.Array  # scalar bool
    mission_phase: chex.Array  # scalar int

    host_compromised: chex.Array  # (num_hosts,) int — 0=None, 1=User, 2=Privileged
    host_services: chex.Array  # (num_hosts, num_services) bool
    host_service_reliability: chex.Array  # (num_hosts, num_services) int32 — 0-100
    host_decoys: chex.Array  # (num_hosts, num_decoy_types) bool
    host_decoy_reliability: chex.Array  # (num_hosts, num_decoy_types) int32 — 0-100
    ot_service_stopped: chex.Array  # (num_hosts,) bool

    red_sessions: chex.Array  # (num_red_agents, num_hosts) bool
    red_session_count: chex.Array  # (num_red_agents, num_hosts) int32 — exact session multiplicity
    # (num_red_agents, num_hosts) int32 — only RedAbstractSession-type (primary+phishing)
    red_abstract_session_count: chex.Array
    red_suspicious_process_count: chex.Array  # (num_red_agents, num_hosts) int — known suspicious user pids
    red_privilege: chex.Array  # (num_red_agents, num_hosts) int — 0/1/2
    red_discovered_hosts: chex.Array  # (num_red_agents, num_hosts) bool
    red_scanned_hosts: chex.Array  # (num_red_agents, num_hosts) bool
    red_scanned_source_hosts: chex.Array  # (num_red_agents, num_hosts, num_hosts) bool
    red_scan_source_pid: chex.Array  # (num_red_agents, num_hosts) int32 — PID owning scan memory per source host
    red_scan_anchor_host: chex.Array  # (num_red_agents,) int — host owning CybORG-like scan memory session
    red_primary_is_abstract: chex.Array  # (num_red_agents,) bool — session-0 equiv is abstract
    red_primary_pid: chex.Array  # (num_red_agents,) int32 — session-0 equiv PID, -1 if absent

    red_scan_success: chex.Array  # (num_red_agents,) bool — scan action succeeded this step (for FSM)
    red_exploit_success: chex.Array  # (num_red_agents,) bool — exploit succeeded (pre-reassign)
    red_discover_success: chex.Array  # (num_red_agents,) bool — discover action executed this step (for FSM)
    red_activity_this_step: chex.Array  # (num_hosts,) int — 0=None, 1=Scan, 2=Exploit
    host_activity_detected: chex.Array  # (num_hosts,) bool — network_connections current
    old_host_activity_detected: chex.Array  # (num_hosts,) bool — network_connections aged
    host_exploit_detected: chex.Array  # (num_hosts,) bool — malicious_processes current
    old_host_exploit_detected: chex.Array  # (num_hosts,) bool — malicious_processes aged
    host_suspicious_process: chex.Array  # (num_hosts,) bool
    host_has_malware: chex.Array  # (num_hosts,) bool

    blocked_zones: chex.Array  # (num_subnets, num_subnets) bool
    messages: chex.Array  # (num_blue_agents, num_blue_agents, message_length) float

    fsm_host_states: chex.Array  # (num_red_agents, num_hosts) int — FSM state per red agent per host
    fsm_host_entered: chex.Array  # (num_red_agents, num_hosts) bool — host entered FSM
    red_fsm_delayed_states: chex.Array  # (num_red_agents, num_hosts) int — hidden FSM state to apply next step
    red_fsm_delayed_pending: chex.Array  # scalar bool — apply delayed FSM state at next step start

    green_lwf_this_step: chex.Array  # (num_hosts,) bool — green LocalWork failed on that source host
    green_asf_this_step: chex.Array  # (num_hosts,) bool — green AccessService failed from that source host

    red_session_sandboxed: chex.Array  # (num_red_agents, num_hosts) bool — sandboxed exploit sessions
    red_session_is_abstract: chex.Array  # (num_red_agents, num_hosts) bool — True for exploit-created sessions
    red_abstract_host_rank: chex.Array  # (num_red_agents, num_hosts) int32 — min abstract-session order per host
    red_next_abstract_rank: chex.Array  # (num_red_agents,) int32 — next abstract-session order value
    red_session_pids: chex.Array  # (num_red_agents, num_hosts, max_tracked_session_pids) int32
    red_session_abstract_pids: chex.Array  # (num_red_agents, num_hosts, max_tracked_session_pids) int32
    red_session_privileged_pids: chex.Array  # (num_red_agents, num_hosts, max_tracked_session_pids) int32
    red_next_pid: chex.Array  # scalar int32 — next PID to allocate
    blue_suspicious_pids: chex.Array  # (num_blue_agents, num_hosts, max_tracked_suspicious_pids) int32
    host_process_creation_pids: chex.Array  # (num_hosts, max_tracked_suspicious_pids) int32
    # pending monitor events
    host_decoy_process_pids: chex.Array  # (num_hosts, num_decoy_types) int32 — live decoy process pids
    host_orphaned_decoy_max_pid: chex.Array  # (num_hosts,) int32 — max PID of orphaned decoy procs
    host_max_pid: chex.Array  # (num_hosts,) int32 — running max PID per host

    red_pending_ticks: chex.Array  # (num_red_agents,) int32 — 0 = idle
    red_pending_action: chex.Array  # (num_red_agents,) int32 — queued action index
    red_pending_key: chex.Array  # (num_red_agents, 2) uint32 — stored RNG key
    red_pending_source_kind: chex.Array  # (num_red_agents,) int32 — 0 none, 1 host, 2 session binding, 3 bound none
    red_pending_source_host: chex.Array  # (num_red_agents,) int32 — queued scan source (anchor) host
    # (num_red_agents,) int32 — creation-time abstract session count for exploit 1/N roll
    red_pending_visible_sessions: chex.Array
    # (num_red_agents,) int32 — cumulative count of unique abstract session IDs
    # ever observed.  Replicates CybORG's server_session dict which never removes
    # destroyed sessions: each new phishing/reassignment event adds a new session
    # ID, and IDs are never deleted even after Blue Restore.
    red_server_session_count: chex.Array

    blue_pending_ticks: chex.Array  # (num_blue_agents,) int32 — 0 = idle
    blue_pending_action: chex.Array  # (num_blue_agents,) int32 — queued action index

    red_pending_fsm_action: chex.Array  # (num_red_agents,) int32 — stored FSM action type for deferred actions
    red_pending_target_host: chex.Array  # (num_red_agents,) int32 — stored target host for deferred actions
    red_pending_target_subnet: chex.Array  # (num_red_agents,) int32 — stored target subnet for deferred discover

    red_impact_attempted: chex.Array  # (num_hosts,) bool — any red Impact reached execution this step
    # (num_red_agents, num_hosts) bool — per-agent record of which target hosts an Impact reached
    # execution on this step.  Materialised so that ``red_impact_attempted`` can be regated against
    # post-reassignment session counts at end-of-step (matching CybORG's BlueRewardMachine, which
    # only counts Impact reward for an agent that still has at least one active session at reward
    # computation time — see CybORG/Shared/BlueRewardMachine.py:108).
    red_impact_attempted_by_agent: chex.Array

    red_agent_active: chex.Array  # (num_red_agents,) bool — dynamically activated via session reassignment


def create_initial_const(cfg: ScenarioConfig = CC4_CONFIG) -> SimulatorConst:
    n_hosts = cfg.num_hosts
    n_subnets = cfg.num_subnets
    n_blue = cfg.num_blue_agents
    n_red = cfg.num_red_agents
    n_services = cfg.num_services
    n_phases = cfg.mission_phases
    max_steps = cfg.max_steps
    return SimulatorConst(
        host_active=jnp.zeros(n_hosts, dtype=jnp.bool_),
        host_subnet=jnp.zeros(n_hosts, dtype=jnp.int32),
        host_is_router=jnp.zeros(n_hosts, dtype=jnp.bool_),
        host_is_server=jnp.zeros(n_hosts, dtype=jnp.bool_),
        host_is_user=jnp.zeros(n_hosts, dtype=jnp.bool_),
        subnet_adjacency=jnp.zeros((n_subnets, n_subnets), dtype=jnp.bool_),
        data_links=jnp.zeros((n_hosts, n_hosts), dtype=jnp.bool_),
        initial_services=jnp.zeros((n_hosts, n_services), dtype=jnp.bool_),
        host_has_bruteforceable_user=jnp.zeros(n_hosts, dtype=jnp.bool_),
        host_has_rfi=jnp.zeros(n_hosts, dtype=jnp.bool_),
        host_respond_to_ping=jnp.zeros(n_hosts, dtype=jnp.bool_),
        host_initial_max_pid=jnp.zeros(n_hosts, dtype=jnp.int32),
        blue_agent_subnets=jnp.zeros((n_blue, n_subnets), dtype=jnp.bool_),
        blue_agent_hosts=jnp.zeros((n_blue, n_hosts), dtype=jnp.bool_),
        red_start_hosts=jnp.zeros(n_red, dtype=jnp.int32),
        red_agent_subnets=jnp.zeros((n_red, n_subnets), dtype=jnp.bool_),
        red_initial_discovered_hosts=jnp.zeros((n_red, n_hosts), dtype=jnp.bool_),
        red_initial_scanned_hosts=jnp.zeros((n_red, n_hosts), dtype=jnp.bool_),
        host_info_links=jnp.zeros((n_hosts, n_hosts), dtype=jnp.bool_),
        green_agent_host=jnp.full(n_hosts, -1, dtype=jnp.int32),
        green_agent_active=jnp.zeros(n_hosts, dtype=jnp.bool_),
        num_green_agents=jnp.int32(0),
        phase_rewards=jnp.zeros((n_phases, n_subnets, 3), dtype=jnp.float32),
        phase_boundaries=jnp.zeros(n_phases, dtype=jnp.int32),
        allowed_subnet_pairs=jnp.zeros((n_phases, n_subnets, n_subnets), dtype=jnp.bool_),
        obs_host_map=jnp.full((n_subnets, cfg.obs_hosts_per_subnet), n_hosts, dtype=jnp.int32),
        blue_obs_subnets=jnp.full((n_blue, cfg.blue_max_observed_subnets), -1, dtype=jnp.int32),
        comms_policy=jnp.zeros((n_phases, n_subnets, n_subnets), dtype=jnp.bool_),
        max_steps=max_steps,
        num_hosts=jnp.int32(0),
        green_agents_active=jnp.array(True),
    )


def create_initial_state(cfg: ScenarioConfig = CC4_CONFIG) -> SimulatorState:
    n_hosts = cfg.num_hosts
    n_subnets = cfg.num_subnets
    n_blue = cfg.num_blue_agents
    n_red = cfg.num_red_agents
    n_services = cfg.num_services
    n_decoys = cfg.num_decoy_types
    return SimulatorState(
        time=0,
        done=jnp.array(False),
        mission_phase=jnp.array(0, dtype=jnp.int32),
        host_compromised=jnp.zeros(n_hosts, dtype=jnp.int32),
        host_services=jnp.zeros((n_hosts, n_services), dtype=jnp.bool_),
        host_service_reliability=jnp.full((n_hosts, n_services), 100, dtype=jnp.int32),
        host_decoys=jnp.zeros((n_hosts, n_decoys), dtype=jnp.bool_),
        host_decoy_reliability=jnp.full((n_hosts, n_decoys), 100, dtype=jnp.int32),
        ot_service_stopped=jnp.zeros(n_hosts, dtype=jnp.bool_),
        red_sessions=jnp.zeros((n_red, n_hosts), dtype=jnp.bool_),
        red_session_count=jnp.zeros((n_red, n_hosts), dtype=jnp.int32),
        red_abstract_session_count=jnp.zeros((n_red, n_hosts), dtype=jnp.int32),
        red_suspicious_process_count=jnp.zeros((n_red, n_hosts), dtype=jnp.int32),
        red_privilege=jnp.zeros((n_red, n_hosts), dtype=jnp.int32),
        red_discovered_hosts=jnp.zeros((n_red, n_hosts), dtype=jnp.bool_),
        red_scanned_hosts=jnp.zeros((n_red, n_hosts), dtype=jnp.bool_),
        red_scanned_source_hosts=jnp.zeros((n_red, n_hosts, n_hosts), dtype=jnp.bool_),
        red_scan_source_pid=jnp.full((n_red, n_hosts), -1, dtype=jnp.int32),
        red_scan_anchor_host=jnp.full(n_red, -1, dtype=jnp.int32),
        red_primary_is_abstract=jnp.ones(n_red, dtype=jnp.bool_),
        red_primary_pid=jnp.full(n_red, -1, dtype=jnp.int32),
        red_scan_success=jnp.zeros(n_red, dtype=jnp.bool_),
        red_exploit_success=jnp.zeros(n_red, dtype=jnp.bool_),
        red_discover_success=jnp.zeros(n_red, dtype=jnp.bool_),
        red_activity_this_step=jnp.zeros(n_hosts, dtype=jnp.int32),
        host_activity_detected=jnp.zeros(n_hosts, dtype=jnp.bool_),
        old_host_activity_detected=jnp.zeros(n_hosts, dtype=jnp.bool_),
        host_exploit_detected=jnp.zeros(n_hosts, dtype=jnp.bool_),
        old_host_exploit_detected=jnp.zeros(n_hosts, dtype=jnp.bool_),
        host_suspicious_process=jnp.zeros(n_hosts, dtype=jnp.bool_),
        host_has_malware=jnp.zeros(n_hosts, dtype=jnp.bool_),
        blocked_zones=jnp.zeros((n_subnets, n_subnets), dtype=jnp.bool_),
        messages=jnp.zeros((n_blue, n_blue, cfg.message_length), dtype=jnp.float32),
        fsm_host_states=jnp.zeros((n_red, n_hosts), dtype=jnp.int32),
        fsm_host_entered=jnp.zeros((n_red, n_hosts), dtype=jnp.bool_),
        red_fsm_delayed_states=jnp.zeros((n_red, n_hosts), dtype=jnp.int32),
        red_fsm_delayed_pending=jnp.array(False),
        red_session_sandboxed=jnp.zeros((n_red, n_hosts), dtype=jnp.bool_),
        red_session_is_abstract=jnp.zeros((n_red, n_hosts), dtype=jnp.bool_),
        red_abstract_host_rank=jnp.full(
            (n_red, n_hosts),
            jnp.int32(ABSTRACT_RANK_NONE),
            dtype=jnp.int32,
        ),
        red_next_abstract_rank=jnp.zeros(n_red, dtype=jnp.int32),
        red_session_pids=jnp.full((n_red, n_hosts, cfg.max_tracked_session_pids), -1, dtype=jnp.int32),
        red_session_abstract_pids=jnp.full((n_red, n_hosts, cfg.max_tracked_session_pids), -1, dtype=jnp.int32),
        red_session_privileged_pids=jnp.full((n_red, n_hosts, cfg.max_tracked_session_pids), -1, dtype=jnp.int32),
        red_next_pid=jnp.array(5000, dtype=jnp.int32),
        blue_suspicious_pids=jnp.full((n_blue, n_hosts, cfg.max_tracked_suspicious_pids), -1, dtype=jnp.int32),
        host_process_creation_pids=jnp.full((n_hosts, cfg.max_tracked_suspicious_pids), -1, dtype=jnp.int32),
        host_decoy_process_pids=jnp.full((n_hosts, n_decoys), -1, dtype=jnp.int32),
        host_orphaned_decoy_max_pid=jnp.zeros(n_hosts, dtype=jnp.int32),
        host_max_pid=jnp.zeros(n_hosts, dtype=jnp.int32),
        green_lwf_this_step=jnp.zeros(n_hosts, dtype=jnp.bool_),
        green_asf_this_step=jnp.zeros(n_hosts, dtype=jnp.bool_),
        red_pending_ticks=jnp.zeros(n_red, dtype=jnp.int32),
        red_pending_action=jnp.zeros(n_red, dtype=jnp.int32),
        red_pending_key=jnp.zeros((n_red, 2), dtype=jnp.uint32),
        red_pending_source_kind=jnp.zeros(n_red, dtype=jnp.int32),
        red_pending_source_host=jnp.full(n_red, -1, dtype=jnp.int32),
        red_pending_visible_sessions=jnp.ones(n_red, dtype=jnp.int32),
        red_server_session_count=jnp.zeros(n_red, dtype=jnp.int32),
        blue_pending_ticks=jnp.zeros(n_blue, dtype=jnp.int32),
        blue_pending_action=jnp.zeros(n_blue, dtype=jnp.int32),
        red_pending_fsm_action=jnp.zeros(n_red, dtype=jnp.int32),
        red_pending_target_host=jnp.zeros(n_red, dtype=jnp.int32),
        red_pending_target_subnet=jnp.zeros(n_red, dtype=jnp.int32),
        red_impact_attempted=jnp.zeros(n_hosts, dtype=jnp.bool_),
        red_impact_attempted_by_agent=jnp.zeros((n_red, n_hosts), dtype=jnp.bool_),
        red_agent_active=jnp.zeros(n_red, dtype=jnp.bool_),
    )
