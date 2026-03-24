"""Recording infrastructure for CybORG green agent random calls.

Wraps CybORG's np_random objects to capture random values used by green agents,
then converts them into the precomputed (MAX_STEPS, GLOBAL_MAX_HOSTS, 8) array
that the JAX green agent implementation can consume.
"""

from dataclasses import dataclass, field

import numpy as np

from jaxborg.constants import (
    DECOY_IDS,
    GLOBAL_MAX_HOSTS,
    MAX_STEPS,
    NUM_BLUE_AGENTS,
    NUM_GREEN_RANDOM_FIELDS,
    NUM_RED_AGENTS,
    NUM_SERVICES,
    SERVICE_IDS,
)
from jaxborg.recording_rng import RecordingNPRandom

GREEN_SLEEP_JAX = 0
GREEN_LOCAL_WORK_JAX = 1
GREEN_ACCESS_SERVICE_JAX = 2

_ACTION_TYPE_TO_JAX = {
    "Sleep": GREEN_SLEEP_JAX,
    "GreenLocalWork": GREEN_LOCAL_WORK_JAX,
    "GreenAccessService": GREEN_ACCESS_SERVICE_JAX,
}

_LOCAL_WORK_DECOY_SERVICE_TO_TOKEN = {
    "haraka": NUM_SERVICES + DECOY_IDS["HarakaSMPT"],
    "apache2": NUM_SERVICES + DECOY_IDS["Apache"],
    "tomcat": NUM_SERVICES + DECOY_IDS["Tomcat"],
    "vsftpd": NUM_SERVICES + DECOY_IDS["Vsftpd"],
}

_DETECTION_SYNC_ACTIONS = {"AggressiveServiceDiscovery", "StealthServiceDiscovery"}


@dataclass
class RedActionRngUsage:
    agent_idx: int
    action_type: str
    random_calls: list[float] = field(default_factory=list)
    choice_sizes: list[int] = field(default_factory=list)
    choice_indices: list[int] = field(default_factory=list)
    integer_ranges: list[tuple[int, int]] = field(default_factory=list)

    def summary(self) -> str:
        return _summarize_rng_calls(self.random_calls, self.choice_sizes, self.integer_ranges)


@dataclass
class BlueActionRngUsage:
    agent_idx: int
    action_type: str
    random_calls: list[float] = field(default_factory=list)
    choice_sizes: list[int] = field(default_factory=list)
    integer_ranges: list[tuple[int, int]] = field(default_factory=list)

    def summary(self) -> str:
        return _summarize_rng_calls(self.random_calls, self.choice_sizes, self.integer_ranges)


@dataclass
class StepRandomSyncReport:
    detection_randoms: list[float] = field(default_factory=list)
    red_action_rng_usage: list[RedActionRngUsage] = field(default_factory=list)
    blue_action_rng_usage: list[BlueActionRngUsage] = field(default_factory=list)
    unsupported_detection_actions: list[str] = field(default_factory=list)
    unsupported_random_actions: list[str] = field(default_factory=list)
    red_pid_collisions: list[str] = field(default_factory=list)
    blue_decoy_pid_collisions: list[str] = field(default_factory=list)
    red_privesc_choices: dict[int, int] = field(default_factory=dict)
    red_session_check_choices: dict[int, int] = field(default_factory=dict)
    red_session_check_hosts: dict[int, int] = field(default_factory=dict)
    green_execution_order: list[int] = field(default_factory=list)
    full_execution_order: list[int] = field(default_factory=list)

    @property
    def detection_sync_supported(self) -> bool:
        return not self.unsupported_detection_actions

    @property
    def has_issues(self) -> bool:
        return bool(
            self.unsupported_detection_actions
            or self.unsupported_random_actions
            or self.red_pid_collisions
            or self.blue_decoy_pid_collisions
        )

    def format(self, step_idx: int) -> str:
        lines = [f"Random sync mismatch at step {step_idx}:"]
        if self.unsupported_detection_actions:
            lines.append("  Unsupported detection sync actions:")
            lines.extend(f"    - {entry}" for entry in self.unsupported_detection_actions)
        if self.unsupported_random_actions:
            lines.append("  Unsupported state RNG actions:")
            lines.extend(f"    - {entry}" for entry in self.unsupported_random_actions)
        if self.red_pid_collisions:
            lines.append("  Multiple red PID deltas in one step:")
            lines.extend(f"    - {entry}" for entry in self.red_pid_collisions)
        if self.blue_decoy_pid_collisions:
            lines.append("  Multiple blue decoy PID deltas in one step:")
            lines.extend(f"    - {entry}" for entry in self.blue_decoy_pid_collisions)
        return "\n".join(lines)


class GreenRecorder:
    """Records CybORG green agent random calls and converts to JAX precomputed format."""

    def __init__(self):
        self._state_recorder = None
        self._agent_recorders = {}
        self._action_log = []
        self._agent_to_host_idx = {}
        self._per_step_data = []

    def install(self, cyborg_env, mappings):
        """Wrap state.np_random and each green agent's np_random.

        Also wraps execute_action to track per-action state.np_random boundaries.
        """
        controller = cyborg_env.environment_controller
        self._controller = controller
        state = controller.state

        self._state_recorder = RecordingNPRandom(state.np_random)
        state.np_random = self._state_recorder
        self._host_idx_to_hostname = {}
        for host in state.hosts.values():
            host.np_random = self._state_recorder

        # Build IP → JAX host_idx mapping for server ordering parity
        self._ip_to_host_idx = {}
        for ip, hostname in state.ip_addresses.items():
            if hostname in mappings.hostname_to_idx:
                self._ip_to_host_idx[ip] = mappings.hostname_to_idx[hostname]

        for name, interface in controller.agent_interfaces.items():
            if not name.startswith("green_agent_"):
                continue
            agent = interface.agent
            recorder = RecordingNPRandom(agent.np_random)
            agent.np_random = recorder
            self._agent_recorders[name] = recorder

            own_ip = getattr(agent, "own_ip", None)
            hostname = state.ip_addresses.get(own_ip) if own_ip is not None else None
            if hostname and hostname in mappings.hostname_to_idx:
                host_idx = mappings.hostname_to_idx[hostname]
                self._agent_to_host_idx[name] = host_idx
                self._host_idx_to_hostname[host_idx] = hostname

        orig_execute = controller.execute_action

        def tracked_execute(action):
            start_idx = len(self._state_recorder.log)
            result = orig_execute(action)
            end_idx = len(self._state_recorder.log)
            agent_name = getattr(action, "agent", None)
            action_type = type(action).__name__
            self._action_log.append((agent_name, action_type, start_idx, end_idx))
            return result

        controller.execute_action = tracked_execute

    def extract_step(self, step_idx):
        """After a CybORG step: segment logs, convert to (GLOBAL_MAX_HOSTS, 8) uniforms."""
        fields = np.zeros((GLOBAL_MAX_HOSTS, NUM_GREEN_RANDOM_FIELDS), dtype=np.float32)
        red_pid_deltas = np.zeros((NUM_RED_AGENTS,), dtype=np.int32)
        blue_decoy_pid_deltas = np.zeros((NUM_BLUE_AGENTS,), dtype=np.int32)
        report = StepRandomSyncReport()

        # Capture CybORG's FULL execution order (all agents in slot-index form).
        # CybORG shuffles all same-priority actions each step.  Green agents are
        # slot NUM_BLUE_AGENTS + host_idx, red agents are NUM_BLUE_AGENTS +
        # GLOBAL_MAX_HOSTS + agent_idx, blue agents are agent_idx.
        green_exec_order = []
        full_exec_order = []
        seen_slots = set()
        for agent_name, _at, _s, _e in self._action_log:
            if not agent_name:
                continue
            if agent_name.startswith("green_agent_"):
                hidx = self._agent_to_host_idx.get(agent_name)
                if hidx is not None:
                    if hidx not in green_exec_order:
                        green_exec_order.append(hidx)
                    slot = NUM_BLUE_AGENTS + hidx
                    if slot not in seen_slots:
                        full_exec_order.append(slot)
                        seen_slots.add(slot)
            elif agent_name.startswith("red_agent_"):
                ridx = int(agent_name.split("_")[-1])
                if 0 <= ridx < NUM_RED_AGENTS:
                    slot = NUM_BLUE_AGENTS + GLOBAL_MAX_HOSTS + ridx
                    if slot not in seen_slots:
                        full_exec_order.append(slot)
                        seen_slots.add(slot)
            elif agent_name.startswith("blue_agent_"):
                bidx = int(agent_name.split("_")[-1])
                if 0 <= bidx < NUM_BLUE_AGENTS:
                    slot = bidx
                    if slot not in seen_slots:
                        full_exec_order.append(slot)
                        seen_slots.add(slot)
        report.green_execution_order = green_exec_order
        report.full_execution_order = full_exec_order

        for agent_name, action_type, start, end in self._action_log:
            calls = self._state_recorder.log[start:end]
            if agent_name is None or not agent_name.startswith("green_agent_"):
                if agent_name is not None and agent_name.startswith("red_agent_"):
                    ridx = int(agent_name.split("_")[-1])
                    if 0 <= ridx < NUM_RED_AGENTS:
                        delta = _extract_create_pid_delta(calls)
                        if delta > 0:
                            if red_pid_deltas[ridx] != 0:
                                report.red_pid_collisions.append(
                                    f"{agent_name}:{action_type} saw repeated create_pid deltas "
                                    f"{int(red_pid_deltas[ridx])} and {delta}"
                                )
                            red_pid_deltas[ridx] = delta
                if (
                    agent_name is not None
                    and agent_name.startswith("blue_agent_")
                    and action_type
                    in (
                        "DeployDecoy",
                        "Remove",
                    )
                ):
                    bidx = int(agent_name.split("_")[-1])
                    if 0 <= bidx < NUM_BLUE_AGENTS:
                        delta = _extract_create_pid_delta(calls)
                        if delta > 0:
                            if blue_decoy_pid_deltas[bidx] != 0:
                                report.blue_decoy_pid_collisions.append(
                                    f"{agent_name}:{action_type} saw repeated create_pid deltas "
                                    f"{int(blue_decoy_pid_deltas[bidx])} and {delta}"
                                )
                            blue_decoy_pid_deltas[bidx] = delta
                if agent_name is not None:
                    _record_non_green_random_usage(report, agent_name, action_type, calls)
                continue
            host_idx = self._agent_to_host_idx.get(agent_name)
            if host_idx is None:
                continue

            jax_action = _ACTION_TYPE_TO_JAX.get(action_type, GREEN_SLEEP_JAX)
            fields[host_idx, 0] = (jax_action + 0.5) / 3.0
            hostname = self._host_idx_to_hostname.get(host_idx)
            host = self._controller.state.hosts.get(hostname) if hostname is not None else None
            _map_calls_to_fields(fields, host_idx, action_type, calls, self._ip_to_host_idx, host)

        self._per_step_data.append(fields.copy())

        self._state_recorder.log.clear()
        for r in self._agent_recorders.values():
            r.log.clear()
        self._action_log.clear()

        return fields, red_pid_deltas, blue_decoy_pid_deltas, report

    def to_jax_array(self):
        """Return (MAX_STEPS, GLOBAL_MAX_HOSTS, 8) array."""
        import jax.numpy as jnp

        result = np.zeros((MAX_STEPS, GLOBAL_MAX_HOSTS, NUM_GREEN_RANDOM_FIELDS), dtype=np.float32)
        for i, step_data in enumerate(self._per_step_data):
            if i < MAX_STEPS:
                result[i] = step_data
        return jnp.array(result)


def _map_calls_to_fields(fields, host_idx, action_type, calls, ip_to_host_idx=None, host=None):
    """Map state.np_random calls to the 8-field format based on action type."""
    if action_type == "GreenLocalWork":
        _map_local_work_calls(fields, host_idx, calls, host)
    elif action_type == "GreenAccessService":
        _map_access_service_calls(fields, host_idx, calls, ip_to_host_idx)


def _extract_create_pid_delta(calls):
    for call in calls:
        if call[0] != "integers":
            continue
        low = call[2] if len(call) >= 4 else 0
        high = call[3] if len(call) >= 4 else call[2]
        if low == 1 and high == 10:
            return int(call[1])
    return 0


def _record_non_green_random_usage(report: StepRandomSyncReport, agent_name: str, action_type: str, calls):
    random_calls = [float(call[1]) for call in calls if call[0] == "random"]
    choice_sizes = [int(call[2]) for call in calls if call[0] == "choice" and int(call[2]) > 1]
    choice_indices = [int(call[1]) for call in calls if call[0] == "choice" and int(call[2]) > 1]
    integer_ranges = [
        (int(call[2]), int(call[3]))
        for call in calls
        if call[0] == "integers" and not _is_create_pid_call(call) and not _is_ephemeral_port_call(call)
    ]

    if agent_name.startswith("red_agent_"):
        if random_calls or choice_sizes or integer_ranges:
            report.red_action_rng_usage.append(
                RedActionRngUsage(
                    agent_idx=int(agent_name.split("_")[-1]),
                    action_type=action_type,
                    random_calls=random_calls,
                    choice_sizes=choice_sizes,
                    choice_indices=choice_indices,
                    integer_ranges=integer_ranges,
                )
            )
            return

    if agent_name.startswith("blue_agent_"):
        if random_calls or choice_sizes or integer_ranges:
            report.blue_action_rng_usage.append(
                BlueActionRngUsage(
                    agent_idx=int(agent_name.split("_")[-1]),
                    action_type=action_type,
                    random_calls=random_calls,
                    choice_sizes=choice_sizes,
                    integer_ranges=integer_ranges,
                )
            )
            return

    if random_calls or choice_sizes or integer_ranges:
        report.unsupported_random_actions.append(
            f"{agent_name}:{action_type} used {_summarize_rng_calls(random_calls, choice_sizes, integer_ranges)}"
        )


def _is_create_pid_call(call) -> bool:
    low = call[2] if len(call) >= 4 else 0
    high = call[3] if len(call) >= 4 else call[2]
    return low == 1 and high == 10


def _is_ephemeral_port_call(call) -> bool:
    low = call[2] if len(call) >= 4 else 0
    high = call[3] if len(call) >= 4 else call[2]
    return low == 49152 and high == 60000


def _summarize_rng_calls(random_calls, choice_sizes, integer_ranges) -> str:
    parts = []
    if random_calls:
        parts.append(f"{len(random_calls)} random()")
    if choice_sizes:
        sizes = ", ".join(str(size) for size in choice_sizes)
        parts.append(f"choice(n={sizes})")
    if integer_ranges:
        ranges = ", ".join(f"[{low},{high})" for low, high in integer_ranges)
        parts.append(f"integers{ranges}")
    return ", ".join(parts) if parts else "no captured calls"


def _map_local_work_calls(fields, host_idx, calls, host=None):
    """Map GreenLocalWork state.np_random calls to fields.

    Call pattern:
    [0] choice(available_services) -> chosen local service token
    [1] integers(0, 100) -> reliability roll
    If reliability passes (work succeeds):
        [2] random() -> FP roll
        [3] random() -> phishing roll
        Any further calls are from PhishingEmail sub-action.
        If phishing fires:
            [n] choice(red_agents) -> phishing source red agent (field 5)
            [n+1] integers(1, 10) -> host.create_pid delta (field 7)
    """
    if len(calls) < 2:
        return

    fields[host_idx, 1] = float(_resolve_local_work_service_token(calls[0], host))

    rel_call = calls[1]
    rel_val = rel_call[1]
    rel_low = rel_call[2] if len(rel_call) >= 4 else 0
    rel_high = rel_call[3] if len(rel_call) >= 4 else rel_call[2]
    rel_range = max(rel_high - rel_low, 1)
    rel_shifted = rel_val - rel_low
    fields[host_idx, 2] = (rel_shifted + 0.5) / max(rel_range, 1)

    trailing = calls[2:]
    random_calls = [call[1] for call in trailing if call[0] == "random"]
    if len(random_calls) >= 1:
        fields[host_idx, 3] = random_calls[0]
    if len(random_calls) >= 2:
        fields[host_idx, 4] = random_calls[1]

    # Extract phishing source red agent from PhishingEmail's choice call.
    # PhishingEmail._create_new_session calls state.np_random.choice(red_agents)
    # where red_agents is a list of (agentname, hostname) tuples.  The choice
    # result is a numpy ndarray (e.g. ['red_agent_0', 'hostname']).  The call
    # appears after the FP/phishing random() calls in the trailing span.
    for call in trailing:
        if call[0] != "choice":
            continue
        # call[3] is the chosen element — a numpy ndarray or tuple.
        chosen = call[3]
        try:
            agent_name = str(chosen[0]) if hasattr(chosen, "__len__") else str(chosen)
        except (IndexError, TypeError):
            agent_name = ""
        if agent_name.startswith("red_agent_"):
            agent_idx = int(agent_name.split("_")[-1])
            # Encode as agent_idx + 1 so that 0.0 = no phishing source.
            fields[host_idx, 5] = float(agent_idx + 1)
        break

    for call in trailing:
        if call[0] != "integers":
            continue
        low = call[2] if len(call) >= 4 else 0
        high = call[3] if len(call) >= 4 else call[2]
        if low == 1 and high == 10:
            pid_delta = call[1]
            fields[host_idx, 7] = (pid_delta - 0.5) / 9.0
            break


def _resolve_local_work_service_token(choice_call, host) -> int:
    if host is None or len(choice_call) < 4:
        return 0

    chosen_service = choice_call[3]
    for service_name, service in host.services.items():
        if service is not chosen_service:
            continue
        is_exploit_decoy = False
        for process in host.processes:
            if process.pid == service.process:
                is_exploit_decoy = getattr(getattr(process, "decoy_type", None), "name", "") == "EXPLOIT"
                break
        if is_exploit_decoy:
            decoy_key = str(service_name).lower()
            if decoy_key in _LOCAL_WORK_DECOY_SERVICE_TO_TOKEN:
                return int(_LOCAL_WORK_DECOY_SERVICE_TO_TOKEN[decoy_key])
        svc_key = getattr(service_name, "name", str(service_name)).upper()
        if svc_key in SERVICE_IDS:
            return int(SERVICE_IDS[svc_key])
        break
    return 0


def _map_access_service_calls(fields, host_idx, calls, ip_to_host_idx=None):
    """Map GreenAccessService state.np_random calls to fields.

    Call pattern:
    [0] choice(reachable_hosts) -> destination server selection
    If not blocked:
        [1] random() -> FP roll

    Field 5 stores the actual JAX host index of the chosen destination
    (not an index into CybORG's list) to avoid server ordering mismatches.
    """
    if len(calls) < 1:
        return

    chosen_ip = calls[0][3]
    if ip_to_host_idx and chosen_ip in ip_to_host_idx:
        fields[host_idx, 5] = float(ip_to_host_idx[chosen_ip])
    else:
        dest_idx, n_reachable = calls[0][1], calls[0][2]
        fields[host_idx, 5] = (dest_idx + 0.5) / max(n_reachable, 1)

    for call in calls[1:]:
        if call[0] == "random":
            fields[host_idx, 6] = call[1]
            break
