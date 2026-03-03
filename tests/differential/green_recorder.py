"""Recording infrastructure for CybORG green agent random calls.

Wraps CybORG's np_random objects to capture random values used by green agents,
then converts them into the precomputed (MAX_STEPS, GLOBAL_MAX_HOSTS, 8) array
that the JAX green agent implementation can consume.
"""

import numpy as np

from jaxborg.constants import GLOBAL_MAX_HOSTS, MAX_STEPS, NUM_BLUE_AGENTS, NUM_GREEN_RANDOM_FIELDS, NUM_RED_AGENTS
from tests.differential.recording_rng import RecordingNPRandom

GREEN_SLEEP_JAX = 0
GREEN_LOCAL_WORK_JAX = 1
GREEN_ACCESS_SERVICE_JAX = 2

_ACTION_TYPE_TO_JAX = {
    "Sleep": GREEN_SLEEP_JAX,
    "GreenLocalWork": GREEN_LOCAL_WORK_JAX,
    "GreenAccessService": GREEN_ACCESS_SERVICE_JAX,
}


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
        state = controller.state

        self._state_recorder = RecordingNPRandom(state.np_random)
        state.np_random = self._state_recorder
        for host in state.hosts.values():
            host.np_random = self._state_recorder

        for name, interface in controller.agent_interfaces.items():
            if not name.startswith("green_agent_"):
                continue
            agent = interface.agent
            recorder = RecordingNPRandom(agent.np_random)
            agent.np_random = recorder
            self._agent_recorders[name] = recorder

            hostname = state.ip_addresses.get(agent.own_ip)
            if hostname and hostname in mappings.hostname_to_idx:
                self._agent_to_host_idx[name] = mappings.hostname_to_idx[hostname]

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

        for agent_name, action_type, start, end in self._action_log:
            calls = self._state_recorder.log[start:end]
            if agent_name is None or not agent_name.startswith("green_agent_"):
                if agent_name is not None and agent_name.startswith("red_agent_"):
                    ridx = int(agent_name.split("_")[-1])
                    if 0 <= ridx < NUM_RED_AGENTS:
                        delta = _extract_create_pid_delta(calls)
                        if delta > 0:
                            red_pid_deltas[ridx] = delta
                if agent_name is not None and agent_name.startswith("blue_agent_") and action_type == "DeployDecoy":
                    bidx = int(agent_name.split("_")[-1])
                    if 0 <= bidx < NUM_BLUE_AGENTS:
                        delta = _extract_create_pid_delta(calls)
                        if delta > 0:
                            blue_decoy_pid_deltas[bidx] = delta
                continue
            host_idx = self._agent_to_host_idx.get(agent_name)
            if host_idx is None:
                continue

            jax_action = _ACTION_TYPE_TO_JAX.get(action_type, GREEN_SLEEP_JAX)
            fields[host_idx, 0] = (jax_action + 0.5) / 3.0
            _map_calls_to_fields(fields, host_idx, action_type, calls)

        self._per_step_data.append(fields.copy())

        self._state_recorder.log.clear()
        for r in self._agent_recorders.values():
            r.log.clear()
        self._action_log.clear()

        return fields, red_pid_deltas, blue_decoy_pid_deltas

    def to_jax_array(self):
        """Return (MAX_STEPS, GLOBAL_MAX_HOSTS, 8) array."""
        import jax.numpy as jnp

        result = np.zeros((MAX_STEPS, GLOBAL_MAX_HOSTS, NUM_GREEN_RANDOM_FIELDS), dtype=np.float32)
        for i, step_data in enumerate(self._per_step_data):
            if i < MAX_STEPS:
                result[i] = step_data
        return jnp.array(result)


def _map_calls_to_fields(fields, host_idx, action_type, calls):
    """Map state.np_random calls to the 8-field format based on action type."""
    if action_type == "GreenLocalWork":
        _map_local_work_calls(fields, host_idx, calls)
    elif action_type == "GreenAccessService":
        _map_access_service_calls(fields, host_idx, calls)


def _extract_create_pid_delta(calls):
    for call in calls:
        if call[0] != "integers":
            continue
        low = call[2] if len(call) >= 4 else 0
        high = call[3] if len(call) >= 4 else call[2]
        if low == 1 and high == 10:
            return int(call[1])
    return 0


def _map_local_work_calls(fields, host_idx, calls):
    """Map GreenLocalWork state.np_random calls to fields.

    Call pattern:
    [0] choice(available_services) -> service selection
    [1] integers(0, 100) -> reliability roll
    If reliability passes (work succeeds):
        [2] random() -> FP roll
        [3] random() -> phishing roll
        Any further calls are from PhishingEmail sub-action.
        If a session is created:
            [n] integers(1, 10) -> host.create_pid delta
    """
    if len(calls) < 2:
        return

    _, svc_idx, n_services = calls[0]
    fields[host_idx, 1] = (svc_idx + 0.5) / max(n_services, 1)

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

    for call in trailing:
        if call[0] != "integers":
            continue
        low = call[2] if len(call) >= 4 else 0
        high = call[3] if len(call) >= 4 else call[2]
        if low == 1 and high == 10:
            pid_delta = call[1]
            fields[host_idx, 7] = (pid_delta - 0.5) / 9.0
            break


def _map_access_service_calls(fields, host_idx, calls):
    """Map GreenAccessService state.np_random calls to fields.

    Call pattern:
    [0] choice(reachable_hosts) -> destination server selection
    If not blocked:
        [1] random() -> FP roll
    """
    if len(calls) < 1:
        return

    _, dest_idx, n_reachable = calls[0]
    fields[host_idx, 5] = (dest_idx + 0.5) / max(n_reachable, 1)

    for call in calls[1:]:
        if call[0] == "random":
            fields[host_idx, 6] = call[1]
            break
