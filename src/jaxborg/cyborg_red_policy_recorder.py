"""Record CybORG FiniteStateRedAgent choice outcomes as JAX-consumable choice tapes."""

from ipaddress import IPv4Address, IPv4Network, ip_address

import numpy as np

from jaxborg.constants import GLOBAL_MAX_HOSTS, MAX_STEPS, NUM_RED_AGENTS, NUM_RED_POLICY_RANDOM_FIELDS, NUM_SUBNETS

_ACTION_NAME_TO_FSM = {
    "DiscoverRemoteSystems": 0,
    "AggressiveServiceDiscovery": 1,
    "StealthServiceDiscovery": 2,
    "DiscoverDeception": 3,
    "ExploitRemoteService": 4,
    "PrivilegeEscalate": 5,
    "Impact": 6,
    "DegradeServices": 7,
    "Withdraw": 8,
}


class _ChoiceRecordingRNG:
    def __init__(self, orig):
        self._orig = orig
        self.choice_log = []

    def choice(self, a, *args, **kwargs):
        result = self._orig.choice(a, *args, **kwargs)
        options = list(a) if hasattr(a, "__len__") else list(range(int(a)))
        p = kwargs.get("p", args[0] if args else None)
        if p is not None:
            p = [float(x) for x in p]
        self.choice_log.append((options, result, p))
        return result

    def __getattr__(self, name):
        return getattr(self._orig, name)


class RedPolicyRecorder:
    """Capture per-step red-agent choice outcomes as exact choice tokens encoded in [0, 1)."""

    def __init__(self):
        self._tape = np.full((MAX_STEPS, NUM_RED_AGENTS, NUM_RED_POLICY_RANDOM_FIELDS), 0.5, dtype=np.float32)

    def install(self, cyborg_env, mappings):
        controller = cyborg_env.environment_controller
        tape = self._tape
        step_ref = {"value": 0}

        orig_step = controller.step

        def _wrapped_step(actions=None, skip_valid_action_check=False):
            out = orig_step(actions, skip_valid_action_check)
            step_ref["value"] += 1
            return out

        controller.step = _wrapped_step

        for agent_name, interface in controller.agent_interfaces.items():
            if not agent_name.startswith("red_agent_"):
                continue
            agent_idx = int(agent_name.split("_")[-1])
            agent = interface.agent
            recorder = _ChoiceRecordingRNG(agent.np_random)
            agent.np_random = recorder
            orig_get_action = agent.get_action

            def _make_wrapped(orig_fn, ridx, rec):
                def _wrapped(agent_self, observation, action_space):
                    step_idx = int(step_ref["value"])
                    rec.choice_log.clear()
                    action = orig_fn(observation, action_space)
                    if step_idx < MAX_STEPS:
                        tape[step_idx, ridx] = _extract_step_uniforms(rec.choice_log, mappings)
                    rec.choice_log.clear()
                    return action

                return _wrapped

            agent.get_action = _make_wrapped(orig_get_action, agent_idx, recorder).__get__(agent, type(agent))

    def to_jax_array(self):
        import jax.numpy as jnp

        return jnp.asarray(self._tape)

    def extract_step(self, step_idx: int) -> np.ndarray:
        """Return the recorded choice-token row for a single step."""
        return np.array(self._tape[int(step_idx)], copy=True)


def _extract_step_uniforms(choice_log, mappings):
    slots = np.full(NUM_RED_POLICY_RANDOM_FIELDS, 0.5, dtype=np.float32)

    for options, result, probs in choice_log:
        if isinstance(result, np.ndarray) and result.shape == ():
            result = result.item()
        elif isinstance(result, np.generic):
            result = result.item()

        if isinstance(result, str) and result.startswith("red_agent_"):
            continue
        host_ip = None
        if isinstance(result, (str, np.str_)):
            try:
                host_ip = ip_address(str(result))
            except ValueError:
                host_ip = None
        elif isinstance(result, IPv4Address):
            host_ip = result

        if host_ip in mappings.ip_to_hostname:
            host_idx = mappings.hostname_to_idx[mappings.ip_to_hostname[host_ip]]
            slots[0] = _token_midpoint(host_idx, GLOBAL_MAX_HOSTS)
            continue
        if isinstance(result, type) and result.__name__ in _ACTION_NAME_TO_FSM:
            del options, probs
            slots[1] = _token_midpoint(_ACTION_NAME_TO_FSM[result.__name__], len(_ACTION_NAME_TO_FSM))
            continue
        if isinstance(result, IPv4Network):
            subnet_idx = mappings.cidr_to_subnet_idx[result]
            del options, probs
            slots[2] = _token_midpoint(subnet_idx, NUM_SUBNETS)
            continue

    return slots


def _token_midpoint(chosen_idx, total_count):
    return np.float32((int(chosen_idx) + 0.5) / int(total_count))
