"""NumPy-only CybORG adapter for simultaneous learned Blue/Red policies.

``BlueFlatWrapper`` remains the source of the historical 210/242 Blue
contract.  Red uses the backend-neutral compact policy contract and is
translated to native CybORG actions immediately before ``parallel_step``.

The adapter intentionally derives Red knowledge from the agent's own action
space and sessions.  It never reads target services, topology internals, or
another agent's sessions when constructing an observation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from jaxborg.actions.red_policy import (
    RED_POLICY_ACTION_DIM,
    RED_POLICY_AGGRESSIVE_SCAN_START,
    RED_POLICY_DEGRADE_START,
    RED_POLICY_DISCOVER_DECEPTION_START,
    RED_POLICY_DISCOVER_START,
    RED_POLICY_EXPLOIT_START,
    RED_POLICY_IMPACT_START,
    RED_POLICY_PRIVESC_START,
    RED_POLICY_SLEEP,
    RED_POLICY_STEALTH_SCAN_START,
    RED_POLICY_WITHDRAW_START,
    RedPolicyActionType,
    decode_red_policy_action,
)
from jaxborg.constants import (
    BLUE_OBS_SIZE,
    GLOBAL_MAX_HOSTS,
    NUM_BLUE_AGENTS,
    NUM_RED_AGENTS,
    NUM_SUBNETS,
    RED_OBS_SIZE,
)
from jaxborg.evaluation.cyborg_env_factory import make_cyborg_env, reset_cyborg_env
from jaxborg.parity.translate import CC4Mappings, build_mappings_from_cyborg
from jaxborg.scenarios.cc4.game_variant import GameVariant

BLUE_ACTION_DIM = 242
BLUE_AGENT_IDS = tuple(f"blue_agent_{i}" for i in range(NUM_BLUE_AGENTS))
RED_AGENT_IDS = tuple(f"red_agent_{i}" for i in range(NUM_RED_AGENTS))
POLICY_AGENT_IDS = BLUE_AGENT_IDS + RED_AGENT_IDS


@dataclass(frozen=True)
class TeamSpec:
    agent_ids: tuple[str, ...]
    obs_dim: int
    action_dim: int


TEAM_SPECS = {
    "blue": TeamSpec(BLUE_AGENT_IDS, BLUE_OBS_SIZE, BLUE_ACTION_DIM),
    "red": TeamSpec(RED_AGENT_IDS, RED_OBS_SIZE, RED_POLICY_ACTION_DIM),
}


def _success_is_true(value: Any) -> bool:
    name = getattr(value, "name", None)
    if name is not None:
        return str(name).upper() == "TRUE"
    return str(value).upper() in {"TRUE", "1"}


class CyborgJointAdapter:
    """Expose both policy teams over one raw CybORG ``parallel_step``.

    Observations and infos are keyed by the canonical ``blue_agent_*`` and
    ``red_agent_*`` names.  Rewards are the existing Blue game score for every
    Blue policy row and its exact negative for every Red policy row.
    """

    def __init__(self, variant: GameVariant, seed: int):
        from CybORG.Agents.Wrappers import BlueFlatWrapper

        self.variant = variant
        self._seed_rng = random.Random(seed)
        self.raw_env = make_cyborg_env(variant, self._seed_rng.randrange(2**31), wrapper_class=None)
        self.blue_wrapper = BlueFlatWrapper(self.raw_env, pad_spaces=True)
        self.mappings: CC4Mappings | None = None
        self._discovered: list[set[int]] = [set() for _ in RED_AGENT_IDS]
        self._scanned_by_primary: list[set[int]] = [set() for _ in RED_AGENT_IDS]
        self._primary_identity: list[tuple[str, int] | None] = [None for _ in RED_AGENT_IDS]

    def reset(self, *, ep_seed: int | None = None):
        """Reset one episode and return ``(observations, infos)``."""

        if ep_seed is None:
            ep_seed = self._seed_rng.randrange(2**31)
        reset = reset_cyborg_env(self.blue_wrapper, self.variant, ep_seed=ep_seed)
        self.mappings = build_mappings_from_cyborg(self.raw_env)
        self._discovered = [set() for _ in RED_AGENT_IDS]
        self._scanned_by_primary = [set() for _ in RED_AGENT_IDS]
        self._primary_identity = [None for _ in RED_AGENT_IDS]
        red_observations = {agent: self.raw_env.get_observation(agent) for agent in RED_AGENT_IDS}
        self._update_discovery_memory(red_observations)
        self._sync_primary_identities()
        return self._collect_observations(reset.obs), self._collect_infos(reset.info)

    def step(self, actions: dict[str, int]):
        """Submit all 11 policy actions from the same pre-step state."""

        missing = set(POLICY_AGENT_IDS) - set(actions)
        if missing:
            raise ValueError(f"joint action is missing policy agents: {sorted(missing)}")

        native: dict[str, Any] = {}
        for agent_name in BLUE_AGENT_IDS:
            action_idx = int(actions[agent_name])
            if not self._actor_available(agent_name):
                action_idx = self._blue_sleep_index(agent_name)
            choices = self.blue_wrapper.actions(agent_name)
            if not 0 <= action_idx < len(choices):
                raise ValueError(f"Blue action index out of range for {agent_name}: {action_idx}")
            native[agent_name] = choices[action_idx]

        for agent_name in RED_AGENT_IDS:
            action_idx = int(actions[agent_name])
            if not self._actor_available(agent_name):
                action_idx = RED_POLICY_SLEEP
            native[agent_name] = self.red_action_to_cyborg(agent_name, action_idx)

        raw_obs, raw_rewards, raw_dones, raw_info = self.raw_env.parallel_step(
            native,
            skip_valid_action_check=True,
        )
        self._update_scan_memory(raw_obs)
        self._update_discovery_memory(raw_obs)
        self._sync_primary_identities()

        blue_obs = {
            agent: self.blue_wrapper.observation_change(agent, raw_obs.get(agent, {})) for agent in BLUE_AGENT_IDS
        }
        observations = self._collect_observations(blue_obs)
        infos = self._collect_infos(raw_info)

        blue_score = float(sum(raw_rewards[BLUE_AGENT_IDS[0]].values()))
        rewards = {agent: blue_score for agent in BLUE_AGENT_IDS}
        rewards.update({agent: -blue_score for agent in RED_AGENT_IDS})
        terminated = {agent: bool(raw_dones.get(agent, False)) for agent in POLICY_AGENT_IDS}
        truncated = dict(terminated)
        return observations, rewards, terminated, truncated, infos

    def close(self) -> None:
        close = getattr(self.raw_env, "close", None)
        if close is not None:
            try:
                close()
            except AttributeError as exc:
                # CybORG's simulation-only constructor does not initialise the
                # emulator GUI flag that its generic close() path reads.
                if "_disable_gui" not in str(exc):
                    raise

    # ------------------------------------------------------------------
    # Red policy contract

    def red_observation(self, agent_idx: int) -> np.ndarray:
        if self.mappings is None:
            raise RuntimeError("reset() must be called before observing")
        agent_name = RED_AGENT_IDS[agent_idx]
        state = self.raw_env.environment_controller.state
        interface = self.raw_env.environment_controller.agent_interfaces[agent_name]
        active = bool(interface.active)
        busy = self._is_busy(agent_name)

        phase = np.zeros(3, dtype=np.float32)
        phase[min(max(int(state.mission_phase), 0), 2)] = 1.0
        time_value = min(float(self.raw_env.environment_controller.step_count) / max(self.variant.num_steps, 1), 1.0)

        identity = np.zeros(NUM_RED_AGENTS, dtype=np.float32)
        identity[agent_idx] = 1.0
        allowed = np.zeros(NUM_SUBNETS, dtype=np.float32)
        allowed_names = {str(name).lower() for name in interface.allowed_subnets}
        for subnet_idx, name in self.mappings.subnet_names.items():
            if name.lower() in allowed_names:
                allowed[subnet_idx] = 1.0

        discovered, scanned, own_sessions, privileged, primary = self._red_host_planes(agent_idx)
        if not active:
            # Discovery persists internally across deactivation, but the
            # policy-facing JAX/CybORG contract hides all host planes until
            # the agent can act again.
            for plane in (discovered, scanned, own_sessions, privileged, primary):
                plane.fill(0.0)
        obs = np.concatenate(
            (
                phase,
                np.asarray([time_value, float(active), float(busy)], dtype=np.float32),
                identity,
                allowed,
                discovered,
                scanned,
                own_sessions,
                privileged,
                primary,
            ),
            dtype=np.float32,
        )
        if obs.shape != (RED_OBS_SIZE,):  # pragma: no cover - import/layout invariant
            raise RuntimeError(f"unexpected learned-Red observation shape: {obs.shape}")
        return obs

    def red_action_mask(self, agent_idx: int) -> np.ndarray:
        if self.mappings is None:
            raise RuntimeError("reset() must be called before requesting masks")
        agent_name = RED_AGENT_IDS[agent_idx]
        mask = np.zeros(RED_POLICY_ACTION_DIM, dtype=np.float32)
        mask[RED_POLICY_SLEEP] = 1.0
        if not self._actor_available(agent_name):
            return mask

        interface = self.raw_env.environment_controller.agent_interfaces[agent_name]
        allowed_names = {str(name).lower() for name in interface.allowed_subnets}
        for subnet_idx, name in self.mappings.subnet_names.items():
            if name.lower() in allowed_names:
                mask[RED_POLICY_DISCOVER_START + subnet_idx] = 1.0

        discovered, scanned, own_sessions, privileged, _ = self._red_host_planes(agent_idx)
        for host_idx in range(GLOBAL_MAX_HOSTS):
            if discovered[host_idx]:
                mask[RED_POLICY_AGGRESSIVE_SCAN_START + host_idx] = 1.0
                mask[RED_POLICY_STEALTH_SCAN_START + host_idx] = 1.0
                mask[RED_POLICY_DISCOVER_DECEPTION_START + host_idx] = 1.0
            if scanned[host_idx]:
                mask[RED_POLICY_EXPLOIT_START + host_idx] = 1.0
            if own_sessions[host_idx]:
                mask[RED_POLICY_PRIVESC_START + host_idx] = 1.0
                mask[RED_POLICY_WITHDRAW_START + host_idx] = 1.0
            if privileged[host_idx]:
                mask[RED_POLICY_IMPACT_START + host_idx] = 1.0
                mask[RED_POLICY_DEGRADE_START + host_idx] = 1.0
        return mask

    def red_action_to_cyborg(self, agent_name: str, action_idx: int):
        """Translate compact Red actions, consistently binding session zero."""

        from CybORG.Simulator.Actions import (
            AggressiveServiceDiscovery,
            DegradeServices,
            DiscoverDeception,
            DiscoverRemoteSystems,
            ExploitRemoteService,
            Impact,
            PrivilegeEscalate,
            Sleep,
            StealthServiceDiscovery,
            Withdraw,
        )

        if self.mappings is None:
            raise RuntimeError("reset() must be called before translating actions")
        action_type, target = decode_red_policy_action(action_idx)
        if action_type is RedPolicyActionType.SLEEP:
            return Sleep()
        if action_type is RedPolicyActionType.DISCOVER:
            cidr = self.mappings.subnet_cidrs.get(target)
            if cidr is None:
                return Sleep()
            return DiscoverRemoteSystems(subnet=cidr, session=0, agent=agent_name)
        if target not in self.mappings.idx_to_hostname:
            return Sleep()

        hostname = self.mappings.idx_to_hostname[target]
        ip = self.mappings.hostname_to_ip[hostname]
        params = {"session": 0, "agent": agent_name}
        if action_type is RedPolicyActionType.AGGRESSIVE_SCAN:
            return AggressiveServiceDiscovery(ip_address=ip, **params)
        if action_type is RedPolicyActionType.STEALTH_SCAN:
            return StealthServiceDiscovery(ip_address=ip, **params)
        if action_type is RedPolicyActionType.DISCOVER_DECEPTION:
            return DiscoverDeception(ip_address=ip, **params)
        if action_type is RedPolicyActionType.EXPLOIT:
            # Keep the simulator's generic selector: target services and decoys
            # are deliberately absent from the learned policy observation.
            return ExploitRemoteService(ip_address=ip, **params)
        if action_type is RedPolicyActionType.PRIVESC:
            return PrivilegeEscalate(hostname=hostname, **params)
        if action_type is RedPolicyActionType.IMPACT:
            return Impact(hostname=hostname, **params)
        if action_type is RedPolicyActionType.DEGRADE:
            return DegradeServices(hostname=hostname, **params)
        if action_type is RedPolicyActionType.WITHDRAW:
            return Withdraw(ip_address=ip, hostname=hostname, **params)
        raise AssertionError(f"unhandled Red policy action type: {action_type}")

    # ------------------------------------------------------------------
    # State projection helpers

    def _collect_observations(self, blue_obs: dict[str, Any]) -> dict[str, np.ndarray]:
        observations = {agent: np.asarray(blue_obs[agent], dtype=np.float32) for agent in BLUE_AGENT_IDS}
        observations.update({agent: self.red_observation(i) for i, agent in enumerate(RED_AGENT_IDS)})
        return observations

    def _collect_infos(self, blue_info: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
        infos: dict[str, dict[str, Any]] = {}
        for agent in BLUE_AGENT_IDS:
            mask = np.asarray(self.blue_wrapper.action_mask(agent), dtype=np.float32)
            actor_active = self._actor_available(agent)
            if not actor_active:
                mask[:] = 0.0
                mask[self._blue_sleep_index(agent)] = 1.0
            infos[agent] = {
                "action_mask": mask,
                "actor_active": actor_active,
                "critic_active": True,
                "team": "blue",
            }
        for i, agent in enumerate(RED_AGENT_IDS):
            interface = self.raw_env.environment_controller.agent_interfaces[agent]
            infos[agent] = {
                "action_mask": self.red_action_mask(i),
                "actor_active": self._actor_available(agent),
                # Busy active agents remain useful value samples. Inactive
                # agents are excluded until their first actual decision.
                "critic_active": bool(interface.active),
                "team": "red",
            }
        return infos

    def _red_host_planes(self, agent_idx: int):
        assert self.mappings is not None
        state = self.raw_env.environment_controller.state
        agent_name = RED_AGENT_IDS[agent_idx]
        discovered = np.zeros(GLOBAL_MAX_HOSTS, dtype=np.float32)
        scanned = np.zeros(GLOBAL_MAX_HOSTS, dtype=np.float32)
        own_sessions = np.zeros(GLOBAL_MAX_HOSTS, dtype=np.float32)
        privileged = np.zeros(GLOBAL_MAX_HOSTS, dtype=np.float32)
        primary = np.zeros(GLOBAL_MAX_HOSTS, dtype=np.float32)

        for idx in self._discovered[agent_idx]:
            if idx < GLOBAL_MAX_HOSTS:
                discovered[idx] = 1.0
        for idx in self._scanned_by_primary[agent_idx]:
            if idx < GLOBAL_MAX_HOSTS:
                scanned[idx] = 1.0

        for session_id, session in state.sessions.get(agent_name, {}).items():
            idx = self.mappings.hostname_to_idx.get(session.hostname)
            if idx is None or idx >= GLOBAL_MAX_HOSTS:
                continue
            own_sessions[idx] = 1.0
            is_privileged = getattr(session, "has_privileged_access", lambda: False)()
            if is_privileged:
                privileged[idx] = 1.0
            if session_id == 0:
                primary[idx] = 1.0
        return discovered, scanned, own_sessions, privileged, primary

    def _update_scan_memory(self, observations: dict[str, dict]) -> None:
        assert self.mappings is not None
        scan_types = ("AggressiveServiceDiscovery", "StealthServiceDiscovery", "DiscoverNetworkServices")
        for agent_idx, agent_name in enumerate(RED_AGENT_IDS):
            obs = observations.get(agent_name, {})
            action = obs.get("action")
            if type(action).__name__ not in scan_types or not _success_is_true(obs.get("success")):
                continue
            ip = getattr(action, "ip_address", None)
            hostname = self.mappings.ip_to_hostname.get(ip)
            target = self.mappings.hostname_to_idx.get(hostname)
            if target is not None:
                self._scanned_by_primary[agent_idx].add(target)

    def _update_discovery_memory(self, observations: dict[str, dict]) -> None:
        """Accumulate hosts from each active agent's filtered observation.

        CybORG pre-seeds a configured start host in every Red action space,
        including inactive agents.  Reading the action space when an agent
        activates would expose that host even though ``FiniteStateRedAgent``
        never processed it.  Using the same filtered observations as the FSM
        keeps learned Red's discovery memory aligned with ``host_states``.
        """

        assert self.mappings is not None
        controller = self.raw_env.environment_controller
        for agent_idx, agent_name in enumerate(RED_AGENT_IDS):
            if not controller.agent_interfaces[agent_name].active:
                continue
            for host_id, host_details in observations.get(agent_name, {}).items():
                if host_id in {"success", "action", "message"} or not isinstance(host_details, dict):
                    continue

                hostname = host_id if host_id in self.mappings.hostname_to_idx else None
                if hostname is None:
                    hostname = self.mappings.ip_to_hostname.get(host_id)
                if hostname is None:
                    hostname = host_details.get("System info", {}).get("Hostname")
                if hostname is None:
                    for interface in host_details.get("Interface", ()):
                        hostname = self.mappings.ip_to_hostname.get(interface.get("ip_address"))
                        if hostname is not None:
                            break
                target = self.mappings.hostname_to_idx.get(hostname)
                if target is not None:
                    self._discovered[agent_idx].add(target)

    def _sync_primary_identities(self) -> None:
        state = self.raw_env.environment_controller.state
        for agent_idx, agent_name in enumerate(RED_AGENT_IDS):
            session = state.sessions.get(agent_name, {}).get(0)
            identity = None if session is None else (str(session.hostname), int(session.pid))
            previous = self._primary_identity[agent_idx]
            if previous is not None and previous != identity:
                self._scanned_by_primary[agent_idx].clear()
            self._primary_identity[agent_idx] = identity

    def _actor_available(self, agent_name: str) -> bool:
        interface = self.raw_env.environment_controller.agent_interfaces[agent_name]
        return bool(interface.active) and not self._is_busy(agent_name)

    def _is_busy(self, agent_name: str) -> bool:
        return self.raw_env.environment_controller.actions_in_progress.get(agent_name) is not None

    def _blue_sleep_index(self, agent_name: str) -> int:
        for idx, label in enumerate(self.blue_wrapper.action_labels(agent_name)):
            if label == "Sleep":
                return idx
        raise RuntimeError(f"BlueFlatWrapper has no Sleep action for {agent_name}")


__all__ = [
    "BLUE_ACTION_DIM",
    "BLUE_AGENT_IDS",
    "CyborgJointAdapter",
    "POLICY_AGENT_IDS",
    "RED_AGENT_IDS",
    "TEAM_SPECS",
    "TeamSpec",
]
