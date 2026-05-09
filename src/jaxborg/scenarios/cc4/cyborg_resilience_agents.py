"""CybORG red agent variants for jaxborg experiments.

Per-episode role assignment matches the JAX side: each episode randomly
picks 3 of the operational-zone server hostnames as auth/db/web. The
assignment is **global to the episode** — every red agent in one CybORG
episode shares the same map so the resilience metric (which scores
against a single canonical role map) stays coherent.

Coordination is by external injection: callers build the map once per
episode via :func:`inject_role_map` and the helper pushes it into every
:class:`ResilienceRedAgent` in the env. All training and eval call sites
do this after every ``env.reset()``; ``set_role_map`` must be called
before the agent picks its first target.
"""

from __future__ import annotations

import random

from CybORG.Agents.SimpleAgents.FiniteStateRedAgent import FiniteStateRedAgent

from jaxborg.scenarios.cc4.topology_roles import (
    OPERATIONAL_SERVER_RE,
    ROLE_AUTH,
    ROLE_DB,
    ROLE_WEB,
    assign_resilience_roles,
)


class ResilienceRedAgent(FiniteStateRedAgent):
    """FiniteStateRedAgent with biased host selection toward operational zone servers.

    Operational Zone A/B server hosts are ``_target_weight`` times more likely
    to be chosen than other hosts. Mirrors the JAX 'resilience' role-biased
    selector behaviour so CybORG training sees the same adversarial pressure
    as JAX training.

    Use ``ResilienceRedAgent.with_weight(w)`` to produce a subclass with a
    specific weight baked in — required because EnterpriseScenarioGenerator
    controls agent construction and only passes (name, np_random, agent_subnets).
    """

    _target_weight: float = 5.0

    def __init__(self, name=None, np_random=None, agent_subnets=None):
        super().__init__(name=name, np_random=np_random, agent_subnets=agent_subnets)
        self._role_map: dict[str, int] = {}

    @classmethod
    def with_weight(cls, target_weight: float) -> type:
        """Return a subclass with target_weight baked in as a class attribute."""
        return type(cls.__name__, (cls,), {"_target_weight": target_weight})

    def set_role_map(self, role_map: dict[str, int]) -> None:
        """Set the episode's role map. Called by inject_role_map after env.reset()."""
        self._role_map = dict(role_map)

    def _choose_host(self, host_options: list) -> str:
        weights = []
        for ip in host_options:
            info = self.host_states.get(ip) or {}
            hostname = info.get("hostname") or ""
            weights.append(self._target_weight if OPERATIONAL_SERVER_RE.match(hostname) else 1.0)
        total = sum(weights)
        probs = [w / total for w in weights]
        return self.np_random.choice(host_options, p=probs)


class _CIARedAgent(ResilienceRedAgent):
    """Base for CIA-targeted agents; subclasses declare which roles to target.

    ``_target_roles`` selects which of {ROLE_AUTH, ROLE_DB, ROLE_WEB} the bias
    points at. Hosts whose role is in that set get ``_target_weight``; all
    others (including untagged op-zone servers) keep weight 1.0.

    Action selection at the root-access state ``R`` is also biased: the stock
    ``FiniteStateRedAgent`` matrix splits ``R`` between Discover (0.50),
    Impact (0.25), and Degrade (0.25); CIA agents shift mass toward Impact
    (0.45) and Degrade (0.45) with only 0.10 left on Discover. Mirrors the
    JAX ``_CIA_PROB_MATRIX``.
    """

    _target_roles: frozenset[int] = frozenset({ROLE_AUTH, ROLE_DB, ROLE_WEB})

    def _choose_host(self, host_options: list) -> str:
        weights = []
        for ip in host_options:
            info = self.host_states.get(ip) or {}
            hostname = info.get("hostname") or ""
            role = self._role_map.get(hostname)
            w = self._target_weight if role in self._target_roles else 1.0
            weights.append(w)
        total = sum(weights)
        probs = [w / total for w in weights]
        return self.np_random.choice(host_options, p=probs)

    def state_transitions_probability(self):
        m = super().state_transitions_probability()
        m["R"] = [0.10, None, None, None, None, None, 0.45, 0.45, 0.0]
        return m


class CRedAgent(_CIARedAgent):
    """Targets auth + db servers (CIA: Confidentiality)."""

    _target_roles = frozenset({ROLE_AUTH, ROLE_DB})


class IRedAgent(_CIARedAgent):
    """Targets auth + web servers (CIA: Integrity)."""

    _target_roles = frozenset({ROLE_AUTH, ROLE_WEB})


class ARedAgent(_CIARedAgent):
    """Targets auth + db + web servers (CIA: Availability)."""

    _target_roles = frozenset({ROLE_AUTH, ROLE_DB, ROLE_WEB})


def inject_role_map(env, ep_seed: int) -> dict[str, int]:
    """Build a per-episode role map from the env's full host list and inject it.

    Computes the role map deterministically from ``ep_seed`` + the env's
    full host list, then pushes it into every ``ResilienceRedAgent`` (or
    subclass) in the env. Returns the map so callers can write it to a
    trajectory header.

    Call after ``env = make_env(seed)`` (and after each ``env.reset()`` if
    reusing the env across episodes) and before the first rollout step.
    """
    ec = env.unwrapped.environment_controller
    hostnames = list(ec.state.hosts.keys())
    rng = random.Random(ep_seed)
    role_map = assign_resilience_roles(hostnames, rng)
    for ai in ec.agent_interfaces.values():
        agent = getattr(ai, "agent", None)
        if isinstance(agent, ResilienceRedAgent):
            agent.set_role_map(role_map)
    return role_map
