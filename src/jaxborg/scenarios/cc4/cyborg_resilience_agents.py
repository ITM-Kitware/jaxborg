"""CybORG red agent variants for jaxborg experiments.

Per-episode role assignment matches the JAX side: at each new episode, three
operational-zone server hostnames are randomly chosen as auth/db/web roles
and rotate across the op-zone server set rather than pinning to the same
3 hostnames.

The role map is **global to the episode**: every red agent in one CybORG
episode must use the same auth/db/web hosts (otherwise different reds
would bias toward different "auth" hosts and the resilience metric — which
scores impact actions against a single canonical role assignment — becomes
incoherent). Coordination uses an external injection pattern:

* The caller builds the role map once per episode via :func:`inject_role_map`
  (deterministic from ``ep_seed`` + the env's full host list).
* :func:`inject_role_map` walks the env's agent_interfaces and pushes the
  map into every :class:`ResilienceRedAgent` (or subclass) via
  :meth:`set_role_map`.
* The trajectory recorder writes the same map to its header so the
  resilience scorer scores exactly the hosts the red biased toward.

Both training (`scripts/train/algorithms/ippo_cyborg.py`) and eval
(`scripts/eval/cc4_trajectory_eval.py`, `cyborg_runner`, `jax_runner`)
call ``inject_role_map`` after every ``env.reset()`` when the red is one
of resilience / c / i / a (and aliases). If ``inject_role_map`` is *not*
called, agents fall back to a private lazy-init from ``self.np_random``
that produces a valid (but agent-local) map — useful for tests and ad-hoc
rollouts where global agreement isn't required.
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
        # Lazy: populated on first _choose_host call once ≥3 op-zone servers
        # are visible in host_states. Stored as {hostname -> role_int}.
        self._role_map: dict[str, int] | None = None

    @classmethod
    def with_weight(cls, target_weight: float) -> type:
        """Return a subclass with target_weight baked in as a class attribute."""
        return type(cls.__name__, (cls,), {"_target_weight": target_weight})

    def set_role_map(self, role_map: dict[str, int]) -> None:
        """Pre-set the episode's role map (used by eval to coordinate with the recorder)."""
        self._role_map = dict(role_map)

    def _ensure_role_map(self) -> None:
        """Build per-episode role map once ≥3 op-zone servers are discovered.

        Shuffles candidates with self.np_random for per-episode variation.
        """
        if self._role_map is not None:
            return
        op_servers = sorted(
            {
                info["hostname"]
                for info in self.host_states.values()
                if info.get("hostname") and OPERATIONAL_SERVER_RE.match(info["hostname"])
            }
        )
        if len(op_servers) < 3:
            return
        order = list(self.np_random.permutation(len(op_servers)))
        roles = (ROLE_AUTH, ROLE_DB, ROLE_WEB)
        self._role_map = {op_servers[order[i]]: roles[i] for i in range(3)}

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
    """

    _target_roles: frozenset[int] = frozenset({ROLE_AUTH, ROLE_DB, ROLE_WEB})

    def _choose_host(self, host_options: list) -> str:
        self._ensure_role_map()
        weights = []
        for ip in host_options:
            info = self.host_states.get(ip) or {}
            hostname = info.get("hostname") or ""
            role = (self._role_map or {}).get(hostname)
            if role is not None and role in self._target_roles:
                w = self._target_weight
            else:
                w = 1.0
            weights.append(w)
        total = sum(weights)
        probs = [w / total for w in weights]
        return self.np_random.choice(host_options, p=probs)


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

    Computes the role map from all hostnames the env currently has, using
    ``random.Random(ep_seed)`` so the same seed reproduces the same map.
    Pushes the map into every ``ResilienceRedAgent`` (or subclass) wired
    into the env via the agent_interfaces table. Returns the map so callers
    can write it to a trajectory header.

    Use after ``env = make_env(seed)`` and before the first ``env.reset()``
    or rollout step.
    """
    ec = env.unwrapped.environment_controller
    hostnames = list(ec.state.hosts.keys())
    rng = random.Random(ep_seed)
    role_map = assign_resilience_roles(hostnames, rng=rng)
    for ai in ec.agent_interfaces.values():
        agent = getattr(ai, "agent", None)
        if isinstance(agent, ResilienceRedAgent):
            agent.set_role_map(role_map)
    return role_map
