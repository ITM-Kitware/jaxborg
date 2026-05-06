"""CybORG red agent variants for jaxborg experiments."""

from __future__ import annotations

import re

from CybORG.Agents.SimpleAgents.FiniteStateRedAgent import FiniteStateRedAgent

_OP_SERVER_RE = re.compile(r"^operational_zone_[ab]_subnet_server_host_\d+$")
_OP_SERVER_IDX_RE = re.compile(r"(\d+)$")


def _op_server_role_idx(hostname: str) -> int | None:
    """Return 0 (auth), 1 (db), or 2 (web) for an op-zone server, else None.

    CybORG has no per-episode role randomization, so we use a fixed convention:
    the trailing host index mod 3 maps to auth/db/web in the same order that
    JAX's resilience_topology uses as its baseline.
    """
    if not _OP_SERVER_RE.match(hostname):
        return None
    m = _OP_SERVER_IDX_RE.search(hostname)
    return int(m.group()) % 3 if m else 0


class ResilienceRedAgent(FiniteStateRedAgent):
    """FiniteStateRedAgent with biased host selection toward operational zone servers.

    Operational Zone A/B server hosts (the auth/db/web candidates) are
    _target_weight times more likely to be chosen than other hosts.
    This mirrors the JAX resilience_red_select_actions behaviour so CybORG
    training sees the same adversarial pressure as JAX training.

    Use ``ResilienceRedAgent.with_weight(w)`` to produce a subclass with a
    specific weight baked in — required because EnterpriseScenarioGenerator
    controls agent construction and only passes (name, np_random, agent_subnets).
    """

    _target_weight: float = 5.0

    @classmethod
    def with_weight(cls, target_weight: float) -> type:
        """Return a subclass with target_weight baked in as a class attribute."""
        return type(cls.__name__, (cls,), {"_target_weight": target_weight})

    def _choose_host(self, host_options: list) -> str:
        weights = []
        for ip in host_options:
            info = self.host_states.get(ip) or {}
            hostname = info.get("hostname") or ""
            weights.append(self._target_weight if _OP_SERVER_RE.match(hostname) else 1.0)
        total = sum(weights)
        probs = [w / total for w in weights]
        return self.np_random.choice(host_options, p=probs)


class _CIARedAgent(ResilienceRedAgent):
    """Base for CIA-targeted agents; subclasses declare which role indices to target.

    Role index convention (fixed, since CybORG has no per-episode randomization):
      0 → auth server  (mirrors JAX RESILIENCE_ROLE_AUTH)
      1 → db server    (mirrors JAX RESILIENCE_ROLE_DB)
      2 → web server   (mirrors JAX RESILIENCE_ROLE_WEB)

    Non-targeted op-zone servers keep weight 1.0; targeted ones use _target_weight.
    """

    _target_role_indices: frozenset[int] = frozenset({0, 1, 2})

    def _choose_host(self, host_options: list) -> str:
        weights = []
        for ip in host_options:
            info = self.host_states.get(ip) or {}
            hostname = info.get("hostname") or ""
            role_idx = _op_server_role_idx(hostname)
            if role_idx is not None and role_idx in self._target_role_indices:
                w = self._target_weight
            else:
                w = 1.0
            weights.append(w)
        total = sum(weights)
        probs = [w / total for w in weights]
        return self.np_random.choice(host_options, p=probs)


class CRedAgent(_CIARedAgent):
    """Targets auth + db servers (CIA: Confidentiality)."""

    _target_role_indices = frozenset({0, 1})


class IRedAgent(_CIARedAgent):
    """Targets auth + web servers (CIA: Integrity)."""

    _target_role_indices = frozenset({0, 2})


class ARedAgent(_CIARedAgent):
    """Targets auth + db + web servers (CIA: Availability)."""

    _target_role_indices = frozenset({0, 1, 2})
