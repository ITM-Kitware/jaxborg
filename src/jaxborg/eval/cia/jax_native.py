"""JAX-native CIA scorer.

Mirror of :func:`jaxborg.eval.cia.cc4_cia_metric.score_episode` operating on
arrays captured during a JAX rollout, instead of the CybORG-trajectory JSONL
format.

We need this for axis-B (router-topology variation) eval: CybORG only has the
default ``_ROUTER_LINKS`` topology, so JAX-trained policies on varied
topologies must be scored within the JAX backend.

Event derivation from JAX state (per host index ``h``):

* C event (red ExploitRemoteService success): ``host_compromised[t, h]`` rises
  from 0 → ≥1.
* I event (red PrivilegeEscalate success): ``host_compromised[t, h]`` rises to
  the privileged level (= COMPROMISE_PRIVILEGED).
* A event (red Impact / DegradeServices success): ``red_impact_attempted[t, h]``
  is True (the underlying CC4 state already gates this on success).
* Blue Remove success: blue agent emitted action in ``[BLUE_REMOVE_START,
  BLUE_REMOVE_END)`` AND target host's compromise dropped (host_compromised
  fell from ≥1 → 0 between t-1 and t).
* Blue Restore success: blue agent emitted action in
  ``[BLUE_RESTORE_START, BLUE_RESTORE_END)`` AND target host's compromise
  cleared this step.

Host weight buckets follow the same regex rules as
``cc4_cia_metric.cc4_host_weight``, but we compute them directly from
topology arrays (host_subnet / host_is_router / host_is_server / host_is_user)
since we don't have CybORG hostnames at this layer.

The composite resilience score uses
``CIA_COMPOSITE_WEIGHT = {C:0.4, I:0.4, A:0.2}`` and
``RESTORATION_WEIGHT = 0.8`` to match :mod:`cc4_cia_metric`.
"""

from __future__ import annotations

import numpy as np

from jaxborg.actions.action_defs import (
    BLUE_ACTION_HOST_SLOTS,
    BLUE_REMOVE_END,
    BLUE_REMOVE_START,
    BLUE_RESTORE_END,
    BLUE_RESTORE_START,
)
from jaxborg.constants import (
    GLOBAL_MAX_HOSTS,
    SUBNET_IDS,
)
from jaxborg.eval.cia.cc4_cia_metric import (
    AVAIL_MULTIPLIER,
    CIA_COMPOSITE_WEIGHT,
    RESTORATION_WEIGHT,
    CIAEpisodeScore,
)

# Host-weight buckets — match cc4_host_weight regex rules but keyed by
# (subnet_id, role) tuples instead of hostname strings.
_OPERATIONAL_SUBS = {SUBNET_IDS["OPERATIONAL_ZONE_A"], SUBNET_IDS["OPERATIONAL_ZONE_B"]}
_RESTRICTED_SUBS = {SUBNET_IDS["RESTRICTED_ZONE_A"], SUBNET_IDS["RESTRICTED_ZONE_B"]}
_MIDTIER_SUBS = {
    SUBNET_IDS["PUBLIC_ACCESS_ZONE"],
    SUBNET_IDS["OFFICE_NETWORK"],
    SUBNET_IDS["ADMIN_NETWORK"],
}
_CONTRACTOR_SUB = SUBNET_IDS["CONTRACTOR_NETWORK"]
_INTERNET_SUB = SUBNET_IDS["INTERNET"]


def jax_host_weights(
    host_active: np.ndarray,
    host_subnet: np.ndarray,
    host_is_router: np.ndarray,
    host_is_server: np.ndarray,
    host_is_user: np.ndarray,
) -> np.ndarray:
    """Vectorized host-weight assignment matching ``cc4_host_weight`` rules.

    Returns ``(GLOBAL_MAX_HOSTS,)`` float weights. Inactive hosts get 0.0
    (excluded from the CIA sum and normalization).
    """
    w = np.full(GLOBAL_MAX_HOSTS, 0.1, dtype=np.float32)  # _FALLBACK_WEIGHT
    sub = host_subnet
    for h in range(GLOBAL_MAX_HOSTS):
        if not host_active[h]:
            w[h] = 0.0
            continue
        s = int(sub[h])
        if host_is_router[h]:
            w[h] = 0.3
        elif s in _OPERATIONAL_SUBS:
            w[h] = 0.5 if host_is_server[h] else 0.15
        elif s in _RESTRICTED_SUBS:
            w[h] = 0.4 if host_is_server[h] else 0.15
        elif s in _MIDTIER_SUBS:
            w[h] = 0.3 if host_is_server[h] else 0.1
        elif s == _CONTRACTOR_SUB:
            w[h] = 0.2 if host_is_server[h] else 0.1
        elif s == _INTERNET_SUB:
            w[h] = 0.05
    return w


def score_jax_episode(
    *,
    actions: np.ndarray,  # (T, NUM_BLUE_AGENTS) int — blue action ids
    host_compromised: np.ndarray,  # (T, GLOBAL_MAX_HOSTS) int — 0=None,1=User,2=Priv
    red_impact: np.ndarray,  # (T, GLOBAL_MAX_HOSTS) bool — Impact reached execution
    host_active: np.ndarray,  # (GLOBAL_MAX_HOSTS,) bool
    host_subnet: np.ndarray,  # (GLOBAL_MAX_HOSTS,) int
    host_is_router: np.ndarray,
    host_is_server: np.ndarray,
    host_is_user: np.ndarray,
    total_reward: float,
) -> CIAEpisodeScore:
    """Compute CIAEpisodeScore from per-step JAX state arrays.

    Replays the same per-host C/I/A/restoring tracking as
    :func:`score_episode`, but events come from state deltas + blue action
    decoding rather than CybORG action-class strings.
    """
    weights = jax_host_weights(
        host_active=host_active,
        host_subnet=host_subnet,
        host_is_router=host_is_router,
        host_is_server=host_is_server,
        host_is_user=host_is_user,
    )
    Z = float(weights.sum()) or 1.0
    n_hosts = GLOBAL_MAX_HOSTS

    C = np.ones(n_hosts, dtype=np.float32)
    I = np.ones(n_hosts, dtype=np.float32)  # noqa: E741
    A = np.ones(n_hosts, dtype=np.float32)
    restoring = np.zeros(n_hosts, dtype=np.float32)

    cs: list[float] = []
    is_: list[float] = []
    as_: list[float] = []
    rs: list[float] = []
    red_counts: dict[str, int] = {}
    blue_counts: dict[str, int] = {}

    T = host_compromised.shape[0]
    prev_compromised = np.zeros(n_hosts, dtype=np.int32)

    for t in range(T):
        # Restore window: apply pending restoration before this step's events
        in_restore = restoring >= 1.0
        restoring = np.where(in_restore, 0.0, restoring)
        A = np.where(in_restore, 1.0, A)

        cur = host_compromised[t]

        # Red events (state-derived, treated as success):
        # C: 0 → ≥1 transition (host newly compromised at user level)
        c_evt = (prev_compromised == 0) & (cur >= 1) & host_active
        # I: any → ≥2 transition (privesc success)
        i_evt = (prev_compromised < 2) & (cur >= 2) & host_active
        # A: red_impact_attempted true this step
        a_evt = red_impact[t].astype(bool) & host_active

        red_counts["ExploitRemoteService"] = red_counts.get("ExploitRemoteService", 0) + int(c_evt.sum())
        red_counts["PrivilegeEscalate"] = red_counts.get("PrivilegeEscalate", 0) + int(i_evt.sum())
        red_counts["Impact"] = red_counts.get("Impact", 0) + int(a_evt.sum())

        C = np.where(c_evt, -1.0, C)
        I = np.where(i_evt, -1.0, I)
        # Availability: chain multiplier on consecutive impacts
        A_after_impact = np.where(A < 0, A * AVAIL_MULTIPLIER, -1.0)
        A = np.where(a_evt, A_after_impact, A)

        # Blue events: decode action ids per blue agent.
        # Remove: action in [BLUE_REMOVE_START, BLUE_REMOVE_END), target host index =
        # action - BLUE_REMOVE_START. A successful Remove restores C and I if compromised.
        # Restore: action in [BLUE_RESTORE_START, BLUE_RESTORE_END), target = action -
        # BLUE_RESTORE_START. Restore wipes everything to clean and starts the
        # 1-step restoration window for A.
        for ag in range(actions.shape[1]):
            a_id = int(actions[t, ag])
            if BLUE_REMOVE_START <= a_id < BLUE_REMOVE_END:
                slot = a_id - BLUE_REMOVE_START
                if 0 <= slot < BLUE_ACTION_HOST_SLOTS:
                    h = slot
                    if h < n_hosts and host_active[h]:
                        # Treat as success only if the host was compromised
                        # (mirrors score_episode's "if host not in network" gate).
                        if C[h] < 0 or I[h] < 0:
                            blue_counts["Remove"] = blue_counts.get("Remove", 0) + 1
                            if C[h] < 0:
                                C[h] = 1.0
                            if I[h] < 0:
                                I[h] = 1.0
            elif BLUE_RESTORE_START <= a_id < BLUE_RESTORE_END:
                slot = a_id - BLUE_RESTORE_START
                if 0 <= slot < BLUE_ACTION_HOST_SLOTS:
                    h = slot
                    if h < n_hosts and host_active[h]:
                        blue_counts["Restore"] = blue_counts.get("Restore", 0) + 1
                        A[h] = 0.0
                        restoring[h] = 1.0
                        C[h] = 1.0
                        I[h] = RESTORATION_WEIGHT

        c = float(np.dot(C, weights) / Z)
        i = float(np.dot(I, weights) / Z)
        a = float(np.dot(A, weights) / Z)
        r = CIA_COMPOSITE_WEIGHT["C"] * c + CIA_COMPOSITE_WEIGHT["I"] * i + CIA_COMPOSITE_WEIGHT["A"] * a
        cs.append(c)
        is_.append(i)
        as_.append(a)
        rs.append(r)

        prev_compromised = cur

    n = len(rs) or 1
    return CIAEpisodeScore(
        steps=len(rs),
        total_reward=float(total_reward),
        C_mean=sum(cs) / n,
        I_mean=sum(is_) / n,
        A_mean=sum(as_) / n,
        R_mean=sum(rs) / n,
        C_min=min(cs) if cs else 0.0,
        I_min=min(is_) if is_ else 0.0,
        A_min=min(as_) if as_ else 0.0,
        R_min=min(rs) if rs else 0.0,
        red_event_counts=red_counts,
        blue_event_counts=blue_counts,
    )
