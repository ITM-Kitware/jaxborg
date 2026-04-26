"""CC4-aware CIA + resilience scorer.

Scores trajectories produced by `scripts/eval/cc4_trajectory_eval.py`. Decoupled
from rollout so we can re-score with new metrics without re-running CybORG.

Event mapping (extends CIATriadMetric to CC4's richer action set):
  C: red ExploitRemoteService success → -1; blue Remove → +1
  I: red PrivilegeEscalate success → -1; blue Remove → +1; blue Restore → +RESTORATION_WEIGHT
  A: red Impact / DegradeServices success → -1 (×AVAIL_MULTIPLIER on consecutive impacts);
     blue Restore → +1 next step

Composite: 0.4*C + 0.4*I + 0.2*A. Under-weighting A is intentional —
CC4's training reward (LWF/ASF/Impact) is availability-flavored; under-weighting
A here keeps the resilience scalar structurally decoupled from reward.

Host weights are CC4 zone-based (security criticality, distinct from
BlueRewardMachine.phase_rewards). See `cc4_host_weight`.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

CIA_COMPOSITE_WEIGHT = {"C": 0.4, "I": 0.4, "A": 0.2}
RESTORATION_WEIGHT = 0.8
AVAIL_MULTIPLIER = 1.0

RED_C_EVENTS = frozenset({"ExploitRemoteService"})
RED_I_EVENTS = frozenset({"PrivilegeEscalate"})
RED_A_EVENTS = frozenset({"Impact", "DegradeServices"})
BLUE_REMOVE = frozenset({"Remove"})
BLUE_RESTORE = frozenset({"Restore"})

_HOST_WEIGHT_RULES = (
    (re.compile(r"^operational_zone_[ab]_subnet_server_host_\d+$"), 0.5),
    (re.compile(r"^restricted_zone_[ab]_subnet_server_host_\d+$"), 0.4),
    (re.compile(r"^public_access_zone_subnet_server_host_\d+$"), 0.3),
    (re.compile(r"^office_network_subnet_server_host_\d+$"), 0.3),
    (re.compile(r"^admin_network_subnet_server_host_\d+$"), 0.3),
    (re.compile(r"^contractor_network_subnet_server_host_\d+$"), 0.2),
    (re.compile(r"^.+_subnet_router$"), 0.3),
    (re.compile(r"^operational_zone_[ab]_subnet_user_host_\d+$"), 0.15),
    (re.compile(r"^restricted_zone_[ab]_subnet_user_host_\d+$"), 0.15),
    (re.compile(r"^public_access_zone_subnet_user_host_\d+$"), 0.1),
    (re.compile(r"^office_network_subnet_user_host_\d+$"), 0.1),
    (re.compile(r"^admin_network_subnet_user_host_\d+$"), 0.1),
    (re.compile(r"^contractor_network_subnet_user_host_\d+$"), 0.1),
    (re.compile(r"^root_internet_host_\d+$"), 0.05),
)
_FALLBACK_WEIGHT = 0.1


def cc4_host_weight(hostname: str) -> float:
    for pat, w in _HOST_WEIGHT_RULES:
        if pat.match(hostname):
            return w
    return _FALLBACK_WEIGHT


@dataclass
class _HostState:
    weight: float
    C: float = 1.0
    I: float = 1.0  # noqa: E741 — canonical C/I/A triad notation
    A: float = 1.0
    restoring: float = 0.0


@dataclass
class CIAEpisodeScore:
    steps: int
    total_reward: float
    C_mean: float
    I_mean: float
    A_mean: float
    R_mean: float
    C_min: float
    I_min: float
    A_min: float
    R_min: float
    red_event_counts: dict = field(default_factory=dict)
    blue_event_counts: dict = field(default_factory=dict)


def score_episode(header: dict, steps: list[dict], total_reward: float) -> CIAEpisodeScore:
    """Replay a recorded trajectory and produce per-episode CIA + resilience.

    `header`: must include `hosts` (list of hostnames present in the network).
    `steps`: list of step records — each has `red` and `blue` agent dicts of
             {cls, host, ip, success}, plus `t` and `reward`.
    """
    network = {h: _HostState(weight=cc4_host_weight(h)) for h in header["hosts"]}
    z = sum(h.weight for h in network.values()) or 1.0
    cs, is_, as_, rs = [], [], [], []
    red_counts: dict[str, int] = {}
    blue_counts: dict[str, int] = {}

    for step in steps:
        for h in network.values():
            if h.restoring >= 1.0:
                h.restoring = 0.0
                h.A = 1.0

        for ag, rec in step["red"].items():
            if rec.get("success") != "TRUE":
                continue
            cls = rec.get("cls")
            host = rec.get("host")
            red_counts[cls] = red_counts.get(cls, 0) + 1
            if host not in network:
                continue
            h = network[host]
            if cls in RED_C_EVENTS:
                h.C = -1.0
            elif cls in RED_I_EVENTS:
                h.I = -1.0
            elif cls in RED_A_EVENTS:
                if h.A < 0:
                    h.A *= AVAIL_MULTIPLIER
                else:
                    h.A = -1.0

        for ag, rec in step["blue"].items():
            if rec.get("success") != "TRUE":
                continue
            cls = rec.get("cls")
            host = rec.get("host")
            blue_counts[cls] = blue_counts.get(cls, 0) + 1
            if host not in network:
                continue
            h = network[host]
            if cls in BLUE_REMOVE:
                if h.C < 0:
                    h.C = 1.0
                if h.I < 0:
                    h.I = 1.0
            elif cls in BLUE_RESTORE:
                h.A = 0.0
                h.restoring = 1.0
                h.C = 1.0
                h.I = RESTORATION_WEIGHT

        c = sum(h.C * h.weight for h in network.values()) / z
        i = sum(h.I * h.weight for h in network.values()) / z
        a = sum(h.A * h.weight for h in network.values()) / z
        r = CIA_COMPOSITE_WEIGHT["C"] * c + CIA_COMPOSITE_WEIGHT["I"] * i + CIA_COMPOSITE_WEIGHT["A"] * a
        cs.append(c)
        is_.append(i)
        as_.append(a)
        rs.append(r)

    n = len(rs) or 1
    return CIAEpisodeScore(
        steps=len(rs),
        total_reward=total_reward,
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


def score_trajectory_file(path: Path) -> CIAEpisodeScore:
    """Score a single .jsonl trajectory file (header + step records + footer)."""
    header = None
    steps = []
    total_reward = 0.0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        t = rec.get("type")
        if t == "header":
            header = rec
        elif t == "step":
            steps.append(rec)
        elif t == "footer":
            total_reward = rec.get("total_reward", 0.0)
    if header is None:
        raise ValueError(f"no header in {path}")
    return score_episode(header, steps, total_reward)
