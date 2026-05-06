"""Resilience CIA metric computed from CC4 trajectory files.

Scores trajectories based on red Impact/DegradeServices attacks on three
resilience-critical servers identified by ``host_role_map``:

  ROLE_AUTH = 1  (authentication server)
  ROLE_DB   = 2  (database server)
  ROLE_WEB  = 3  (frontend web server)

CIA rules (per step):
  C: decreases by CIA_DROP_WEIGHT["C"] when AUTH or DB is currently impacted
  I: decreases by CIA_DROP_WEIGHT["I"] when AUTH or WEB is currently impacted
  A: decreases by CIA_DROP_WEIGHT["A"] when any of AUTH, DB, WEB is impacted

Resilience: weighted sum  R = w_C*C + w_I*I + w_A*A
            weights taken from CIA_COMPOSITE_WEIGHT (class-level, override to tune).

All weights are configurable either at the class level (subclass) or per-instance
by passing ``composite_weight`` / ``drop_weight`` to the constructor.

Usage::

    metric = ResilienceMetric(host_role_map={"op_zone_a_server_0": ROLE_AUTH,
                                              "op_zone_a_server_1": ROLE_DB,
                                              "op_zone_b_server_2": ROLE_WEB})
    result = metric.score_trajectory_file(Path("episode.jsonl"))
    results = metric.score_trajectory_dir(Path("trajs/"))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jaxborg.scenarios.cc4.topology_roles import ROLE_AUTH, ROLE_DB, ROLE_WEB

_C_ROLES = frozenset({ROLE_AUTH, ROLE_DB})
_I_ROLES = frozenset({ROLE_AUTH, ROLE_WEB})
_A_ROLES = frozenset({ROLE_AUTH, ROLE_DB, ROLE_WEB})

_RED_IMPACT_EVENTS = frozenset({"Impact", "DegradeServices"})
_BLUE_RESTORE_EVENT = frozenset({"Restore"})


@dataclass
class ResilienceEpisodeScore:
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
    impact_counts: dict = field(default_factory=dict)


class ResilienceMetric:
    """CIA resilience metric focused on auth/db/web server availability.

    Args:
        host_role_map:    Mapping of hostname → role int (ROLE_AUTH/DB/WEB).
                          Hosts not in this map are ignored.
        composite_weight: Override CIA_COMPOSITE_WEIGHT per instance.
        drop_weight:      Override CIA_DROP_WEIGHT per instance.
    """

    # Class-level defaults — override in subclasses or via constructor args.
    CIA_COMPOSITE_WEIGHT: dict[str, float] = {"C": 1 / 3, "I": 1 / 3, "A": 1 / 3}
    CIA_DROP_WEIGHT: dict[str, float] = {"C": 10.0, "I": 10.0, "A": 10.0}

    def __init__(
        self,
        host_role_map: dict[str, int] | None = None,
        composite_weight: dict[str, float] | None = None,
        drop_weight: dict[str, float] | None = None,
    ) -> None:
        # If None, score_episode reads the map from the trajectory header's
        # "host_resilience_roles" field (written by cc4_trajectory_eval.py).
        self.host_role_map = host_role_map
        self._composite = composite_weight if composite_weight is not None else self.CIA_COMPOSITE_WEIGHT
        self._drop = drop_weight if drop_weight is not None else self.CIA_DROP_WEIGHT

    def score_episode(
        self,
        header: dict[str, Any],
        steps: list[dict[str, Any]],
        total_reward: float,
    ) -> ResilienceEpisodeScore:
        """Replay one trajectory and compute per-episode CIA resilience scores."""
        role_map = self.host_role_map
        if role_map is None:
            role_map = {h: int(r) for h, r in header.get("host_resilience_roles", {}).items()}
        if not role_map:
            raise ValueError(
                "host_role_map is empty: pass it to the constructor or include "
                "'host_resilience_roles' in the trajectory header."
            )

        # Per-host impact state: True = currently impacted by red.
        impacted: dict[str, bool] = {h: False for h in role_map}
        impact_counts: dict[str, int] = {}

        drop_C = self._drop["C"]
        drop_I = self._drop["I"]
        drop_A = self._drop["A"]
        w_C = self._composite["C"]
        w_I = self._composite["I"]
        w_A = self._composite["A"]

        cs, is_, as_, rs = [], [], [], []

        for step in steps:
            # --- Red events: Impact/Degrade marks host as impacted ---
            for rec in step["red"].values():
                if rec.get("success") != "TRUE":
                    continue
                cls = rec.get("cls", "")
                host = rec.get("host")
                if cls in _RED_IMPACT_EVENTS and host in impacted:
                    impacted[host] = True
                    impact_counts[host] = impact_counts.get(host, 0) + 1

            # --- Blue events: Restore clears impact state ---
            for rec in step["blue"].values():
                if rec.get("success") != "TRUE":
                    continue
                if rec.get("cls") in _BLUE_RESTORE_EVENT:
                    host = rec.get("host")
                    if host in impacted:
                        impacted[host] = False

            # --- Per-step CIA scores ---
            C = I = A = 0.0  # noqa: E741 — CIA triad domain notation
            for host, role in role_map.items():
                if impacted[host]:
                    if role in _C_ROLES:
                        C -= drop_C
                    if role in _I_ROLES:
                        I -= drop_I  # noqa: E741
                    if role in _A_ROLES:
                        A -= drop_A

            R = w_C * C + w_I * I + w_A * A
            cs.append(C)
            is_.append(I)
            as_.append(A)
            rs.append(R)

        n = len(rs) or 1
        return ResilienceEpisodeScore(
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
            impact_counts=impact_counts,
        )

    def score_trajectory_file(self, path: Path) -> ResilienceEpisodeScore:
        """Parse a ``.jsonl`` trajectory file and call ``score_episode``."""
        header = None
        steps: list[dict] = []
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
            raise ValueError(f"no header record in {path}")
        return self.score_episode(header, steps, total_reward)

    def score_trajectory_dir(self, traj_dir: Path, glob: str = "*.jsonl") -> list[ResilienceEpisodeScore]:
        """Score all trajectory files in a directory."""
        files = sorted(Path(traj_dir).glob(glob))
        if not files:
            raise FileNotFoundError(f"no files matching {glob!r} in {traj_dir}")
        return [self.score_trajectory_file(f) for f in files]
