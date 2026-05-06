"""Shared rollout result data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepSnapshot:
    reward: float
    cumulative_reward: float
    hosts_compromised_user: int
    hosts_compromised_priv: int
    red_sessions_total: int
    mission_phase: int


@dataclass
class EpisodeResult:
    actions_by_agent: list = field(default_factory=list)  # [agent_idx][step] = action_id
    rewards: list = field(default_factory=list)  # per-step rewards
    cumulative_reward: float = 0.0
    trajectory: list = field(default_factory=list)  # list[StepSnapshot]
    ria_total: float = 0.0  # sum of RIA (Red Impact) reward over episode
    lwf_total: float = 0.0  # sum of LWF (Local Work Fails) reward over episode
    asf_total: float = 0.0  # sum of ASF (Access Service Fails) reward over episode
    blue_busy_by_agent: list = field(default_factory=list)  # [agent_idx][step] = 0/1
    phase_per_step: list = field(default_factory=list)  # [step] = phase


@dataclass
class JaxRollout:
    actions: Any
    rewards: Any
    episodes: list[EpisodeResult]


@dataclass
class CyborgRollout:
    actions: Any
    rewards: Any
    actions_by_agent: list
    ria: Any
    lwf: Any
    asf: Any
    busy_by_agent: list
    phase_per_step: list


@dataclass
class TransferComparison:
    jax_actions: Any
    jax_rewards: Any
    jax_episodes: list[EpisodeResult]
    cyborg_actions: Any
    cyborg_rewards: Any
    cyborg_actions_by_agent: list
    cyborg_ria: Any | None = None
    cyborg_lwf: Any | None = None
    cyborg_asf: Any | None = None
    cyborg_busy_by_agent: list | None = None
    cyborg_phase_per_step: list | None = None
