"""Blue-perspective CC4 env with internal FSM red.

Wraps :class:`jaxborg.env.ScenarioEnv`, picks red actions each step via a
pluggable :class:`~jaxborg.scenarios.cc4.red_selectors.RedSelector`, and
exposes only blue agents to training.

The selector is decided at construction time; recipes pick a name and
:func:`make_red_selector` builds the callable. PR-style "subclass the env to
swap the red" is no longer required.

Per-episode metadata that selectors need (e.g. resilience role assignments)
is computed once at reset by an ``extras_factory(key, const) -> dict`` and
carried on :class:`FsmRedEnvState`. Today the only extras key is
``host_resilience_role``; new metrics extend the schema by adding an extras
factory and consuming the new key in the appropriate selector or scorer.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Box, Discrete

from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END
from jaxborg.actions.masking import compute_blue_action_mask
from jaxborg.constants import BLUE_OBS_SIZE, GLOBAL_MAX_HOSTS, NUM_BLUE_AGENTS, NUM_RED_AGENTS
from jaxborg.env import ScenarioEnv, ScenarioEnvState
from jaxborg.scenarios.cc4.red_fsm import (
    fsm_red_apply_delayed_update,
    fsm_red_schedule_post_step_update,
)
from jaxborg.scenarios.cc4.red_selectors import RedSelector, fsm_selector
from jaxborg.state import SimulatorConst

# An ExtrasFactory builds the per-episode extras dict at reset.
# Default = no extras. Resilience uses one to assign auth/db/web roles.
ExtrasFactory = Callable[[chex.PRNGKey, SimulatorConst], Dict[str, jax.Array]]


def _empty_extras_factory(key: chex.PRNGKey, const: SimulatorConst) -> Dict[str, jax.Array]:
    # Always carry host_resilience_role so the pytree shape is stable across
    # selector choices — selectors that don't care just see zeros.
    del key, const
    return {"host_resilience_role": jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32)}


@struct.dataclass
class FsmRedEnvState:
    """Flat extension of :class:`ScenarioEnvState` with per-episode extras.

    Field layout matches ``ScenarioEnvState`` (``state``, ``const``) plus
    ``extras``, so existing callers reading ``env_state.state`` /
    ``env_state.const`` keep working. ``extras`` is a stable-shape dict;
    today the only key is ``"host_resilience_role"``.
    """

    state: Any  # SimulatorState — typed loosely to avoid circular import
    const: Any  # SimulatorConst
    extras: Dict[str, jax.Array]


class FsmRedCC4Env(MultiAgentEnv):
    """Blue-only CC4 env with a pluggable internal FSM red.

    Args:
        num_steps:        Episode length.
        topology_mode:    Forwarded to :class:`ScenarioEnv` (``generative`` or
                          ``snapshot`` paired with ``topology_path``).
        training_mode:    Forwarded to :class:`ScenarioEnv`.
        topology_path:    Forwarded to :class:`ScenarioEnv`.
        red_selector:     Callable from :mod:`red_selectors`. Default is the
                          vanilla FSM selector (CybORG-parity).
        extras_factory:   ``(key, const) -> dict``. Default returns
                          ``{"host_resilience_role": zeros}`` so role-biased
                          selectors degrade gracefully when no factory is
                          supplied (no bias is applied).
        name:             Optional override for ``self.name``. Lets recipe-built
                          envs report a meaningful name in logs.
    """

    def __init__(
        self,
        num_steps: int = 500,
        *,
        topology_mode: str = "generative",
        training_mode: bool = False,
        topology_path: str | Path | Sequence[str | Path] | None = None,
        red_selector: RedSelector = fsm_selector,
        extras_factory: ExtrasFactory = _empty_extras_factory,
        name: Optional[str] = None,
    ):
        self._env = ScenarioEnv(
            num_steps=num_steps,
            topology_mode=topology_mode,
            training_mode=training_mode,
            topology_path=topology_path,
        )
        self._red_selector = red_selector
        self._extras_factory = extras_factory
        self._name = name or "FsmRedCC4"
        self.agents = list(self._env.blue_agents)

        super().__init__(num_agents=NUM_BLUE_AGENTS)

        for agent in self.agents:
            self.action_spaces[agent] = Discrete(BLUE_ALLOW_TRAFFIC_END)
            self.observation_spaces[agent] = Box(low=0.0, high=1.0, shape=(BLUE_OBS_SIZE,), dtype=jnp.float32)

    # ------------------------------------------------------------------
    # Reset

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], FsmRedEnvState]:
        key, key_extras = jax.random.split(key)
        obs, inner = self._env.reset(key)
        inner = self._strip_inactive_red_reset_knowledge(inner)
        extras = self._extras_factory(key_extras, inner.const)
        blue_obs = {a: obs[a] for a in self.agents}
        return blue_obs, FsmRedEnvState(state=inner.state, const=inner.const, extras=extras)

    def _strip_inactive_red_reset_knowledge(self, env_state: ScenarioEnvState) -> ScenarioEnvState:
        """Match native FiniteStateRedAgent reset knowledge.

        CybORG's controller action spaces may know additional hosts for inactive
        red agents at reset, but the FiniteStateRedAgent.host_states used by
        native action selection does not. Keep the richer reset knowledge in
        ScenarioEnv for translated-action differential replay; clear it here
        so the FSM env matches CybORG's FSM.
        """
        state = env_state.state
        inactive = ~state.red_agent_active
        state = state.replace(
            red_discovered_hosts=jnp.where(inactive[:, None], False, state.red_discovered_hosts),
            red_scanned_hosts=jnp.where(inactive[:, None], False, state.red_scanned_hosts),
            red_scanned_source_hosts=jnp.where(inactive[:, None, None], False, state.red_scanned_source_hosts),
            red_scan_source_pid=jnp.where(inactive[:, None], jnp.int32(-1), state.red_scan_source_pid),
            red_scan_anchor_host=jnp.where(inactive, jnp.int32(-1), state.red_scan_anchor_host),
            red_primary_is_abstract=jnp.where(inactive, True, state.red_primary_is_abstract),
            red_primary_pid=jnp.where(inactive, jnp.int32(-1), state.red_primary_pid),
            fsm_host_entered=jnp.where(inactive[:, None], False, state.fsm_host_entered),
        )
        return ScenarioEnvState(state=state, const=env_state.const)

    # ------------------------------------------------------------------
    # Step

    @partial(jax.jit, static_argnums=[0])
    def step(
        self,
        key: chex.PRNGKey,
        state: FsmRedEnvState,
        actions: Dict[str, chex.Array],
        reset_state: Optional[FsmRedEnvState] = None,
    ) -> Tuple[Dict[str, chex.Array], FsmRedEnvState, Dict[str, float], Dict[str, bool], Dict]:
        key, key_reset, key_extras = jax.random.split(key, 3)
        obs_st, states_st, rewards, dones, infos = self.step_env(key, state, actions)

        if reset_state is not None:
            inner_re_state = reset_state.state
            inner_re_const = reset_state.const
            extras_re = reset_state.extras
        else:
            scenario_st = ScenarioEnvState(state=states_st.state, const=states_st.const)
            inner_re = self._env._reset_state(scenario_st, key_reset)
            inner_re = self._strip_inactive_red_reset_knowledge(inner_re)
            inner_re_state = inner_re.state
            inner_re_const = inner_re.const
            extras_re = self._extras_factory(key_extras, inner_re_const)

        states_re = FsmRedEnvState(state=inner_re_state, const=inner_re_const, extras=extras_re)
        obs_re = self._get_blue_obs(states_re)

        states = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y),
            states_re,
            states_st,
        )
        obs = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y),
            obs_re,
            obs_st,
        )
        return obs, states, rewards, dones, infos

    @partial(jax.jit, static_argnums=[0])
    def step_env(
        self,
        key: chex.PRNGKey,
        env_state: FsmRedEnvState,
        blue_actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], FsmRedEnvState, Dict[str, float], Dict[str, bool], Dict]:
        host_resilience_role = env_state.extras["host_resilience_role"]

        key, key_red = jax.random.split(key)
        red_keys = jax.random.split(key_red, NUM_RED_AGENTS)

        state_before = fsm_red_apply_delayed_update(env_state.state)
        active_before = state_before.red_agent_active
        (
            red_action_arr,
            target_hosts_arr,
            target_subnets_arr,
            fsm_actions_arr,
            eligible_arr,
            sim_state,
        ) = self._red_selector(state_before, env_state.const, host_resilience_role, red_keys)

        red_actions = {f"red_{r}": red_action_arr[r] for r in range(NUM_RED_AGENTS)}
        target_hosts = [target_hosts_arr[r] for r in range(NUM_RED_AGENTS)]
        target_subnets = [target_subnets_arr[r] for r in range(NUM_RED_AGENTS)]
        fsm_actions = [fsm_actions_arr[r] for r in range(NUM_RED_AGENTS)]
        eligible_flags = [eligible_arr[r] for r in range(NUM_RED_AGENTS)]
        inner = ScenarioEnvState(state=sim_state, const=env_state.const)

        all_actions = {**blue_actions, **red_actions}
        obs, inner, rewards, dones, info = self._env.step_env(key, inner, all_actions)

        # Post-fix: CybORG's reassignment adds start host to aspace.ip_address,
        # but the FSM agent's host_states only includes hosts observed via
        # _process_new_observations. Strip start host discovery for agents
        # activated this step so the FSM-visible state matches CybORG's FSM.
        newly_active = ~active_before & inner.state.red_agent_active
        discovered = inner.state.red_discovered_hosts
        for r in range(NUM_RED_AGENTS):
            start_h = inner.const.red_start_hosts[r]
            has_session_at_start = inner.state.red_sessions[r, start_h]
            discovered = jnp.where(
                newly_active[r] & ~has_session_at_start,
                discovered.at[r, start_h].set(False),
                discovered,
            )
        inner = ScenarioEnvState(
            state=inner.state.replace(red_discovered_hosts=discovered),
            const=inner.const,
        )

        executed_flags = [inner.state.red_pending_ticks[r] == 0 for r in range(NUM_RED_AGENTS)]
        new_state = fsm_red_schedule_post_step_update(
            state_before,
            inner.state,
            inner.const,
            target_hosts,
            target_subnets,
            fsm_actions,
            eligible_flags,
            executed_flags,
        )
        out_state = FsmRedEnvState(state=new_state, const=inner.const, extras=env_state.extras)

        blue_obs = {a: obs[a] for a in self.agents}
        blue_rewards = {a: rewards[a] for a in self.agents}
        blue_dones = {a: dones[a] for a in self.agents}
        blue_dones["__all__"] = dones["__all__"]
        return blue_obs, out_state, blue_rewards, blue_dones, info

    # ------------------------------------------------------------------
    # Observers

    @partial(jax.jit, static_argnums=[0])
    def _get_blue_obs(self, env_state: FsmRedEnvState) -> Dict[str, chex.Array]:
        scenario = ScenarioEnvState(state=env_state.state, const=env_state.const)
        obs = self._env.get_obs(scenario)
        return {a: obs[a] for a in self.agents}

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, env_state: FsmRedEnvState) -> Dict[str, chex.Array]:
        return self._get_blue_obs(env_state)

    @partial(jax.jit, static_argnums=[0])
    def get_avail_actions(self, env_state: FsmRedEnvState) -> Dict[str, chex.Array]:
        return {
            f"blue_{i}": compute_blue_action_mask(env_state.const, i, env_state.state) for i in range(NUM_BLUE_AGENTS)
        }

    @property
    def name(self) -> str:
        return self._name

    @property
    def agent_classes(self) -> dict:
        return {"blue_agents": self.agents}


# ---------------------------------------------------------------------------
# Convenience constructor — recipe-friendly.


def make_fsm_red_env(
    num_steps: int = 500,
    *,
    topology_mode: str = "generative",
    training_mode: bool = False,
    topology_path: str | Path | Sequence[str | Path] | None = None,
    red_agent: str = "fsm",
    target_weight: float = 5.0,
    role_assignment: str | None = None,
    **selector_kwargs: Any,
) -> FsmRedCC4Env:
    """Build an env from recipe-style names.

    ``red_agent``: registry name (e.g. ``"fsm"``, ``"resilience"``, ``"cia_a"``).
    ``role_assignment``: extras-factory name (currently only ``"resilience"``
    is supplied; default ``None`` means no role assignment, and role-biased
    selectors run with no bias). Auto-defaults to ``"resilience"`` when
    ``red_agent`` requires roles.
    """
    from jaxborg.scenarios.cc4.red_selectors import make_red_selector

    selector = make_red_selector(red_agent, target_weight=target_weight, **selector_kwargs)

    needs_roles = red_agent not in ("fsm", "finite_state")
    factory_name = role_assignment if role_assignment is not None else ("resilience" if needs_roles else None)
    if factory_name == "resilience":
        from jaxborg.scenarios.cc4.resilience_topology import _assign_resilience_roles

        def extras_factory(key, const):
            del key
            return {"host_resilience_role": _assign_resilience_roles(const)}
    elif factory_name is None:
        extras_factory = _empty_extras_factory
    else:
        raise ValueError(f"Unknown role_assignment: {factory_name!r}")

    return FsmRedCC4Env(
        num_steps=num_steps,
        topology_mode=topology_mode,
        training_mode=training_mode,
        topology_path=topology_path,
        red_selector=selector,
        extras_factory=extras_factory,
        name=f"FsmRedCC4_{red_agent}",
    )
