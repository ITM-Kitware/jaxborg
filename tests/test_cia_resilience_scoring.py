from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.constants import GLOBAL_MAX_HOSTS
from jaxborg.evaluation.cia import (
    ROLE_AUTH,
    ROLE_DB,
    ROLE_WEB,
    ResilienceMetric,
    mean_resilience_episode,
    resilience_episode_records,
    score_resilience_state,
    summarize_resilience_episodes,
)
from jaxborg.state import create_initial_state


def _role_array() -> jax.Array:
    return jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32).at[0].set(ROLE_AUTH).at[1].set(ROLE_DB).at[2].set(ROLE_WEB)


@pytest.mark.parametrize(
    ("state_update", "expected"),
    [
        ("auth_ot", [-10.0, -10.0, -10.0]),
        ("db_service", [-10.0, 0.0, -10.0]),
        ("web_decoy", [0.0, -10.0, -10.0]),
    ],
)
def test_jax_state_scorer_handles_each_resilience_role(state_update: str, expected: list[float]) -> None:
    state = create_initial_state()
    if state_update == "auth_ot":
        state = state.replace(ot_service_stopped=state.ot_service_stopped.at[0].set(True))
    elif state_update == "db_service":
        state = state.replace(
            host_service_reliability=state.host_service_reliability.at[1, 0].set(50),
        )
    else:
        state = state.replace(
            host_decoy_reliability=state.host_decoy_reliability.at[2, 0].set(75),
        )

    actual = jax.jit(score_resilience_state)(state, _role_array())

    np.testing.assert_array_equal(actual, expected)


def test_jax_state_scorer_returns_zero_for_healthy_state_and_sums_simultaneous_impacts() -> None:
    state = create_initial_state()
    roles = _role_array()
    np.testing.assert_array_equal(score_resilience_state(state, roles), [0.0, 0.0, 0.0])

    impacted = state.replace(
        ot_service_stopped=state.ot_service_stopped.at[0].set(True),
        host_service_reliability=state.host_service_reliability.at[1, 0].set(90),
        host_decoy_reliability=state.host_decoy_reliability.at[2, 0].set(90),
    )
    np.testing.assert_array_equal(score_resilience_state(impacted, roles), [-20.0, -20.0, -30.0])


def test_episode_reducer_does_not_double_count_repeated_degradation() -> None:
    repeated_degradation = jnp.asarray(
        [
            [-10.0, -10.0, -10.0],
            [-10.0, -10.0, -10.0],
            [0.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(
        mean_resilience_episode(repeated_degradation),
        [-20.0 / 3, -20.0 / 3, -20.0 / 3],
    )
    np.testing.assert_array_equal(
        mean_resilience_episode(repeated_degradation, jnp.asarray([True, False, True])),
        [-5.0, -5.0, -5.0],
    )


def test_episode_summary_uses_sample_std_and_is_json_friendly() -> None:
    scores = jnp.asarray([[0.0, -10.0, -20.0], [-10.0, -20.0, -30.0]])
    summary = summarize_resilience_episodes(scores)

    assert summary.n == 2
    assert summary.mean == {"c": -5.0, "i": -15.0, "a": -25.0}
    assert summary.std == pytest.approx({"c": math.sqrt(50), "i": math.sqrt(50), "a": math.sqrt(50)})
    assert summary.to_dict() == {
        "n": 2,
        "c": {"mean": -5.0, "std": pytest.approx(math.sqrt(50))},
        "i": {"mean": -15.0, "std": pytest.approx(math.sqrt(50))},
        "a": {"mean": -25.0, "std": pytest.approx(math.sqrt(50))},
    }
    assert resilience_episode_records(scores) == [
        {"c": 0.0, "i": -10.0, "a": -20.0},
        {"c": -10.0, "i": -20.0, "a": -30.0},
    ]
    assert summarize_resilience_episodes([[1.0, 2.0, 3.0]]).std == {"c": 0.0, "i": 0.0, "a": 0.0}


def _event(cls: str, host: str, success: str = "TRUE") -> dict[str, str]:
    return {"cls": cls, "host": host, "success": success}


def _step(*, red: list[dict[str, str]] | None = None, blue: list[dict[str, str]] | None = None) -> dict:
    return {
        "red": {f"red_agent_{i}": event for i, event in enumerate(red or [])},
        "blue": {f"blue_agent_{i}": event for i, event in enumerate(blue or [])},
    }


def _offline_metric() -> ResilienceMetric:
    return ResilienceMetric({"auth": ROLE_AUTH, "database": ROLE_DB, "web": ROLE_WEB})


def test_offline_repeated_degradation_persists_until_successful_restore() -> None:
    score = _offline_metric().score_episode(
        {},
        [
            _step(red=[_event("DegradeServices", "auth")]),
            _step(red=[_event("DegradeServices", "auth")]),
            _step(blue=[_event("Restore", "auth")]),
        ],
        total_reward=4.0,
    )

    assert score.C_mean == pytest.approx(-20.0 / 3)
    assert score.I_mean == pytest.approx(-20.0 / 3)
    assert score.A_mean == pytest.approx(-20.0 / 3)
    assert score.C_min == -10.0
    assert score.impact_counts == {"auth": 2}


def test_offline_failed_attack_and_restore_do_not_change_impact_state() -> None:
    score = _offline_metric().score_episode(
        {},
        [
            _step(red=[_event("Impact", "auth", "FALSE")]),
            _step(red=[_event("Impact", "auth")]),
            _step(blue=[_event("Restore", "auth", "FALSE")]),
        ],
        total_reward=0.0,
    )

    assert score.C_mean == pytest.approx(-20.0 / 3)
    assert score.I_mean == pytest.approx(-20.0 / 3)
    assert score.A_mean == pytest.approx(-20.0 / 3)
    assert score.impact_counts == {"auth": 1}


def test_offline_same_step_restore_then_attack_leaves_host_impacted() -> None:
    score = _offline_metric().score_episode(
        {},
        [
            _step(
                blue=[_event("Restore", "auth")],
                red=[_event("Impact", "auth")],
            )
        ],
        total_reward=0.0,
    )

    assert (score.C_mean, score.I_mean, score.A_mean) == (-10.0, -10.0, -10.0)
    assert score.impact_counts == {"auth": 1}
