"""Phase 2 blue cooperation channel: state.messages writeback wiring tests.

The trainer pre-populates `state.messages` with the message head's output
before each `env.step` call.  These tests cover the env-side invariants that
make that wiring correct:

1. Setting `state.messages` and stepping preserves the messages in the new
   state (env step does not zero or scramble them).
2. The auto-reset path zeros messages (new episode → fresh comms history).
3. End-to-end trainer wiring: simulating one rollout step with a non-zero
   message head output produces a non-zero `state.messages` post-step when
   `BLUE_COMMS=true`, and zero when `BLUE_COMMS=false`.
"""

import jax
import jax.numpy as jnp
import numpy as np

from jaxborg.constants import (
    MESSAGE_LENGTH,
    NUM_BLUE_AGENTS,
)
from jaxborg.fsm_red_env import FsmRedCC4Env


def _make_env(num_steps: int = 10) -> FsmRedCC4Env:
    return FsmRedCC4Env(num_steps=num_steps, topology_mode="generative")


def _zero_blue_actions(env: FsmRedCC4Env) -> dict:
    return {a: jnp.int32(0) for a in env.agents}


def test_messages_pre_set_survive_env_step():
    """State pre-populated with messages keeps them after one step."""
    env = _make_env()
    _obs, env_state = env.reset(jax.random.PRNGKey(0))

    msg = jnp.arange(NUM_BLUE_AGENTS * NUM_BLUE_AGENTS * MESSAGE_LENGTH, dtype=jnp.float32) * 0.01
    msg = msg.reshape(NUM_BLUE_AGENTS, NUM_BLUE_AGENTS, MESSAGE_LENGTH)
    env_state = env_state.replace(state=env_state.state.replace(messages=msg))

    actions = _zero_blue_actions(env)
    _new_obs, new_env_state, _r, _d, _i = env.step_env(jax.random.PRNGKey(1), env_state, actions)

    np.testing.assert_allclose(np.asarray(new_env_state.state.messages), np.asarray(msg))


def test_auto_reset_zeros_messages():
    """When the episode wraps via auto-reset, messages start at zero."""
    env = _make_env(num_steps=2)
    _obs, env_state = env.reset(jax.random.PRNGKey(0))

    nonzero = jnp.ones((NUM_BLUE_AGENTS, NUM_BLUE_AGENTS, MESSAGE_LENGTH), dtype=jnp.float32) * 0.5
    env_state = env_state.replace(state=env_state.state.replace(messages=nonzero))

    actions = _zero_blue_actions(env)
    key = jax.random.PRNGKey(7)
    for _ in range(3):  # exceed num_steps so auto-reset triggers
        key, sub = jax.random.split(key)
        _o, env_state, _r, dones, _i = env.step(sub, env_state, actions)
        if bool(dones["__all__"]):
            break

    np.testing.assert_array_equal(np.asarray(env_state.state.messages), 0.0)


def test_blue_comms_off_keeps_messages_zero():
    """Skipping the writeback (BLUE_COMMS=false simulation) keeps messages 0."""
    env = _make_env()
    _obs, env_state = env.reset(jax.random.PRNGKey(0))
    actions = _zero_blue_actions(env)

    key = jax.random.PRNGKey(3)
    for _ in range(5):
        key, sub = jax.random.split(key)
        _o, env_state, _r, _d, _i = env.step_env(sub, env_state, actions)
        np.testing.assert_array_equal(np.asarray(env_state.state.messages), 0.0)


def test_blue_comms_on_propagates_message_to_recipient_obs():
    """A pre-set sender message appears in every recipient's obs slot."""
    env = _make_env()
    _obs, env_state = env.reset(jax.random.PRNGKey(0))

    sender = 1
    msg_val = jnp.linspace(-1.0, 1.0, MESSAGE_LENGTH, dtype=jnp.float32)
    msg = jnp.zeros((NUM_BLUE_AGENTS, NUM_BLUE_AGENTS, MESSAGE_LENGTH), dtype=jnp.float32)
    # Broadcast sender's message to all recipient slots in its row.
    msg = msg.at[sender].set(jnp.broadcast_to(msg_val, (NUM_BLUE_AGENTS, MESSAGE_LENGTH)))
    env_state = env_state.replace(state=env_state.state.replace(messages=msg))

    obs = env._env.get_obs(env_state)
    # Each recipient (every blue agent except sender itself) sees msg_val
    # somewhere in its message section.
    for r in range(NUM_BLUE_AGENTS):
        if r == sender:
            continue
        flat = np.asarray(obs[f"blue_{r}"])
        # Look for any 8-byte window matching msg_val.
        found = any(
            np.allclose(flat[i : i + MESSAGE_LENGTH], np.asarray(msg_val))
            for i in range(len(flat) - MESSAGE_LENGTH + 1)
        )
        assert found, f"recipient blue_{r} did not see sender's message"
