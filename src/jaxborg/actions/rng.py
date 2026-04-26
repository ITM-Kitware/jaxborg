import jax
import jax.numpy as jnp

from jaxborg.state import SimulatorConst, SimulatorState


def sample_detection_random(state: SimulatorState, const: SimulatorConst, key: jax.Array):
    """Return (random_float, updated_state). Uses precomputed sequence if enabled, else JAX RNG."""
    return jax.lax.cond(
        const.use_detection_randoms,
        lambda s: _from_sequence(s, const),
        lambda s: (jax.random.uniform(key), s),
        state,
    )


def _from_sequence(state: SimulatorState, const: SimulatorConst):
    idx = state.detection_random_index
    val = const.detection_randoms[idx]
    new_state = state.replace(detection_random_index=idx + 1)
    return val, new_state


def sample_green_random(const: SimulatorConst, time, host_idx, field_idx, key, *, int_range=None):
    """Return a random value. Uses precomputed green_randoms if enabled, else JAX RNG.

    When int_range is provided, returns an int32 in [0, int_range).
    When int_range is None, returns a float32 uniform in [0, 1).
    """
    if int_range is not None:

        def from_precomputed(_):
            v = const.green_randoms[time, host_idx, field_idx]
            # Clamp to [0, int_range-1] — tokens can be exactly 1.0 for
            # fields that encode a direct index (e.g. service token).
            return jnp.minimum(jnp.floor(v * int_range).astype(jnp.int32), jnp.maximum(int_range - 1, 0))

        def from_rng(_):
            return jax.random.randint(key, (), 0, jnp.maximum(int_range, 1))
    else:

        def from_precomputed(_):
            return const.green_randoms[time, host_idx, field_idx]

        def from_rng(_):
            return jax.random.uniform(key)

    return jax.lax.cond(const.use_green_randoms, from_precomputed, from_rng, None)


def sample_red_policy_random(const: SimulatorConst, time, agent_id, field_idx, key):
    """Return a precomputed red-policy choice token encoded in [0, 1), else JAX uniform."""

    def from_precomputed(_):
        return const.red_policy_randoms[time, agent_id, field_idx]

    def from_rng(_):
        return jax.random.uniform(key)

    return jax.lax.cond(const.use_red_policy_randoms, from_precomputed, from_rng, None)


def sample_red_pid_delta(const: SimulatorConst, time, agent_id, key):
    """Return Host.create_pid delta in [1, 9] for exploit session creation."""

    def from_precomputed(_):
        return jnp.maximum(const.red_pid_deltas[time, agent_id], jnp.int32(1))

    def from_rng(_):
        return jax.random.randint(key, (), minval=1, maxval=10, dtype=jnp.int32)

    return jax.lax.cond(const.use_red_pid_deltas, from_precomputed, from_rng, None)


def sample_red_privesc_choice(const: SimulatorConst, time, agent_id, key, num_sessions):
    """Return privesc session choice index in [0, num_sessions).

    Uses precomputed CybORG choice if available, else JAX RNG."""

    def from_precomputed(_):
        return jnp.clip(const.red_privesc_choices[time, agent_id], 0, jnp.maximum(num_sessions - 1, 0))

    def from_rng(_):
        return jax.random.randint(key, (), minval=0, maxval=jnp.maximum(num_sessions, 1), dtype=jnp.int32)

    return jax.lax.cond(const.use_red_privesc_choices, from_precomputed, from_rng, None)


def sample_red_session_check_choice(const: SimulatorConst, time, agent_id, key, num_sessions_on_host):
    """Return session-check within-host slot index in [0, num_sessions_on_host).

    CybORG's RedSessionCheck picks a random session via np_random.choice(all_sessions).
    The harness remaps this to a within-host slot index so JAX picks the same
    session on the promoted host.  Uses precomputed CybORG choice if available,
    else JAX RNG (uniform over sessions on the chosen host).
    """

    def from_precomputed(_):
        return jnp.clip(
            const.red_session_check_choices[time, agent_id],
            0,
            jnp.maximum(num_sessions_on_host - 1, 0),
        )

    def from_rng(_):
        return jax.random.randint(key, (), minval=0, maxval=jnp.maximum(num_sessions_on_host, 1), dtype=jnp.int32)

    return jax.lax.cond(const.use_red_session_check_choices, from_precomputed, from_rng, None)


def sample_exploit_session_choice(const: SimulatorConst, time, agent_id, key, visible_sessions):
    """Return exploit session choice index in [0, visible_sessions).

    CybORG's FSM picks uniformly from server_session (abstract sessions in
    allowed subnets).  Only session 0 holds scan data, so the exploit succeeds
    iff the choice is 0.  Uses precomputed CybORG choice if available, else
    JAX RNG.
    """

    def from_precomputed(_):
        return const.red_exploit_session_choices[time, agent_id]

    def from_rng(_):
        return jax.random.randint(key, (), minval=0, maxval=jnp.maximum(visible_sessions, 1), dtype=jnp.int32)

    return jax.lax.cond(const.use_red_exploit_session_choices, from_precomputed, from_rng, None)


def sample_blue_decoy_type_choice(const: SimulatorConst, time, agent_id, compatibility, key):
    """Return a compatible decoy type index, selected randomly from available types.

    Uses precomputed CybORG choice if available, else JAX RNG from the
    provided key.  The returned index is a raw decoy type (0–3) guaranteed
    to be compatible with the host.
    """

    def from_precomputed(_):
        return const.blue_decoy_type_choices[time, agent_id]

    def from_fallback_rng(_):
        perm = jax.random.permutation(key, 4, independent=True)
        # Pick the first compatible type in the random permutation
        scores = jnp.where(compatibility, perm, jnp.int32(100))
        return jnp.argmin(scores).astype(jnp.int32)

    return jax.lax.cond(const.use_blue_decoy_type_choices, from_precomputed, from_fallback_rng, None)


def sample_blue_decoy_pid_delta(const: SimulatorConst, time, agent_id, key, respawn_index=0):
    """Return Host.create_pid delta in [1, 9] for blue decoy process creation.

    respawn_index selects which precomputed delta to use when multiple decoys
    are respawned in a single Remove action (0 for DeployDecoy or first respawn).
    """

    def from_precomputed(_):
        return jnp.maximum(const.blue_decoy_pid_deltas[time, agent_id, respawn_index], jnp.int32(1))

    def from_fallback_rng(_):
        return jax.random.randint(key, (), minval=1, maxval=10, dtype=jnp.int32)

    return jax.lax.cond(const.use_blue_decoy_pid_deltas, from_precomputed, from_fallback_rng, None)
