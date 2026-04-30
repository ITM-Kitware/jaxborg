import jax
import jax.numpy as jnp

from scripts.train.algorithms.ippo_jax import compute_value_loss


def test_unclipped_value_loss_keeps_gradient_for_large_target_error():
    old_value = jnp.array([0.0], dtype=jnp.float32)
    target = jnp.array([-154.0], dtype=jnp.float32)
    value = jnp.array([-1.0], dtype=jnp.float32)

    clipped_grad = jax.grad(lambda v: compute_value_loss(v, old_value, target, clip_eps=0.2, clip_value_loss=True))(
        value
    )
    unclipped_grad = jax.grad(lambda v: compute_value_loss(v, old_value, target, clip_eps=0.2, clip_value_loss=False))(
        value
    )

    assert float(clipped_grad[0]) == 0.0
    assert float(unclipped_grad[0]) > 100.0
