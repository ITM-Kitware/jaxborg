"""Optimizer-step parity: torch.optim.Adam vs optax.adam.

Given identical params + grads + (m, v, step) state, both optimizers must
produce identical post-step params to ~1e-6. This test isolates Adam math
(bias correction timing, eps placement, sqrt order) from any upstream
loss-function asymmetry.

If this test fails, the −227 pt matched-training gap localizes to optimizer
math; if it passes, optimizer math is ruled out and divergence lives upstream.

Cited by: plans/jax/cc4/prompts/training-loop-parity-prompt.md (Tier 1.2).
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import torch


def _torch_adam_step(params_np, grads_np, *, lr, eps, betas, m_init, v_init, t_init):
    """Run one torch.optim.Adam step and return post-step params (numpy)."""
    p = torch.from_numpy(params_np.copy()).requires_grad_(True)
    p.grad = torch.from_numpy(grads_np.copy())

    opt = torch.optim.Adam([p], lr=lr, eps=eps, betas=betas)

    # Prime the optimizer state to (m_init, v_init, step=t_init) so we are
    # not just measuring step 1 — for any t≥1, bias correction differs.
    state = opt.state[p]
    state["step"] = torch.tensor(float(t_init))
    state["exp_avg"] = torch.from_numpy(m_init.copy())
    state["exp_avg_sq"] = torch.from_numpy(v_init.copy())

    opt.step()
    return p.detach().numpy()


def _optax_adam_step(params_np, grads_np, *, lr, eps, betas, m_init, v_init, t_init):
    """Run one optax.adam step and return post-step params (numpy)."""
    params = jnp.asarray(params_np)
    grads = jnp.asarray(grads_np)

    tx = optax.adam(learning_rate=lr, b1=betas[0], b2=betas[1], eps=eps)
    state = tx.init(params)
    # state is a tuple: (ScaleByAdamState(count, mu, nu), EmptyState)
    # Prime mu/nu/count to match torch's exp_avg/exp_avg_sq/step.
    scale_state = state[0]
    primed = scale_state._replace(
        count=jnp.asarray(t_init, dtype=scale_state.count.dtype),
        mu=jnp.asarray(m_init),
        nu=jnp.asarray(v_init),
    )
    state = (primed,) + state[1:]

    updates, _ = tx.update(grads, state, params)
    new_params = optax.apply_updates(params, updates)
    return np.asarray(new_params)


@pytest.mark.parametrize("t_init", [0, 1, 10, 100])
@pytest.mark.parametrize("lr", [3e-4, 1e-3])
@pytest.mark.parametrize("eps", [1e-5, 1e-8])
def test_adam_single_step_parity(t_init, lr, eps):
    """torch.optim.Adam ≈ optax.adam on one step from a primed state."""
    rng = np.random.default_rng(0)
    n = 256
    params = rng.standard_normal(n).astype(np.float32)
    grads = rng.standard_normal(n).astype(np.float32) * 0.1
    m = rng.standard_normal(n).astype(np.float32) * 0.01 if t_init > 0 else np.zeros(n, dtype=np.float32)
    v = (rng.standard_normal(n).astype(np.float32) * 0.001) ** 2 if t_init > 0 else np.zeros(n, dtype=np.float32)

    betas = (0.9, 0.999)

    p_torch = _torch_adam_step(params, grads, lr=lr, eps=eps, betas=betas, m_init=m, v_init=v, t_init=t_init)
    p_optax = _optax_adam_step(params, grads, lr=lr, eps=eps, betas=betas, m_init=m, v_init=v, t_init=t_init)

    np.testing.assert_allclose(
        p_torch,
        p_optax,
        atol=1e-6,
        rtol=1e-6,
        err_msg=f"torch.Adam vs optax.adam diverged at t_init={t_init}, lr={lr}, eps={eps}",
    )


def test_adam_many_steps_parity():
    """200 sequential Adam steps with shared grad streams stay aligned to ~1e-5.

    Compounds float-op-order differences across many steps; if torch and
    optax accumulate moments differently the gap will widen. 200 steps
    matches ~one PPO update (4 epochs × 16 minibatches = 64 minibatch
    updates), with a 3× safety margin.
    """
    rng = np.random.default_rng(1)
    n = 1024
    params_np = rng.standard_normal(n).astype(np.float32)
    grad_stream = [rng.standard_normal(n).astype(np.float32) * 0.05 for _ in range(200)]

    lr, eps, betas = 3e-4, 1e-5, (0.9, 0.999)

    # Torch path
    p_torch = torch.from_numpy(params_np.copy()).requires_grad_(True)
    opt_torch = torch.optim.Adam([p_torch], lr=lr, eps=eps, betas=betas)
    for g in grad_stream:
        p_torch.grad = torch.from_numpy(g.copy())
        opt_torch.step()
    final_torch = p_torch.detach().numpy()

    # Optax path
    p_jax = jnp.asarray(params_np)
    tx = optax.adam(learning_rate=lr, b1=betas[0], b2=betas[1], eps=eps)
    state = tx.init(p_jax)

    @jax.jit
    def step(p, s, g):
        u, s = tx.update(g, s, p)
        return optax.apply_updates(p, u), s

    for g in grad_stream:
        p_jax, state = step(p_jax, state, jnp.asarray(g))
    final_optax = np.asarray(p_jax)

    np.testing.assert_allclose(
        final_torch,
        final_optax,
        atol=1e-5,
        rtol=1e-5,
        err_msg="torch.Adam and optax.adam drift apart over 200 steps",
    )
