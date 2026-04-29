"""Tier 2.5 replay-trajectory parity: real rollout, both update bodies.

Builds a small rollout from `FsmRedCC4Env` (real CC4 obs / action masks /
rewards / dones), then feeds the same flattened minibatch into:

  - the JAX `_loss_fn` body inlined from `scripts/train/ippo_jax.py`
  - the torch loss body inlined from `scripts/train/ppo_cleanrl_cyborg.py`
    (post-`unbiased=False` ddof fix)

with **identical** initial params (Flax SharedActorCritic init →
converted to torch PPOAgent weights) and asserts:

  - per-tensor gradients within ≤1e-4
  - post-Adam-step params within ≤1e-4

The point: `test_ppo_update_parity.py` (Tier 1.1) tests on synthetic IID
Gaussian minibatches. This test feeds *real-distribution* data — sparse
action masks, very-negative log-probs on rare actions, mixed-magnitude
GAE — through the same machinery. If Tier 1 passes but this fails, the
parity break lives in real-distribution numerics rather than the loss
math itself.

Cited by: plans/jax/cc4/prompts/training-loop-parity-prompt.md (Tier 2.5).
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import torch
import torch.nn as nn
from flax.training.train_state import TrainState

from jaxborg.actions.masking import compute_blue_action_mask
from jaxborg.constants import BLUE_OBS_SIZE
from jaxborg.fsm_red_env import FsmRedCC4Env
from jaxborg.policy import SharedActorCritic

# Real CC4 dims; small hidden + small rollout for fast test wall time.
OBS_DIM = BLUE_OBS_SIZE  # 213 = 210 base + 3 Phase 3 mission-multiplier slots
ACT_DIM = 242
HIDDEN_DIM = 64

NUM_ENVS = 4
NUM_STEPS = 32
NUM_AGENTS = 5
NUM_MINIBATCHES = 4  # flat batch / 4 → minibatch size 160

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
LR = 3e-4
ADAM_EPS = 1e-5


# ── Mini PPOAgent: real OBS/ACT dims, smaller hidden ──────────────────────


class TinyPPOAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(OBS_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.actor = nn.Linear(HIDDEN_DIM, ACT_DIM)
        self.critic = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, obs, action_mask):
        h = torch.tanh(self.fc1(obs))
        h = torch.tanh(self.fc2(h))
        logits = self.actor(h)
        logits = logits + (action_mask.float() - 1.0) * 1e8
        value = self.critic(h).squeeze(-1)
        return logits, value


def _flax_to_torch_agent(flax_params: dict) -> TinyPPOAgent:
    """Build a torch PPOAgent whose weights match the given flax params."""
    agent = TinyPPOAgent()
    p = flax_params["params"]
    with torch.no_grad():
        agent.fc1.weight.copy_(torch.from_numpy(np.asarray(p["Dense_0"]["kernel"]).T.copy()))
        agent.fc1.bias.copy_(torch.from_numpy(np.asarray(p["Dense_0"]["bias"]).copy()))
        agent.fc2.weight.copy_(torch.from_numpy(np.asarray(p["Dense_1"]["kernel"]).T.copy()))
        agent.fc2.bias.copy_(torch.from_numpy(np.asarray(p["Dense_1"]["bias"]).copy()))
        agent.actor.weight.copy_(torch.from_numpy(np.asarray(p["Dense_2"]["kernel"]).T.copy()))
        agent.actor.bias.copy_(torch.from_numpy(np.asarray(p["Dense_2"]["bias"]).copy()))
        agent.critic.weight.copy_(torch.from_numpy(np.asarray(p["Dense_3"]["kernel"]).T.copy()).reshape(1, HIDDEN_DIM))
        agent.critic.bias.copy_(torch.from_numpy(np.asarray(p["Dense_3"]["bias"]).copy()).reshape(1))
    return agent


# ── Rollout collection on the real CC4 env ────────────────────────────────


def _collect_rollout(network, params, seed: int = 0):
    """Run a small rollout on FsmRedCC4Env. Returns flattened minibatch tensors.

    Returned arrays have leading dim batch_size = NUM_STEPS * NUM_ENVS * NUM_AGENTS.
    """
    env = FsmRedCC4Env(num_steps=500, topology_mode="generative", training_mode=True)
    agents = list(env.agents)

    init_key = jax.random.PRNGKey(seed)
    keys = jax.random.split(init_key, NUM_ENVS)
    obs, env_state = jax.vmap(env.reset)(keys)

    rng = jax.random.PRNGKey(seed + 100)
    agent_ids = jnp.arange(NUM_AGENTS)
    mask_over_envs = jax.vmap(compute_blue_action_mask, in_axes=(0, None, 0))
    mask_over_agents = jax.vmap(mask_over_envs, in_axes=(None, 0, None))

    obs_buf = []
    action_buf = []
    avail_buf = []
    logp_buf = []
    value_buf = []
    reward_buf = []
    done_buf = []

    # Plain Python loop — small rollout, no JIT needed for the test.
    for _ in range(NUM_STEPS):
        obs_batch = jnp.stack([obs[a] for a in agents], axis=-2)  # (env, agent, obs)
        avail_batch = mask_over_agents(env_state.const, agent_ids, env_state.state).transpose(1, 0, 2)

        rng, sample_key = jax.random.split(rng)
        flat_obs = obs_batch.reshape(-1, OBS_DIM)
        flat_avail = avail_batch.reshape(-1, ACT_DIM)
        pi, value = network.apply(params, flat_obs, flat_avail)
        action = pi.sample(seed=sample_key)
        log_prob = pi.log_prob(action)

        action_grid = action.reshape(NUM_ENVS, NUM_AGENTS)
        env_act = {agents[i]: action_grid[:, i] for i in range(NUM_AGENTS)}

        rng, step_key = jax.random.split(rng)
        step_keys = jax.random.split(step_key, NUM_ENVS)
        new_obs, new_env_state, rewards, dones, _info = jax.vmap(env.step)(step_keys, env_state, env_act)

        team_reward = rewards[agents[0]]  # (NUM_ENVS,)
        team_done = dones[agents[0]]

        obs_buf.append(obs_batch)
        action_buf.append(action_grid)
        avail_buf.append(avail_batch)
        logp_buf.append(log_prob.reshape(NUM_ENVS, NUM_AGENTS))
        value_buf.append(value.reshape(NUM_ENVS, NUM_AGENTS))
        reward_buf.append(team_reward)
        done_buf.append(team_done)

        obs = new_obs
        env_state = new_env_state

    # Bootstrap last value
    last_obs_batch = jnp.stack([obs[a] for a in agents], axis=-2)
    flat_last = last_obs_batch.reshape(-1, OBS_DIM)
    _, last_val = network.apply(params, flat_last)
    last_val = last_val.reshape(NUM_ENVS, NUM_AGENTS)

    obs_t = jnp.stack(obs_buf)  # (T, E, A, obs)
    action_t = jnp.stack(action_buf)  # (T, E, A)
    avail_t = jnp.stack(avail_buf)  # (T, E, A, ACT)
    logp_t = jnp.stack(logp_buf)  # (T, E, A)
    value_t = jnp.stack(value_buf)  # (T, E, A)
    reward_t = jnp.stack(reward_buf)  # (T, E) team-shared
    done_t = jnp.stack(done_buf)  # (T, E)

    # GAE — replicate ippo_jax.py convention. Done broadcast across agents.
    advantages = jnp.zeros_like(value_t)
    gae = jnp.zeros((NUM_ENVS, NUM_AGENTS), dtype=jnp.float32)
    next_value = last_val
    advs_rev = []
    for t in reversed(range(NUM_STEPS)):
        d = done_t[t][:, None]  # (E, 1)
        r = reward_t[t][:, None]  # (E, 1) -> broadcast over agents
        v = value_t[t]  # (E, A)
        delta = r + GAMMA * next_value * (1 - d) - v
        gae = delta + GAMMA * GAE_LAMBDA * (1 - d) * gae
        advs_rev.append(gae)
        next_value = v
    advantages = jnp.stack(list(reversed(advs_rev)), axis=0)  # (T, E, A)
    targets = advantages + value_t

    # Flatten to (T*E*A, ...)
    def flat(x):
        new_shape = (NUM_STEPS * NUM_ENVS * NUM_AGENTS,) + tuple(x.shape[3:])
        return x.reshape(new_shape)

    return {
        "obs": flat(obs_t),
        "action": flat(action_t),
        "avail": flat(avail_t),
        "old_logp": flat(logp_t),
        "old_value": flat(value_t),
        "advantage": flat(advantages),
        "target": flat(targets),
    }


# ── Loss bodies (inlined from each script, post-fix) ──────────────────────


def _torch_loss(agent, obs, actions, mask, old_logp, mb_adv_np, mb_ret_np):
    """Mirrors ppo_cleanrl_cyborg.py inner-loop loss math (post-ddof fix)."""
    obs_t = torch.from_numpy(np.asarray(obs))
    act_t = torch.from_numpy(np.asarray(actions))
    mask_t = torch.from_numpy(np.asarray(mask))
    old_lp_t = torch.from_numpy(np.asarray(old_logp))
    mb_adv = torch.from_numpy(np.asarray(mb_adv_np))
    mb_ret = torch.from_numpy(np.asarray(mb_ret_np))

    logits, value = agent(obs_t, mask_t)
    dist = torch.distributions.Categorical(logits=logits)
    new_lp = dist.log_prob(act_t)
    ent = dist.entropy()

    adv = (mb_adv - mb_adv.mean()) / (mb_adv.std(unbiased=False) + 1e-8)
    logratio = new_lp - old_lp_t
    ratio = logratio.exp()
    pg_loss1 = -adv * ratio
    pg_loss2 = -adv * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    vf_loss = 0.5 * ((value - mb_ret) ** 2).mean()
    entropy_loss = ent.mean()

    return pg_loss - ENT_COEF * entropy_loss + VF_COEF * vf_loss


def _jax_loss(params, network, obs, actions, mask, old_logp, mb_adv, mb_targets):
    gae = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
    pi, value = network.apply(params, obs, mask)
    log_prob = pi.log_prob(actions)
    value_loss = 0.5 * jnp.mean(jnp.square(value - mb_targets))
    ratio = jnp.exp(log_prob - old_logp)
    loss_actor1 = ratio * gae
    loss_actor2 = jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * gae
    loss_actor = -jnp.mean(jnp.minimum(loss_actor1, loss_actor2))
    entropy = pi.entropy().mean()
    return loss_actor + VF_COEF * value_loss - ENT_COEF * entropy


# ── The test ──────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_replay_minibatch_update_parity():
    """Real CC4 rollout fed through both PPO update bodies → identical update."""
    network = SharedActorCritic(action_dim=ACT_DIM, hidden_dim=HIDDEN_DIM, activation="tanh")
    init_key = jax.random.PRNGKey(123)
    init_x = jnp.zeros((OBS_DIM,), dtype=jnp.float32)
    flax_params = network.init(init_key, init_x)

    rollout = _collect_rollout(network, flax_params, seed=42)

    # Sanity-check rollout shapes
    batch_size = NUM_STEPS * NUM_ENVS * NUM_AGENTS
    assert rollout["obs"].shape == (batch_size, OBS_DIM)
    assert rollout["action"].shape == (batch_size,)

    # Take first minibatch (no shuffle — we want deterministic comparison)
    mb_size = batch_size // NUM_MINIBATCHES
    sl = slice(0, mb_size)
    mb = {k: rollout[k][sl] for k in rollout}

    # Build matched torch agent
    agent = _flax_to_torch_agent(flax_params)

    # Torch update
    optim_torch = torch.optim.Adam(agent.parameters(), lr=LR, eps=ADAM_EPS)
    loss_t = _torch_loss(agent, mb["obs"], mb["action"], mb["avail"], mb["old_logp"], mb["advantage"], mb["target"])
    optim_torch.zero_grad()
    loss_t.backward()
    torch_grads = {n: p.grad.detach().numpy().copy() for n, p in agent.named_parameters()}
    optim_torch.step()
    torch_post = {n: p.detach().numpy().copy() for n, p in agent.named_parameters()}

    # JAX update
    tx = optax.adam(LR, eps=ADAM_EPS)
    train_state = TrainState.create(apply_fn=network.apply, params=flax_params, tx=tx)
    grad_fn = jax.value_and_grad(_jax_loss)
    _, grads = grad_fn(
        train_state.params,
        network,
        mb["obs"],
        mb["action"],
        mb["avail"],
        mb["old_logp"],
        mb["advantage"],
        mb["target"],
    )
    train_state = train_state.apply_gradients(grads=grads)

    # Re-pack JAX grads/params in torch's named-param layout
    def _jax_to_torch_layout(tree):
        p = tree["params"]
        return {
            "fc1.weight": np.asarray(p["Dense_0"]["kernel"]).T,
            "fc1.bias": np.asarray(p["Dense_0"]["bias"]),
            "fc2.weight": np.asarray(p["Dense_1"]["kernel"]).T,
            "fc2.bias": np.asarray(p["Dense_1"]["bias"]),
            "actor.weight": np.asarray(p["Dense_2"]["kernel"]).T,
            "actor.bias": np.asarray(p["Dense_2"]["bias"]),
            "critic.weight": np.asarray(p["Dense_3"]["kernel"]).T.reshape(1, HIDDEN_DIM),
            "critic.bias": np.asarray(p["Dense_3"]["bias"]).reshape(1),
        }

    jax_grads = _jax_to_torch_layout(grads)
    jax_post = _jax_to_torch_layout(train_state.params)

    # Per-tensor gradient parity
    for name, t_grad in torch_grads.items():
        np.testing.assert_allclose(
            jax_grads[name],
            t_grad,
            atol=1e-4,
            rtol=1e-4,
            err_msg=f"replay gradient mismatch on {name}",
        )

    # Post-Adam-step parity
    for name, t_post in torch_post.items():
        np.testing.assert_allclose(
            jax_post[name],
            t_post,
            atol=1e-4,
            rtol=1e-4,
            err_msg=f"replay post-Adam mismatch on {name}",
        )
