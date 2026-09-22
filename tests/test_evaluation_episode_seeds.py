from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jaxborg.evaluation import cyborg_runner, jax_runner
from jaxborg.evaluation.episode_seeds import expand_episode_seeds
from jaxborg.scenarios.cc4.game_variants import CC4_STOCK


def test_adjacent_seeds_produce_independent_paired_episodes():
    seeds = list(range(1000, 1100))
    expanded = expand_episode_seeds(seeds, 6)
    assert expanded == list(range(6000, 6600))
    assert len(set(expanded)) == 600
    assert expand_episode_seeds([1001], 6) == expanded[6:12]
    assert expand_episode_seeds(seeds, 1) == seeds


@pytest.mark.parametrize(
    "seeds,episodes",
    [([], 1), ([1, 1], 1), ([-1], 1), ([True], 1), ([1], 0), ([1], True), ([2**32 - 1], 2)],
)
def test_invalid_seed_requests_fail_before_rollout(seeds, episodes):
    with pytest.raises(ValueError):
        expand_episode_seeds(seeds, episodes)


@pytest.mark.parametrize("backend", ["jax", "cyborg"])
@pytest.mark.parametrize("workers", [1, 2])
def test_native_runners_use_the_same_episode_seeds_across_worker_counts(monkeypatch, backend, workers):
    module = jax_runner if backend == "jax" else cyborg_runner
    created = []
    recipe = {"cage4_enhanced_obs": False}
    if backend == "jax":
        monkeypatch.setattr(module, "load_jax_checkpoint", lambda _: (None, None, recipe))
        monkeypatch.setattr(module, "run_episode", lambda *args, ep_seed, **kwargs: float(ep_seed))
        evaluate = module.evaluate_jax_on_cyborg
    else:
        monkeypatch.setattr(module, "load_torch_policy", lambda _: (None, recipe))
        monkeypatch.setattr(module, "rollout_episode", lambda *args, ep_seed, **kwargs: float(ep_seed))
        evaluate = module.evaluate_on_cyborg
    monkeypatch.setattr(module, "make_cyborg_env", lambda variant, seed, **kwargs: created.append(seed))

    class InlinePool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, function, args):
            return map(function, args)

    monkeypatch.setattr(module.concurrent.futures, "ProcessPoolExecutor", InlinePool)
    rewards, seeds, *_ = evaluate(
        "model.safetensors" if backend == "jax" else "model.pt",
        variant=CC4_STOCK,
        seeds=[1000, 1001],
        episodes_per_seed=6,
        workers=workers,
        progress=False,
    )
    assert seeds == list(range(6000, 6012))
    assert rewards == [float(seed) for seed in seeds]
    assert sorted(created) == seeds


def test_torch_rollouts_seed_policy_sampling_per_episode(monkeypatch):
    monkeypatch.setattr(cyborg_runner, "reset_cyborg_env", lambda *args, **kwargs: SimpleNamespace(obs={}, info={}))
    monkeypatch.setattr(cyborg_runner, "_pad_obs_mask", lambda *args: (np.zeros((5, 1)), np.ones((5, 2))))

    class Policy:
        def get_action_and_value(self, *args):
            return torch.randint(0, 2, (5,)), None, None, None

    class Env:
        def __init__(self):
            self.actions = []

        def step(self, actions):
            self.actions.append(actions)
            return {}, {"blue_agent_0": 0}, {"__all__": False}, {"__all__": False}, {}

    def actions(seed):
        env = Env()
        cyborg_runner.rollout_episode(env, replace(CC4_STOCK, num_steps=8), seed, Policy(), deterministic=False)
        return env.actions

    assert actions(42) == actions(42)
    assert actions(42) != actions(43)
