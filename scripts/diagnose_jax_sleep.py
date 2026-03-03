"""Measure JAXborg sleep baseline reward over a full episode."""

import jax
import jax.numpy as jnp

from jaxborg.actions.encoding import BLUE_SLEEP
from jaxborg.constants import NUM_BLUE_AGENTS
from jaxborg.fsm_red_env import FsmRedCC4Env

SEEDS = [42, 123, 456]


def main():
    env = FsmRedCC4Env(num_steps=500)

    for seed in SEEDS:
        key = jax.random.PRNGKey(seed)
        _, state = env.reset(key)
        total = 0.0
        for step in range(500):
            key, step_key = jax.random.split(key)
            actions = {f"blue_{i}": jnp.int32(BLUE_SLEEP) for i in range(NUM_BLUE_AGENTS)}
            _, state, rewards, _, _ = env.step(step_key, state, actions)
            total += float(rewards["blue_0"])
        print(f"Seed {seed}: JAXborg sleep baseline = {total:.1f}")


if __name__ == "__main__":
    main()
