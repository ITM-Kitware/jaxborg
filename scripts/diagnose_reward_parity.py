"""Compare JAXborg vs CybORG rewards step-by-step with sleep policy.

Runs both systems with identical topology and sleep actions for N steps.
Reports per-step reward differences to find dynamics gaps.
"""

from statistics import mean

import jax
import jax.numpy as jnp
import numpy as np
from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Simulator.Actions import Sleep
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.actions.encoding import BLUE_SLEEP
from jaxborg.constants import NUM_BLUE_AGENTS
from jaxborg.fsm_red_env import FsmRedCC4Env

SEED = 42
NUM_STEPS = 100


def main():
    # --- CybORG side ---
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500,
    )
    cyborg = CybORG(sg, "sim", seed=SEED)
    env = BlueFlatWrapper(env=cyborg, pad_spaces=True)
    env.reset()

    # --- JAXborg side: use FsmRedCC4Env with same topology ---
    fsm_env = FsmRedCC4Env(num_steps=500)
    key = jax.random.PRNGKey(SEED)
    jax_obs, jax_state = fsm_env.reset(key)

    print("=" * 70)
    print("REWARD PARITY: Sleep Blue in CybORG vs JAXborg (FsmRedCC4Env)")
    print(f"Seed={SEED}, Steps={NUM_STEPS}")
    print("=" * 70)
    print(f"{'Step':>5s} {'CybORG':>10s} {'JAXborg':>10s} {'Diff':>10s} {'CumCyb':>10s} {'CumJax':>10s}")
    print("-" * 70)

    cum_cyborg = 0.0
    cum_jax = 0.0
    diffs = []

    for step in range(NUM_STEPS):
        # CybORG step: all blue sleep
        cyborg_actions = {a: Sleep() for a in env.agents}
        _, cyborg_rewards, _, _, _ = env.step(actions=cyborg_actions)
        cyborg_r = mean(cyborg_rewards.values())

        # JAXborg step: all blue sleep
        key, step_key = jax.random.split(key)
        blue_actions = {f"blue_{i}": jnp.int32(BLUE_SLEEP) for i in range(NUM_BLUE_AGENTS)}
        jax_obs, jax_state, jax_rewards, jax_dones, _ = fsm_env.step(step_key, jax_state, blue_actions)
        jax_r = float(jax_rewards["blue_0"])

        diff = cyborg_r - jax_r
        cum_cyborg += cyborg_r
        cum_jax += jax_r
        diffs.append(diff)

        if step < 20 or step % 10 == 0 or abs(diff) > 0.5:
            print(f"{step:5d} {cyborg_r:10.2f} {jax_r:10.2f} {diff:10.2f} {cum_cyborg:10.1f} {cum_jax:10.1f}")

    print("-" * 70)
    print(f"Total: CybORG={cum_cyborg:.1f}  JAXborg={cum_jax:.1f}  Diff={cum_cyborg - cum_jax:.1f}")
    print(f"Mean step diff: {np.mean(diffs):.3f} ± {np.std(diffs):.3f}")
    nonzero_diffs = [d for d in diffs if abs(d) > 0.01]
    print(f"Steps with diff > 0.01: {len(nonzero_diffs)}/{NUM_STEPS}")


if __name__ == "__main__":
    main()
