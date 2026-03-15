"""Differential fuzzer: runs episodes across many seeds to find CybORG/JAX divergences."""

import time
from dataclasses import dataclass

from CybORG.Agents import (
    EnterpriseGreenAgent,
    FiniteStateRedAgent,
    MonitorAgent,
    SleepAgent,
    cc4BlueRandomAgent,
)

from tests.differential.harness import CC4DifferentialHarness
from tests.differential.state_comparator import _ERROR_FIELDS, format_diffs


@dataclass
class MismatchReport:
    seed: int
    step: int
    field_name: str
    host_or_agent: str
    cyborg_value: object
    jax_value: object
    all_diffs_str: str

    def __str__(self):
        return (
            f"Mismatch at seed={self.seed}, step={self.step}: "
            f"{self.field_name} [{self.host_or_agent}] "
            f"cyborg={self.cyborg_value} jax={self.jax_value}\n"
            f"All diffs:\n{self.all_diffs_str}"
        )


BLUE_AGENT_CLASSES = {
    "sleep": SleepAgent,
    "monitor": MonitorAgent,
    "random": cc4BlueRandomAgent,
}

BLUE_ACTION_SOURCES = {"sleep", "cyborg_policy"}


def run_differential_fuzz(
    seeds=range(20),
    max_steps_per_seed=100,
    verbose=False,
    mismatch_mode: str = "error",
    blue_agent: str = "sleep",
    blue_action_source: str = "sleep",
    strict_random_sync: bool = False,
    check_obs: bool = True,
) -> MismatchReport | None:
    if mismatch_mode not in {"error", "all"}:
        raise ValueError("mismatch_mode must be one of: error, all")
    if blue_agent not in BLUE_AGENT_CLASSES:
        raise ValueError(f"blue_agent must be one of: {sorted(BLUE_AGENT_CLASSES)}")
    if blue_action_source not in BLUE_ACTION_SOURCES:
        raise ValueError(f"blue_action_source must be one of: {sorted(BLUE_ACTION_SOURCES)}")

    blue_cls = BLUE_AGENT_CLASSES[blue_agent]
    use_cyborg_blue_policy = blue_action_source == "cyborg_policy"

    wall_start = time.time()
    for seed in seeds:
        seed_start = time.time()
        if verbose:
            print(f"--- Seed {seed} (blue_agent={blue_agent}, blue_action_source={blue_action_source}) ---")

        harness = CC4DifferentialHarness(
            seed=seed,
            max_steps=500,
            blue_cls=blue_cls,
            green_cls=EnterpriseGreenAgent,
            red_cls=FiniteStateRedAgent,
            sync_green_rng=True,
            strict_random_sync=strict_random_sync,
            use_cyborg_blue_policy=use_cyborg_blue_policy,
            check_obs=check_obs,
        )
        harness.reset()

        for t in range(max_steps_per_seed):
            try:
                result = harness.full_step()
            except AssertionError as err:
                return MismatchReport(
                    seed=seed,
                    step=t,
                    field_name="random_sync",
                    host_or_agent="",
                    cyborg_value="strict_random_sync",
                    jax_value="unsupported",
                    all_diffs_str=str(err),
                )

            if mismatch_mode == "all":
                candidate_diffs = result.diffs
            else:
                candidate_diffs = [d for d in result.diffs if d.field_name in _ERROR_FIELDS]

            if candidate_diffs:
                d = candidate_diffs[0]
                return MismatchReport(
                    seed=seed,
                    step=t,
                    field_name=d.field_name,
                    host_or_agent=d.host_or_agent,
                    cyborg_value=d.cyborg_value,
                    jax_value=d.jax_value,
                    all_diffs_str=format_diffs(result.diffs),
                )

        if verbose:
            elapsed = time.time() - seed_start
            print(f"  Seed {seed}: {max_steps_per_seed} steps clean ({elapsed:.1f}s)")

    if verbose:
        total = time.time() - wall_start
        print(f"\nTotal: {total:.1f}s")
    return None


if __name__ == "__main__":
    print("Running differential fuzzer...")
    report = run_differential_fuzz(verbose=True)
    if report:
        print(f"\nFOUND MISMATCH:\n{report}")
    else:
        print("\nNo mismatches found across all seeds!")
