import pytest

from tests.differential.fuzzer import run_differential_fuzz

pytestmark = pytest.mark.slow

STRICT_RANDOM_BLUE_SEEDS = list(range(1000))
STRICT_RANDOM_BLUE_STEPS = 500


def _seed_id(seed: int) -> str:
    return f"seed_{seed:02d}"


class TestStrictRandomBlueParityCampaign:
    @pytest.mark.parametrize("seed", STRICT_RANDOM_BLUE_SEEDS, ids=_seed_id)
    def test_random_blue_strict_parity(self, seed):
        """Run one strict random-blue differential rollout per seed.

        This is intentionally one seed per pytest item so xdist can distribute
        the campaign across workers. Nested multiprocessing inside the fuzzer is
        disabled here on purpose.
        """
        report = run_differential_fuzz(
            seeds=[seed],
            max_steps_per_seed=STRICT_RANDOM_BLUE_STEPS,
            verbose=False,
            mismatch_mode="all",
            blue_agent="random",
            blue_action_source="cyborg_policy",
            strict_random_sync=True,
            check_obs=True,
            check_masks=True,
            parallel=1,
        )
        assert report is None, str(report)
