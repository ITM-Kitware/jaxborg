from tests.differential.fuzzer import run_differential_fuzz


def test_fuzzer_runs_with_cyborg_random_blue_policy():
    report = run_differential_fuzz(
        seeds=[0],
        max_steps_per_seed=20,
        mismatch_mode="all",
        blue_agent="random",
        blue_action_source="cyborg_policy",
        verbose=False,
    )
    assert report is None
