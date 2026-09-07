from __future__ import annotations

from importlib import import_module
from pathlib import Path

_CIA_CONFIG = {
    "enabled": True,
    "metric": "resilience",
    "role_assignment": "fixed_per_topology",
}


def test_builtin_cia_paths_preserve_shared_role_map_ids_for_copied_topology(tmp_path, monkeypatch):
    # Import MLflow's checkpoint module before JAX; importing Matplotlib after
    # JAX under pytest's fd capture can close the worker's captured stream.
    training_checkpoint = import_module("jaxborg.evaluation.training_checkpoint")
    TrainingCheckpointEvaluation = training_checkpoint.TrainingCheckpointEvaluation
    evaluate_training_checkpoint = training_checkpoint.evaluate_training_checkpoint

    import jax

    from jaxborg.evaluation import matchup_runner
    from jaxborg.evaluation.cia.fixed_topology import build_evaluation_cases
    from jaxborg.evaluation.matchup_runner import LoadedMatchupPolicy
    from jaxborg.scenarios.cc4.game_variants import CIA_RESILIENCE
    from jaxborg.scenarios.cc4.topology import build_topology, save_topology

    original = tmp_path / "original.snapshot.npz"
    copied = tmp_path / "copied.snapshot.npz"
    const = build_topology(jax.random.PRNGKey(31), op_zone_min_servers=3)
    save_topology(const, original, metadata={"copy": False})
    save_topology(const, copied, metadata={"copy": True})
    bank = (original, copied)
    expected_cases = build_evaluation_cases(bank, [17], 1)
    expected_ids = [case.role_map_id for case in expected_cases]
    assert expected_ids[0] == expected_ids[1]

    monkeypatch.setattr(
        matchup_runner,
        "load_matchup_policy",
        lambda path, *, team, backend: LoadedMatchupPolicy(
            team,
            "jax",
            None,
            None,
            {"path": str(path)},
        ),
    )
    monkeypatch.setattr(matchup_runner, "make_joint_jax_env", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        matchup_runner,
        "run_matchup_episode",
        lambda *args, **kwargs: (0.0, [0.0, 0.0, 0.0]),
    )
    learned = matchup_runner.evaluate_matchup(
        "blue.safetensors",
        "red.safetensors",
        backend="jax",
        variant=CIA_RESILIENCE,
        seeds=[17],
        episodes_per_seed=1,
        progress=False,
        topology_path=bank,
        topology_sampling="exhaustive",
        cia=_CIA_CONFIG,
    )
    assert learned.episode_role_map_ids == expected_ids

    monkeypatch.setattr(matchup_runner, "evaluate_matchup", lambda *args, **kwargs: learned)
    monkeypatch.setattr(
        training_checkpoint,
        "project_eval",
        lambda *args, **kwargs: {
            "TOPOLOGY_BANK": bank,
            "TOPOLOGY_SAMPLING": "exhaustive",
            "CIA": _CIA_CONFIG,
        },
    )
    recipe = {
        "meta": {"name": "cross-path"},
        "train": {"teams": "both"},
        "eval": {"variant": "cia_resilience", "cia": _CIA_CONFIG},
        "mlflow": {"checkpoint_eval": {"seed": 17}},
    }
    checkpoint = evaluate_training_checkpoint(
        "checkpoint.safetensors",
        backend="jax",
        recipe=recipe,
        seed=0,
        episodes_per_seed=1,
    )
    assert isinstance(checkpoint, TrainingCheckpointEvaluation)
    assert checkpoint.episode_role_map_ids == expected_ids

    from jaxborg.evaluation import play_priors
    from jaxborg.evaluation.play_priors import PeriodicCheckpoint, _result_row

    monkeypatch.setattr(play_priors, "_git_commit", lambda: "test-commit")
    play_priors_row = _result_row(
        evaluation=learned,
        recipe=recipe,
        backend="jax",
        focal_team="blue",
        current=PeriodicCheckpoint(Path("current.safetensors"), 2),
        prior=PeriodicCheckpoint(Path("prior.safetensors"), 1),
        current_index=2,
        seeds=(17,),
        episodes_per_seed=1,
        deterministic=True,
        wall_time_s=0.0,
        eval_id="cross-path",
    )
    assert play_priors_row["episode_role_map_ids"] == expected_ids

    from jaxborg.evaluation import jax_scripted_red
    from jaxborg.evaluation.jax_scripted_red import (
        JaxScriptedRedEpisode,
        evaluate_jax_scripted_reds,
    )

    monkeypatch.setattr(jax_scripted_red, "_git_commit", lambda: "test-commit")
    model = tmp_path / "model.safetensors"
    model.touch()
    scripted_rows = evaluate_jax_scripted_reds(
        model,
        base_variant=CIA_RESILIENCE,
        topology_paths=bank,
        reds=("fsm",),
        seeds=(17,),
        deterministic=True,
        recipe=recipe,
        policy_loader=lambda path, *, team, backend: LoadedMatchupPolicy(
            team,
            "jax",
            None,
            None,
            {"bundle_trainable": True},
        ),
        env_factory=lambda *args, **kwargs: object(),
        episode_runner=lambda *args, **kwargs: JaxScriptedRedEpisode(
            reward=0.0,
            cia=(0.0, 0.0, 0.0),
        ),
    )
    assert scripted_rows[0]["episode_role_map_ids"] == expected_ids
