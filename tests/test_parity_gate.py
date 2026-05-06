from pathlib import Path

from scripts.dev.parity.gate import (
    GateConfig,
    aggregate_transfer_results,
    build_eval_command,
    build_test_commands,
    build_train_command,
    parse_int_set,
    planned_training_checkpoints,
    run_gate,
)


def _config(tmp_path: Path, **kwargs) -> GateConfig:
    values = {
        "root": tmp_path,
        "exp_dir": tmp_path / "exp",
        "run_id": "parity_test",
        "recipe": "default",
        "seeds": (42, 100, 200),
        "dry_run": True,
    }
    values.update(kwargs)
    return GateConfig(**values)


def test_parse_int_set_accepts_ranges_and_lists():
    assert parse_int_set("42,100-102,42") == (42, 100, 101, 102)


def test_srun_train_command_uses_one_gpu_and_expected_checkpoint(tmp_path):
    config = _config(tmp_path, train=True, tag_prefix="gate", total_timesteps=123, num_envs=4)

    command = build_train_command(config, seed=42)

    assert command.argv[:4] == ["srun", "--gres=gpu:1", "--mem=64G", "--partition=community"]
    assert "--total-timesteps" in command.argv
    assert "--num-envs" in command.argv
    assert command.unset_env == ("JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES")
    assert command.env["JAXBORG_EXP_DIR"] == str(config.exp_dir)
    assert command.result_path == config.exp_dir / "ippo_jax" / "gate_default_seed42" / "model_gate_default_seed42.pkl"


def test_train_command_accepts_slurm_gpu_bind(tmp_path):
    config = _config(tmp_path, train=True, train_launcher="sbatch", slurm_gpu_bind="map_gpu:1")

    command = build_train_command(config, seed=42)

    assert "--gpu-bind=map_gpu:1" in command.argv
    assert command.argv[command.argv.index("--gpu-bind=map_gpu:1") + 1] == "--gres=gpu:1"


def test_eval_command_is_cpu_only_stochastic_unmatched_by_default(tmp_path):
    checkpoint = tmp_path / "exp" / "ippo_jax" / "tag" / "model_tag.pkl"
    config = _config(tmp_path, eval_episodes=7, eval_workers=3)

    command = build_eval_command(config, checkpoint, 0)

    assert command.env["JAX_PLATFORMS"] == "cpu"
    assert command.env["CUDA_VISIBLE_DEVICES"] == ""
    assert command.env["JAXBORG_TRANSFER_WORKERS"] == "3"
    assert "--deterministic" not in command.argv
    assert "--matched" not in command.argv
    assert command.argv[command.argv.index("--episodes") + 1] == "7"
    assert command.result_path == config.run_dir / "eval" / "00_tag" / "tost_result.json"


def test_fast_preflight_uses_repo_fast_suite(tmp_path):
    config = _config(tmp_path, run_fast_tests=True)

    command = build_test_commands(config)[0]

    assert command.name == "pytest:fast-suite"
    assert command.argv == ["uv", "run", "pytest"]
    assert command.env["JAX_PLATFORMS"] == "cpu"
    assert command.env["CUDA_VISIBLE_DEVICES"] == ""
    assert command.log_path == config.run_dir / "logs" / "pytest_fast_suite.log"


def test_planned_training_checkpoints_follow_tag_prefix(tmp_path):
    config = _config(tmp_path, train=True, tag_prefix="merge_gate", seeds=(1, 2))

    assert planned_training_checkpoints(config) == [
        config.exp_dir / "ippo_jax" / "merge_gate_default_seed1" / "model_merge_gate_default_seed1.pkl",
        config.exp_dir / "ippo_jax" / "merge_gate_default_seed2" / "model_merge_gate_default_seed2.pkl",
    ]


def test_aggregate_transfer_results_pools_rewards():
    results = [
        {
            "checkpoint": "a.pkl",
            "episodes": 3,
            "seed": 0,
            "equivalent": True,
            "mean_diff": 0.0,
            "jax_rewards": [100.0, 101.0, 99.0],
            "cyborg_rewards": [100.5, 100.0, 99.5],
        },
        {
            "checkpoint": "b.pkl",
            "episodes": 3,
            "seed": 0,
            "equivalent": True,
            "mean_diff": 0.0,
            "jax_rewards": [98.0, 102.0, 100.0],
            "cyborg_rewards": [98.5, 101.0, 100.5],
        },
    ]

    pooled = aggregate_transfer_results(results, margin=5.0, alpha=0.05)

    assert pooled["jax_n"] == 6
    assert pooled["cyborg_n"] == 6
    assert len(pooled["checkpoints"]) == 2
    assert pooled["equivalent"] is True


def test_dry_run_writes_summary_without_executing_commands(tmp_path):
    checkpoint = tmp_path / "missing_model.pkl"
    config = _config(tmp_path, checkpoints=(checkpoint,), dry_run=True)

    summary = run_gate(config)

    assert summary["passed"] is None
    assert (config.run_dir / "summary.json").exists()
    assert "scripts/dev/transfer.py" in summary["commands"]["evaluation"][0]
