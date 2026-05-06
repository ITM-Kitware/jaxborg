"""Merge-gate automation for JAXborg/CybORG parity checks.

The gate intentionally treats training and transfer evaluation as subprocesses:
training may need Slurm GPUs, while transfer evaluation is forced onto CPU and
uses ``scripts/dev/transfer.py`` as the behavior source of truth.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scripts.dev.parity.bootstrap import EXP_DIR as DEFAULT_EXP_DIR
from scripts.dev.parity.bootstrap import ROOT
from scripts.dev.parity.stats import tost_equivalence

DEFAULT_SEEDS = (42, 100, 200)
CPU_ONLY_ENV = {
    "JAX_PLATFORMS": "cpu",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}
TRAIN_ENV_UNSET = ("JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES")

for key, value in CPU_ONLY_ENV.items():
    os.environ.setdefault(key, value)


@dataclass(frozen=True)
class PlannedCommand:
    name: str
    argv: list[str]
    cwd: Path
    env: dict[str, str] = field(default_factory=dict)
    unset_env: tuple[str, ...] = ()
    log_path: Path | None = None
    result_path: Path | None = None

    def display(self) -> str:
        parts: list[str] = []
        if self.unset_env:
            parts.append("env")
            for key in self.unset_env:
                parts.extend(["-u", key])
        parts.extend(f"{k}={v}" for k, v in sorted(self.env.items()))
        parts.extend(self.argv)
        return shlex.join(parts)


@dataclass(frozen=True)
class GateConfig:
    root: Path
    exp_dir: Path
    run_id: str
    recipe: str = "default"
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    train: bool = False
    checkpoints: tuple[Path, ...] = ()
    tag_prefix: str | None = None
    total_timesteps: int | None = None
    num_envs: int | None = None
    train_launcher: str = "srun"
    parallel_train: int = 1
    slurm_gres: str = "gpu:1"
    slurm_mem: str = "64G"
    slurm_time: str = "12:00:00"
    slurm_gpu_bind: str | None = None
    train_cuda_visible_devices: str | None = None
    eval_episodes: int = 100
    eval_seed: int = 0
    eval_workers: int = 10
    parallel_eval: int = 1
    no_scan: bool = False
    deterministic: bool = False
    tost_margin: float = 200.0
    tost_alpha: float = 0.05
    require_each_equivalent: bool = False
    run_fast_tests: bool = False
    run_slow_tests: bool = False
    dry_run: bool = False

    @property
    def run_dir(self) -> Path:
        return self.exp_dir / "parity_gate" / self.run_id

    @property
    def resolved_tag_prefix(self) -> str:
        return self.tag_prefix or self.run_id


def parse_int_set(spec: str) -> tuple[int, ...]:
    """Parse ``42,100,200`` or inclusive ranges like ``42-45``."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            out.update(range(int(start), int(end) + 1))
        else:
            out.add(int(part))
    if not out:
        raise ValueError("seed list is empty")
    return tuple(sorted(out))


def repo_root() -> Path:
    return ROOT


def default_exp_dir(root: Path) -> Path:
    return Path(os.environ.get("JAXBORG_EXP_DIR", DEFAULT_EXP_DIR)).resolve()


def default_run_id() -> str:
    return time.strftime("parity_%Y%m%d_%H%M%S")


def checkpoint_for_tag(exp_dir: Path, tag: str) -> Path:
    return exp_dir / "ippo_jax" / tag / f"model_{tag}.pkl"


def tag_for_seed(config: GateConfig, seed: int) -> str:
    return f"{config.resolved_tag_prefix}_{config.recipe}_seed{seed}"


def planned_training_checkpoints(config: GateConfig) -> list[Path]:
    return [checkpoint_for_tag(config.exp_dir, tag_for_seed(config, seed)) for seed in config.seeds]


def build_train_command(config: GateConfig, seed: int) -> PlannedCommand:
    tag = tag_for_seed(config, seed)
    base = [
        "uv",
        "run",
        "python",
        "-u",
        "scripts/train/algorithms/ippo_jax.py",
        "--recipe",
        config.recipe,
        "--seed",
        str(seed),
        "--tag",
        tag,
    ]
    if config.total_timesteps is not None:
        base.extend(["--total-timesteps", str(config.total_timesteps)])
    if config.num_envs is not None:
        base.extend(["--num-envs", str(config.num_envs)])

    env = {
        "JAXBORG_EXP_DIR": str(config.exp_dir),
        "PYTHONUNBUFFERED": "1",
    }
    if config.train_cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = config.train_cuda_visible_devices
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    log_path = config.run_dir / "logs" / f"train_{tag}.log"

    if config.train_launcher == "local":
        argv = base
    elif config.train_launcher == "srun":
        argv = [
            "srun",
            f"--gres={config.slurm_gres}",
            f"--mem={config.slurm_mem}",
            "--partition=community",
            f"--time={config.slurm_time}",
            *base,
        ]
        if config.slurm_gpu_bind:
            argv.insert(1, f"--gpu-bind={config.slurm_gpu_bind}")
    elif config.train_launcher == "sbatch":
        wrapped = shlex.join([f"{k}={v}" for k, v in env.items()] + base)
        argv = [
            "sbatch",
            "--wait",
            "--parsable",
            f"--gres={config.slurm_gres}",
            f"--mem={config.slurm_mem}",
            "--partition=community",
            f"--time={config.slurm_time}",
            f"--job-name={tag}",
            f"--output={log_path}",
            "--wrap",
            wrapped,
        ]
        if config.slurm_gpu_bind:
            argv.insert(3, f"--gpu-bind={config.slurm_gpu_bind}")
        env = {"PYTHONUNBUFFERED": "1"}
    else:
        raise ValueError(f"unknown train launcher: {config.train_launcher}")

    return PlannedCommand(
        name=f"train:{tag}",
        argv=argv,
        cwd=config.root,
        env=env,
        unset_env=TRAIN_ENV_UNSET,
        log_path=log_path,
        result_path=checkpoint_for_tag(config.exp_dir, tag),
    )


def build_eval_command(config: GateConfig, checkpoint: Path, index: int) -> PlannedCommand:
    label = checkpoint.parent.name or checkpoint.stem
    eval_dir = config.run_dir / "eval" / f"{index:02d}_{label}"
    result_path = eval_dir / "tost_result.json"
    argv = [
        "uv",
        "run",
        "python",
        "-u",
        "scripts/dev/transfer.py",
        "--checkpoint",
        str(checkpoint),
        "--episodes",
        str(config.eval_episodes),
        "--seed",
        str(config.eval_seed),
        "--tost-margin",
        str(config.tost_margin),
        "--tost-alpha",
        str(config.tost_alpha),
        "--tost-output",
        str(result_path),
    ]
    if config.no_scan:
        argv.append("--no-scan")
    if config.deterministic:
        argv.append("--deterministic")

    env = {
        **CPU_ONLY_ENV,
        "CUDA_VISIBLE_DEVICES": "",
        "JAXBORG_EXP_DIR": str(eval_dir),
        "JAXBORG_TRANSFER_WORKERS": str(config.eval_workers),
        "PYTHONUNBUFFERED": "1",
    }
    return PlannedCommand(
        name=f"eval:{label}",
        argv=argv,
        cwd=config.root,
        env=env,
        log_path=config.run_dir / "logs" / f"eval_{index:02d}_{label}.log",
        result_path=result_path,
    )


def build_test_commands(config: GateConfig) -> list[PlannedCommand]:
    commands: list[PlannedCommand] = []
    env = {
        **CPU_ONLY_ENV,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONUNBUFFERED": "1",
    }
    if config.run_fast_tests:
        commands.append(
            PlannedCommand(
                name="pytest:fast-suite",
                argv=["uv", "run", "pytest"],
                cwd=config.root,
                env=env,
                log_path=config.run_dir / "logs" / "pytest_fast_suite.log",
            )
        )
    if config.run_slow_tests:
        commands.append(
            PlannedCommand(
                name="pytest:slow-parity",
                argv=[
                    "uv",
                    "run",
                    "pytest",
                    "-p",
                    "no:xdist",
                    "-m",
                    "slow",
                    "tests/differential",
                    "tests/l3",
                ],
                cwd=config.root,
                env=env,
                log_path=config.run_dir / "logs" / "pytest_slow_parity.log",
            )
        )
    return commands


def run_logged_command(command: PlannedCommand) -> int:
    if command.log_path:
        command.log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in command.unset_env:
        env.pop(key, None)
    env.update(command.env)
    print(f"\n==> {command.name}")
    print(command.display())
    if command.log_path:
        print(f"log: {command.log_path}")

    log_context = command.log_path.open("w") if command.log_path else contextlib.nullcontext(None)
    with log_context as log_file:
        process = subprocess.Popen(
            command.argv,
            cwd=command.cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            if log_file is not None:
                log_file.write(line)
                log_file.flush()
        return process.wait()


def run_command_group(commands: list[PlannedCommand], parallel: int, *, dry_run: bool = False) -> None:
    if not commands:
        return
    parallel = max(1, parallel)
    if dry_run:
        for command in commands:
            print(f"[dry-run] {command.name}: {command.display()}")
        return
    if parallel == 1 or len(commands) == 1:
        for command in commands:
            rc = run_logged_command(command)
            if rc != 0:
                raise RuntimeError(f"{command.name} failed with exit code {rc}")
        return

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(run_logged_command, command): command for command in commands}
        for future in as_completed(futures):
            command = futures[future]
            rc = future.result()
            if rc != 0:
                raise RuntimeError(f"{command.name} failed with exit code {rc}")


def load_transfer_results(result_paths: list[Path]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in result_paths:
        if not path.exists():
            raise FileNotFoundError(f"missing transfer result: {path}")
        results.append(json.loads(path.read_text()))
    return results


def aggregate_transfer_results(
    results: list[dict[str, Any]],
    *,
    margin: float,
    alpha: float,
    paired: bool = False,
) -> dict[str, Any]:
    jax_rewards: list[float] = []
    cyborg_rewards: list[float] = []
    checkpoints: list[dict[str, Any]] = []
    for result in results:
        jax_rewards.extend(float(x) for x in result["jax_rewards"])
        cyborg_rewards.extend(float(x) for x in result["cyborg_rewards"])
        checkpoints.append(
            {
                "checkpoint": result.get("checkpoint", ""),
                "episodes": result.get("episodes", len(result["jax_rewards"])),
                "seed": result.get("seed"),
                "equivalent": result.get("equivalent"),
                "mean_diff": result.get("mean_diff"),
                "ci_lower": result.get("ci_lower"),
                "ci_upper": result.get("ci_upper"),
                "p_upper": result.get("p_upper"),
                "p_lower": result.get("p_lower"),
            }
        )

    pooled = tost_equivalence(jax_rewards, cyborg_rewards, margin=margin, alpha=alpha, paired=paired)
    pooled.update(
        {
            "jax_mean": sum(jax_rewards) / len(jax_rewards),
            "cyborg_mean": sum(cyborg_rewards) / len(cyborg_rewards),
            "jax_n": len(jax_rewards),
            "cyborg_n": len(cyborg_rewards),
            "checkpoints": checkpoints,
        }
    )
    return pooled


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def write_summary(config: GateConfig, summary: dict[str, Any]) -> Path:
    path = config.run_dir / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")
    return path


def run_gate(config: GateConfig) -> dict[str, Any]:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    (config.run_dir / "logs").mkdir(parents=True, exist_ok=True)

    test_commands = build_test_commands(config)
    train_commands = [build_train_command(config, seed) for seed in config.seeds] if config.train else []
    planned_checkpoints = planned_training_checkpoints(config) if config.train else []
    checkpoints = [*planned_checkpoints, *config.checkpoints]
    if not checkpoints:
        raise ValueError("provide --train or at least one --checkpoint")

    print(f"Parity gate run: {config.run_id}")
    print(f"run dir: {config.run_dir}")
    print(f"recipe: {config.recipe}")
    print(f"checkpoints: {len(checkpoints)}")

    run_command_group(test_commands, parallel=1, dry_run=config.dry_run)
    run_command_group(train_commands, parallel=config.parallel_train, dry_run=config.dry_run)

    if not config.dry_run:
        missing = [path for path in planned_checkpoints if not path.exists()]
        if missing:
            raise FileNotFoundError(f"training completed but checkpoints are missing: {missing}")

    eval_commands = [build_eval_command(config, checkpoint, i) for i, checkpoint in enumerate(checkpoints)]
    run_command_group(eval_commands, parallel=config.parallel_eval, dry_run=config.dry_run)

    summary: dict[str, Any] = {
        "run_id": config.run_id,
        "recipe": config.recipe,
        "seeds": list(config.seeds),
        "run_dir": config.run_dir,
        "train": config.train,
        "checkpoints": checkpoints,
        "commands": {
            "tests": [command.display() for command in test_commands],
            "training": [command.display() for command in train_commands],
            "evaluation": [command.display() for command in eval_commands],
        },
        "dry_run": config.dry_run,
    }

    if config.dry_run:
        summary["passed"] = None
        summary_path = write_summary(config, summary)
        print(f"wrote dry-run summary: {summary_path}")
        return summary

    transfer_results = load_transfer_results([command.result_path for command in eval_commands if command.result_path])
    pooled = aggregate_transfer_results(
        transfer_results,
        margin=config.tost_margin,
        alpha=config.tost_alpha,
        paired=False,
    )
    each_ok = all(bool(result.get("equivalent")) for result in transfer_results)
    passed = bool(pooled["equivalent"]) and (each_ok or not config.require_each_equivalent)
    summary.update(
        {
            "pooled_tost": pooled,
            "require_each_equivalent": config.require_each_equivalent,
            "each_checkpoint_equivalent": each_ok,
            "passed": passed,
        }
    )
    summary_path = write_summary(config, summary)

    verdict = "PASS" if passed else "FAIL"
    print("\n" + "=" * 70)
    print(f"PARITY GATE: {verdict}")
    print("=" * 70)
    print(f"pooled mean diff: {pooled['mean_diff']:+.2f}")
    print(f"{int((1 - config.tost_alpha) * 100)}% CI: [{pooled['ci_lower']:+.2f}, {pooled['ci_upper']:+.2f}]")
    print(f"p_upper={pooled['p_upper']:.4f} p_lower={pooled['p_lower']:.4f} margin=+/-{pooled['margin']:.1f}")
    print(f"summary: {summary_path}")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate JAXborg parity before merging")
    parser.add_argument("--recipe", default="default")
    parser.add_argument("--seeds", default="42,100,200", help="Training seeds, e.g. 42,100,200 or 42-44")
    parser.add_argument("--train", action="store_true", help="Train JAX policies before evaluation")
    parser.add_argument("--checkpoint", action="append", default=[], help="Existing model_<tag>.pkl to evaluate")
    parser.add_argument("--tag-prefix", default=None, help="Training tag prefix (default: run id)")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--exp-dir", default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--train-launcher", choices=("srun", "sbatch", "local"), default="srun")
    parser.add_argument("--parallel-train", type=int, default=1)
    parser.add_argument("--slurm-gres", default="gpu:1")
    parser.add_argument("--slurm-mem", default="64G")
    parser.add_argument("--slurm-time", default="12:00:00")
    parser.add_argument("--slurm-gpu-bind", default=None, help="Optional Slurm GPU binding, e.g. map_gpu:1")
    parser.add_argument(
        "--train-cuda-visible-devices",
        default=None,
        help="Optional explicit CUDA_VISIBLE_DEVICES for training jobs, e.g. 1",
    )
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--eval-workers", type=int, default=10)
    parser.add_argument("--parallel-eval", type=int, default=1)
    parser.add_argument("--no-scan", action="store_true")
    parser.add_argument("--deterministic", action="store_true", help="Debug only: use argmax instead of sampling")
    parser.add_argument("--tost-margin", type=float, default=200.0)
    parser.add_argument("--tost-alpha", type=float, default=0.05)
    parser.add_argument("--require-each-equivalent", action="store_true")
    parser.add_argument("--run-fast-tests", action="store_true")
    parser.add_argument("--run-slow-tests", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> GateConfig:
    root = repo_root()
    exp_dir = Path(args.exp_dir).resolve() if args.exp_dir else default_exp_dir(root)
    checkpoints = tuple(Path(p).resolve() for p in args.checkpoint)
    run_id = args.run_id or default_run_id()
    return GateConfig(
        root=root,
        exp_dir=exp_dir,
        run_id=run_id,
        recipe=args.recipe,
        seeds=parse_int_set(args.seeds),
        train=args.train,
        checkpoints=checkpoints,
        tag_prefix=args.tag_prefix,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        train_launcher=args.train_launcher,
        parallel_train=args.parallel_train,
        slurm_gres=args.slurm_gres,
        slurm_mem=args.slurm_mem,
        slurm_time=args.slurm_time,
        slurm_gpu_bind=args.slurm_gpu_bind,
        train_cuda_visible_devices=args.train_cuda_visible_devices,
        eval_episodes=args.eval_episodes,
        eval_seed=args.eval_seed,
        eval_workers=args.eval_workers,
        parallel_eval=args.parallel_eval,
        no_scan=args.no_scan,
        deterministic=args.deterministic,
        tost_margin=args.tost_margin,
        tost_alpha=args.tost_alpha,
        require_each_equivalent=args.require_each_equivalent,
        run_fast_tests=args.run_fast_tests,
        run_slow_tests=args.run_slow_tests,
        dry_run=args.dry_run,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        summary = run_gate(config_from_args(parse_args(argv)))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if summary.get("passed") is False:
        return 1
    return 0
