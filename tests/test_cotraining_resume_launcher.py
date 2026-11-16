import json
import os
import subprocess
from pathlib import Path


def test_pbt_launcher_resumes_then_evaluates_even_with_inherited_train_only(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uv = fake_bin / "uv"
    uv.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "if '-c' in args:\n"
        "    print('ippo 1')\n"
        "else:\n"
        "    with open(os.environ['CALL_LOG'], 'a') as out:\n"
        "        out.write(json.dumps(args) + '\\n')\n"
        "    if '--tag' in args:\n"
        "        tag = args[args.index('--tag') + 1]\n"
        "        model = Path(os.environ['JAXBORG_EXP_DIR']) / 'ippo_jax' / tag / f'model_{tag}.safetensors'\n"
        "        model.parent.mkdir(parents=True, exist_ok=True)\n"
        "        model.touch()\n"
    )
    uv.chmod(0o755)
    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()
    call_log = tmp_path / "calls.jsonl"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CALL_LOG": str(call_log),
        "JAXBORG_REPO_DIR": str(repo),
        "JAXBORG_EXP_DIR": str(exp_dir),
        "JAXBORG_LOG_DIR": str(tmp_path / "logs"),
        "JAXBORG_MIN_FREE_GB": "0",
        "JAXBORG_RECIPES": "cotraining/pbt/example",
        "JAXBORG_SEEDS": "42 100",
        "JAXBORG_EVAL_ONLY": "0",
        "JAXBORG_TRAIN_ONLY": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "SLURM_JOB_ID": "12345",
    }
    result = subprocess.run(
        ["bash", str(repo / "scripts/temp/run_cotraining_separate_pbt.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in call_log.read_text().splitlines()]
    assert len(calls) == 4
    for train in calls[:2]:
        assert "scripts/train/algorithms/ippo_jax.py" in train
        assert train[train.index("--resume") + 1] == "latest"
    for evaluation in calls[2:]:
        assert "scripts/eval/run_after_training.py" in evaluation
        assert "--resume" in evaluation
        assert "--cross-seed-red" in evaluation
        assert "--recipe" in evaluation
