#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == allocated ]]; then
    [[ $# == 3 ]] || exit 1
    : "${SLURM_JOB_ID:?Run inside a Slurm allocation}"
    : "${JAXBORG_LAUNCH_RECORD:?Pin source and environment before submission}"
    [[ ${SLURM_JOB_PARTITION:-} == community ]] || { echo "Use community" >&2; exit 1; }
    ROOT="$(git rev-parse --show-toplevel)"
    cd "$ROOT"
    export PYTHONPATH="$ROOT/src:$ROOT"
    unset JAX_PLATFORMS
    source scripts/jax_env.sh
    "$ROOT/.venv/bin/python" -m jaxborg.launch verify --gpu "$JAXBORG_LAUNCH_RECORD"
    "$ROOT/.venv/bin/python" - "$ROOT" "$3" "$2" <<'PY'
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1]).resolve()
manifest_path = Path(sys.argv[2]).resolve()
manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(manifest_bytes)
origins = {}
for name in ("jaxborg.tracking", "jaxborg.recipe", "scripts.train.algorithms.ippo_jax_joint"):
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"Cannot locate required project module: {name}")
    origin = Path(spec.origin).resolve()
    if not origin.is_relative_to(root):
        raise RuntimeError(f"Project module resolves outside the pinned checkout: {name}: {origin}")
    origins[name] = str(origin)

def path_record(path):
    path = Path(path)
    return {
        "path": str(path),
        "realpath": str(path.resolve()),
        "symlink_target": os.readlink(path) if path.is_symlink() else None,
    }

runtime_keys = (
    "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "JAX_PLATFORMS",
    "JAX_PLATFORM_NAME", "JAX_ENABLE_X64", "JAX_DEFAULT_MATMUL_PRECISION", "XLA_FLAGS",
    "JAX_ENABLE_COMPILATION_CACHE", "JAX_COMPILATION_CACHE_DIR",
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES", "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_ALLOCATOR", "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NVIDIA_TF32_OVERRIDE",
    "JAXBORG_EXPECTED_SHA", "JAXBORG_LAUNCH_RECORD", "JAXBORG_EXP_DIR",
    "JAXBORG_MLFLOW_EXPERIMENT", "JAXBORG_CAMPAIGN",
)
slurm_keys = (
    "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST",
    "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS", "SLURM_GPUS_ON_NODE",
    "SLURM_GPUS_PER_NODE", "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE",
    "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_NTASKS", "SLURM_JOB_NUM_NODES",
)
gpu_command = ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv"]
gpu = subprocess.run(gpu_command, capture_output=True, text=True, check=True)
record = {
    "schema_version": 1,
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "checkout": str(root),
    "source_revision": manifest["source_revision"],
    "action": sys.argv[3],
    "manifest": str(manifest_path),
    "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    "python_version": sys.version,
    "interpreter": path_record(sys.executable),
    "venv": path_record(root / ".venv"),
    "venv_python": path_record(root / ".venv/bin/python"),
    "sys_prefix": sys.prefix,
    "sys_base_prefix": sys.base_prefix,
    "sys_path": sys.path,
    "project_module_origins": origins,
    "environment": {key: os.environ.get(key) for key in runtime_keys},
    "slurm": {key: os.environ.get(key) for key in slurm_keys},
    "gpu_query": {
        "command": gpu_command,
        "stdout": gpu.stdout,
        "stderr": gpu.stderr,
        "scope": "Read-only device inventory; CUDA_VISIBLE_DEVICES records the Slurm-assigned visibility.",
    },
}
destination = Path(manifest["report_dir"]) / "provenance" / f"gpu-runtime-{os.environ['SLURM_JOB_ID']}.json"
destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open("x") as stream:
    stream.write(json.dumps(record, indent=2) + "\n")
print(f"GPU runtime provenance: {destination}", flush=True)
PY
    exec "$ROOT/.venv/bin/python" scripts/experiments/response_budget.py "$2" --manifest "$3"
fi
[[ $# == 2 && ( $1 == smoke || $1 == run || $1 == pilot ) ]] || {
    echo "Usage: $0 <smoke|run|pilot> /absolute/manifest.json" >&2; exit 1;
}
: "${JAXBORG_EXP_DIR:?Set the explicit absolute external experiment root}"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT"
export JAXBORG_EXPECTED_SHA="$(git rev-parse HEAD)"
export JAXBORG_LAUNCH_RECORD="$(JAX_PLATFORMS=cpu "$ROOT/.venv/bin/python" -m jaxborg.launch pin)"
read -r JAXBORG_CAMPAIGN MEMORY CPUS JAXBORG_MLFLOW_EXPERIMENT < <(
    JAX_PLATFORMS=cpu "$ROOT/.venv/bin/python" - "$2" "$1" <<'PY'
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("response_budget", "scripts/experiments/response_budget.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
m = module.load_protocol(sys.argv[1])
r = m["resources"]
if r["partition"] != "community" or r["gpus_per_job"] != 1 or r["time_limit"] != "04:00:00" or r["max_concurrent_jobs"] != 1:
    raise ValueError("budget diagnostic requires one community GPU and a four-hour cap")
if sys.argv[2] == "run":
    module.oracle.require_smoke(m, sys.argv[1])
print(m["campaign"], r["memory_gb"], r["cpus_per_task"], m["tracking"]["experiment"])
PY
)
export JAXBORG_CAMPAIGN JAXBORG_MLFLOW_EXPERIMENT
mkdir -p "$JAXBORG_EXP_DIR/launches/slurm"
exec sbatch --parsable --partition=community --gres=gpu:1 --mem="${MEMORY}G" --cpus-per-task="$CPUS" \
    --time=04:00:00 --job-name="$JAXBORG_CAMPAIGN" --chdir="$ROOT" \
    --output="$JAXBORG_EXP_DIR/launches/slurm/%j.out" --error="$JAXBORG_EXP_DIR/launches/slurm/%j.err" \
    "$ROOT/scripts/experiments/submit_response_budget.sh" allocated "$1" "$2"
