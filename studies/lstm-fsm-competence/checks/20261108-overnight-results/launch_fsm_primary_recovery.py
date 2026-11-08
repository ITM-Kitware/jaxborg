"""Retain and validate the six-cell recovery plan; submit only with --submit."""

import argparse
import hashlib
import subprocess
from pathlib import Path

REMOTE = r"""
import fcntl,hashlib,json,os,pathlib,subprocess
home=pathlib.Path.home()
source=home/'src/cyber/jaxborg/lstm-fsm-resumable-v3-20261107'
root=home/'jaxborg-exp/shared/studies/lstm-fsm-competence-20261107-v3'
transfer=home/'jaxborg-exp/shared/studies/defender-transfer-20261107-v4/inputs/manifest.json'
recovery=root/'recoveries/primary-20261108'
revision='0aef148bbebb45818e71b20e9b5a52be031f93f1'
lock=(root/'.primary-recovery-submission.lock').open('a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
helper=root/'orchestration/recover_fsm_primary.py'
data=payload['helper'].encode()
assert hashlib.sha256(data).hexdigest()==payload['helper_sha256']
if helper.exists():
    if helper.read_bytes()!=data:raise ValueError('Retained recovery helper differs')
else:
    with helper.open('xb') as stream:stream.write(data)
args=[str(source/'.venv/bin/python'),str(helper),'--source-root',str(source),'--study-root',str(root),'--recovery-root',str(recovery),'--transfer-manifest',str(transfer),'--expected-revision',revision]
env=dict(os.environ,JAX_PLATFORMS='cpu',JAXBORG_EVAL_BATCH_SIZE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
if not (recovery/'plan.json').exists():
    subprocess.run(args+['prepare'],cwd=source,env=env,check=True)
plan=json.loads((recovery/'plan.json').read_text())
if plan['orchestrator_sha256']!=payload['helper_sha256'] or len(plan['missing'])!=6 or len(plan['existing'])!=14:
    raise ValueError('Prepared plan differs from reviewed six-cell recovery')
print(json.dumps({'recovery_root':str(recovery),'verified_existing':len(plan['existing']),'missing':len(plan['missing']),'helper_sha256':payload['helper_sha256']}),flush=True)
if payload['submit']:
    marker=recovery/'submitted.job'
    if marker.exists():raise RuntimeError('Recovery already submitted; inspect existing job')
    script=recovery/'launch.sh'
    text='''#!/usr/bin/env bash
set -euo pipefail
SOURCE_ROOT=${1:?}
HELPER=${2:?}
shift 2
source "$HOME/jaxborg-exp/shared/store.env"
export MLFLOW_TRACKING_URI="sqlite:///$JAXBORG_EXP_DIR/mlflow.db"
export MLFLOW_EXPERIMENT_NAME=lstm-fsm-competence
export JAXBORG_MLFLOW_EXPERIMENT="$MLFLOW_EXPERIMENT_NAME"
export JAX_PLATFORMS=cpu JAXBORG_EVAL_BATCH_SIZE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_COMPILATION_CACHE_DIR="$HOME/jaxborg-exp/shared/studies/lstm-fsm-competence-20261107/cpu-compilation-cache"
cd "$SOURCE_ROOT"
exec "$SOURCE_ROOT/.venv/bin/python" "$HELPER" "$@" run
'''
    if script.exists():
        if script.read_text()!=text:raise ValueError('Recovery launcher differs')
    else:script.write_text(text)
    command=['sbatch','--parsable','--partition=community','--cpus-per-task=8','--mem=64G','--time=360','--job-name=lstm-primary-recovery',f'--chdir={source}',f'--output={recovery}/%j.out',f'--error={recovery}/%j.err',str(script),str(source),str(helper),*args[2:]]
    job=subprocess.check_output(command,cwd=source,env=env,text=True).strip()
    if not job.isdigit():raise ValueError('Unexpected Slurm job identity')
    with marker.open('x') as stream:stream.write(job+'\n')
    submission={'job_id':job,'command':command,'helper_sha256':payload['helper_sha256'],
                'launch_sha256':hashlib.sha256(text.encode()).hexdigest()}
    with (recovery/'submission.json').open('x') as stream:json.dump(submission,stream,indent=2)
    print('SUBMITTED',job,flush=True)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).with_name("recover_fsm_primary.py").read_text()
    payload = {"helper": source, "helper_sha256": hashlib.sha256(source.encode()).hexdigest(), "submit": args.submit}
    script = "payload=" + repr(payload) + "\n" + REMOTE
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "itm", "python3", "-"],
        input=script,
        text=True,
        check=True,
    )


if __name__ == "__main__":
    main()
