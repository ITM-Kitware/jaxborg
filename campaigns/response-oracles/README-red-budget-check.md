# Late IPPO Red response budget check

This bounded experiment asks whether one fresh Red response can improve over
original Red against frozen original Blue from IPPO co-training seed 42 at
49,968,000 steps, and whether allowing 20M response steps improves the response
selected by 10M. It is one training seed on topology seed 0, not a convergence
certificate or an estimate of training-population uncertainty.

Run one continuous fresh shared-IPPO Red trajectory with 20,000,000 requested
steps, completing 416 updates / 19,968,000 actual steps. Preserve the historical
trainer, architecture, observation contract, stock CC4 reward, gamma .99,
GAE lambda .95, learning rate .0003 with annealing disabled, 96 parallel
environments, and 500-step episodes. Only checkpoint cadence changes to retain
updates 104, 208, 312 and 416: 4.992M, 9.984M, 14.976M and 19.968M steps.
Original Red weights are never used to initialize the fresh challenger.

The YAML fixes training seed 44001 and separate validation, final-test, smoke
and bootstrap roots before training. A seed ledger records the archive search
and its limitations. Evaluate all four checkpoints and original Red on 100
common validation episodes. Independently select the minimum mean Blue return
among original plus checkpoints through update 208 and through update 416.
Exact ties favor original, then earlier checkpoints. Persist both identities,
checkpoint hashes and all validation scores before opening the test cohort.

Evaluate original and each distinct selected policy on the same 600 new episode
roots. Reused matchup labels are not independent observations. The primary
comparison is the selected 20M response's Red gain over original Red. The
secondary budget comparison is the selected 20M response's Red gain over the
selected 10M response. Compute each from paired episode differences; report
pointwise 95% percentile bootstrap intervals with 10,000 resamples. A positive
primary lower bound supports a profitable deviation at the late IPPO pair.
A positive secondary lower bound supports useful additional search for this
trajectory. A null or original-policy fallback does not establish convergence,
a best response, equilibrium, or a global upper bound on exploitability.

Reserve one GPU through Slurm community, with 48 GB host memory, 8 CPUs and a
four-hour hard limit including the actual GPU smoke, training and evaluation.
Historical A6000 timings suggest about three GPU hours. Keep CUDA_VISIBLE_DEVICES
as assigned by Slurm and preserve the pinned checkout until the job finishes.
Use the existing shared experiment store and the original compatible GPU runtime.
Never overwrite old campaigns or silently restart an incomplete training run.

## Reproduce the protocol

Use the clean retained checkout and its existing `.venv`. Set `PYTHONPATH`
explicitly because the shared runtime's editable installation points to the
older launch checkout. The full commit recorded as `source_revision` in the
prepared manifest is the authoritative code identity. Retain that commit in
the study's Git bundle and record the bundle filename and checksum in launch
provenance; the branch name alone is insufficient to reconstruct the code.

```bash
cd /home/local/KHQ/paul.elliott/src/cyber/jaxborg/stage-c-red-budget-check
export JAXBORG_EXP_DIR=/home/local/KHQ/paul.elliott/jaxborg-exp/shared
export JAXBORG_MLFLOW_EXPERIMENT=stage-c-red-budget-49968000
export PYTHONPATH="$PWD/src:$PWD"
BUDGET_MANIFEST=/home/local/KHQ/paul.elliott/src/cyber/plans/jax/cc4/equilibrium-experiments/results/stage-c/red-budget-49968000-20261021/manifest.json

JAX_PLATFORMS=cpu .venv/bin/python scripts/experiments/response_budget.py prepare \
  --campaign campaigns/response-oracles/ippo-seed42-red-budget-49968000.yaml \
  --defender ippo-49968000
JAX_PLATFORMS=cpu .venv/bin/python scripts/experiments/response_budget.py dry-run \
  --manifest "$BUDGET_MANIFEST"
bash scripts/experiments/submit_response_budget.sh pilot "$BUDGET_MANIFEST"
```

Run `prepare` only once for this report directory. The `pilot` submission runs
the actual GPU smoke followed by the one training attempt and prescribed
evaluations under the four-hour Slurm cap. It verifies the clean pinned source
and dependency environment inside the allocation. Preserve Slurm's assigned
device visibility. Do not run training or rollout evaluation on the login node.

After the evaluations finish, regenerate the report from archived episode
records without training or new rollouts:

```bash
JAX_PLATFORMS=cpu .venv/bin/python scripts/experiments/response_budget.py aggregate \
  --manifest "$BUDGET_MANIFEST"
```

The manifest export must match its immutable
`runs:/<protocol_owner_run_id>/protocol/manifest.json` artifact. It records the
campaign YAML reference, source bundle and sidecar hashes, topology identity,
training and evaluation recipe paths and hashes, prescribed seed cohorts,
dependency identity and actual update budgets. The protocol owner also archives
the recipes under `protocol/recipe-red-seed44001.yaml` and
`protocol/recipe-evaluation.yaml`. Preserve these artifacts together with the
seed ledger, its audit coverage and limitations, runtime details, source Git
bundle, launch record, Slurm job identifier and Slurm output/error logs.

The training run owns the complete raw `logs/metrics.jsonl`, source archive,
effective configuration, dependency records and all four checkpoint bundles
with sidecars. Intermediate bundle names contain their actual step counts;
the last bundle is `model_<tag>.safetensors`. The report's `summary.json`
records each checkpoint's canonical reference and hashes, both locked
selections, all validation references and the distinct confirmation references.
Each referenced evaluation owns `evaluations/result.json`, including the
ordered episode seeds and raw Blue/Red returns. Keep these canonical records;
the exported summary and CSV files alone do not preserve the entire experiment.

Fresh reruns and independent reevaluation of saved policies are supported.
Use a new campaign/report location for a fresh rerun and document any seed or
runtime changes. Reevaluation uses `scripts/eval/eval_matchup.py` inside a
community allocation with the archived evaluation recipe, explicit original
Blue and selected Red bundle references, and an explicit episode cohort.
Keep new evaluation records separate from the frozen selection and original
confirmation. Existing state can recover completed stages, but the portable
policy bundles do not support full midrun restoration of the optimizer, random
keys, environment state or reward normalizer. An interrupted training attempt
must not silently become a new fresh attempt. Bitwise replay is not guaranteed
on a different GPU or runtime.
