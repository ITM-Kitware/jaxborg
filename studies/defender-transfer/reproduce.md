# Reproduce the defender-transfer diagnostic

## Rebuild recorded results

Use the source and selection pinned in the
[learning check](checks/lstm-learning-20261107/README.md#reproduce-without-rollouts)
or [matched screen](benchmark.md#execute-and-reconstruct). Those pages give the
commands for the reported results. The general transfer analyzer accepts:

```bash
python studies/defender-transfer/analyze.py \
  --manifest /path/to/study/inputs/manifest.json \
  --root /path/to/study/evidence \
  --selection /path/to/study/selection.json \
  --output /path/to/new-report
```

## Run a new evaluation

Use a clean checkout of the intended revision and a new output root. Follow
[project setup](../../docs/setup.md) and the machine's AGENTS.md. The archive
defaults to `/data/shared/jaxborg/jaxborg-separate`; pass another archive as the
fourth launcher argument if needed. Setup installs the locked dependencies and
the CUDA plugins listed in `cuda-runtime.txt`.

```bash
UV_BIN="$HOME/.local/bin/uv" studies/defender-transfer/setup.sh
ROOT=/path/to/new-study
studies/defender-transfer/launch.sh submit "$ROOT" smoke
```

After checking the smoke receipts, run the final stage, then the early stage:

```bash
studies/defender-transfer/launch.sh submit "$ROOT" final
studies/defender-transfer/launch.sh submit "$ROOT" early
```

`selection.json` identifies the successful evaluations used by analysis. Smoke
results are excluded. Changing the source or selection creates a new execution
or analysis, preserving the recorded comparison.
