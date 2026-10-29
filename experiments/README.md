# Experiments

Keep each study's question, methods, and analysis in `experiments/<question>/`.
Keep generated data, checkpoints, logs, figures, and reports outside Git.

## Choose a workflow

- [Capture a checkpoint evaluation](../docs/evaluation-capture.md): evaluate a
  Torch policy in CybORG and retain inputs, source, episodes, and execution status.
- [Record joint training and evaluation](../docs/joint-execution-records.md):
  retain JAX training updates, owned checkpoints, matchups and trajectories.
- [Import existing evidence](../docs/local-evidence-producers.md): read a retained
  frozen-defender bundle or co-training SQLite store into version 1 records.
- [Retrieve retained evidence](../docs/evidence-retrieval.md): find an exact MLflow
  publication and verify its downloaded inputs, records, and saved selection.
- [Rebuild the frozen-defender report](frozen-defender-response/README.md): analyze
  that study's saved evidence with its scientific checks and explicit selection.

Shared training launch and logging conventions are tracked in
[#25](https://github.com/ITM-Kitware/jaxborg/issues/25), with broader integration in
[#28](https://github.com/ITM-Kitware/jaxborg/issues/28).

## Study files and settings

```text
experiments/<question>/
  README.md          question, method, commands, input/result references
  analyze.py         study-specific analysis, when needed
```

Start with the question, conditions, budget, seed/evaluation procedure, and input
and result references. Add analysis code and settings as needed; no fixed file
set is required. Use the repository's Python environment and dependencies.

Executable training and evaluation settings use the existing recipe schema and
loader, including study-local recipes passed by path, plus supported command
overrides. Keep reusable presets in `recipes/` and reusable simulator, policy,
trainer, and evaluator behavior in shared modules. Seed or budget changes need
no new directory or commit; maintained code and presets follow ordinary review.

A protocol declares expected scientific trial cells and their selection rule.
Attempts record executions and retry relationships. Select attempts and
evaluations explicitly when collating them; retries and evaluation episodes do
not create independent training replicates. Retain fully resolved runtime
settings, source/environment identity, inputs, measurements, and outputs with
each execution. These records describe what ran and are not editable presets.

## Analysis and reporting

Analysis takes explicit saved inputs and writes to a new output directory.
Retain plotted values and settings with figures. Use checks, statistics, and
presentation appropriate to the question; reports should explain results,
methods, and interpretation limits, with routine execution details in linked
records.

The frozen-defender example checks a historical result against a retained
reference summary, using an input bundle supplied separately by its author.
Other studies need not copy its validator or report format.
