# Studies

Keep each study's question, methods, and analysis in `studies/<question>/`.
Keep generated data, checkpoints, logs, figures, reports, and other run
evidence outside the checkout.

## Choose a workflow

- [Capture a checkpoint evaluation](../docs/evaluation-capture.md): evaluate a
  Torch policy in CybORG and retain inputs, source, episodes, and execution status.
- [Record joint training and evaluation](../docs/joint-execution-records.md):
  retain JAX training updates, owned checkpoints, matchups and trajectories.
- [Compare Red update schedules](red-update-schedules/README.md): declare fresh
  baseline, quarter-rate and adaptive runs and reconstruct explicit comparisons.
- [Run the payoff/blocking example](reward-offset/README.md): compose four
  conditions and rebuild a comparison from explicitly selected evidence.
- [Import existing evidence](../docs/local-evidence-producers.md): read a retained
  frozen-defender bundle or co-training SQLite store into version 1 records.
- [Browse runs and download evidence](../docs/evidence-retrieval.md): use native
  MLflow views/APIs, verify downloaded evidence and reconstruct selected reports.
- [Rebuild the frozen-defender report](frozen-defender-response/README.md): analyze
  that study's saved evidence with its scientific checks and explicit selection.

Shared training launch and logging conventions are tracked in
[#25](https://github.com/ITM-Kitware/jaxborg/issues/25), with broader integration in
[#28](https://github.com/ITM-Kitware/jaxborg/issues/28).

## Study files and settings

```text
studies/<question>/
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

A study defines a scientific question, methods, and analysis. Its protocol
defines expected trials, their conditions, replicate identities, and selection
rule. Attempts are executions of trials; retries are new attempts linked to an
earlier attempt. Select attempts and evaluations explicitly when collating them;
retries and evaluation episodes do not create independent training replicates.
An MLflow experiment is a tracking container independent of a study. Retain
fully resolved runtime settings, source/environment identity, inputs,
measurements, and outputs with each execution. These records describe what ran
and are not editable presets.

## Opponent control and learning

Scripted Red chooses actions inside its environment, including its memory
updates. Network-controlled Red supplies policy actions through the caller.
Freezing a network controls whether its parameters update; it still acts.
The existing `make_joint_train` entrypoint supports training one team against
a frozen policy or training both teams.

Environment factories share variant settings while preserving these execution
differences. A study can supply a custom environment builder to the supported
training and evaluation paths. A common opponent interface and any later
rollout unification are separate work under [#52](https://github.com/ITM-Kitware/jaxborg/issues/52).

## Analysis and reporting

Analysis takes explicit saved inputs and writes to a new output directory.
Retain plotted values and settings with figures. Use checks, statistics, and
presentation appropriate to the question; reports should explain results,
methods, and interpretation limits, with routine execution details in linked
records.

Saved selections bind exact analyzer source and repository-relative paths.
Reconstruct a saved report with its retained source; a different analyzer
identity cannot replace it.

The frozen-defender example checks a historical result against a retained
reference summary, using an input bundle supplied separately by its author.
Other studies need not copy its validator or report format.
