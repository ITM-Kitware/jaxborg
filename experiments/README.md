# Experiments

An experiment directory gives a scientific question, its methods, and its analysis
a maintained home alongside the shared code. Repeated runs keep their actual
settings and generated files in artifact storage outside the checkout.

The first example is [frozen-defender response](frozen-defender-response/README.md).
It reproduces a completed study from saved measurements. Shared training launch
and logging conventions are tracked separately in
[#25](https://github.com/ITM-Kitware/jaxborg/issues/25); broader integration is
tracked in [#28](https://github.com/ITM-Kitware/jaxborg/issues/28).

## Choose a workflow

- [Capture a checkpoint evaluation](../docs/evaluation-capture.md): run a Torch
  policy in CybORG and retain inputs, source, episode results, and execution status.
- [Import existing evidence](../docs/local-evidence-producers.md): read a retained
  frozen-defender bundle or co-training SQLite store into version 1 records.
- [Retrieve retained evidence](../docs/evidence-retrieval.md): discover an exact
  MLflow publication and verify its downloaded inputs, records and saved selection.
- [Rebuild the frozen-defender report](frozen-defender-response/README.md): analyze
  that study's saved evidence with its scientific checks and explicit selection.

Capture and import both use the shared record contract. Select attempts and
evaluations explicitly when collating them; retries do not create new replicates.

## What to check in

```text
experiments/<question>/
  README.md          question, method, commands, input/result references
  analyze.py         analysis specific to this question, when needed
```

This is a convention, not a required set of files. Start with a README describing
the question, compared conditions, budget, seed/evaluation procedure, and where to
find the inputs and results. Add study-specific analysis code and analysis
settings when useful. Executable training and evaluation settings use the existing
recipe schema and loader, including recipes kept within a study directory and
passed by path. Keep reusable presets in `recipes/` and reusable simulator,
policy, trainer, and evaluator behavior in the shared implementation.

A protocol declares expected trial cells and an explicit selection rule.
Attempts record retry relationships. Neither defines another executable
configuration or launcher.

Use the repository's `pyproject.toml`, `uv.lock`, and Python selection. NumPy and
Matplotlib are shared dependencies. Generated data, checkpoints, logs, tables,
figures, and reports belong outside Git.

## Running another trial

Choose an existing recipe, then supply the trial's settings through the command's
supported overrides. Study code may reference recipes and those overrides. Changing a
seed or budget does not require a new directory or commit. Changing maintained
code or an intentionally versioned preset follows the ordinary review process.

Each execution should retain its fully resolved settings, including defaults and
overrides, with the code/environment identity, input references, measurements,
and outputs. The saved configuration records what ran; it is not another editable
copy of a preset. The frozen-defender example retains this evidence for its
analysis command. The checkpoint evaluation workflow above captures it for Torch
policies evaluated in CybORG. Live training capture and JAX checkpoint evaluation
capture remain follow-up work.

For example, population-based training would need shared code for population
updates and study settings for its budget and replacement rules. Its runs would
also retain checkpoint ancestry and actual parameter changes over time. Creating
an experiment directory alone does not add that training capability.

## Analysis and reporting

An analysis should take explicit saved inputs and write to a new output directory.
Keep the plotted values and settings with its figures so another person can
repeat the calculation. Choose checks, statistics, and presentation for the
question being studied. A new study need not copy the frozen-defender validator
or emit the same report files.

The frozen-defender script checks a historical result against a retained reference
summary. It is useful for reviewing that study and demonstrating the file and
configuration conventions; it is not a general reporting API. Its input bundle
is currently supplied separately by the study author.

Reports focus on results, methods, and caveats that affect interpretation. Routine
execution and verification details belong in the linked execution record.
