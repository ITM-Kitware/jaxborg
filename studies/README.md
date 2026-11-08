# Studies

A study's README is the entry point for its question, methods, current status,
and results. Keep study definitions and analysis code in this directory; keep
generated evidence outside the source checkout.

## Start here

- [Project setup](../docs/setup.md): environment, storage, and MLflow.
- [Shared training guide](../docs/training.md): recipes and execution.
- [Shared evaluation guide](../docs/evaluation.md): evaluators and outputs.

## Read an existing study

1. Open its README to understand the question, what was compared, and what ran.
2. Follow its result references to the report, selected runs, and measurements.
3. Use its analysis commands and retained source when checking a result.

### Studies and examples

These examples illustrate different research questions and ways to compose
existing code.

| Study | What it illustrates |
| --- | --- |
| [Red update schedules](red-update-schedules/README.md) | Comparing baseline, quarter-rate, and adaptive training recipes with explicit run selection and reconstructable tables. |
| [Joint payoff and blocking](reward-offset/README.md) | Composing a reward change and a blocking rule in a small four-condition execution example. |
| [Frozen-defender response](frozen-defender-response/README.md) | Rebuilding a completed study's report from saved measurements. Its reference input bundle must be obtained separately from the study author. |
| [Nash co-training retrospective](nash-co-training/README.md) | Replaying unilateral deviations and saved-policy cross-play from archived runs and selections. |

## Create a study

### Study files and settings

Start with a README in a directory named for the research question. Add analysis
code and settings as needed; the optional filenames below are examples.

```text
studies/<short-study-name>/
  README.md          research summary and links to evidence and commands
  reproduce.md       optional detailed execution and reproduction instructions
  analyze.py         optional study-specific analysis
  recipe.yaml        optional study-specific settings using the existing schema
```

### Write the research summary

Lead with the question and finding. Explain what the result means and the next
research decision. For unfinished work, say what remains unknown.

- Give the actual comparison, important controls, budget, independent seeds and
  evaluation cases. Link exact methods and configuration.
- Include only limitations that affect the conclusion.
- Keep useful ITM output paths and links to results, evidence and replay commands.

Omit routine scheduling, resource quotas, wall-time caps, device assignment,
generic safeguards and job chronology. Delete prose that repeats configuration;
do not move it into another document or script comments. Mention failures only
when they explain missing evidence or affect interpretation.

Aim for a few short paragraphs and an optional results table. Use `reproduce.md`
for commands and non-obvious prerequisites. Reproducibility comes from retained
code, inputs and results, not narration of execution settings.

### Study settings

- Keep settings specific to the study beside its README. See
  [training recipes](../docs/training.md#recipes-and-teams) for the shared format
  and reusable presets.
- Distinguish planned settings from the resolved settings retained with each
  attempt. Defaults and command overrides can change what actually runs.

### Where results live

- **Source checkout:** the study README, maintained analysis code, and settings.
- **External results storage:** generated data, checkpoints, logs, tables,
  figures, and reports. The study README should tell a reader how to find them.
- **MLflow:** a place to browse runs and retrieve their retained evidence. Its
  experiment name is a tracking destination, independent of the study's identity.

## Understand the study records

### The plan: what should run

- **Study:** a scientific question, its methods, and its analysis.
- **Condition:** one set of settings being compared, such as baseline training
  or quarter-rate Red updates.
- **Replicate identity:** a label distinguishing independent repetitions.
  In a training comparison, this usually identifies a separately trained policy,
  for example `training-42`. The label alone does not establish independence.
- **Expected trial:** one planned case under a condition, identified
  before execution. For example, baseline training with training seed 42.
  A planned trial remains expected even if it fails or has not started.
- **Selection rule:** the stated rule for which attempts, checkpoints, and
  evaluations enter the analysis. For example, use each trial's completed final
  checkpoint and its declared test evaluations. Record the exact choices.
- **Protocol:** the saved plan connecting those conditions, expected trials,
  replicate identities, and selection rule.

For example, two conditions with training seeds 42 and 100 give four expected
trials. Each condition has two planned training replicates. Matching seed labels
across conditions do not by themselves establish a paired experiment.

### The execution: what actually happened

- **Attempt:** one execution of a trial, with its own status, settings, source,
  software environment, inputs, measurements, and outputs.
- **Retry:** a new attempt at the same trial, linked to the earlier attempt.
  A retry does not add an independent training replicate.
- **Evaluation:** measurements of a particular checkpoint under specified test
  settings. More evaluation episodes do not create more independently trained
  policies.
- **Saved selection:** the exact attempt, checkpoint, and evaluation identities
  used in one comparison. Select them explicitly so a later run or retry does
  not silently change an existing result.

## Analyze and report results

- Analyze explicit saved inputs and write each report to a new output directory.
- Explain the results, methods, and interpretation limits. Link routine execution
  details from the report to the retained records.
- Retain the numbers and settings used for any figures.
- Keep the exact analyzer source and its paths relative to the repository root
  with a saved selection. Reconstruct a saved report using that source;
  a changed analyzer produces a new analysis.

## Execution and evidence references

Use these instructions for the operations your study needs.

- [Joint training and evaluation records](../docs/joint-execution-records.md):
  retain JAX training updates, owned checkpoints, matchups, and trajectories.
- [Checkpoint evaluation capture](../docs/evaluation-capture.md): evaluate a
  Torch policy in CybORG and retain inputs, source, episodes, and execution status.
- [MLflow browsing and evidence retrieval](../docs/evidence-retrieval.md):
  find runs, download evidence, verify its bytes, and reconstruct selected reports.
- [Existing-evidence import](../docs/local-evidence-producers.md): read a retained
  frozen-defender bundle or co-training SQLite store into version 1 records.
