# Nash co-training retrospective

## Question

Can a unilateral policy replacement improve either team's return against
its co-trained opponent? How much does the answer depend on the opponent
held fixed?
These comparisons test whether co-training produced a pair that withstands
the available saved policies and independently trained attackers.

## Method

- **Saved policies.** Hold one team fixed and substitute a saved checkpoint
  for the other. Evaluate all nine matchups among three ages from the
  seed-43 MAPPO trajectory in a separate CPU matrix.
- **Fresh Red responses.** Train Red for about 10M transitions against each
  of two frozen Blue policies, using three matched initialization seeds per
  target. Select one attacker per Blue from the six trained Reds and three
  saved Reds on 512 validation roots per cell. Freeze the selection before
  confirmation on 4,096 held-out roots per cell, with saved Red150 as the
  common baseline.
- **Analysis.** Compare paired own-return differences on aligned episode
  roots. Keep each protocol's opponent set and correction family intact.
  The manifests and frozen selections identify the exact checkpoints and
  evaluations.

## Status and findings

The completed fresh Red search found a profitable unilateral deviation
against the 150M pair. The selected attacker gained 530.16 points over
Red150 against the same Blue, with a corrected interval of [517.54, 542.79].
That rules out exact equilibrium for this pair in the evaluated setting.

In the completed CPU matrix, each Red checkpoint pair reversed ranking
across opponents. The supported better-response graph contained no cycle.
Repeated temporal cycling remains unresolved.

The earlier GPU matrix is partial, and the planned 20M attacker search was
cancelled before starting. The completed comparisons use one defender
trajectory and one topology. Their intervals quantify episode variation
conditional on the selected policies. Best-response saturation and broader
generalization remain open.

## Evidence

Obtain the unpacked `reproducibility-20261102` bundle from the study owner.
The [archive index](archive-studies.json) locates the twelve original protocols,
their execution status, reports, and evidence directories. Their manifests
retain the producing source and comparison inputs.

Browse historical fixed-panel runs and reports under
**Jaxborg Nash / checkpoint-progress** in the local MLflow import. The
**historical-context** archive run holds the consolidated reports and replay
artifacts, including the frozen selection and matchup table.

## Commands

See [Replay and browse the Nash retrospective](reproduce.md) to rebuild the
comparisons from retained measurements and find the results in MLflow.
