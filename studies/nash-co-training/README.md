# Nash co-training retrospective

## Question

As Red and Blue co-train, does a later defender handle a fixed pool of
attackers better, and can an attacker trained against that defender profitably
replace its original Red opponent? The retained work answers five related
questions with different comparison sets.

| Question | Archived evidence |
| --- | --- |
| [Did performance improve against fixed opponents?](#checkpoint-progress) | Historical screens, fresh-root confirmation, and a 150M training replication |
| [Can a saved policy profitably replace one member of a pair?](#saved-policy-deviations) | Unilateral checkpoint substitutions at nominal 100M and 150M |
| [Does a bounded fresh attacker search find a better replacement?](#matched-red-response) | Fixed-Blue Red training, validation, and held-out confirmation |
| [Do saved-policy rankings depend on the opponent?](#saved-policy-cross-play) | A partial GPU matrix and a separate complete CPU matrix |
| [What changes with a larger attacker search budget?](#red-search-budget) | A registered protocol cancelled before training |

The [archive study index](archive-studies.json) maps these groups to all twelve
historical study IDs and their evidence directories. Paths in the index are
relative to the unpacked reproduction bundle. Each study keeps its own
manifest, execution status, and comparison membership.

## Checkpoint progress

The seed-42 work started with fixed-Red tests near 50M, then compared later
training budgets. A 72-cell screen selected IPPO Blue92.16M and MAPPO Blue96M.
Both outperformed their own Blue99.984M on a fresh cohort of 2,048 paired roots
per matchup against six Reds from their respective trajectories. A separate
24-cell comparison used the same twelve opponents for both selected Blues and
favored the selected MAPPO Blue. The comparison describes those policies on
that pool and topology.

The fresh seed-43 MAPPO run reached 150M steps. Its 37-cell evaluation
included a fixed panel of six seed-42 opponents and used 1,024 paired roots.
On that panel, Blue124.8M and Blue150M gained 81.99 and 57.51 points,
respectively, over Blue99.84M. Panel effects average opponents within each
root before calculating uncertainty. Late-checkpoint behavior and recurrence
diagnostics found continuing policy changes, with repeated recurrence still
unresolved.

The evidence is in `reports/training-results/`, including the consolidated
report, individual manifests and reports, and the replication input manifest
and raw cells. Preserve screening and confirmation cohorts separately. Each
panel has its own opponent distribution and corrected contrast family.

## Saved-policy deviations

Fix one member of a co-trained pair and replace the other with an earlier
checkpoint. The seed-42 IPPO final pair admitted a profitable Blue
replacement (+154.92), and the seed-42 MAPPO final pair admitted a profitable
Red replacement (+38.10). The follow-up behavior measurements describe those
trajectories.

At the seed-43 150M pair, the five-cell GPU audit compared each team's
99.84M and 124.8M checkpoints with its final checkpoint against the
opponent's final policy. Red gained 39.12 and 24.17 points with the earlier
checkpoints. The four contrasts use 4,096 paired roots per cell and a common
family correction. Scientific evaluation and analysis completed. Slurm
recorded a failure in final archiving, and the report was recovered from
retained outputs.

The original payoff and behavior reports are in
`reports/training-results/evidence/studies/final-pair-behavior-20261029/`.
The later audit, including its completion and recovery provenance, is in
`reports/overnight-experiments/150m-checkpoint-audit/`.

## Matched Red response

Train Red independently against frozen Blue124.8M and Blue150M, using three
initialization seeds per target. Initial Red parameters match across the two
targets within each seed. Each attempt requested 10M steps and completed
9.984M. The six trained Reds and three saved Reds played both defenders on
512 validation roots per cell. The recorded lowest Blue mean and tie order
select one Red per defender. The study froze those choices before
confirmation on 4,096 fresh roots per cell.

The chosen Reds gained 497.62 and 530.16 points against the respective
defenders, both relative to the same saved Red150. The 150M gain rules out
exact equilibrium for that named pair in the evaluated setting. Blue150
outperformed Blue124.8M against the attacker selected for Blue124.8M and
underperformed it against the attacker selected for Blue150. These are
bounded searches on one defender trajectory and one topology. The intervals
measure evaluation uncertainty conditional on the selected policies.
Best-response saturation and global exploitability remain open.

The archive at `reports/overnight-experiments/matched-red-response/` retains
the complete candidate pool, frozen selection, validation and confirmation
cells, and all four registered contrasts.

## Saved-policy cross-play

The GPU protocol expected nine cells and referenced five already evaluated
in the audit. It reached its scheduling deadline before starting the other
four. Six of 24 planned contrasts were available, using the full family
correction.

A separate CPU protocol completed all nine matchups of the 99.84M, 124.8M,
and 150M policies, with the same 4,096 roots per cell. Every Red checkpoint
pair reversed ranking across the three Blues. The 15 supported better-response
edges formed an acyclic graph over those nine saved policy pairs. The matrix
answers a matchup question conditional on one training trajectory and one
topology. Establishing repeated temporal cycling would require chronological
recurrence evidence. The CPU matrix was exploratory, motivated by earlier
audit outcomes. GPU overlaps serve as descriptive comparisons.

The partial and complete reports live in
`reports/overnight-experiments/crossplay-100m-125m-150m/` and
`reports/overnight-experiments/cpu-crossplay-100m-125m-150m-20261102/`.

## Red search budget

The optional protocol planned one fresh continuous Red trajectory per frozen
Blue, with checkpoints at 9.984M and 19.968M steps. Within-trajectory midpoint
and final comparisons would isolate search budget for those two trajectories.
The proposed contrasts also used the saved Red150 and parent study's selected
Red as comparators.

The job was cancelled before starting after its dependency reached its
deadline. It produced zero training attempts and zero evaluation cells. The
effect of extending the search remains unanswered. The planned manifest and
terminal record are in `reports/overnight-experiments/red-search-budget/`.

## Reproduce and browse

Obtain the `reproducibility-20261102` bundle from the study owner and start
with its `REPRODUCIBILITY.md`. The producing revision for the latest training
and evaluations is `6f17b5546bc7fd2773d30d33444f44f1bcbb1b7b`. The published
reproduction tools are at `cedd69080d644729294a36a6ec37a907533c9cf3`.
Historical runs retain their own source provenance. MLflow can browse this
evidence and the associated measurements. The archived manifests, candidate
pools, and selections determine which observations enter each analysis.

See [replay and local tracking instructions](reproduce.md).
