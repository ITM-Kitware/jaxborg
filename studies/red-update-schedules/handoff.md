# Independent reconstruction and bounded follow-up

This is an execution worksheet, not a claim that any assessment has occurred.
Complete a short-run handoff before substantive compute and repeat the affected
checks for the substantive evidence. Keep filled worksheets outside the checkout
with the evidence. Share public claims only when their evidence is accessible.

Provide the reviewer the execution store/MLflow locator and access instructions,
selected run IDs, comparison bundle with an authenticated manifest digest,
producing source/configuration and dependency artifacts, required checkpoint/input
identities, this study's README, and the two implementation diffs. Do not require
this conversation, private paths or the original checkout. Initially withhold the
implementer's usability verdict. Record missing information as a failed attempt.

1. From a fresh directory/client, find the selected checkpoint and its settings.
   Follow its producing execution, source snapshot, effective evaluation settings
   and exact topology/checkpoint hashes. Verify retrieved bytes. Record the actual
   subset of shared helpers/dependencies needed for retrieval and source audit.
2. Retrieve the complete comparison bundle, verify its trusted manifest digest,
   and run the retained `analyze.py rebuild` command. Compare the regenerated
   files. Remove access to the producing checkout before this step. A lightweight
   report reconstruction does not establish that checkpoint reevaluation works.
3. Inspect expected/missing cells, every training-seed value, episode/topology role
   identities, weighting, optimizer/schedule differences and timing scopes.
   Check training-update and adaptive gate diagnostics against the retained raw
   measurements. Describe incomplete evidence rather than filling its gaps.
4. After the first comparison is available, choose one small meaningful follow-up
   using measurements actually retained. The implementer must not prescribe the
   answer or supply a pre-scripted demonstration. Make the change in study-local
   analysis code, retain revised source with `revise`, and preserve the original
   comparison. One exercise suffices. Additional training or reevaluation needs a
   separately agreed compute budget; adaptive is already a core condition.
5. Record commands, code/configuration edits, metadata bookkeeping, undocumented
   steps, help needed, failures, and hands-on time when measured. If live help or
   metadata repair was necessary, record the attempt as unsuccessful, fix the
   bounded defect, and repeat the affected exercise without deleting the failure.

Compare the same small task set against the existing tools on compatible copies
of these fresh pilot inputs: locating the checkpoint/settings, producing the first
table, and implementing the chosen follow-up. Existing starting points include the
[native result loader](../../plots/cotraining/data.py) and
[MLflow analysis](../../plots/plot_mlflow.py). Inspect their actual input assumptions
before reuse; do not adopt historical checkpoint/recency selection implicitly.
Do not duplicate full training or build a replacement old workflow to manufacture
a baseline. Mark a code/documentation walkthrough as an estimate. Unsupported
retrieval guarantees are capability gaps, not zero effort.

Use one row per observed task/attempt in a retained CSV or Markdown worksheet:

| Task and workflow | Setup or recurring | Commands/manual steps | Code/config edits | Metadata handling | Help/failure | Hands-on time, if measured | Machine runtime | Evidence link |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pending | Pending | Pending | Pending | Pending | Pending | Unmeasured | Unmeasured | Pending |

Separate one-time integration/environment work from recurring research work and
machine runtime. Note tooling familiarity, unavailable baseline functionality,
actual storage/transport costs and whether timings were measured or estimated.
State both added retention benefits and friction; successful reconstruction alone
does not establish usefulness or quantified time savings. A software reviewer
cannot establish representative researcher experience.

The implementer's account should distinguish reused training/evaluation/checkpoint
machinery, complementary retention/selection, necessary integration refactors,
and inherited features the exercise did not use. The independent assessment
should identify hidden dependencies, scientific validity gaps, failed handoffs,
and components to keep, simplify, remove or defer. Record the next scientific
decision with the per-seed performance and comparable measured costs, permitting
an inconclusive result. Do not label unexecuted criteria as passed.
