# Stage B research/logging integration

The dedicated `stage-b/logging-integration` branch rebases the compatible HMARL
revision `2367e284d6e54ebadc277c692660c80219cbdc93` onto the logging core
`55ee8bd878bce4261aa24f3e199ed277bdcdda50`, preserving research merge topology.
Existing HMARL, co-training and logging branches/worktrees are unchanged.
The newer co-training evaluation-count commit `0fa1327` is not imported: this
pilot has its own fixed episode counts. No source checkout used by another job
is changed. No upstream branch is pushed or retargeted by this local integration.

Conflict resolutions retain the research trainer/policy implementation and the
shared `Run`, artifact resolution/publication and launch interface. Explicit
adapters restore ownership for JAX joint/stock trainers, CybORG trainers and the
learned matchup evaluator. The rollout/PPO implementations, simulator, observation
and reward rules, policy classes and topology generator match the original HMARL
revision. Rebased intermediate commits are integration history; launch only the
final tested, committed adapter revision.

`eval.allow_training_topologies: true` explicitly permits evaluation on the
training topology for a fixed-game response search. The existing held-out-layout
validation stays the default. Stage B holds out episode/PRNG roots instead of
layouts. Policy execution remains stochastic. Original Red is a validation
fallback candidate. Neither CIA/resilience rules nor historical/cross-seed sweeps
are used. Requested budgets are 10M each; original 96 × 500 rollout sizing completes
9,984,000 steps per attempt, with no automatic budget extensions.

## Readiness and scope

CPU tests verify recipe settings, stream separation, fallback selection, paired
bootstrap arithmetic (including negative gains), actual JAX Red updates with
450-wide frozen Blue, saved/reloaded policies, completed checkpoint retention on
cancellation, evaluation lineage, fingerprint-based reuse and launch pinning.
The actual enhanced-observation singleton GPU smoke is a separate prerequisite.
A CPU test or the main-based issue #25 smoke does not satisfy it. The submission
wrapper rejects a pilot launch until canonical smoke evidence covers the same
manifest, source SHA and isolated environment. It rejects silent CPU fallback.

This supports the Stage B path. The entire issue #28 stack is not declared adopted:
legacy historical/HMARL evaluation suites, inline checkpoint evaluation and the
separate PR #26 co-training trajectory exporter still need independent adapters
and verification. Stage B requires no trajectory export and disables those suites.
Main-based trajectory adapters inherited from the logging core remain present.
No Stage C jobs are scheduled by these commands.

## Reproduce from a clean pinned checkout

Commit first. Prepare an unused detached launch worktree at the full implementation
SHA and its own `uv sync --frozen --extra cuda` environment. Keep both unchanged
through smoke and pilot execution. Set `JAXBORG_EXP_DIR` to an explicit absolute
external root, e.g. `/home/local/KHQ/paul.elliott/src/cyber/jaxborg-exp/oracle-stage-b`.
Use the checkout's `.venv/bin/python`; do not sync inside an allocation.

```bash
.venv/bin/python scripts/experiments/oracle_stage_b.py prepare
.venv/bin/python scripts/experiments/oracle_stage_b.py dry-run --manifest /absolute/results/stage-b/manifest.json
./scripts/experiments/submit_stage_b.sh smoke /absolute/results/stage-b/manifest.json
# After that job has succeeded and smoke.json has canonical verified evidence:
./scripts/experiments/submit_stage_b.sh run /absolute/results/stage-b/manifest.json
# Re-aggregate finished evaluations without any GPU allocation:
JAX_PLATFORMS=cpu .venv/bin/python scripts/experiments/oracle_stage_b.py aggregate --manifest /absolute/results/stage-b/manifest.json
```

`prepare` verifies the exact paired source, archives its sidecar/hashes, regenerates
the original singleton using source-identical generator bytes/lock, publishes
canonical `runs:/...` inputs and writes recipes/seed splits before test outcomes.
`dry-run` prints resolved settings and planned commands without scheduling.
`smoke` performs a four-episode original baseline, two real Red updates and four
saved/reloaded trained episodes. `run` trains three attempts sequentially, evaluates
all four validation candidates, saves the selected identity, runs the two paired
600-episode tests and exports data, plots, report and Stage C handoff.

The manifest owns exact recipe hashes, source/dependency provenance and stream
roots. Each run owns its checkpoints/sidecars, metrics and console logs. Batch
startup logs live under `$JAXBORG_EXP_DIR/launches/slurm/<job-id>.{out,err}`.
The retained launch record pins checkout, interpreter, full SHA and dependencies;
verification and GPU discovery happen inside the community allocation. Respect
Slurm's `CUDA_VISIBLE_DEVICES`. Use one job at a time.

Pipeline `state.json` records completed training/evaluation IDs and selection.
Re-running the pipeline verifies completed canonical artifacts and reuses explicit
validated evaluation fingerprints, returning the original evaluation ID. It skips
completed attempts. Failed training remains failed; portable weights do not save
optimizer/PRNG/environment state, so incomplete training cannot be resumed by
passing a run ID. Inspect failures before authorizing another full attempt. Every
failed/partial run and any consumed extra budget must remain in the final record.
