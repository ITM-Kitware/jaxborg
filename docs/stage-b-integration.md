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
wrapper rejects a `run` launch until canonical smoke evidence covers the same
manifest, source SHA and isolated environment. The combined `pilot` action performs
its own smoke inside the allocation and proceeds only after that smoke succeeds.
It rejects silent CPU fallback.

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
external root from the campaign YAML,
e.g. `/home/local/KHQ/paul.elliott/jaxborg-exp/shared`.
Use the checkout's `.venv/bin/python`; do not sync inside an allocation.

```bash
.venv/bin/python scripts/experiments/response_oracle.py prepare \
  --campaign campaigns/response-oracles/mappo-seed42.yaml --defender mappo-49968000
.venv/bin/python scripts/experiments/response_oracle.py dry-run --manifest /absolute/results/stage-b/manifest.json
./scripts/experiments/submit_response_oracle.sh smoke /absolute/results/stage-b/manifest.json
# After that job has succeeded and smoke.json has canonical verified evidence:
./scripts/experiments/submit_response_oracle.sh run /absolute/results/stage-b/manifest.json
# Re-aggregate finished evaluations without any GPU allocation:
JAX_PLATFORMS=cpu .venv/bin/python scripts/experiments/response_oracle.py aggregate --manifest /absolute/results/stage-b/manifest.json
```

The inherited global JAX override stripped the CUDA extra from `uv.lock`; smoke
job 3851 failed at GPU discovery before any rollout. The CUDA extra now explicitly
requests the plugin at the existing locked JAX versions. The repaired lock adds
only CUDA backend distributions and retains every existing distribution's version
and source. The first failed checkout/environment remains unchanged for provenance.
See [uv override semantics](https://docs.astral.sh/uv/reference/settings/#override-dependencies).

`prepare` verifies the exact paired source, archives its sidecar/hashes, regenerates
the original singleton using source-identical generator bytes and unchanged existing
dependency versions, publishes
canonical `runs:/...` inputs and writes recipes/seed splits before test outcomes.
The manifest records original/current lock hashes and the explicit GPU additions;
the updated lock is not represented as identical to the historical lock.
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
Slurm's `CUDA_VISIBLE_DEVICES`. Campaign YAML records the allocation resources.
The user authorized two concurrent response jobs; submit dependency-constrained
jobs so this cap also covers the existing IPPO pilot.

Pipeline `state.json` records completed training/evaluation IDs and selection.
Re-running the pipeline verifies completed canonical artifacts and reuses explicit
validated evaluation fingerprints, returning the original evaluation ID. It skips
completed attempts. Failed training remains failed; portable weights do not save
optimizer/PRNG/environment state, so incomplete training cannot be resumed by
passing a run ID. Inspect failures before authorizing another full attempt. Every
failed/partial run and any consumed extra budget must remain in the final record.

## Declarative MAPPO expansion

`campaigns/response-oracles/mappo-seed42.yaml` defines both seed-42 MAPPO
checkpoints (9,600,000 and 49,968,000 source steps), the shared MLflow root,
three fresh Red seeds, training budget, disjoint evaluation roots, smoke settings,
validation selection, bootstrap settings and community Slurm resources.
The repository uses its native recipe YAML loader; it has no Hydra dependency.
The campaign runner resolves ordinary trainer recipes and invokes the existing
`ippo_jax.py` and `eval_matchup.py` entrypoints. Selection and paired statistics
remain Python operations, with their settings archived from the campaign YAML.

Fresh Reds use the same IPPO source recipe as the original pilot, including its
shared 256-by-256 local actor/critic, core optimizer and JAX rollout settings.
The MAPPO defender retains its full source architecture and weights. Its
centralized critic stays frozen; inference uses the original actor. No MAPPO
Red override is inherited. Each defender must pass its own enhanced GPU smoke.

Source the persistent store profile, prepare each named defender and use the
combined action to execute smoke, training, validation and paired final tests:

```bash
source /home/local/KHQ/paul.elliott/jaxborg-exp/shared/store.env
.venv/bin/python scripts/experiments/response_oracle.py prepare \
  --campaign campaigns/response-oracles/mappo-seed42.yaml --defender mappo-49968000
./scripts/experiments/submit_response_oracle.sh pilot /absolute/defender/manifest.json
# Hold a second campaign until another response job releases its slot:
JAXBORG_SLURM_DEPENDENCY=afterany:JOB_ID \
  ./scripts/experiments/submit_response_oracle.sh pilot /absolute/second-defender/manifest.json
```

Each manifest archives the campaign YAML, resolved recipes and canonical source
inputs. The two MAPPO studies share one database and artifact store; their
campaign IDs and report directories are distinct. Completed collection links
include the actual defender algorithm and source steps. No Stage C is launched.

## Overnight Stage C preparation

`campaigns/response-oracles/ippo-seed42-red-curve.yaml` adds the four remaining
IPPO seed-42 snapshots, reusing the fixed Red template and protocol. The original
9.6M Stage B point stays in its producing store and retains its original source
revision. Verify the source/game/budget/seed contracts and executable science
file identities before treating these points as one curve; a full Git SHA is
not represented as identical across the later config/logging changes.

The user authorized free additional GPUs for the 13-hour overnight window. The
new campaign requests 48G per GPU job so four jobs can fit the node's 254000M
memory reservation limit. Existing 64G MAPPO allocations are retained. Slurm
community scheduling reserves the GPUs and respects other users' allocations.
The Red curve jobs depend on successful IPPO Stage B completion and on the
already queued MAPPO 9.6M job starting.

`recipes/verification/stage_c_blue_smoke.yaml` uses the ordinary trainer with
`train.teams: blue`, original frozen Red, two 2000-step updates and a required
four-episode saved/reloaded matchup. Its post-training hook preserves explicit
`JAXBORG_EXP_DIR`; canonical Run checkpoint paths must not redirect evaluation
into a new run-local MLflow store. This smoke does not authorize the 15 full
Blue-response attempts by itself. The complete Stage C table and approximate
NashConv remain pending both response directions.
