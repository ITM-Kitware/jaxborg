# Checkpoint study reproduction

The tools in this directory analyze retained episode outcomes and generate
commands for independently re-evaluating saved policies. Training and evaluation
implementations are in [the repository source](../../src/jaxborg), including
[matchup evaluation](../../scripts/eval/eval_matchup.py) and
[response campaigns](../../src/jaxborg/response_campaign.py).

## Research questions

- Does additional co-training improve a team against fixed opponents?
- Does a named co-trained pair admit a profitable unilateral policy replacement?
- Do saved-policy rankings reverse when the opponent changes?
- Is a saved-policy better-response graph cyclic, and what additional evidence
  would be required to establish repeated temporal cycling during training?
- How does an independently trained attacker's search budget affect a fixed
  defender comparison?

These questions require different comparisons. Evaluation against a moving
co-trained opponent does not by itself measure improvement against a controlled
opponent distribution. A bounded attacker search can find profitable deviations
without supplying a global exploitability upper bound.

## Inputs

Supply a reproduction bundle directory containing `artifact-catalog.json`,
content-addressed `objects/`, and `reports/`. Checkpoints and episode outcomes
are separate inputs, rather than Git objects. The catalog records original
paths, SHA-256 values, file sizes and their roles. Restoring files preserves
their bytes and recreates paths below the selected destination.

The statistical input layout is:

```text
reports/
  training-results/evidence/studies/training-replication-20261029/
  overnight-experiments/
    150m-checkpoint-audit/evidence/
    matched-red-response/evidence/
    cpu-crossplay-100m-125m-150m-20261102/evidence/
```

Each study's evidence directory supplies its registered manifest, raw episode
outcomes, original analysis, and provenance records. The response study also
supplies its complete candidate pool, validation outcomes and frozen selection.

## Verify and recompute

These commands use Python's standard library and do not execute rollouts:

```bash
python3 research/reproduction/reproduce.py --bundle /path/to/bundle check
python3 research/reproduction/reproduce.py --bundle /path/to/bundle \
  statistics --output /path/to/new-analysis.json
```

`statistics` recomputes the registered 24 saved-policy matrix contrasts, four
checkpoint-audit contrasts, four fresh-response confirmation contrasts, and
28 fixed-panel training comparisons. Paired differences use aligned episode
roots. Equal-opponent panels average within each root before calculating
uncertainty, preserving covariance. Family sizes remain fixed at their
registered values.

The complete provenance verifiers additionally check policy identities, frozen
parameters, selection timing, source records and supported response graphs:

```bash
python3 research/reproduction/verify_cpu.py \
  --study-root /path/to/bundle/reports/overnight-experiments/cpu-crossplay-100m-125m-150m-20261102
python3 research/reproduction/verify_response.py \
  --study-root /path/to/bundle/reports/overnight-experiments/matched-red-response
```

These verifiers write derived reports beside their study inputs. Keep raw
outcomes and input manifests intact. `consolidate_training.py` separately
accepts `--bundle`, `--remote-verification` and `--output` for the earlier
fixed-panel analysis and writes machine-readable summaries.

## Generate new evaluation commands

Restore the supplied checkpoint and topology files, and use a checkout at the
source revision recorded in the study manifest:

```bash
python3 research/reproduction/reproduce.py --bundle /path/to/bundle \
  restore --directory /path/to/restored
python3 research/reproduction/reproduce.py --bundle /path/to/bundle \
  eval-command --blue-steps 150000000 --red-steps 99840000 \
  --restored /path/to/restored --checkout /path/to/evaluation-checkout \
  --output /path/to/new-run/result.json > /path/to/new-run.sh
```

The generated shell command sets an explicit CPU platform and tracking store,
selects both policy files and the fixed topology, and uses the registered
episode cohort. Generating a command does not execute it. Use a fresh output
path and run rollouts only inside the appropriate resource allocation.

For ITM CPU evaluation inside Slurm, both `JAX_PLATFORMS=cpu` and
`JAXBORG_ALLOW_CPU=1` are required. Preserve assigned device visibility. The
[CPU affinity helpers](protocols/cpu_crossplay/cpu_affinity.py) map Slurm CPU
numbers to Linux CPU numbers; do not reuse another allocation's CPU IDs.

## Protocol runners

`protocols/` contains the standalone coordinators for checkpoint auditing,
fixed-Blue Red response training, CPU/GPU cross-play and attacker-budget
calibration. Their prepared manifests bind input bytes, source revision,
launcher hashes, seeds, sample sizes and resource settings. Keep the source
checkout pinned for an allocation's duration, use matching archived launcher
bytes for an existing manifest, and register fresh paths and hashes for a new
study. Completed study roots are not templates for overwriting prior outcomes.

For preparation scripts, configure the store, pinned evaluation checkout and
seed-audit collection explicitly:

```bash
export JAXBORG_EXP_DIR=/path/to/experiment-store
export JAXBORG_EVALUATION_CHECKOUT=/path/to/pinned-evaluation-checkout
export JAXBORG_SEED_AUDIT_ROOT=/path/to/seed-ledgers-and-manifests
export JAXBORG_STUDY_ROOT=/path/to/fresh-study-directory
```

Copy a protocol's scripts into a fresh study directory outside the source
checkout before preparing its manifest. The published scripts are formatted
and parameterized; existing manifests require their matching archived launcher
bytes. Generate new launcher hashes when registering a fresh study.
The registration scripts retain dated resource settings. Register a new
execution window before any fresh allocation; do not extend an old job's
cutoff or reuse its completed output paths.

Candidate selection belongs to validation data. Confirmation outcomes must not
change the candidate set, seed choices, sample counts or frozen selection.
Likewise, a ranking reversal or a loop in a saved-policy game does not by
itself establish repeated temporal cycling. These distinctions belong to the
interpretation of a study, independently of whether its execution completes.
