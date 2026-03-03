# CybORG-First Parity Execution Plan

## Goal
Replace workaround-heavy behavior with direct CybORG mechanic parity for session/PID/Remove paths, then continue parity loop from a cleaner base.

## Phase 1: Lock Spec And Baseline
1. Freeze and publish source-backed parity spec.
2. Maintain workaround inventory with open/closed items.
3. Add TODO markers in code for each open shim.

Exit criteria:
- `docs/parity/cyborg_remove_pid_model_spec.md` accepted as source-of-truth.
- Every known shim is listed with explicit removal condition.

## Phase 2: Data Model Alignment
1. Make suspicious PID memory representation match CybORG list behavior.
2. Remove dependence on independent budget counters (or strictly derive them from list state).
3. Ensure session PID lifecycle is strictly host/session derived.

Exit criteria:
- No parity-critical path depends on synthetic budget semantics.
- Differential tests cover stale/live/mixed PID rows and list ordering effects.

## Phase 3: Event-Driven Suspicious PID Ingestion
1. Implement Monitor-equivalent ingestion contract in JAX path.
2. Stop direct exploit/phishing writes that bypass event semantics.
3. Validate event-to-suspicious-memory transitions with explicit differential tests.

Exit criteria:
- Suspicious PID additions are explainable via CybORG event sources.
- Remove mismatches tied to stale PID churn are eliminated in targeted seeds.

## Phase 4: Remove Pipeline Hardening
1. Keep `Remove` strictly PID-driven.
2. Remove remaining fallback guards not present in CybORG.
3. Validate against explicit multi-session host cases (abstract+concrete+stale combinations).

Exit criteria:
- `Remove` behavior reproducibly matches CybORG on mechanism-specific tests.

## Phase 5: Parity Loop Discipline
1. Run short loop (`FUZZ_SEEDS=5 FUZZ_STEPS=30`) until first mismatch moves off PID/Remove path.
2. Increase to medium (`20x100`), then stress (`50x200`).
3. One-gap-at-a-time process: reproduce -> explicit differential regression -> minimal fix -> rerun.

Exit criteria:
- Full stress loop first mismatch is outside the completed mechanic area, or clean.

## Commit Policy
1. `test:` commit for failing explicit regression.
2. `fix:` commit for minimal production change.
3. `refactor:` commit to remove obsolete workaround.

No mixed workaround introduction and cleanup in the same commit unless unavoidable.
