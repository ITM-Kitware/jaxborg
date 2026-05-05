# JAXborg Performance Autoresearch Agent

You are running one iteration of an autoresearch-style optimization loop in the
`opt` worktree.

This loop follows the Karpathy pattern: a frozen benchmark measures the result,
the coding agent proposes one code change, and the outer shell script keeps the
change only if it improves the benchmark while preserving correctness. Do not
modify the benchmark or the loop. Do not commit; the outer script owns commits,
logging, and reset.

## Worktree

`${WORKTREE}`

## Current Iteration

Iteration: `${ITERATION}`

Best known median steps/sec: `${BEST_STEADY_SPS}`
Best known compile estimate seconds: `${BEST_COMPILE_TIME}`

Recent results:

```text
${RECENT_RESULTS}
```

Experiment memory:

```text
${EXPERIMENT_MEMORY}
```

Do not repeat discarded experiments unless the new attempt is materially
different. If you build on a previous failed idea, state exactly what changed
in `EXPERIMENT_DESCRIPTION`.

## Goal

Improve JAXborg training performance without changing behavior.

Primary metric:

- Maximize `steady_sps_median`.

Secondary metric:

- Reduce `compile_time_estimate_s`.

Steps/sec is more important than compile time. A compile-time improvement does
not count if steady-state throughput meaningfully regresses.

## Frozen Measurement

The outer script will run:

```bash
${CORRECTNESS_CMD}
${BENCHMARK_CMD}
```

Correctness must pass before any performance result is considered. The
benchmark clears the XLA cache every run and reports `PERF_JSON` from
`scripts/dev/optimize_perf/benchmark.py`.

## Editable Scope

You may edit only:

```text
${EDITABLE_SCOPE}
```

Do not edit:

- `scripts/dev/optimize_perf/benchmark.py`
- `scripts/dev/optimize_perf/loop_claude.sh`
- `scripts/dev/optimize_perf/prompt.md`
- `scripts/eval/`
- `tests/`
- `recipes/`
- `pyproject.toml`
- `uv.lock`
- CybORG installed package files

If you need a new test, stop and explain it in your final response instead of
editing tests during this optimization loop.

## Behavior Constraints

Preserve the base CybORG CAGE 4 semantics. Do not weaken, remove, bypass, or
special-case:

- reward logic
- observations
- action masks
- action ordering
- red/green/blue RNG behavior
- topology generation or CybORG-derived topology extraction
- training/evaluation episode length semantics
- parity harness assumptions

The benchmark always uses the default generated topology path. Do not add or
switch topology modes as part of an optimization.

Any change that improves throughput by doing less simulation, changing action
availability, dropping agents, shortening rollouts, changing reward scaling, or
altering CybORG parity is invalid.

## Good Optimization Targets

Prefer local, mechanically defensible JAX improvements:

- Move shape/static computations out of jitted functions when possible.
- Avoid rebuilding constant arrays, vmaps, masks, or Python containers inside
  hot jitted paths when they can be captured once.
- Reduce redundant reshapes/transposes or repeated network applications.
- Improve scan/vmap structure without changing outputs.
- Simplify PyTree structure only if behavior remains identical.
- Reduce compile-time complexity if steady-state throughput is not harmed.

## Instructions

1. Inspect the current code and previous results.
2. Make one focused optimization attempt.
3. Run only cheap local checks if useful. Do not run the full GPU benchmark;
   the outer script will do that through Slurm.
4. Leave the working tree with your candidate change.
5. Do not commit.
6. End with these exact lines:

```text
EXPERIMENT_DESCRIPTION: <short tab-free description of the idea>
EXPECTED_IMPACT: <why this should improve steady_sps_median or compile time>
RISK: <main behavior/correctness risk, or "low">
```
