# AGENTS.md

## Study workflow

Start with the [study guide](studies/README.md).

Write for a researcher deciding what the result means and what to do next.
Lead with the question and finding, then the comparison and limitations that
affect the conclusion. Keep useful ITM result paths and evidence/replay links.

Do not narrate routine execution settings: Slurm partitions, CPU/GPU/RAM
allocations, wall-time caps, device assignment, or generic launcher safeguards.
Those settings already exist in configuration. Delete redundant prose; do not
move it into another document or script comments. Mention an execution detail
only when it changes the scientific interpretation or is essential to reproduce
the result. Keep README summaries short; link exact methods and records.

## Tests

```bash
uv run pytest path/to/modified_test.py               # run tests added or modified by the current change
```

Run only the test files or focused test cases added or modified for the current
work. Do not run the full suite, slow suite, or all tests unless the user
explicitly requests it.

## Linting

Run `uv run ruff check --fix . && uv run ruff format .` before committing.

## Evaluation resources

Prefer CPU for scalar/serial Jaxborg evaluation and small evaluation workloads;
keep GPUs primarily available for training. For substantially batched or
vectorized evaluation, choose the backend using representative end-to-end
timing of the actual evaluator and batch size. Include startup and compilation
costs when assessing short jobs.

For CPU-only JAX evaluation, set `JAX_PLATFORMS=cpu` before Python imports and
omit GPU requests from the Slurm allocation. Keep using the normal recorded
evaluation workflow and retain the selected backend and resource settings.
