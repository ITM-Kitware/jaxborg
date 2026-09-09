# AGENTS.md

## Tests

```bash
uv run pytest path/to/modified_test.py               # run tests added or modified by the current change
```

Run only the test files or focused test cases added or modified for the current
work. Do not run the full suite, slow suite, or all tests unless the user
explicitly requests it.

## Linting

Run `uv run ruff check --fix . && uv run ruff format .` before committing.
