# CLAUDE.md

## Experiment workflow

For experiment work, start with the [experiment guide](experiments/README.md).
It links the supported evaluation capture, historical evidence import, and
saved-analysis workflows. Read the relevant workflow before running it.

## Commands

```bash
uv run pytest                                        # fast suite (~7 min, excludes `-m slow`)
uv run pytest -m slow                                # slow L3 fuzz + full-episode parity (~40 min)
uv run pytest -m ""                                  # everything
```

## Linting

Run `uv run ruff check --fix . && uv run ruff format .` before committing.
