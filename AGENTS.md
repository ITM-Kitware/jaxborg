# AGENTS.md

Read [CLAUDE.md](CLAUDE.md) for full project context, architecture, commands, and development workflow.

## Parity Loop

The goal: systematically find and fix every CybORG/JAX state divergence until full functional parity is achieved.

### Quick start

```bash
# Default
bash scripts/parity_loop.sh

# Start small for fast iteration
FUZZ_SEEDS=5 FUZZ_STEPS=30 bash scripts/parity_loop.sh

# Full stress test
FUZZ_SEEDS=50 FUZZ_STEPS=200 bash scripts/parity_loop.sh

# Multiple fuzz batches
MAX_ITERATIONS=5 bash scripts/parity_loop.sh
```

### How it works

1. **Fuzzer** runs `tests/differential/fuzzer.py` across N seeds x M steps
2. On first mismatch, prints: seed, step, field, CybORG value, JAX value, and exits non-zero
3. Harness mode is strict-only: no CybORG->JAX state patching/injection is allowed during replay
4. You then fix the gap manually (or with Codex/Claude), following the "Fixing Differential Gaps" workflow in `CLAUDE.md`
5. Rerun the loop

### Running with Codex CLI directly

If you want to invoke Codex manually for a single gap:

```bash
codex exec --full-auto "The differential fuzzer found a CybORG/JAX state divergence.

Mismatch at seed=0, step=33: host_compromised [host_56] cyborg=0 jax=1

Follow the 'Fixing Differential Gaps' workflow in CLAUDE.md:
1. Write a failing regression test that reproduces this mismatch
2. Read CybORG source at .venv/lib/python3.11/site-packages/CybORG/
3. Fix the JAX code in src/jaxborg/
4. Verify: uv run pytest tests/ -v -x
5. Lint: uv run ruff check --fix . && uv run ruff format .
6. Commit with a message describing the gap and fix."
```

### Running with Claude Code directly

```bash
claude -p "..." --allowedTools "Read,Write,Edit,Grep,Glob,Bash,Task,WebFetch"
```

### Progressive fuzzing strategy

Start with short runs for fast feedback, increase coverage as gaps are fixed:

| Phase | Seeds | Steps | When |
|-------|-------|-------|------|
| Early | 5 | 30 | First few iterations, finding obvious gaps |
| Middle | 20 | 100 | After easy fixes, broader coverage |
| Final | 50 | 200 | Stress testing near-parity |
