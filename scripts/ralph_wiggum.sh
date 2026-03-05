#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MAX_RETRIES=${MAX_RETRIES:-5}
COMPLETED=0

CYBORG_PREFIX=".venv/lib/python3.11/site-packages"

while true; do
    NEXT_JSON=$(uv run python -c "
import json
from tests.catalog import get_next_incomplete
s = get_next_incomplete()
if s is None:
    print('DONE')
else:
    print(json.dumps({
        'id': s.id,
        'name': s.name,
        'description': s.description,
        'cyborg_source_paths': s.cyborg_source_paths,
        'jax_target_files': s.jax_target_files,
        'depends_on': s.depends_on,
    }))
")

    if [ "$NEXT_JSON" = "DONE" ]; then
        echo "All subsystems complete!"
        break
    fi

    ID=$(echo "$NEXT_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
    NAME=$(echo "$NEXT_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['name'])")
    DESC=$(echo "$NEXT_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['description'])")
    CYBORG_PATHS=$(echo "$NEXT_JSON" | python3 -c "import json,sys; print(' '.join(json.load(sys.stdin)['cyborg_source_paths']))")
    JAX_FILES=$(echo "$NEXT_JSON" | python3 -c "import json,sys; print(' '.join(json.load(sys.stdin)['jax_target_files']))")

    echo ""
    echo "================================================================"
    echo "  Subsystem $ID: $NAME ($((COMPLETED + 1))/22)"
    echo "  $DESC"
    echo "================================================================"
    echo ""

    ATTEMPT=0
    PASSED=false

    while [ $ATTEMPT -lt $MAX_RETRIES ]; do
        ATTEMPT=$((ATTEMPT + 1))
        echo "--- Attempt $ATTEMPT/$MAX_RETRIES ---"

        claude -p "You are porting CC4 (CAGE Challenge 4) from CybORG to JAX.
Working directory: $(pwd)

Subsystem $ID: $NAME
Description: $DESC

CybORG source files to read for exact behavior (prefix with $CYBORG_PREFIX/):
$CYBORG_PATHS

JAX target files to implement/modify:
$JAX_FILES

## Differential Testing Requirements

Your tests MUST be differential: run the same actions in both CybORG and JAX, then compare results.
CybORG is installed and importable. Use it as the ground-truth oracle.

Pattern for every test:
1. Create a CybORG env: EnterpriseScenarioGenerator(blue_agent_class=SleepAgent, green_agent_class=EnterpriseGreenAgent, red_agent_class=FiniteStateRedAgent, steps=500), seed=42
2. Build JAX const from it: build_const_from_cyborg(cyborg_env)
3. Create JAX state: create_initial_state() (or appropriate starting state)
4. Execute the SAME action sequence in both environments
5. Compare the relevant state fields — they must match

Use tests/conftest.py which provides a cyborg_env fixture.
See tests/subsystems/test_static_topology.py TestDifferentialWithCybORG for reference.

For action subsystems: inject specific actions into CybORG, apply the same via JAX apply_red_action/apply_blue_action, compare host_compromised, red_sessions, red_privilege, red_activity_this_step, etc.

Do NOT write tests that only check JAX in isolation. Every test must compare JAX output against CybORG output.

## Workflow

1. Read the CybORG source files to understand exact behavior.
   Installed CybORG is at $CYBORG_PREFIX/CybORG/
2. Read current JAX target files + src/jaxborg/constants.py + src/jaxborg/state.py for schema.
3. Implement the JAX code for this subsystem.
4. Create DIFFERENTIAL tests in tests/subsystems/test_${NAME}.py
   - Every test must run both CybORG and JAX and compare.
5. Run: uv run pytest tests/subsystems/test_${NAME}.py -v
6. Fix failures until tests pass.
7. Run: uv run pytest tests/ -v -m "not slow" to verify no regressions.
8. Mark subsystem as passing:
   uv run python -c \"from tests.catalog import mark_passing; mark_passing($ID)\"
9. Commit: git add -A && git commit -m 'subsystem $ID: $NAME'

## Constraints

- All host-indexed arrays padded to GLOBAL_MAX_HOSTS=137, use host_active mask
- 5 blue agents, 6 red agents (hardcoded, no masking)
- Use flax.struct.dataclass for state, jax.lax.cond for branching
- Reference CC2 port at /home/paulhax/src/cyber/jaxmarl/integration/jaxmarl/environments/cage/ for patterns
- pyproject.toml has pythonpath=[\".\"] so tests can import tests.catalog etc.
- Do not add comments describing what you changed
- Prefer functional programming
" \
            --allowedTools "Read,Edit,Write,Bash(uv run*),Bash(git add*),Bash(git commit*),Bash(git status*),Bash(git diff*),Bash(ls*),Bash(python3*),Grep,Glob"

        if uv run pytest "tests/subsystems/test_${NAME}.py" -v 2>/dev/null; then
            PASSED=true
            break
        else
            echo "Tests failed on attempt $ATTEMPT"
        fi
    done

    if [ "$PASSED" = false ]; then
        echo "FAILED after $MAX_RETRIES attempts on subsystem $ID: $NAME"
        echo "Stopping loop. Fix manually, then re-run this script."
        exit 1
    fi

    COMPLETED=$((COMPLETED + 1))
    echo ""
    echo "Subsystem $ID ($NAME) complete. [$COMPLETED done]"
done

echo ""
echo "Finished. $COMPLETED subsystems completed."
