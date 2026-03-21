"""Mutation testing for JaxBorg differential test suite.

Introduces small, targeted bugs into JaxBorg and checks whether the
differential fuzzer catches them.  Each mutation runs in a subprocess
to avoid JIT cache interference.

Usage:
    uv run python scripts/mutation_test.py [--seeds 3] [--steps 30]
"""

import argparse
import json
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class MutationDef:
    name: str
    description: str
    module: str
    target_fn: str
    patch_code: str  # Python code that replaces the function body


MUTATIONS: list[MutationDef] = [
    MutationDef(
        name="exploit_skip_session_creation",
        description="SSH exploit succeeds but does not create a session",
        module="jaxborg.actions.red_exploit",
        target_fn="apply_red_exploit_ssh",
        patch_code=textwrap.dedent("""\
            # Mutation: return state unchanged (no session creation)
            def mutant(state, const, agent_id, target_host, key):
                return state
        """),
    ),
    MutationDef(
        name="monitor_skip_aging",
        description="Monitor does not age old_host_activity_detected",
        module="jaxborg.actions.blue_monitor",
        target_fn="apply_blue_monitor",
        patch_code=textwrap.dedent("""\
            import functools
            _orig = __orig_fn__
            @functools.wraps(_orig)
            def mutant(state, const, agent_id=None):
                result = _orig(state, const, agent_id)
                # Mutation: keep old activity flags unchanged
                return result.replace(old_host_activity_detected=state.old_host_activity_detected)
        """),
    ),
    MutationDef(
        name="reward_flip_sign",
        description="Negate the reward signal",
        module="jaxborg.rewards",
        target_fn="compute_rewards",
        patch_code=textwrap.dedent("""\
            import functools
            _orig = __orig_fn__
            @functools.wraps(_orig)
            def mutant(*args, **kwargs):
                return -_orig(*args, **kwargs)
        """),
    ),
    MutationDef(
        name="remove_always_noop",
        description="Blue Remove never removes any sessions",
        module="jaxborg.actions.blue_remove",
        target_fn="apply_blue_remove",
        patch_code=textwrap.dedent("""\
            def mutant(state, const, agent_id, target_host):
                return state
        """),
    ),
    MutationDef(
        name="discover_always_fails",
        description="Red DiscoverRemoteSystems never discovers hosts",
        module="jaxborg.actions.red_discover",
        target_fn="apply_red_discover",
        patch_code=textwrap.dedent("""\
            def mutant(state, const, agent_id, target_subnet, key):
                return state
        """),
    ),
    MutationDef(
        name="phase_never_advances",
        description="Mission phase stays at 0",
        module="jaxborg.rewards",
        target_fn="advance_mission_phase",
        patch_code=textwrap.dedent("""\
            def mutant(state, const):
                return state
        """),
    ),
    MutationDef(
        name="compromise_level_off_by_one",
        description="Set compromise to PRIVILEGED instead of USER on exploit",
        module="jaxborg.actions.red_common",
        target_fn="apply_exploit_success",
        patch_code=textwrap.dedent("""\
            import functools
            _orig = __orig_fn__
            @functools.wraps(_orig)
            def mutant(state, const, agent_id, target_host, key):
                result = _orig(state, const, agent_id, target_host, key)
                # Mutation: always set privilege to 2 (PRIVILEGED) instead of 1 (USER)
                return result.replace(
                    host_compromised=result.host_compromised.at[target_host].set(2),
                    red_privilege=result.red_privilege.at[agent_id, target_host].set(2),
                )
        """),
    ),
    MutationDef(
        name="scan_marks_wrong_host",
        description="Scan marks host 0 as scanned instead of the target",
        module="jaxborg.actions.red_scan",
        target_fn="apply_red_scan",
        patch_code=textwrap.dedent("""\
            import functools
            _orig = __orig_fn__
            @functools.wraps(_orig)
            def mutant(state, const, agent_id, target_host, key):
                result = _orig(state, const, agent_id, target_host, key)
                # Mutation: also mark host 0 as scanned
                return result.replace(
                    red_scanned_hosts=result.red_scanned_hosts.at[agent_id, 0].set(True)
                )
        """),
    ),
]


_RUNNER_TEMPLATE = textwrap.dedent("""\
import sys
sys.path.insert(0, "{root}")

import importlib
from unittest.mock import patch

# Import the module and get the original function
mod = importlib.import_module("{module}")
__orig_fn__ = getattr(mod, "{target_fn}")

# Build the mutant
exec_globals = {{"__orig_fn__": __orig_fn__}}
exec('''{patch_code}''', exec_globals)
mutant_fn = exec_globals["mutant"]

# Patch and run the fuzzer
with patch.object(mod, "{target_fn}", mutant_fn):
    from tests.differential.fuzzer import run_differential_fuzz
    report = run_differential_fuzz(
        seeds=range({seeds}),
        max_steps_per_seed={steps},
        mismatch_mode="error",
    )
    if report is not None:
        print(f"CAUGHT at seed={{report.seed}} step={{report.step}}: {{report.field_name}}")
        sys.exit(0)
    else:
        print("SURVIVED")
        sys.exit(1)
""")


def run_mutation(mutation: MutationDef, seeds: int, steps: int, verbose: bool) -> tuple[bool, str]:
    """Run a single mutation test. Returns (caught: bool, detail: str)."""
    script = _RUNNER_TEMPLATE.format(
        root=str(ROOT),
        module=mutation.module,
        target_fn=mutation.target_fn,
        patch_code=mutation.patch_code.replace("'", "\\'"),
        seeds=seeds,
        steps=steps,
    )

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(ROOT),
        )
        output = result.stdout.strip()
        if verbose:
            if result.stderr:
                print(f"  stderr: {result.stderr[-200:]}")
        caught = result.returncode == 0
        return caught, output
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, f"ERROR: {e}"


def main():
    parser = argparse.ArgumentParser(description="Mutation testing for JaxBorg")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds per mutation")
    parser.add_argument("--steps", type=int, default=30, help="Steps per seed")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    args = parser.parse_args()

    results = []
    caught_count = 0
    total = len(MUTATIONS)

    print(f"Running {total} mutations ({args.seeds} seeds x {args.steps} steps each)\n")

    for i, mutation in enumerate(MUTATIONS, 1):
        print(f"[{i}/{total}] {mutation.name}: {mutation.description}")
        t0 = time.time()
        caught, detail = run_mutation(mutation, args.seeds, args.steps, args.verbose)
        elapsed = time.time() - t0

        status = "CAUGHT" if caught else "SURVIVED"
        print(f"  -> {status} ({elapsed:.1f}s) {detail}\n")

        if caught:
            caught_count += 1

        results.append(
            {
                "name": mutation.name,
                "description": mutation.description,
                "caught": caught,
                "detail": detail,
                "elapsed_s": round(elapsed, 1),
            }
        )

    score = caught_count / total * 100 if total > 0 else 0
    print("=" * 60)
    print(f"Mutation Score: {caught_count}/{total} ({score:.0f}%) caught")
    print("=" * 60)

    survived = [r for r in results if not r["caught"]]
    if survived:
        print("\nSurviving mutations (TEST GAPS):")
        for r in survived:
            print(f"  - {r['name']}: {r['description']}")

    if args.json:
        report = {"score": score, "caught": caught_count, "total": total, "mutations": results}
        report_path = ROOT / "mutation_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        print(f"\nJSON report written to {report_path}")


if __name__ == "__main__":
    main()
