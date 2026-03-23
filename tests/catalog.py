import json
from dataclasses import dataclass, field
from pathlib import Path

CATALOG_STATUS_PATH = Path(__file__).parent / "catalog_status.json"


@dataclass
class Subsystem:
    id: int
    name: str
    description: str
    depends_on: list[int] = field(default_factory=list)
    cyborg_source_paths: list[str] = field(default_factory=list)
    jax_target_files: list[str] = field(default_factory=list)
    verification_level: str = "L1"  # L1=property, L2=interaction, L3=rollout, L4=transfer


SUBSYSTEMS: list[Subsystem] = [
    # --- L1: Property tests (individual component parity) ---
    Subsystem(
        id=1,
        name="static_topology",
        description="Static topology construction: hosts, subnets, adjacency, host properties",
        depends_on=[],
        cyborg_source_paths=[
            "CybORG/Simulator/Scenarios/EnterpriseScenarioGenerator.py",
            "CybORG/Simulator/State.py",
        ],
        jax_target_files=["src/jaxborg/topology.py", "src/jaxborg/state.py"],
        verification_level="L1",
    ),
    Subsystem(
        id=2,
        name="red_discover",
        description="Red: discover remote systems",
        depends_on=[1],
        cyborg_source_paths=[
            "CybORG/Simulator/Actions/ConcreteActions/DiscoverRemoteSystems.py",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
        verification_level="L1",
    ),
    Subsystem(
        id=3,
        name="red_scan",
        description="Red: scan host (DiscoverNetworkServices)",
        depends_on=[2],
        cyborg_source_paths=[
            "CybORG/Simulator/Actions/ConcreteActions/DiscoverNetworkServices.py",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
        verification_level="L1",
    ),
    Subsystem(
        id=4,
        name="red_exploit_ssh",
        description="Red: exploit — SSH brute force (simplest exploit)",
        depends_on=[3],
        cyborg_source_paths=[
            "CybORG/Simulator/Actions/ConcreteActions/ExploitActions/SSHBruteForce.py",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
        verification_level="L1",
    ),
    Subsystem(
        id=5,
        name="red_exploit_remaining",
        description="Red: exploit — remaining types (FTP, HTTP, HTTPS, Haraka, SQL, EternalBlue, BlueKeep)",
        depends_on=[4],
        cyborg_source_paths=[
            "CybORG/Simulator/Actions/ConcreteActions/ExploitActions/",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
        verification_level="L1",
    ),
    Subsystem(
        id=6,
        name="red_privesc",
        description="Red: privilege escalation",
        depends_on=[4],
        cyborg_source_paths=[
            "CybORG/Simulator/Actions/ConcreteActions/EscalateActions/",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
        verification_level="L1",
    ),
    Subsystem(
        id=7,
        name="red_impact",
        description="Red: impact",
        depends_on=[6],
        cyborg_source_paths=[
            "CybORG/Simulator/Actions/ConcreteActions/Impact.py",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
        verification_level="L1",
    ),
    Subsystem(
        id=8,
        name="blue_monitor",
        description="Blue: monitor (activity detection)",
        depends_on=[1],
        cyborg_source_paths=[
            "CybORG/Simulator/Actions/ConcreteActions/Monitor.py",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
        verification_level="L1",
    ),
    Subsystem(
        id=9,
        name="blue_analyse",
        description="Blue: analyse",
        depends_on=[8],
        cyborg_source_paths=[
            "CybORG/Simulator/Actions/ConcreteActions/Analyse.py",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
        verification_level="L1",
    ),
    Subsystem(
        id=10,
        name="blue_remove",
        description="Blue: remove",
        depends_on=[9],
        cyborg_source_paths=[
            "CybORG/Simulator/Actions/ConcreteActions/Remove.py",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
        verification_level="L1",
    ),
    Subsystem(
        id=11,
        name="blue_restore",
        description="Blue: restore",
        depends_on=[10],
        cyborg_source_paths=[
            "CybORG/Simulator/Actions/ConcreteActions/Restore.py",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
        verification_level="L1",
    ),
    Subsystem(
        id=12,
        name="blue_decoys",
        description="Blue: decoys (all types, OS restrictions, exploit blocking matrix)",
        depends_on=[11],
        cyborg_source_paths=[
            "CybORG/Simulator/Actions/ConcreteActions/DecoyActions/",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
        verification_level="L1",
    ),
    Subsystem(
        id=13,
        name="rewards",
        description="Rewards (confidentiality + availability + restore/impact costs)",
        depends_on=[7, 11],
        cyborg_source_paths=[
            "CybORG/Shared/BlueRewardMachine.py",
        ],
        jax_target_files=["src/jaxborg/rewards.py"],
        verification_level="L1",
    ),
    Subsystem(
        id=14,
        name="observations",
        description="Observations (blue obs encoding per agent, red obs encoding)",
        depends_on=[8],
        cyborg_source_paths=[
            "CybORG/Agents/Wrappers/BlueFlatWrapper.py",
            "CybORG/Agents/Wrappers/EnterpriseMAE.py",
        ],
        jax_target_files=["src/jaxborg/observations.py"],
        verification_level="L1",
    ),
    # --- L2: Interaction tests (cross-module state dependencies) ---
    Subsystem(
        id=15,
        name="phase_transitions",
        description="Phase transitions (0->1->2, reward weight changes, allowed subnet pairs)",
        depends_on=[13],
        cyborg_source_paths=[
            "CybORG/Simulator/Scenarios/EnterpriseScenarioGenerator.py",
            "CybORG/Shared/BlueRewardMachine.py",
        ],
        jax_target_files=["src/jaxborg/state.py", "src/jaxborg/rewards.py"],
        verification_level="L2",
    ),
    Subsystem(
        id=16,
        name="green_agents",
        description="Green agents (false positives, phishing)",
        depends_on=[1],
        cyborg_source_paths=[
            "CybORG/Agents/SimpleAgents/EnterpriseGreenAgent.py",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
        verification_level="L2",
    ),
    Subsystem(
        id=17,
        name="blue_traffic_zones",
        description="Blue: block/allow traffic zones",
        depends_on=[11],
        cyborg_source_paths=[
            "CybORG/Simulator/Actions/ConcreteActions/ControlTraffic.py",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
        verification_level="L2",
    ),
    Subsystem(
        id=18,
        name="multi_agent_messaging",
        description="Multi-agent observations and messaging (8-bit inter-agent vectors)",
        depends_on=[14],
        cyborg_source_paths=[
            "CybORG/Agents/Wrappers/EnterpriseMAE.py",
        ],
        jax_target_files=["src/jaxborg/observations.py"],
        verification_level="L2",
    ),
    Subsystem(
        id=19,
        name="dynamic_topology",
        description="Dynamic topology (pad-to-max with host_active masking)",
        depends_on=[1],
        cyborg_source_paths=[
            "CybORG/Simulator/Scenarios/EnterpriseScenarioGenerator.py",
        ],
        jax_target_files=["src/jaxborg/topology.py", "src/jaxborg/state.py"],
        verification_level="L2",
    ),
    Subsystem(
        id=20,
        name="fsm_red_agent",
        description="FiniteStateRedAgent (8-state FSM, probabilistic transitions)",
        depends_on=[7],
        cyborg_source_paths=[
            "CybORG/Agents/SimpleAgents/FiniteStateRedAgent.py",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
        verification_level="L2",
    ),
    Subsystem(
        id=21,
        name="cc4_new_red_actions",
        description="CC4-new red actions (AggressiveServiceDiscovery, StealthServiceDiscovery, etc.)",
        depends_on=[5],
        cyborg_source_paths=[
            "CybORG/Simulator/Actions/ConcreteActions/",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
        verification_level="L2",
    ),
    # --- L3: Rollout comparison (full episodes, matched seeds) ---
    Subsystem(
        id=22,
        name="full_episode_fuzzing",
        description="Full episode fuzzing (all subsystems, 200+ seeds, extended episodes)",
        depends_on=list(range(1, 22)),
        cyborg_source_paths=[],
        jax_target_files=[],
        verification_level="L3",
    ),
]

SUBSYSTEMS_BY_ID = {s.id: s for s in SUBSYSTEMS}


def _load_status() -> dict[int | str, object]:
    if not CATALOG_STATUS_PATH.exists():
        return {}
    raw_status = json.loads(CATALOG_STATUS_PATH.read_text())
    status: dict[int | str, object] = {}
    for key, value in raw_status.items():
        try:
            parsed_key: int | str = int(key)
        except ValueError:
            parsed_key = key
        status[parsed_key] = value
    return status


def _save_status(status: dict[int | str, object]) -> None:
    sorted_items = sorted(status.items(), key=lambda x: str(x[0]))
    CATALOG_STATUS_PATH.write_text(json.dumps({str(k): v for k, v in sorted_items}, indent=2) + "\n")


def get_next_incomplete() -> Subsystem | None:
    status = _load_status()
    for s in SUBSYSTEMS:
        if status.get(s.id) == "passing":
            continue
        deps_met = all(status.get(d) == "passing" for d in s.depends_on)
        if deps_met:
            return s
    return None


def mark_passing(subsystem_id: int) -> None:
    status = _load_status()
    status[subsystem_id] = "passing"
    _save_status(status)


def is_all_done() -> bool:
    status = _load_status()
    return all(status.get(s.id) == "passing" for s in SUBSYSTEMS)


def coverage_summary() -> dict:
    """Return verification coverage summary by level (L1/L2/L3/L4).

    Counts passing/total subsystems at each verification level and loads
    L3/L4 results from catalog_status.json if present.
    """
    status = _load_status()
    by_level: dict[str, dict] = {}
    for level in ("L1", "L2", "L3", "L4"):
        subs = [s for s in SUBSYSTEMS if s.verification_level == level]
        passing = sum(1 for s in subs if status.get(s.id) == "passing")
        by_level[level] = {
            "total": len(subs),
            "passing": passing,
            "subsystems": [s.name for s in subs],
        }

    # L3 rollout coverage from status file
    l3_coverage = status.get("l3_coverage")
    if l3_coverage:
        by_level["L3"]["rollout_coverage"] = l3_coverage

    # L4 transfer result from status file
    l4_result = status.get("l4_tost")
    if l4_result:
        by_level["L4"]["tost_result"] = l4_result

    return by_level


def update_l3_coverage(seeds: int, steps: int, clean: bool) -> None:
    """Record L3 rollout coverage in catalog status."""
    status = _load_status()
    status["l3_coverage"] = {"seeds": seeds, "steps": steps, "clean": clean}
    _save_status(status)


def update_l4_tost(equivalent: bool, margin: float, mean_diff: float, episodes: int) -> None:
    """Record L4 TOST equivalence result in catalog status."""
    status = _load_status()
    status["l4_tost"] = {
        "equivalent": equivalent,
        "margin": margin,
        "mean_diff": mean_diff,
        "episodes": episodes,
    }
    _save_status(status)


VERIFICATION_STATUS_PATH = Path(__file__).parent.parent / ".agent_handoff" / "verification_status.json"

_VERIFICATION_LEVELS = ("l1", "l2", "l3", "l4")
_DEFAULT_VERIFICATION = {lvl: {"status": "unknown", "iterations": 0} for lvl in _VERIFICATION_LEVELS}


def load_verification_status() -> dict:
    """Load the Karten loop verification status."""
    if VERIFICATION_STATUS_PATH.exists():
        return json.loads(VERIFICATION_STATUS_PATH.read_text())
    return dict(_DEFAULT_VERIFICATION)


def update_verification_status(level: str, status: str, **details) -> None:
    """Update verification status for a level."""
    data = load_verification_status()
    entry = data.setdefault(level, {"status": "unknown", "iterations": 0})
    entry["status"] = status
    entry["iterations"] = entry.get("iterations", 0) + 1
    entry.update(details)
    VERIFICATION_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    VERIFICATION_STATUS_PATH.write_text(json.dumps(data, indent=2))


def get_current_level() -> str | None:
    """Return the lowest non-passing verification level, or None if all pass."""
    data = load_verification_status()
    for lvl in _VERIFICATION_LEVELS:
        if data.get(lvl, {}).get("status") != "passing":
            return lvl
    return None


def print_coverage_summary() -> None:
    """Print hierarchical verification coverage summary."""
    summary = coverage_summary()
    print("\n" + "=" * 60)
    print("HIERARCHICAL VERIFICATION COVERAGE")
    print("=" * 60)
    for level, data in summary.items():
        total = data["total"]
        passing = data["passing"]

        if level == "L3":
            if "rollout_coverage" in data:
                rc = data["rollout_coverage"]
                clean_str = "CLEAN" if rc["clean"] else "FAILING"
                print(f"  L3: {rc['seeds']} seeds x {rc['steps']} steps = {clean_str}")
            else:
                pct = (passing / total * 100) if total > 0 else 0
                print(f"  L3: {passing}/{total} ({pct:.0f}%) — no rollout data yet")

        elif level == "L4":
            if "tost_result" in data:
                tr = data["tost_result"]
                eq_str = "EQUIVALENT" if tr["equivalent"] else "NOT EQUIVALENT"
                print(f"  L4: TOST {eq_str} (margin={tr['margin']}, diff={tr['mean_diff']:+.2f}, n={tr['episodes']})")
            else:
                print("  L4: no transfer evaluation yet")

        else:
            pct = (passing / total * 100) if total > 0 else 0
            print(f"  {level}: {passing}/{total} ({pct:.0f}%)")

    print("=" * 60)
