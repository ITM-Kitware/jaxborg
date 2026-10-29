"""Capture local Torch checkpoint evaluations as retained experiment records."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from jaxborg.evaluation.capture_store import (
    Capture,
    add_artifact,
    declare_protocol,
    finish_evaluation,
    load_records,
    record_episode,
    start_evaluation,
)

_SOURCE_ROOT = Path(__file__).resolve().parents[3]
_CHECKPOINT_ROLE = "policy checkpoint"
_RECIPE_ROLE = "policy recipe"


def _outside_source(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_relative_to(_SOURCE_ROOT):
        raise ValueError("Generated evidence must be outside the source checkout")
    return path


def _sidecar_path(model: Path) -> Path:
    stem = model.stem.removeprefix("model_")
    candidates = [model.with_name(f"recipe_{stem}.{suffix}") for suffix in ("yaml", "yml")]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Required recipe sidecar not found next to {model}")


def _read_recipe(path: Path) -> dict:
    import yaml

    recipe = yaml.safe_load(path.read_text())
    if not isinstance(recipe, dict) or not isinstance(recipe.get("arch"), dict):
        raise ValueError("Recipe sidecar must contain an arch mapping")
    if not isinstance(recipe["arch"].get("name"), str):
        raise ValueError("Recipe sidecar must declare arch.name")
    return recipe


def _integer(value, name, *, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")


def _metric_contract(protocol):
    metrics = protocol.get("metric_definitions", [])
    metric = next((metric for metric in metrics if metric.get("name") == "blue_return"), {})
    if (
        metric.get("unit") != "score"
        or metric.get("aggregation") not in {"mean", "arithmetic mean"}
        or "episode" not in metric.get("sampling_unit", "").lower()
        or "polic" not in metric.get("sampling_unit", "").lower()
        or metric.get("uncertainty") != "none"
    ):
        raise ValueError(
            "Protocol must define blue_return in score units as an arithmetic mean over fixed-policy episodes, "
            "with uncertainty none"
        )


def _fact(value, source="configuration.json"):
    return {"value": value, "source": source}


def run_evaluation(
    *,
    root: Path,
    protocol: dict,
    trial_id: str,
    model: Path,
    seeds: list[int],
    horizon: int,
    retry_of: str | None = None,
    deterministic: bool = False,
    policy_seed: int = 0,
    allow_dirty: bool = False,
) -> Capture:
    """Retain and evaluate a checkpoint, sealing failures before re-raising them.

    Invalid launch settings and recipe sidecars fail before an attempt is
    declared. Once started, input-copy, model-load, and rollout failures are
    retained. The protocol's trial identities and selection remain explicit.
    """
    from jaxborg.constants import BLUE_OBS_SIZE
    from jaxborg.evaluation.cyborg_runner import torch_policy_configuration
    from jaxborg.evaluation.provenance import collect_provenance
    from jaxborg.recipe import eval_variant

    root = _outside_source(root)
    model = Path(model).expanduser().resolve()
    if model.suffix != ".pt":
        raise ValueError("Evaluation capture currently requires a Torch .pt checkpoint")
    _integer(horizon, "horizon", minimum=1, maximum=500)
    _integer(policy_seed, "policy_seed", maximum=2**63 - 1)
    if not isinstance(deterministic, bool):
        raise ValueError("deterministic must be a boolean")
    if not seeds:
        raise ValueError("At least one explicit environment seed is required")
    for seed in seeds:
        _integer(seed, "environment seed", maximum=2**32 - 1)
    if len(set(seeds)) != len(seeds):
        raise ValueError("Environment seeds must be unique")
    _integer(policy_seed + len(seeds) - 1, "last policy seed", maximum=2**63 - 1)
    _metric_contract(protocol)
    recipe_path = _sidecar_path(model)
    recipe = _read_recipe(recipe_path)
    variant = replace(eval_variant(recipe), num_steps=horizon)
    provenance = collect_provenance(allow_dirty=allow_dirty)
    policy_seeds = [policy_seed + index for index in range(len(seeds))]
    config = {
        "schema_version": 1,
        "eval_env": "cyborg",
        "policy_backend": "torch",
        "model_filename": model.name,
        "recipe_filename": recipe_path.name,
        "recipe": recipe,
        "policy": torch_policy_configuration(recipe),
        "variant": asdict(variant),
        "horizon": horizon,
        "seeds": list(seeds),
        "policy_seeds": policy_seeds,
        "episodes_per_seed": 1,
        "deterministic": deterministic,
        "policy_device": "cpu",
        "observation_size": BLUE_OBS_SIZE,
        "action_size": 242,
        "blue_agents": 5,
        "observation_contract": "five padded BlueEnterpriseWrapper observations; BLUE_OBS_SIZE entries per agent",
        "action_contract": "242 EnterpriseMAE action indices per Blue agent, masked by the live action mask",
        "reward_contract": "sum of blue_agent_0 rewards returned by EnterpriseMAE.step across completed episode steps",
        "wrapper": "CybORG.Agents.Wrappers.EnterpriseMAE",
        "procedure": "jaxborg.evaluation.cyborg_runner.load_torch_policy and rollout_episode",
        "execution": provenance.configuration,
    }
    facts = {
        **provenance.facts,
        "eval_env": _fact("cyborg"),
        "policy_backend": _fact("torch"),
        "game_contract": _fact(asdict(variant)),
        "seeds": _fact(list(seeds)),
        "policy_seeds": _fact(policy_seeds),
        "policies": _fact({"blue": {"checkpoint_role": _CHECKPOINT_ROLE}, "red": {"scripted": variant.red_agent}}),
        "step_accounting": _fact("Step counts cover completed episodes; partial episode steps are not retained"),
        "policy_device": _fact("cpu"),
    }
    capture = start_evaluation(
        root,
        protocol=protocol,
        trial_id=trial_id,
        retry_of=retry_of,
        requested_episodes=len(seeds),
        resolved_config=config,
        facts=facts,
        inputs={_CHECKPOINT_ROLE: model, _RECIPE_ROLE: recipe_path},
        source_files=provenance.source_files,
    )
    try:
        retained_hashes = json.loads((capture.directory / "source-files.json").read_text())
        if retained_hashes != provenance.hashes:
            raise ValueError("Retained evaluation source differs from its captured identity")
        if _read_recipe(capture.inputs[_RECIPE_ROLE]) != recipe:
            raise ValueError("Recipe sidecar changed while its input bytes were retained")

        import torch
        from CybORG.Agents.Wrappers import EnterpriseMAE

        from jaxborg.constants import CC4_CONFIG
        from jaxborg.evaluation import cyborg_runner
        from jaxborg.scenarios.cc4.topology import build_const_from_cyborg, save_topology

        completed = []
        topologies = []
        # Policy construction initializes parameters before loading their saved values.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(policy_seed)
            policy, loaded_recipe = cyborg_runner.load_torch_policy(capture.inputs[_CHECKPOINT_ROLE])
        if loaded_recipe != recipe:
            raise ValueError("Loaded policy recipe differs from the retained configuration")
        devices = sorted({str(parameter.device) for parameter in policy.parameters()})
        if devices != ["cpu"]:
            raise ValueError("Retained policy must execute on its declared CPU device")

        with TemporaryDirectory(prefix="jaxborg-evaluation-") as temporary:
            temporary = Path(temporary)
            runtime = temporary / "runtime.json"
            runtime.write_text(
                json.dumps(
                    {
                        "policy_devices": devices,
                        "policy_class": f"{type(policy).__module__}.{type(policy).__qualname__}",
                        "wrapper": f"{EnterpriseMAE.__module__}.{EnterpriseMAE.__qualname__}",
                        "observation_size": cyborg_runner.OBS_DIM,
                        "action_size": cyborg_runner.ACT_DIM,
                        "blue_agents": cyborg_runner.AGENT_IDS,
                        "reward_agent": cyborg_runner.AGENT_IDS[0],
                    },
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            add_artifact(capture, runtime, role="evaluation runtime")
            for index, (seed, episode_policy_seed) in enumerate(zip(seeds, policy_seeds)):

                def on_reset(env, reset, *, index=index, seed=seed):
                    topology = temporary / f"episode-{index}-topology.npz"
                    save_topology(
                        build_const_from_cyborg(env.env),
                        topology,
                        metadata={
                            "source": "cyborg evaluation reset",
                            "source_seed": seed,
                            "episode_index": index,
                            "variant": asdict(variant),
                            "role_map": reset.role_map,
                        },
                        scenario_config=replace(CC4_CONFIG, max_steps=horizon),
                    )
                    artifact_id = add_artifact(capture, topology, role="evaluation reset topology")
                    topologies.append(artifact_id)
                    metadata = temporary / f"episode-{index}-reset.json"
                    metadata.write_text(
                        json.dumps(
                            {
                                "episode_index": index,
                                "seed": seed,
                                "role_map": reset.role_map,
                                "topology_id": artifact_id,
                            },
                            sort_keys=True,
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    add_artifact(capture, metadata, role="evaluation reset metadata")

                def on_complete(reward, steps, *, index=index, seed=seed, episode_policy_seed=episode_policy_seed):
                    if not 0 < steps <= horizon:
                        raise ValueError("Completed episode step count exceeds the declared horizon")
                    record_episode(
                        capture,
                        episode_index=index,
                        seed=seed,
                        reward=reward,
                        policy_seed=episode_policy_seed,
                        steps=steps,
                    )
                    completed.append(index)

                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(episode_policy_seed)
                    env = cyborg_runner.make_cyborg_env(variant, seed, wrapper_class=EnterpriseMAE)
                    cyborg_runner.rollout_episode(
                        env,
                        variant,
                        seed,
                        policy,
                        deterministic=deterministic,
                        on_reset=on_reset,
                        on_complete=on_complete,
                    )
        if completed != list(range(len(seeds))) or len(topologies) != len(seeds):
            raise ValueError(
                "Successful evaluation requires an observed topology and result for every requested episode"
            )
        provenance.verify_sources()
        load_records(root, verify=True)
        finish_evaluation(capture, status="succeeded")
    except BaseException as error:
        error._evaluation_capture = capture
        finish_evaluation(
            capture, status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed", error=error
        )
        raise
    return capture


def export_records(*, root: Path, output: Path) -> list[dict]:
    """Export verified canonical records without changing any selection."""
    root = _outside_source(root)
    output = _outside_source(output)
    if output.is_relative_to(root):
        raise ValueError("Export output must be separate from the retained evidence store")
    records = load_records(root, verify=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(records, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return records


def _protocol(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Protocol file must contain one protocol record")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Evaluate and retain one attempt for an explicitly named trial")
    run.add_argument("--root", required=True, type=Path)
    run.add_argument("--protocol", required=True, type=Path)
    run.add_argument("--trial", required=True)
    run.add_argument("--model", required=True, type=Path)
    run.add_argument("--seeds", required=True, nargs="+", type=int)
    run.add_argument("--horizon", required=True, type=int)
    run.add_argument("--retry-of")
    run.add_argument("--deterministic", action="store_true")
    run.add_argument("--policy-seed", type=int, default=0)
    run.add_argument("--allow-dirty", action="store_true")
    export = commands.add_parser("export", help="Export records after verifying every retained artifact")
    export.add_argument("--root", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    declare = commands.add_parser("declare", help="Retain the protocol including trials that may never launch")
    declare.add_argument("--root", required=True, type=Path)
    declare.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            capture = run_evaluation(
                root=args.root,
                protocol=_protocol(args.protocol),
                trial_id=args.trial,
                model=args.model,
                seeds=args.seeds,
                horizon=args.horizon,
                retry_of=args.retry_of,
                deterministic=args.deterministic,
                policy_seed=args.policy_seed,
                allow_dirty=args.allow_dirty,
            )
            result = {
                "attempt_id": capture.attempt_id,
                "evaluation_id": capture.evaluation_id,
                "status": "succeeded",
                "completed_episodes": len(args.seeds),
            }
        elif args.command == "export":
            records = export_records(root=args.root, output=args.output)
            result = {"output": str(args.output.expanduser().resolve()), "records": len(records)}
        else:
            protocol = _protocol(args.protocol)
            _metric_contract(protocol)
            declare_protocol(_outside_source(args.root), protocol)
            result = {"protocol_id": protocol["id"], "status": "declared"}
    except (Exception, KeyboardInterrupt) as error:
        failure = {
            "status": "cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        retained = getattr(error, "_evaluation_capture", None)
        if retained is not None:
            failure.update(attempt_id=retained.attempt_id, evaluation_id=retained.evaluation_id)
        print(
            json.dumps(failure, sort_keys=True),
            file=sys.stderr,
        )
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
