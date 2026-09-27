"""Compare native, muted, and foreign Blue messages against one fixed Red."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from jaxborg.evaluation.matchup_runner import MatchupEvaluationContext, evaluate_matchup
from jaxborg.recipe import eval_variant, load, project_eval


def transfer_scores(native, muted, foreign):
    gain = native - muted
    transfer = foreign - muted
    return dict(comm_gain=gain, transfer=transfer, transfer_fraction=transfer / gain if abs(gain) > 1e-8 else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--models", nargs="+", required=True, help="Blue checkpoints from independent training seeds")
    parser.add_argument("--red-path", required=True, help="Fixed learned opponent for every comparison")
    parser.add_argument("--policy-backend", choices=["jax", "cyborg"], default="jax")
    parser.add_argument("--seeds", default="1000,1001,1002", help="Comma-separated evaluation seeds")
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(set(args.models)) < 2:
        parser.error("provide at least two distinct Blue checkpoints")
    recipe = load(args.recipe)
    evaluation = project_eval(recipe, materialize_topologies=True)
    kwargs = dict(
        backend=args.policy_backend,
        variant=eval_variant(recipe),
        seeds=[int(s) for s in args.seeds.split(",")],
        episodes_per_seed=args.episodes_per_seed,
        deterministic=args.deterministic,
        topology_path=evaluation["TOPOLOGY_BANK"] or None,
        topology_sampling=recipe.get("eval", {}).get("topology_sampling", "exhaustive"),
        cia=recipe.get("eval", {}).get("cia"),
        context=MatchupEvaluationContext(),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        for actor in args.models:
            native = evaluate_matchup(actor, args.red_path, **kwargs)
            muted = evaluate_matchup(actor, args.red_path, mute=True, **kwargs)
            for sender in args.models:
                if sender == actor:
                    continue
                foreign = evaluate_matchup(actor, args.red_path, message_sender_path=sender, **kwargs)
                row = dict(
                    recipe=args.recipe,
                    actor=actor,
                    sender=sender,
                    red=args.red_path,
                    policy_backend=args.policy_backend,
                    native_mean=mean(native.blue_returns),
                    muted_mean=mean(muted.blue_returns),
                    foreign_mean=mean(foreign.blue_returns),
                    native_returns=native.blue_returns,
                    muted_returns=muted.blue_returns,
                    foreign_returns=foreign.blue_returns,
                    episode_seeds=native.episode_seeds,
                    episode_topology_paths=native.episode_topology_paths,
                    native_messages=native.per_episode_messages,
                    foreign_messages=foreign.per_episode_messages,
                    **transfer_scores(mean(native.blue_returns), mean(muted.blue_returns), mean(foreign.blue_returns)),
                )
                stream.write(json.dumps(row) + "\n")
                stream.flush()


if __name__ == "__main__":
    main()
