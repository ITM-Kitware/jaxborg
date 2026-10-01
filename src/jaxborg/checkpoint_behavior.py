"""Read-only action/reward diagnostics checked against the pinned matchup evaluator."""

import argparse
import csv
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxborg.actions import action_defs as action
from jaxborg.evaluation.matchup_runner import DEFAULT_EVAL_BATCH_SIZE, MatchupEvaluationContext
from jaxborg.policies import policy_step
from jaxborg.recipe import eval_variant, load
from jaxborg.tracking import Run, assigned_devices, file_hash, input_artifact, resolve_artifact

COMPONENTS = ("reward_ria", "reward_lwf", "reward_asf", "action_cost")
COUNTERS = ("impact_count", "green_lwf_count", "green_asf_count")
ACTION_NAMES = ("sleep", "monitor", "analyse", "remove", "restore", "decoy", "block", "allow")


def _action_type(actions):
    kind = jnp.where(actions == action.BLUE_MONITOR, 1, 0)
    for index, (start, end) in enumerate(
        (
            (action.BLUE_ANALYSE_START, action.BLUE_ANALYSE_END),
            (action.BLUE_REMOVE_START, action.BLUE_REMOVE_END),
            (action.BLUE_RESTORE_START, action.BLUE_RESTORE_END),
            (action.BLUE_DECOY_START, action.BLUE_DECOY_END),
            (action.BLUE_BLOCK_TRAFFIC_START, action.BLUE_BLOCK_TRAFFIC_END),
            (action.BLUE_ALLOW_TRAFFIC_START, action.BLUE_ALLOW_TRAFFIC_END),
        ),
        2,
    ):
        kind = jnp.where((actions >= start) & (actions < end), index, kind)
    return kind


def _episode(blue_weights, red_weights, key, *, env, blue_module, red_module, steps):
    # Same stock feedforward-policy RNG/reset/transition sequence as the original
    # _run_jax_matchup_episode_scan. Instrumentation observes, never changes actions.
    key, reset_key = jax.random.split(key)
    obs, state = env.reset_at_topology(reset_key, jnp.int32(0))
    blue_names, red_names = tuple(env.blue_agents), tuple(env.red_agents)
    initial = {
        name: jnp.float32(0)
        for name in (
            *COMPONENTS,
            *COUNTERS,
            "blue_return",
            "blocked_pair_ticks",
            "illegal_actions",
            "legal_choices",
            "idle_agent_ticks",
            *ACTION_NAMES,
        )
    }

    def step(carry, _):
        key, obs, state, active, totals = carry

        def advance(_):
            masks = env.get_avail_actions(state)
            key1, blue_key = jax.random.split(key)
            blue_obs = jnp.stack([obs[name] for name in blue_names])
            blue_masks = jnp.stack([masks[name] for name in blue_names])
            blue_pi, _, _ = policy_step(blue_module, blue_weights, blue_obs, blue_masks)
            blue_actions = blue_pi.sample(seed=blue_key)
            key2, red_key = jax.random.split(key1)
            red_obs = jnp.stack([obs[name] for name in red_names])
            red_masks = jnp.stack([masks[name] for name in red_names])
            red_pi, _, _ = policy_step(red_module, red_weights, red_obs, red_masks)
            red_actions = red_pi.sample(seed=red_key)
            actions = {
                **{name: blue_actions[i] for i, name in enumerate(blue_names)},
                **{name: red_actions[i] for i, name in enumerate(red_names)},
            }
            key3, step_key = jax.random.split(key2)
            transition_key, _ = jax.random.split(step_key)
            next_obs, next_state, rewards, dones, info = env.step_env(transition_key, state, actions)
            updated = dict(totals)
            for name in (*COMPONENTS, *COUNTERS):
                updated[name] += info[name]
            updated["blue_return"] += rewards[blue_names[0]]
            updated["blocked_pair_ticks"] += state.state.blocked_zones.sum()
            idle = state.state.blue_pending_ticks == 0
            updated["idle_agent_ticks"] += idle.sum()
            updated["legal_choices"] += (blue_masks.sum(axis=-1) * idle).sum()
            updated["illegal_actions"] += (
                ~jnp.take_along_axis(blue_masks, blue_actions[:, None], axis=1).squeeze(-1).astype(bool)
            ).sum()
            kinds = _action_type(blue_actions)
            for index, name in enumerate(ACTION_NAMES):
                updated[name] += ((kinds == index) & idle).sum()
            return key3, next_obs, next_state, ~dones["__all__"], updated

        return jax.lax.cond(active, advance, lambda _: carry, operand=None), None

    carry, _ = jax.lax.scan(step, (key, obs, state, jnp.bool_(True), initial), xs=None, length=steps)
    return carry[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--point", required=True)
    parser.add_argument("--evaluations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assigned_devices()
    manifest = json.loads(Path(args.manifest).read_text())
    point = manifest["points"][args.point]
    outcomes = json.loads(Path(args.evaluations).read_text())["confirmation"]
    config, protocol = manifest["config"], point["protocol"]
    count = config["behavior_trace_episodes"]
    if not 1 <= count <= config["episodes"]["confirmation"]["count"]:
        raise ValueError("trace episodes must be a bounded subset of confirmation")
    context = MatchupEvaluationContext()
    red = context.load_policy(resolve_artifact(protocol["source"]["checkpoint"]), team="red", backend="jax")
    original = context.load_policy(resolve_artifact(protocol["source"]["checkpoint"]), team="blue", backend="jax")
    variant = eval_variant(load(protocol["eval_recipe"]))
    topology = resolve_artifact(protocol["game"]["topology"])
    env = context.environment(variant, [topology])
    scan = partial(_episode, env=env, blue_module=original.module, red_module=red.module, steps=variant.num_steps)
    batched = jax.jit(jax.vmap(scan, in_axes=(None, None, 0)))
    candidates = {c["name"]: c for c in point["candidates"]}
    episode_rows, summaries, evidence = [], [], []
    for name, result in outcomes.items():
        candidate = candidates[name]
        model = resolve_artifact(candidate["checkpoint"])
        if file_hash(model) != candidate["sha256"]:
            raise ValueError("trace model changed")
        blue = context.load_policy(model, team="blue", backend="jax")
        if blue.module != original.module or protocol["source"]["policies"]["red"]["architecture"]["name"] != "shared":
            raise ValueError("trace supports matched feedforward shared policies only")
        # Preserve the original evaluator's batch width to reduce numerical differences.
        seed_list = result["per_episode_seeds"][:DEFAULT_EVAL_BATCH_SIZE]
        seed_list += [seed_list[-1]] * (DEFAULT_EVAL_BATCH_SIZE - len(seed_list))
        keys = jnp.stack([jax.random.PRNGKey(seed) for seed in seed_list])
        rows = {k: np.asarray(v)[:count] for k, v in jax.device_get(batched(blue.weights, red.weights, keys)).items()}
        expected = np.asarray(result["per_episode_blue_returns"][:count])
        matched = np.array_equal(rows["blue_return"], expected)
        evidence.append(
            {
                "candidate": name,
                "returns_match_canonical": matched,
                "max_return_difference": float(np.max(np.abs(rows["blue_return"] - expected))),
            }
        )
        if not matched:
            raise ValueError(f"instrumented trace differs from original evaluator: {evidence[-1]}")
        total_components = sum(rows[k] for k in COMPONENTS)
        if not np.allclose(total_components, rows["blue_return"], atol=0.01) or np.any(rows["illegal_actions"]):
            raise ValueError("trace accounting or action mask failed")
        summary = {"candidate": name, **{k: float(v.mean()) for k, v in rows.items()}}
        summary["mean_blocked_pairs"] = summary["blocked_pair_ticks"] / variant.num_steps
        summary["mean_legal_choices_when_idle"] = summary["legal_choices"] / summary["idle_agent_ticks"]
        for action_name in ACTION_NAMES:
            summary[action_name + "_idle_fraction"] = summary[action_name] / summary["idle_agent_ticks"]
        summaries.append(summary)
        for i, seed in enumerate(result["per_episode_seeds"][:count]):
            episode_rows.append({"candidate": name, "episode_seed": seed, **{k: float(v[i]) for k, v in rows.items()}})
    inputs = [input_artifact(candidates[name]["checkpoint"], role=name) for name in outcomes]
    inputs += [
        input_artifact(protocol["source"]["checkpoint"], role="frozen original Red"),
        input_artifact(protocol["game"]["topology"], role="source topology"),
    ]
    output = Path(args.output)
    with Run(
        {"meta": {"name": config["name"] + "-" + args.point + "-behavior"}},
        backend="jax",
        kind="comparison",
        config=manifest,
        inputs=inputs,
    ) as owner:
        payload = {
            "owner": owner.run_id,
            "trace_episodes": count,
            "evidence": evidence,
            "summaries": summaries,
            "episodes": episode_rows,
            "canonical_reference": f"runs:/{owner.run_id}/diagnostics/behavior.json",
        }
        owner.write_json("diagnostics/behavior.json", payload)
        owner.export("diagnostics/behavior.json", output)
        with owner.path("diagnostics/behavior-summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        owner.publish(owner.path("diagnostics/behavior-summary.csv"), "diagnostics/behavior-summary.csv")
        owner.export("diagnostics/behavior-summary.csv", output.with_name("behavior-summary.csv"))


if __name__ == "__main__":
    main()
