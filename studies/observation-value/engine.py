"""Recorded stepwise rollouts for full and fixed-preference Blue policies."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END, decode_blue_action
from jaxborg.evaluation.capture_store import (
    add_artifact,
    fail_evaluation,
    finish_evaluation,
    json_configuration,
    record_episode,
    start_evaluation,
)
from jaxborg.evaluation.joint_capture import fact, outside_source, retained_json
from jaxborg.evaluation.matchup_runner import load_matchup_policy
from jaxborg.evaluation.provenance import collect_provenance, topology_evidence
from jaxborg.evaluation.tracking import destination_configured
from jaxborg.policies import initial_carry, is_recurrent, policy_step
from jaxborg.policies.categorical import Categorical


@partial(jax.jit, static_argnames=("module", "mode"))
def blue_action(module, weights, observations, masks, carry, key, mode, baseline_logits):
    if mode == "blind":
        logits = jnp.where(masks > 0, baseline_logits, -1e10)
        distribution = Categorical(logits=logits)
    else:
        distribution, _, carry = policy_step(module, weights, observations, masks, carry=carry)
        logits = distribution.logits
    greedy = jnp.argmax(logits, axis=-1)
    selected = greedy if mode == "argmax" else distribution.sample(seed=key)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probabilities = jnp.exp(log_probs)
    top = jnp.sort(probabilities, axis=-1)[:, -2:]
    valid = jnp.sum(masks, axis=-1)
    entropy = -jnp.sum(jnp.where(probabilities > 0, probabilities * log_probs, 0.0), axis=-1)
    diagnostics = jnp.stack(
        (top[:, -1], top[:, -1] - top[:, -2], entropy / jnp.log(jnp.maximum(valid, 2)), greedy, valid),
        axis=-1,
    )
    return selected, carry, diagnostics, probabilities


@partial(jax.jit, static_argnames=("module",))
def red_action(module, weights, observations, masks, carry, reset, key):
    distribution, _, carry = policy_step(module, weights, observations, masks, carry=carry, reset=reset)
    return distribution.sample(seed=key), carry


@jax.jit
def action_targets(actions, const):
    kinds, hosts, _, _, _ = jax.vmap(lambda action, agent: decode_blue_action(action, agent, const))(
        actions, jnp.arange(actions.shape[0])
    )
    return kinds, hosts


def _carry_record(carry):
    if carry is None:
        return {}
    if isinstance(carry, dict):
        return {f"{name}_{key}": value for name, child in carry.items() for key, value in _carry_record(child).items()}
    if isinstance(carry, tuple):
        if len(carry) != 2:
            raise ValueError("Only two-leaf LSTM carries are supported by this study")
        return {"carry_c": np.asarray(carry[0]), "carry_h": np.asarray(carry[1])}
    return {"carry_h": np.asarray(carry)}


def _observation_layout(const):
    subnets = np.asarray(const.blue_obs_subnets)
    n_subnets = (subnets >= 0).sum(axis=-1).astype(np.int32)
    host_map, active = np.asarray(const.obs_host_map), np.asarray(const.host_active)
    host_counts = np.zeros(subnets.shape, dtype=np.int32)
    for agent in range(len(subnets)):
        for slot, subnet in enumerate(subnets[agent]):
            if subnet < 0:
                continue
            hosts = host_map[subnet, :16]
            valid = (hosts >= 0) & (hosts < len(active))
            host_counts[agent, slot] = np.count_nonzero(valid & active[np.clip(hosts, 0, len(active) - 1)])
    return n_subnets, host_counts


def episode(
    policies,
    env,
    seed,
    topology_index,
    mode,
    horizon,
    baseline_logits=None,
    collect=False,
    collection_stride=5,
):
    """Run one native-key episode and retain sparse incoming decision inputs."""
    if mode not in {"sampled", "blind", "argmax"}:
        raise ValueError(f"Unknown Blue selection: {mode}")
    if collection_stride < 1:
        raise ValueError("Decision collection stride must be positive")
    if mode == "blind" and baseline_logits is None:
        raise ValueError("Blind selection requires frozen baseline logits")
    rng, reset_key = jax.random.split(jax.random.PRNGKey(seed))
    observations, state = env.reset_at_topology(reset_key, jnp.int32(topology_index))
    teams = {"blue": tuple(env.blue_agents), "red": tuple(env.red_agents)}
    n_blue = len(teams["blue"])
    if baseline_logits is not None:
        baseline_logits = jnp.asarray(baseline_logits, dtype=jnp.float32)
        if baseline_logits.shape != (n_blue, BLUE_ALLOW_TRAFFIC_END):
            raise ValueError("Frozen baseline dimensions differ from Blue action dimensions")
        if not np.all(np.isfinite(np.asarray(baseline_logits))):
            raise ValueError("Frozen baseline contains nonfinite logits")
    carry = {team: initial_carry(policies[team].module, len(names)) for team, names in teams.items()}
    recurrent_red = is_recurrent(policies["red"].module)
    red_reset = jnp.ones(len(teams["red"]), dtype=jnp.bool_) if recurrent_red else None
    counts = np.zeros((n_blue, BLUE_ALLOW_TRAFFIC_END), dtype=np.int64)
    types = np.zeros((n_blue, 8), dtype=np.int64)
    choices, repeats, changed_repeats = (np.zeros(n_blue, dtype=np.int64) for _ in range(3))
    runs, maximum_runs = np.zeros(n_blue, dtype=np.int64), np.zeros(n_blue, dtype=np.int64)
    previous, previous_obs = np.full(n_blue, -1, dtype=np.int64), [None] * n_blue
    hosts_seen = [set() for _ in range(n_blue)]
    diagnostics_sum = np.zeros(3, dtype=np.float64)
    near_uniform = 0
    reward_sums = {key: 0.0 for key in ("reward_ria", "reward_lwf", "reward_asf", "action_cost")}
    compromised_host_ticks, total, red_total = 0, 0.0, 0.0
    records, fitting = [], []
    n_subnets, host_counts = _observation_layout(state.const) if collect else (None, None)
    for tick in range(horizon):
        masks = env.get_avail_actions(state)
        blue_obs = jnp.stack([observations[name] for name in teams["blue"]])
        blue_masks = jnp.stack([masks[name] for name in teams["blue"]])
        incoming_carry = carry["blue"]
        rng, blue_key = jax.random.split(rng)
        selected, carry["blue"], diag, probabilities = blue_action(
            policies["blue"].module,
            policies["blue"].weights,
            blue_obs,
            blue_masks,
            incoming_carry,
            blue_key,
            mode,
            baseline_logits,
        )
        red_obs = jnp.stack([observations[name] for name in teams["red"]])
        red_masks = jnp.stack([masks[name] for name in teams["red"]])
        rng, red_key = jax.random.split(rng)
        red_selected, carry["red"] = red_action(
            policies["red"].module,
            policies["red"].weights,
            red_obs,
            red_masks,
            carry["red"],
            red_reset,
            red_key,
        )
        actions = {
            **{name: selected[i].astype(jnp.int32) for i, name in enumerate(teams["blue"])},
            **{name: red_selected[i].astype(jnp.int32) for i, name in enumerate(teams["red"])},
        }
        if recurrent_red:
            red_reset = ~state.state.red_agent_active
        busy = np.asarray(state.state.blue_pending_ticks > 0)
        kinds, hosts = action_targets(selected, state.const)
        selected_np, kinds_np, hosts_np, diag_np = map(np.asarray, (selected, kinds, hosts, diag))
        free = (diag_np[:, 4] > 1) & ~busy
        obs_np = np.asarray(blue_obs)
        if collect:
            fitting.append(
                {
                    "fit_probabilities": np.asarray(probabilities),
                    "fit_masks": np.asarray(blue_masks, dtype=np.bool_),
                    "fit_genuine": free,
                }
            )
        if collect and tick % collection_stride == 0 and np.any(free):
            records.append(
                {
                    "observations": obs_np,
                    "masks": np.asarray(blue_masks, dtype=np.bool_),
                    "probabilities": np.asarray(probabilities),
                    "genuine": free,
                    "phase": np.int32(state.state.mission_phase),
                    "topology_index": np.int32(topology_index),
                    "tick": np.int32(tick),
                    "seed": np.int32(seed),
                    "n_subnets": n_subnets,
                    "host_counts": host_counts,
                    **_carry_record(incoming_carry),
                }
            )
        for agent in np.flatnonzero(free):
            action, kind, host = int(selected_np[agent]), int(kinds_np[agent]), int(hosts_np[agent])
            counts[agent, action] += 1
            types[agent, kind] += 1
            choices[agent] += 1
            repeated = previous[agent] == action
            repeats[agent] += int(repeated)
            if repeated and previous_obs[agent] is not None and np.any(obs_np[agent] != previous_obs[agent]):
                changed_repeats[agent] += 1
            runs[agent] = runs[agent] + 1 if repeated else 1
            maximum_runs[agent] = max(maximum_runs[agent], runs[agent])
            previous[agent], previous_obs[agent] = action, obs_np[agent].copy()
            if kind in (2, 3, 4, 5) and host >= 0:
                hosts_seen[agent].add(host)
        diagnostics_sum += diag_np[free, :3].sum(axis=0)
        near_uniform += int(np.sum(diag_np[free, 2] > 0.9))
        rng, step_key = jax.random.split(rng)
        transition_key, _ = jax.random.split(step_key)
        observations, state, rewards, dones, info = env.step_env(transition_key, state, actions)
        total += float(rewards[teams["blue"][0]])
        red_total += float(rewards[teams["red"][0]])
        for key in reward_sums:
            reward_sums[key] += float(info.get(key, 0))
        compromised_host_ticks += int(np.sum(np.asarray(state.state.host_compromised) > 0))
        if bool(dones["__all__"]):
            break
    arrays = {key: np.stack([record[key] for record in records]) for key in records[0]} if records else {}
    if fitting:
        arrays.update({key: np.stack([record[key] for record in fitting]) for key in fitting[0]})
    return {
        "seed": seed,
        "topology_index": topology_index,
        "blue_return": total,
        "red_return": red_total,
        "steps": tick + 1,
        "behavior": {
            "decision_counts": choices.tolist(),
            "action_counts": counts.tolist(),
            "type_counts": types.tolist(),
            "repeats": repeats.tolist(),
            "repeats_with_changed_observations": changed_repeats.tolist(),
            "longest_identical_action_run": maximum_runs.tolist(),
            "unique_host_targets": [len(value) for value in hosts_seen],
            "diagnostic_sums": diagnostics_sum.tolist(),
            "near_uniform_decisions": near_uniform,
            "reward_components": reward_sums,
            "compromised_host_ticks": compromised_host_ticks,
        },
    }, arrays


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(
    *,
    root,
    protocol,
    trial,
    revision,
    blue,
    red,
    topologies,
    seeds,
    mode,
    env,
    variant,
    source_root,
    baseline_path=None,
    collect=False,
    collection_stride=5,
    auxiliary_paths=None,
):
    provenance = collect_provenance(
        expected_revision=revision, entrypoint=source_root / "compare.py", source_roots=(source_root,)
    )
    inputs = {
        f"{team} {role}": Path(policy[role]["path"])
        for team, policy in (("blue", blue), ("red", red))
        for role in ("model", "sidecar")
    }
    expected = {
        f"{team} {role}": policy[role]["sha256"]
        for team, policy in (("blue", blue), ("red", red))
        for role in ("model", "sidecar")
    }
    inputs.update({f"topology {index}": Path(value["path"]) for index, value in enumerate(topologies)})
    expected.update({f"topology {index}": value["sha256"] for index, value in enumerate(topologies)})
    if baseline_path is not None:
        inputs["frozen baseline"] = Path(baseline_path)
    for role, path in (auxiliary_paths or {}).items():
        inputs[f"auxiliary {role}"] = Path(path)
    identities = {role: _digest(path) for role, path in inputs.items()}
    if any(identities[role] != identity for role, identity in expected.items()):
        raise ValueError("Selected evaluation input differs from its frozen manifest")
    if mode == "blind" and baseline_path is None:
        raise ValueError("Blind evaluation requires a retained frozen baseline")
    config = json_configuration(
        {
            "backend": "jax",
            "eval_env": "jax_joint",
            "variant": asdict(variant),
            "blue_selection": mode,
            "red_selection": "sampled",
            "horizon": variant.num_steps,
            "seeds": seeds,
            "episodes_per_seed": 1,
            "topology": topology_evidence(
                [Path(p["path"]) for p in topologies], [p["sha256"] for p in topologies], sampling="exhaustive"
            ),
            "baseline_sha256": identities.get("frozen baseline"),
            "collect_decision_inputs": collect,
            "decision_collection_stride": collection_stride,
            "fitting_collection": "Every tick; genuine-decision flags select calibration fitting rows",
            "auxiliary_sha256": {
                role: identity for role, identity in identities.items() if role.startswith("auxiliary ")
            },
            "procedure": "studies/observation-value/engine.py:episode",
            "execution": provenance.configuration,
            "execution_settings": {
                "jax_platforms": os.environ.get("JAX_PLATFORMS"),
                "xla_flags": os.environ.get("XLA_FLAGS"),
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
            },
            "seed_contract": "reset, Blue, Red and environment key splits match the maintained joint rollout",
            "diagnostics": "Genuine decisions only; forced busy waits and singleton masks excluded",
        }
    )
    capture = start_evaluation(
        outside_source(root / "evidence"),
        protocol=protocol,
        trial_id=trial,
        requested_episodes=len(topologies) * len(seeds),
        resolved_config=config,
        facts={**provenance.facts, "procedure": fact(config["procedure"])},
        inputs=inputs,
        source_files=provenance.source_files,
        track=destination_configured(),
    )
    try:
        provenance.verify_sources()
        for role, identity in identities.items():
            if _digest(capture.inputs[role]) != identity:
                raise ValueError("Retained evaluation input bytes changed")
        policies = {
            team: load_matchup_policy(capture.inputs[f"{team} model"], team=team, backend="jax")
            for team in ("blue", "red")
        }
        baseline_logits = (
            json.loads(capture.inputs["frozen baseline"].read_text())["logits"] if mode == "blind" else None
        )
        rows, decisions = [], []
        for index, topology in enumerate(topologies):
            if _digest(topology["path"]) != topology["sha256"]:
                raise ValueError("Cached environment topology changed")
            for seed in seeds:
                row, records = episode(
                    policies,
                    env,
                    seed,
                    index,
                    mode,
                    variant.num_steps,
                    baseline_logits=baseline_logits,
                    collect=collect,
                    collection_stride=collection_stride,
                )
                row.update(topology=topology["label"], topology_sha256=topology["sha256"])
                record_episode(
                    capture,
                    episode_index=len(rows),
                    seed=seed,
                    policy_seed=seed,
                    reward=row["blue_return"],
                    steps=row["steps"],
                )
                rows.append(row)
                if records:
                    decisions.append(records)
                print(f"{trial} {len(rows)}/{len(topologies) * len(seeds)} return={row['blue_return']:.0f}", flush=True)
        retained_json(capture, "episodes-with-behavior.json", rows, "joint returns and decision statistics")
        records_path = None
        if collect:
            if not decisions:
                raise ValueError("No decision inputs were collected")
            combined = {key: np.concatenate([value[key] for value in decisions]) for key in decisions[0]}
            with tempfile.TemporaryDirectory(prefix="decision-inputs-") as temporary:
                path = Path(temporary) / "decision-inputs.npz"
                np.savez_compressed(path, **combined)
                add_artifact(capture, path, role="actual incoming Blue decision inputs", name=path.name)
            records_path = str(capture.directory / "artifacts/decision-inputs.npz")
        provenance.verify_sources()
        finish_evaluation(capture, status="succeeded")
        return {
            "blue": blue["id"],
            "red": red["id"],
            "mode": mode,
            "trial": trial,
            "attempt_id": capture.attempt_id,
            "evaluation_id": capture.evaluation_id,
            "configuration": config,
            "episodes": rows,
            "records_path": records_path,
        }
    except BaseException as error:
        fail_evaluation(capture, error)
        raise
