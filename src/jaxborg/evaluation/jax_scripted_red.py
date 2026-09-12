"""Evaluate a trained Blue policy against scripted Red in the JAX simulator.

Unlike :mod:`jaxborg.evaluation.scripted_red`, this evaluator consumes JAX
topology snapshots.  It is the paired resilience evaluation path: every
episode receives the fixed AUTH/DB/WEB assignment carried by its evaluation
case before either policy selects an action.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jaxborg.evaluation.cia.fixed_topology import EvaluationCase, build_evaluation_cases
from jaxborg.evaluation.cia.jax_resilience import (
    mean_resilience_episode,
    resilience_episode_records,
    score_resilience_state,
    summarize_resilience_episodes,
)
from jaxborg.evaluation.cia.reporting import cia_mlflow_metrics, cia_summary_dict
from jaxborg.evaluation.jax_env_factory import make_jax_env
from jaxborg.evaluation.matchup_runner import (
    LoadedMatchupPolicy,
    _eval_batch_size,
    _padded_batch,
    _supports_batched_eval,
    _torch_actions,
    cyborg_blue_flat_to_jax_lookup,
    jax_mask_to_cyborg_blue,
    load_matchup_policy,
)
from jaxborg.policies import initial_carry, policy_step
from jaxborg.scenarios.cc4.game_variant import GameVariant
from jaxborg.scenarios.cc4.game_variants import variant_for_red

DEFAULT_SCRIPTED_REDS = ("fsm", "cia_c", "cia_i", "cia_a")
_SUPPORTED_SCRIPTED_REDS = frozenset(DEFAULT_SCRIPTED_REDS)
_EVAL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class JaxScriptedRedEpisode:
    """Reward and temporal CIA score for one fixed-topology episode."""

    reward: float
    cia: tuple[float, float, float]


def _normalise_eval_name(value: str | None) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str) or not _EVAL_NAME_PATTERN.fullmatch(value):
        raise ValueError("evaluation name may contain only letters, numbers, '.', '_' and '-'")
    return value


def _parse_seeds(value: str | int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, bool):
        raise ValueError("seeds must contain non-negative integers")
    if isinstance(value, int):
        seeds = {value}
    elif isinstance(value, str):
        seeds: set[int] = set()
        for raw_token in value.split(","):
            token = raw_token.strip()
            if not token:
                continue
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                try:
                    start, end = int(start_text), int(end_text)
                except ValueError as exc:
                    raise ValueError(f"invalid evaluation seed range: {token!r}") from exc
                if end < start:
                    raise ValueError(f"evaluation seed range must be ascending: {token!r}")
                seeds.update(range(start, end + 1))
            else:
                try:
                    seeds.add(int(token))
                except ValueError as exc:
                    raise ValueError(f"invalid evaluation seed: {token!r}") from exc
    elif isinstance(value, Sequence):
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in value):
            raise ValueError("seeds must contain non-negative integers")
        seeds = set(value)
    else:
        raise ValueError("seeds must be a seed, range string, or integer list")
    if not seeds or min(seeds) < 0:
        raise ValueError("seeds must contain at least one non-negative integer")
    return tuple(sorted(seeds))


def _normalise_reds(value: str | Sequence[str]) -> tuple[str, ...]:
    reds = (value,) if isinstance(value, str) else tuple(value)
    if not reds:
        raise ValueError("reds must contain at least one scripted Red")
    if any(not isinstance(red, str) for red in reds):
        raise ValueError("reds must be a string list")
    unknown = set(reds) - _SUPPORTED_SCRIPTED_REDS
    if unknown:
        raise ValueError(
            f"unsupported scripted Red agents {sorted(unknown)}; expected a subset of {list(DEFAULT_SCRIPTED_REDS)}"
        )
    if len(set(reds)) != len(reds):
        raise ValueError("reds must not contain duplicates")
    return reds


def _normalise_topology_paths(paths: Sequence[str | Path] | None) -> tuple[Path, ...]:
    if paths is None:
        raise ValueError("CIA evaluation requires a non-empty topology bank")
    if isinstance(paths, (str, bytes, Path)):
        paths = (paths,)
    resolved = tuple(Path(path).expanduser().resolve() for path in paths)
    if not resolved:
        raise ValueError("CIA evaluation requires a non-empty topology bank")
    missing = [path for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"topology snapshot not found: {missing[0]}")
    return resolved


def _backend_from_model(model_path: Path) -> str:
    if model_path.suffix == ".safetensors":
        return "jax"
    if model_path.suffix == ".pt":
        return "cyborg"
    raise ValueError(f"cannot detect policy backend from model suffix: {model_path}")


def _fixed_role_state(state: Any, case: EvaluationCase) -> Any:
    """Install the case role map while preserving the FSM extras schema."""

    extras = dict(state.extras)
    extras["host_resilience_role"] = case.role_array
    return state.replace(extras=extras)


@partial(
    jax.jit,
    static_argnames=("policy_module", "env", "num_steps", "deterministic"),
)
def _run_jax_scripted_red_episode_scan(
    policy_weights: Any,
    key: jax.Array,
    topology_index: jax.Array,
    host_resilience_role: jax.Array,
    *,
    policy_module: Any,
    env: Any,
    num_steps: int,
    deterministic: bool,
) -> tuple[jax.Array, jax.Array]:
    """Compile Blue inference, scripted Red, and CIA scoring for an episode."""

    key, reset_key = jax.random.split(key)
    obs, state = env.reset_at_topology(reset_key, topology_index)
    extras = dict(state.extras)
    extras["host_resilience_role"] = host_resilience_role
    state = state.replace(extras=extras)
    blue_agents = tuple(env.agents)
    zero_cia = jnp.zeros(3, dtype=jnp.float32)

    def _active_step(rng, current_obs, current_state, policy_carry):
        masks = env.get_avail_actions(current_state)
        obs_batch = jnp.stack([current_obs[agent] for agent in blue_agents])
        mask_batch = jnp.stack([masks[agent] for agent in blue_agents])
        rng, policy_key = jax.random.split(rng)
        # Blue never goes dormant and the scan stops at termination, so its
        # sequence runs unbroken from the reset for the whole episode.
        pi, _, policy_carry = policy_step(policy_module, policy_weights, obs_batch, mask_batch, carry=policy_carry)
        selected = jnp.argmax(pi.logits, axis=-1) if deterministic else pi.sample(seed=policy_key)
        actions = {agent: jnp.asarray(selected[index], dtype=jnp.int32) for index, agent in enumerate(blue_agents)}
        rng, step_key = jax.random.split(rng)
        # ``FsmRedCC4Env.step`` splits the caller key into transition/reset/
        # extras children. Retain its transition child while bypassing the
        # terminal auto-reset needed for final-state CIA scoring.
        transition_key = jax.random.split(step_key, 3)[0]
        next_obs, next_state, rewards, dones, _ = env.step_env(
            transition_key,
            current_state,
            actions,
        )
        reward = jnp.asarray(rewards[blue_agents[0]], dtype=jnp.float32)
        cia = score_resilience_state(
            next_state.state,
            next_state.extras["host_resilience_role"],
        )
        done = jnp.asarray(dones["__all__"], dtype=jnp.bool_)
        return rng, next_obs, next_state, reward, cia, done, policy_carry

    def _scan_step(carry, _):
        rng, current_obs, current_state, active, reward_sum, cia_sum, valid_steps, policy_carry = carry

        def run_active(_):
            next_rng, next_obs, next_state, reward, cia, done, next_policy_carry = _active_step(
                rng,
                current_obs,
                current_state,
                policy_carry,
            )
            return (
                next_rng,
                next_obs,
                next_state,
                ~done,
                reward_sum + reward,
                cia_sum + cia,
                valid_steps + jnp.int32(1),
                next_policy_carry,
            )

        def keep_terminal(_):
            return (
                rng,
                current_obs,
                current_state,
                active,
                reward_sum,
                cia_sum,
                valid_steps,
                policy_carry,
            )

        return jax.lax.cond(active, run_active, keep_terminal, operand=None), None

    scan_carry = (
        key,
        obs,
        state,
        jnp.bool_(True),
        jnp.float32(0.0),
        zero_cia,
        jnp.int32(0),
        initial_carry(policy_module, len(blue_agents)),
    )
    final_carry, _ = jax.lax.scan(_scan_step, scan_carry, xs=None, length=num_steps)
    reward_sum = final_carry[4]
    cia_sum = final_carry[5]
    valid_steps = final_carry[6]
    cia_mean = jnp.where(valid_steps > 0, cia_sum / jnp.maximum(valid_steps, 1), zero_cia)
    return reward_sum, cia_mean


def _run_jax_scripted_red_episodes_batched(
    policy: LoadedMatchupPolicy,
    *,
    env: Any,
    variant: GameVariant,
    cases: Sequence[EvaluationCase],
    deterministic: bool,
    batch_size: int | None = None,
    progress: bool = False,
    progress_label: str = "",
) -> list[JaxScriptedRedEpisode]:
    """Evaluate many fixed-topology episodes per compiled call via ``jax.vmap``.

    One episode is a 500-step scan over a single environment, which leaves an
    accelerator almost entirely idle -- the work per kernel is tiny and the
    cost is dispatch latency. Mapping the identical per-episode scan over a
    batch gives the device the same shape of work training already runs at.
    Results are unchanged: each case keeps its own key, topology index and
    fixed role map, so this is the sequential sweep evaluated in parallel.
    """
    count = len(cases)
    if count == 0:
        return []
    chunk = batch_size or _eval_batch_size()

    scan = partial(
        _run_jax_scripted_red_episode_scan,
        policy_module=policy.module,
        env=env,
        num_steps=variant.num_steps,
        deterministic=deterministic,
    )
    # Weights are shared across the batch; keys, topologies and roles vary.
    batched = jax.vmap(scan, in_axes=(None, 0, 0, 0))

    keys = jnp.stack([jax.random.PRNGKey(case.episode_seed) for case in cases])
    indices = jnp.asarray([case.topology_index for case in cases], dtype=jnp.int32)
    roles = jnp.stack([case.role_array for case in cases])

    episodes: list[JaxScriptedRedEpisode] = []
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        reward_batch, cia_batch = batched(
            policy.weights,
            _padded_batch(keys, start, stop, chunk),
            _padded_batch(indices, start, stop, chunk),
            _padded_batch(roles, start, stop, chunk),
        )
        reward_batch, cia_batch = jax.device_get((reward_batch, cia_batch))
        rewards = np.asarray(reward_batch)[: stop - start]
        cia_rows = np.asarray(cia_batch)[: stop - start]
        episodes.extend(
            JaxScriptedRedEpisode(reward=float(reward), cia=(float(c), float(i), float(a)))
            for reward, (c, i, a) in zip(rewards, cia_rows, strict=True)
        )
        if progress:
            print(f"  {progress_label}episodes {stop}/{count}", flush=True)
    return episodes


def _torch_blue_actions(
    policy: LoadedMatchupPolicy,
    obs_batch: jax.Array,
    mask_batch: jax.Array,
    lookups: Sequence[np.ndarray],
    *,
    seed: int,
    deterministic: bool,
) -> np.ndarray:
    policy_masks = np.stack(
        [jax_mask_to_cyborg_blue(mask_batch[index], lookup) for index, lookup in enumerate(lookups)]
    )
    cyborg_actions = _torch_actions(policy, obs_batch, policy_masks, seed, deterministic)
    actions = np.asarray(
        [lookups[index][int(action)] for index, action in enumerate(cyborg_actions)],
        dtype=np.int32,
    )
    if np.any(actions < 0):  # pragma: no cover - masked defensive guard
        raise RuntimeError("Torch Blue policy selected a padded CybORG action")
    return actions


def run_jax_scripted_red_episode(
    policy: LoadedMatchupPolicy,
    *,
    env: Any,
    variant: GameVariant,
    case: EvaluationCase,
    deterministic: bool = False,
) -> JaxScriptedRedEpisode:
    """Run one exact case, injecting roles before the first policy action."""

    if policy.backend == "jax":
        reward, episode_cia = _run_jax_scripted_red_episode_scan(
            policy.weights,
            jax.random.PRNGKey(case.episode_seed),
            jnp.asarray(case.topology_index, dtype=jnp.int32),
            case.role_array,
            policy_module=policy.module,
            env=env,
            num_steps=variant.num_steps,
            deterministic=deterministic,
        )
        c, i, a = (float(value) for value in np.asarray(jax.device_get(episode_cia)))
        return JaxScriptedRedEpisode(
            reward=float(jax.device_get(reward)),
            cia=(c, i, a),
        )
    if policy.backend != "cyborg":
        raise ValueError(f"unsupported Blue policy backend: {policy.backend!r}")

    rng = jax.random.PRNGKey(case.episode_seed)
    rng, reset_key = jax.random.split(rng)
    obs, state = env.reset_at_topology(reset_key, case.topology_index)
    state = _fixed_role_state(state, case)
    blue_agents = tuple(env.agents)
    torch_lookups = (
        tuple(cyborg_blue_flat_to_jax_lookup(state.const, agent_id) for agent_id in range(len(blue_agents)))
        if policy.backend == "cyborg"
        else ()
    )
    total_reward = 0.0
    step_cia: list[jax.Array] = []
    for step_index in range(variant.num_steps):
        masks = env.get_avail_actions(state)
        obs_batch = jnp.stack([obs[agent] for agent in blue_agents])
        mask_batch = jnp.stack([masks[agent] for agent in blue_agents])
        torch_seed = case.episode_seed * 1_000_003 + step_index * 17
        selected = _torch_blue_actions(
            policy,
            obs_batch,
            mask_batch,
            torch_lookups,
            seed=torch_seed,
            deterministic=deterministic,
        )
        actions = {agent: jnp.asarray(selected[index], dtype=jnp.int32) for index, agent in enumerate(blue_agents)}

        # `step` auto-resets terminal states. CIA must observe the final state,
        # so use the non-resetting transition API directly while preserving
        # ``FsmRedCC4Env.step``'s transition-key convention.
        rng, step_key = jax.random.split(rng)
        transition_key = jax.random.split(step_key, 3)[0]
        obs, state, rewards, dones, _ = env.step_env(
            transition_key,
            state,
            actions,
        )
        total_reward += float(rewards[blue_agents[0]])
        step_cia.append(score_resilience_state(state.state, state.extras["host_resilience_role"]))
        if bool(dones["__all__"]):
            break

    episode_cia = mean_resilience_episode(jnp.stack(step_cia))
    c, i, a = (float(value) for value in np.asarray(jax.device_get(episode_cia)))
    return JaxScriptedRedEpisode(reward=total_reward, cia=(c, i, a))


def _scripted_variant(base_variant: GameVariant, red: str) -> GameVariant:
    variant = variant_for_red(red, resilience_roles=True)
    return replace(
        variant,
        name=f"{base_variant.name}_fsm" if red == "fsm" else variant.name,
        num_steps=base_variant.num_steps,
        op_zone_servers=base_variant.op_zone_servers,
        resilience_roles=True,
    )


def _topology_role_maps(cases: Sequence[EvaluationCase]) -> list[dict[str, Any]]:
    by_topology: dict[int, EvaluationCase] = {}
    for case in cases:
        by_topology.setdefault(case.topology_index, case)
    return [
        {
            "topology_index": case.topology_index,
            "topology_path": str(case.topology_path),
            **case.audit_role_map(),
        }
        for case in by_topology.values()
    ]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""


def evaluate_jax_scripted_reds(
    model_path: str | Path,
    *,
    base_variant: GameVariant,
    topology_paths: Sequence[str | Path] | None,
    reds: Sequence[str] = DEFAULT_SCRIPTED_REDS,
    seeds: str | int | Sequence[int] = "1000-1009",
    episodes_per_seed: int = 1,
    deterministic: bool = False,
    progress: bool = False,
    eval_name: str | None = None,
    recipe: Mapping[str, Any] | None = None,
    policy_loader: Callable[..., LoadedMatchupPolicy] | None = None,
    env_factory: Callable[..., Any] | None = None,
    episode_runner: Callable[..., JaxScriptedRedEpisode] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate trained Blue against scripted Reds on an exhaustive bank."""

    if not base_variant.resilience_roles:
        raise ValueError("JAX scripted-Red CIA evaluation requires a variant with resilience_roles=True")
    if isinstance(episodes_per_seed, bool) or not isinstance(episodes_per_seed, int) or episodes_per_seed < 1:
        raise ValueError("episodes_per_seed must be a positive integer")
    normalised_reds = _normalise_reds(reds)
    parsed_seeds = _parse_seeds(seeds)
    eval_name = _normalise_eval_name(eval_name)
    resolved_topologies = _normalise_topology_paths(topology_paths)
    source_recipe = dict(recipe or {})
    from jaxborg.evaluation.cia.config import CIAEvalSettings, validate_cia_evaluation

    cia_settings = CIAEvalSettings.from_recipe(source_recipe)
    if not cia_settings.enabled:
        raise ValueError("JAX scripted-Red CIA evaluation requires eval.cia.enabled: true")
    topology_sampling = (source_recipe.get("eval") or {}).get("topology_sampling", "exhaustive")
    validate_cia_evaluation(
        cia_settings,
        variant=base_variant,
        topology_sampling=topology_sampling,
        topology_paths=resolved_topologies,
    )
    cases = build_evaluation_cases(resolved_topologies, parsed_seeds, episodes_per_seed)

    resolved_model = Path(model_path).expanduser().resolve()
    if not resolved_model.is_file():
        raise FileNotFoundError(f"model not found: {resolved_model}")
    backend = _backend_from_model(resolved_model)
    load_policy = policy_loader or load_matchup_policy
    policy = load_policy(resolved_model, team="blue", backend=backend)
    if policy.source.get("bundle_trainable") is False:
        raise ValueError(f"model bundle marks Blue as frozen, not trained: {resolved_model}")
    make_env = env_factory or make_jax_env
    run_episode = episode_runner or run_jax_scripted_red_episode
    run = source_recipe.get("run", {})
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    nonce = f"{time.time_ns() % 1_000_000_000:09d}"
    role_maps = _topology_role_maps(cases)
    rows: list[dict[str, Any]] = []

    for red in normalised_reds:
        variant = _scripted_variant(base_variant, red)
        env = make_env(
            variant,
            training_mode=False,
            topology_path=resolved_topologies,
        )
        started = time.perf_counter()
        episodes: list[JaxScriptedRedEpisode] = []
        # A JAX Blue policy vmaps the whole per-episode scan; Torch Blue and an
        # injected ``episode_runner`` keep the one-case-at-a-time seam.
        if episode_runner is None and policy.backend == "jax" and _supports_batched_eval(env):
            episodes = _run_jax_scripted_red_episodes_batched(
                policy,
                env=env,
                variant=variant,
                cases=cases,
                deterministic=deterministic,
                progress=progress,
                progress_label=f"{red} ",
            )
        else:
            for index, case in enumerate(cases, 1):
                result = run_episode(
                    policy,
                    env=env,
                    variant=variant,
                    case=case,
                    deterministic=deterministic,
                )
                episodes.append(result)
                if progress:
                    print(
                        f"  {red} ep {index}/{len(cases)} "
                        f"(topology={case.topology_path.name}, seed={case.episode_seed}): {result.reward:.1f}",
                        flush=True,
                    )

        rewards = [episode.reward for episode in episodes]
        episode_cia = np.asarray([episode.cia for episode in episodes], dtype=np.float64)
        cia_summary = cia_summary_dict(summarize_resilience_episodes(episode_cia))
        cia_records = resilience_episode_records(episode_cia)
        row = {
            "eval_id": f"{timestamp}_{nonce}_{red}",
            "eval_name": eval_name,
            "suite": "scripted_red",
            "model": str(resolved_model),
            "policy_team": "blue",
            "recipe_name": source_recipe.get("meta", {}).get("name", ""),
            "recipe_path": source_recipe.get("__source_path__", source_recipe.get("meta", {}).get("source_path", "")),
            "trained_backend": backend,
            "policy_backend": backend,
            "eval_env": "jax_fsm",
            "eval_red": red,
            "base_variant": base_variant.name,
            "variant": variant.name,
            "red_agent": variant.red_agent,
            "resilience_roles": True,
            "seeds": list(parsed_seeds),
            "episodes_per_seed": episodes_per_seed,
            "episodes_per_topology": len(parsed_seeds) * episodes_per_seed,
            "stochastic": not deterministic,
            "mean_reward": mean(rewards),
            "std_reward": stdev(rewards) if len(rewards) > 1 else 0.0,
            "n_episodes": len(rewards),
            "wall_time_s": time.perf_counter() - started,
            "git_commit": _git_commit(),
            "train_run_id": run.get("train_run_id"),
            "train_total_steps": run.get("total_steps"),
            "blue_policy": policy.source,
            "topology_paths": [str(path) for path in resolved_topologies],
            "topology_sampling": "exhaustive",
            "per_episode_topology_paths": [str(case.topology_path) for case in cases],
            "per_episode_seeds": [case.episode_seed for case in cases],
            "per_episode": rewards,
            "cia_metric": cia_settings.metric,
            "cia_config": cia_settings.as_dict(),
            "cia_summary": cia_summary,
            "per_episode_cia": cia_records,
            "episode_role_map_ids": [case.role_map_id for case in cases],
            "per_episode_topology_fingerprints": [case.topology_fingerprint for case in cases],
            "topology_role_maps": role_maps,
        }
        rows.append(row)
        print(
            f"Blue vs {red}: reward {row['mean_reward']:.2f} +/- {row['std_reward']:.2f}; "
            f"CIA C={cia_summary['c']['mean']:.2f}, I={cia_summary['i']['mean']:.2f}, "
            f"A={cia_summary['a']['mean']:.2f}",
            flush=True,
        )
    return rows


def _default_output_path(rows: Sequence[Mapping[str, Any]]) -> Path:
    exp_dir = Path(os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp")).expanduser().resolve()
    first = rows[0]
    eval_prefix = str(first["eval_id"]).rsplit("_", 1)[0]
    name = f"_{first['eval_name']}" if first.get("eval_name") else ""
    return (
        exp_dir
        / "eval"
        / (f"{first['recipe_name']}_{Path(str(first['model'])).stem}_jax_scripted_red{name}_{eval_prefix}.jsonl")
    )


def write_results(rows: Sequence[Mapping[str, Any]], output_path: str | Path | None = None) -> Path:
    if not rows:
        raise ValueError("cannot write an empty JAX scripted-Red evaluation")
    path = Path(output_path).expanduser().resolve() if output_path else _default_output_path(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, default=str) + "\n" for row in rows))
    return path


def attach_results_to_mlflow(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    run_id = rows[0].get("train_run_id")
    if not run_id:
        return
    metrics: dict[str, float] = {}
    for row in rows:
        eval_name = row.get("eval_name")
        prefix = (
            f"eval.after_training.{eval_name}.scripted_red.{row['eval_red']}.blue"
            if eval_name
            else f"eval.scripted_red.{row['eval_red']}.blue"
        )
        metrics[f"{prefix}.mean_reward"] = float(row["mean_reward"])
        metrics[f"{prefix}.std_reward"] = float(row["std_reward"])
        metrics[f"{prefix}.episodes"] = float(row["n_episodes"])
        metrics.update(cia_mlflow_metrics(f"{prefix}.cia", row["cia_summary"]))
    from jaxborg.mlflow_setup import attach_eval_metrics

    attach_eval_metrics(str(run_id), metrics)


def _cia_recipe_configuration(recipe: Mapping[str, Any]) -> tuple[str, str]:
    from jaxborg.evaluation.cia.config import CIAEvalSettings

    settings = CIAEvalSettings.from_recipe(recipe)
    if not settings.enabled:
        raise ValueError("JAX scripted-Red CIA evaluation requires eval.cia.enabled: true")
    return settings.metric, settings.role_assignment


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained Blue vs scripted Red on JAX topology snapshots")
    parser.add_argument("--model", required=True, help="Final .safetensors or .pt policy bundle")
    parser.add_argument(
        "--recipe",
        default=os.environ.get("JAXBORG_RECIPE_PATH"),
        help="Recipe name/path (default: post-training environment or model sidecar)",
    )
    parser.add_argument("--reds", nargs="+", choices=DEFAULT_SCRIPTED_REDS, default=list(DEFAULT_SCRIPTED_REDS))
    parser.add_argument("--seeds", default="1000-1009")
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--topology-path",
        action="append",
        default=None,
        help="Override topology snapshot; repeat for a held-out bank",
    )
    parser.add_argument("--topology-sampling", choices=("exhaustive", "random"), default=None)
    parser.add_argument("--name", default=os.environ.get("JAXBORG_EVAL_NAME"))
    parser.add_argument("--output")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    if args.recipe:
        from jaxborg.recipe import load

        recipe = load(args.recipe)
    else:
        from jaxborg.checkpoint import read_sidecar

        recipe = read_sidecar(args.model)

    _cia_recipe_configuration(recipe)
    from jaxborg.recipe import REPO_ROOT, eval_variant, project_eval
    from jaxborg.topology_banks import validate_eval_topology_override

    variant = eval_variant(dict(recipe))
    if not variant.resilience_roles:
        raise ValueError("eval variant must have resilience_roles=True for CIA evaluation")
    sampling = args.topology_sampling or recipe.get("eval", {}).get("topology_sampling", "exhaustive")
    if sampling != "exhaustive":
        raise ValueError("CIA evaluation requires eval.topology_sampling: exhaustive")
    if args.topology_path:
        topology_paths = tuple(Path(path).expanduser().resolve() for path in args.topology_path)
        validate_eval_topology_override(recipe, topology_paths, repo_root=REPO_ROOT)
    else:
        topology_paths = tuple(project_eval(dict(recipe), materialize_topologies=True)["TOPOLOGY_BANK"])

    rows = evaluate_jax_scripted_reds(
        args.model,
        base_variant=variant,
        topology_paths=topology_paths,
        reds=args.reds,
        seeds=args.seeds,
        episodes_per_seed=args.episodes_per_seed,
        deterministic=args.deterministic,
        progress=args.progress,
        eval_name=args.name,
        recipe=recipe,
    )
    output = write_results(rows, args.output)
    print(f"Wrote JAX scripted-Red sweep: {output}", flush=True)
    if not args.no_mlflow:
        try:
            attach_results_to_mlflow(rows)
            if rows[0].get("train_run_id"):
                print(f"Attached scripted-Red metrics to MLflow run {rows[0]['train_run_id']}", flush=True)
        except Exception as exc:
            print(f"MLflow attach warning: {exc}", flush=True)


__all__ = [
    "DEFAULT_SCRIPTED_REDS",
    "JaxScriptedRedEpisode",
    "attach_results_to_mlflow",
    "evaluate_jax_scripted_reds",
    "main",
    "run_jax_scripted_red_episode",
    "write_results",
]


if __name__ == "__main__":
    main()
