"""CEC Phase 1 (axis B router-topology variation) eval harness.

Evaluates a JAX-trained checkpoint on three testbeds that share the same
runtime — only the seed range and per-call env config change:

  1. ``train``     — same env config as training (gen-fixed / gen-base /
                     gen-router) over seeds ``[0, N)``.
  2. ``heldout``   — gen-router env (varied router topologies) over
                     seeds ``[10000, 10000+N)``. Forces evaluation under
                     unseen structural-axis configurations even when the
                     training arm did NOT vary topology.
  3. ``heldout_fsm`` — same env config as training, but reset keys drawn
                     from a disjoint seed range ``[20000, 20000+N)``. This
                     isolates "novel red FSM rollouts" from "novel topology".

For each (checkpoint, testbed) we compute per-episode reward + CIA via
:func:`jaxborg.eval.cia.score_jax_episode` and dump a JSON report.

Usage::

    python scripts/eval/cec_phase1_eval.py \
        --checkpoint /path/to/checkpoint_final.pkl \
        --arm gen-router \
        --episodes 30 \
        --output /path/to/eval_<arm>_<seed>.json

The output JSON has the schema consumed by ``cec_phase1_aggregate.py``.
"""

# ruff: noqa: E402

import argparse
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

# transfer.py is loaded for `make_scan_eval_fn` and `load_checkpoint`.
from transfer import load_checkpoint, make_scan_eval_fn  # noqa: E402

from jaxborg.eval.cia import score_jax_episode  # noqa: E402
from jaxborg.fsm_red_env import FsmRedCC4Env  # noqa: E402

ARM_CONFIGS = {
    # ── Phase 1 arms (axis B/C) ──────────────────────────────────────────
    # gen-fixed: collapse every reset to a single topology (default tree).
    # We use TOPOLOGY_FIXED_KEY=0 to lock the topology; vary_router_links is
    # irrelevant since the key is fixed.
    "gen-fixed": dict(vary_router_links=False, vary_phase_rewards=False, topology_fixed_key=0),
    # gen-base: per-key host counts/services/red entry, default _ROUTER_LINKS
    # tree (no extra edges). The Phase 0 generative-mode default.
    "gen-base": dict(vary_router_links=False, vary_phase_rewards=False, topology_fixed_key=None),
    # gen-router: gen-base + per-key router topology from validated bank.
    "gen-router": dict(vary_router_links=True, vary_phase_rewards=False, topology_fixed_key=None),
    # gen-router-rewards: combines axis B (router topology) and axis C (phase
    # rewards) variation. Most diverse training distribution.
    "gen-router-rewards": dict(vary_router_links=True, vary_phase_rewards=True, topology_fixed_key=None),
    # ── Phase 2 arms (mission-objective family × comms) ──────────────────
    # gen-fixed-nomsg: re-trained baseline on the new architecture.  No env
    # variation, no comms.
    "gen-fixed-nomsg": dict(
        vary_router_links=False,
        vary_phase_rewards=False,
        vary_mission_profile=False,
        topology_fixed_key=0,
    ),
    "gen-fixed-msg": dict(
        vary_router_links=False,
        vary_phase_rewards=False,
        vary_mission_profile=False,
        topology_fixed_key=0,
    ),
    "gen-mission-nomsg": dict(
        vary_router_links=False,
        vary_phase_rewards=False,
        vary_mission_profile=True,
        topology_fixed_key=None,
    ),
    "gen-mission-msg": dict(
        vary_router_links=False,
        vary_phase_rewards=False,
        vary_mission_profile=True,
        topology_fixed_key=None,
    ),
    # ── Phase 3 arms (10x amplify-only mission bank × goal visibility) ───
    # Both arms use the new 10x bank and per-key mission profile sampling.
    # The only difference is whether the multiplier triple is appended to
    # blue obs (obs_mission_goal).  No comms head — Phase 2 found it
    # collapses at 20M.
    "gen-mission-10x-hidden": dict(
        vary_router_links=False,
        vary_phase_rewards=False,
        vary_mission_profile=True,
        obs_mission_goal=False,
        topology_fixed_key=None,
    ),
    "gen-mission-10x-visible": dict(
        vary_router_links=False,
        vary_phase_rewards=False,
        vary_mission_profile=True,
        obs_mission_goal=True,
        topology_fixed_key=None,
    ),
}

TESTBEDS = {
    # Each entry maps to: (per-episode env config, seed_offset). Seeds shift by
    # offset to keep distributions disjoint.
    "train": dict(arm_override=None, seed_offset=0),
    "heldout": dict(arm_override="gen-router", seed_offset=10000),
    "heldout_fsm": dict(arm_override=None, seed_offset=20000),
    # Phase 2: per-profile CIA breakdown.  Forces vary_mission_profile=True so
    # we sample across all profiles within the testbed; downstream aggregation
    # bins per-episode results by const.mission_profile_index.
    "per_mission": dict(
        arm_override=None,
        seed_offset=50000,
        force_kwargs=dict(vary_mission_profile=True),
    ),
    # Phase 2: strict OOD on axis D (allowed_subnet_pairs).  Seen by *all* arms
    # as the same novel config (vary_router_links + vary_phase_rewards +
    # vary_mission_profile + vary_subnet_pairs).
    "heldout_unseen": dict(
        arm_override=None,
        seed_offset=30000,
        force_kwargs=dict(
            vary_router_links=True,
            vary_phase_rewards=True,
            vary_mission_profile=True,
            vary_subnet_pairs=True,
            topology_fixed_key=None,
        ),
    ),
    # Phase 2: thinnest-distribution (compositional-training penalty) — all
    # axes off, fixed topology key 0.
    "heldout_thinner": dict(
        arm_override=None,
        seed_offset=40000,
        force_kwargs=dict(
            vary_router_links=False,
            vary_phase_rewards=False,
            vary_mission_profile=False,
            vary_subnet_pairs=False,
            topology_fixed_key=0,
        ),
    ),
}

DEFAULT_NUM_STEPS = 500


def _run_rollout(env, policy, params, policy_kind, num_episodes, seed_offset, deterministic):
    """Run ``num_episodes`` rollouts and return per-episode arrays.

    Returns dict with keys: rewards (E,), actions (E, T, B), host_compromised
    (E, T, H), red_impact (E, T, H), const_per_episode (list of CC4Const).
    """
    keys = jnp.stack([jax.random.PRNGKey(seed_offset + ep) for ep in range(num_episodes)])
    all_obs, all_env_states = jax.vmap(env.reset)(keys)
    scan_fn = make_scan_eval_fn(env, policy, policy_kind, deterministic)

    t0 = time.perf_counter()
    _, all_step_data = jax.vmap(scan_fn, in_axes=(None, 0, 0, 0))(params, keys, all_env_states, all_obs)
    elapsed = time.perf_counter() - t0
    print(f"    {num_episodes} episodes in {elapsed:.1f}s ({elapsed / num_episodes:.2f}s/ep)", flush=True)

    return {
        "actions": np.asarray(all_step_data["actions"]),
        "rewards": np.asarray(all_step_data["reward_mean"]),
        "host_compromised": np.asarray(all_step_data["host_compromised"]),
        "red_impact": np.asarray(all_step_data["red_impact_attempted"]),
        "ria_per_step": np.asarray(all_step_data["reward_ria"]),
        "lwf_per_step": np.asarray(all_step_data["reward_lwf"]),
        "asf_per_step": np.asarray(all_step_data["reward_asf"]),
        "messages": np.asarray(all_step_data["messages"]),  # (E, T, NUM_BLUE_AGENTS, MESSAGE_LENGTH)
        "env_states": all_env_states,  # we'll pull per-episode const out of this
    }


def _score_all_episodes(rollout, num_episodes):
    """Apply score_jax_episode to each episode in a rollout."""
    scores = []
    actions = rollout["actions"]
    hc = rollout["host_compromised"]
    ri = rollout["red_impact"]
    rewards = rollout["rewards"]
    consts = rollout["env_states"].const

    # consts is a vmap-stacked CC4Const: each leaf has leading dim = num_episodes.
    host_active_all = np.asarray(consts.host_active)
    host_subnet_all = np.asarray(consts.host_subnet)
    host_is_router_all = np.asarray(consts.host_is_router)
    host_is_server_all = np.asarray(consts.host_is_server)
    host_is_user_all = np.asarray(consts.host_is_user)

    for ep in range(num_episodes):
        ep_total_reward = float(rewards[ep].sum())
        s = score_jax_episode(
            actions=actions[ep],
            host_compromised=hc[ep],
            red_impact=ri[ep],
            host_active=host_active_all[ep],
            host_subnet=host_subnet_all[ep],
            host_is_router=host_is_router_all[ep],
            host_is_server=host_is_server_all[ep],
            host_is_user=host_is_user_all[ep],
            total_reward=ep_total_reward,
        )
        scores.append(s)
    return scores


def _norm_scorecard(actions: np.ndarray, host_compromised: np.ndarray) -> dict:
    """Per-episode hand-coded norm-compliance fractions.

    actions: (T, NUM_BLUE_AGENTS) int — action indices in the encoded action
        space.
    host_compromised: (T, GLOBAL_MAX_HOSTS) int — per-step compromise level.

    Returns a dict of percent-compliance scalars in ``[0, 1]``.  Rules are
    intentionally lenient — they look only at action-class statistics so the
    score is independent of which checkpoint produced the trajectory.
    """
    from jaxborg.actions.encoding import (
        BLUE_ANALYSE_END,
        BLUE_ANALYSE_START,
        BLUE_DECOY_END,
        BLUE_DECOY_START,
        BLUE_RESTORE_END,
        BLUE_RESTORE_START,
        BLUE_SLEEP,
    )

    T, B = actions.shape
    flat = actions.reshape(-1)
    sleep_count = int(np.sum(flat == BLUE_SLEEP))
    analyse_count = int(np.sum((flat >= BLUE_ANALYSE_START) & (flat < BLUE_ANALYSE_END)))
    restore_count = int(np.sum((flat >= BLUE_RESTORE_START) & (flat < BLUE_RESTORE_END)))
    decoy_count = int(np.sum((flat >= BLUE_DECOY_START) & (flat < BLUE_DECOY_END)))
    total = T * B

    # Restraint: sleep when no compromise observed.
    no_compromise_steps = np.sum(host_compromised > 0, axis=1) == 0
    sleep_when_quiet = 0.0
    if no_compromise_steps.any():
        sleep_during_quiet = np.sum(actions[no_compromise_steps] == BLUE_SLEEP)
        denom = float(no_compromise_steps.sum() * B)
        if denom > 0:
            sleep_when_quiet = sleep_during_quiet / denom

    return {
        "sleep_fraction": sleep_count / total if total else 0.0,
        "analyse_fraction": analyse_count / total if total else 0.0,
        "restore_fraction": restore_count / total if total else 0.0,
        "decoy_fraction": decoy_count / total if total else 0.0,
        "sleep_when_quiet": float(sleep_when_quiet),
    }


def _message_stats(messages: np.ndarray) -> dict:
    """Per-byte protocol diagnostics across a single episode.

    messages: (T, NUM_BLUE_AGENTS, MESSAGE_LENGTH) float in [-1, 1] (tanh).

    Returns coarse summaries that distinguish "messages collapsed" (all zeros
    or constants → low std + low entropy) from "messages active but null DV"
    (high std but low MI with state events — Phase 3 question).  We bin each
    byte into 8 quantile buckets per agent and compute Shannon entropy.
    """
    if messages.size == 0:
        return {}
    T, B, M = messages.shape
    # Per-byte mean/std over time, averaged across agents.
    per_byte_std = messages.std(axis=0).mean(axis=0)  # (M,)
    per_byte_mean_abs = np.abs(messages).mean(axis=(0, 1))  # (M,)

    # 8-bucket quantile entropy per (agent, byte) → averaged.
    edges = np.quantile(messages.reshape(-1, M), np.linspace(0, 1, 9), axis=0)
    # Avoid duplicate edges when a byte is constant.
    edges = np.where(np.diff(edges, axis=0) == 0, edges[:-1] + 1e-6, edges[1:])
    entropies = []
    for b in range(B):
        for m in range(M):
            buckets = np.searchsorted(edges[:, m], messages[:, b, m], side="right")
            counts = np.bincount(buckets, minlength=9)[:8].astype(np.float32)
            counts = counts / max(counts.sum(), 1.0)
            counts = counts[counts > 0]
            entropies.append(float(-(counts * np.log(counts)).sum()))
    return {
        "msg_per_byte_std_mean": float(np.mean(per_byte_std)),
        "msg_per_byte_std_max": float(np.max(per_byte_std)),
        "msg_per_byte_abs_mean": float(np.mean(per_byte_mean_abs)),
        "msg_quantile_entropy_mean": float(np.mean(entropies)),
        "msg_quantile_entropy_max": float(np.max(entropies)) if entropies else 0.0,
    }


def _summarize(scores, *, mission_indices=None, norm_per_episode=None, msg_per_episode=None):
    if not scores:
        return {}
    rewards = [s.total_reward for s in scores]
    cs = [s.C_mean for s in scores]
    is_ = [s.I_mean for s in scores]
    as_ = [s.A_mean for s in scores]
    rs = [s.R_mean for s in scores]

    def stats(xs):
        n = len(xs)
        m = sum(xs) / n
        if n > 1:
            sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
            stderr = sd / (n**0.5)
        else:
            sd = stderr = 0.0
        return dict(mean=m, sd=sd, stderr=stderr, n=n)

    per_episode = []
    for idx, s in enumerate(scores):
        rec = dict(reward=s.total_reward, C=s.C_mean, I=s.I_mean, A=s.A_mean, R=s.R_mean)
        if mission_indices is not None:
            rec["mission_profile_index"] = int(mission_indices[idx])
        if norm_per_episode is not None:
            rec["norms"] = norm_per_episode[idx]
        per_episode.append(rec)

    summary = {
        "reward": stats(rewards),
        "C_mean": stats(cs),
        "I_mean": stats(is_),
        "A_mean": stats(as_),
        "R_mean": stats(rs),
        "per_episode": per_episode,
    }

    if mission_indices is not None:
        # Bucket reward + R_mean by mission_profile_index for the per_mission DV.
        per_profile: dict = {}
        for s, mp in zip(scores, mission_indices):
            key = int(mp)
            per_profile.setdefault(key, {"reward": [], "R_mean": [], "C_mean": [], "A_mean": []})
            per_profile[key]["reward"].append(s.total_reward)
            per_profile[key]["R_mean"].append(s.R_mean)
            per_profile[key]["C_mean"].append(s.C_mean)
            per_profile[key]["A_mean"].append(s.A_mean)
        summary["per_mission_profile"] = {k: {m: stats(v) for m, v in d.items()} for k, d in per_profile.items()}

    if norm_per_episode is not None:
        keys = sorted({k for ep in norm_per_episode for k in ep.keys()})
        summary["norms"] = {k: stats([ep.get(k, 0.0) for ep in norm_per_episode]) for k in keys}

    if msg_per_episode is not None and msg_per_episode:
        keys = sorted({k for ep in msg_per_episode for k in ep.keys()})
        summary["messages"] = {k: stats([ep.get(k, 0.0) for ep in msg_per_episode]) for k in keys}

    return summary


def main():
    ap = argparse.ArgumentParser(description="CEC Phase 1 axis-B eval (train + heldout testbeds)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--arm",
        required=True,
        choices=list(ARM_CONFIGS.keys()),
        help="Training arm (governs train-testbed env config and heldout_fsm config)",
    )
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--output", required=True, help="Path to write summary JSON")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument(
        "--testbeds",
        default="train,heldout,heldout_fsm",
        help="Comma-separated subset of testbeds to run",
    )
    args = ap.parse_args()

    requested_beds = [b.strip() for b in args.testbeds.split(",") if b.strip()]
    for b in requested_beds:
        if b not in TESTBEDS:
            raise SystemExit(f"unknown testbed {b!r}; valid: {list(TESTBEDS.keys())}")

    print(f"Loading checkpoint {args.checkpoint} ...", flush=True)
    policy, params, policy_kind = load_checkpoint(args.checkpoint)

    output: dict = {
        "checkpoint": str(args.checkpoint),
        "arm": args.arm,
        "episodes": args.episodes,
        "deterministic": bool(args.deterministic),
        "testbeds": {},
    }

    for bed in requested_beds:
        bed_cfg = TESTBEDS[bed]
        arm_for_bed = bed_cfg["arm_override"] or args.arm
        env_kwargs = dict(ARM_CONFIGS[arm_for_bed])
        forced = bed_cfg.get("force_kwargs", {})
        env_kwargs.update(forced)
        print(f"\n[testbed={bed}] arm={arm_for_bed} env_kwargs={env_kwargs}", flush=True)

        env = FsmRedCC4Env(
            num_steps=DEFAULT_NUM_STEPS,
            topology_mode="generative",
            **env_kwargs,
        )
        rollout = _run_rollout(
            env=env,
            policy=policy,
            params=params,
            policy_kind=policy_kind,
            num_episodes=args.episodes,
            seed_offset=bed_cfg["seed_offset"],
            deterministic=args.deterministic,
        )
        scores = _score_all_episodes(rollout, args.episodes)

        consts = rollout["env_states"].const
        mission_indices = None
        if hasattr(consts, "mission_profile_index"):
            mission_indices = np.asarray(consts.mission_profile_index)

        norm_per_episode = [
            _norm_scorecard(rollout["actions"][ep], rollout["host_compromised"][ep]) for ep in range(args.episodes)
        ]

        msg_per_episode = None
        if "messages" in rollout:
            msgs = rollout["messages"]
            msg_per_episode = [_message_stats(msgs[ep]) for ep in range(args.episodes)]

        summary = _summarize(
            scores,
            mission_indices=mission_indices,
            norm_per_episode=norm_per_episode,
            msg_per_episode=msg_per_episode,
        )
        output["testbeds"][bed] = dict(arm_for_bed=arm_for_bed, env_kwargs=env_kwargs, **summary)

        r = summary["reward"]
        ci = summary["R_mean"]
        print(
            f"  reward={r['mean']:+.1f} ± {r['stderr']:.1f}  R_mean={ci['mean']:+.3f} ± {ci['stderr']:.3f}",
            flush=True,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
