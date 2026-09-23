"""Inference-only JAX port of Singh et al.'s released two-subpolicy H-MARL.

The actor inputs, 82/242-action heads, branch masks, file IOC priorities and
first-decoy-per-subnet rule follow upstream h-marl-3policy. Message delivery
uses one 8-bit host reference per sender per tick, with stable ordering in
place of Python set.pop(). See recipes/hmarl/README.md for provenance
and simulation differences.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file

from jaxborg.actions.encoding import (
    BLUE_ANALYSE_START,
    BLUE_DECOY_START,
    BLUE_MONITOR,
    BLUE_REMOVE_START,
    BLUE_RESTORE_START,
    BLUE_SLEEP,
)
from jaxborg.blue_ioc import _host_slots, initial_ioc_memory
from jaxborg.blue_ioc import update_ioc_memory as update_memory
from jaxborg.evaluation.matchup_runner import LoadedMatchupPolicy
from jaxborg.evaluation.stateful_blue import StatefulBluePolicy
from jaxborg.observations import CYBORG_POS_TO_JAX_ID, JAX_ID_TO_CYBORG_POS
from jaxborg.pretrained.hmarl_import import FORMAT, MANIFEST_PATH, UPSTREAM_COMMIT, UPSTREAM_REPO


def actor_logits(weights, agent: int, branch: str, obs):
    """Raw float32 inputs; Torch Linear weights have been transposed on import."""
    x = jnp.asarray(obs, dtype=jnp.float32)
    for layer in range(3):
        prefix = f"agent{agent}.{branch}.{layer}"
        x = x @ weights[f"{prefix}.kernel"] + weights[f"{prefix}.bias"]
        if layer < 2:
            x = jnp.tanh(x)
    return x


def action_layout(subnets: int):
    """Upstream unpadded order; traffic actions are unused by both branches."""
    n = 16 * subnets
    lookup = jnp.concatenate(
        [
            BLUE_ANALYSE_START + jnp.arange(n),
            jnp.array([BLUE_MONITOR]),
            BLUE_REMOVE_START + jnp.arange(n),
            BLUE_RESTORE_START + jnp.arange(n),
            jnp.array([BLUE_SLEEP]),
            jnp.full(16 * subnets, BLUE_SLEEP),
            BLUE_DECOY_START + jnp.arange(n),
        ]
    )
    return lookup


def branch_mask(subnets: int, branch: str, valid_hosts, busy):
    n = subnets * 16
    size = 80 * subnets + 2
    mask = jnp.zeros(size, dtype=jnp.bool_)
    if branch == "investigate":
        # Upstream get_action_indexes skips InvalidAction objects.
        mask = mask.at[:n].set(valid_hosts).at[-n:].set(valid_hosts)
    else:
        # FULL_MASK uses labels and includes padded Invalid Remove/Restore.
        mask = mask.at[n + 1 : 3 * n + 1].set(True)
    sleep_only = jnp.zeros_like(mask).at[3 * n + 1].set(True)
    return jnp.where(busy, sleep_only, mask)


def _sample(logits, mask, key, deterministic):
    masked = jnp.where(mask, logits, jnp.finfo(jnp.float32).min)
    return (
        jnp.argmax(masked).astype(jnp.int32) if deterministic else jax.random.categorical(key, masked).astype(jnp.int32)
    )


def policy_observations(state, const, agent, memory, iocs):
    subnets, hosts, valid = _host_slots(const, agent)
    process = jnp.where(valid, memory["process"][hosts], False)
    network = jnp.where(valid, memory["network"][hosts], False)
    phase = jnp.asarray([state.mission_phase], dtype=jnp.float32)
    investigate = jnp.concatenate([phase, jnp.concatenate([process, network, iocs], axis=1).reshape(-1)])
    recover = jnp.concatenate([phase, iocs.reshape(-1)])
    subnet_bits = jax.nn.one_hot(JAX_ID_TO_CYBORG_POS[subnets], 9)
    blocked = state.blocked_zones[subnets][:, CYBORG_POS_TO_JAX_ID]
    comms = const.comms_policy[state.mission_phase, subnets][:, CYBORG_POS_TO_JAX_ID]
    master = jnp.concatenate(
        [phase, jnp.concatenate([subnet_bits, blocked, comms, process, network, iocs], axis=1).reshape(-1)]
    )
    busy = state.blue_pending_ticks[agent] > 0
    return tuple(jnp.where(busy, jnp.ones_like(obs), obs).astype(jnp.float32) for obs in (master, investigate, recover))


@dataclass(frozen=True)
class HMARLPolicy(StatefulBluePolicy):
    variant: str

    def __post_init__(self):
        if self.variant not in ("expert", "meta"):
            raise ValueError("H-MARL variant must be expert or meta")

    def initialize(self, env_state):
        state = env_state.state
        n_red, n_hosts = state.red_sessions.shape
        connections = jnp.full((n_hosts, n_red), -1, dtype=jnp.int32)
        state = state.replace(blue_decoy_sources=connections, blue_old_decoy_sources=connections)
        memory = initial_ioc_memory(n_hosts)
        return env_state.replace(state=state), memory

    def select_actions(self, weights, env_state, key, carry, *, deterministic):
        state, const = env_state.state, env_state.const
        if state.blue_ioc_memory is not None:
            memory = state.blue_ioc_memory
            ioc_rows = [state.blue_ioc_codes[a, : (48 if a == 4 else 16)].reshape(-1, 16) for a in range(5)]
        else:
            memory, ioc_rows = update_memory(state, const, carry)
        keys = jax.random.split(key, 15).reshape(5, 3, 2)
        actions = []
        for agent in range(5):
            n_subnets = 3 if agent == 4 else 1
            _, _, valid = _host_slots(const, agent)
            master, investigate, recover = policy_observations(state, const, agent, memory, ioc_rows[agent])
            busy = state.blue_pending_ticks[agent] > 0
            if self.variant == "expert":
                branch = jnp.any(ioc_rows[agent] > 0).astype(jnp.int32)
            else:
                master_mask = jnp.array([True, state.time > 0])
                branch = _sample(
                    actor_logits(weights, agent, "master", master), master_mask, keys[agent, 0], deterministic
                )
            branch_actions = []
            for branch_index, (name, obs) in enumerate((("investigate", investigate), ("recover", recover))):
                mask = branch_mask(n_subnets, name, valid.reshape(-1), busy)
                choice = _sample(
                    actor_logits(weights, agent, name, obs), mask, keys[agent, branch_index + 1], deterministic
                )
                # Preserve upstream InvalidAction behavior for absent recovery hosts.
                lookup = action_layout(n_subnets)
                n = n_subnets * 16
                valid_recovery = jnp.tile(valid.reshape(-1), 2)
                lookup = lookup.at[n + 1 : 3 * n + 1].set(
                    jnp.where(valid_recovery, lookup[n + 1 : 3 * n + 1], BLUE_SLEEP)
                )
                branch_actions.append(lookup[choice])
            actions.append(jnp.where(branch == 0, branch_actions[0], branch_actions[1]))
        return jnp.stack(actions), memory


def load_policy(path, *, variant, team="blue", backend="jax"):
    if team != "blue" or backend != "jax":
        raise ValueError("Pretrained H-MARL is a JAX Blue policy")
    with safe_open(path, framework="numpy") as bundle:
        metadata = bundle.metadata()
    expected_manifest = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    if metadata.get("format") != FORMAT or metadata.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("Not a supported imported H-MARL checkpoint; run the H-MARL importer")
    if metadata.get("manifest_sha256") != expected_manifest:
        raise ValueError("H-MARL checkpoint manifest differs from this version of the importer")
    arrays = load_file(path)
    expected_keys = set()
    for agent in range(5):
        subnets = 3 if agent == 4 else 1
        for branch, inputs in (
            ("investigate", 1 + 48 * subnets),
            ("recover", 1 + 16 * subnets),
            ("master", 1 + 75 * subnets),
        ):
            outputs = 2 if branch == "master" else 2 + 80 * subnets
            for layer, shape in enumerate(((inputs, 256), (256, 256), (256, outputs))):
                for name, expected in (("kernel", shape), ("bias", (shape[1],))):
                    key = f"agent{agent}.{branch}.{layer}.{name}"
                    expected_keys.add(key)
                    if key not in arrays or arrays[key].shape != expected or arrays[key].dtype != np.float32:
                        raise ValueError(f"Invalid H-MARL tensor: {key}; expected float32 {expected}")
                    if not np.isfinite(arrays[key]).all():
                        raise ValueError(f"Non-finite H-MARL tensor: {key}")
    if set(arrays) != expected_keys:
        raise ValueError("Unexpected tensors in H-MARL bundle")
    return LoadedMatchupPolicy(
        "blue",
        "jax",
        HMARLPolicy(variant),
        jax.tree.map(jnp.asarray, arrays),
        dict(
            kind=f"pretrained_hmarl_{variant}",
            upstream_repository=UPSTREAM_REPO,
            upstream_commit=UPSTREAM_COMMIT,
            checkpoint="iter_49",
            architecture="256x256 tanh",
            checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            message_order="sorted_subnet_host",
            training_performed=False,
        ),
    )
