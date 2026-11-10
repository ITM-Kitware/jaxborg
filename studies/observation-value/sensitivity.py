"""Matched input interventions on retained Blue decision inputs.

Evidence permutations move an observed host's complete evidence tuple to another
valid slot in the same subnet. They preserve structural input fields, the legal
mask and incoming recurrent carry. These input interventions measure dependence;
they do not establish that either edited input has a realizable trajectory.
"""

from __future__ import annotations

from collections import defaultdict
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from jaxborg.actions.encoding import (
    BLUE_ANALYSE_START,
    BLUE_DECOY_START,
    BLUE_REMOVE_START,
    BLUE_RESTORE_START,
)
from jaxborg.blue_observation_contract import observation_version_from_size
from jaxborg.constants import BLUE_OBS_SIZE, NUM_SUBNETS, OBS_VECTOR_HOSTS_PER_SUBNET
from jaxborg.observations import SUBNET_BLOCK_SIZE
from jaxborg.policies import initial_carry, is_recurrent, policy_step

HOST_SLOTS = 3 * OBS_VECTOR_HOSTS_PER_SUBNET
TARGET_ACTION_STARTS = np.asarray(
    [BLUE_ANALYSE_START, BLUE_REMOVE_START, BLUE_RESTORE_START, BLUE_DECOY_START], dtype=np.int32
)


def _validate(records):
    observations = np.asarray(records["observations"])
    if observations.ndim != 3 or observations.shape[1] != 5:
        raise ValueError("observations must have shape (rows, 5, observation_width)")
    count, agents, width = observations.shape
    if count == 0:
        raise ValueError("At least one retained input row is required")
    observation_version_from_size(width)
    expected = {
        "masks": (count, agents, 242),
        "genuine": (count, agents),
        "phase": (count,),
        "topology_index": (count,),
        "n_subnets": (count, agents),
    }
    for name, shape in expected.items():
        if np.asarray(records[name]).shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(observations).all():
        raise ValueError("Retained observations contain non-finite values")
    masks = np.asarray(records["masks"], dtype=bool)
    if not masks.any(axis=-1).all():
        raise ValueError("Every retained input must have a legal action")
    if not np.equal(observations[:, :, 0], np.asarray(records["phase"])[:, None]).all():
        raise ValueError("Retained phases disagree with observation phase inputs")
    subnets = np.asarray(records["n_subnets"])
    if not np.isin(subnets, (1, 2, 3)).all():
        raise ValueError("n_subnets must contain the actual observed subnet counts")
    if width == BLUE_OBS_SIZE:
        if np.asarray(records.get("host_counts")).shape != (count, agents, 3):
            raise ValueError("Base observations need host_counts with shape (rows, 5, 3)")
        if not np.isin(records["host_counts"], np.arange(OBS_VECTOR_HOSTS_PER_SUBNET + 1)).all():
            raise ValueError("host_counts must contain active non-router slot counts")
    return observations, masks, np.asarray(records["genuine"], dtype=bool)


def _carry_from_records(module, records):
    if not is_recurrent(module):
        return None
    template = initial_carry(module, 5)

    def trunk(value, prefix):
        if isinstance(value, tuple):
            return tuple(np.asarray(records[prefix + name]) for name in ("carry_c", "carry_h"))
        return np.asarray(records[prefix + "carry_h"])

    if isinstance(template, dict):
        carry = {name: trunk(value, name + "_") for name, value in template.items()}
    else:
        carry = trunk(template, "")
    count = len(records["observations"])
    for actual, expected in zip(jax.tree.leaves(carry), jax.tree.leaves(template), strict=True):
        if actual.shape != (count, *expected.shape):
            raise ValueError(f"Incoming carry shape {actual.shape} must be {(count, *expected.shape)}")
        if not np.isfinite(actual).all():
            raise ValueError("Retained carry contains non-finite values")
    return carry


@partial(jax.jit, static_argnames=("module",))
def _probabilities(module, weights, observations, masks, carry):
    if carry is None:
        logits = jax.vmap(lambda obs, mask: policy_step(module, weights, obs, mask)[0].logits)(observations, masks)
    else:
        logits = jax.vmap(lambda obs, mask, state: policy_step(module, weights, obs, mask, carry=state)[0].logits)(
            observations, masks, carry
        )
    return jax.nn.softmax(logits, axis=-1)


def _evaluate(module, weights, observations, masks, carry):
    result = []
    for start in range(0, len(observations), 64):
        stop = min(start + 64, len(observations))
        selection = np.minimum(np.arange(start, start + 64), stop - 1)
        states = None if carry is None else jax.tree.map(lambda value: jnp.asarray(value[selection]), carry)
        probabilities = _probabilities(
            module, weights, jnp.asarray(observations[selection]), jnp.asarray(masks[selection]), states
        )
        result.append(np.asarray(probabilities)[: stop - start])
    probabilities = np.concatenate(result)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=-1), 1, atol=1e-5):
        raise ValueError("Policy probabilities are not finite normalized distributions")
    if np.any(probabilities[~masks] > 1e-7):
        raise ValueError("Policy assigns probability to illegal actions")
    return probabilities


def _evidence_columns(width, subnet, slot):
    block = 1 + subnet * SUBNET_BLOCK_SIZE
    columns = [block + 3 * NUM_SUBNETS + slot, block + 3 * NUM_SUBNETS + OBS_VECTOR_HOSTS_PER_SUBNET + slot]
    if width > BLUE_OBS_SIZE:
        planes = (width - BLUE_OBS_SIZE) // HOST_SLOTS
        host = subnet * OBS_VECTOR_HOSTS_PER_SUBNET + slot
        columns.extend(BLUE_OBS_SIZE + plane * HOST_SLOTS + host for plane in range(1, planes))
    return np.asarray(columns)


def _permute_evidence(records, observations):
    edited = observations.copy()
    source = np.full(observations.shape[:2], -1, dtype=np.int16)
    destination = source.copy()
    width = observations.shape[-1]
    for row in range(len(observations)):
        for agent in range(5):
            for subnet in range(int(records["n_subnets"][row, agent])):
                if width > BLUE_OBS_SIZE:
                    first = BLUE_OBS_SIZE + subnet * OBS_VECTOR_HOSTS_PER_SUBNET
                    valid = np.flatnonzero(observations[row, agent, first : first + OBS_VECTOR_HOSTS_PER_SUBNET] == 1)
                else:
                    valid = np.arange(int(records["host_counts"][row, agent, subnet]))
                if len(valid) < 2:
                    continue
                signatures = [tuple(observations[row, agent, _evidence_columns(width, subnet, slot)]) for slot in valid]
                maximum, minimum = (
                    max(range(len(valid)), key=signatures.__getitem__),
                    min(range(len(valid)), key=signatures.__getitem__),
                )
                if signatures[maximum] == signatures[minimum]:
                    continue
                left, right = int(valid[maximum]), int(valid[minimum])
                left_columns, right_columns = (
                    _evidence_columns(width, subnet, left),
                    _evidence_columns(width, subnet, right),
                )
                edited[row, agent, left_columns] = observations[row, agent, right_columns]
                edited[row, agent, right_columns] = observations[row, agent, left_columns]
                source[row, agent] = subnet * OBS_VECTOR_HOSTS_PER_SUBNET + left
                destination[row, agent] = subnet * OBS_VECTOR_HOSTS_PER_SUBNET + right
                break
    return edited, source, destination


def _donor_carry(records, carry, genuine):
    donors = np.full(genuine.shape, -1, dtype=np.int32)
    if carry is None:
        return None, donors
    edited = jax.tree.map(np.copy, carry)
    actor = carry.get("actor", carry) if isinstance(carry, dict) else carry
    actor_leaves = jax.tree.leaves(actor)
    groups = defaultdict(list)
    for row, agent in zip(*np.nonzero(genuine), strict=True):
        key = (
            int(records["topology_index"][row]),
            int(records["phase"][row]),
            int(agent),
            np.asarray(records["masks"][row, agent], dtype=bool).tobytes(),
        )
        groups[key].append(int(row))
    for (_, _, agent, _), rows in groups.items():
        signatures = {row: b"".join(value[row, agent].tobytes() for value in actor_leaves) for row in rows}
        representatives = {}
        for row in rows:
            representatives.setdefault(signatures[row], row)
        if len(representatives) < 2:
            continue
        ordered = list(representatives)
        next_donor = {
            signature: representatives[ordered[(index + 1) % len(ordered)]] for index, signature in enumerate(ordered)
        }
        for row in rows:
            donor = next_donor[signatures[row]]
            donors[row, agent] = donor
            for out, original in zip(jax.tree.leaves(edited), jax.tree.leaves(carry), strict=True):
                out[row, agent] = original[donor, agent]
    return edited, donors


def _distances(original, changed):
    p, q = original.astype(np.float64), changed.astype(np.float64)
    midpoint = (p + q) / 2
    tiny = np.finfo(np.float64).tiny
    log_midpoint = np.log(np.maximum(midpoint, tiny))
    js = 0.5 * np.sum(
        p * (np.log(np.maximum(p, tiny)) - log_midpoint) + q * (np.log(np.maximum(q, tiny)) - log_midpoint), axis=-1
    )
    return {
        "tv": np.abs(p - q).sum(axis=-1) / 2,
        "js_nats": np.maximum(js, 0),
        "argmax_changed": np.argmax(p, axis=-1) != np.argmax(q, axis=-1),
    }


def _summary(distances, eligible):
    count = int(eligible.sum())
    result = {"decisions": count}
    for name, values in distances.items():
        selected = np.asarray(values)[eligible]
        result[name + "_mean"] = float(selected.mean()) if count else None
        if name != "argmax_changed":
            result[name + "_median"] = float(np.median(selected)) if count else None
            result[name + "_p90"] = float(np.quantile(selected, 0.9)) if count else None
    return result


def _target_probabilities(probabilities, slots):
    indices = np.maximum(slots, 0)[..., None] + TARGET_ACTION_STARTS
    return np.take_along_axis(probabilities, indices, axis=-1)


def analyze_inputs(module, weights, records: dict[str, np.ndarray]) -> tuple[dict, dict[str, np.ndarray]]:
    """Return descriptive dependence measurements without changing rollout state.

    Arrays have leading dimensions ``(rows, 5)``. Carry fields are incoming
    ``carry_c``/``carry_h`` for LSTM or ``carry_h`` for GRU; separate trunks add
    ``actor_`` and ``critic_`` prefixes. Enhanced inputs supply their own host
    validity plane. Base inputs require contiguous active ``host_counts``.
    """
    observations, masks, genuine = _validate(records)
    carry = _carry_from_records(module, records)
    original = _evaluate(module, weights, observations, masks, carry)
    edited, source, destination = _permute_evidence(records, observations)
    evidence_probabilities = _evaluate(module, weights, edited, masks, carry)
    evidence_eligible = genuine & (source >= 0)
    evidence_distances = _distances(original, evidence_probabilities)
    measurements = {
        "original_probabilities": original,
        "evidence_probabilities": evidence_probabilities,
        "evidence_observations": edited,
        "evidence_eligible": evidence_eligible,
        "source_slot": source,
        "destination_slot": destination,
        **{"evidence_" + name: values for name, values in evidence_distances.items()},
    }
    target_summary = {}
    for name, slots in (("source", source), ("destination", destination)):
        before = _target_probabilities(original, slots)
        after = _target_probabilities(evidence_probabilities, slots)
        measurements[name + "_action_probabilities_before"] = before
        measurements[name + "_action_probabilities_after"] = after
        measurements[name + "_target_mass_change"] = (after - before).sum(axis=-1)
        target_summary[name + "_target_mass_change_mean"] = (
            float(measurements[name + "_target_mass_change"][evidence_eligible].mean())
            if evidence_eligible.any()
            else None
        )
        for index, action in enumerate(("Analyse", "Remove", "Restore", "Decoy")):
            target_summary[name + "_" + action + "_probability_change_mean"] = (
                float((after - before)[..., index][evidence_eligible].mean()) if evidence_eligible.any() else None
            )
    summary = {
        "retained_rows": len(observations),
        "genuine_decisions": int(genuine.sum()),
        "evidence_permutation": {**_summary(evidence_distances, evidence_eligible), **target_summary},
        "evidence_selection": (
            "First observed subnet with unequal evidence; lexicographically greatest and least valid host tuples."
        ),
        "target_action_order": ["Analyse", "Remove", "Restore", "Decoy"],
        "interpretation": (
            "Input dependence only. Spatial evidence permutations are controlled counterfactual inputs; "
            "trajectory realizability is not established. Closed-loop returns determine functional value."
        ),
    }
    if "probabilities" in records:
        retained = np.asarray(records["probabilities"])
        if retained.shape != original.shape:
            raise ValueError("Retained probabilities have the wrong shape")
        error = float(np.abs(retained - original).max())
        summary["retained_probability_max_abs_error"] = error
        if error > 1e-4:
            raise ValueError(f"Recomputed probabilities differ from retained rollout probabilities by {error}")
    if carry is not None:
        initialized = initial_carry(module, 5)
        reset_carry = jax.tree.map(
            lambda value: np.broadcast_to(np.asarray(value), (len(observations), *value.shape)), initialized
        )
        reset_probabilities = _evaluate(module, weights, observations, masks, reset_carry)
        reset_distances = _distances(original, reset_probabilities)
        donor_carry, donors = _donor_carry(records, carry, genuine)
        donor_probabilities = _evaluate(module, weights, observations, masks, donor_carry)
        donor_distances = _distances(original, donor_probabilities)
        donor_eligible = genuine & (donors >= 0)
        measurements.update(
            reset_carry_probabilities=reset_probabilities,
            reset_carry_eligible=genuine,
            donor_carry_probabilities=donor_probabilities,
            donor_carry_eligible=donor_eligible,
            donor_row=donors,
            **{"reset_carry_" + name: values for name, values in reset_distances.items()},
            **{"donor_carry_" + name: values for name, values in donor_distances.items()},
        )
        summary["initial_carry_intervention"] = _summary(reset_distances, genuine)
        summary["matched_carry_intervention"] = _summary(donor_distances, donor_eligible)
        summary["carry_interpretation"] = (
            "Donors are actual incoming carries matched on receiving agent, topology, phase and exact legal mask. "
            "Combining a donor history with the recipient observation remains counterfactual. "
            "Initialized carry at later ticks is a memory-reset intervention, not a valid replacement history."
        )
    else:
        summary["carry_interpretation"] = "Feedforward actor; no recurrent carry."
    return summary, measurements
