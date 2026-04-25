"""Diagnostic: how much actual diversity does generative-mode build_topology produce?

Generates N topologies with distinct PRNG keys and reports the number of distinct
values across topology fields that vary per-key, plus per-field histograms. Used to
verify gen-base is consuming its diversity before we commit to a training matrix.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
from collections import Counter

import jax
import numpy as np

from jaxborg.topology import build_topology


def fingerprint(arr) -> bytes:
    return np.asarray(arr).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base_key = jax.random.PRNGKey(args.seed)
    keys = jax.random.split(base_key, args.n)

    build_jit = jax.jit(lambda k: build_topology(k, num_steps=500, training_mode=False))
    print("compiling...", flush=True)
    _ = build_jit(keys[0])
    print(f"compiled. building {args.n} topologies sequentially (JIT'd)...", flush=True)

    fields = [
        "host_active",
        "host_subnet",
        "host_is_server",
        "host_is_user",
        "initial_services",
        "data_links",
        "red_start_hosts",
        "red_initial_discovered_hosts",
        "host_initial_max_pid",
        "blue_agent_hosts",
        "phase_rewards",
        "allowed_subnet_pairs",
        "comms_policy",
        "subnet_adjacency",
    ]

    distinct = {f: set() for f in fields}
    red_start_counter = Counter()
    nhost_counter = Counter()

    for i, k in enumerate(keys):
        c = build_jit(k)
        for f in fields:
            distinct[f].add(fingerprint(getattr(c, f)))
        red_start_counter[tuple(np.asarray(c.red_start_hosts).tolist())] += 1
        nhost_counter[int(np.asarray(c.num_hosts))] += 1
        if (i + 1) % 64 == 0:
            print(f"  built {i + 1}/{args.n}", flush=True)

    print(f"\n=== diversity report (n={args.n}) ===")
    for f in fields:
        n = len(distinct[f])
        flag = "" if n > 1 else "  ← CONSTANT (no variation)"
        print(f"  {f:32s} distinct={n:5d}{flag}")

    print("\nnum_hosts histogram (top 10):")
    for v, c in sorted(nhost_counter.items()):
        print(f"  {v:4d}: {c}")

    print(f"\nred_start_hosts: {len(red_start_counter)} distinct tuples in {args.n} samples")
    print("  top 5 most-frequent red_start tuples:")
    for tup, cnt in red_start_counter.most_common(5):
        print(f"    {cnt:4d}× {tup}")


if __name__ == "__main__":
    main()
