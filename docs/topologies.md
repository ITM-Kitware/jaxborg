# Controlling CC4 topologies

JAXborg can train a Blue policy against a fixed, pre-built pool of topology
snapshots instead of generating a new static CC4 topology on every reset. The
pool is configured in the training recipe and is supported by both JAX
Blue-versus-scripted-Red training and JAX Blue/Red co-training.

## Support by training path

| Training path | Pre-built topology pool |
| --- | --- |
| JAX Blue versus scripted FSM Red | Supported |
| JAX with a learned or frozen learned Red, including co-training | Supported |
| CybORG/CleanRL | Not currently supported |

The CybORG/CleanRL path continues to use `EnterpriseScenarioGenerator` and
regenerates the scenario normally. Supplying `train.topology_bank` to that
backend does not currently affect its environments.

## Use the checked-in topology bank

The repository includes 16 snapshots in `scripts/dev/topology_bank/`. The
topology-only example is `recipes/cec_phase6_topo_10M.yaml`:

```yaml
train:
  episode_length: 500
  total_timesteps: 10000000
  variant: cc4_stock
  topology_bank:
    - scripts/dev/topology_bank/shape_00.snapshot.npz
    - scripts/dev/topology_bank/shape_01.snapshot.npz
    - scripts/dev/topology_bank/shape_02.snapshot.npz
    # ...list exactly the snapshots that should be eligible...
```

Launch that recipe with:

```bash
./scripts/train/run.sh jax cec_phase6_topo_10M 42
```

For a shorter run, override its duration:

```bash
./scripts/train/run.sh jax cec_phase6_topo_10M 42 \
  --total-timesteps 3000000
```

`recipes/cec_phase6_C11.yaml` uses the same 16-topology pool together with
mission, phase-boundary, and crown-jewel reward banks. Use that recipe when
the goal is broader environment randomisation rather than topology-only
variation.

The same `train.topology_bank` field works in a JAX co-training recipe:

```yaml
train:
  teams: both
  variant: cc4_stock
  topology_bank:
    - scripts/dev/topology_bank/shape_00.snapshot.npz
    - scripts/dev/topology_bank/shape_01.snapshot.npz
```

Both policies act in one shared environment, so one topology is sampled per
environment reset and is seen by both teams. Blue and Red cannot configure
different topology pools.

## Declare disjoint training and evaluation seed ranges

Use `topology_generation` when the pool should be generated from an exact
contiguous seed range instead of enumerating snapshot paths by hand. This
example creates 100 CAGE-generated training layouts from seeds 0 through 99
and 20 held-out layouts from seeds 100 through 119:

```yaml
train:
  teams: both
  variant: cc4_stock
  topology_generation:
    generator: cyborg
    seed_start: 0
    count: 100
    cache_dir: topology_banks/my_experiment/train

eval:
  variant: cc4_stock
  topology_generation:
    generator: cyborg
    seed_start: 100
    count: 20
    cache_dir: topology_banks/my_experiment/test
  topology_sampling: exhaustive
```

`seed_end` is an inclusive alternative to `count`; specify exactly one:

```yaml
topology_generation:
  generator: cyborg
  seed_start: 100
  seed_end: 119
  cache_dir: topology_banks/my_experiment/test
```

A generated bank is limited to 10,000 entries so a malformed range cannot
exhaust memory during recipe validation. In practice, loaded banks will hit
host/device-memory limits well before that ceiling.

Choose `generator: cyborg` for the real CAGE 4
`EnterpriseScenarioGenerator`, or `generator: jax` for JAXborg's pure-JAX
generator. Missing files are generated lazily when the training or evaluation
pool is first needed and then reused. Their deterministic names include the
generator and zero-padded source seed, for example
`cyborg_seed_0000000100.snapshot.npz`.

Cache reuse checks the source seed, snapshot format, required arrays, and
scenario-configuration digest. The generating JAXborg revision and CybORG
version remain in each snapshot's metadata for audit. Cached files are
immutable: materialization publishes them atomically, so concurrent training
jobs cannot observe or overwrite a partial snapshot. To regenerate a range
under newer generator code, choose a new `cache_dir` or remove the old cache
deliberately.

To inspect or pre-build both pools before launching a job:

```bash
uv run materialize-topologies --recipe my_recipe --dry-run
uv run materialize-topologies --recipe my_recipe
```

Use `--scope train` or `--scope eval` to materialize only one side. Relative
cache paths are resolved from the repository root.

Recipe loading rejects overlapping canonical paths and overlapping
`(generator, source_seed)` provenance before generation starts. Cached files
with unexpected provenance are never overwritten. When both sides use
explicit `topology_bank` lists, their snapshots must contain source metadata
so disjointness can be verified. These checks prevent seed/path leakage; they
do not attempt to prove that two different generator seeds can never happen to
produce structurally identical layouts.

## Control the number of training topologies

For an explicit bank, the number of paths listed under
`train.topology_bank` is the pool size. For a generated bank,
`train.topology_generation.count` is the pool size:

- one listed snapshot produces a fixed training topology;
- N listed snapshots restrict training to those N topologies;
- omitting the field restores generative topology construction on every reset.

`jax.num_envs` controls the number of parallel environment instances. It does
not control the number of distinct topologies.

At every episode reset, each parallel environment samples one bank entry
uniformly with replacement. A fixed training seed reproduces the sampling
sequence. Sampling is not round-robin, and there is no quota that guarantees
each entry will be used the same number of times. Weighted sampling,
curricula, and without-replacement sampling are not currently implemented.

Explicit paths may be absolute or relative. Relative paths are resolved from
the repository root, not from the recipe's directory. Globs and directory
shorthand are not expanded, so every eligible explicit snapshot must be
listed.

## Build a custom diversified bank

The bank builder accepts both a count and a base seed:

```bash
uv run python scripts/dev/build_topology_bank.py \
  --out-dir topology_banks/my_bank \
  --count 8 \
  --seed 0
```

List the resulting eight files under `train.topology_bank` in a copied recipe.
The builder deliberately varies several structural axes:

- router adjacency;
- operational-zone server counts;
- seed-driven host counts and services;
- a cross-segment allow-list edge.

The checked-in 16-snapshot bank combines four perturbation patterns with base
seeds 0 through 3. These snapshots provide deliberate JAXborg domain
randomisation; they are not all untouched outputs from the upstream CAGE 4
generator.

## Build a pool from the real CybORG generator

To restrict training to layouts produced directly by CybORG/CAGE, export one
snapshot per chosen seed:

```bash
uv run export-cyborg-topology \
  --seed 0 \
  --out topology_banks/cage/shape_00.snapshot.npz
```

Repeat with different seeds, then enumerate the generated files in
`train.topology_bank`. For example, this creates eight snapshots:

```bash
for seed in $(seq 0 7); do
  printf -v name 'shape_%02d.snapshot.npz' "$seed"
  uv run export-cyborg-topology \
    --seed "$seed" \
    --out "topology_banks/cage/$name"
done
```

`export-generated-topology` is also available, but it uses JAXborg's pure-JAX
generator rather than a live CybORG `EnterpriseScenarioGenerator`.

## What a snapshot fixes

A snapshot stores the static simulator constants, including host layout and
types, links, initial services and PIDs, Blue and Red mappings and start hosts,
Green host assignments, mission phase tables, reward tables, and allow-list
configuration.

It does not serialize a complete episode. Per-step Green, Red, Blue, action,
and simulator randomness continues normally after reset. The pool therefore
controls the static environments encountered by the policy without turning
training into deterministic trajectory replay.

Snapshots contain a scenario-configuration digest. Loading fails explicitly
if a snapshot was produced for incompatible host, subnet, mission-phase, or
other fixed dimensions.

## Evaluate the held-out pool

Evaluation never falls back to `train.topology_bank`; the evaluation pool must
be declared independently under `eval`. Both `eval.topology_generation` and an
explicit `eval.topology_bank` are supported by the JAX-native learned-policy
matchup, checkpoint matchup, Phase 6 scripted-Red evaluator, and JAX baseline
evaluator.

`eval.topology_sampling` controls coverage:

- `exhaustive` (the default) runs every requested dynamics seed/episode on
  every held-out topology using paired rollout keys. The matchup and Phase 6
  JSON outputs record the exact per-episode topology path;
- `random` preserves training-style uniform sampling with replacement from
  the evaluation pool and therefore does not guarantee equal coverage.

For learned Blue-versus-learned Red evaluation, the recipe is enough:

```bash
uv run python scripts/eval/eval_matchup.py \
  --recipe my_recipe \
  --seeds 1000-1009 \
  --episodes-per-seed 1
```

With exhaustive sampling, this example runs each of the ten dynamics seeds on
each held-out topology. `--episodes-per-seed` remains episodes per seed per topology.
The periodic co-training checkpoint evaluator uses the same held-out recipe
pool automatically and reports aggregate team means. The baseline evaluator
prints aggregate results rather than writing per-episode topology assignments.

For JAX-native Blue-versus-scripted-Red evaluation, the checkpoint sidecar's
`eval` pool is loaded automatically:

```bash
uv run python scripts/eval/cec_phase6_eval_jax.py \
  --model path/to/model.safetensors \
  --eval-red fsm \
  --episodes 5
```

Here `--episodes 5` means five dynamics rollouts per held-out topology under
exhaustive sampling. Any of these JAX-native commands can override the recipe
by repeating `--topology-path FILE`, and can request replacement sampling with
`--topology-sampling random`.

The real CybORG/CleanRL trainer and CybORG contract evaluators still cannot
consume JAX snapshot banks. `eval.scripted_red.seeds` controls full CybORG
episode seeds—not this held-out snapshot split—and changes both static topology
and episode dynamics.

The joint JAX path currently forwards the topology bank only. The independent
mission, phase-boundary, and phase-reward diversity banks used by
`cec_phase6_C11.yaml` are not yet wired into co-training.

## Implementation references

- `src/jaxborg/topology_banks.py`: expands seed ranges, materializes caches, and validates splits.
- `src/jaxborg/recipe.py`: projects train and evaluation topology pools.
- `scripts/train/algorithms/ippo_jax.py`: passes the pool to the Blue trainer.
- `scripts/train/algorithms/ippo_jax_joint.py`: passes the shared pool to joint training.
- `src/jaxborg/env.py`: loads, stacks, and samples topology constants.
- `src/jaxborg/scenarios/cc4/topology.py`: snapshot serialization and validation.
- `src/jaxborg/scenarios/cc4/topology_cli.py`: generated and CybORG exporters.
- `src/jaxborg/topology_bank_cli.py`: recipe pool inspection and pre-generation CLI.
- `scripts/dev/build_topology_bank.py`: diversified bank construction.
