# Pretrained H-MARL evaluation

These recipes evaluate **H-MARL Expert** and **H-MARL Meta** from
[Singh et al., arXiv:2410.17351](https://arxiv.org/abs/2410.17351).
Both policy inference and environment rollouts run in JAX. There is no training
configuration, optimizer, or training entry point in this implementation.

```bash
uv run python scripts/eval/eval_hmarl.py --recipe hmarl_expert
uv run python scripts/eval/eval_hmarl.py --recipe hmarl_meta
```

The first invocation downloads the released weights, checks their SHA-256 hashes,
and converts the actors into `.cache/pretrained/hmarl/actors.safetensors`.
The two recipes share this cache. Subsequent invocations work offline once the
weights and topology bank are present. The binary weights are not added to Git.
To prepare weights without running evaluation:

```bash
uv run python scripts/eval/eval_hmarl.py --recipe hmarl_expert --prepare-only
```

A local clone of the upstream repository can supply the weights using
`--upstream-dir /path/to/Hierarchical-MARL`. This option is used only when the
converted cache is missing. To explicitly reconvert:

```bash
uv run python -m jaxborg.pretrained.hmarl_import --upstream-dir /path/to/Hierarchical-MARL
```

The CLI defaults to CPU, matching the existing scripted-Red evaluation script.
On a CUDA-enabled JAX installation, prefix the commands with `JAX_PLATFORMS=cuda`.
CPU defaults to a scalar episode scan, avoiding costly vectorization of a single
simulator. Accelerators use the existing evaluator's vmapped batch default.
`JAXBORG_EVAL_BATCH_SIZE` overrides the batch size on either backend. The first
episode/batch includes JAX compilation and can take several minutes.

## Two-GPU IPPO-LSTM comparison

From the repository root:

```bash
# Inspect the complete command plan without starting jobs.
bash scripts/temp/run_lstm_hmarl.sh --dry-run

mkdir -p outputs
sbatch scripts/temp/run_lstm_hmarl.sh
# Or, inside an existing two-GPU allocation:
# bash scripts/temp/run_lstm_hmarl.sh
```

The launcher first trains both `cotraining_lstm` and
`cotraining_lstm_env_diversity`, each with seeds **42, 100, 200** and its
unchanged 50M-step budget. Both use the new 450-input enhanced observations.
It then evaluates all six final LSTM Blues against FSM and CIA C/I/A.
Finally, **Expert and Meta run on one GPU each**; each faces the same four
scripted opponents and all **six final cotrained Reds**, covering both training
conditions and all seeds. No historical checkpoints are selected, and H-MARL
is never trained. All matchups use the evaluation contract below: 600 episodes
per opponent, 500 steps each (6,000 episodes per H-MARL policy).

The launcher explicitly schedules these suites; the cotraining recipes' automatic
history/cross-play and native CybORG suites are deferred and are not part of this
comparison. Comparing the LSTM Blues with H-MARL on learned Reds would additionally
require evaluating those Blues against the same Red pool with appropriate
training-seed exclusions; the common FSM/CIA sweep is directly comparable here.

Models, sidecars, frozen recipe copies, logs, JSONL results and MLflow records
are saved under **`jaxborg-harml-comparison/`** (spelling intentional).
`JAXBORG_COMPARISON_DIR` overrides this root, independently of any old
`JAXBORG_EXP_DIR` value. Each invocation gets a unique run ID:

```text
jaxborg-harml-comparison/
  ippo_jax/<recipe>_seed<seed>_<run-id>/model_<tag>.safetensors
  eval/<run-id>/<tag>_scripted.jsonl
  eval/<run-id>/hmarl_expert.jsonl
  eval/<run-id>/hmarl_meta.jsonl
  logs/<run-id>/
  recipes/<run-id>/
  mlflow.db
```

`JAXBORG_SEEDS="42 100 200"` controls training seeds. The two device IDs supplied
by `CUDA_VISIBLE_DEVICES` are preserved, including UUIDs. With no device selection,
the default is `0,1`. Failed training prevents evaluation; existing model directories
are never overwritten. After successful training, rerun evaluations with
`--eval-only --run-id <run-id>` and the same seed/root settings. Recipe snapshots
and saved model sidecars keep those reruns on the original observation contract.

The existing `scripts/temp/sync/pull_runs.sh` now downloads this folder into
`remote/jaxborg-harml-comparison/` and rewrites its MLflow artifact paths, alongside
the original experiment folder. Custom comparison-root names need a matching sync
filter. The sync script remains in the repository's ignored `scripts/temp/` tree.

For independent H-MARL evaluations, repeat `--red-model` with exact final bundle
paths; these are evaluated after the scripted opponents. `--learned-only` skips
the scripted sweep:

```bash
uv run python scripts/eval/eval_hmarl.py --recipe hmarl_expert \
  --red-model /path/to/model_cotraining_lstm_seed42_RUNID.safetensors \
  --red-model /path/to/model_cotraining_lstm_env_diversity_seed42_RUNID.safetensors
```

Learned-opponent rows include the Red model path, training condition, seed, step
count, policy provenance, per-episode rewards/CIA, topology and role-map identities.
Each completed learned matchup is saved before starting the next one.

## Evaluation contract

Both recipes match the JAX scripted-Red sweep in
`recipes/cotraining/cotraining_lstm_env_diversity.yaml`:

| Setting | Value |
| --- | --- |
| Red opponents | `fsm`, `cia_c`, `cia_i`, `cia_a` |
| Episode length | 500 environment steps, with native action durations |
| Evaluation seeds | 1000–1009 |
| Episodes per seed per topology | 6 |
| Topology seeds | 100–109, generated by JAX |
| Operational-zone servers | 3 |
| Topology sampling | Exhaustive |
| Policy sampling | Stochastic |
| CIA metric | Resilience, fixed roles per topology |
| Budget | 600 episodes per Red, 2,400 per recipe |

The existing topology cache, episode-seed expansion, role-map fingerprints,
reward calculation and temporal CIA scoring are reused. Results use the existing
scripted-Red JSONL schema under `jaxborg-exp/eval/`, with additional checkpoint
provenance and episode duration. The `trained_backend` is `rllib_torch`, while
`policy_backend` and the simulator are JAX. These are pretrained checkpoints,
so there is no local training seed or local training MLflow run to attach to.

For a short smoke run on one existing topology:

```bash
JAXBORG_EVAL_BATCH_SIZE=1 uv run python scripts/eval/eval_hmarl.py \
  --recipe hmarl_meta --reds fsm cia_c cia_i cia_a \
  --seeds 1000 --episodes-per-seed 1 --episode-length 20 \
  --topology-path .bank_cache/topologies/cotraining/eval_ops3/jax_ops3_seed_0000000100.snapshot.npz
```

Omit `--episode-length` to retain the 500-step duration. CLI overrides are recorded
in the result. `--output` selects a JSONL path. `--deterministic` switches both
master and subpolicy decisions to argmax.

## Imported checkpoints and architecture

The importer is pinned to
[upstream commit 6fe960f5931f2d7dc5ef522f09ca4a179f377ee4](https://github.com/adityavs14/Hierarchical-MARL/tree/6fe960f5931f2d7dc5ef522f09ca4a179f377ee4).
The per-file hashes are in `src/jaxborg/pretrained/hmarl_manifest.json`.

- Both models use `h-marl-3policy/saved_policies/sub/iter_49/policies/Agent{0..4}_{investigate,recover}`.
- Meta also uses `h-marl-3policy/saved_policies/master/iter_49/policies/Agent{0..4}_master`.
- Expert chooses Recover when an IOC is present and Investigate otherwise.
- Meta uses its learned two-action master. Its initial decision is Investigate,
  matching the upstream wrapper's reset mask.

Here `h-marl-3policy` means one master plus **two** subpolicies. The separate
three-subpolicy/control-traffic experiment is not selected by these recipes.

Each released actor has two independent 256-unit tanh hidden layers and a linear
logit head. Observations are raw float32 values (`NoFilter`, preprocessing
disabled), as specified in the saved RLlib configuration. Branch agents have
49/17/76 inputs for Investigate/Recover/Master; the headquarters agent has
145/49/226. Primitive heads have 82 or 242 actions, and master heads have 2.
The saved checkpoint dimensions take precedence over the paper's approximate
space-size descriptions. Critics and optimizer state are not needed to evaluate.
The importer reads only the leading NumPy array dictionary through a restricted
unpickler; it never reconstructs RLlib objects or executes checkpoint functions.

## Observation and simulation scope

The adapter preserves per-defender policies, subnet/host order, persistent
process/network alerts, file IOC priorities (root=1, user=2), first decoy IOC per
subnet (3), branch-specific action masks, and Sleep-only behavior while busy.
Recovery clears local IOC memory. The optional telemetry records observed remote
hosts connecting to decoys; it does not reveal latent Red compromise/session
state to the policy. New cotraining runs with `cage4_enhanced_obs: true` also use
this telemetry and the shared `jaxborg.blue_ioc` memory/delivery implementation.
Unenhanced runs and old enhanced-v1 checkpoints retain their previous contracts.
See [enhanced observations](../../docs/cage4_enhanced_observations.md) for the
450-input cotraining layout and retraining requirements.

Upstream communicates one 8-bit `(subnet, host slot)` reference per sender per
step. The JAX adapter preserves that bandwidth and next-step delivery, but sends
pending messages in sorted subnet/slot order rather than Python `set.pop()` order.
Upstream's full recovery mask includes padded invalid host actions; those remain
selectable and are translated to a native Sleep/no-op. Valid host actions keep
their native durations. The adapter does not enable traffic-control actions.

These are evaluations of the released actors in **JAXborg's simulator**, not a
claim to reproduce the paper's CybORG scores or random trajectories exactly.
Simulator parity differences, the message ordering above, and the cotraining
held-out topology/CIA protocol can change rewards. Numerical actor equivalence
is checked independently of those environment differences.
