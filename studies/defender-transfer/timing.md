# CPU versus GPU evaluation timing

This engineering check asks where to run the existing scalar defender evaluator.
It performs no training and adds no independent research episodes to the transfer
comparison.

## Fixed workload

Use saved `blue-single-42-final`, stock native case 0, and its exact policy and
dynamics seeds from the completed matched-system benchmark manifest. Run the same
500-step episode four times on each backend: one compile-inclusive first call,
then three warmed repeats. CPU and CUDA use separate processes, fresh separate
JAX compilation caches, and the same eight allowed CPU slots. The CUDA process
also uses the one GPU assigned by Slurm. Workers run sequentially, CPU first.

The timed call is the unchanged `benchmark_rollout.run_episode`. It synchronizes
results and includes topology loading, device transfers, simulator/policy work,
validation and trajectory construction. Model loading and synchronization,
worker initialization, evidence writes and total worker-process time are reported
separately. A fresh JAX cache does not mean a cold operating-system file cache.

Report warm median/range, episodes per second, process CPU time, first-call time,
and reward/action agreement. Different floating-point backends may produce
different sampled trajectories even with identical input seeds. Repeated inputs
are timing repeats, not independent scientific replicates. One policy and one
scenario provide a small scheduling pilot, not a general performance claim.

This tests scalar evaluation. It does not settle GPU `vmap`/batch-16 throughput
or multiple concurrent CPU workers. Preserve this distinction when choosing
resources for the larger transfer harness.

## Execution and records

Status: prepared; no timing result is claimed yet.

```bash
bash studies/defender-transfer/time_evaluation_launch.sh submit \
  "$HOME/jaxborg-exp/shared/studies/defender-eval-timing-20261107" \
  "$HOME/jaxborg-exp/shared/studies/defender-benchmark-20261107/inputs/manifest.json"
```

Use clean committed source and the existing study runtime setup. The launcher
requests `community`, one GPU, eight CPU slots, 32 GiB, and a 30-minute limit.
Slurm binds the task to assigned cores; each worker records its actual affinity.
The original device assignment is preserved, with JAX_PLATFORMS selecting CPU or
CUDA before imports. Outputs are external to the source checkout.

The output root retains the source archive, immutable plan/protocol, copied
weights/recipe/snapshot, captured execution source and dependencies, all timed
trajectories, per-backend receipts, exact selection, and `summary.json`. Shared
MLflow experiment: `defender-eval-throughput`. Existing evidence roots are never
silently reused.
