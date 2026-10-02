# Matched Blue credit replication

The protocol replicates the seed-11001 learning-mechanism study with historical
fresh Blue seed 22001 at 1.92M, against frozen original Red from source 9.6M,
co-training seed 42. Both arms use the same calibration and starting full state,
20 updates (960,000 added steps), stock CC4, topology seed 0, stochastic actions,
500-step episodes, gamma .99 and the inherited architecture/optimizer.
Only GAE lambda changes from .95 to 1. The historical optimizer state is absent;
these remain warm starts with fresh Adam and calibrated normalization.

`blue-credit-replication-control.yaml` and `blue-credit-replication-lambda1.yaml`
reserve a new final cohort, **10200000–10200127**, before execution. Original
co-trained Blue, the starting policy and both final policies are evaluated on
that cohort, both unrestricted and with Block excluded. Intermediate models,
states and validation results are retained at updates 5, 10, 15 and 20; full
minibatch diagnostics at 1, 5, 10 and 20. The final endpoint is fixed in advance
and is not selected using confirmation results.

The control performs the initial legal-action forks; its variance profile and
fork captures support later matched credit replay. Strict open-mission Block
and mission-route reopening Allow diagnostics account for the reverse bit.
These additional classifications are read-only; they change no learning loss,
policy sampling, observation, action rule or reward. Legacy classifications are
retained, including newly blocked own bits whose reverse direction is closed.

Use a clean pinned checkout, the matching GPU environment and one existing
shared MLflow store outside Git. On this machine, reserve GPUs only through
Slurm community. Each job requests one GPU for at most two hours. The second job
depends on successful completion of the first and reuses its calibrated state.
Keep the checkout unchanged until both jobs finish.

```bash
export JAXBORG_DIAGNOSTIC_PYTHON=/absolute/path/to/compatible/venv/bin/python
export JAXBORG_EXP_DIR=/absolute/path/to/existing/shared/tracking-root

blue_control_job=$(bash scripts/experiments/submit_blue_learning_mechanism.sh \
  campaigns/response-oracles/blue-credit-replication-control.yaml \
  /path/to/equilibrium-data /path/to/replication/control \
  /path/to/original-0d577fa-checkout)
blue_control_job=${blue_control_job%%;*}

JAXBORG_SLURM_AFTEROK="$blue_control_job" \
  bash scripts/experiments/submit_blue_learning_mechanism.sh \
  campaigns/response-oracles/blue-credit-replication-lambda1.yaml \
  /path/to/equilibrium-data /path/to/replication/lambda1 \
  /path/to/original-0d577fa-checkout \
  --controlled-from /path/to/replication/control
```

The data directory must retain `eval/blue_diagnosis/checkpoints.csv`, all three
seed-22001/original models selected by that index, their recipes, the source
topology and `eval/blue_diagnosis/ippo-9600000/evaluations.json`. The archival GPU
check now selects the actual fresh training seed instead of hardcoding 11001.
The original reference checkout must pin
`0d577faab9ac3e0a06376e99e9592aadf128c130`. The launcher rejects CPU fallback and
records exact source identity, dependencies and Slurm allocation. Compilation
cache defaults to the shared tracking root to avoid additional writable roots.

Analyze completed results without a GPU, keeping exports separate from raw data:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src:. python -m scripts.experiments.analyze_blue_credit_replication \
  --control /path/to/replication/control --variant /path/to/replication/lambda1 \
  --output /path/to/replication/analysis
```

The pair auditor requires both runs completed, only effective lambda differing,
identical numeric calibrated states, complete registered training budgets and
confirmation cohorts. It checks duplicate control/starting/original evaluations
when the cohort is common, reproduces paired intervals, audits reward sums and
legality, and plots raw curves and component gains. It reports one independent
training seed for the pair; reset episodes are not independent training runs.
With `--credit /path/to/credit-replay`, it additionally recomputes every archived
conditional estimate from raw arrays, including the multiplicity correction.
It uses NumPy, PyYAML and Matplotlib; it does not require JAX or MLflow for analysis.

Proceed to a controlled actor-advantage lambda × critic-target lambda 2×2 only
after examining this new replication. These diagonal arms use one lambda for
both; they are not the full factorial experiment. The opt-in mixed-lambda
implementation, separation checks and bounded launch protocol are documented
in [README-blue-credit-factorial.md](README-blue-credit-factorial.md).
Preserve failed attempts, use a newly reserved confirmation cohort for that
step, and keep fresh starts, original-Blue warm starts and Block exclusion as
distinct conditions. No default lambda or stock environment rule is changed.
