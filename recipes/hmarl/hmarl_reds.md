# H-MARL Red evaluation

[Singh et al., section 5.2](https://arxiv.org/abs/2410.17351) evaluates four
finite-state Reds. JAXborg provides these selectors in both JAX and CybORG:

| Selector | Change from the default FSM |
| --- | --- |
| `fsm` | Stock CC4; equal aggressive/stealth scan and Impact/Degrade choices |
| `aggressive` | All service discovery uses the aggressive scan (1 tick, 0.75 detection probability) |
| `stealthy` | All service discovery uses the stealth scan (3 ticks, 0.25 detection probability) |
| `impact` | All Degrade probability moves to Impact; scanning remains stock |

These implement the paper's description. The released GitHub code imports the
stock `FiniteStateRedAgent` and does not include separate variant definitions.
Only the scan columns in states K/KD or the objective columns in R/RD change.
Discover remains 0.5 in K and R; host choice stays uniform, with stock deception,
privilege escalation, action durations, and pending-action handling.

Run the four-opponent JAX suite on a final Blue checkpoint (including LSTM):

```bash
uv run python scripts/eval/eval_hmarl_reds.py \
  --model /path/to/model_RUN.safetensors
```

The model's sidecar supplies its recipe. `--recipe` can select an explicit recipe.
For the released H-MARL actors, the same entry point loads the correct Expert or
Meta policy and prepares missing weights using the existing importer:

```bash
uv run python scripts/eval/eval_hmarl_reds.py --recipe hmarl_expert
uv run python scripts/eval/eval_hmarl_reds.py --recipe hmarl_meta
```

The standard and diverse IPPO, IPPO-LSTM, MAPPO and MAPPO-LSTM recipes schedule this script
as `hmarl-reds` in `eval.after_training`, selecting Aggressive, Stealthy and Impact.
Default/FSM is already covered by their `scripted-reds` suite. H-MARL's regular
`eval_hmarl.py` sweep includes all three variants alongside FSM and CIA C/I/A.
Invoked directly, the dedicated H-MARL Red script defaults to
`fsm aggressive stealthy impact`; `--reds` selects a subset.

The configured protocol is 10 held-out topologies × 10 seeds × 6 episodes =
600 episodes per opponent, each 500 steps. The evaluator reuses the exhaustive
bank, episode seeds, fixed CIA role maps, and recurrent/stateful policy carry.
It requires the existing CIA-enabled evaluation configuration and bank. These
are JAXborg results under that protocol, not reproductions of the paper's scores.

`--seeds`, `--episodes-per-seed`, `--episode-length`, `--topology-path`, and
`--deterministic` override the protocol. For a smoke run, select one existing
held-out topology and a short duration. CPU defaults to scalar scans; use
`JAX_PLATFORMS=cuda` for GPU batching or `JAXBORG_EVAL_BATCH_SIZE` to set its size.
Results are JSONL (`suite: hmarl_reds`) with per-opponent rewards, CIA metrics,
seeds, topology identities, policy provenance and episode length. `--output`
sets the destination; otherwise results go under `$JAXBORG_EXP_DIR/eval/`.
Training runs receive MLflow metrics under `eval.after_training.hmarl-reds`;
`--no-mlflow` disables attachment.
