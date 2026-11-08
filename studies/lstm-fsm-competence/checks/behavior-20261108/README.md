# Behavior across checkpoints and opponents

**Completed:** all 16 traces reproduce their original returns. Read the
[behavior findings](results.md): later Blue Analyses more and restores fewer
hosts, with both earlier targeted recovery and costly missed recoveries.

The [protocol](protocol.json) compares seed 42 at 49.968M and 124.992M steps against
FSM and diverse-seed100 Red, on four retained layouts and episode root 21000.
These are exploratory cases, not a replacement for the full score panels.

ITM output:

```text
/home/local/KHQ/paul.elliott/jaxborg-exp/shared/studies/lstm-fsm-competence-20261107-v3/checks/behavior-20261108-3d97d4c4/
```

The [launch record](launch-record.json) retains the exact inputs and evaluation
source. The [analysis record](analysis-record.json) retains artifact hashes,
exact producing helper bytes and the focused observation-parity check.

Rebuild the numerical summary from the retained traces:

```bash
python studies/lstm-fsm-competence/checks/behavior-20261108/analyze.py \
  --plan /path/to/check/plan.json --output /path/to/new-analysis
```

[run.py](run.py) provides preparation and individual-cell commands.
[case_diagnostics.py](case_diagnostics.py) and [mechanism_probe.py](mechanism_probe.py)
reconstruct host-level costs and observation/action sequences. Executable copies
are lint-formatted; the analysis record preserves the exact original bytes.
