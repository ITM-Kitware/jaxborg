# Blue behavior check

The CPU check started November 8 at 12:33 p.m. Eastern. See the
[launch record](launch-record.json) for the frozen plan and validation results.
ITM output:

```text
/home/local/KHQ/paul.elliott/jaxborg-exp/shared/studies/lstm-fsm-competence-20261107-v3/checks/behavior-20261108-3d97d4c4/
```

This exploratory check asks what Blue changes between **49.968M and 124.992M** training steps against FSM and the archived diverse-seed100 Red. The original competence evaluations retained returns and FSM CIA summaries, but no action traces; they cannot establish a behavioral explanation.

The frozen [protocol](protocol.json) selects training seed42, all four retained layouts200–203 and episode root21000: **16 episodes**, paired by checkpoint within each opponent. The root was chosen by index before examining its returns. This small panel can explain particular cases, not estimate the behavior of all seeds or opponents. Every archived Red is unseen during this Blue-only FSM training. Cross-opponent episodes use different RNG contracts.

[run.py](run.py) freezes exact checkpoint, sidecar, topology and canonical episode identities, then runs one recorded cell per invocation. Each traced return must equal its retained same-case return. A discrepancy is retained and blocks interpretation. Original training source remains `0aef148bbebb45818e71b20e9b5a52be031f93f1`; the tracing source is pinned separately in the new plan.

[analyze.py](analyze.py) reports genuine idle choices, forced busy Sleep, completed attempts, legal category opportunities, reachable host coverage, reward components and Green failure counts. Raw traces also retain policy observations, observed-host mappings, mission phase, route blocks and pre/post simulator state. Automatic monitoring runs every step. Completion and whole-step state changes do not prove an individual action succeeded; latent compromise is distinct from Blue's observed evidence. Green counts have no attempted-service denominator. The analyzer supports partial panels and records missing cells.

Prepare once in a fresh directory beneath the existing external study, then invoke each of the four cell IDs separately:

```bash
python studies/lstm-fsm-competence/checks/behavior-20261108/run.py prepare \
  --study-root /path/to/lstm-fsm-competence-20261107-v3 \
  --transfer-manifest /path/to/defender-transfer-20261107-v4/inputs/manifest.json \
  --output /path/to/lstm-fsm-competence-20261107-v3/checks/behavior-20261108 \
  --expected-revision FULL_TRACING_COMMIT

JAX_PLATFORMS=cpu python studies/lstm-fsm-competence/checks/behavior-20261108/run.py run-cell \
  --plan /path/to/checks/behavior-20261108/plan.json \
  --cell 49968000-fsm --expected-revision FULL_TRACING_COMMIT

python studies/lstm-fsm-competence/checks/behavior-20261108/analyze.py \
  --plan /path/to/checks/behavior-20261108/plan.json \
  --output /path/to/checks/behavior-20261108/analysis
```

The other cells are `124992000-fsm`, `49968000-red-diverse-100-final` and `124992000-red-diverse-100-final`. Outputs comprise a frozen plan/protocol, four cell receipts, 16 episode trajectories owned by their evaluations in the existing evidence store, and an analysis summary with its exact analyzer source. Existing records remain intact.
