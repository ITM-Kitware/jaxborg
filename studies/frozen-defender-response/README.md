# Frozen-defender response

Can freshly trained Red challengers improve their return against one frozen
Blue policy? The retained MAPPO study reports about **112 additional Red return**
for the validation-selected challenger over Original Red on **600 paired
held-out cases**, with a paired 95% interval of approximately **87 to 137**.

Three challengers trained against the same frozen Blue. Selection used separate
validation scores, with Original Red as a fallback; final test scores did not
select the winner. Both selected policies faced the same Blue and test cases.
This demonstrates a profitable response under the evaluated conditions.

| Opponent facing frozen Blue | Mean Blue return |
| --- | ---: |
| Original Red | -2837.945 |
| Selected challenger | -2949.953 |

Lower Blue return means higher Red return in this zero-sum comparison.
The uncertainty concerns evaluation cases conditional on these selected policies
from one original training run. It does not measure variation across independently
trained defenders or establish equilibrium. The single topology was reconstructed,
rather than retained from the original training execution.

The next research question is whether the response gain persists across
independently trained defenders and attacker searches.

## Evidence and reconstruction

Obtain the reference input bundle from the study author. It is held outside Git,
and **a shared download location has not yet been established**. The bundle
contains the protocol, selection, producer records, raw measurements, and
reference report. [Bundle identities and reconstruction commands](reproduce.md)
describe how to obtain the same tables and figures.

[analyze.py](analyze.py) reconstructs this historical report from saved
measurements using the project lockfile. It checks selection and recorded
policy identities without loading model tensors or rerunning policy evaluation.
Independent policy reevaluation additionally requires weights, sidecars, topology
bytes, and the producing execution environment. New studies should follow the
[study guide](../README.md) and define checks suited to their question.
