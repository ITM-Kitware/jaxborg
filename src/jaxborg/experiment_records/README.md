# Retained experiment records, version 1

`validate_records(records)` validates the canonical executable specifications in
`core.SCHEMAS` and their semantic constraints, then returns a detached ID index.
The specifications and validator together define the schema; there is no second
schema file to keep synchronized. JSON records use `schema_version: 1`, `kind`,
and a globally unique `id`. Store-qualified producer IDs belong in these IDs,
not readable run names. Unknown versions and top-level fields are rejected.

The four input kinds are:

| Kind | Required fields beyond the envelope |
| --- | --- |
| `protocol` | `study_id`, `revision`, `selection_rule`, `required_facts`, `trials`, `metric_definitions` |
| `attempt` | `protocol_id`, `trial_id`, `status`, `execution_kind`, `retry_of`, `facts`, `budgets`, `inputs` |
| `artifact` | `producer_id`, `uri`, `sha256`, `size_bytes`, `role` |
| `evaluation` | `attempt_id`, `inputs`, `status`, `facts`, `budgets`, `metrics`, `raw_artifacts` |

Each trial cell has `id`, `condition_id`, `replicate_id`, and nullable
`matched_unit_id`. Cells exist before attempts, so an unlaunched replicate remains
visible. Replicate and matched identities are protocol decisions, not seed/name
parsing. Attempts retain their trial through retries; `retry_of` is nullable and
must reference an attempt of the same trial without cycles. Imported artifacts
may have a null producer; do not invent a training run.

An attempt or evaluation status is `planned`, `running`, `succeeded`, `failed`,
or `cancelled`. Inputs and raw evidence reference artifact IDs. Artifact producer
references, when present, point to an attempt or evaluation. Artifact hashes use
lowercase SHA-256 and sizes use bytes. Validation checks metadata format and graph
consistency; it does not retrieve artifact URIs or verify their contents.

Facts are named values with provenance, for example
`{"seed": {"value": 42, "source": "manifest.json#/seed"}}`. An unavailable fact
is `{"value": null, "reason": "not retained", "source": "legacy manifest"}`. A budget has `unit`, `requested`,
and `completed`; counts are nonnegative numbers or unknown values in that same
null/reason/source form. Units must be explicit, such as `environment_steps` or
`episodes`. Different units are never converted implicitly.

Protocol metric definitions have `name`, `unit`, `direction` (`higher`, `lower`,
`neutral`), `aggregation`, `sampling_unit`, and `uncertainty`. Evaluation metrics
have `name`, numeric `value`, and matching `unit`. Keep episode SD and independent
training replication distinct in those definitions.

An optional `extensions` object maps named namespaces to objects containing
`schema_version` (positive integer), `source` (retained source reference), and
`values` (object). Its shape is validated; scientific semantics remain study-specific. Record custom mechanism source hashes,
explicit arguments, and schedule evidence as artifacts and sourced facts; a function
name alone cannot establish equivalence.

## Explicit selection

```python
from jaxborg.experiment_records import collate

result = collate(
    records,
    protocol_id="study:protocol:1",
    selected_attempts={"trial:control:1": "store:attempt:123"},
    selected_evaluations=["store:evaluation:456"],
    analysis={"source": "analysis source hash", "configuration": {}},
)
```

The protocol declares a revision, `selection_rule: "explicit"`, and
`required_facts: {"attempt": ["contract"], "evaluation": ["cohort"]}` (lists may
be empty). Selected records must contain known sourced values for every required
fact. A protocol can therefore make missing cohort or contract facts block selection.
These checks establish required presence; study-specific matching still determines
whether known values are compatible.

Unknown `protocol_id`, `trial_id`, `retry_of`, or evaluation `attempt_id` can use
the sourced null/reason fact object. Discovery retains these records without
inventing a scientific replicate. Known trials require a known matching protocol.
Unresolved membership stays in exclusions and cannot be selected. Unknown retry
history remains visible when a protocol explicitly selects one attempt for a
known trial. That choice does not establish the absence of earlier attempts and
does not add another replicate. If genealogy is required for a comparison, the
protocol must require a sourced retry-history fact. Null declares no retry
relationship and must not substitute for missing historical evidence.

Only explicitly selected successful attempts/evaluations are included. A selected
evaluation must belong to a selected attempt. Unselected attempts/evaluations stay
in `excluded`; unselected cells stay in `missing`, distinguishing no attempt from
no explicit selection. `rows` exposes trial/condition/replicate/matched identities
and selected record IDs for existing analysis consumers. The versioned `kind: "collation"` result retains an explicit selection and all
validated record snapshots, metric definitions, analysis settings, and a SHA-256
of canonical JSON before adding the digest. Its content digest supplies its immutable `id`. `validate_collation(result)`
recomputes the selection, rows, exclusions, and digest from the retained inputs
and rejects inconsistencies, even if an altered row received a recomputed digest.
Reordering the input records does not change this result. Changing the candidate set produces a new snapshot.

For file workflows:

```sh
python -m jaxborg.experiment_records records.json selection.json collation.json
```

The selection file contains the four keyword arguments from the Python example.
The CLI creates its output exclusively and refuses to overwrite retained evidence.
No selection comes from names, timestamps, or successful status alone.

Scientific matching remains the study's responsibility in this first slice.
Successful metadata validation does not establish comparable games, cohorts,
selection procedures, or budgets. Cross-study indexing must not pool outcomes.
Required known facts are enforced by collation; the study must still declare the
appropriate requirements and compatibility rules before analysis.
