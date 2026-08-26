# Schema-v1 diagnostic reason proposal — independent review required

Status: proposed static catalog only; not frozen and not COMPLETE-capable until
independently reviewed, encoded canonically, and incorporated into semantic
member 8 with updated dependent hashes.

The reason population below maps one-to-one, in the same order, from the
authoritative counter list in Measurement Contract section 6.5. Numeric IDs are
sparse by diagnostic class so future append-only additions do not renumber an
existing meaning. Every counter instance is bound to one nonzero process
generation and uses the exact counter-scope grammar and projection below.

| ID | Name | Exact class flags | Required claim gate |
|---:|---|---:|---|
| 1 | `ring_full` | `0x0101` EVENT_LOSS + EXACT_EVENT_GATE | exact event |
| 2 | `unregistered_active_thread` | `0x0101` EVENT_LOSS + EXACT_EVENT_GATE | exact event |
| 3 | `registration_capacity_failure` | `0x0101` EVENT_LOSS + EXACT_EVENT_GATE | exact event |
| 16 | `recursion_bypass` | `0x0102` UNACCOUNTED_BYPASS + EXACT_EVENT_GATE | exact event |
| 17 | `profiler_internal_bypass` | `0x0004` INTENTIONAL_OUTSIDE_DOMAIN | domain disclosure |
| 18 | `unsupported_api_or_domain` | `0x0004` INTENTIONAL_OUTSIDE_DOMAIN | denominator/domain |
| 32 | `size_unknown` | `0x0008` UNKNOWN_VALUE | requested-byte estimands |
| 48 | `sample_membership_insertion_failure` | `0x0210` SAMPLED_MEMBERSHIP + SAMPLED_GATE | sampled lifetime |
| 49 | `membership_lookup_failure` | `0x0210` SAMPLED_MEMBERSHIP + SAMPLED_GATE | sampled lifetime |
| 50 | `bounded_probe_exhaustion` | `0x0210` SAMPLED_MEMBERSHIP + SAMPLED_GATE | sampled lifetime |
| 51 | `invalid_or_ambiguous_pointer_pairing` | `0x0210` SAMPLED_MEMBERSHIP + SAMPLED_GATE | sampled lifetime |
| 64 | `clock_regression_or_invalid_counter` | `0x0140` COUNTER_OR_CLOCK_INVALID + EXACT_EVENT_GATE | timing/lifetime |
| 80 | `writer_io_or_finalization_failure` | `0x0120` WRITER_OR_FINALIZATION + EXACT_EVENT_GATE | stream completeness |
| 96 | `diagnostic_counter_saturation` | `0x0180` SATURATION + EXACT_EVENT_GATE | affected exact total |

## Exact class and claim-gate meaning

Class flags describe the scientific consequence, not merely the component that
increments a counter. A reason may therefore carry one base class and a claim-
gate class. `EXACT_EVENT_GATE` means the reason must be exactly zero for an
emitted-stream exactness claim in A03, A04, or A05. It does not claim that A03
captures the full admitted-event population. `SAMPLED_GATE` marks the additional
A03 selected-membership and paired-lifetime conjunction. Thus A03 requires both
its applicable `EXACT_EVENT_GATE` reasons and IDs 48--51 to be zero; A04/A05
never reinterpret an A03 membership reason as an exact-mode producer counter.

`profiler_internal_bypass` and `unsupported_api_or_domain` are not asserted to
be zero. Their complete, catalog-reconciled totals restrict and disclose the
domain. `size_unknown` restricts requested-byte and size-dependent estimands
without pretending event loss.

ID 80 retains `WRITER_OR_FINALIZATION | EXACT_EVENT_GATE` (`0x0120`). A writer
I/O/finalization failure also determines the applicable negative lifecycle and
cannot support an exact emitted-stream claim. ID 96 retains
`SATURATION | EXACT_EVENT_GATE` (`0x0180`) because Measurement Contract section
10.1 explicitly requires diagnostic-counter saturation to be zero. The existing
class bits express both consequences; no new archive-completeness bit is needed.

## Counter scopes and required populations

This proposal introduces a diagnostic-only `counter_scope_kind`; it is distinct
from the opening header's measurement `scope_kind`.

`process_generation` is nonzero `u64`, `counter_scope_kind` is `u16`, and
`counter_scope_id` is `u32`. Values or scope-ID combinations not listed below
are invalid in schema v1.

| Value | Name | Exact `counter_scope_id` meaning |
|---:|---|---|
| 1 | `producer_thread` | the stable nonzero `thread_index` of one final READY producer slot in this generation |
| 2 | `registration` | exactly 1, the singleton bounded registration/admission counter set |
| 3 | `writer` | exactly 1, the singleton payload-writer/finalizer counter set |
| 4 | `diagnostic_aggregate` | exactly 1, the singleton saturation-transition counter set |

Let `P` be the exact set of final READY producer slots below the reservation
high-water snapshot that belong to the active generation. A not-yet-READY
reservation is not in `P`; no thread identity is fabricated for it. Let `R`,
`W`, and `G` denote the singleton scopes 2, 3, and 4 above. The required
contributing-instance population is:

| Reason IDs | A02 | A03 | A04 | A05 |
|---|---|---|---|---|
| 1 | not applicable | one per `P` | one per `P` | one per `P` |
| 2, 3 | `R` | `R` | `R` | `R` |
| 16, 17, 18, 32 | one per `P` | one per `P` | one per `P` | one per `P` |
| 48, 49, 50, 51 | not applicable | one per `P` | not applicable | not applicable |
| 64 | not applicable | one per `P` | one per `P` | one per `P` |
| 80 | `W` | `W` | `W` | `W` |
| 96 | `G` | `G` | `G` | `G` |

A00/A01 perform no profiler diagnostic mutation and produce no diagnostic
instance. A06 remains outside the schema-v1 first slice. This proposal does not
make any of those modes COMPLETE-capable. A reason marked not applicable has a
schema-known empty contributing set, not an unavailable population; its explicit
reason total is canonical zero. If `P` is known to be empty, its per-producer
set is likewise an exact empty set. Failure to establish `P`, `R`, `W`, or `G`,
or a missing required row/value within a nonempty set, is
`PARTIAL_COUNTER_POPULATION`, never an implicit zero.

Concrete rows have the unique key
`(process_generation, counter_scope_kind, counter_scope_id, reason_id)` and sort
strictly by that tuple. For the active mode, the diagnostics object contains
exactly the contributing rows selected by the table: no missing, duplicate, or
extra scope/reason pair is permitted. Each row carries its stored `u64` value
and its counter-saturated state.

For each of the 14 reasons, including those with an exact empty contributing
set, the canonical diagnostics object also contains exactly one derived reason-
total row in increasing `reason_id` order. A derived total is not another
contributing counter instance and is never added to its leaves. Its projection
to the Artifact 10 section 6.3 terminal entry is exact:

- `saturating_total` is the mathematical sum of available stored contributing
  values, clamped to `UINT64_MAX`;
- `nonzero_counter_instances` is the number of contributing values greater
  than zero;
- `saturated_counter_instances` is the number of contributing counters whose
  saturation state is set;
- `TOTAL_SATURATED` is set iff a contributing counter saturated or the
  mathematical sum of available unsaturated values exceeded `UINT64_MAX`;
- `PARTIAL_COUNTER_POPULATION` is set iff a required instance or value is
  unavailable, independently of saturation.

Every terminal reason entry must equal its derived object row. The fixed loss
and bypass aggregates remain the saturating sums of complete reason totals with
class bit 0 and bit 1 respectively. They count reason occurrences, not inferred
unique transactions. No process aggregate is stored as a contributing row, so
per-thread/per-writer leaves cannot be double-counted.

The diagnostics object's concrete `entries` array contains only the contributing
counter-instance rows, and the object-table `entry_count` equals its length. The
14 derived `reason_totals` form a separate required array with the exact
terminal fields `reason_id`, `class_flags`, `summary_flags`,
`saturating_total`, `nonzero_counter_instances`, and
`saturated_counter_instances`; it is not included in object-table
`entry_count`.

## Exact reason predicates

IDs 2 and 3 remain distinct authoritative populations and use the registration
singleton because a failed registration has no valid `thread_index`:

- ID 3 increments once when an ACTIVE supported-call capture attempt takes the
  bounded first-use registration branch and that reservation fails specifically
  because the preallocated registration capacity is exhausted. The real API is
  still called, but no event can be emitted.
- ID 2 increments once when an ACTIVE supported-call capture attempt cannot be
  associated with a READY producer slot for any other registration/invariant
  cause. It is not the ordinary first use that successfully registers.
- The primary causal branches are mutually exclusive for one call. A
  capacity-failure call increments ID 3 and does not also increment ID 2.

IDs 17 and 18 are fail-closed classifications, not escape hatches:

- ID 17 is permitted only for an exact static-catalog entry that declares the
  operation an intentional profiler-internal exclusion.
- ID 18 is permitted only for an exact coverage-policy origin classified as a
  known unsupported API/domain population.
- A new or unknown API, origin, alias, module, or domain, and any unclassified
  bypass, fails coverage/admission. It cannot be converted to ID 17 or 18 to
  preserve eligibility.

IDs 48--51 exist only in A03 and map the section 10.1 paired-lifetime
conjunction without overlap at their primary failure decision:

- ID 48 is failure to commit a selected-live-instance insertion, including
  membership-table capacity exhaustion after a conclusive bounded search.
- ID 49 is a conclusive bounded lookup that cannot resolve a selected
  predecessor which the archived transition requires.
- ID 50 is exhaustion of the bounded collision/probe budget before insertion
  or lookup can reach a conclusive result.
- ID 51 is a completed lookup whose selected pointer/instance transition state
  is invalid or genuinely ambiguous under the frozen realloc/free rules.

This maps insertion and capacity to ID 48, lookup loss to ID 49, bounded
collision resolution to ID 50, and transition/pairing loss to ID 51. Offline
A04/A05 reconstruction never increments ID 51; an invalid exact-mode pairing is
a semantic-validation failure and makes the corresponding analysis claim
ineligible through that gate.

ID 96 increments once for the first unsaturated-to-saturated transition of each
underlying authoritative diagnostic counter instance in the generation. It
does not increment repeatedly while that counter remains saturated, and the ID
96 singleton excludes itself from its source population. Saturation of the ID
96 counter is represented by its own counter-saturated state and the ordinary
`TOTAL_SATURATED` projection; it never recursively increments itself. Aggregate
sum overflow without an input-counter transition sets `TOTAL_SATURATED` for the
affected reason but does not increment ID 96.

Let `T` be the mathematical number of first saturation transitions excluding
the ID 96 singleton itself. When every required contributing instance and value
is available, the exact reconciliation is
`terminal.saturated_counter_instances = T`, and the ID 96 stored value is also
exactly `T`. Schema v1 requires the ID 96 singleton's own saturation state to be
clear: the bounded v1 population contains fewer than `10 * UINT32_MAX`
per-producer counters plus the fixed singletons, so its once-per-other-instance
transition count cannot reach `UINT64_MAX`. A nonzero ID 96 self-saturation
state is corrupt, unadmitted evidence rather than a valid saturated singleton.

Under `PARTIAL_COUNTER_POPULATION`, no equality between ID 96 and the terminal
aggregate is asserted. ID 96's available stored value and the terminal
aggregate are independent lower bounds. The sole exact terminal relation then
remains the Artifact 10 relation
`terminal.saturated_counter_instances =
sum(reason_totals[*].saturated_counter_instances)` over the available derived
reason rows; an unavailable leaf cannot be reconstructed from ID 96.

No source, runtime, ring, writer, archive, or generated instance may consume
these IDs until the proposal is accepted and member 8 is regenerated with new
canonical bytes and SHA-256.
