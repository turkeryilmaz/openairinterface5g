# Effective configuration object (schema-bundle member 11)

This version-controlled slice freezes a fail-closed schema, validator, explicit
constructor, and canonical serializer for external object kind 10,
`metadata/effective-config.json`.  It binds the fully materialized profiler
CLI values, pre-ACTIVE role and scope, module-selection inputs, and (for A03)
the caller-supplied 64-bit seed and threshold.  No generated run instance is
stored here and the code performs no host discovery, random acquisition,
default selection, or evidence write.

`validate_module_selection_bindings()` evaluates the already frozen coverage
policy's `always`, `configuration_equals`, and `role_equals` predicates.  It
requires every referenced configuration key to exist and proves that each
build-domain row's run-time `configured` value equals the effective role and
selection result.  It complements rather than replaces the coverage module's
full build/run validators. The kind-10 digest must equal both the opening
`configuration_instance_sha256` and run coverage's same-named field. It is
deliberately independent of build coverage's immutable
`build_configuration_sha256`, because run ID, output directory, and the A03
trial seed are materialized separately for each run of one build.

The module imports the accepted canonical JSON codec from the sibling
`semantic` slice.  It does not carry a second JSON implementation.

The effective-configuration schema is registered as append-only object type 11
at `definition/effective-config-schema-v1.json`.  Every generated kind-10
object binds that exact member.  The schema definition is version 1.1 because
it binds event-semantics member 3 version 1.1; the external effective-
configuration instance remains version 1.0 and is byte-compatible.  The
bundle schema, generated bundle, and inventory advance separately to version
1.5. `BUNDLE_MEMBER_PROPOSAL`, `BUNDLE_ENTRY_PROPOSAL`, and
`BUNDLE_CROSS_RELATION_PROPOSAL` expose the exact registered rows.

Numeric values have no hidden defaults.  Artifact 9 explicitly leaves their
defaults pilot-derived, so callers must pass every value.  Synthetic unit-test
values demonstrate mechanics only and are not host observations or admissible
scientific evidence.  The A03 seed, threshold, deterministic selection
mapping, and literal vectors are bundle-bound. Scientific admission still
requires measured provenance and complete runtime, coverage, and event
evidence.

The schema freezes configuration meanings rather than only JSON widths:

- `flush_records` is the record-count ceiling for a normal writer payload
  chunk; a record-threshold chunk has exactly that count, while a timer or final
  chunk may be partial. `flush_us` is elapsed monotonic microseconds since the
  oldest staged record and zero disables only this time trigger.
- `ring_records` is record slots per registered producer ring.
  `max_threads` is the preallocated producer-descriptor capacity and is at most
  `UINT32_MAX-1`, leaving the all-ones value reserved.
- `table_entries` is the total process membership capacity across all shards,
  while `table_probes` is the maximum slots examined by one membership
  operation.
- In A03, `sample_threshold` is `q` in `U < q`, with
  `U = F(x) xor K` and inclusion probability `q / 2^64`; `q` is exactly
  `1..2^64-1`. Therefore A03 represents neither `p=0` nor exact `p=1`; its
  greatest representable probability is `1-2^-64`. The seed is exactly
  eight bytes rendered as lower-case `%016x`: the first hex byte is the most
  significant byte of the equivalent unsigned 64-bit integer. Provenance and
  status are closed pairs: measured Linux `getrandom()` is `1/1`; an explicit
  synthetic fixture is `2/2` and fails
  `validate_sample_seed_admissibility()`. Outside A03, seed is null, threshold
  is zero, and provenance/status are `20/20`.

`wire_object_binding_fields()` returns fields with the exact Artifact10 wire
names and types, including `object_flags` and the 32-byte binary SHA-256. Its
focused test round-trips the result through the independent container-wire
codec.

Focused check:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B tests/test_effective_config_v1.py -v
```
