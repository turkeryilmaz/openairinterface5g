# Coverage policy and instance schemas

This version-controlled layer owns schema-bundle members 9 and 10:

- `archive/definition/coverage-policy-v1.json` (`object_type=9`)
- `archive/definition/coverage-instance-schema-v1.json` (`object_type=10`)

Member 9 is coverage-policy definition version 2.0. Member 10 and all generated
build/run instances remain version 1.0. Both files are exact canonical JSON
literals. `coverage_catalog_v1.py` restates their values independently, validates
their literal bytes, and validates
caller-supplied generated `catalog/build-coverage.json` and
`catalog/run-coverage.json` instances.

The policy freezes Linux GNU ELF64 little-endian x86-64/AArch64 ABI oracles,
the API-ID 1..12 symbol/signature/wrapper/real-symbol set, required final-link
identity/import/export/relocation/wrap/dependency/runtime evidence, sparse
classification/admission/load/evidence/verdict/failure IDs, realloc-zero
oracle binding, path domains, module-selection grammar, and fail-closed rules.
The realloc oracle binding is exactly object type 3 at
`definition/event-semantics-v1.json`, hashed over its canonical member bytes
including the sole final LF. Each realloc-admitting logical-ELF row must equal
that frozen digest; its separately measured `realloc_zero_policy_id` records
which admitted host behavior occurred, so the static binding invents no host
result. Policy v1.1 appended the fail-closed rule
`active_realloc_importers_share_one_policy`, and v1.2 froze architecture-exact
known-unsupported symbol versions. Policy v2.0 promotes `reallocarray`,
`aligned_alloc`, `posix_memalign`, `memalign`, `valloc`, `pvalloc`, `strdup`,
and `strndup` into the supported first-slice denominator. The former deferred
origin IDs 105--108 are removed rather than reinterpreted; only IDs 101--104
remain classification 10. Run validation derives the exact realloc
policy/oracle pairs from validated, admitted API-3 or API-5 rows that are
configured, observed, identity-valid, and in load state 1. More than one pair
requires failure 22 and an ineligible verdict; inactive and active non-realloc
rows do not contribute.

The public resolver accepts no caller policy or oracle. It snapshots and fully
validates the build and run mappings, requires measured eligible evidence, and
returns an immutable `resolved` pair or `not_applicable`. It rejects mixed-policy
retained negatives, other ineligible runs, and synthetic fixtures. The semantic
oracle in every resolved result is the exact member-3 digest above.

The instance schema keeps a module's `admission_state_id` separate from its
sorted per-origin `classifications`. Run instances use `module_population`,
the exact expected/observed union, with explicit load state and nullable load
identity only when not observed.

Build identity records the digest of the immutable build-time configuration as
`build_configuration_sha256`. It is not the digest of kind 10: run coverage's
`configuration_instance_sha256` binds the exact per-run effective
configuration, including run ID, output directory, and any A03 trial seed.
Multiple distinct run configurations may therefore bind the same immutable
build coverage. Exact-domain eligibility is derived from the validated build,
the run population and failures; no build/run configuration-digest equality is
part of that derivation. Failure 34 remains reserved for a concrete configured
module population mismatch.

The run validator requires the expected configuration-instance digest supplied
from the opening header/kind-10 equality gate. A run digest that differs from
that value requires failure 34 and an ineligible verdict; it is never compared
with `build_configuration_sha256`. Thus the same build coverage can be reused
by distinct trial seeds and other per-run effective configurations without
weakening the opening/kind10/run identity check.

No generated build or run instance is shipped here. Build-evidence source and
instances advance independently to version 1.1 so the final ELF parser consumes
the policy-owned architecture oracle instead of assuming the ABI baseline for
every symbol. The coverage-policy member is propagated by the atomic schema
bundle v1.5 checkpoint; member 10 and generated build/run instances remain byte-
identical v1.0. The unit tests
construct only in-memory fixtures. Fixtures are explicitly labelled
`evidence_origin_id=2`, require failure 90, and cannot validate as a final or
eligible result. A real final instance must be produced from the exact final
link/runtime census, use `evidence_origin_id=1`, and satisfy every derived
identity, population, failure, and verdict relation; reconnaissance rows are
never admission input.

Run the focused checks without bytecode output:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B tests/test_coverage_catalog_v1.py -v
```

The module is standard-library only, performs no discovery and no writes, and
does not fabricate missing evidence. Callers remain responsible for acquiring
and canonically serializing the final-link and pre-ACTIVE runtime observations
that the validators consume.
