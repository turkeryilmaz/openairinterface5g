# OAI memory-profiler schema-v1 semantic catalog

Status: version-controlled static semantic and runtime-schema definition closure

This versioned tree freezes the schema-v1 semantic definitions. `archive/` mirrors the
paths that static objects will have inside an evidence archive.  Every JSON
file there is canonical UTF-8 JSON: one object, no BOM or whitespace, ASCII
identifier keys in byte order, and exactly one final LF.

The frozen slice covers:

- canonical JSON and path rules;
- schema-bundle member IDs, archive paths, external-object paths, and static
  cross-hash relations;
- API IDs 1 through 12, event kinds, result kinds, every event flag and mask;
- all 57 malloc-family, aligned-allocation, release, and string-duplication transitions and mode-specific emission;
- exact A03 selection-mask composition and A04/A05 forbidden masks;
- realloc-zero-policy IDs and the complete relevant outcome table;
- exact context and phase schemas and process-generation-scoped resolution;
- the static callsite schema surface and a complete-capable diagnostic
  definition with all 14 reason IDs, exact class masks, counter scopes,
  A02--A05 populations, concrete/total row schemas, and terminal projections;
- append-only member 11, the config-owned effective-configuration schema at
  `definition/effective-config-schema-v1.json`, including its exact
  kind-10 instance-schema relation;
- append-only member 12, the sampling-owned byte-identical selection-rule
  definition at `definition/selection-rule-v1.json`;
- append-only member 13, the runtime-owned shared schema for thread, module,
  and clock catalog instances at `definition/runtime-catalog-schema-v1.json`.

The schema intentionally resolves an inconsistency in the preceding design
prose.  A03 emits a successful malloc/calloc only when
`SUCCESSOR_SELECTED` (bit 16) is set.  Thus A03 permits and requires bit 16 for
those records, while it forbids predecessor bits 14, 15, and 17.  A04/A05
continue to require all bits 14 through 17 to be zero.

`semantic_catalog_v1.py` is a standard-library-only independent canonical
parser, validator, bundle generator, and cross-binding checker.  It never
writes files.  Its `bundle` command emits candidate bytes to stdout; literal
publication remains a patch-based review action.

For defensive parsing, canonical input must be `bytes` no larger than
`CANONICAL_RAW_MAX_BYTES` (64 MiB), with no more than
`CANONICAL_JSON_MAX_NESTING_DEPTH` (64) open JSON containers. Before UTF-8
decoding, the parser's byte-level depth scan tracks quoted and escaped string
content, so brackets and braces inside a string do not contribute to nesting.
`load_canonical` uses `stat().st_size` only as an early preflight; the
in-memory parse repeats the length check, but callers remain responsible for
ensuring a source file is immutable while it is read.

Diagnostic member 8 incorporates the independently accepted design preserved
in `DIAGNOSTIC_REASON_PROPOSAL.md` at SHA-256
`3e2e2c3add69a04249f7d01ddec72e95f063885f11fdc830db7b1544003f3d8d`.
It closes the semantic reason/scope/projection gate that Artifact 10 left open;
it does not supersede Artifact 10's terminal layout, widths, flags, or object
binding rules.  The proposal remains historical design evidence; member 8 is
the canonical archive definition.

Member 11 preserves two distinct digest domains.  Coverage member 10 names
immutable build inputs `build_configuration_sha256`; it is never equated to
the per-run `metadata/effective-config.json` bytes.  The latter bind only to
the opening header, run coverage, object kind 10, and member 11 through the
frozen relations.  The config-owned validator also requires A03's active
threshold `q` to be in 1 through `UINT64_MAX`; other modes require zero.

Not frozen here:

- generated build-coverage or run-coverage instance rows;
- real Git/build/ELF/module/configuration/run/process identities;
- concrete runtime context, callsite, thread, module, clock, diagnostics, or
  status rows produced by OAI;
- final admission verdicts or hashes that depend on a real build or run;
- runtime behavior or integration correctness without measured build/run evidence.

Static coverage-policy and coverage-instance-schema members are owned by the
adjacent `coverage/` slice and are validated through its
independent exact-literal validator.  Effective-config member 11 is owned by
the adjacent `config/` slice and is likewise accepted only
when its member, entry, relation, canonical bytes, and exact definition all
match that owner.  Selection-rule member 12 is owner-validated through the
adjacent `sampling/` slice. Runtime member 13 is owner-validated through the
adjacent `runtime/` slice, including exact member/entry/relation proposals and
canonical bytes. The inventory pins all thirteen registered static members,
which can produce and validate one exact in-memory v1.5 candidate
`catalog/schema-bundle.json`.

The version split is intentional: member 2, the generated bundle, and the
inventory are v1.5; event-semantics member 3, API member 4, and config-schema
member 11 are v1.1; coverage-policy member 9 is v2.0. Members 1, 5--8, 10,
12, and 13 remain v1.0, as do external effective-configuration and generated
build/run instances. Member 9 v2.0 promotes deferred origins 105--108 into the
12-API supported denominator and freezes architecture-exact symbol versions;
its former classification-10 IDs are not silently reused.

The earlier ten-member candidate (1,726 bytes, SHA-256
`501eca2a2af21f338b665f94deac19df39865f293540d31429c2080d385363b2`)
is retained only as superseded historical evidence.  The selection mapping is registered as member 12. Runtime instance schemas
are registered as member 13, while no concrete producer-emitted instance is
claimed; the callsite instance grammar remains supplied by member 5.
The semantic files and their hash inventory remain a deterministic static
freeze, not evidence that any real stream, build, run, configuration, or
coverage instance is semantically admissible.  The adjacent executable
`event_classifier_v1.py` derives all 57 transitions, operand profiles,
mode-specific selection cases, callsite rules, and exact composed masks from
decoded evidence.  Its A03 replay independently reconciles persisted selected
successor identities and predecessor endpoints, including final live-address
uniqueness.  Exact A04/A05 per-thread sequence gaps remain structurally retained
completeness failures rather than corrupt-container errors; the verifier reports
the missing population and keeps scientific admission false.  A03 gaps are
expected sampling and are not counted.  The classifier does not infer
unpersisted transactions or claim A04/A05 offline heap reconstruction.

Runtime member 13 closes the former schema ambiguity without claiming hardware
or producer behavior. Thread rows are retained registration history with exact
ordinals and never-reused indices. Module rows reconcile every observed loaded
run row one-to-one, include exact sorted mapping segments, and bind the
restricted base-namespace map projection digest. Clock schema v1 admits only
architecture-derived exact rates (x86 CPUID.15 or AArch64 CNTFRQ_EL0) and
requires ordered opening/final acquisition brackets whose observed error stays
within the opening header's predeclared bound. These checks establish artifact
consistency at sampled anchors; they do not prove behavior between samples or
independent calibration accuracy.

The producer's boundary-straddling bit remains authoritative when optional
counters are unavailable, but a set boundary bit is rejected if the terminal
seal bracket was never established.  A valid exit strictly before
cutoff-before or strictly after cutoff-after proves which side of the seal
completed; equality or an exit inside the exchange bracket remains
producer-resolved.

Run the bounded check without bytecode output:

```text
env -i HOME=/tmp LANG=C LC_ALL=C PATH=/usr/bin PYTHONDONTWRITEBYTECODE=1 \
  PYTHONHASHSEED=0 TZ=UTC /usr/bin/python3 -B \
  tests/test_semantic_catalog_v1.py -v

env -i HOME=/tmp LANG=C LC_ALL=C PATH=/usr/bin PYTHONDONTWRITEBYTECODE=1 \
  PYTHONHASHSEED=0 TZ=UTC /usr/bin/python3 -B \
  tests/test_event_classifier_v1.py -v
```
