# Schema-v1 concrete diagnostics catalog

This version-controlled slice implements the canonical `status/diagnostics.json`
instance grammar and its exact projection into Artifact 10 diagnostic-total
rows and fixed trailer aggregates.  It consumes the accepted semantic member-8
definition without redefining canonical JSON, reason IDs, flags, or mode scope
populations.

The validator requires explicit evidence-population inputs.  A missing required
counter scope or value is represented by `PARTIAL_COUNTER_POPULATION`; it is
never synthesized as zero.  Complete populations require exact rows, ID 96
reconciliation, and exact terminal projection.  The module performs no runtime
discovery, file writes, stream I/O, or admission decision.

Run the bounded deterministic suite without bytecode:

```text
env PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest -v \
  tools/profiling/memory/catalog_v1/diagnostics/tests/test_diagnostic_instance_v1.py
```

This schema code does not claim a measured diagnostic instance,
COMPLETE archive admission, runtime writer correctness, or profiler coverage.
