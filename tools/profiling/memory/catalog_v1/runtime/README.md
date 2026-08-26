# Runtime catalog schema v1 (bundle member 13)

This version-controlled owner freezes the single schema-bundle definition shared
by external object kinds 5 (`catalog/thread.json`), 6
(`catalog/module.json`), and 7 (`catalog/clock.json`). Its canonical member is
`archive/definition/runtime-catalog-schema-v1.json`. The module is pure,
standard-library only, and performs no host discovery or archive writes.

Thread rows are retained registration history, not a final slot snapshot. A
COMPLETE population has contiguous registration ordinals from one, unique
never-reused indices in registration order, exact record resolution, and an
exact match between supplied records and the validated declared record count. A
FAILED/INCOMPLETE negative-partial population may retain unresolved record
indices explicitly; it can never become scientific admission. Diagnostic
producer scopes are derived from the validated thread population rather than a
separate caller list.

Module rows cover every observed loaded run-coverage row exactly once, including
`observed_unexpected` state 20 with a null build logical ID. V1 supports only
the base loader namespace, rejects duplicate build identities and zero hashes,
and hashes the exact canonical projection `{device,inode,loaded_path,
namespace_id,segments}`. Segments are nonempty, sorted, nonoverlapping, bounded,
and carry exact `/proc/maps`-style permission grammar. This intentionally does
not claim `dlmopen` namespaces or simultaneous multi-load support.

The clock catalog contains one opening-bound record-clock row. Schema v1 records
a producer-attested architectural source and rational counter rate, and
cross-checks architecture, the opening header, terminal repetition, acquisition
brackets, and arithmetic. At least two ordered counter/monotonic acquisition
brackets bind the opening and terminal anchors. The declared opening error bound
must cover every observed transform-to-bracket endpoint distance; the artifact
also freezes the maximum observed distance outside a bracket. V1 preserves no
raw CPUID.15 or CNTFRQ_EL0 witness and therefore cannot independently rederive
or calibrate the rate. It validates archived observations and arithmetic, not
unsampled hardware stability or independent metrology calibration.
Measured-affine admission is deliberately deferred.

The deterministic no-bytecode owner suite is:

```text
env PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest -v \
  tools/profiling/memory/catalog_v1/runtime/tests/test_runtime_catalog_v1.py
```

Registration of the definition does not by itself prove that a particular OAI
build emitted a valid instance. Runtime behavior, lifecycle integration, exact
build/run/configuration identity, and RF observations remain measured-evidence gates.
