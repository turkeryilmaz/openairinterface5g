# Schema-v1 callsite catalog

This version-controlled slice implements the observed `catalog/callsite.json`
instance grammar already frozen by semantic definition member 5.  It validates
strict row identity, active process generation, module-generation/ID
resolution, canonical bytes, object-table metadata, and every event-record
callsite reference.  A05 requires nonzero resolved callsites; other frozen
modes require canonical zero and an empty observed catalog.

The archive descriptor is derived from one canonical validated byte snapshot;
its count, size, and digest are never supplied independently.  A COMPLETE A05
terminal requires equality between the distinct persisted record references
and catalog IDs.  FAILED/INCOMPLETE callers may explicitly request weaker
one-way resolution and must retain that limitation in terminal status.

The accepted static definition freezes uniqueness only by process generation
and callsite ID.  This validator therefore does not invent a second raw-address
injectivity rule; any future interning-key uniqueness must first be added to
the named definition artifact.

The validator does not discover modules, symbolize addresses, mutate an
archive, implement callsite interning, or claim completeness.  Its module keys
must come from a separately validated module catalog.

Run its short synthetic suite without bytecode:

```text
env PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest -v \
  tools/profiling/memory/catalog_v1/callsite/tests/test_callsite_catalog_v1.py
```
