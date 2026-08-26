# Schema-v1 status-chain catalog

This version-controlled slice defines and validates the acyclic relationship
between `status/pre-footer-status.json`,
`status/post-close-verification.json`, and the archive manifest described by
Artifact 10. It performs no file I/O, stream decoding, catalog or ELF
validation, scientific admission, or archive publication.

`status_chain_v1.py` supplies strict canonical-JSON parsing, exact top-level
schemas, terminal outcome/boundary validation, manifest validation, and a pure
cross-artifact COMPLETE-candidate binder. Negative terminals remain valid
negative evidence but cannot be promoted. Missing evidence is rejected rather
than represented as zero. Canonical member names match exactly
`[a-z][a-z0-9_]*`.

The parser is deliberately bounded at this trust boundary. Schema v1 admits at
most 16 KiB of raw canonical bytes for a pre-footer status, 16 KiB for a
post-close receipt, and 64 MiB for the archive manifest; the binder applies the
appropriate population-specific ceiling before UTF-8 decoding or `json.loads`.
The manifest is additionally limited to 8192 entries, which is substantially
larger than a normal archive while keeping validation work bounded. JSON
containers may nest to depth 64. A byte-level scan tracks only braces and
brackets outside strings, honoring backslash escapes, before decoding; recursive
walking and serialization use the same depth ceiling, with a typed
`RecursionError` backstop. The generic canonical parser uses the manifest
ceiling because it has no artifact-kind argument.

The binder accepts a frozen `VerifiedStreamIdentity` produced by the separate
accepted exact-EOF decoder. The identity carries the stream path, physical byte
count, whole-stream SHA-256, all four footer-domain SHA-256 values, and the
accepted verifier-definition SHA-256. It also carries the exact byte count and
SHA-256 from the stream's decoded object-kind-12 binding. Those two values must
equal the supplied pre-footer status bytes, so a recomputed manifest cannot pair
a verified stream with a different process generation or terminal outcome.
Every receipt identity field must equal the trusted stream identity. The stream
path must differ from the pre-footer status, post-close receipt, and manifest
paths, and the manifest must therefore bind three distinct pre-footer, receipt,
and stream nodes.

Run the deterministic standard-library tests with:

```text
env PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 TZ=UTC python3 -B -m unittest -v test_status_chain_v1.py
```

Constructing `VerifiedStreamIdentity` is a trust-boundary operation: the caller
must already have checked the stream, exact EOF, footer, hash domains, tables,
external-object bytes, and exact accepted verifier definition. This slice does
not perform those checks and cannot turn structurally unchecked input into
verified scientific evidence.
