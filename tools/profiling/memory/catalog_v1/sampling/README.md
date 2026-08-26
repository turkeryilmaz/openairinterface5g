# Schema-v1 sampled-selection candidate

This version-controlled slice closes the deterministic mapping left open by the
measurement contract's A03 equation.  It defines the exact instance-key byte
encoding, a bounded portable 64-bit mixer, the seed interpretation, and the
strict-threshold decision.  Schema-bundle v1.1 registers its byte-identical
definition as append-only member 12.  That registration does not authorize
A03 archive admission or runtime activation.

The owner module exposes the exact member-12 proposal and validates the
canonical literal at `archive/definition/selection-rule-v1.json`.  The
definition itself remains version 1.0; member 2 and the generated twelve-member
schema definition are version 1.1. The current authoritative generated bundle
has the members and version declared by `semantic/semantic_catalog_v1.py`'s
`MEMBER_REGISTRY` and `BUNDLE_VERSION` (currently thirteen members and v1.5).
The canonical definition includes the complete ordered mapping and `mix64`
operation sequences, rather than relying on its Python name or finite vectors to
define the algorithm.

Seed text follows the effective-configuration member exactly: lowercase
`%016x` numeric big-endian hexadecimal, with the first hex byte representing
the most-significant byte of the equivalent u64.  Statuses 1, 2, and 20 mean
measured exact-eight-byte `getrandom`, explicit synthetic fixture, and
non-A03 not-applicable respectively.

The mapping is deliberately non-cryptographic.  Its scientific requirement is
equal first-order inclusion probability under an independently uniform
64-bit seed, not mutually independent selections or collision resistance.
Collisions are possible and are preserved as a stated limitation.  Exact
synthetic calibration across many independently acquired seeds remains a later
admission gate.

`selection_rule_v1.py` is the accepted static definition and pure oracle.
`tests/test_selection_rule_v1.py` uses frozen key/mapping/decision literals and
adversarial type, range, provenance, and threshold boundaries. All source stays
under this versioned catalog root.
