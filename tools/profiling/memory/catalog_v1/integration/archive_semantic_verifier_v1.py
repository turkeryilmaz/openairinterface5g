#!/usr/bin/env python3
"""Composed exact-archive verifier for the schema-v1 contracts.

This version-controlled, standard-library-only module is intentionally a
verifier, not a writer. It composes the independently defined wire, semantic, coverage,
configuration, sampled-selection, callsite, bundle-registered runtime-catalog,
diagnostic, and status-chain modules.  It never discovers host state, repairs
evidence, substitutes defaults, or promotes a negative terminal outcome.

The thread/module/clock schemas are frozen as bundle member 13 and their
instances are validated and reconciled. Complete scientific admission remains
contingent on validated measured artifacts and is never inferred from static definitions.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
CATALOG_ROOT = HERE.parent
TOOLS_ROOT = CATALOG_ROOT.parent
REPOSITORY_ROOT = CATALOG_ROOT.parents[3]


class ArchiveVerificationError(ValueError):
    """Deterministic rejection with the composed validation stage attached."""

    def __init__(self, stage: str, detail: str):
        super().__init__(f"{stage}: {detail}")
        self.stage = stage
        self.detail = detail


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module specification unavailable: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


WIRE = _load(
    "_oai_memprof_integration_wire_v1",
    TOOLS_ROOT / "oai_memprof_container_wire.py",
)
SEMANTIC = _load(
    "_oai_memprof_integration_semantic_v1",
    CATALOG_ROOT / "semantic/semantic_catalog_v1.py",
)
EVENT_CLASSIFIER = _load(
    "_oai_memprof_integration_event_classifier_v1",
    CATALOG_ROOT / "semantic/event_classifier_v1.py",
)
COVERAGE = _load(
    "_oai_memprof_integration_coverage_v1",
    CATALOG_ROOT / "coverage/coverage_catalog_v1.py",
)
CONFIG = _load(
    "_oai_memprof_integration_config_v1",
    CATALOG_ROOT / "config/effective_config_v1.py",
)
SELECTION = _load(
    "_oai_memprof_integration_selection_v1",
    CATALOG_ROOT / "sampling/selection_rule_v1.py",
)
CALLSITE = _load(
    "_oai_memprof_integration_callsite_v1",
    CATALOG_ROOT / "callsite/callsite_catalog_v1.py",
)
RUNTIME = _load(
    "_oai_memprof_integration_runtime_v1",
    CATALOG_ROOT / "runtime/runtime_catalog_v1.py",
)
DIAGNOSTICS = _load(
    "_oai_memprof_integration_diagnostics_v1",
    CATALOG_ROOT / "diagnostics/diagnostic_instance_v1.py",
)
STATUS = _load(
    "_oai_memprof_integration_status_v1",
    CATALOG_ROOT / "status/status_chain_v1.py",
)
_HANDOFF_DECODER_PATH = (
    REPOSITORY_ROOT / "tools/profiling/memory/oai_memprof_process_handoff.py"
).resolve()
HANDOFF = _load(
    "_oai_memprof_integration_process_handoff_v1", _HANDOFF_DECODER_PATH
)
_BUILD_EVIDENCE_PATH = (
    REPOSITORY_ROOT / "tools/profiling/memory/oai_memprof_build_evidence.py"
).resolve()
BUILD_EVIDENCE = _load(
    "_oai_memprof_integration_build_evidence_v1", _BUILD_EVIDENCE_PATH
)

ACCEPTED_VERIFIER_DEFINITION_PATH = "definition/archive-semantic-verifier-v1.py"
ACCEPTED_PROCESS_HANDOFF_PATH = "streams/process-handoff.bin"
ACCEPTED_PRODUCER_DEFINITION_PATH = (
    "definition/oai-memprof-archive-composer-v1.py"
)
ACCEPTED_HANDOFF_DECODER_DEFINITION_PATH = (
    "definition/oai-memprof-process-handoff-v1.py"
)
ACCEPTED_BUILD_EVIDENCE_DEFINITION_PATH = (
    "definition/oai-memprof-build-evidence-v1.py"
)
ACCEPTED_EVENT_CLASSIFIER_SHA256 = (
    "57c5588bf78ae1729b00bcf480096921ec4672d498a7e2a510016e53ab1939c1"
)
_ACCEPTED_VERIFIER_DEFINITION_BYTES = Path(__file__).resolve().read_bytes()
_EVENT_CLASSIFIER_PATH = (
    CATALOG_ROOT / "semantic/event_classifier_v1.py"
).resolve()
if Path(EVENT_CLASSIFIER.__file__).resolve() != _EVENT_CLASSIFIER_PATH:
    raise RuntimeError("event classifier loaded from an unexpected path")
_EVENT_CLASSIFIER_DEFINITION_BYTES = _EVENT_CLASSIFIER_PATH.read_bytes()
if hashlib.sha256(_EVENT_CLASSIFIER_DEFINITION_BYTES).hexdigest() != (
    ACCEPTED_EVENT_CLASSIFIER_SHA256
):
    raise RuntimeError("event classifier source differs from the accepted digest")

if Path(HANDOFF.__file__).resolve() != _HANDOFF_DECODER_PATH:
    raise RuntimeError("process-handoff decoder loaded from an unexpected path")
_HANDOFF_DECODER_DEFINITION_BYTES = _HANDOFF_DECODER_PATH.read_bytes()
if Path(BUILD_EVIDENCE.__file__).resolve() != _BUILD_EVIDENCE_PATH:
    raise RuntimeError("build-evidence module loaded from an unexpected path")
_BUILD_EVIDENCE_DEFINITION_BYTES = _BUILD_EVIDENCE_PATH.read_bytes()
_PRODUCER_PATH = (
    REPOSITORY_ROOT / "tools/profiling/memory/oai_memprof_archive_composer.py"
).resolve()
_PRODUCER_DEFINITION_BYTES = _PRODUCER_PATH.read_bytes()

TRUSTED_RELEASE_AUTHORITY_PATH = "definition/trusted-release-authority-v1.json"
TRUSTED_RELEASE_AUTHORITY_SCHEMA = "oai_memprof_trusted_release_authority_v1"
# These are code-definition paths only. The authority intentionally excludes
# itself, manifests, generated config/opening/handoff objects, tests, caches,
# README material, and the legacy R0 surface, so its digest is non-circular.
TRUSTED_RELEASE_SOURCE_PATHS = {
    "definition/oai-memprof-container-wire-v1.py": "tools/profiling/memory/oai_memprof_container_wire.py",
    "definition/catalog-v1/semantic-catalog-v1.py": "tools/profiling/memory/catalog_v1/semantic/semantic_catalog_v1.py",
    "definition/catalog-v1/event-classifier-v1.py": "tools/profiling/memory/catalog_v1/semantic/event_classifier_v1.py",
    "definition/catalog-v1/coverage-catalog-v1.py": "tools/profiling/memory/catalog_v1/coverage/coverage_catalog_v1.py",
    "definition/catalog-v1/effective-config-v1.py": "tools/profiling/memory/catalog_v1/config/effective_config_v1.py",
    "definition/catalog-v1/selection-rule-v1.py": "tools/profiling/memory/catalog_v1/sampling/selection_rule_v1.py",
    "definition/catalog-v1/callsite-catalog-v1.py": "tools/profiling/memory/catalog_v1/callsite/callsite_catalog_v1.py",
    "definition/catalog-v1/runtime-catalog-v1.py": "tools/profiling/memory/catalog_v1/runtime/runtime_catalog_v1.py",
    "definition/catalog-v1/diagnostic-instance-v1.py": "tools/profiling/memory/catalog_v1/diagnostics/diagnostic_instance_v1.py",
    "definition/catalog-v1/status-chain-v1.py": "tools/profiling/memory/catalog_v1/status/status_chain_v1.py",
    "definition/oai-memprof-process-handoff-v1.py": "tools/profiling/memory/oai_memprof_process_handoff.py",
    "definition/oai-memprof-build-evidence-v1.py": "tools/profiling/memory/oai_memprof_build_evidence.py",
    "definition/oai-memprof-archive-composer-v1.py": "tools/profiling/memory/oai_memprof_archive_composer.py",
    "definition/archive-semantic-verifier-v1.py": "tools/profiling/memory/catalog_v1/integration/archive_semantic_verifier_v1.py",
    "definition/oai-memprof-softmodem-launcher-v1.py": "tools/profiling/memory/oai_memprof_softmodem_launcher.py",
}
if len(TRUSTED_RELEASE_SOURCE_PATHS) != 15:
    raise RuntimeError("trusted-release source population must contain exactly fifteen paths")

_TRUSTED_RELEASE_SOURCE_SNAPSHOTS: dict[str, bytes] = {}
for _trusted_path, _repository_relative_path in TRUSTED_RELEASE_SOURCE_PATHS.items():
    _source_path = (REPOSITORY_ROOT / _repository_relative_path).resolve()
    if not _source_path.is_file():
        raise RuntimeError(f"trusted-release source unavailable: {_repository_relative_path}")
    # This import-time fixed repository snapshot identifies the executing source
    # population. Runtime authority inputs are separately frozen from an explicit
    # controller-provided root; imported module paths are never trust inputs.
    _TRUSTED_RELEASE_SOURCE_SNAPSHOTS[_trusted_path] = _source_path.read_bytes()
if _TRUSTED_RELEASE_SOURCE_SNAPSHOTS[ACCEPTED_VERIFIER_DEFINITION_PATH] != _ACCEPTED_VERIFIER_DEFINITION_BYTES:
    raise RuntimeError("trusted-release verifier snapshot differs from accepted verifier snapshot")
if _TRUSTED_RELEASE_SOURCE_SNAPSHOTS[ACCEPTED_PRODUCER_DEFINITION_PATH] != _PRODUCER_DEFINITION_BYTES:
    raise RuntimeError("trusted-release producer snapshot differs from accepted producer snapshot")
if _TRUSTED_RELEASE_SOURCE_SNAPSHOTS[ACCEPTED_HANDOFF_DECODER_DEFINITION_PATH] != _HANDOFF_DECODER_DEFINITION_BYTES:
    raise RuntimeError("trusted-release handoff snapshot differs from accepted handoff snapshot")
if _TRUSTED_RELEASE_SOURCE_SNAPSHOTS[ACCEPTED_BUILD_EVIDENCE_DEFINITION_PATH] != _BUILD_EVIDENCE_DEFINITION_BYTES:
    raise RuntimeError("trusted-release build-evidence snapshot differs from accepted build-evidence snapshot")

_MAP_RE = re.compile(
    rb"\A([0-9a-f]+)-([0-9a-f]+) ([r-][w-][x-][ps]) ([0-9a-f]+) "
    rb"([0-9a-f]+):([0-9a-f]+) ([0-9]+)(?: +(.*))?\Z"
)
_GLIBC_RUNTIME_BASENAME_RE = re.compile(
    r"\A(?:libc\.so\.6|libc-[0-9]+(?:\.[0-9]+)*\.so)\Z"
)
_GLIBC_RUNTIME_ARTIFACT_PATH = "input/build-evidence/libc.so.6"
_UINT64_MAX = (1 << 64) - 1
MAX_TRUSTED_RELEASE_AUTHORITY_BYTES = 1 << 20
MAX_TRUSTED_RELEASE_SOURCE_BYTES = 4 << 20
MAX_TRUSTED_RELEASE_SOURCE_TOTAL_BYTES = 16 << 20


ACCEPTED_MEMBER_SHA256 = {
    1: "af27d4cae8233c0e049a014e30e3844d86e3d74c3a59143b89794f655ec2b88b",
    2: "4b1560613f7ed69d3bc0c6f9a9d8067091354cccf03c71e626e870b3549b69fa",
    3: "8dbe428939592cdfc86ba8730078563672de4a13081ebdb868fa97f543dfab89",
    4: "93056c4cfd071c1df396ba09bf82b4cbe807923977c4bca988b0aee1b8c94610",
    5: "510c852c65888dcf563d10e9e416ad0ab96c8503af1e412e06ff75dcb14caa18",
    6: "a05ee0bb16451fa11c965b8db50bdba3de080473329b44a5f8ffdfb8110c9333",
    7: "5bc30647a2512ab89e1d1507d3175068586bcb2ae020c19fab7f90696f3c1c1f",
    8: "f46d99a638da45105006fa5bdd70547674aa948fd1c35012c68d5dce2a274162",
    9: "2d01a2e3f066787378e7bebfe50f618d94dd46e4f38ede300adc0ad178f31969",
    10: "86176978f48ef9e888bfb373de713a9729e47e0bce4d9cc2fa8c917fc13e6773",
    11: "592bbf3d41790752140f567213ab6cf688c0d571fc89b9dda132875302b8f9cb",
    12: "6168e7d23ae1a514cca8b111bd0a99b0a6b7a903c59fbc00291ca949ce1110c8",
    13: "70626b468c7ebb89c8a053103957e400418221fcdf38395444a967d0dea917a1",
}
ACCEPTED_SCHEMA_BUNDLE_SHA256 = (
    "50fc22d910fe0c6c6934b6ad45ada7609ffefeb929eed1675f64a3ad86bb0c9f"
)

EXTERNAL_BY_KIND = {
    kind: (path, flags) for kind, path, flags in SEMANTIC.EXTERNAL_OBJECTS
}
EXTERNAL_KIND_BY_PATH = {
    path: kind for kind, (path, _flags) in EXTERNAL_BY_KIND.items()
}

PROCESS_AUTHENTICATION_BLOCKER = (
    "producer-authenticated process handoff and trusted offline projection are unavailable"
)
BUILD_EVIDENCE_AUTHENTICATION_BLOCKER = (
    "config-bound authenticated measured build evidence is unavailable"
)
TRUSTED_RELEASE_AUTHORITY_BLOCKER = (
    "externally pinned trusted-release authority is unavailable"
)
SCIENTIFIC_ADMISSION_BLOCKERS = (
    PROCESS_AUTHENTICATION_BLOCKER,
    BUILD_EVIDENCE_AUTHENTICATION_BLOCKER,
    TRUSTED_RELEASE_AUTHORITY_BLOCKER,
)


@dataclass(frozen=True)
class ArchiveVerificationResult:
    """Immutable result of one complete composed verification attempt."""

    terminal_outcome: str
    stream_sha256: str
    stream_bytes: int
    external_object_sha256: tuple[tuple[int, str], ...]
    verified_stream: Any
    status_promotion: Any | None
    scientific_admission_complete: bool
    admission_blockers: tuple[str, ...]


@dataclass(frozen=True)
class _TrustedReleaseAuthority:
    raw: bytes
    expected_sha256: str
    git_commit: str
    git_tree: str
    artifacts: Mapping[str, bytes]


def _fail(stage: str, detail: str) -> None:
    raise ArchiveVerificationError(stage, detail)


def _call(stage: str, operation: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return operation(*args, **kwargs)
    except ArchiveVerificationError:
        raise
    except Exception as error:
        _fail(stage, str(error))


def _snapshot_bytes(value: Any, where: str) -> bytes:
    if not isinstance(value, bytes):
        _fail("input", f"{where}: immutable bytes required")
    return bytes(value)


def _archive_path(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("input", f"{where}: nonempty archive-relative path required")
    if (
        value.startswith("/")
        or "\\" in value
        or any(component in ("", ".", "..") for component in value.split("/"))
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        _fail("input", f"{where}: normalized archive-relative POSIX path required")
    return value


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _trusted_release_hex(value: Any, width: int, where: str) -> str:
    if not isinstance(value, str) or re.fullmatch(rf"[0-9a-f]{{{width}}}", value) is None:
        _fail("trusted_release_authority", f"{where}: lowercase {width // 2}-byte hex required")
    return value


def _trusted_release_exact_object(
    value: Any, keys: set[str], where: str
) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        _fail("trusted_release_authority", f"{where}: exact members {tuple(sorted(keys))!r} required")
    return value


def _trusted_release_source_map(source_bytes: Any) -> dict[str, bytes]:
    if not isinstance(source_bytes, Mapping):
        _fail("input", "trusted_release_source_bytes: mapping required")
    supplied: dict[str, bytes] = {}
    for path, raw in source_bytes.items():
        normalized = _archive_path(path, f"trusted_release_source_bytes[{path!r}]")
        if normalized in supplied:
            _fail("trusted_release_authority", "source mapping contains a duplicate path")
        supplied[normalized] = _snapshot_bytes(
            raw, f"trusted_release_source_bytes[{normalized!r}]"
        )
    expected_paths = set(TRUSTED_RELEASE_SOURCE_PATHS)
    if set(supplied) != expected_paths:
        _fail(
            "trusted_release_authority",
            "source mapping exact path population required",
        )
    total = 0
    for path in sorted(expected_paths):
        raw = supplied[path]
        if not 0 < len(raw) <= MAX_TRUSTED_RELEASE_SOURCE_BYTES:
            _fail("trusted_release_authority", f"source bytes out of bounds: {path}")
        total += len(raw)
        if total > MAX_TRUSTED_RELEASE_SOURCE_TOTAL_BYTES:
            _fail("trusted_release_authority", "source aggregate bytes exceeded")
        if raw != _TRUSTED_RELEASE_SOURCE_SNAPSHOTS[path]:
            _fail(
                "trusted_release_authority",
                f"source bytes differ from the import-time executing snapshot: {path}",
            )
    return supplied


def _decode_trusted_release_authority(
    authority_bytes: Any,
    expected_authority_sha256: Any,
    source_bytes: Any,
) -> _TrustedReleaseAuthority:
    raw = _snapshot_bytes(authority_bytes, "trusted_release_authority_bytes")
    if not 0 < len(raw) <= MAX_TRUSTED_RELEASE_AUTHORITY_BYTES:
        _fail("trusted_release_authority", "authority bytes out of bounds")
    expected_sha256 = _trusted_release_hex(
        expected_authority_sha256,
        64,
        "trusted_release_authority_sha256",
    )
    if _hash(raw) != expected_sha256:
        _fail("trusted_release_authority", "authority digest differs from the external pin")
    supplied = _trusted_release_source_map(source_bytes)
    value = _call("trusted_release_authority", COVERAGE.parse_canonical, raw)
    authority = _trusted_release_exact_object(
        value,
        {"schema", "git", "sources"},
        "authority",
    )
    if authority["schema"] != TRUSTED_RELEASE_AUTHORITY_SCHEMA:
        _fail("trusted_release_authority", "authority schema mismatch")
    git = _trusted_release_exact_object(authority["git"], {"clean", "commit", "tree"}, "authority.git")
    if git["clean"] is not True:
        _fail("trusted_release_authority", "authority.git.clean must be exact true")
    commit = _trusted_release_hex(git["commit"], 40, "authority.git.commit")
    tree = _trusted_release_hex(git["tree"], 40, "authority.git.tree")
    rows = authority["sources"]
    if not isinstance(rows, list) or len(rows) != len(TRUSTED_RELEASE_SOURCE_PATHS):
        _fail("trusted_release_authority", "authority.sources exact population required")
    expected_order = tuple(sorted(TRUSTED_RELEASE_SOURCE_PATHS))
    observed_paths: list[str] = []
    for index, row in enumerate(rows):
        source = _trusted_release_exact_object(
            row,
            {"bytes", "path", "sha256"},
            f"authority.sources[{index}]",
        )
        path = _archive_path(source["path"], f"authority.sources[{index}].path")
        byte_count = source["bytes"]
        if type(byte_count) is not int or not 0 < byte_count <= MAX_TRUSTED_RELEASE_SOURCE_BYTES:
            _fail("trusted_release_authority", f"authority.sources[{index}].bytes: bounded integer required")
        digest = _trusted_release_hex(source["sha256"], 64, f"authority.sources[{index}].sha256")
        observed_paths.append(path)
        source_raw = supplied.get(path)
        if source_raw is None:
            _fail("trusted_release_authority", f"authority source is absent from supplied bytes: {path}")
        if byte_count != len(source_raw) or digest != _hash(source_raw):
            _fail("trusted_release_authority", f"authority source byte/digest mismatch: {path}")
    if tuple(observed_paths) != expected_order:
        _fail("trusted_release_authority", "authority.sources must be exact, unique, and path-sorted")
    return _TrustedReleaseAuthority(
        raw=raw,
        expected_sha256=expected_sha256,
        git_commit=commit,
        git_tree=tree,
        artifacts={TRUSTED_RELEASE_AUTHORITY_PATH: raw, **supplied},
    )


def _validate_trusted_release_git(
    authority: _TrustedReleaseAuthority, build: Mapping[str, Any]
) -> None:
    identity = build.get("build_identity") if isinstance(build, Mapping) else None
    if not isinstance(identity, Mapping):
        _fail("trusted_release_authority", "build identity required for authority binding")
    if identity.get("dirty") is not False:
        _fail("trusted_release_authority", "build identity must be clean for authority binding")
    if identity.get("source_commit") != authority.git_commit:
        _fail("trusted_release_authority", "authority git commit differs from build identity")
    if identity.get("source_tree") != authority.git_tree:
        _fail("trusted_release_authority", "authority git tree differs from build identity")


def validate_trusted_release_authority(
    authority_bytes: Any,
    expected_authority_sha256: Any,
    source_bytes: Any,
    *,
    build: Mapping[str, Any],
) -> dict[str, bytes]:
    """Validate an externally pinned authority against local source snapshots.

    This trusts a separately supplied controller/verifier digest and frozen
    input bytes. It cannot defend against malicious modified running code,
    coordinated same-UID rewrites outside each frozen read, or prove Git
    authorship or durable storage.
    """

    authority = _decode_trusted_release_authority(
        authority_bytes,
        expected_authority_sha256,
        source_bytes,
    )
    _validate_trusted_release_git(authority, build)
    return {path: bytes(raw) for path, raw in authority.artifacts.items()}


def accepted_trusted_release_source_bytes() -> dict[str, bytes]:
    """Return copies of the import-time 15-file executing source snapshot."""

    return {
        path: bytes(raw)
        for path, raw in _TRUSTED_RELEASE_SOURCE_SNAPSHOTS.items()
    }


def make_trusted_release_authority_bytes(
    *, commit: str, tree: str, source_bytes: Mapping[str, bytes]
) -> bytes:
    """Build deterministic authority bytes; callers must pin its digest separately."""

    commit = _trusted_release_hex(commit, 40, "authority.git.commit")
    tree = _trusted_release_hex(tree, 40, "authority.git.tree")
    sources = _trusted_release_source_map(source_bytes)
    return COVERAGE.canonical_bytes(
        {
            "schema": TRUSTED_RELEASE_AUTHORITY_SCHEMA,
            "git": {"clean": True, "commit": commit, "tree": tree},
            "sources": [
                {"bytes": len(raw), "path": path, "sha256": _hash(raw)}
                for path, raw in sorted(sources.items())
            ],
        }
    )


def _merge_authenticated_artifacts(
    destination: dict[str, bytes], incoming: Mapping[str, bytes], where: str
) -> None:
    for path, raw in incoming.items():
        previous = destination.get(path)
        if previous is not None:
            if previous != raw:
                _fail("cross_bindings", f"{where}: conflicting authenticated artifact: {path}")
            continue
        destination[path] = raw


def _accepted_static_members() -> tuple[dict[int, bytes], bytes]:
    semantic_root = CATALOG_ROOT / "semantic"
    loaded = _call("schema_bundle", SEMANTIC.validate_semantic_root, semantic_root)
    members = {object_type: raw for object_type, (_path, raw) in loaded.items()}
    coverage_members = _call("schema_bundle", COVERAGE.static_members)
    member_paths: dict[int, Path] = {}
    for object_type, _name, archive_path, owner in SEMANTIC.MEMBER_REGISTRY:
        if owner == "semantic":
            root = semantic_root
        elif owner == "coverage":
            root = CATALOG_ROOT / "coverage"
            raw = coverage_members.get(archive_path)
            if raw is None:
                _fail("schema_bundle", f"accepted member {object_type} unavailable")
            _call("schema_bundle", COVERAGE.validate_static_member, archive_path, raw)
            members[object_type] = raw
        elif owner == "config":
            root = CATALOG_ROOT / "config"
            if not isinstance(CONFIG.SCHEMA_BYTES, bytes):
                _fail("schema_bundle", "CONFIG.SCHEMA_BYTES: immutable bytes required")
            raw = bytes(CONFIG.SCHEMA_BYTES)
            schema_path = root / "archive" / archive_path
            try:
                literal_raw = schema_path.read_bytes()
            except OSError as error:
                _fail("schema_bundle", f"accepted member {object_type} unavailable: {error}")
            if raw != literal_raw:
                _fail("schema_bundle", f"accepted member {object_type} literal/module mismatch")
            schema_value = _call("schema_bundle", SEMANTIC.parse_canonical, raw)
            _call("schema_bundle", CONFIG.validate_schema_definition, schema_value)
            members[object_type] = raw
        elif owner == "sampling":
            root = CATALOG_ROOT / "sampling"
            raw = _call("schema_bundle", SELECTION.definition_bytes)
            definition_path = root / "archive" / archive_path
            try:
                literal_raw = definition_path.read_bytes()
            except OSError as error:
                _fail("schema_bundle", f"accepted member {object_type} unavailable: {error}")
            if raw != literal_raw:
                _fail("schema_bundle", f"accepted member {object_type} literal/module mismatch")
            definition_value = _call("schema_bundle", SEMANTIC.parse_canonical, raw)
            _call("schema_bundle", SELECTION.validate_definition, definition_value)
            members[object_type] = raw
        elif owner == "runtime":
            root = CATALOG_ROOT / "runtime"
            raw = _call("schema_bundle", RUNTIME.definition_bytes)
            definition_path = root / "archive" / archive_path
            try:
                literal_raw = definition_path.read_bytes()
            except OSError as error:
                _fail("schema_bundle", f"accepted member {object_type} unavailable: {error}")
            if raw != literal_raw:
                _fail("schema_bundle", f"accepted member {object_type} literal/module mismatch")
            _call("schema_bundle", RUNTIME.validate_definition_bytes, raw)
            members[object_type] = raw
        else:
            _fail("schema_bundle", f"unknown member owner {owner!r}")
        member_paths[object_type] = root / "archive" / archive_path

    expected_types = {row[0] for row in SEMANTIC.MEMBER_REGISTRY}
    if set(members) != expected_types or set(ACCEPTED_MEMBER_SHA256) != expected_types:
        _fail("schema_bundle", "exact accepted registered member set unavailable")
    for object_type, expected in ACCEPTED_MEMBER_SHA256.items():
        observed = _hash(members[object_type])
        if observed != expected:
            _fail(
                "schema_bundle",
                f"accepted member {object_type} digest drift: {observed}",
            )

    bundle = _call("schema_bundle", SEMANTIC.build_bundle, member_paths)
    _call("schema_bundle", SEMANTIC.validate_bundle, bundle, members)
    if _hash(bundle) != ACCEPTED_SCHEMA_BUNDLE_SHA256:
        _fail("schema_bundle", "accepted registered bundle digest drift")
    return members, bundle


def accepted_schema_bundle_bytes() -> bytes:
    """Return the uniquely generated bundle for the accepted local literals."""

    return _accepted_static_members()[1]

def accepted_verifier_definition_bytes() -> bytes:
    """Return a copy of the verifier source bytes frozen at module import."""

    return bytes(_ACCEPTED_VERIFIER_DEFINITION_BYTES)


def accepted_build_evidence_definition_bytes() -> bytes:
    """Return the import-time trusted measured-build evidence source snapshot."""

    return bytes(_BUILD_EVIDENCE_DEFINITION_BYTES)


def accepted_event_classifier_definition_bytes() -> bytes:
    """Return the exact classifier source snapshot accepted at module import."""

    return bytes(_EVENT_CLASSIFIER_DEFINITION_BYTES)



def _entry_count(kind: int, value: Mapping[str, Any]) -> int:
    if kind in range(1, 8):
        entries = value.get("entries")
        if not isinstance(entries, list):
            _fail("external_semantics", f"object kind {kind}: entries array required")
        return len(entries)
    if kind == 8:
        return len(value["entries"])
    if kind == 9:
        return len(value["module_population"])
    if kind in (10, 12):
        return 1
    if kind == 11:
        return len(value["entries"])
    _fail("external_semantics", f"object kind {kind}: unsupported")


def _uuid_text(raw: bytes) -> str:
    try:
        return str(uuid.UUID(bytes=raw))
    except (ValueError, AttributeError) as error:
        _fail("cross_bindings", f"UUID conversion failed: {error}")


def _validate_primary_build(opening: Any, build: Mapping[str, Any]) -> None:
    identity = build["build_identity"]
    rows = {row["logical_id"]: row for row in build["entries"]}
    primary = rows[identity["primary_logical_elf_id"]]
    if primary["sha256"] != opening.primary_binary_sha256.hex():
        _fail("cross_bindings", "opening.primary_binary_sha256: build mismatch")
    try:
        build_id_bytes = bytes.fromhex(primary["build_id"])
    except ValueError as error:
        _fail("cross_bindings", f"primary Build-ID is not hex: {error}")
    if _hash(build_id_bytes) != opening.primary_build_id_sha256.hex():
        _fail("cross_bindings", "opening.primary_build_id_sha256: build mismatch")
    source = opening.source_object_value[: opening.source_object_length].hex()
    if identity["source_commit"] != source:
        _fail("cross_bindings", "opening source object: build commit mismatch")
    expected_architecture = 1 if opening.clock_kind == 1 else 2
    if build["architecture_id"] != expected_architecture:
        _fail("cross_bindings", "opening clock architecture: build mismatch")


def _validate_run_opening(opening: Any, run: Mapping[str, Any]) -> None:
    relations = {
        "process_generation": opening.process_generation,
        "role_id": opening.role_kind,
        "run_uuid": _uuid_text(opening.run_uuid),
        "process_uuid": _uuid_text(opening.process_uuid),
        "configuration_instance_sha256": opening.configuration_instance_sha256.hex(),
    }
    for field, expected in relations.items():
        if run[field] != expected:
            _fail("cross_bindings", f"run_coverage.{field}: opening mismatch")


def _validate_records(
    container: Any,
    *,
    api: Mapping[str, Any],
    context: Mapping[str, Any],
    mode_id: int,
    event_semantics: Mapping[str, Any],
    realloc_zero_policy_id: int | None,
) -> tuple[tuple[Any, ...], int]:
    pairs = {(row["event_kind"], row["api_id"]) for row in api["entries"]}
    context_keys = {
        (row["process_generation"], row["context_id"])
        for row in context["entries"]
    }
    records = [record for chunk in container.chunks for record in chunk.records]
    for index, record in enumerate(records):
        if (record.event_kind, record.api_id) not in pairs:
            _fail("record_bindings", f"record[{index}]: unresolved API/event pair")
        generation = container.opening_header.process_generation
        if record.context_id and (generation, record.context_id) not in context_keys:
            _fail("record_bindings", f"record[{index}]: unresolved context")
    classifications = _call(
        "record_semantics",
        EVENT_CLASSIFIER.classify_records,
        event_semantics,
        records,
        mode_id=mode_id,
        realloc_zero_policy_id=realloc_zero_policy_id,
        cutoff_before_counter=container.trailer_body.header.cutoff_before_counter,
        cutoff_after_counter=container.trailer_body.header.cutoff_after_counter,
        seal_bracket_available=(
            container.trailer_body.header.finalization_stage >= 1
        ),
    )
    gaps = _call(
        "record_semantics",
        EVENT_CLASSIFIER.exact_mode_sequence_gap_count,
        records,
        mode_id=mode_id,
    )
    return classifications, gaps


def _validate_a03_selection(
    records: Sequence[Any],
    classifications: Sequence[Any],
    *,
    process_generation: int,
    config: Mapping[str, Any],
) -> None:
    """Bind A03 record selection bits to the registered mapping and trial seed."""

    if config["mode_id"] != 3:
        return
    seed_k = _call(
        "sampling",
        SELECTION.parse_seed_hex,
        config["sample_seed_hex"],
        config["sample_seed_provenance_id"],
        config["sample_seed_status_id"],
        require_publication_seed=False,
    )
    threshold = _call(
        "sampling",
        SELECTION.validate_threshold,
        config["sample_threshold"],
        active_a03=True,
    )
    for index, record in enumerate(records):
        successor_created = bool(record.flags & (1 << 11))
        successor_selected = bool(record.flags & (1 << 16))
        expected_successor = successor_created and _call(
            "sampling",
            SELECTION.selected,
            process_generation,
            record.thread_index,
            record.thread_sequence,
            seed_k=seed_k,
            threshold=threshold,
        )
        if successor_selected != expected_successor:
            _fail("sampling", f"record[{index}]: successor-selected bit/mapping mismatch")

        match_valid = bool(record.flags & (1 << 14))
        predecessor_selected = bool(record.flags & (1 << 15))
        if match_valid:
            origin = _call(
                "sampling",
                SELECTION.validate_instance_key,
                process_generation,
                record.arg1,
                record.arg2,
            )
            expected_predecessor = _call(
                "sampling",
                SELECTION.selected,
                *origin,
                seed_k=seed_k,
                threshold=threshold,
            )
            if bool(record.flags & (1 << 17)) != (record.arg1 != record.thread_index):
                _fail("sampling", f"record[{index}]: cross-thread endpoint mismatch")
        else:
            expected_predecessor = False
        if predecessor_selected != expected_predecessor:
            _fail("sampling", f"record[{index}]: predecessor-selected bit/mapping mismatch")
        if not successor_selected and not predecessor_selected:
            _fail("sampling", f"record[{index}]: emitted A03 record has no selected endpoint")
    _call(
        "record_replay",
        EVENT_CLASSIFIER.reconcile_a03_predecessors,
        records,
        classifications,
        process_generation=process_generation,
    )


def _compare_pre_footer(pre: Mapping[str, Any], trailer: Any) -> None:
    relations = {
        "active_generation": trailer.active_generation,
        "active_start_counter": trailer.active_start_counter,
        "active_start_monotonic_raw_ns": trailer.active_start_monotonic_raw_ns,
        "cutoff_after_counter": trailer.cutoff_after_counter,
        "cutoff_before_counter": trailer.cutoff_before_counter,
        "final_counter": trailer.final_counter,
        "final_monotonic_raw_ns": trailer.final_monotonic_raw_ns,
        "final_realtime_unix_ns": trailer.final_realtime_unix_ns,
        "finalization_stage": trailer.finalization_stage,
        "lifecycle_state": trailer.lifecycle_state,
        "payload_writer_state": trailer.payload_writer_state,
        "process_generation": trailer.process_generation,
        "quiescence_complete_counter": trailer.quiescence_complete_counter,
        "reason_code": trailer.terminal_reason_code,
        "scope_kind": trailer.scope_kind,
        "terminal_flags": trailer.terminal_flags,
    }
    for field, expected in relations.items():
        if pre[field] != expected:
            _fail("terminal_binding", f"pre_footer.{field}: trailer mismatch")


def _trusted_stream_identity(
    raw: bytes,
    container: Any,
    *,
    stream_path: str,
    pre_footer_row: Any,
    accepted_verifier_definition_sha256: str,
) -> Any:
    footer_offset = len(raw) - WIRE.FOOTER_BYTES
    footer_preimage = raw[footer_offset : footer_offset + 224]
    return STATUS.VerifiedStreamIdentity(
        stream_path=stream_path,
        physical_bytes=len(raw),
        whole_stream_sha256=_hash(raw),
        pre_footer_status_bytes=pre_footer_row.byte_count,
        pre_footer_status_sha256=pre_footer_row.sha256.hex(),
        footer_preimage_sha256=_hash(footer_preimage),
        opening_header_sha256=container.footer.opening_header_sha256.hex(),
        prefix_sha256=container.footer.prefix_sha256.hex(),
        trailer_body_sha256=container.footer.trailer_body_sha256.hex(),
        verifier_definition_sha256=accepted_verifier_definition_sha256,
    )



def _catalog_bytes(
    catalog_id: str,
    entries: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
) -> bytes:
    return SEMANTIC.canonical_bytes(
        {
            "catalog_id": catalog_id,
            "entries": [dict(row) for row in entries],
            "schema": dict(schema),
            "version": {"major": 1, "minor": 0},
        }
    )


def _parse_process_maps(raw: bytes) -> dict[tuple[int, int, str], tuple[dict[str, Any], ...]]:
    groups: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for index, line in enumerate(raw.splitlines()):
        match = _MAP_RE.fullmatch(line)
        if match is None:
            _fail("process_handoff", f"proc maps line {index + 1}: grammar mismatch")
        start, end, permissions, offset, major, minor, inode, path = match.groups()
        inode_value = int(inode)
        if (
            inode_value == 0
            or not path
            or not path.startswith(b"/")
            or path.endswith(b" (deleted)")
        ):
            continue
        try:
            loaded_path = path.decode("utf-8", "strict")
            permissions_text = permissions.decode("ascii", "strict")
        except UnicodeDecodeError as error:
            _fail("process_handoff", f"proc maps line {index + 1}: invalid text: {error}")
        key = (os.makedev(int(major, 16), int(minor, 16)), inode_value, loaded_path)
        groups.setdefault(key, []).append(
            {
                "end_address": int(end, 16),
                "file_offset": int(offset, 16),
                "permissions": permissions_text,
                "start_address": int(start, 16),
            }
        )
    result: dict[tuple[int, int, str], tuple[dict[str, Any], ...]] = {}
    for key, segments in groups.items():
        segments.sort(key=lambda row: row["start_address"])
        result[key] = tuple(segments)
    return result


def _validate_authenticated_runtime_libc(
    handoff: Any,
    build: Mapping[str, Any],
    build_evidence_artifacts: Mapping[str, bytes],
) -> None:
    """Bind the unique mapped glibc identity to measured dependency bytes.

    The producer authenticates the live path/device/inode-to-byte relation while
    that inode is still available.  This offline projection independently binds
    the persisted maps identity to the measured dependency digest and artifact.
    """

    dependencies = [
        row
        for row in build["dependencies"]
        if row["dependency_id"] == "glibc_runtime"
    ]
    if len(dependencies) != 1:
        _fail("build_evidence", "exactly one glibc_runtime dependency is required")
    dependency = dependencies[0]
    if dependency["name"] != "libc.so.6" or dependency["evidence_state_id"] != 1:
        _fail("build_evidence", "glibc_runtime dependency is not measured libc.so.6")
    libc_raw = build_evidence_artifacts.get(_GLIBC_RUNTIME_ARTIFACT_PATH)
    if libc_raw is None or _hash(libc_raw) != dependency["sha256"]:
        _fail("build_evidence", "mapped glibc dependency artifact digest mismatch")

    maps = _parse_process_maps(handoff.maps_bytes)
    candidates = [
        (device, inode, loaded_path)
        for (device, inode, loaded_path), segments in maps.items()
        if _GLIBC_RUNTIME_BASENAME_RE.fullmatch(os.path.basename(loaded_path))
        and any("x" in segment["permissions"] for segment in segments)
    ]
    if len(candidates) != 1:
        _fail(
            "process_handoff",
            "authenticated maps require exactly one executable glibc runtime identity",
        )


def _handoff_clock_row(handoff: Any) -> dict[str, Any]:
    samples = (handoff.opening_sample, handoff.writer.final_sample)
    sample_rows = [
        {
            "counter": sample.counter,
            "monotonic_raw_after_ns": sample.monotonic_raw_after_ns,
            "monotonic_raw_before_ns": sample.monotonic_raw_before_ns,
            "realtime_unix_ns": sample.realtime_unix_ns,
            "sample_ordinal": index,
        }
        for index, sample in enumerate(samples, 1)
    ]
    numerator = handoff.opening.counter_frequency_numerator
    denominator = handoff.opening.counter_frequency_denominator

    def predicted(counter: int) -> int:
        return handoff.opening.start_monotonic_raw_ns + (
            (counter - handoff.opening.start_counter)
            * 1_000_000_000
            * denominator
        ) // numerator

    observed_max = max(
        max(
            sample.monotonic_raw_before_ns - predicted(sample.counter),
            predicted(sample.counter) - sample.monotonic_raw_after_ns,
            0,
        )
        for sample in samples
    )
    realtime_discontinuity = any(
        abs(
            (sample.realtime_unix_ns - samples[0].realtime_unix_ns)
            - (
                (
                    sample.monotonic_raw_before_ns
                    + sample.monotonic_raw_after_ns
                )
                // 2
                - handoff.opening.start_monotonic_raw_ns
            )
        )
        > handoff.opening.calibration_error_bound_ns
        for sample in samples
    )
    return {
        "acquisition_source_id": handoff.writer.clock_info.acquisition_source_id,
        "acquisition_status_id": 1,
        "architecture_id": handoff.writer.clock_info.architecture_id,
        "calibration_error_bound_ns": handoff.opening.calibration_error_bound_ns,
        "calibration_kind": handoff.opening.calibration_kind,
        "calibration_span_ns": handoff.opening.calibration_span_ns,
        "clock_kind": handoff.opening.clock_kind,
        "counter_frequency_denominator": denominator,
        "counter_frequency_numerator": numerator,
        "counter_invalid_observed": (
            handoff.writer.clock_status != 0
            or any(thread.counter_invalids for thread in handoff.threads)
        ),
        "counter_stability_status_id": 1,
        "observed_max_error_ns": observed_max,
        "process_generation": handoff.opening.process_generation,
        "realtime_discontinuity_observed": realtime_discontinuity,
        "samples": sample_rows,
        "start_counter": handoff.opening.start_counter,
        "start_monotonic_raw_ns": handoff.opening.start_monotonic_raw_ns,
        "start_realtime_unix_ns": handoff.opening.start_realtime_unix_ns,
    }


def _handoff_diagnostic_rows(handoff: Any) -> list[dict[str, Any]]:
    all_reason_ids = (1, 16, 17, 18, 32, 48, 49, 50, 51, 64)
    mode_row = DIAGNOSTICS.MODE_ROWS.get(handoff.writer.runtime_snapshot.mode_id)
    if mode_row is None:
        _fail("process_handoff", "diagnostic mode is not registered")
    producer_reason_ids = set(mode_row[4])
    rows: list[dict[str, Any]] = []
    for thread in handoff.threads:
        for index, reason_id in enumerate(all_reason_ids):
            if reason_id not in producer_reason_ids:
                continue
            rows.append(
                {
                    "counter_scope_id": thread.thread_index,
                    "counter_scope_kind": 1,
                    "process_generation": handoff.opening.process_generation,
                    "reason_id": reason_id,
                    "saturated": bool(
                        thread.diagnostic_saturated_mask & (1 << index)
                    ),
                    "value": thread.diagnostic_values[index],
                }
            )
    registration_values = (
        handoff.unregistered_active_thread_failures,
        handoff.writer.runtime_snapshot.registration_capacity_failures,
    )
    for index, reason_id in enumerate((2, 3)):
        rows.append(
            {
                "counter_scope_id": 1,
                "counter_scope_kind": 2,
                "process_generation": handoff.opening.process_generation,
                "reason_id": reason_id,
                "saturated": bool(
                    handoff.registration_diagnostic_saturated_mask & (1 << index)
                ),
                "value": registration_values[index],
            }
        )
    rows.extend(
        (
            {
                "counter_scope_id": 1,
                "counter_scope_kind": 3,
                "process_generation": handoff.opening.process_generation,
                "reason_id": 80,
                "saturated": (
                    handoff.writer_io_or_finalization_failures == _UINT64_MAX
                ),
                "value": handoff.writer_io_or_finalization_failures,
            },
            {
                "counter_scope_id": 1,
                "counter_scope_kind": 4,
                "process_generation": handoff.opening.process_generation,
                "reason_id": 96,
                "saturated": False,
                "value": handoff.diagnostic_saturation_transitions,
            },
        )
    )
    return sorted(
        rows,
        key=lambda row: (
            row["process_generation"],
            row["counter_scope_kind"],
            row["counter_scope_id"],
            row["reason_id"],
        ),
    )


def _handoff_terminal_projection(writer: Any) -> tuple[int, int, int, str]:
    """Mirror the exact terminal outcomes accepted by the C offline finalizer."""

    if not writer.prefooter_closed:
        _fail("process_handoff", "authenticated terminal requires a closed pre-footer stream")
    if writer.status == 0:
        if writer.runtime_status != 0 or writer.clock_status != 0:
            _fail("process_handoff", "COMPLETE requires a successful producer handoff")
        return (5, 0, 5, "complete")
    if writer.status == 9:
        return (6, 6, 6, "failed")
    if writer.status == 11:
        if writer.clock_status == 0:
            _fail("process_handoff", "CLOCK_ERROR requires a nonzero clock status")
        return (7, 7, 5, "incomplete")
    _fail(
        "process_handoff",
        f"writer status {writer.status} has no authenticated terminal representation",
    )


def _handoff_pre_footer(
    handoff: Any, diagnostic_projection: Mapping[str, Any]
) -> dict[str, Any]:
    clock_row = _handoff_clock_row(handoff)
    lifecycle_state, reason_code, payload_writer_state, terminal_outcome = (
        _handoff_terminal_projection(handoff.writer)
    )
    terminal_flags = 0x017F if payload_writer_state == 6 else 0x01FF
    if terminal_outcome == "complete":
        terminal_flags |= (1 << 9) | (1 << 10) | (1 << 11)
    if diagnostic_projection["aggregate_saturated"]:
        terminal_flags |= 1 << 12
    if clock_row["realtime_discontinuity_observed"]:
        terminal_flags |= 1 << 14
    if clock_row["counter_invalid_observed"]:
        terminal_flags |= 1 << 15
    writer = handoff.writer
    return {
        "active_generation": handoff.opening.process_generation,
        "active_start_counter": handoff.opening.start_counter,
        "active_start_monotonic_raw_ns": handoff.opening.start_monotonic_raw_ns,
        "cutoff_after_counter": writer.seal_after_sample.counter,
        "cutoff_before_counter": writer.seal_before_sample.counter,
        "diagnostic_population_partial": False,
        "final_counter": writer.final_sample.counter,
        "final_monotonic_raw_ns": (
            writer.final_sample.monotonic_raw_before_ns
            + writer.final_sample.monotonic_raw_after_ns
        )
        // 2,
        "final_realtime_unix_ns": writer.final_sample.realtime_unix_ns,
        "finalization_stage": 6,
        "lifecycle_state": lifecycle_state,
        "payload_writer_state": payload_writer_state,
        "process_generation": handoff.opening.process_generation,
        "quiescence_complete_counter": writer.drain_complete_sample.counter,
        "reason_code": reason_code,
        "schema": STATUS.SCHEMA_PRE_FOOTER,
        "scope_kind": handoff.opening.scope_kind,
        "terminal_flags": terminal_flags,
    }


def _sat_add(left: int, right: int) -> int:
    return _UINT64_MAX if right > _UINT64_MAX - left else left + right


def _record_requested_bytes(record: Any) -> int:
    if record.api_id in (1, 3, 5, 9, 10):
        return record.arg0
    if record.api_id == 2:
        return record.arg2
    if record.api_id in (6, 7, 8):
        return record.arg1
    return 0


def validate_handoff_runtime_configuration(
    handoff: Any,
    config: Mapping[str, Any],
    trusted_realloc_policy_id: int | None,
) -> None:
    """Bind the decoded producer controls before any archive append."""

    _call(
        "process_handoff",
        CONFIG.validate_effective_configuration,
        config,
    )
    if trusted_realloc_policy_id is not None and (
        type(trusted_realloc_policy_id) is not int
        or trusted_realloc_policy_id not in (1, 2)
    ):
        _fail("process_handoff", "trusted realloc policy 1 or 2 required")
    flush_interval_ns = config["flush_us"] * 1_000
    if flush_interval_ns > _UINT64_MAX:
        _fail(
            "process_handoff",
            "effective flush interval is not representable in nanoseconds",
        )
    writer = handoff.writer
    sampled = config["mode_id"] == 3
    expected_table_entries = config["table_entries"] if sampled else 0
    expected_sample_seed = int(config["sample_seed_hex"], 16) if sampled else 0
    expected_sample_threshold = config["sample_threshold"] if sampled else 0
    expected_table_probes = config["table_probes"] if sampled else 0
    expected_table_shards = (
        1 << (min(expected_table_entries, 256).bit_length() - 1)
        if sampled
        else 0
    )
    if (
        writer.runtime_snapshot.mode_id != config["mode_id"]
        or writer.runtime_snapshot.table_entries != expected_table_entries
        or writer.runtime_snapshot.sample_seed != expected_sample_seed
        or writer.runtime_snapshot.sample_threshold != expected_sample_threshold
        or writer.runtime_snapshot.table_probes != expected_table_probes
        or writer.runtime_snapshot.table_shards != expected_table_shards
        or handoff.ring_records != config["ring_records"]
        or handoff.flush_records != config["flush_records"]
        or handoff.flush_interval_ns != flush_interval_ns
        or (
            trusted_realloc_policy_id is not None
            and handoff.realloc_zero_policy_id != trusted_realloc_policy_id
        )
    ):
        _fail(
            "process_handoff",
            "runtime configuration differs from authenticated catalog/coverage",
        )


def _validate_authenticated_process_handoff(
    handoff_raw: bytes,
    raw: bytes,
    container: Any,
    objects: Mapping[str, bytes],
    parsed: Mapping[int, Mapping[str, Any]],
    module_rows: Mapping[tuple[int, int], Mapping[str, Any]],
    diagnostic_projection: Mapping[str, Any],
    pre_footer_raw: bytes,
    config: Mapping[str, Any],
    trusted_realloc_policy_id: int | None,
) -> Any:
    handoff = _call(
        "process_handoff", HANDOFF.decode_process_handoff, handoff_raw
    )
    opening = container.opening_header
    trailer = container.trailer_body.header
    records = tuple(
        record for chunk in container.chunks for record in chunk.records
    )
    if handoff.opening_raw != raw[: WIRE.OPENING_HEADER_BYTES]:
        _fail("process_handoff", "opening bytes differ from authenticated producer handoff")
    if handoff.bootstrap_bytes != objects[EXTERNAL_BY_KIND[10][0]]:
        _fail("process_handoff", "effective configuration differs from authenticated bootstrap")
    validate_handoff_runtime_configuration(
        handoff,
        config,
        trusted_realloc_policy_id,
    )
    writer = handoff.writer
    expected_terminal_outcome = _handoff_terminal_projection(writer)[3]
    prefix_bytes = container.footer.prefix_bytes
    if (
        writer.stream_bytes != prefix_bytes
        or writer.stream_bytes != container.footer.trailer_offset
        or writer.stream_bytes != trailer.chunks_end_offset
        or writer.chunk_count != trailer.chunk_count
        or writer.record_count != trailer.record_count
        or writer.payload_bytes != trailer.payload_bytes
    ):
        _fail("process_handoff", "writer prefix/count projection mismatch")
    prefix = raw[:prefix_bytes]
    if (
        hashlib.sha256(prefix).digest() != handoff.prefix_sha256
        or container.footer.prefix_sha256 != handoff.prefix_sha256
    ):
        _fail("process_handoff", "producer-authenticated prefix digest mismatch")
    if opening.process_generation != handoff.opening.process_generation:
        _fail("process_handoff", "opening process generation mismatch")

    expected_threads = [
        {
            "process_generation": thread.process_generation,
            "registration_ordinal": thread.registration_ordinal,
            "thread_index": thread.thread_index,
        }
        for thread in handoff.threads
    ]
    if parsed[5]["entries"] != expected_threads:
        _fail("process_handoff", "thread catalog differs from producer snapshot")

    if writer.runtime_snapshot.mode_id == 4:
        records_by_thread: dict[int, list[Any]] = {
            thread.thread_index: [] for thread in handoff.threads
        }
        for record in records:
            if record.thread_index not in records_by_thread:
                _fail("process_handoff", "record refers to a non-authenticated thread")
            records_by_thread[record.thread_index].append(record)
        for thread in handoff.threads:
            thread_records = records_by_thread[thread.thread_index]
            sequences = tuple(record.thread_sequence for record in thread_records)
            if sequences != tuple(range(1, thread.thread_sequence + 1)):
                _fail("process_handoff", "exact-mode thread sequence differs from handoff")
            api_counts = tuple(
                sum(record.api_id == api_id for record in thread_records)
                for api_id in range(1, HANDOFF.API_SLOT_COUNT + 1)
            )
            requested = 0
            for record in thread_records:
                requested = _sat_add(requested, _record_requested_bytes(record))
            if (
                api_counts != thread.api_attempts
                or len(thread_records) != thread.completed_transactions
                or requested != thread.requested_bytes
            ):
                _fail("process_handoff", "record population differs from thread counters")
    elif writer.runtime_snapshot.mode_id == 3:
        authenticated_threads = {thread.thread_index for thread in handoff.threads}
        if any(record.thread_index not in authenticated_threads for record in records):
            _fail(
                "process_handoff", "sampled record refers to a non-authenticated thread"
            )
    elif records:
        _fail("process_handoff", "counter-only handoff unexpectedly has event records")

    maps = _parse_process_maps(handoff.maps_bytes)
    for row in module_rows.values():
        key = (row["device"], row["inode"], row["loaded_path"])
        if key not in maps or tuple(row["segments"]) != maps[key]:
            _fail("process_handoff", "module map differs from authenticated proc snapshot")

    expected_clock = _catalog_bytes(
        "oai_memprof_clock", [_handoff_clock_row(handoff)], RUNTIME.CLOCK_SCHEMA
    )
    if objects[EXTERNAL_BY_KIND[7][0]] != expected_clock:
        _fail("process_handoff", "clock catalog differs from authenticated samples")

    expected_diagnostics = _call(
        "process_handoff",
        DIAGNOSTICS.make_diagnostics_bytes,
        definition_sha256=ACCEPTED_MEMBER_SHA256[8],
        mode_id=writer.runtime_snapshot.mode_id,
        process_generation=opening.process_generation,
        counter_rows=_handoff_diagnostic_rows(handoff),
        ready_thread_indices=tuple(
            thread.thread_index for thread in handoff.threads
        ),
        producer_population_complete=True,
    )
    if objects[EXTERNAL_BY_KIND[11][0]] != expected_diagnostics:
        _fail("process_handoff", "diagnostics differ from authenticated counters")
    expected_pre_footer = STATUS.canonical_bytes(
        _handoff_pre_footer(handoff, diagnostic_projection)
    )
    if pre_footer_raw != expected_pre_footer:
        _fail("process_handoff", "pre-footer status differs from authenticated finalization")
    return handoff, expected_terminal_outcome


def verify_archive_candidate(
    stream_bytes: bytes,
    external_object_bytes: Mapping[str, bytes],
    *,
    stream_path: str,
    verifier_definition_path: str,
    verifier_definition_bytes: bytes,
    unavailable_counter_keys: Sequence[Sequence[int]] = (),
    registration_available: bool = True,
    writer_available: bool = True,
    aggregate_available: bool = True,
    process_handoff_bytes: bytes | None = None,
    producer_definition_path: str | None = None,
    producer_definition_bytes: bytes | None = None,
    handoff_decoder_definition_path: str | None = None,
    handoff_decoder_definition_bytes: bytes | None = None,
    build_evidence_definition_path: str | None = None,
    build_evidence_definition_bytes: bytes | None = None,
    build_evidence_bytes: bytes | None = None,
    build_evidence_artifact_bytes: Mapping[str, bytes] | None = None,
    trusted_release_authority_bytes: bytes | None = None,
    trusted_release_authority_sha256: str | None = None,
    trusted_release_source_bytes: Mapping[str, bytes] | None = None,
    post_close_receipt_bytes: bytes | None = None,
    manifest_bytes: bytes | None = None,
) -> ArchiveVerificationResult:
    """Verify one immutable candidate without repairing or defaulting evidence.

    The exact structural stream is always decoded before any external bytes.
    FAILED/INCOMPLETE terminals are returned as retained negative evidence and
    are never passed to the COMPLETE-only status promotion function.
    """

    raw = _snapshot_bytes(stream_bytes, "stream_bytes")
    stream_path = _archive_path(stream_path, "stream_path")
    verifier_path = _archive_path(
        verifier_definition_path, "verifier_definition_path"
    )
    verifier_definition_raw = _snapshot_bytes(
        verifier_definition_bytes, "verifier_definition_bytes"
    )
    if (
        verifier_path != ACCEPTED_VERIFIER_DEFINITION_PATH
        or verifier_definition_raw != _ACCEPTED_VERIFIER_DEFINITION_BYTES
    ):
        _fail("input", "candidate verifier path/bytes differ from the internal accepted source snapshot")
    accepted_verifier_definition_sha256 = _hash(_ACCEPTED_VERIFIER_DEFINITION_BYTES)
    process_authentication_values = (
        process_handoff_bytes,
        producer_definition_path,
        producer_definition_bytes,
        handoff_decoder_definition_path,
        handoff_decoder_definition_bytes,
    )
    if any(value is not None for value in process_authentication_values) and not all(
        value is not None for value in process_authentication_values
    ):
        _fail("input", "complete process-handoff authentication tuple required")
    authenticated_artifacts: dict[str, bytes] = {}
    authenticated_process_artifacts: dict[str, bytes] = {}
    authenticated_build_artifacts: dict[str, bytes] = {}
    trusted_release_authority: _TrustedReleaseAuthority | None = None
    authenticated_handoff_raw: bytes | None = None
    if all(value is not None for value in process_authentication_values):
        authenticated_handoff_raw = _snapshot_bytes(
            process_handoff_bytes, "process_handoff_bytes"
        )
        producer_path = _archive_path(
            producer_definition_path, "producer_definition_path"
        )
        producer_raw = _snapshot_bytes(
            producer_definition_bytes, "producer_definition_bytes"
        )
        handoff_decoder_path = _archive_path(
            handoff_decoder_definition_path,
            "handoff_decoder_definition_path",
        )
        handoff_decoder_raw = _snapshot_bytes(
            handoff_decoder_definition_bytes,
            "handoff_decoder_definition_bytes",
        )
        if (
            producer_path != ACCEPTED_PRODUCER_DEFINITION_PATH
            or producer_raw != _PRODUCER_DEFINITION_BYTES
            or handoff_decoder_path != ACCEPTED_HANDOFF_DECODER_DEFINITION_PATH
            or handoff_decoder_raw != _HANDOFF_DECODER_DEFINITION_BYTES
        ):
            _fail(
                "input",
                "process-handoff producer/decoder path or bytes differ from internal source snapshots",
            )
        authenticated_process_artifacts = {
            ACCEPTED_PROCESS_HANDOFF_PATH: authenticated_handoff_raw,
            producer_path: producer_raw,
            handoff_decoder_path: handoff_decoder_raw,
        }
        _merge_authenticated_artifacts(
            authenticated_artifacts,
            authenticated_process_artifacts,
            "process-handoff authentication",
        )

    build_authentication_values = (
        build_evidence_definition_path,
        build_evidence_definition_bytes,
        build_evidence_bytes,
        build_evidence_artifact_bytes,
    )
    if any(value is not None for value in build_authentication_values) and not all(
        value is not None for value in build_authentication_values
    ):
        _fail("input", "complete measured build-evidence authentication tuple required")
    authenticated_build_evidence_raw: bytes | None = None
    authenticated_build_evidence_artifacts: dict[str, bytes] = {}
    if all(value is not None for value in build_authentication_values):
        build_definition_path = _archive_path(
            build_evidence_definition_path, "build_evidence_definition_path"
        )
        build_definition_raw = _snapshot_bytes(
            build_evidence_definition_bytes, "build_evidence_definition_bytes"
        )
        if (
            build_definition_path != ACCEPTED_BUILD_EVIDENCE_DEFINITION_PATH
            or build_definition_raw != _BUILD_EVIDENCE_DEFINITION_BYTES
        ):
            _fail(
                "input",
                "build-evidence definition path or bytes differ from internal source snapshot",
            )
        authenticated_build_evidence_raw = _snapshot_bytes(
            build_evidence_bytes, "build_evidence_bytes"
        )
        try:
            authenticated_build_evidence_artifacts = {
                _archive_path(path, f"build_evidence_artifact_bytes[{path!r}]"):
                _snapshot_bytes(raw, f"build_evidence_artifact_bytes[{path!r}]")
                for path, raw in build_evidence_artifact_bytes.items()
            }
        except AttributeError:
            _fail("input", "build_evidence_artifact_bytes: mapping required")
        if not authenticated_build_evidence_artifacts:
            _fail("input", "build_evidence_artifact_bytes: nonempty mapping required")
        authenticated_build_artifacts = {
            build_definition_path: build_definition_raw,
            BUILD_EVIDENCE.EVIDENCE_ARCHIVE_PATH: authenticated_build_evidence_raw,
            **authenticated_build_evidence_artifacts,
        }
        if len(authenticated_build_artifacts) != len(authenticated_build_evidence_artifacts) + 2:
            _fail("cross_bindings", "build-evidence artifact paths must be distinct")
        _merge_authenticated_artifacts(
            authenticated_artifacts,
            authenticated_build_artifacts,
            "build-evidence authentication",
        )

    trusted_release_values = (
        trusted_release_authority_bytes,
        trusted_release_authority_sha256,
        trusted_release_source_bytes,
    )
    if any(value is not None for value in trusted_release_values) and not all(
        value is not None for value in trusted_release_values
    ):
        _fail("input", "complete trusted-release authority tuple required")
    if all(value is not None for value in trusted_release_values):
        trusted_release_authority = _decode_trusted_release_authority(
            trusted_release_authority_bytes,
            trusted_release_authority_sha256,
            trusted_release_source_bytes,
        )
    try:
        objects = {
            _archive_path(path, f"external_object_bytes[{path!r}]"): _snapshot_bytes(
                value, f"external_object_bytes[{path!r}]"
            )
            for path, value in external_object_bytes.items()
        }
    except AttributeError:
        _fail("input", "external_object_bytes: mapping required")

    container = _call("structural", WIRE.decode_container, raw)
    opening = container.opening_header
    trailer = container.trailer_body.header
    records = tuple(record for chunk in container.chunks for record in chunk.records)
    attempt_blockers: list[str] = []
    rows_by_kind = {
        row.object_kind: row for row in container.trailer_body.object_entries
    }
    expected_paths = {
        EXTERNAL_BY_KIND[kind][0] for kind in rows_by_kind
    }
    if set(objects) != expected_paths:
        _fail(
            "external_set",
            f"exact decoded object-path set required; missing={sorted(expected_paths - set(objects))!r} "
            f"extra={sorted(set(objects) - expected_paths)!r}",
        )
    reserved_chain_paths = {
        STATUS.PRE_FOOTER_PATH,
        STATUS.POST_CLOSE_PATH,
        STATUS.MANIFEST_PATH,
    }
    if stream_path in objects or stream_path in reserved_chain_paths:
        _fail("cross_bindings", "stream path aliases another archive role")
    if (
        verifier_path in objects
        or verifier_path == stream_path
        or verifier_path in reserved_chain_paths
    ):
        _fail("cross_bindings", "accepted verifier path aliases another archive role")
    if authenticated_artifacts:
        occupied = set(objects) | reserved_chain_paths | {stream_path}
        if set(authenticated_artifacts) & occupied:
            _fail("cross_bindings", "authenticated artifact aliases another archive role")
        verifier_artifact = authenticated_artifacts.get(verifier_path)
        if verifier_artifact is not None and verifier_artifact != verifier_definition_raw:
            _fail("cross_bindings", "authenticated verifier artifact differs from accepted verifier bytes")
        if authenticated_process_artifacts and len(authenticated_process_artifacts) != 3:
            _fail("cross_bindings", "authenticated process artifact paths must be distinct")
        if authenticated_build_artifacts and len(authenticated_build_artifacts) < 3:
            _fail("cross_bindings", "authenticated build-evidence path set is incomplete")

    parsed: dict[int, dict[str, Any]] = {}
    for kind in sorted(rows_by_kind):
        row = rows_by_kind[kind]
        path, flags = EXTERNAL_BY_KIND[kind]
        if row.format_id != 1 or row.schema_revision != 1 or row.object_flags != flags:
            _fail("external_binding", f"object kind {kind}: fixed metadata mismatch")
        object_raw = objects[path]
        if row.byte_count != len(object_raw):
            _fail("external_binding", f"object kind {kind}: byte count mismatch")
        if row.sha256.hex() != _hash(object_raw):
            _fail("external_binding", f"object kind {kind}: digest mismatch")
        parsed[kind] = _call("external_canonical", SEMANTIC.parse_canonical, object_raw)

    for kind, value in parsed.items():
        count = _call("external_semantics", _entry_count, kind, value)
        if rows_by_kind[kind].entry_count != count:
            _fail("external_binding", f"object kind {kind}: entry count mismatch")

    members, expected_bundle = _accepted_static_members()
    if 1 not in parsed:
        _fail("schema_bundle", "object kind 1 is mandatory")
    _call("schema_bundle", SEMANTIC.validate_bundle, objects[EXTERNAL_BY_KIND[1][0]], members)
    if objects[EXTERNAL_BY_KIND[1][0]] != expected_bundle:
        _fail("schema_bundle", "archive bundle bytes differ from accepted registered bundle")

    bundle_entries = {row["object_type"]: row for row in parsed[1]["entries"]}
    if opening.schema_bundle_definition_sha256.hex() != _hash(objects[EXTERNAL_BY_KIND[1][0]]):
        _fail("cross_bindings", "opening schema-bundle digest mismatch")
    if opening.callsite_catalog_definition_sha256.hex() != bundle_entries[5]["sha256"]:
        _fail("cross_bindings", "opening callsite-definition digest mismatch")

    if 2 in parsed:
        _call("external_semantics", SEMANTIC.validate_api_catalog, parsed[2], bundle_entries[3]["sha256"])
        if objects[EXTERNAL_BY_KIND[2][0]] != members[4]:
            _fail("cross_bindings", "external API bytes differ from bundle member 4")
        if opening.api_catalog_definition_sha256.hex() != bundle_entries[4]["sha256"]:
            _fail("cross_bindings", "opening API-definition digest mismatch")

    phase_ids = {
        row["phase_id"] for row in SEMANTIC.parse_canonical(members[7])["entries"]
    }
    if 3 in parsed:
        _call(
            "external_semantics",
            SEMANTIC.validate_context_catalog,
            parsed[3],
            context_schema_sha256=bundle_entries[6]["sha256"],
            phase_ids=phase_ids,
        )
        if any(
            row["process_generation"] != opening.process_generation
            for row in parsed[3]["entries"]
        ):
            _fail("cross_bindings", "context catalog contains another generation")
    build: Mapping[str, Any] | None = None
    run: Mapping[str, Any] | None = None
    if 8 in parsed:
        build = _call(
            "coverage",
            COVERAGE.validate_build_coverage_bytes,
            objects[EXTERNAL_BY_KIND[8][0]],
            api_definition_sha256=bundle_entries[4]["sha256"],
        )
        if (
            build["policy"]["sha256"] != bundle_entries[9]["sha256"]
            or build["schema"]["sha256"] != bundle_entries[10]["sha256"]
        ):
            _fail("coverage", "build coverage definition/bundle relation mismatch")
        _validate_primary_build(opening, build)
    if 9 in parsed:
        if build is None:
            _fail("coverage", "run coverage requires bound build coverage")
        run = _call(
            "coverage",
            COVERAGE.validate_run_coverage_bytes,
            objects[EXTERNAL_BY_KIND[9][0]],
            build_coverage=build,
            api_definition_sha256=bundle_entries[4]["sha256"],
            expected_configuration_instance_sha256=(
                opening.configuration_instance_sha256.hex()
            ),
        )
        if (
            run["policy"]["sha256"] != bundle_entries[9]["sha256"]
            or run["schema"]["sha256"] != bundle_entries[10]["sha256"]
        ):
            _fail("coverage", "run coverage definition/bundle relation mismatch")
        _validate_run_opening(opening, run)

    config: Mapping[str, Any] | None = None
    if 10 in parsed:
        config = _call(
            "configuration",
            CONFIG.validate_effective_configuration_bytes,
            objects[EXTERNAL_BY_KIND[10][0]],
        )
        if config["schema"]["sha256"] != bundle_entries[11]["sha256"]:
            _fail("configuration", "effective-config schema/bundle relation mismatch")
        config_binding = _call(
            "configuration", CONFIG.wire_object_binding_fields, config
        )
        for field, expected in config_binding.items():
            if getattr(rows_by_kind[10], field) != expected:
                _fail("configuration", f"kind-10 wire binding mismatch: {field}")
        _call(
            "configuration",
            CONFIG.validate_pre_active_bindings,
            config,
            opening_configuration_sha256=opening.configuration_instance_sha256.hex(),
            opening_configured_thread_capacity=opening.configured_thread_capacity,
            opening_role_kind=opening.role_kind,
            opening_scope_kind=opening.scope_kind,
            run_coverage=run,
        )
        if config["mode_id"] == 3 and (
            config["sample_seed_provenance_id"], config["sample_seed_status_id"]
        ) != (1, 1):
            attempt_blockers.append(
                "synthetic A03 seed validates mechanics only and is not scientifically admissible"
            )
        else:
            _call("configuration", CONFIG.validate_sample_seed_admissibility, config)
        if build is not None and run is not None:
            _call(
                "configuration",
                CONFIG.validate_module_selection_bindings,
                config,
                build_coverage=build,
                run_coverage=run,
            )
    build_evidence_projection = False
    if authenticated_build_evidence_raw is not None:
        if build is None or config is None:
            _fail(
                "build_evidence",
                "authenticated build evidence requires build coverage and effective configuration",
            )
        selection_values = {
            row["key"]: row["value"] for row in config["selection_values"]
        }
        if selection_values.get("build_evidence_sha256") != _hash(
            authenticated_build_evidence_raw
        ):
            _fail(
                "build_evidence",
                "effective configuration does not bind the exact build-evidence digest",
            )
        derived_build = _call(
            "build_evidence",
            BUILD_EVIDENCE.validate_build_evidence_bytes,
            authenticated_build_evidence_raw,
            authenticated_build_evidence_artifacts,
            objects[EXTERNAL_BY_KIND[8][0]],
            api_definition_sha256=bundle_entries[4]["sha256"],
        )
        if derived_build != build:
            _fail("build_evidence", "derived build object differs after exact validation")
        build_evidence_projection = True

    trusted_release_projection = False
    if trusted_release_authority is not None:
        if build is None or config is None:
            _fail(
                "trusted_release_authority",
                "authority requires measured build coverage and effective configuration",
            )
        if build.get("evidence_origin_id") != 1 or build.get("verdict_id") != 1:
            _fail(
                "trusted_release_authority",
                "authority requires measured complete build coverage",
            )
        _validate_trusted_release_git(trusted_release_authority, build)
        selection_values = {
            row["key"]: row["value"] for row in config["selection_values"]
        }
        if (
            selection_values.get("trusted_release_authority_sha256")
            != trusted_release_authority.expected_sha256
        ):
            _fail(
                "trusted_release_authority",
                "effective configuration does not bind the external authority digest",
            )
        authority_artifacts = trusted_release_authority.artifacts
        occupied = set(objects) | reserved_chain_paths | {stream_path}
        if set(authority_artifacts) & occupied:
            _fail("cross_bindings", "trusted-release artifact aliases another archive role")
        verifier_artifact = authority_artifacts.get(verifier_path)
        if verifier_artifact is not None and verifier_artifact != verifier_definition_raw:
            _fail("cross_bindings", "trusted-release verifier bytes differ from accepted verifier bytes")
        allowed_release_overlaps = {
            ACCEPTED_PRODUCER_DEFINITION_PATH,
            ACCEPTED_HANDOFF_DECODER_DEFINITION_PATH,
            ACCEPTED_BUILD_EVIDENCE_DEFINITION_PATH,
        }
        for path, artifact_raw in authority_artifacts.items():
            previous = authenticated_artifacts.get(path)
            if previous is None:
                continue
            if path not in allowed_release_overlaps or previous != artifact_raw:
                _fail(
                    "cross_bindings",
                    "trusted-release source collides with another authenticated artifact",
                )
        _merge_authenticated_artifacts(
            authenticated_artifacts,
            authority_artifacts,
            "trusted-release authority",
        )
        trusted_release_projection = True

    complete_terminal = trailer.lifecycle_state == STATUS.LIFECYCLE_COMPLETE
    if config is None and complete_terminal:
        _fail("configuration", "effective configuration object is required")
    runtime_population_state = (
        RUNTIME.POPULATION_COMPLETE
        if complete_terminal
        else RUNTIME.POPULATION_NEGATIVE_PARTIAL
    )
    if complete_terminal and not {5, 6, 7}.issubset(parsed):
        _fail("runtime_catalog", "COMPLETE requires thread, module, and clock catalogs")

    module_rows: Mapping[tuple[int, int], Mapping[str, Any]] = {}
    if 6 in parsed:
        _module, module_rows = _call(
            "runtime_catalog",
            RUNTIME.validate_module_catalog_bytes,
            objects[EXTERNAL_BY_KIND[6][0]],
            expected_process_generation=opening.process_generation,
            population_state=runtime_population_state,
            lifecycle_state=trailer.lifecycle_state,
        )

    thread_population: Any | None = None
    if 5 in parsed and config is not None:
        _thread, thread_population = _call(
            "runtime_catalog",
            RUNTIME.validate_thread_catalog_bytes,
            objects[EXTERNAL_BY_KIND[5][0]],
            expected_process_generation=opening.process_generation,
            configured_thread_capacity=opening.configured_thread_capacity,
            record_count=trailer.record_count,
            mode_id=config["mode_id"],
            population_state=runtime_population_state,
            lifecycle_state=trailer.lifecycle_state,
        )
    elif 5 in parsed:
        attempt_blockers.append(
            "thread catalog could not be mode-bound because effective configuration is absent"
        )

    callsite_keys: set[tuple[int, int]] = set()
    callsite_module_keys: set[tuple[int, int]] = set()
    if 4 in parsed and config is not None:
        _callsite, callsite_keys = _call(
            "callsite",
            CALLSITE.validate_callsite_catalog_bytes,
            objects[EXTERNAL_BY_KIND[4][0]],
            definition_sha256=bundle_entries[5]["sha256"],
            expected_process_generation=opening.process_generation,
            mode_id=config["mode_id"],
            module_keys=module_rows.keys(),
        )
        callsite_module_keys = {
            (row["module_generation"], row["module_id"])
            for row in _callsite["entries"]
        }
        _call(
            "callsite",
            CALLSITE.reconcile_records,
            callsite_keys,
            process_generation=opening.process_generation,
            mode_id=config["mode_id"],
            records=records,
            complete=complete_terminal,
        )
    elif 4 in parsed:
        attempt_blockers.append(
            "callsite catalog could not be mode-bound because effective configuration is absent"
        )

    if module_rows and run is not None:
        _call(
            "runtime_catalog",
            RUNTIME.reconcile_module_relations,
            module_rows,
            callsite_module_keys=callsite_module_keys,
            run_module_population=run["module_population"],
        )
    elif module_rows:
        attempt_blockers.append(
            "module catalog could not be reconciled because run coverage is absent"
        )

    clock_row: Mapping[str, Any] | None = None
    if 7 in parsed:
        architecture_id = (
            build["architecture_id"]
            if build is not None
            else {1: 1, 2: 2}[opening.clock_kind]
        )
        if build is None:
            attempt_blockers.append(
                "clock catalog architecture could not be cross-bound to absent build coverage"
            )
        opening_clock = {
            field: getattr(opening, field) for field in RUNTIME.OPENING_CLOCK_FIELDS
        }
        _clock, clock_row = _call(
            "runtime_catalog",
            RUNTIME.validate_clock_catalog_bytes,
            objects[EXTERNAL_BY_KIND[7][0]],
            opening_identity=opening_clock,
            architecture_id=architecture_id,
            final_counter=trailer.final_counter,
            final_monotonic_raw_ns=trailer.final_monotonic_raw_ns,
            final_realtime_unix_ns=trailer.final_realtime_unix_ns,
            counter_invalid_observed=bool(trailer.terminal_flags & (1 << 15)),
            realtime_discontinuity_observed=bool(
                trailer.terminal_flags & (1 << 14)
            ),
        )

    if thread_population is not None:
        unresolved_threads = _call(
            "runtime_catalog",
            RUNTIME.reconcile_thread_records,
            thread_population,
            process_generation=opening.process_generation,
            records=records,
        )
        if unresolved_threads:
            attempt_blockers.append(
                f"negative thread population retains {len(unresolved_threads)} unresolved record references"
            )
    if clock_row is not None:
        _call(
            "runtime_catalog",
            RUNTIME.reconcile_record_counters,
            clock_row,
            final_counter=trailer.final_counter,
            records=records,
        )

    realloc_policy_id = None
    if 2 in parsed and 3 in parsed and config is not None:
        if any(record.api_id == 3 for record in records) or authenticated_handoff_raw is not None:
            if build is None or run is None:
                _fail("record_semantics", "realloc records require build and run coverage")
            resolution = _call(
                "record_semantics",
                COVERAGE.resolve_run_realloc_zero_policy,
                run,
                build_coverage=build,
                api_definition_sha256=bundle_entries[4]["sha256"],
                expected_configuration_instance_sha256=(
                    opening.configuration_instance_sha256.hex()
                ),
            )
            if resolution.status != "resolved":
                _fail("record_semantics", "active realloc policy is not resolved")
            realloc_policy_id = resolution.policy_id
        event_definition = SEMANTIC.parse_canonical(members[3])
        classifications, sequence_gaps = _validate_records(
            container,
            api=parsed[2],
            context=parsed[3],
            mode_id=config["mode_id"],
            event_semantics=event_definition,
            realloc_zero_policy_id=realloc_policy_id,
        )
        if sequence_gaps:
            attempt_blockers.append(
                f"exact-mode thread sequence population has {sequence_gaps} missing transactions"
            )
        _validate_a03_selection(
            records,
            classifications,
            process_generation=opening.process_generation,
            config=config,
        )
    elif records:
        attempt_blockers.append(
            "records could not be fully API/context/mode-bound because a required catalog is absent"
        )

    diagnostic_projection: Mapping[str, Any] | None = None
    if 11 in parsed and config is not None:
        _diagnostic, diagnostic_projection = _call(
            "diagnostics",
            DIAGNOSTICS.validate_diagnostics_bytes,
            objects[EXTERNAL_BY_KIND[11][0]],
            definition_sha256=bundle_entries[8]["sha256"],
            expected_mode_id=config["mode_id"],
            expected_process_generation=opening.process_generation,
            ready_thread_indices=(
                thread_population.ready_thread_indices
                if thread_population is not None
                else ()
            ),
            producer_population_complete=(
                authenticated_handoff_raw is not None
                or (
                    thread_population.population_complete
                    if thread_population is not None else False
                )
            ),
            unavailable_counter_keys=unavailable_counter_keys,
            registration_available=registration_available,
            writer_available=writer_available,
            aggregate_available=aggregate_available,
        )
        _call(
            "diagnostics",
            DIAGNOSTICS.reconcile_terminal,
            diagnostic_projection,
            terminal_entries=container.trailer_body.diagnostic_entries,
            terminal_flags=trailer.terminal_flags,
            diagnostic_loss_sum=trailer.diagnostic_loss_sum,
            diagnostic_bypass_sum=trailer.diagnostic_bypass_sum,
            saturated_counter_instances=trailer.saturated_counter_instances,
        )
    elif 11 in parsed:
        attempt_blockers.append(
            "diagnostics could not be mode-bound because effective configuration is absent"
        )

    if 12 not in parsed:
        _fail("terminal_binding", "pre-footer status object is mandatory")
    pre_footer_raw = objects[EXTERNAL_BY_KIND[12][0]]
    pre_footer = _call("terminal_binding", STATUS.parse_canonical, pre_footer_raw)
    _call("terminal_binding", STATUS.validate_pre_footer, pre_footer)
    _compare_pre_footer(pre_footer, trailer)
    if diagnostic_projection is not None and (
        pre_footer["diagnostic_population_partial"]
        != diagnostic_projection["population_partial"]
    ):
        _fail("terminal_binding", "pre-footer diagnostic partial state mismatch")

    authenticated_projection = False
    authenticated_handoff: Any | None = None
    authenticated_terminal_outcome: str | None = None
    if authenticated_handoff_raw is not None:
        if unavailable_counter_keys or not (
            registration_available and writer_available and aggregate_available
        ):
            _fail(
                "process_handoff",
                "authenticated terminal requires the full diagnostic population",
            )
        if diagnostic_projection is None:
            _fail("process_handoff", "authenticated diagnostics are unavailable")
        authenticated_handoff, authenticated_terminal_outcome = _validate_authenticated_process_handoff(
            authenticated_handoff_raw,
            raw,
            container,
            objects,
            parsed,
            module_rows,
            diagnostic_projection,
            pre_footer_raw,
            config,
            realloc_policy_id,
        )
        authenticated_projection = True
    if authenticated_handoff is not None and build_evidence_projection:
        _validate_authenticated_runtime_libc(
            authenticated_handoff,
            build,
            authenticated_build_evidence_artifacts,
        )

    stream_identity = _trusted_stream_identity(
        raw,
        container,
        stream_path=stream_path,
        pre_footer_row=rows_by_kind[12],
        accepted_verifier_definition_sha256=accepted_verifier_definition_sha256,
    )
    authentication_blockers = (
        (() if authenticated_projection else (PROCESS_AUTHENTICATION_BLOCKER,))
        + (() if build_evidence_projection else (BUILD_EVIDENCE_AUTHENTICATION_BLOCKER,))
        + (() if trusted_release_projection else (TRUSTED_RELEASE_AUTHORITY_BLOCKER,))
    )
    terminal_outcome = {
        STATUS.LIFECYCLE_COMPLETE: "complete",
        STATUS.LIFECYCLE_FAILED: "failed",
        STATUS.LIFECYCLE_INCOMPLETE: "incomplete",
    }[trailer.lifecycle_state]
    if (
        authenticated_terminal_outcome is not None
        and terminal_outcome != authenticated_terminal_outcome
    ):
        _fail("process_handoff", "authenticated terminal outcome projection mismatch")
    digest_rows = tuple(
        (kind, rows_by_kind[kind].sha256.hex()) for kind in sorted(rows_by_kind)
    )
    if terminal_outcome != "complete":
        return ArchiveVerificationResult(
            terminal_outcome=terminal_outcome,
            stream_sha256=_hash(raw),
            stream_bytes=len(raw),
            external_object_sha256=digest_rows,
            verified_stream=stream_identity,
            status_promotion=None,
            scientific_admission_complete=False,
            admission_blockers=("negative terminal outcome is retained and never promoted",)
            + tuple(attempt_blockers)
            + authentication_blockers,
        )

    if post_close_receipt_bytes is None or manifest_bytes is None:
        _fail("status_chain", "COMPLETE requires post-close receipt and manifest bytes")
    receipt_raw = _snapshot_bytes(post_close_receipt_bytes, "post_close_receipt_bytes")
    manifest_raw = _snapshot_bytes(manifest_bytes, "manifest_bytes")
    manifest_value = _call("status_chain", STATUS.parse_canonical, manifest_raw)
    manifest_entries = _call("status_chain", STATUS.validate_manifest, manifest_value)
    manifest_by_path = {entry.path: entry for entry in manifest_entries}
    expected_manifest_paths = set(objects) | set(authenticated_artifacts) | {
        STATUS.POST_CLOSE_PATH,
        stream_path,
        verifier_path,
    }
    if set(manifest_by_path) != expected_manifest_paths:
        _fail("status_chain", "manifest exact archive-role path set mismatch")
    expected_manifest = {
        path: (len(object_raw), _hash(object_raw)) for path, object_raw in objects.items()
    }
    for path, expected in expected_manifest.items():
        entry = manifest_by_path.get(path)
        if entry is None or (entry.byte_count, entry.sha256) != expected:
            _fail("status_chain", f"manifest external-object binding mismatch: {path}")
    for path, artifact_raw in authenticated_artifacts.items():
        entry = manifest_by_path.get(path)
        if entry is None or (entry.byte_count, entry.sha256) != (
            len(artifact_raw),
            _hash(artifact_raw),
        ):
            _fail("status_chain", f"manifest authenticated-artifact binding mismatch: {path}")
    verifier_entry = manifest_by_path.get(verifier_path)
    if verifier_entry is None or (
        verifier_entry.byte_count,
        verifier_entry.sha256,
    ) != (len(verifier_definition_raw), accepted_verifier_definition_sha256):
        _fail("status_chain", "manifest accepted-verifier binding mismatch")
    promotion = _call(
        "status_chain",
        STATUS.bind_complete_candidate,
        pre_footer_raw,
        receipt_raw,
        manifest_raw,
        verified_stream=stream_identity,
    )
    return ArchiveVerificationResult(
        terminal_outcome=terminal_outcome,
        stream_sha256=_hash(raw),
        stream_bytes=len(raw),
        external_object_sha256=digest_rows,
        verified_stream=stream_identity,
        status_promotion=promotion,
        scientific_admission_complete=(
            authenticated_projection
            and build_evidence_projection
            and trusted_release_projection
            and not attempt_blockers
        ),
        admission_blockers=tuple(attempt_blockers) + authentication_blockers,
    )


__all__ = [
    "ACCEPTED_BUILD_EVIDENCE_DEFINITION_PATH",
    "ACCEPTED_EVENT_CLASSIFIER_SHA256",
    "ACCEPTED_HANDOFF_DECODER_DEFINITION_PATH",
    "ACCEPTED_PROCESS_HANDOFF_PATH",
    "ACCEPTED_PRODUCER_DEFINITION_PATH",
    "ACCEPTED_MEMBER_SHA256",
    "ACCEPTED_VERIFIER_DEFINITION_PATH",
    "ACCEPTED_SCHEMA_BUNDLE_SHA256",
    "ArchiveVerificationError",
    "BUILD_EVIDENCE_AUTHENTICATION_BLOCKER",
    "ArchiveVerificationResult",
    "PROCESS_AUTHENTICATION_BLOCKER",
    "SCIENTIFIC_ADMISSION_BLOCKERS",
    "TRUSTED_RELEASE_AUTHORITY_BLOCKER",
    "TRUSTED_RELEASE_AUTHORITY_PATH",
    "TRUSTED_RELEASE_AUTHORITY_SCHEMA",
    "TRUSTED_RELEASE_SOURCE_PATHS",
    "accepted_build_evidence_definition_bytes",
    "accepted_event_classifier_definition_bytes",
    "accepted_schema_bundle_bytes",
    "accepted_trusted_release_source_bytes",
    "accepted_verifier_definition_bytes",
    "make_trusted_release_authority_bytes",
    "validate_trusted_release_authority",
    "validate_handoff_runtime_configuration",
    "verify_archive_candidate",
]
