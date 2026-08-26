#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-CSSL-1.0
"""Fail-closed disabled-mode graph-absence gate for OAI memory profiler R0.

Exit zero is bounded structural/content evidence only: common capture origin,
capture status/CWD, and final-ELF absence require independent external evidence.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import os
import re
import shlex
import stat
import sys
import tempfile
from pathlib import Path
from typing import BinaryIO, Callable, Sequence
from unittest import mock


SCHEMA = "oai.memprof.r0.absence-result/v11"
SELF_TEST_SCHEMA = "oai.memprof.r0.absence-self-test/v18"
CATALOG_VERSION = "oai-memprof-r0-absence-catalog/19"
CATALOG_SERIALIZATION_DOMAIN = "oai.memprof.r0.absence-catalog"
CATALOG_SERIALIZATION_VERSION = "length-prefixed-fields/v1"

DEFAULT_MAX_EVIDENCE_BYTES = 256 * 1024 * 1024
HARD_MAX_EVIDENCE_BYTES = 512 * 1024 * 1024
MAX_LINE_BYTES = 4 * 1024 * 1024
READ_CHUNK_BYTES = 64 * 1024
MAX_REPORTED_MATCHES = 128
MAX_MATCH_TEXT_CHARS = 160
MAX_DIAGNOSTIC_CHARS = 512
MAX_PATH_CHARS = 4096
MAX_JSON_BYTES = 64 * 1024

APIS = (
    "malloc",
    "calloc",
    "realloc",
    "free",
    "reallocarray",
    "aligned_alloc",
    "posix_memalign",
    "memalign",
    "valloc",
    "pvalloc",
    "strdup",
    "strndup",
)
R0_FINAL_TARGETS = ("nr-softmodem", "nr-uesoftmodem")

CATALOG_SCAN_MARKERS = (
    "memprof",
    "r0_",
    "__wrap_",
    "__real_",
    "profiling/memory",
    "profiling.memory",
    "compare_literal",
    "event_record_v1_literal",
    "chunk_header_v1.hex",
    "diagnostic_total_entry_v1.hex",
    "event_record_v1.hex",
    "event_total_entry_v1.hex",
    "footer_preimage_v1.hex",
    "object_binding_entries_v1.hex",
    "opening_header_v1.hex",
    "active_stream_opening_v1.hex",
    "process_handoff_v1.hex",
    "trailer_header_v1.hex",
)
WRAP_SCAN_MARKERS = ("wrap", "--w", "-wr")
WRAP_QUOTING_CHARACTERS = frozenset(map(chr, (92, 39, 34)))
WRAP_QUOTING_TRANSLATION = str.maketrans({92: None, 39: None, 34: None})
INVALID_LINE_CHARACTER = re.compile(r"[\x00-\x08\x0a-\x0c\x0e-\x1f\x7f]|\r(?=.)")

GNU_LD_REFERENCE_VERSION = "GNU ld 2.42"
GNU_LD_WRAP_GRAMMAR_VERSION = "oai.memprof.gnu-ld-wrap-text-grammar/v2"
GNU_LD_WRAP_OPTION_NAMES = ("wr", "wra", "wrap")
GNU_LD_WRAP_CLAIM = "frozen_textual_grammar_only"
GNU_LD_WRAP_EXPANSION_CLAIM = "shell_expansion_constructed_spellings_outside_grammar"
GNU_LD_WRAP_EXPANSION_EXAMPLE = r"--wr${EMPTY}ap=malloc"
FINAL_LINK_GRAMMAR_VERSION = "oai.memprof.r0.final-link-text-grammar/v3"
ROOT_GRAMMAR_VERSION = "oai.memprof.r0.declared-root-grammar/v1"
FINAL_ELF_REQUIREMENT = "independent_final_elf_validation_required"
ABSENCE_SCOPE_CLAIM = "bounded_structural_content_only"
CONTENT_IDENTITY_CLAIM = "self_hashed_content_only"
CAPTURE_ORIGIN_CLAIM = "external_terminal_ledger_required"
EXIT_ZERO_COMMON_ORIGIN_CLAIM = "not_common_origin_proof"
RELATIVE_OUTPUT_CLAIM = "bare_basename_external_cwd_premise_required"
FINAL_LINK_DRIVER_CLAIM = "exact_textual_usr_bin_cxx_only"
FINAL_LINK_SCAFFOLD_CLAIM = "exact_cmake_ninja_colon_guard_only"
FINAL_OUTPUT_CONTAINMENT_CLAIM = "lexical_bare_target_basename_only"
FINAL_OUTPUT_IDENTITY_CLAIM = "external_final_file_validation_required"
ROOT_DOMAIN_CLAIM = "canonical_ascii_path_components_only"
ROOT_COMPONENT_EXPRESSION = r"[A-Za-z0-9._+-]+"
FINAL_LINK_ARTIFACT_COMPONENT_EXPRESSION = r"[A-Za-z0-9_.+@=-]+"
FINAL_LINK_LIBRARY_SUFFIX_EXPRESSION = r"(?:\.a|\.so(?:\.[0-9]+)*)"
RAW_OUTPUT_TRAILING_WRAPPER_CHARS = "\"'`(){}"
MALFORMED_RAW_OUTPUT_EXPRESSIONS = (
    r"(?:^|[\t ;&|<>])-o[\t ]+(?P<value>[^\t ;&|<>]+)",
    r"(?:^|[\t ;&|<>])-o(?P<value>[^\t ;&|<>]+)",
    r"(?:^|[\t ;&|<>])--output(?:=|[\t ]+)(?P<value>[^\t ;&|<>]+)",
    r"(?:^|[\t ;&|<>])-Wl,(?:-o,|--output=)(?P<value>[^\t ;&|<>]+)",
    r"(?:^|[\t ;&|<>])-Wl,(?:-o|--output)[\t ]+(?P<value>[^\t ;&|<>]+)",
    r"(?:^|[\t ;&|<>])-Xlinker[\t ]+(?:-o|--output)"
    r"(?:[\t ]+-Xlinker)?[\t ]+(?P<value>[^\t ;&|<>]+)",
)
R0_FINAL_LINK_DRIVER = "/usr/bin/c++"
FINAL_LINK_DEPENDENCY_OPTION_TEMPLATE = "-Wl,--dependency-file=CMakeFiles/{target}.dir/link.d"
FINAL_LINK_PRE_OPTION_EXPRESSIONS = (
    r"-D[A-Za-z_][A-Za-z0-9_]*(?:=.*)?",
    r"-O[A-Za-z0-9_.+-]+",
    r"-W(?!l,)[A-Za-z0-9_.+,=-]+",
    r"-f[A-Za-z0-9_.+-]+(?:=.*)?",
    r"-g[A-Za-z0-9_.+-]*",
    r"-m[A-Za-z0-9_.+-]+(?:=.*)?",
    r"-l(?:gcc|rt)",
    r"-pipe",
    r"-rdynamic",
)
COMPILER_NON_LINK_OPTIONS = (
    "--help",
    "--target-help",
    "--version",
    "-###",
    "-E",
    "-M",
    "-MM",
    "-S",
    "-c",
    "-fsyntax-only",
    "-r",
    "-shared",
)
COMPILER_NON_LINK_PREFIXES = (
    "--help=",
    "-dumpmachine",
    "-dumpversion",
    "-dumpfullversion",
    "-dumpspecs",
    "-flinker-output=rel",
    "-flinker-output=nolto-rel",
    "-print-",
    "--print-",
)

PRODUCTION_TARGETS = (
    "oai_memprof_runtime",
    "oai_memprof_wrap_c",
    "oai_memprof_wire",
    "oai_memprof_container_wire",
    "oai_memprof_active_core",
    "oai_memprof_active_runtime",
    "oai_memprof_active_wrap_c",
    "oai_memprof_stream_writer",
    "oai_memprof_stream_finalizer",
    "oai_memprof_clock",
    "oai_memprof_process_handoff",
    "oai_memprof_process_session",
    "oai_memprof_softmodem_session",
    "oai_memprof_archive_append",
)

PRODUCTION_OUTPUTS = (
    "liboai_memprof_runtime.so",
    "liboai_memprof_runtime.so.1",
    "liboai_memprof_runtime.so.1.0.0",
    "liboai_memprof_wrap_c.a",
    "liboai_memprof_wire.a",
    "liboai_memprof_container_wire.a",
    "liboai_memprof_active_core.a",
    "liboai_memprof_active_runtime.so",
    "liboai_memprof_active_runtime.so.1",
    "liboai_memprof_active_runtime.so.1.0.0",
    "liboai_memprof_active_wrap_c.a",
    "liboai_memprof_stream_writer.a",
    "liboai_memprof_stream_finalizer.a",
    "liboai_memprof_clock.a",
    "liboai_memprof_process_handoff.a",
    "liboai_memprof_process_session.a",
    "liboai_memprof_softmodem_session.a",
    "liboai_memprof_active_runtime.map",
    "dfts.oai-memprof.map",
    "ldpc.oai-memprof.map",
    "ldpc_orig.oai-memprof.map",
    "nr-softmodem.oai-memprof.map",
    "nr-uesoftmodem.oai-memprof.map",
    "oai_usrpdevif.oai-memprof.map",
    "params_libconfig.oai-memprof.map",
    "oai_memprof_archive_append",
)

PRODUCTION_SOURCES = (
    "oai_memprof_runtime.c",
    "oai_memprof_runtime.map",
    "oai_memprof_runtime_abi.h",
    "oai_memprof_wire.c",
    "oai_memprof_wire.h",
    "oai_memprof_wire.py",
    "oai_memprof_wrap_internal.h",
    "oai_memprof_wrap_malloc.c",
    "oai_memprof_wrap_calloc.c",
    "oai_memprof_wrap_realloc.c",
    "oai_memprof_wrap_free.c",
    "oai_memprof_container_wire.c",
    "oai_memprof_container_wire.h",
    "oai_memprof_container_wire.py",
    "oai_memprof_active_core.c",
    "oai_memprof_active_core.h",
    "oai_memprof_active_runtime.c",
    "oai_memprof_active_runtime.map",
    "oai_memprof_active_runtime_abi.h",
    "oai_memprof_active_wrap_internal.h",
    "oai_memprof_active_wrap_malloc.c",
    "oai_memprof_active_wrap_calloc.c",
    "oai_memprof_active_wrap_realloc.c",
    "oai_memprof_active_wrap_free.c",
    "oai_memprof_active_wrap_reallocarray.c",
    "oai_memprof_active_wrap_aligned_alloc.c",
    "oai_memprof_active_wrap_posix_memalign.c",
    "oai_memprof_active_wrap_memalign.c",
    "oai_memprof_active_wrap_valloc.c",
    "oai_memprof_active_wrap_pvalloc.c",
    "oai_memprof_active_wrap_strdup.c",
    "oai_memprof_active_wrap_strndup.c",
    "oai_memprof_stream_writer.c",
    "oai_memprof_stream_writer.h",
    "oai_memprof_stream_finalizer.c",
    "oai_memprof_stream_finalizer.h",
    "oai_memprof_clock.c",
    "oai_memprof_clock.h",
    "oai_memprof_process_handoff.c",
    "oai_memprof_process_handoff.h",
    "oai_memprof_process_handoff.py",
    "oai_memprof_process_session.c",
    "oai_memprof_process_session.h",
    "oai_memprof_softmodem_session.c",
    "oai_memprof_softmodem_session.h",
    "oai_memprof_archive_append.c",
    "oai_memprof_archive_composer.py",
    "oai_memprof_build_evidence.py",
    "oai_memprof_trusted_release_authority.py",
    "oai_memprof_softmodem_launcher.py",
)

R0_SOURCE_AND_TOOL_NAMES = (
    "check_oai_memprof_r0_absence.py",
    "check_oai_memprof_r0_elf.py",
    "compare_literal.cmake",
    "event_record_v1_literal.hex",
    "r0_actual_fixture.h",
    "r0_actual_fixture_common.h",
    "r0_actual_fixture_dso.c",
    "r0_actual_fixture_exe.c",
    "r0_bounded_process.py",
    "r0_harness_common.py",
    "r0_raw_emit.c",
    "r0_raw_emit.h",
    "r0_scripted_backend.c",
    "r0_scripted_backend.h",
    "r0_scripted_oracle.c",
    "r0_scripted_passthrough.c",
    "run_r0_actual_differential.py",
    "test_oai_memprof_wire.c",
    "test_oai_memprof_wire.py",
    "test_r0_bounded_process.py",
    "test_r0_harness.py",
    "validate_r0_scripted_oracle.py",
    "chunk_header_v1.hex",
    "diagnostic_total_entry_v1.hex",
    "event_record_v1.hex",
    "event_total_entry_v1.hex",
    "footer_preimage_v1.hex",
    "object_binding_entries_v1.hex",
    "opening_header_v1.hex",
    "trailer_header_v1.hex",
    "test_oai_memprof_container_wire.c",
    "test_oai_memprof_container_wire.py",
    "active_stream_opening_v1.hex",
    "test_oai_memprof_active_core.c",
    "test_oai_memprof_active_wrappers.c",
    "test_oai_memprof_stream_writer.c",
    "test_oai_memprof_stream_finalizer.c",
    "validate_oai_memprof_stream_finalizer.py",
    "test_oai_memprof_clock.c",
    "process_handoff_v1.hex",
    "test_oai_memprof_process_handoff.c",
    "test_oai_memprof_process_handoff.py",
    "test_oai_memprof_process_session.c",
    "test_oai_memprof_softmodem_session.c",
    "test_oai_memprof_archive_producer.c",
    "test_oai_memprof_archive_composer.py",
    "test_oai_memprof_softmodem_launcher.py",
    "test_oai_memprof_trusted_release_authority.py",
)

R0_TARGET_AND_TEST_NAMES = (
    "test_oai_memprof_wire",
    "oai_memprof_wire_c",
    "oai_memprof_wire_cross_literal",
    "oai_memprof_wire_python",
    "test_oai_memprof_r0_scripted_a00",
    "test_oai_memprof_r0_scripted_a01",
    "test_oai_memprof_r0_mutant_duplicate_real",
    "test_oai_memprof_r0_mutant_operand",
    "test_oai_memprof_r0_mutant_errno",
    "test_oai_memprof_r0_mutant_result",
    "test_oai_memprof_r0_mutant_suppress_free_null",
    "test_oai_memprof_r0_mutant_context",
    "oai_memprof_r0_actual_exe_object",
    "oai_memprof_r0_actual_dso_object",
    "oai_memprof_r0_actual_dso_a00",
    "oai_memprof_r0_actual_dso_a01",
    "test_oai_memprof_r0_actual_a00",
    "test_oai_memprof_r0_actual_a01",
    "oai_memprof_r0_scripted_pair",
    "oai_memprof_r0_scripted_bounds",
    "oai_memprof_r0_mutation_duplicate_real",
    "oai_memprof_r0_mutation_operand",
    "oai_memprof_r0_mutation_errno",
    "oai_memprof_r0_mutation_result",
    "oai_memprof_r0_mutation_suppress_free_null",
    "oai_memprof_r0_mutation_context",
    "oai_memprof_r0_actual_a00",
    "oai_memprof_r0_actual_a01",
    "oai_memprof_r0_actual_differential",
    "oai_memprof_r0_elf_validator_selftest",
    "oai_memprof_r0_elf",
    "oai_memprof_r0_absence_validator_selftest",
    "oai_memprof_r0_harness_selftest",
    "oai_memprof_r0_a00_exe.map",
    "oai_memprof_r0_a01_exe.map",
    "oai_memprof_r0_a00_dso.map",
    "oai_memprof_r0_a01_dso.map",
    "liboai_memprof_r0_actual_dso_a00.so",
    "liboai_memprof_r0_actual_dso_a01.so",
    "test_oai_memprof_container_wire",
    "oai_memprof_container_wire_c",
    "oai_memprof_container_wire_python",
    "test_oai_memprof_active_core",
    "test_oai_memprof_active_wrappers",
    "test_oai_memprof_stream_writer",
    "oai_memprof_active_core_c",
    "oai_memprof_active_wrappers_c",
    "oai_memprof_stream_writer_positive_c",
    "oai_memprof_stream_writer_counters_c",
    "oai_memprof_stream_writer_timer_c",
    "oai_memprof_stream_writer_short_c",
    "oai_memprof_stream_writer_failure_c",
    "test_oai_memprof_stream_finalizer",
    "oai_memprof_stream_finalizer_positive_c",
    "oai_memprof_stream_finalizer_short_c",
    "oai_memprof_stream_finalizer_failure_c",
    "oai_memprof_stream_finalizer_corrupt_c",
    "oai_memprof_stream_finalizer_mismatch_c",
    "oai_memprof_stream_finalizer_identity_c",
    "oai_memprof_stream_finalizer_offline_c",
    "test_oai_memprof_clock",
    "oai_memprof_clock_c",
    "test_oai_memprof_process_handoff",
    "oai_memprof_process_handoff_c",
    "oai_memprof_process_handoff_python",
    "test_oai_memprof_process_session",
    "oai_memprof_process_session_positive_c",
    "oai_memprof_process_session_existing_c",
    "test_oai_memprof_softmodem_session",
    "oai_memprof_softmodem_session_disabled_c",
    "oai_memprof_softmodem_session_partial_c",
    "oai_memprof_softmodem_session_legacy-path_c",
    "oai_memprof_softmodem_session_role-mismatch_c",
    "oai_memprof_softmodem_session_positive_c",
    "oai_memprof_softmodem_session_sampled_c",
    "oai_memprof_softmodem_session_fd-roots-replaced_c",
    "oai_memprof_softmodem_session_configuration-mismatch_c",
    "oai_memprof_softmodem_session_insecure-streams_c",
    "oai_memprof_softmodem_launcher_python",
    "test_oai_memprof_archive_producer",
    "test_oai_memprof_archive_producer.map",
    "oai_memprof_archive_composer_python",
    "oai_memprof_trusted_release_authority_python",
)

REVIEWER_NAMESPACE_VECTORS = (
    ("namespace:upper:OAI_MEMPROF", "-DOAI_MEMPROF_CONTROL_PRESENT_OFF=1"),
    ("namespace:upper:OAI_MEMPROF", "-DOAI_MEMPROF_RUNTIME_ABI_VERSION=1"),
    ("namespace:upper:OAI_MEMPROF", "-DOAI_MEMPROF_EVENT_V1_WIRE_SIZE=128"),
    ("namespace:lower:oai_memprof", "U oai_memprof_event_v1_encode"),
    ("namespace:lower:oai_memprof", "U oai_memprof_event_v1_decode"),
    ("namespace:lower:oai_memprof", "oai_memprof_event_v1_t"),
    ("namespace:lower:oai_memprof", "oai_memprof_wire_status_t"),
    ("namespace:linker:-loai_memprof", "-loai_memprof_runtime"),
    ("namespace:linker:-loai_memprof", "-loai_memprof_wire"),
    ("namespace:linker:-loai_memprof", "-loai_memprof_wrap_c"),
    ("namespace:python:tools.profiling.memory",
     "tools.profiling.memory.tests.test_oai_memprof_wire"),
    ("namespace:upper:OAI_MEMPROF", "-DOAI_MEMPROF_CONTAINER_V1_FOOTER_SIZE=256"),
    ("namespace:lower:oai_memprof", "U oai_memprof_container_v1_footer_decode"),
    ("namespace:linker:-loai_memprof", "-loai_memprof_container_wire"),
    ("namespace:python:tools.profiling.memory",
     "tools.profiling.memory.tests.test_oai_memprof_container_wire"),
    ("namespace:lower:oai_memprof", "U oai_memprof_active_runtime_activate_v1"),
    ("namespace:linker:-loai_memprof", "-loai_memprof_active_runtime"),
    ("namespace:linker:-loai_memprof", "-loai_memprof_stream_writer"),
    ("namespace:linker:-loai_memprof", "-loai_memprof_stream_finalizer"),
    ("namespace:lower:oai_memprof", "U oai_memprof_stream_finalize_v1"),
    ("namespace:lower:oai_memprof", "U oai_memprof_stream_finalize_offline_v1"),
    ("namespace:lower:oai_memprof", "U oai_memprof_clock_sample_v1"),
    ("namespace:linker:-loai_memprof", "-loai_memprof_clock"),
    ("namespace:linker:-loai_memprof", "-loai_memprof_process_handoff"),
    ("namespace:linker:-loai_memprof", "-loai_memprof_process_session"),
    ("namespace:linker:-loai_memprof", "-loai_memprof_softmodem_session"),
    ("namespace:python:tools.profiling.memory",
     "tools.profiling.memory.tests.test_oai_memprof_process_handoff"),
    ("namespace:python:tools.profiling.memory",
     "tools.profiling.memory.tests.test_oai_memprof_softmodem_launcher"),
)

REVIEWER_WRAP_ESCAPE_VECTORS = (
    r"ld --wr\ap=malloc",
    r"cc -Wl,--wr\ap=malloc -o x",
)

ADJACENT_QUOTING_WRAP_VECTORS = (
    ("--w''rap=malloc", "--wrap=malloc"),
    ('--w""rap=malloc', "--wrap=malloc"),
    ("-w''rap=malloc", "-wrap=malloc"),
    ('-w""rap=malloc', "-wrap=malloc"),
    ("-Wl,--w''rap=malloc", "-Wl,--wrap=malloc"),
    ('-Wl,--w""rap=malloc', "-Wl,--wrap=malloc"),
    ("-Xlinker --w''rap=malloc", "-Xlinker --wrap=malloc"),
    ('-Xlinker --w""rap=malloc', "-Xlinker --wrap=malloc"),
)

R0_CMAKE_NAME_CORRESPONDENCE_VECTORS = (
    (
        "r0-name:oai_memprof_r0_harness_selftest",
        "add_test(NAME oai_memprof_r0_harness_selftest COMMAND test_r0_harness.py)",
    ),
)

EXPECTED_CATALOG_IDS_V19 = (
    'build-subtree-relative',
    'control-load:oai_memprof_control_load_v1',
    'control:oai_memprof_control_v1',
    'namespace:hyphen:oai-memprof',
    'namespace:lib:liboai_memprof',
    'namespace:linker:-loai_memprof',
    'namespace:lower:oai_memprof',
    'namespace:python:tools.profiling.memory',
    'namespace:upper:OAI_MEMPROF',
    'output:dfts.oai-memprof.map',
    'output:ldpc.oai-memprof.map',
    'output:ldpc_orig.oai-memprof.map',
    'output:liboai_memprof_active_core.a',
    'output:liboai_memprof_active_runtime.map',
    'output:liboai_memprof_active_runtime.so',
    'output:liboai_memprof_active_runtime.so.1',
    'output:liboai_memprof_active_runtime.so.1.0.0',
    'output:liboai_memprof_active_wrap_c.a',
    'output:liboai_memprof_clock.a',
    'output:liboai_memprof_container_wire.a',
    'output:liboai_memprof_process_handoff.a',
    'output:liboai_memprof_process_session.a',
    'output:liboai_memprof_runtime.so',
    'output:liboai_memprof_runtime.so.1',
    'output:liboai_memprof_runtime.so.1.0.0',
    'output:liboai_memprof_softmodem_session.a',
    'output:liboai_memprof_stream_finalizer.a',
    'output:liboai_memprof_stream_writer.a',
    'output:liboai_memprof_wire.a',
    'output:liboai_memprof_wrap_c.a',
    'output:nr-softmodem.oai-memprof.map',
    'output:nr-uesoftmodem.oai-memprof.map',
    'output:oai_memprof_archive_append',
    'output:oai_usrpdevif.oai-memprof.map',
    'output:params_libconfig.oai-memprof.map',
    'python-source-subtree-relative',
    'r0-cmake-namespace',
    'r0-file:active_stream_opening_v1.hex',
    'r0-file:check_oai_memprof_r0_absence.py',
    'r0-file:check_oai_memprof_r0_elf.py',
    'r0-file:chunk_header_v1.hex',
    'r0-file:compare_literal.cmake',
    'r0-file:diagnostic_total_entry_v1.hex',
    'r0-file:event_record_v1.hex',
    'r0-file:event_record_v1_literal.hex',
    'r0-file:event_total_entry_v1.hex',
    'r0-file:footer_preimage_v1.hex',
    'r0-file:object_binding_entries_v1.hex',
    'r0-file:opening_header_v1.hex',
    'r0-file:process_handoff_v1.hex',
    'r0-file:r0_actual_fixture.h',
    'r0-file:r0_actual_fixture_common.h',
    'r0-file:r0_actual_fixture_dso.c',
    'r0-file:r0_actual_fixture_exe.c',
    'r0-file:r0_bounded_process.py',
    'r0-file:r0_harness_common.py',
    'r0-file:r0_raw_emit.c',
    'r0-file:r0_raw_emit.h',
    'r0-file:r0_scripted_backend.c',
    'r0-file:r0_scripted_backend.h',
    'r0-file:r0_scripted_oracle.c',
    'r0-file:r0_scripted_passthrough.c',
    'r0-file:run_r0_actual_differential.py',
    'r0-file:test_oai_memprof_active_core.c',
    'r0-file:test_oai_memprof_active_wrappers.c',
    'r0-file:test_oai_memprof_archive_composer.py',
    'r0-file:test_oai_memprof_archive_producer.c',
    'r0-file:test_oai_memprof_clock.c',
    'r0-file:test_oai_memprof_container_wire.c',
    'r0-file:test_oai_memprof_container_wire.py',
    'r0-file:test_oai_memprof_process_handoff.c',
    'r0-file:test_oai_memprof_process_handoff.py',
    'r0-file:test_oai_memprof_process_session.c',
    'r0-file:test_oai_memprof_softmodem_launcher.py',
    'r0-file:test_oai_memprof_softmodem_session.c',
    'r0-file:test_oai_memprof_stream_finalizer.c',
    'r0-file:test_oai_memprof_stream_writer.c',
    'r0-file:test_oai_memprof_trusted_release_authority.py',
    'r0-file:test_oai_memprof_wire.c',
    'r0-file:test_oai_memprof_wire.py',
    'r0-file:test_r0_bounded_process.py',
    'r0-file:test_r0_harness.py',
    'r0-file:trailer_header_v1.hex',
    'r0-file:validate_oai_memprof_stream_finalizer.py',
    'r0-file:validate_r0_scripted_oracle.py',
    'r0-name:liboai_memprof_r0_actual_dso_a00.so',
    'r0-name:liboai_memprof_r0_actual_dso_a01.so',
    'r0-name:oai_memprof_active_core_c',
    'r0-name:oai_memprof_active_wrappers_c',
    'r0-name:oai_memprof_archive_composer_python',
    'r0-name:oai_memprof_clock_c',
    'r0-name:oai_memprof_container_wire_c',
    'r0-name:oai_memprof_container_wire_python',
    'r0-name:oai_memprof_process_handoff_c',
    'r0-name:oai_memprof_process_handoff_python',
    'r0-name:oai_memprof_process_session_existing_c',
    'r0-name:oai_memprof_process_session_positive_c',
    'r0-name:oai_memprof_r0_a00_dso.map',
    'r0-name:oai_memprof_r0_a00_exe.map',
    'r0-name:oai_memprof_r0_a01_dso.map',
    'r0-name:oai_memprof_r0_a01_exe.map',
    'r0-name:oai_memprof_r0_absence_validator_selftest',
    'r0-name:oai_memprof_r0_actual_a00',
    'r0-name:oai_memprof_r0_actual_a01',
    'r0-name:oai_memprof_r0_actual_differential',
    'r0-name:oai_memprof_r0_actual_dso_a00',
    'r0-name:oai_memprof_r0_actual_dso_a01',
    'r0-name:oai_memprof_r0_actual_dso_object',
    'r0-name:oai_memprof_r0_actual_exe_object',
    'r0-name:oai_memprof_r0_elf',
    'r0-name:oai_memprof_r0_elf_validator_selftest',
    'r0-name:oai_memprof_r0_harness_selftest',
    'r0-name:oai_memprof_r0_mutation_context',
    'r0-name:oai_memprof_r0_mutation_duplicate_real',
    'r0-name:oai_memprof_r0_mutation_errno',
    'r0-name:oai_memprof_r0_mutation_operand',
    'r0-name:oai_memprof_r0_mutation_result',
    'r0-name:oai_memprof_r0_mutation_suppress_free_null',
    'r0-name:oai_memprof_r0_scripted_bounds',
    'r0-name:oai_memprof_r0_scripted_pair',
    'r0-name:oai_memprof_softmodem_launcher_python',
    'r0-name:oai_memprof_softmodem_session_configuration-mismatch_c',
    'r0-name:oai_memprof_softmodem_session_disabled_c',
    'r0-name:oai_memprof_softmodem_session_fd-roots-replaced_c',
    'r0-name:oai_memprof_softmodem_session_insecure-streams_c',
    'r0-name:oai_memprof_softmodem_session_legacy-path_c',
    'r0-name:oai_memprof_softmodem_session_partial_c',
    'r0-name:oai_memprof_softmodem_session_positive_c',
    'r0-name:oai_memprof_softmodem_session_role-mismatch_c',
    'r0-name:oai_memprof_softmodem_session_sampled_c',
    'r0-name:oai_memprof_stream_finalizer_corrupt_c',
    'r0-name:oai_memprof_stream_finalizer_failure_c',
    'r0-name:oai_memprof_stream_finalizer_identity_c',
    'r0-name:oai_memprof_stream_finalizer_mismatch_c',
    'r0-name:oai_memprof_stream_finalizer_offline_c',
    'r0-name:oai_memprof_stream_finalizer_positive_c',
    'r0-name:oai_memprof_stream_finalizer_short_c',
    'r0-name:oai_memprof_stream_writer_counters_c',
    'r0-name:oai_memprof_stream_writer_failure_c',
    'r0-name:oai_memprof_stream_writer_positive_c',
    'r0-name:oai_memprof_stream_writer_short_c',
    'r0-name:oai_memprof_stream_writer_timer_c',
    'r0-name:oai_memprof_trusted_release_authority_python',
    'r0-name:oai_memprof_wire_c',
    'r0-name:oai_memprof_wire_cross_literal',
    'r0-name:oai_memprof_wire_python',
    'r0-name:test_oai_memprof_active_core',
    'r0-name:test_oai_memprof_active_wrappers',
    'r0-name:test_oai_memprof_archive_producer',
    'r0-name:test_oai_memprof_archive_producer.map',
    'r0-name:test_oai_memprof_clock',
    'r0-name:test_oai_memprof_container_wire',
    'r0-name:test_oai_memprof_process_handoff',
    'r0-name:test_oai_memprof_process_session',
    'r0-name:test_oai_memprof_r0_actual_a00',
    'r0-name:test_oai_memprof_r0_actual_a01',
    'r0-name:test_oai_memprof_r0_mutant_context',
    'r0-name:test_oai_memprof_r0_mutant_duplicate_real',
    'r0-name:test_oai_memprof_r0_mutant_errno',
    'r0-name:test_oai_memprof_r0_mutant_operand',
    'r0-name:test_oai_memprof_r0_mutant_result',
    'r0-name:test_oai_memprof_r0_mutant_suppress_free_null',
    'r0-name:test_oai_memprof_r0_scripted_a00',
    'r0-name:test_oai_memprof_r0_scripted_a01',
    'r0-name:test_oai_memprof_softmodem_session',
    'r0-name:test_oai_memprof_stream_finalizer',
    'r0-name:test_oai_memprof_stream_writer',
    'r0-name:test_oai_memprof_wire',
    'r0-optional-lib-namespace',
    'r0-target-namespace',
    'raw-schema:oai-memprof-r0-raw-v1',
    'real:aligned_alloc',
    'real:calloc',
    'real:free',
    'real:malloc',
    'real:memalign',
    'real:posix_memalign',
    'real:pvalloc',
    'real:realloc',
    'real:reallocarray',
    'real:strdup',
    'real:strndup',
    'real:valloc',
    'runtime-soname',
    'runtime-symbol-version',
    'source-subtree-relative',
    'source:oai_memprof_active_core.c',
    'source:oai_memprof_active_core.h',
    'source:oai_memprof_active_runtime.c',
    'source:oai_memprof_active_runtime.map',
    'source:oai_memprof_active_runtime_abi.h',
    'source:oai_memprof_active_wrap_aligned_alloc.c',
    'source:oai_memprof_active_wrap_calloc.c',
    'source:oai_memprof_active_wrap_free.c',
    'source:oai_memprof_active_wrap_internal.h',
    'source:oai_memprof_active_wrap_malloc.c',
    'source:oai_memprof_active_wrap_memalign.c',
    'source:oai_memprof_active_wrap_posix_memalign.c',
    'source:oai_memprof_active_wrap_pvalloc.c',
    'source:oai_memprof_active_wrap_realloc.c',
    'source:oai_memprof_active_wrap_reallocarray.c',
    'source:oai_memprof_active_wrap_strdup.c',
    'source:oai_memprof_active_wrap_strndup.c',
    'source:oai_memprof_active_wrap_valloc.c',
    'source:oai_memprof_archive_append.c',
    'source:oai_memprof_archive_composer.py',
    'source:oai_memprof_build_evidence.py',
    'source:oai_memprof_clock.c',
    'source:oai_memprof_clock.h',
    'source:oai_memprof_container_wire.c',
    'source:oai_memprof_container_wire.h',
    'source:oai_memprof_container_wire.py',
    'source:oai_memprof_process_handoff.c',
    'source:oai_memprof_process_handoff.h',
    'source:oai_memprof_process_handoff.py',
    'source:oai_memprof_process_session.c',
    'source:oai_memprof_process_session.h',
    'source:oai_memprof_runtime.c',
    'source:oai_memprof_runtime.map',
    'source:oai_memprof_runtime_abi.h',
    'source:oai_memprof_softmodem_launcher.py',
    'source:oai_memprof_softmodem_session.c',
    'source:oai_memprof_softmodem_session.h',
    'source:oai_memprof_stream_finalizer.c',
    'source:oai_memprof_stream_finalizer.h',
    'source:oai_memprof_stream_writer.c',
    'source:oai_memprof_stream_writer.h',
    'source:oai_memprof_trusted_release_authority.py',
    'source:oai_memprof_wire.c',
    'source:oai_memprof_wire.h',
    'source:oai_memprof_wire.py',
    'source:oai_memprof_wrap_calloc.c',
    'source:oai_memprof_wrap_free.c',
    'source:oai_memprof_wrap_internal.h',
    'source:oai_memprof_wrap_malloc.c',
    'source:oai_memprof_wrap_realloc.c',
    'target:oai_memprof_active_core',
    'target:oai_memprof_active_runtime',
    'target:oai_memprof_active_wrap_c',
    'target:oai_memprof_archive_append',
    'target:oai_memprof_clock',
    'target:oai_memprof_container_wire',
    'target:oai_memprof_process_handoff',
    'target:oai_memprof_process_session',
    'target:oai_memprof_runtime',
    'target:oai_memprof_softmodem_session',
    'target:oai_memprof_stream_finalizer',
    'target:oai_memprof_stream_writer',
    'target:oai_memprof_wire',
    'target:oai_memprof_wrap_c',
    'wire-namespace:OAI_MEMPROF_WIRE',
    'wrapper:aligned_alloc',
    'wrapper:calloc',
    'wrapper:free',
    'wrapper:malloc',
    'wrapper:memalign',
    'wrapper:posix_memalign',
    'wrapper:pvalloc',
    'wrapper:realloc',
    'wrapper:reallocarray',
    'wrapper:strdup',
    'wrapper:strndup',
    'wrapper:valloc',
)
EXPECTED_CATALOG_COUNT_V19 = 263
EXPECTED_CATALOG_DIGEST_V19 = '14a3d10c88e4da56f1b48d2732309471e7472c15e6aad0cea1db24ec5ffef270'


class AbsenceError(RuntimeError):
    """An invalid or unavailable input whose absence cannot be interpreted."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise AbsenceError(code, message)


def bounded_text(value: object, limit: int = MAX_DIAGNOSTIC_CHARS) -> str:
    text = str(value).replace("\x00", "\\0")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def normalized_root(value: str, role: str) -> str:
    require(0 < len(value) <= MAX_PATH_CHARS, "invalid_root", f"{role} has invalid length")
    require("\x00" not in value and "\n" not in value and "\r" not in value,
            "invalid_root", f"{role} contains a forbidden character")
    require(os.path.isabs(value), "invalid_root", f"{role} must be absolute: {value}")
    normalized = os.path.normpath(value)
    require(normalized != os.path.sep, "invalid_root", f"{role} must not be the filesystem root")
    require(value == normalized, "invalid_root",
            f"{role} must use one canonical lexical spelling")
    components = normalized.split(os.path.sep)[1:]
    require(bool(components) and all(
                component not in {".", ".."}
                and re.fullmatch(ROOT_COMPONENT_EXPRESSION, component) is not None
                for component in components
            ),
            "invalid_root",
            f"{role} must contain only slash-separated ASCII path components matching "
            f"{ROOT_COMPONENT_EXPRESSION}")
    return normalized


@dataclasses.dataclass(frozen=True)
class CatalogEntry:
    category: str
    catalog_id: str
    expression: str
    sample: str


@dataclasses.dataclass(frozen=True)
class Match:
    role: str
    line: int
    column: int
    category: str
    catalog_id: str
    matched: str
    line_sha256: str
    spelling: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "matched", bounded_text(self.matched, MAX_MATCH_TEXT_CHARS))
        if self.spelling is not None:
            object.__setattr__(self, "spelling", bounded_text(self.spelling, MAX_MATCH_TEXT_CHARS))

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "catalog_id": self.catalog_id,
            "category": self.category,
            "column": self.column,
            "line": self.line,
            "line_sha256": self.line_sha256,
            "matched": self.matched,
            "role": self.role,
        }
        if self.spelling is not None:
            result["spelling"] = self.spelling
        return result


@dataclasses.dataclass(frozen=True)
class EvidenceSummary:
    role: str
    path: str
    byte_count: int
    line_count: int
    sha256: str
    device: int
    inode: int

    def as_dict(self) -> dict[str, object]:
        return {
            "bytes": self.byte_count,
            "device": self.device,
            "inode": self.inode,
            "lines": self.line_count,
            "path": self.path,
            "role": self.role,
            "sha256": self.sha256,
        }


def exact_expression(value: str, name_chars: str = r"A-Za-z0-9_") -> str:
    return rf"(?<![{name_chars}]){re.escape(value)}(?![{name_chars}])"


def build_catalog() -> tuple[CatalogEntry, ...]:
    entries: list[CatalogEntry] = [
        CatalogEntry(
            "build_subtree",
            "build-subtree-relative",
            r"(?<![A-Za-z0-9_.+-])common/utils/memprof/CMakeFiles(?=/|(?:\\)+|$|[\s:$])",
            "common/utils/memprof/CMakeFiles/oai_memprof_runtime.dir",
        ),
        CatalogEntry(
            "source_subtree",
            "source-subtree-relative",
            r"(?<![A-Za-z0-9_.+-])common/utils/memprof(?=/|(?:\\)+|$|[\s:$])",
            "common/utils/memprof/oai_memprof_runtime.c",
        ),
        CatalogEntry(
            "source_subtree",
            "python-source-subtree-relative",
            r"(?<![A-Za-z0-9_.+-])tools/profiling/memory(?=/|(?:\\)+|$|[\s:$])",
            "tools/profiling/memory/oai_memprof_wire.py",
        ),
    ]

    for value in PRODUCTION_TARGETS:
        entries.append(CatalogEntry("production_target", f"target:{value}",
                                    exact_expression(value, r"A-Za-z0-9_.+-"),
                                    f"build {value}: phony"))
    for value in PRODUCTION_OUTPUTS:
        entries.append(CatalogEntry("production_output", f"output:{value}",
                                    exact_expression(value, r"A-Za-z0-9_.+-"), f"-o {value}"))
    for value in PRODUCTION_SOURCES:
        entries.append(CatalogEntry("production_source", f"source:{value}",
                                    exact_expression(value, r"A-Za-z0-9_.+-"), f"cc -c {value}"))
    for value in R0_SOURCE_AND_TOOL_NAMES:
        entries.append(CatalogEntry("r0_source_or_tool", f"r0-file:{value}",
                                    exact_expression(value, r"A-Za-z0-9_.+-"), f"python3 {value}"))
    for value in R0_TARGET_AND_TEST_NAMES:
        entries.append(CatalogEntry("r0_target_or_test", f"r0-name:{value}",
                                    exact_expression(value, r"A-Za-z0-9_.+-"),
                                    f"build {value}: phony"))

    entries.extend(
        (
            CatalogEntry("r0_namespace", "r0-optional-lib-namespace",
                         r"(?<![A-Za-z0-9_.+-])(?:lib)?"
                         r"oai_memprof_r0_[A-Za-z0-9_+-]{1,128}"
                         r"(?:\.a|\.so(?:\.[0-9]{1,20}){0,8})",
                         "-lo /tmp/liboai_memprof_r0_future_optional.so"),
            CatalogEntry("r0_namespace", "r0-target-namespace",
                         r"(?<![A-Za-z0-9_.+-])(?:test_)?"
                         r"oai_memprof_r0_[A-Za-z0-9_+-]{1,128}",
                         "build oai_memprof_r0_future_gate: phony"),
            CatalogEntry("r0_namespace", "r0-cmake-namespace",
                         r"(?<![A-Z0-9_])(?:-D)?OAI_MEMPROF_R0_[A-Z0-9_]{1,128}",
                         "FLAGS = -DOAI_MEMPROF_R0_FUTURE=1"),
        )
    )

    entries.extend(
        (
            CatalogEntry("runtime_abi", "runtime-soname",
                         exact_expression("liboai_memprof_runtime.so.1", r"A-Za-z0-9_.+-"),
                         "NEEDED liboai_memprof_runtime.so.1"),
            CatalogEntry("runtime_abi", "runtime-symbol-version",
                         exact_expression("OAI_MEMPROF_RUNTIME_1.0", r"A-Za-z0-9_.+-"),
                         "oai_memprof_control_v1@@OAI_MEMPROF_RUNTIME_1.0"),
            CatalogEntry("control_symbol", "control:oai_memprof_control_v1",
                         exact_expression("oai_memprof_control_v1"),
                         "U oai_memprof_control_v1"),
            CatalogEntry("control_symbol", "control-load:oai_memprof_control_load_v1",
                         exact_expression("oai_memprof_control_load_v1"),
                         "call oai_memprof_control_load_v1"),
            CatalogEntry("raw_schema", "raw-schema:oai-memprof-r0-raw-v1",
                         exact_expression("oai-memprof-r0-raw-v1", r"A-Za-z0-9_.+-"),
                         "META|schema|oai-memprof-r0-raw-v1"),
            CatalogEntry("wire_namespace", "wire-namespace:OAI_MEMPROF_WIRE",
                         r"(?<![A-Z0-9_])OAI_MEMPROF_WIRE_[A-Z0-9_]{1,128}",
                         "OAI_MEMPROF_WIRE_WRONG_SIZE"),
        )
    )
    entries.extend(
        (
            CatalogEntry("broad_namespace", "namespace:lower:oai_memprof",
                         r"(?<![A-Za-z0-9_])oai_memprof_[A-Za-z0-9_+-]{1,128}",
                         "U oai_memprof_future_api"),
            CatalogEntry("broad_namespace", "namespace:upper:OAI_MEMPROF",
                         r"(?<![A-Za-z0-9_])(?:-D)?OAI_MEMPROF_[A-Z0-9_]{1,128}",
                         "FLAGS = -DOAI_MEMPROF_FUTURE_ABI=1"),
            CatalogEntry("broad_namespace", "namespace:hyphen:oai-memprof",
                         r"(?<![A-Za-z0-9_-])oai-memprof-[A-Za-z0-9_.+-]{1,128}",
                         "META|schema|oai-memprof-future-v1"),
            CatalogEntry("broad_namespace", "namespace:lib:liboai_memprof",
                         r"(?<![A-Za-z0-9_])liboai_memprof_[A-Za-z0-9_+-]{1,128}"
                         r"(?:\.a|\.so(?:\.[0-9]{1,20}){0,8})?",
                         "NEEDED liboai_memprof_future.so.2"),
            CatalogEntry("broad_namespace", "namespace:linker:-loai_memprof",
                         r"(?<![A-Za-z0-9_-])-loai_memprof_[A-Za-z0-9_+-]{1,128}",
                         "cc x.o -loai_memprof_future -o x"),
            CatalogEntry("broad_namespace", "namespace:python:tools.profiling.memory",
                         r"(?<![A-Za-z0-9_.])tools\.profiling\.memory"
                         r"[A-Za-z0-9_.+-]{0,128}",
                         "python3 -m tools.profiling.memory.future"),
        )
    )
    for api in APIS:
        entries.append(CatalogEntry("wrapper_symbol", f"wrapper:{api}",
                                    exact_expression(f"__wrap_{api}"), f"U __wrap_{api}"))
        entries.append(CatalogEntry("real_symbol", f"real:{api}",
                                    exact_expression(f"__real_{api}"), f"U __real_{api}"))
    return tuple(entries)


def append_canonical_field(parts: list[bytes], name: str, value: str) -> None:
    for text in (name, value):
        encoded = text.encode("utf-8", errors="strict")
        parts.extend((len(encoded).to_bytes(8, "big"), encoded))


def canonical_catalog_serialization(entries: Sequence[CatalogEntry]) -> bytes:
    """Serialize the ordered catalog contract without separators or implicit fields."""

    parts: list[bytes] = []
    header = (
        ("domain", CATALOG_SERIALIZATION_DOMAIN),
        ("serialization_version", CATALOG_SERIALIZATION_VERSION),
        ("catalog_version", CATALOG_VERSION),
        ("gnu_ld_reference", GNU_LD_REFERENCE_VERSION),
        ("gnu_ld_wrap_grammar", GNU_LD_WRAP_GRAMMAR_VERSION),
        ("gnu_ld_wrap_claim", GNU_LD_WRAP_CLAIM),
        ("gnu_ld_wrap_expansion_claim", GNU_LD_WRAP_EXPANSION_CLAIM),
        ("gnu_ld_wrap_expansion_example", GNU_LD_WRAP_EXPANSION_EXAMPLE),
        ("final_link_grammar", FINAL_LINK_GRAMMAR_VERSION),
        ("root_grammar", ROOT_GRAMMAR_VERSION),
        ("final_elf_requirement", FINAL_ELF_REQUIREMENT),
        ("absence_scope_claim", ABSENCE_SCOPE_CLAIM),
        ("content_identity_claim", CONTENT_IDENTITY_CLAIM),
        ("capture_origin_claim", CAPTURE_ORIGIN_CLAIM),
        ("exit_zero_common_origin_claim", EXIT_ZERO_COMMON_ORIGIN_CLAIM),
        ("relative_output_claim", RELATIVE_OUTPUT_CLAIM),
        ("final_link_driver_claim", FINAL_LINK_DRIVER_CLAIM),
        ("final_link_scaffold_claim", FINAL_LINK_SCAFFOLD_CLAIM),
        ("final_output_containment_claim", FINAL_OUTPUT_CONTAINMENT_CLAIM),
        ("final_output_identity_claim", FINAL_OUTPUT_IDENTITY_CLAIM),
        ("root_domain_claim", ROOT_DOMAIN_CLAIM),
    )
    parts.append(len(header).to_bytes(8, "big"))
    for name, value in header:
        append_canonical_field(parts, name, value)

    parts.append(len(R0_FINAL_TARGETS).to_bytes(8, "big"))
    for target in R0_FINAL_TARGETS:
        append_canonical_field(parts, "r0_final_target", target)

    parts.append(len(GNU_LD_WRAP_OPTION_NAMES).to_bytes(8, "big"))
    for option_name in GNU_LD_WRAP_OPTION_NAMES:
        append_canonical_field(parts, "gnu_ld_wrap_option_name", option_name)

    append_canonical_field(parts, "r0_final_link_driver", R0_FINAL_LINK_DRIVER)
    append_canonical_field(parts, "root_component_expression", ROOT_COMPONENT_EXPRESSION)
    append_canonical_field(
        parts, "final_link_artifact_component_expression",
        FINAL_LINK_ARTIFACT_COMPONENT_EXPRESSION,
    )
    append_canonical_field(
        parts, "final_link_library_suffix_expression", FINAL_LINK_LIBRARY_SUFFIX_EXPRESSION
    )
    append_canonical_field(
        parts, "raw_output_trailing_wrapper_chars", RAW_OUTPUT_TRAILING_WRAPPER_CHARS
    )

    parts.append(len(MALFORMED_RAW_OUTPUT_EXPRESSIONS).to_bytes(8, "big"))
    for expression in MALFORMED_RAW_OUTPUT_EXPRESSIONS:
        append_canonical_field(parts, "malformed_raw_output_expression", expression)

    parts.append(len(FINAL_LINK_PRE_OPTION_EXPRESSIONS).to_bytes(8, "big"))
    for expression in FINAL_LINK_PRE_OPTION_EXPRESSIONS:
        append_canonical_field(parts, "final_link_pre_option_expression", expression)
    append_canonical_field(
        parts, "final_link_dependency_option_template",
        FINAL_LINK_DEPENDENCY_OPTION_TEMPLATE,
    )

    parts.append(len(COMPILER_NON_LINK_OPTIONS).to_bytes(8, "big"))
    for option in COMPILER_NON_LINK_OPTIONS:
        append_canonical_field(parts, "compiler_non_link_option", option)

    parts.append(len(COMPILER_NON_LINK_PREFIXES).to_bytes(8, "big"))
    for prefix in COMPILER_NON_LINK_PREFIXES:
        append_canonical_field(parts, "compiler_non_link_prefix", prefix)

    parts.append(len(ROOT_TOKEN_DELIMITERS).to_bytes(8, "big"))
    for delimiter in sorted(ROOT_TOKEN_DELIMITERS):
        append_canonical_field(parts, "root_token_delimiter", delimiter)

    parts.append(len(ROOT_JOINED_OPTION_PREFIXES).to_bytes(8, "big"))
    for prefix in ROOT_JOINED_OPTION_PREFIXES:
        append_canonical_field(parts, "root_joined_option_prefix", prefix)

    parts.append(len(REVIEWER_NAMESPACE_VECTORS).to_bytes(8, "big"))
    for catalog_id, vector in REVIEWER_NAMESPACE_VECTORS:
        append_canonical_field(parts, "reviewer_namespace_catalog_id", catalog_id)
        append_canonical_field(parts, "reviewer_namespace_vector", vector)

    parts.append(len(REVIEWER_WRAP_ESCAPE_VECTORS).to_bytes(8, "big"))
    for vector in REVIEWER_WRAP_ESCAPE_VECTORS:
        append_canonical_field(parts, "reviewer_wrap_escape_vector", vector)

    parts.append(len(ADJACENT_QUOTING_WRAP_VECTORS).to_bytes(8, "big"))
    for vector, normalized in ADJACENT_QUOTING_WRAP_VECTORS:
        append_canonical_field(parts, "adjacent_quoting_wrap_vector", vector)
        append_canonical_field(parts, "adjacent_quoting_wrap_normalized", normalized)

    parts.append(len(R0_CMAKE_NAME_CORRESPONDENCE_VECTORS).to_bytes(8, "big"))
    for catalog_id, vector in R0_CMAKE_NAME_CORRESPONDENCE_VECTORS:
        append_canonical_field(parts, "r0_cmake_name_catalog_id", catalog_id)
        append_canonical_field(parts, "r0_cmake_name_vector", vector)

    parts.append(len(entries).to_bytes(8, "big"))
    for ordinal, entry in enumerate(entries):
        parts.append(ordinal.to_bytes(8, "big"))
        append_canonical_field(parts, "catalog_id", entry.catalog_id)
        append_canonical_field(parts, "category", entry.category)
        append_canonical_field(parts, "expression", entry.expression)
        append_canonical_field(parts, "sample", entry.sample)
    return b"".join(parts)


def catalog_identity(entries: Sequence[CatalogEntry]) -> tuple[tuple[str, ...], str]:
    identifiers = tuple(sorted(entry.catalog_id for entry in entries))
    digest = hashlib.sha256(canonical_catalog_serialization(entries)).hexdigest()
    return identifiers, digest


def validate_catalog_freeze(entries: Sequence[CatalogEntry]) -> str:
    identifiers, digest = catalog_identity(entries)
    require(len(set(identifiers)) == len(identifiers), "catalog_duplicate_id",
            "the catalog contains duplicate semantic IDs")
    require(len(identifiers) == EXPECTED_CATALOG_COUNT_V19,
            "catalog_count_mismatch", "the v19 catalog count changed without a version update")
    require(identifiers == EXPECTED_CATALOG_IDS_V19,
            "catalog_id_set_mismatch", "the v19 catalog ID set changed without a version update")
    require(digest == EXPECTED_CATALOG_DIGEST_V19,
            "catalog_digest_mismatch", "the v19 catalog contract changed without a version update")
    return digest


class Catalog:
    def __init__(self, entries: Sequence[CatalogEntry], declared_roots: Sequence[str]):
        require(bool(entries), "invalid_catalog", "absence catalog must not be empty")
        require(len(declared_roots) == 2, "invalid_root_contract",
                "catalog requires ordered source and build roots")
        self.entries = tuple(entries)
        self.declared_roots = tuple(declared_roots)
        self.source_root, self.build_root = self.declared_roots
        grouped: dict[str, list[CatalogEntry]] = {}
        for entry in entries:
            grouped.setdefault(entry.expression, []).append(entry)
        self.groups = tuple(tuple(group) for group in grouped.values())
        alternatives = [
            f"(?P<E{index}>{group[0].expression})" for index, group in enumerate(self.groups)
        ]
        self.expression = re.compile("|".join(alternatives))

    def scan_line(
        self,
        role: str,
        line_number: int,
        line: str,
        retain_limit: int = MAX_REPORTED_MATCHES,
    ) -> tuple[list[Match], int]:
        require(0 <= retain_limit <= MAX_REPORTED_MATCHES, "invalid_match_limit",
                f"invalid retained-match limit: {retain_limit}")
        folded_line = line.casefold()
        if not any(marker in folded_line for marker in CATALOG_SCAN_MARKERS):
            if not any(marker in folded_line for marker in WRAP_SCAN_MARKERS):
                if not any(character in folded_line for character in WRAP_QUOTING_CHARACTERS):
                    return [], 0
                normalized_wrap_line = folded_line.translate(WRAP_QUOTING_TRANSLATION)
                if not any(marker in normalized_wrap_line for marker in WRAP_SCAN_MARKERS):
                    return [], 0
        scanned_line = mask_declared_roots(line, self.declared_roots)
        digest = hashlib.sha256(line.encode("utf-8")).hexdigest()
        matches, total_matches = scan_wrap_options(
            role, line_number, scanned_line, digest, retain_limit
        )
        folded_line = scanned_line.casefold()
        if not any(marker in folded_line for marker in CATALOG_SCAN_MARKERS):
            return matches, total_matches

        room = retain_limit - len(matches)
        for found in self.expression.finditer(scanned_line):
            index = int(found.lastgroup[1:])
            group = self.groups[index]
            total_matches += len(group)
            for entry in group:
                if room > 0:
                    matches.append(
                        Match(
                            role=role,
                            line=line_number,
                            column=found.start() + 1,
                            category=entry.category,
                            catalog_id=entry.catalog_id,
                            matched=found.group(0),
                            line_sha256=digest,
                        )
                    )
                    room -= 1
        return matches, total_matches


WRAP_SYMBOL = re.compile(r"^[A-Za-z_][A-Za-z0-9_.$@+-]{0,127}$")
LD_WRAP_OPTION = re.compile(
    r"^(?P<option>-{1,2}(?:wr|wra|wrap))(?:(?P<equals>=)(?P<symbol>.*))?$"
)
SHELL_CONTROL_CHARACTERS = ";&|<>"
ASCII_SHELL_WHITESPACE = frozenset(" \t\r\n\v\f")


@dataclasses.dataclass(frozen=True)
class ShellLexicalAnalysis:
    active_text: str
    forbidden_features: tuple[str, ...]


def analyze_shell_line(line: str) -> ShellLexicalAnalysis:
    """Locate real shell comments and nonliteral constructs without expanding them."""

    quote: str | None = None
    escaped = False
    word_start = True
    comment_start: int | None = None
    forbidden: set[str] = set()
    for index, character in enumerate(line):
        if quote == "single":
            if character == "'":
                quote = None
            continue
        if escaped:
            escaped = False
            word_start = False
            continue
        if character == "\\":
            forbidden.add("shell_quoting_or_escape")
            escaped = True
            word_start = False
            continue
        if quote == "double":
            if character == '"':
                quote = None
            elif character in {"$", "`"}:
                forbidden.add("shell_expansion")
            continue
        if character == "'":
            forbidden.add("shell_quoting_or_escape")
            quote = "single"
            word_start = False
        elif character == '"':
            forbidden.add("shell_quoting_or_escape")
            quote = "double"
            word_start = False
        elif character in ASCII_SHELL_WHITESPACE:
            word_start = True
        elif character == "#" and word_start:
            forbidden.add("shell_comment")
            comment_start = index
            break
        elif character == "#":
            forbidden.add("unquoted_hash")
            word_start = False
        elif character in {"$", "`"}:
            forbidden.add("shell_expansion")
            word_start = False
        elif character in {"<", ">"}:
            forbidden.add("shell_redirection")
            word_start = True
        elif character in {"(", ")", "{", "}"}:
            forbidden.add("shell_group_or_expansion")
            word_start = True
        elif character in {"*", "?", "["}:
            forbidden.add("shell_pathname_expansion")
            word_start = False
        elif character == "~" and word_start:
            forbidden.add("shell_tilde_expansion")
            word_start = False
        else:
            word_start = character in SHELL_CONTROL_CHARACTERS
    active_text = line if comment_start is None else line[:comment_start].rstrip()
    return ShellLexicalAnalysis(active_text, tuple(sorted(forbidden)))


def shell_tokenize(line: str) -> tuple[str, ...]:
    """Tokenize shell text, normalizing quoting while retaining control operators."""

    lexer = shlex.shlex(line, posix=True, punctuation_chars=SHELL_CONTROL_CHARACTERS)
    lexer.whitespace_split = True
    lexer.commenters = ""
    return tuple(lexer)


def is_shell_control_token(token: str) -> bool:
    return bool(token) and all(character in SHELL_CONTROL_CHARACTERS for character in token)


def split_shell_command_tokens(
    tokens: Sequence[str],
) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...]]:
    segments: list[tuple[str, ...]] = []
    controls: list[str] = []
    current: list[str] = []
    for token in tokens:
        if is_shell_control_token(token):
            segments.append(tuple(current))
            controls.append(token)
            current = []
        else:
            current.append(token)
    segments.append(tuple(current))
    return tuple(segments), tuple(controls)


def parse_ld_wrap_option(token: str) -> tuple[str, bool, str | None] | None:
    matched = LD_WRAP_OPTION.fullmatch(token)
    if matched is None:
        return None
    return matched.group("option"), matched.group("equals") is not None, matched.group("symbol")


def is_wrap_like_token(token: str) -> bool:
    folded = token.casefold()
    if "wrap" in folded:
        return True
    option = folded.partition("=")[0]
    return option in {"--w", "--wr", "--wra", "--wrap", "-wr", "-wra", "-wrap"}


def scan_wrap_options(
    role: str,
    line_number: int,
    line: str,
    digest: str,
    retain_limit: int,
) -> tuple[list[Match], int]:
    """Apply the frozen GNU ld 2.42 textual wrap grammar; ELF proof is separate."""

    require(0 <= retain_limit <= MAX_REPORTED_MATCHES, "invalid_match_limit",
            f"invalid wrap retained-match limit: {retain_limit}")
    try:
        tokens = shell_tokenize(line)
    except ValueError:
        folded = line.casefold().replace("\\", "").replace("'", "").replace('"', "")
        if not any(marker in folded for marker in ("wrap", "--w", "-wra", "-wr")):
            return [], 0
        column = folded.find("wrap")
        if column < 0:
            column = folded.find("--w")
        if column < 0:
            column = 0
        return ([Match(role, line_number, column + 1, "wrap_syntax",
                       "gnu-wrap-syntax-unrecognized", "unparseable-wrap-syntax", digest,
                       "unrecognized")] if retain_limit else []), 1

    matches: list[Match] = []
    total_matches = 0
    consumed: set[int] = set()
    search_from = 0

    def report(symbol: str, spelling: str, source: str, recognized: bool) -> None:
        nonlocal total_matches, search_from
        total_matches += 1
        column = line.find(source, search_from)
        if column < 0:
            folded = line.casefold()
            for needle in ("--wrap", "-wrap", "--wra", "-wra", "--wr", "-wr", "wrap"):
                column = folded.find(needle, search_from)
                if column >= 0:
                    break
        if column < 0:
            column = 0
        search_from = column + 1
        if len(matches) >= retain_limit:
            return
        valid_symbol = bool(WRAP_SYMBOL.fullmatch(symbol))
        catalog_id = "gnu-wrap-option" if recognized and valid_symbol else "gnu-wrap-syntax-unrecognized"
        normalized = f"--wrap={symbol}" if recognized and valid_symbol else source
        matches.append(Match(role, line_number, column + 1,
                             "wrap_option" if catalog_id == "gnu-wrap-option" else "wrap_syntax",
                             catalog_id, normalized, digest, spelling))

    index = 0
    while index < len(tokens):
        if index in consumed:
            index += 1
            continue
        token = tokens[index]
        if token.startswith("-Wl,"):
            fields = token[4:].split(",")
            field_index = 0
            while field_index < len(fields):
                field = fields[field_index]
                parsed = parse_ld_wrap_option(field)
                if parsed is not None:
                    option, has_equals, symbol = parsed
                    if has_equals:
                        report(symbol or "", f"-Wl,{option}=VALUE", field, True)
                    elif field_index + 1 < len(fields):
                        field_index += 1
                        report(fields[field_index], f"-Wl,{option},VALUE", token, True)
                    elif index + 1 < len(tokens):
                        consumed.add(index + 1)
                        report(tokens[index + 1], f"-Wl,{option} VALUE", token, True)
                    else:
                        report("", "unrecognized", field, False)
                elif is_wrap_like_token(field):
                    report("", "unrecognized", field, False)
                field_index += 1
        elif token == "-Xlinker" and index + 1 < len(tokens):
            candidate = tokens[index + 1]
            parsed = parse_ld_wrap_option(candidate)
            if parsed is not None:
                option, has_equals, symbol = parsed
                consumed.update((index, index + 1))
                if has_equals:
                    report(symbol or "", f"-Xlinker {option}=VALUE",
                           f"{token} {candidate}", True)
                elif index + 3 < len(tokens) and tokens[index + 2] == "-Xlinker":
                    consumed.update((index + 2, index + 3))
                    report(tokens[index + 3], f"-Xlinker {option} -Xlinker VALUE",
                           f"{token} {candidate}", True)
                else:
                    report("", "unrecognized", f"{token} {candidate}", False)
            elif is_wrap_like_token(candidate):
                consumed.update((index, index + 1))
                report("", "unrecognized", f"{token} {candidate}", False)
        else:
            parsed = parse_ld_wrap_option(token)
            if parsed is not None:
                option, has_equals, symbol = parsed
                consumed.add(index)
                if has_equals:
                    report(symbol or "", f"{option}=VALUE", token, True)
                elif index + 1 < len(tokens):
                    consumed.add(index + 1)
                    report(tokens[index + 1], f"{option} VALUE", token, True)
                else:
                    report("", "unrecognized", token, False)
            elif is_wrap_like_token(token):
                consumed.add(index)
                report("", "unrecognized", token, False)
        index += 1
    return matches, total_matches


def ninja_escape_path(value: str) -> str:
    return value.replace("$", "$$").replace(" ", "$ ").replace(":", "$:")


ROOT_TOKEN_DELIMITERS = ASCII_SHELL_WHITESPACE
ROOT_JOINED_OPTION_PREFIXES = (
    "-B",
    "-I",
    "-L",
    "-S",
    "-o",
    "-Wl,-rpath,",
    "-isystem",
    "-iquote",
    "-include",
)


def root_occurrence_has_left_boundary(line: str, start: int) -> bool:
    if start == 0 or line[start - 1] in ROOT_TOKEN_DELIMITERS:
        return True
    for prefix in ROOT_JOINED_OPTION_PREFIXES:
        prefix_start = start - len(prefix)
        if prefix_start < 0 or line[prefix_start:start] != prefix:
            continue
        if prefix_start == 0 or line[prefix_start - 1] in ROOT_TOKEN_DELIMITERS:
            return True
    return False


def root_occurrence_has_right_boundary(line: str, end: int) -> bool:
    return end == len(line) or line[end] == "/" or line[end] in ROOT_TOKEN_DELIMITERS


def iter_boundary_root_occurrences(line: str, value: str):
    search_from = 0
    while True:
        start = line.find(value, search_from)
        if start < 0:
            return
        end = start + len(value)
        if (root_occurrence_has_left_boundary(line, start)
                and root_occurrence_has_right_boundary(line, end)):
            yield start, end
            search_from = end
        else:
            search_from = start + 1


def declared_root_variants(root: str) -> tuple[str, ...]:
    require(ninja_escape_path(root) == root, "invalid_root",
            "controlled declared roots must not require Ninja escaping")
    return (root,)


def line_binds_declared_root(line: str, root: str) -> bool:
    return any(
        next(iter_boundary_root_occurrences(line, variant), None) is not None
        for variant in declared_root_variants(root)
    )


def mask_boundary_root_occurrences(line: str, value: str) -> str:
    pieces: list[str] = []
    cursor = 0
    found = False
    for start, end in iter_boundary_root_occurrences(line, value):
        pieces.extend((line[cursor:start], "_" * (end - start)))
        cursor = end
        found = True
    if not found:
        return line
    pieces.append(line[cursor:])
    return "".join(pieces)


def mask_declared_roots(line: str, declared_roots: Sequence[str]) -> str:
    """Mask controlled-domain declared roots without changing byte/character columns."""

    masked = line
    variants = {
        variant
        for root in declared_roots
        for variant in declared_root_variants(root)
    }
    for variant in sorted(variants, key=len, reverse=True):
        masked = mask_boundary_root_occurrences(masked, variant)
    require(len(masked) == len(line), "root_mask_length", "declared-root masking changed line length")
    return masked


TARGET_SENTINELS = (
    "all: phony",
    "build.ninja: RERUN_CMAKE",
    "clean: CLEAN",
    "help: HELP",
)


@dataclasses.dataclass(frozen=True)
class OutputMention:
    segment_index: int
    value: str
    spelling: str


def output_mentions_from_segment(
    segment: Sequence[str],
    segment_index: int,
) -> tuple[OutputMention, ...]:
    mentions: list[OutputMention] = []
    index = 0
    while index < len(segment):
        token = segment[index]
        if token in {"-o", "--output"} and index + 1 < len(segment):
            mentions.append(OutputMention(segment_index, segment[index + 1], f"{token} VALUE"))
            index += 2
            continue
        elif token.startswith("-o") and len(token) > 2:
            mentions.append(OutputMention(segment_index, token[2:], "-oVALUE"))
        elif token.startswith("--output="):
            mentions.append(OutputMention(segment_index, token.partition("=")[2], "--output=VALUE"))
        elif token.startswith("-Wl,"):
            fields = token[4:].split(",")
            for field_index, field in enumerate(fields):
                if field in {"-o", "--output"} and field_index + 1 < len(fields):
                    mentions.append(
                        OutputMention(segment_index, fields[field_index + 1], "-Wl,OUTPUT")
                    )
                elif field.startswith("-o") and len(field) > 2:
                    mentions.append(OutputMention(segment_index, field[2:], "-Wl,-oVALUE"))
                elif field.startswith("--output="):
                    mentions.append(
                        OutputMention(segment_index, field.partition("=")[2], "-Wl,--output=VALUE")
                    )
        elif token == "-Xlinker" and index + 1 < len(segment):
            option = segment[index + 1]
            if option.startswith("--output="):
                mentions.append(
                    OutputMention(segment_index, option.partition("=")[2], "-Xlinker --output=VALUE")
                )
                index += 2
                continue
            elif option in {"-o", "--output"}:
                if index + 3 < len(segment) and segment[index + 2] == "-Xlinker":
                    mentions.append(
                        OutputMention(segment_index, segment[index + 3], "-Xlinker OUTPUT")
                    )
                    index += 4
                    continue
                elif index + 2 < len(segment):
                    mentions.append(
                        OutputMention(segment_index, segment[index + 2], "-Xlinker OUTPUT")
                    )
                    index += 3
                    continue
        index += 1
    return tuple(mentions)


def output_mentions(tokens: Sequence[str]) -> tuple[OutputMention, ...]:
    segments, _ = split_shell_command_tokens(tokens)
    return tuple(
        mention
        for segment_index, segment in enumerate(segments)
        for mention in output_mentions_from_segment(segment, segment_index)
    )


def output_mention_target(mention: OutputMention, targets: Sequence[str]) -> str | None:
    normalized = os.path.normpath(mention.value)
    basename = os.path.basename(normalized)
    return basename if basename in targets else None


def raw_output_value_target(value: str, targets: Sequence[str]) -> str | None:
    normalized_value = value.strip(RAW_OUTPUT_TRAILING_WRAPPER_CHARS)
    normalized = os.path.normpath(normalized_value)
    basename = os.path.basename(normalized)
    return basename if basename in targets else None


def line_has_raw_frozen_output_candidate(line: str, targets: Sequence[str]) -> bool:
    """Bind target text to an output operand, including forbidden/malformed shell text."""

    try:
        mentions = output_mentions(shell_tokenize(line))
    except ValueError:
        normalized_line = line.replace("\\", "").replace("'", "").replace('"', "")
        return any(
            raw_output_value_target(found.group("value"), targets) is not None
            for expression in MALFORMED_RAW_OUTPUT_EXPRESSIONS
            for found in re.finditer(expression, normalized_line)
        )
    return any(raw_output_value_target(mention.value, targets) is not None
               for mention in mentions)


def safe_artifact_path(value: str, *, allow_absolute: bool) -> bool:
    if not value or (os.path.isabs(value) and not allow_absolute):
        return False
    components = value.split("/")
    if allow_absolute and components and components[0] == "":
        components = components[1:]
    return bool(components) and all(
        component not in {"", ".", ".."}
        and re.fullmatch(FINAL_LINK_ARTIFACT_COMPONENT_EXPRESSION, component) is not None
        for component in components
    )


def is_object_input(value: str) -> bool:
    return value.endswith(".o") and safe_artifact_path(value, allow_absolute=False)


def is_target_owned_object(value: str, target: str) -> bool:
    prefix = f"CMakeFiles/{target}.dir/"
    return value.startswith(prefix) and len(value) > len(prefix) + 2 and is_object_input(value)


def is_library_argument(value: str) -> bool:
    if re.fullmatch(r"-l[A-Za-z0-9_.+-]+", value):
        return True
    if re.search(FINAL_LINK_LIBRARY_SUFFIX_EXPRESSION + r"$", value) is None:
        return False
    return safe_artifact_path(value, allow_absolute=True)


def is_pre_object_option(value: str, target: str) -> bool:
    return (
        value == FINAL_LINK_DEPENDENCY_OPTION_TEMPLATE.format(target=target)
        or any(re.fullmatch(expression, value) is not None
               for expression in FINAL_LINK_PRE_OPTION_EXPRESSIONS)
    )


def validate_strict_final_link_argv(
    argv: Sequence[str],
    target: str,
    build_root: str,
    line_number: int,
) -> None:
    require(bool(argv) and argv[0] == R0_FINAL_LINK_DRIVER,
            "final_link_driver_not_exact",
            f"commands line {line_number} final target does not use exact textual driver "
            f"{R0_FINAL_LINK_DRIVER}")
    for token in argv[1:]:
        if token in COMPILER_NON_LINK_OPTIONS or any(
            token.startswith(prefix) for prefix in COMPILER_NON_LINK_PREFIXES
        ):
            raise AbsenceError(
                "compiler_non_link_mode",
                f"commands line {line_number} contains non-link/query mode {token}",
            )
        require(token != "--" and not token.startswith("@"),
                "unsupported_final_link_token",
                f"commands line {line_number} contains an option terminator or response file")

    mentions = output_mentions_from_segment(argv, 0)
    require(len(mentions) <= 1,
            "multiple_output_options",
            f"commands line {line_number} contains multiple output options")
    require(len(mentions) == 1 and mentions[0].spelling == "-o VALUE",
            "unsupported_final_output_spelling",
            f"commands line {line_number} requires exactly one separate -o VALUE spelling")
    require(mentions[0].value == target,
            "final_output_not_bare_basename",
            f"commands line {line_number} output must be exact bare basename {target}")
    output_index = tuple(argv).index("-o")
    require(output_index + 1 < len(argv) and argv[output_index + 1] == target,
            "malformed_output_option",
            f"commands line {line_number} has an invalid final -o operand")

    before_output = tuple(argv[1:output_index])
    object_start = next(
        (index for index, token in enumerate(before_output) if is_object_input(token)),
        None,
    )
    require(object_start is not None and object_start < len(before_output),
            "missing_final_link_input_block",
            f"commands line {line_number} has no contiguous object input block")
    pre_options = before_output[:object_start]
    objects = before_output[object_start:]
    require(all(is_pre_object_option(token, target) for token in pre_options),
            "unsupported_final_link_pre_option",
            f"commands line {line_number} has a token outside the frozen pre-object grammar")
    require(bool(objects) and all(is_object_input(token) for token in objects),
            "invalid_final_link_input_block",
            f"commands line {line_number} object inputs are not one contiguous safe block")
    require(any(is_target_owned_object(token, target) for token in objects),
            "missing_target_owned_object",
            f"commands line {line_number} has no object owned by target {target}")

    post_output = tuple(argv[output_index + 2:])
    expected_rpath = f"-Wl,-rpath,{build_root}"
    start_group = "-Wl,--start-group"
    end_group = "-Wl,--end-group"
    require(post_output.count(expected_rpath) == 1,
            "missing_exact_build_rpath",
            f"commands line {line_number} lacks one exact build-root rpath")
    require(post_output.count(start_group) == 1 and post_output.count(end_group) == 1
            and post_output.index(start_group) < post_output.index(end_group),
            "invalid_link_group_scaffold",
            f"commands line {line_number} lacks one ordered start/end linker group")
    require(all(
                token in {expected_rpath, start_group, end_group}
                or is_library_argument(token)
                for token in post_output
            ),
            "unsupported_final_link_post_argument",
            f"commands line {line_number} has an argument outside the frozen post-output grammar")


def direct_final_link_segment(
    segments: Sequence[Sequence[str]],
    controls: Sequence[str],
    line_number: int,
) -> tuple[str, ...]:
    """Admit only CMake/Ninja's exact guarded single-command scaffold."""

    require(tuple(controls) == ("&&", "&&") and len(segments) == 3
            and tuple(segments[0]) == (":",) and tuple(segments[2]) == (":",),
            "final_link_scaffold_not_exact",
            f"commands line {line_number} is not exact ': && DRIVER ... && :' syntax")
    return tuple(segments[1])


def line_may_contain_frozen_target(line: str, targets: Sequence[str]) -> bool:
    """Conservatively retain raw and quote/backslash-joined target spellings."""

    if any(target in line for target in targets):
        return True
    normalized_line = line.translate(WRAP_QUOTING_TRANSLATION)
    return any(target in normalized_line for target in targets)


def observe_frozen_final_outputs(
    line: str,
    line_number: int,
    build_root: str,
    counts: dict[str, int],
) -> str:
    """Count only structurally valid final `-o` arguments for the frozen R0 targets."""

    analysis = analyze_shell_line(line)
    targets = tuple(counts)
    if not line_may_contain_frozen_target(line, targets):
        return analysis.active_text if not analysis.forbidden_features else ""
    try:
        tokens = shell_tokenize(analysis.active_text)
    except ValueError as error:
        raise AbsenceError(
            "malformed_command_line",
            f"commands line {line_number} cannot be shell-tokenized: {error}",
        ) from error
    mentions = output_mentions(tokens)
    frozen = tuple(
        (mention, target)
        for mention in mentions
        if (target := output_mention_target(mention, targets)) is not None
    )
    if analysis.forbidden_features and (
        frozen or line_has_raw_frozen_output_candidate(line, targets)
    ):
        raise AbsenceError(
            "forbidden_final_link_shell_syntax",
            f"commands line {line_number} final-output text occurs with forbidden shell "
            f"features: {', '.join(analysis.forbidden_features)}",
        )
    if not frozen:
        return analysis.active_text if not analysis.forbidden_features else ""

    require(len(frozen) == 1,
            "multiple_output_options",
            f"commands line {line_number} contains multiple frozen output mentions")
    segments, controls = split_shell_command_tokens(tokens)
    direct = direct_final_link_segment(segments, controls, line_number)
    mention, target = frozen[0]
    require(mention.segment_index == 1,
            "final_output_not_direct_link",
            f"commands line {line_number} frozen output is outside the guarded driver segment")
    validate_strict_final_link_argv(direct, target, build_root, line_number)
    counts[target] += 1
    return analysis.active_text


class RoleValidator:
    def __init__(
        self,
        role: str,
        declared_roots: Sequence[str],
        build_root: str,
    ):
        self.role = role
        self.nonempty = 0
        self.first_line_is_cmake_header = False
        self.last_line_is_default_all = False
        self.line_count = 0
        self.ninja_version_count = 0
        self.ninja_version_line: int | None = None
        self.first_build_line: int | None = None
        self.default_all_count = 0
        self.default_all_line: int | None = None
        self.build_edge = False
        self.declared_roots = tuple(declared_roots)
        self.root_seen = [False] * len(self.declared_roots)
        self.target_sentinels = {sentinel: 0 for sentinel in TARGET_SENTINELS}
        self.build_root = build_root
        self.final_target_counts = {target: 0 for target in R0_FINAL_TARGETS}

    def observe(self, line: str, line_number: int) -> None:
        self.line_count = line_number
        if line_number == 1:
            self.first_line_is_cmake_header = line == "# CMAKE generated file: DO NOT EDIT!"
        self.last_line_is_default_all = line == "default all"
        stripped = line.strip()
        if stripped:
            self.nonempty += 1
        if self.role == "build_ninja":
            for index, root in enumerate(self.declared_roots):
                if not self.root_seen[index]:
                    self.root_seen[index] = line_binds_declared_root(line, root)
        if self.role == "build_ninja":
            if re.fullmatch(r"ninja_required_version = [0-9]+(?:\.[0-9]+)+", line):
                self.ninja_version_count += 1
                if self.ninja_version_line is None:
                    self.ninja_version_line = line_number
            if line.startswith("build "):
                self.build_edge = True
                if self.first_build_line is None:
                    self.first_build_line = line_number
            if line == "default all":
                self.default_all_count += 1
                if self.default_all_line is None:
                    self.default_all_line = line_number
        elif self.role == "targets":
            if line in self.target_sentinels:
                self.target_sentinels[line] += 1
        elif self.role == "commands":
            if all(self.root_seen) and not line_may_contain_frozen_target(
                line, tuple(self.final_target_counts)
            ):
                return
            active_root_text = observe_frozen_final_outputs(
                line, line_number, self.build_root, self.final_target_counts
            )
            if active_root_text:
                for index, root in enumerate(self.declared_roots):
                    if not self.root_seen[index]:
                        self.root_seen[index] = line_binds_declared_root(active_root_text, root)

    def validate(self) -> None:
        require(self.nonempty > 0, "empty_evidence", f"{self.role} evidence has no records")
        if self.role == "build_ninja":
            require(self.first_line_is_cmake_header,
                    "malformed_build_ninja", "build_ninja does not start with the exact CMake header")
            require(self.ninja_version_count == 1 and self.build_edge,
                    "malformed_build_ninja",
                    "build_ninja lacks one exact Ninja version record or any build edge")
            require(self.default_all_count == 1 and self.default_all_line == self.line_count and
                    self.last_line_is_default_all,
                    "truncated_build_ninja", "build_ninja does not terminate with one exact default all")
            require(self.ninja_version_line is not None and self.first_build_line is not None and
                    self.default_all_line is not None and
                    self.ninja_version_line < self.first_build_line < self.default_all_line,
                    "malformed_build_ninja", "build_ninja header/version/build/default ordering is invalid")
        elif self.role == "targets":
            missing = [sentinel for sentinel, count in self.target_sentinels.items() if count != 1]
            require(not missing, "incomplete_target_list",
                    "targets evidence lacks one exact occurrence of: " + ", ".join(missing))
        elif self.role == "commands":
            incomplete = [target for target, count in self.final_target_counts.items() if count != 1]
            require(not incomplete, "incomplete_command_list",
                    "commands evidence lacks one exact final-link output for: " + ", ".join(incomplete))
        if self.role in {"build_ninja", "commands"}:
            missing_roots = [root for root, seen in zip(self.declared_roots, self.root_seen) if not seen]
            require(not missing_roots, "unbound_declared_root",
                    f"{self.role} does not bind every declared source/build root: "
                    + ", ".join(missing_roots))


def validate_line_bytes(raw: bytes, role: str, line_number: int) -> str:
    require(len(raw) <= MAX_LINE_BYTES, "line_over_limit",
            f"{role} line {line_number} exceeds {MAX_LINE_BYTES} bytes")
    try:
        line = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise AbsenceError("non_utf8_evidence",
                           f"{role} line {line_number} is not valid UTF-8: {error}") from error
    invalid = INVALID_LINE_CHARACTER.search(line)
    if invalid is not None:
        character = invalid.group(0)
        if character == "\r":
            message = f"{role} line {line_number} contains embedded carriage return"
        elif character == "\x7f":
            message = (
                f"{role} line {line_number} contains DEL at column {invalid.start() + 1}"
            )
        else:
            message = (
                f"{role} line {line_number} contains control byte at column "
                f"{invalid.start() + 1}"
            )
        raise AbsenceError("control_character", message)
    return line[:-1] if line.endswith("\r") else line


def scan_regular_evidence(
    role: str,
    path: Path,
    catalog: Catalog,
    max_bytes: int,
    *,
    forbidden_identities: set[tuple[int, int]] | None = None,
    retain_limit: int = MAX_REPORTED_MATCHES,
    chunk_bytes: int = READ_CHUNK_BYTES,
) -> tuple[EvidenceSummary, list[Match], int]:
    """Open once, validate one regular snapshot, hash it, and scan by bounded lines."""

    require(0 < chunk_bytes <= READ_CHUNK_BYTES, "invalid_chunk_size",
            f"invalid read chunk size: {chunk_bytes}")
    require(0 <= retain_limit <= MAX_REPORTED_MATCHES, "invalid_match_limit",
            f"invalid evidence retained-match limit: {retain_limit}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise AbsenceError("evidence_open_failed", f"cannot open {role} evidence {path}: {error}") from error

    hasher = hashlib.sha256()
    buffer = b""
    byte_count = 0
    line_count = 0
    matches: list[Match] = []
    total_matches = 0
    validator = RoleValidator(role, catalog.declared_roots, catalog.build_root)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), "evidence_not_regular",
                f"{role} evidence is not a regular file: {path}")
        file_identity = (before.st_dev, before.st_ino)
        require(forbidden_identities is None or file_identity not in forbidden_identities,
                "evidence_identity_alias",
                f"{role} evidence aliases an earlier role by device/inode: {path}")
        require(0 < before.st_size <= max_bytes, "evidence_size_invalid",
                f"{role} evidence size {before.st_size} is outside 1..{max_bytes}")

        while True:
            try:
                chunk = os.read(descriptor, chunk_bytes)
            except BlockingIOError as error:
                raise AbsenceError(
                    "evidence_read_would_block",
                    f"regular {role} evidence unexpectedly would block: {path}",
                ) from error
            if not chunk:
                break
            byte_count += len(chunk)
            require(byte_count <= max_bytes, "evidence_over_limit",
                    f"{role} evidence grew beyond {max_bytes} bytes while being read")
            hasher.update(chunk)
            buffer += chunk
            require(len(buffer) <= MAX_LINE_BYTES or b"\n" in buffer,
                    "line_over_limit", f"{role} contains a line exceeding {MAX_LINE_BYTES} bytes")
            while True:
                newline = buffer.find(b"\n")
                if newline < 0:
                    break
                raw_line = buffer[:newline]
                buffer = buffer[newline + 1 :]
                line_count += 1
                line = validate_line_bytes(raw_line, role, line_count)
                room = retain_limit - len(matches)
                validator.observe(line, line_count)
                line_matches, line_match_count = catalog.scan_line(
                    role, line_count, line, room
                )
                total_matches += line_match_count
                matches.extend(line_matches)
        require(not buffer, "unterminated_evidence",
                f"{role} evidence does not end with LF and may be truncated")
        after = os.fstat(descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size,
                           before.st_mtime_ns, before.st_ctime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size,
                          after.st_mtime_ns, after.st_ctime_ns)
        require(identity_before == identity_after and byte_count == before.st_size,
                "evidence_changed", f"{role} evidence changed while being read: {path}")
        validator.validate()
    finally:
        os.close(descriptor)

    return (
        EvidenceSummary(role, str(path), byte_count, line_count, hasher.hexdigest(),
                        before.st_dev, before.st_ino),
        matches,
        total_matches,
    )


def emit_result(
    payload: dict[str, object],
    intended_exit: int,
    stream: BinaryIO | None = None,
) -> int:
    bounded_payload = dict(payload)
    bounded_payload.setdefault("catalog_version", CATALOG_VERSION)
    encoded = (json.dumps(bounded_payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    actual_exit = intended_exit
    if len(encoded) > MAX_JSON_BYTES:
        encoded = (json.dumps(
            {
                **result_contract(),
                "catalog_version": CATALOG_VERSION,
                "problem": {"code": "result_over_limit", "message": "structured result exceeded bound"},
                "schema": SCHEMA,
                "verdict": "invalid",
            },
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n").encode("utf-8")
        actual_exit = 2
    require(len(encoded) <= MAX_JSON_BYTES, "internal_result_bound", "fallback JSON exceeds output bound")
    (sys.stdout.buffer if stream is None else stream).write(encoded)
    return actual_exit


def scan_text_for_test(
    catalog: Catalog,
    text: str,
    retain_limit: int = MAX_REPORTED_MATCHES,
) -> tuple[list[Match], int]:
    result: list[Match] = []
    total = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        matches, count = catalog.scan_line(
            "self_test", line_number, line, retain_limit - len(result)
        )
        result.extend(matches)
        total += count
    return result, total


def expect_absence_error(expected_code: str, operation: Callable[[], object]) -> None:
    try:
        operation()
    except AbsenceError as error:
        require(error.code == expected_code, "self_test_wrong_error",
                f"expected {expected_code}, observed {error.code}")
    else:
        raise AbsenceError("self_test_missing_error", f"expected error was not raised: {expected_code}")


def ordinary_compile_regression_lines(source_root: str) -> tuple[str, ...]:
    sources = (
        ("nr-softmodem", "executables/nr-gnb.c"),
        ("nr-uesoftmodem", "executables/nr-ue.c"),
    )
    result: list[str] = []
    for target, source in sources:
        object_path = f"CMakeFiles/{target}.dir/{source}.o"
        result.append(
            f"/usr/bin/ccache /usr/bin/cc -I{source_root}/safe "
            f"-DCMAKE_BUILD_TYPE=\\\"RelWithDebInfo\\\" "
            f"-MD -MT {object_path} -MF {object_path}.d "
            f"-o {object_path} -c {source_root}/{source}"
        )
    return tuple(result)


def write_valid_evidence(
    root: Path,
    source_root: str,
    build_root: str,
) -> dict[str, Path]:
    paths = {
        "build_ninja": root / "build.ninja",
        "targets": root / "targets.txt",
        "commands": root / "commands.txt",
    }
    paths["build_ninja"].write_text(
        "# CMAKE generated file: DO NOT EDIT!\n"
        "# Generated by self-test\n"
        "ninja_required_version = 1.5\n"
        f"# source root = {ninja_escape_path(source_root)}\n"
        f"# build root = {ninja_escape_path(build_root)}\n"
        "build safe: phony\n"
        "default all\n",
        encoding="utf-8",
        newline="",
    )
    paths["targets"].write_text(
        "safe: phony\n" + "\n".join(TARGET_SENTINELS) + "\n",
        encoding="utf-8",
        newline="",
    )
    command_lines = [
        f"{R0_FINAL_LINK_DRIVER} -I{source_root}/safe -c safe.c -o safe.o",
        *(
        f": && {R0_FINAL_LINK_DRIVER} -DSELF_TEST=1 -mcpu=native -march=native "
        f"-lgcc -lrt -Wl,--dependency-file=CMakeFiles/{target}.dir/link.d "
        f"CMakeFiles/{target}.dir/selftest.c.o -o {target} "
        f"-Wl,-rpath,{build_root} -Wl,--start-group libsafe.a "
        f"-Wl,--end-group -lm && :"
        for target in R0_FINAL_TARGETS
        ),
        *ordinary_compile_regression_lines(source_root),
    ]
    paths["commands"].write_text("\n".join(command_lines) + "\n", encoding="utf-8", newline="")
    return paths


def invoke_for_test(arguments: Sequence[str]) -> tuple[int, dict[str, object], bytes]:
    output = io.BytesIO()
    status = main(arguments, output)
    raw = output.getvalue()
    require(0 < len(raw) <= MAX_JSON_BYTES and raw.endswith(b"\n"),
            "self_test_json_bound", "CLI result is empty, over-limit, or not LF-terminated")
    try:
        payload = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AbsenceError("self_test_json_parse", f"CLI result is not one UTF-8 JSON object: {error}") from error
    require(isinstance(payload, dict), "self_test_json_parse", "CLI result is not a JSON object")
    return status, payload, raw


def run_self_test(stream: BinaryIO | None = None) -> int:
    checks = 0
    source_root = "/source/OAI_MEMPROF_WIRE_collision"
    build_root = "/build/oai_memprof_r0_--wrap-collision"
    entries = build_catalog()
    catalog = Catalog(entries, (source_root, build_root))
    try:
        catalog_digest = validate_catalog_freeze(entries)
        checks += 4  # exact ID set, uniqueness, count, and digest

        first = entries[0]
        literal_mutations = (
            ("catalog_id_set_mismatch",
             CatalogEntry(first.category, first.catalog_id + ".literal-mutant",
                          first.expression, first.sample)),
            ("catalog_digest_mismatch",
             CatalogEntry(first.category + ".literal-mutant", first.catalog_id,
                          first.expression, first.sample)),
            ("catalog_digest_mismatch",
             CatalogEntry(first.category, first.catalog_id,
                          first.expression + "(?:literal-mutant)", first.sample)),
            ("catalog_digest_mismatch",
             CatalogEntry(first.category, first.catalog_id,
                          first.expression, first.sample + " literal-mutant")),
        )
        for expected_code, mutant in literal_mutations:
            expect_absence_error(
                expected_code,
                lambda mutant=mutant: validate_catalog_freeze((mutant,) + entries[1:]),
            )
            checks += 1

        expect_absence_error("catalog_count_mismatch", lambda: validate_catalog_freeze(entries[:-1]))
        checks += 1
        addition = CatalogEntry("literal-addition", "literal-addition:id",
                                r"(?<![A-Za-z])literal-addition", "literal-addition")
        expect_absence_error(
            "catalog_count_mismatch", lambda: validate_catalog_freeze(entries + (addition,))
        )
        checks += 1
        expect_absence_error(
            "catalog_digest_mismatch",
            lambda: validate_catalog_freeze((entries[1], entries[0]) + entries[2:]),
        )
        checks += 1

        required_semantic_ids = {
            "control-load:oai_memprof_control_load_v1",
            "output:liboai_memprof_container_wire.a",
            "output:liboai_memprof_softmodem_session.a",
            "output:dfts.oai-memprof.map",
            "output:ldpc.oai-memprof.map",
            "output:ldpc_orig.oai-memprof.map",
            "output:nr-softmodem.oai-memprof.map",
            "output:nr-uesoftmodem.oai-memprof.map",
            "output:oai_usrpdevif.oai-memprof.map",
            "output:params_libconfig.oai-memprof.map",
            "raw-schema:oai-memprof-r0-raw-v1",
            "wire-namespace:OAI_MEMPROF_WIRE",
            "r0-optional-lib-namespace",
            "r0-name:liboai_memprof_r0_actual_dso_a00.so",
            "r0-name:liboai_memprof_r0_actual_dso_a01.so",
            "r0-name:oai_memprof_r0_harness_selftest",
            "namespace:hyphen:oai-memprof",
            "namespace:lib:liboai_memprof",
            "namespace:linker:-loai_memprof",
            "namespace:lower:oai_memprof",
            "namespace:python:tools.profiling.memory",
            "namespace:upper:OAI_MEMPROF",
            "r0-file:test_oai_memprof_container_wire.c",
            "r0-file:test_oai_memprof_container_wire.py",
            "r0-file:test_oai_memprof_softmodem_launcher.py",
            "r0-file:test_oai_memprof_softmodem_session.c",
            "r0-name:oai_memprof_container_wire_c",
            "r0-name:oai_memprof_container_wire_python",
            "r0-name:test_oai_memprof_container_wire",
            "r0-name:oai_memprof_softmodem_launcher_python",
            "r0-name:test_oai_memprof_softmodem_session",
            "source:oai_memprof_container_wire.c",
            "source:oai_memprof_container_wire.h",
            "source:oai_memprof_container_wire.py",
            "source:oai_memprof_softmodem_launcher.py",
            "source:oai_memprof_softmodem_session.c",
            "source:oai_memprof_softmodem_session.h",
            "target:oai_memprof_container_wire",
            "target:oai_memprof_softmodem_session",
        }
        identifiers, _ = catalog_identity(entries)
        require(required_semantic_ids <= set(identifiers), "self_test_catalog_core",
                "the frozen catalog lacks a required DSO/core semantic ID")
        checks += len(required_semantic_ids)

        for entry in catalog.entries:
            observed, _ = scan_text_for_test(catalog, entry.sample)
            require(any(match.catalog_id == entry.catalog_id for match in observed),
                    "self_test_catalog_miss", f"catalog entry is insensitive: {entry.catalog_id}")
            checks += 1

        for api in APIS:
            for prefix, category in (
                ("__wrap_", "wrapper"),
                ("__real_", "real"),
            ):
                symbol = f"{prefix}{api}"
                observed, _ = scan_text_for_test(catalog, f"U {symbol}")
                require(any(match.catalog_id == f"{category}:{api}" and
                            match.matched == symbol
                            for match in observed),
                        "self_test_api_symbol_miss",
                        f"{category} spelling escaped for {api}")
                checks += 1

        observed, _ = scan_text_for_test(catalog, "U __real_aligned_alloc")
        require(any(match.catalog_id == "real:aligned_alloc" and
                    match.matched == "__real_aligned_alloc"
                    for match in observed),
                "self_test_non_original_api",
                "the isolated aligned_alloc real spelling escaped the catalog")
        checks += 1

        wrap_spellings: list[tuple[str, str]] = []
        for option_name in GNU_LD_WRAP_OPTION_NAMES:
            for dash_count in (1, 2):
                option = "-" * dash_count + option_name
                wrap_spellings.extend(
                    (
                        (f"ld x.o {option}={{api}} -o x", f"{option}=VALUE"),
                        (f"ld x.o {option} {{api}} -o x", f"{option} VALUE"),
                        (f"cc x.o -Wl,{option}={{api}} -o x",
                         f"-Wl,{option}=VALUE"),
                        (f"cc x.o -Wl,{option},{{api}} -o x",
                         f"-Wl,{option},VALUE"),
                        (f"cc x.o -Wl,{option} {{api}} -o x",
                         f"-Wl,{option} VALUE"),
                        (f"cc x.o -Xlinker {option}={{api}} -o x",
                         f"-Xlinker {option}=VALUE"),
                        (f"cc x.o -Xlinker {option} -Xlinker {{api}} -o x",
                         f"-Xlinker {option} -Xlinker VALUE"),
                    )
                )
        for api in APIS:
            for template, expected_spelling in wrap_spellings:
                observed, _ = scan_text_for_test(catalog, template.format(api=api))
                require(any(match.catalog_id == "gnu-wrap-option" and
                            match.spelling == expected_spelling and match.matched == f"--wrap={api}"
                            for match in observed),
                        "self_test_wrap_miss", f"wrap normalization missed {template}")
                checks += 1

        for escaped in REVIEWER_WRAP_ESCAPE_VECTORS:
            observed, _ = scan_text_for_test(catalog, escaped)
            require(any(match.catalog_id == "gnu-wrap-option" and
                        match.matched == "--wrap=malloc" for match in observed),
                    "self_test_wrap_escape",
                    f"shlex-unescaped accepted wrap spelling escaped detection: {escaped}")
            checks += 1

        for vector, normalized in ADJACENT_QUOTING_WRAP_VECTORS:
            tokens = shell_tokenize(vector)
            require(" ".join(tokens) == normalized,
                    "self_test_adjacent_quote_tokenization",
                    f"adjacent shell quoting did not normalize as frozen: {vector}")
            observed, _ = scan_text_for_test(catalog, vector)
            require(any(match.catalog_id == "gnu-wrap-option" and
                        match.matched == "--wrap=malloc" for match in observed),
                    "self_test_adjacent_quote_wrap",
                    f"adjacent shell quoting escaped wrap detection: {vector}")
            checks += 2

        for malformed in (
            "cc x.o -Wl,--wrap:malloc -o x",
            "cc x.o -Xlinker --wrap -Xlinker",
            "ld x.o --wrap -o x",
            "cc x.o wrapper-object.o -o x",
            "cc x.o '--wrap=malloc -o x",
            "ld --w=malloc",
            "ld --WrAp=malloc",
            "cc -Wl,--WrAp=malloc -o x",
            "cc -Xlinker --Wr=malloc -o x",
            "ld --wrapx=malloc",
            "cc -Wl,--w,malloc -o x",
        ):
            observed, count = scan_text_for_test(catalog, malformed)
            require(count > 0 and any(match.catalog_id == "gnu-wrap-syntax-unrecognized"
                                      for match in observed),
                    "self_test_wrap_fail_closed", f"unrecognized wrap syntax escaped: {malformed}")
            checks += 1

        observed, count = scan_text_for_test(catalog, GNU_LD_WRAP_EXPANSION_EXAMPLE)
        require(not observed and count == 0,
                "self_test_wrap_expansion_residual",
                "the declared shell-expansion residual unexpectedly entered the textual grammar")
        checks += 1

        for control in (
            "common/utils/memprofile/readme.txt",
            "not_oai_memprof_runtime_extra",
            "check_oai_memprof_r0_elf.pyi",
            "__real_free_list",
            "xoai_memprof_future",
            "XOAI_MEMPROF_FUTURE",
            "my-oai-memprof-future",
            "myliboai_memprof_future.so",
            "x-loai_memprof_future",
            "my.tools.profiling.memory.future",
            "cc -w x.c",
            "ld --wide x.o",
        ):
            observed, count = scan_text_for_test(catalog, control)
            require(not observed and count == 0, "self_test_false_positive",
                    f"false-positive control matched: {control}")
            checks += 1

        for expected_id, vector in REVIEWER_NAMESPACE_VECTORS:
            observed, _ = scan_text_for_test(catalog, vector)
            require(any(match.catalog_id == expected_id for match in observed),
                    "self_test_reviewer_namespace_vector",
                    f"reviewer namespace vector escaped {expected_id}: {vector}")
            checks += 1

        for expected_id, vector in R0_CMAKE_NAME_CORRESPONDENCE_VECTORS:
            observed, _ = scan_text_for_test(catalog, vector)
            require(any(match.catalog_id == expected_id for match in observed),
                    "self_test_cmake_name_correspondence",
                    f"CMake-name vector escaped exact catalog ID {expected_id}")
            checks += 1

        for optional_library in (
            "oai_memprof_r0_future_optional.so",
            "liboai_memprof_r0_future_optional.so.1.2",
        ):
            observed, _ = scan_text_for_test(catalog, optional_library)
            require(any(match.catalog_id == "r0-optional-lib-namespace" for match in observed),
                    "self_test_optional_lib_prefix",
                    f"optional-lib namespace missed: {optional_library}")
            checks += 1

        long_namespace = "oai_memprof_r0_" + "x" * 512
        observed, _ = scan_text_for_test(catalog, long_namespace)
        require(any(match.catalog_id == "r0-target-namespace" for match in observed) and
                all(len(match.matched) <= MAX_MATCH_TEXT_CHARS for match in observed),
                "self_test_namespace_bound",
                "an overlength R0 namespace escaped or retained an overlength match")
        checks += 1

        root_only = f"-I{source_root}/safe -L{build_root}/safe"
        masked = mask_declared_roots(root_only, (source_root, build_root))
        require(len(masked) == len(root_only), "self_test_root_mask", "root mask shifted columns")
        observed, count = scan_text_for_test(catalog, root_only)
        require(not observed and count == 0, "self_test_root_mask",
                "a catalog-like declared root produced a match")
        controlled_only = f"root {source_root} build {build_root}"
        observed, count = scan_text_for_test(catalog, controlled_only)
        require(not observed and count == 0, "self_test_root_mask",
                "a controlled declared root produced a match")
        suffix_line = f"{source_root}/common/utils/memprof/file.c"
        observed, _ = scan_text_for_test(catalog, suffix_line)
        require(any(match.catalog_id == "source-subtree-relative" and
                    match.column == len(source_root) + 2 for match in observed),
                "self_test_root_mask", "root masking hid a forbidden suffix or shifted its column")
        checks += 4

        for suffix in (
            "_future", "-sibling", ".suffix", "X", ":sibling", ",sibling",
            ";sibling", ")sibling",
        ):
            for root_value in (source_root, build_root):
                sibling = root_value + suffix
                require(not line_binds_declared_root(sibling, root_value),
                        "self_test_root_boundary",
                        f"forbidden root sibling bound the declared root: {sibling}")
                require(mask_declared_roots(sibling, (source_root, build_root)) == sibling,
                        "self_test_root_boundary",
                        f"forbidden root sibling was masked: {sibling}")
                checks += 2

        for root_value in (source_root, build_root):
            for sibling in (
                f'"{root_value} sibling"',
                root_value + r"\ sibling",
                root_value + "$ sibling",
                root_value + "$:sibling",
            ):
                require(not line_binds_declared_root(sibling, root_value),
                        "self_test_root_boundary",
                        f"quoted/escaped/Ninja root sibling bound declared root: {sibling}")
                require(mask_declared_roots(sibling, (source_root, build_root)) == sibling,
                        "self_test_root_boundary",
                        f"quoted/escaped/Ninja root sibling was masked: {sibling}")
                checks += 2

        for invalid_root in (
            "/source/with space",
            "/source/with'quote",
            "/source/with$dollar",
            "/source/with:colon",
            "/source/with,comma",
            "/source/with:semicolon",
            "/source/with)paren",
            "/source/../escape",
            "/source//double",
        ):
            expect_absence_error(
                "invalid_root", lambda invalid_root=invalid_root: normalized_root(
                    invalid_root, "self_test_root"
                )
            )
            checks += 1

        for compile_line in ordinary_compile_regression_lines(source_root):
            mentioned_targets = tuple(
                target for target in R0_FINAL_TARGETS if target in compile_line
            )
            require(len(mentioned_targets) == 1 and "-o CMakeFiles/" in compile_line,
                    "self_test_compile_shape",
                    "ordinary-compile regression lacks one target-owned object output")
            checks += 1
            analysis = analyze_shell_line(compile_line)
            require("shell_quoting_or_escape" in analysis.forbidden_features,
                    "self_test_compile_shape",
                    "ordinary-compile regression does not retain the replayed quoting shape")
            checks += 1
            require(not line_has_raw_frozen_output_candidate(
                        compile_line, R0_FINAL_TARGETS
                    ),
                    "self_test_compile_output_classification",
                    "target namespace in an object path was classified as a final output")
            checks += 1
            compile_counts = {target: 0 for target in R0_FINAL_TARGETS}
            active = observe_frozen_final_outputs(
                compile_line, 1, build_root, compile_counts
            )
            require(active == "" and not any(compile_counts.values()),
                    "self_test_compile_output_classification",
                    "ordinary compile command changed final-target counts or bound root text")
            checks += 1

        malformed_comment_candidate = (
            f": && /usr/bin/cc --version # -o '{build_root}/{R0_FINAL_TARGETS[0]}"
        )
        require(line_has_raw_frozen_output_candidate(
                    malformed_comment_candidate, R0_FINAL_TARGETS
                ),
                "self_test_malformed_raw_output_candidate",
                "malformed-comment fallback missed a frozen output operand")
        checks += 1
        malformed_compile_control = ordinary_compile_regression_lines(source_root)[0] + " '"
        require(not line_has_raw_frozen_output_candidate(
                    malformed_compile_control, R0_FINAL_TARGETS
                ),
                "self_test_malformed_raw_output_control",
                "malformed ordinary compile line became a frozen output candidate")
        checks += 1

        fast_final_counts = {target: 0 for target in R0_FINAL_TARGETS}
        clean_non_target_command = "/usr/bin/c++ -Wall -c safe.cc -o safe.o"
        with mock.patch.object(
            sys.modules[__name__],
            "shell_tokenize",
            side_effect=AssertionError("non-target command reached shlex"),
        ):
            require(
                observe_frozen_final_outputs(
                    clean_non_target_command, 1, build_root, fast_final_counts
                ) == clean_non_target_command
                and not any(fast_final_counts.values()),
                "self_test_final_target_prefilter",
                "a clean non-target command did not bypass final-output tokenization",
            )
        checks += 1

        fast_role = RoleValidator("commands", (source_root, build_root), build_root)
        fast_role.root_seen[:] = [True, True]
        with mock.patch.object(
            sys.modules[__name__],
            "observe_frozen_final_outputs",
            side_effect=AssertionError("bound-root non-target command reached full analysis"),
        ):
            fast_role.observe(clean_non_target_command, 41)
        require(
            fast_role.line_count == 41
            and fast_role.nonempty == 1
            and fast_role.root_seen == [True, True],
            "self_test_bound_root_non_target_prefilter",
            "stateful non-target fast path changed general accounting or root state",
        )
        checks += 1

        repeated = " ".join("--wrap=malloc" for _ in range(512))
        observed, count = scan_text_for_test(catalog, repeated, retain_limit=3)
        require(len(observed) == 3 and count == 512 and
                all(len(match.matched) <= MAX_MATCH_TEXT_CHARS for match in observed),
                "self_test_match_bound", "retained match storage/count is not bounded")
        checks += 1

        with tempfile.TemporaryDirectory(prefix="oai-r0-absence-selftest-") as temporary:
            root = Path(temporary)
            paths = write_valid_evidence(root, source_root, build_root)
            arguments = (
                "--build-ninja", str(paths["build_ninja"]),
                "--targets", str(paths["targets"]),
                "--commands", str(paths["commands"]),
                "--source-root", source_root,
                "--build-root", build_root,
            )
            status, payload, _ = invoke_for_test(arguments)
            provenance = payload.get("provenance", {})
            claim = payload.get("claim", {})
            require(status == 0 and payload.get("verdict") == "pass" and
                    payload.get("match_count") == 0 and
                    payload.get("expected_final_targets") == list(R0_FINAL_TARGETS) and
                    provenance.get("content_identity") == CONTENT_IDENTITY_CLAIM and
                    provenance.get("capture_origin") == CAPTURE_ORIGIN_CLAIM and
                    provenance.get("relative_final_outputs") == RELATIVE_OUTPUT_CLAIM and
                    claim.get("exit_zero_common_origin") == EXIT_ZERO_COMMON_ORIGIN_CLAIM and
                    claim.get("final_elf") == FINAL_ELF_REQUIREMENT,
                    "self_test_valid_gate", "complete clean evidence did not pass")
            checks += 1

            clean_command_lines = paths["commands"].read_text(
                encoding="utf-8"
            ).splitlines()
            full_gate_wrap_vectors = (
                tuple(vector for vector, _ in ADJACENT_QUOTING_WRAP_VECTORS)
                + REVIEWER_WRAP_ESCAPE_VECTORS
            )
            for vector in full_gate_wrap_vectors:
                wrap_commands = root / (
                    "commands-adjacent-wrap-" + hashlib.sha256(vector.encode()).hexdigest()[:12]
                    + ".txt"
                )
                candidate_lines = list(clean_command_lines)
                candidate_lines.append("/usr/bin/true " + vector)
                wrap_commands.write_text("\n".join(candidate_lines) + "\n",
                                         encoding="utf-8", newline="")
                wrap_arguments = list(arguments)
                wrap_arguments[5] = str(wrap_commands)
                status, payload, _ = invoke_for_test(wrap_arguments)
                require(status == 1 and payload.get("verdict") == "fail" and
                        any(match.get("catalog_id") == "gnu-wrap-option"
                            for match in payload.get("matches", [])),
                        "self_test_adjacent_quote_full_gate",
                        f"full gate accepted adjacent-quoted wrap syntax: {vector}")
                checks += 1

            root_sibling_vectors = (
                ("underscore", source_root + "_future"),
                ("hyphen", source_root + "-sibling"),
                ("dot", source_root + ".suffix"),
                ("alphanumeric", source_root + "X"),
                ("colon", source_root + ":sibling"),
                ("comma", source_root + ",sibling"),
                ("semicolon", source_root + ";sibling"),
                ("right-paren", source_root + ")sibling"),
                ("quoted-space", f'"{source_root} sibling"'),
                ("escaped-space", source_root + r"\ sibling"),
                ("ninja-space", source_root + "$ sibling"),
                ("ninja-colon", source_root + "$:sibling"),
            )
            pristine_build_text = paths["build_ninja"].read_text(encoding="utf-8")
            for label, sibling_source in root_sibling_vectors:
                sibling_commands = root / f"commands-root-sibling-{label}.txt"
                sibling_command_lines = list(clean_command_lines)
                sibling_command_lines[0] = sibling_command_lines[0].replace(
                    source_root, sibling_source, 1
                )
                sibling_commands.write_text(
                    "\n".join(sibling_command_lines) + "\n",
                    encoding="utf-8",
                    newline="",
                )
                sibling_arguments = list(arguments)
                sibling_arguments[5] = str(sibling_commands)
                status, payload, _ = invoke_for_test(sibling_arguments)
                require(status == 2 and payload.get("problem", {}).get("code") ==
                        "unbound_declared_root",
                        "self_test_root_sibling_full_gate",
                        f"commands evidence bound a forbidden root sibling: {label}")
                checks += 1

                sibling_build = root / f"build-root-sibling-{label}.ninja"
                sibling_build.write_text(
                    pristine_build_text.replace(source_root, sibling_source),
                    encoding="utf-8",
                    newline="",
                )
                sibling_build_arguments = list(arguments)
                sibling_build_arguments[1] = str(sibling_build)
                status, payload, _ = invoke_for_test(sibling_build_arguments)
                require(status == 2 and payload.get("problem", {}).get("code") ==
                        "unbound_declared_root",
                        "self_test_root_sibling_full_gate",
                        f"build evidence bound a forbidden root sibling: {label}")
                checks += 1

                build_sibling = sibling_source.replace(source_root, build_root, 1)
                sibling_build_root = root / f"build-build-root-sibling-{label}.ninja"
                sibling_build_root.write_text(
                    pristine_build_text.replace(build_root, build_sibling),
                    encoding="utf-8",
                    newline="",
                )
                sibling_build_root_arguments = list(arguments)
                sibling_build_root_arguments[1] = str(sibling_build_root)
                status, payload, _ = invoke_for_test(sibling_build_root_arguments)
                require(status == 2 and payload.get("problem", {}).get("code") ==
                        "unbound_declared_root",
                        "self_test_root_sibling_full_gate",
                        f"build evidence bound a forbidden build-root sibling: {label}")
                checks += 1

            for invalid_root in (
                "/source/with space",
                "/source/with'quote",
                "/source/with$dollar",
                "/source/with:colon",
                "/source/with,comma",
                "/source/with:semicolon",
                "/source/with)paren",
                "/source/../escape",
                "/source//double",
            ):
                invalid_root_arguments = list(arguments)
                invalid_root_arguments[7] = invalid_root
                status, payload, _ = invoke_for_test(invalid_root_arguments)
                require(status == 2 and payload.get("problem", {}).get("code") ==
                        "invalid_root",
                        "self_test_cli_root_domain",
                        f"CLI accepted a root outside the frozen domain: {invalid_root}")
                checks += 1

            alias_arguments = list(arguments)
            alias_arguments[3] = str(paths["build_ninja"])
            status, payload, _ = invoke_for_test(alias_arguments)
            require(status == 2 and payload.get("problem", {}).get("code") == "evidence_identity_alias",
                    "self_test_role_alias", "device/inode role alias was not rejected")
            checks += 1

            truncated_build = root / "build-truncated.ninja"
            truncated_build.write_bytes(paths["build_ninja"].read_bytes().removesuffix(b"default all\n"))
            truncated_arguments = list(arguments)
            truncated_arguments[1] = str(truncated_build)
            status, payload, _ = invoke_for_test(truncated_arguments)
            require(status == 2 and payload.get("problem", {}).get("code") == "truncated_build_ninja",
                    "self_test_truncation", "truncated build.ninja was not rejected")
            checks += 1

            truncated_targets = root / "targets-truncated.txt"
            truncated_targets.write_text("all: phony\nbuild.ninja: RERUN_CMAKE\nclean: CLEAN\n",
                                         encoding="utf-8", newline="")
            truncated_arguments = list(arguments)
            truncated_arguments[3] = str(truncated_targets)
            status, payload, _ = invoke_for_test(truncated_arguments)
            require(status == 2 and payload.get("problem", {}).get("code") == "incomplete_target_list",
                    "self_test_truncation", "truncated targets evidence was not rejected")
            checks += 1

            truncated_commands = root / "commands-truncated.txt"
            first_command = paths["commands"].read_text(encoding="utf-8").splitlines()[0]
            truncated_commands.write_text(first_command + "\n", encoding="utf-8", newline="")
            truncated_arguments = list(arguments)
            truncated_arguments[5] = str(truncated_commands)
            status, payload, _ = invoke_for_test(truncated_arguments)
            require(status == 2 and payload.get("problem", {}).get("code") == "incomplete_command_list",
                    "self_test_truncation", "truncated commands evidence was not rejected")
            checks += 1

            duplicate_commands = root / "commands-duplicate-output.txt"
            command_lines = paths["commands"].read_text(encoding="utf-8").splitlines()
            command_lines[1] = command_lines[1].replace(
                " && :", " -o unrelated && :", 1
            )
            duplicate_commands.write_text("\n".join(command_lines) + "\n",
                                          encoding="utf-8", newline="")
            duplicate_arguments = list(arguments)
            duplicate_arguments[5] = str(duplicate_commands)
            status, payload, _ = invoke_for_test(duplicate_arguments)
            require(status == 2 and payload.get("problem", {}).get("code") ==
                    "multiple_output_options",
                    "self_test_duplicate_final_output",
                    "a duplicate final-output token in one command was not rejected")
            checks += 1

            first_target = R0_FINAL_TARGETS[0]
            first_object = f"CMakeFiles/{first_target}.dir/selftest.c.o"
            valid_first_link = clean_command_lines[1]

            def require_invalid_command_case(
                label: str,
                adversarial_line: str,
                expected_code: str,
            ) -> None:
                adversarial_commands = root / f"commands-v8-adversarial-{label}.txt"
                adversarial_lines = list(clean_command_lines)
                adversarial_lines[1] = adversarial_line
                adversarial_commands.write_text(
                    "\n".join(adversarial_lines) + "\n",
                    encoding="utf-8",
                    newline="",
                )
                adversarial_arguments = list(arguments)
                adversarial_arguments[5] = str(adversarial_commands)
                observed_status, observed_payload, _ = invoke_for_test(adversarial_arguments)
                observed_code = observed_payload.get("problem", {}).get("code")
                require(observed_status == 2 and observed_code == expected_code,
                        "self_test_final_output_provenance",
                        f"{label}: expected invalid {expected_code}, observed "
                        f"status={observed_status} code={observed_code}")

            commented_fake = (
                f": && /usr/bin/cc --version # -I{source_root}/clean/safe "
                f"-L{build_root}/clean/safe -o {build_root}/{first_target} && :"
            )
            long_output_comment = (
                f": && /usr/bin/ld --output=/outside/{first_target} input.o "
                f"# -o {build_root}/{first_target} && :"
            )
            outside_then_echo = (
                f": && {R0_FINAL_LINK_DRIVER} {first_object} -o /outside/authentic-link "
                f"-Wl,-rpath,{build_root} -Wl,--start-group libsafe.a "
                f"-Wl,--end-group -lm && /usr/bin/echo -o {first_target}"
            )
            outside_then_printf = outside_then_echo.replace(
                "/usr/bin/echo", "/usr/bin/printf marker", 1
            )
            shell_adversarial_cases = (
                ("commented-fake", commented_fake),
                ("long-output-comment", long_output_comment),
                (
                    "redirection-held-output",
                    valid_first_link.replace(
                        f"-o {first_target}", "-o unrelated", 1
                    ).replace(
                        first_object,
                        f"> -o {first_target} {first_object}",
                        1,
                    ),
                ),
                (
                    "command-substitution-held-output",
                    valid_first_link.replace(
                        f"-o {first_target}", "-o unrelated", 1
                    ).replace(
                        first_object,
                        f"$(/usr/bin/true -o {first_target}) {first_object}",
                        1,
                    ),
                ),
                (
                    "backtick-held-output",
                    valid_first_link.replace(
                        f"-o {first_target}", "-o unrelated", 1
                    ).replace(
                        first_object,
                        f"`/usr/bin/true -o {first_target}` {first_object}",
                        1,
                    ),
                ),
                (
                    "process-substitution-held-output",
                    valid_first_link.replace(
                        f"-o {first_target}", "-o unrelated", 1
                    ).replace(
                        first_object,
                        f"<(/usr/bin/true -o {first_target}) {first_object}",
                        1,
                    ),
                ),
                (
                    "arithmetic-substitution",
                    valid_first_link.replace(first_object, f"$((1)) {first_object}", 1),
                ),
                (
                    "quoted-hash",
                    valid_first_link.replace(
                        "-DSELF_TEST=1", '"-DQUOTED_HASH=#" -DSELF_TEST=1', 1
                    ),
                ),
                (
                    "quoted-driver",
                    valid_first_link.replace(
                        R0_FINAL_LINK_DRIVER, f'"{R0_FINAL_LINK_DRIVER}"', 1
                    ),
                ),
                (
                    "quoted-output",
                    valid_first_link.replace(
                        f"-o {first_target}", f"-o '{first_target}'", 1
                    ),
                ),
                (
                    "split-quoted-output",
                    valid_first_link.replace(
                        f"-o {first_target}", "-o nr-soft'modem'", 1
                    ),
                ),
                (
                    "split-escaped-output",
                    valid_first_link.replace(
                        f"-o {first_target}", r"-o nr-soft\modem", 1
                    ),
                ),
                (
                    "escaped-driver",
                    valid_first_link.replace(
                        R0_FINAL_LINK_DRIVER, r"/usr/bin/c\+\+", 1
                    ),
                ),
            )
            for label, adversarial_line in shell_adversarial_cases:
                require_invalid_command_case(
                    label, adversarial_line, "forbidden_final_link_shell_syntax"
                )
                checks += 1

            for label, adversarial_line in (
                ("outside-then-echo", outside_then_echo),
                ("outside-then-printf", outside_then_printf),
                ("bare-link", valid_first_link.removeprefix(": && ").removesuffix(" && :")),
                ("post-link-action", valid_first_link.replace(" && :", " && echo done", 1)),
            ):
                require_invalid_command_case(
                    label, adversarial_line, "final_link_scaffold_not_exact"
                )
                checks += 1

            for mode in (
                "-c", "-S", "-E", "-fsyntax-only", "--version", "--help",
                "--target-help", "-M", "-MM", "-shared", "-r",
            ):
                require_invalid_command_case(
                    "mode-" + hashlib.sha256(mode.encode()).hexdigest()[:12],
                    valid_first_link.replace(
                        R0_FINAL_LINK_DRIVER,
                        f"{R0_FINAL_LINK_DRIVER} {mode}",
                        1,
                    ),
                    "compiler_non_link_mode",
                )
                checks += 1

            for label, replacement in (
                ("gcc-driver", "/usr/bin/cc"),
                ("cxx-alias", "/bin/c++"),
                ("ld-relocatable", "/usr/bin/ld -r"),
                ("ld-long-output", "/usr/bin/ld"),
            ):
                driver_line = valid_first_link.replace(
                    R0_FINAL_LINK_DRIVER, replacement, 1
                )
                if label == "ld-long-output":
                    driver_line = driver_line.replace(
                        f"-o {first_target}", f"--output={first_target}", 1
                    )
                require_invalid_command_case(
                    label, driver_line, "final_link_driver_not_exact"
                )
                checks += 1

            for mode in ("--version", "-c", "-E"):
                require_invalid_command_case(
                    "cc-mode-" + hashlib.sha256(mode.encode()).hexdigest()[:12],
                    valid_first_link.replace(
                        R0_FINAL_LINK_DRIVER, f"/usr/bin/cc {mode}", 1
                    ),
                    "final_link_driver_not_exact",
                )
                checks += 1

            for label, replacement, expected_code in (
                ("joined-o", f"-o{first_target}", "unsupported_final_output_spelling"),
                ("long-output", f"--output={first_target}",
                 "unsupported_final_output_spelling"),
                ("long-output-separated", f"--output {first_target}",
                 "unsupported_final_output_spelling"),
                ("wl-output", f"-Wl,-o,{first_target}",
                 "unsupported_final_output_spelling"),
                ("xlinker-output", f"-Xlinker -o -Xlinker {first_target}",
                 "unsupported_final_output_spelling"),
                ("absolute-outside", f"-o /outside/{first_target}",
                 "final_output_not_bare_basename"),
                ("absolute-declared-build", f"-o {build_root}/{first_target}",
                 "final_output_not_bare_basename"),
                ("dot-relative", f"-o ./{first_target}",
                 "final_output_not_bare_basename"),
                ("nested-relative", f"-o elsewhere/{first_target}",
                 "final_output_not_bare_basename"),
                ("trailing-slash", f"-o {first_target}/",
                 "final_output_not_bare_basename"),
                ("parent-component", f"-o link/../{first_target}",
                 "final_output_not_bare_basename"),
            ):
                require_invalid_command_case(
                    label,
                    valid_first_link.replace(f"-o {first_target}", replacement, 1),
                    expected_code,
                )
                checks += 1

            for label, inserted in (
                ("preobject-lgcc-lookalike", "-lgcc_s "),
                ("preobject-unrelated-library", "-lpthread "),
                ("dependency-wrong-target",
                 "-Wl,--dependency-file=CMakeFiles/other.dir/link.d "),
                ("dependency-parent-component",
                 "-Wl,--dependency-file=CMakeFiles/../nr-softmodem.dir/link.d "),
            ):
                require_invalid_command_case(
                    label,
                    valid_first_link.replace(
                        first_object + " ", inserted + first_object + " ", 1
                    ),
                    "unsupported_final_link_pre_option",
                )
                checks += 1

            for label, inserted, expected_code in (
                ("response-file", "@args.rsp ", "unsupported_final_link_token"),
                ("option-terminator", "-- ", "unsupported_final_link_token"),
                ("missing-owned-object", "CMakeFiles/other.dir/selftest.c.o ",
                 "missing_target_owned_object"),
                ("missing-object-block", "libinput.a ", "missing_final_link_input_block"),
            ):
                require_invalid_command_case(
                    label,
                    valid_first_link.replace(first_object + " ", inserted, 1),
                    expected_code,
                )
                checks += 1

            require_invalid_command_case(
                "noncontiguous-object-block",
                valid_first_link.replace(
                    first_object,
                    f"{first_object} libgap.a CMakeFiles/{first_target}.dir/second.c.o",
                    1,
                ),
                "invalid_final_link_input_block",
            )
            checks += 1

            require_invalid_command_case(
                "unsupported-post-argument",
                valid_first_link.replace(
                    f"-o {first_target} ", f"-o {first_target} --unsupported ", 1
                ),
                "unsupported_final_link_post_argument",
            )
            checks += 1

            require_invalid_command_case(
                "wrong-rpath-root",
                valid_first_link.replace(
                    f"-Wl,-rpath,{build_root}", "-Wl,-rpath,/outside", 1
                ),
                "missing_exact_build_rpath",
            )
            checks += 1

            unbound_commands = root / "commands-unbound-source-root.txt"
            unbound_lines = list(clean_command_lines)
            unbound_lines[0] = f"{R0_FINAL_LINK_DRIVER} -c safe.c -o safe.o"
            unbound_commands.write_text(
                "\n".join(unbound_lines) + "\n", encoding="utf-8", newline=""
            )
            unbound_arguments = list(arguments)
            unbound_arguments[5] = str(unbound_commands)
            status, payload, _ = invoke_for_test(unbound_arguments)
            require(status == 2 and payload.get("problem", {}).get("code") ==
                    "unbound_declared_root",
                    "self_test_unbound_evidence", "unbound commands evidence was not rejected")
            checks += 1

            symlink = root / "commands-link.txt"
            symlink.symlink_to(paths["commands"])
            expect_absence_error(
                "evidence_open_failed",
                lambda: scan_regular_evidence("commands", symlink, catalog, 4096),
            )
            checks += 1

            fifo = root / "commands.fifo"
            os.mkfifo(fifo)
            expect_absence_error(
                "evidence_not_regular",
                lambda: scan_regular_evidence("commands", fifo, catalog, 4096),
            )
            checks += 1

            overlong = root / "overlong-line.txt"
            overlong.write_bytes(b"x" * (MAX_LINE_BYTES + 1) + b"\n")
            expect_absence_error(
                "line_over_limit",
                lambda: scan_regular_evidence(
                    "commands", overlong, catalog, MAX_LINE_BYTES + 2
                ),
            )
            checks += 1

            invalid_utf8 = root / "invalid-utf8.txt"
            invalid_utf8.write_bytes(b"\xff\n")
            expect_absence_error(
                "non_utf8_evidence",
                lambda: scan_regular_evidence("commands", invalid_utf8, catalog, 4096),
            )
            checks += 1

            require(
                validate_line_bytes(b"safe\ttext", "commands", 7) == "safe\ttext"
                and validate_line_bytes(b"safe\r", "commands", 8) == "safe",
                "self_test_line_validation",
                "valid tab or terminal carriage return changed meaning",
            )
            checks += 1

            invalid_line_cases = (
                (b"safe\x00x", "commands line 9 contains control byte at column 5"),
                (b"safe\nx", "commands line 9 contains control byte at column 5"),
                (b"safe\rx", "commands line 9 contains embedded carriage return"),
                (b"safe\x7fx", "commands line 9 contains DEL at column 5"),
                (b"a\x7f\x00", "commands line 9 contains DEL at column 2"),
            )
            for raw_line, expected_message in invalid_line_cases:
                try:
                    validate_line_bytes(raw_line, "commands", 9)
                except AbsenceError as error:
                    require(
                        error.code == "control_character" and str(error) == expected_message,
                        "self_test_line_validation",
                        f"line validation changed precedence/message: {error.code}: {error}",
                    )
                else:
                    raise AbsenceError(
                        "self_test_line_validation",
                        f"invalid line was accepted: {raw_line!r}",
                    )
                checks += 1

            overlimit = root / "evidence-over-limit.txt"
            overlimit.write_bytes(b"x" * 4097)
            expect_absence_error(
                "evidence_size_invalid",
                lambda: scan_regular_evidence("commands", overlimit, catalog, 4096),
            )
            checks += 1

            unterminated = root / "unterminated.txt"
            unterminated.write_bytes(b"/usr/bin/true")
            expect_absence_error(
                "unterminated_evidence",
                lambda: scan_regular_evidence("commands", unterminated, catalog, 4096),
            )
            checks += 1

            with mock.patch.object(
                os, "read", side_effect=BlockingIOError(11, "injected would-block")
            ):
                expect_absence_error(
                    "evidence_read_would_block",
                    lambda: scan_regular_evidence("commands", paths["commands"], catalog, 4096),
                )
            checks += 1

        overflow_output = io.BytesIO()
        overflow_status = emit_result({"blob": "x" * (MAX_JSON_BYTES + 1)}, 0, overflow_output)
        overflow_payload = json.loads(overflow_output.getvalue().decode("utf-8"))
        require(overflow_status == 2 and overflow_payload.get("catalog_version") == CATALOG_VERSION and
                overflow_payload.get("problem", {}).get("code") == "result_over_limit" and
                overflow_payload.get("provenance", {}).get("capture_origin") ==
                CAPTURE_ORIGIN_CLAIM,
                "self_test_output_overflow", "output overflow did not produce bounded invalid exit 2")
        checks += 1

        status, payload, _ = invoke_for_test(("--max-evidence-bytes", "not-an-integer"))
        require(status == 2 and payload.get("catalog_version") == CATALOG_VERSION and
                payload.get("problem", {}).get("code") == "argument_parse_error",
                "self_test_cli_error", "argparse failure did not produce bounded JSON exit 2")
        checks += 1

        status, payload, _ = invoke_for_test(("--not-a-checker-option",))
        require(status == 2 and payload.get("catalog_version") == CATALOG_VERSION and
                payload.get("problem", {}).get("code") == "argument_parse_error",
                "self_test_cli_error", "unknown CLI option did not produce bounded JSON exit 2")
        checks += 1

        status, payload, _ = invoke_for_test(("--expected-final-target", "weaker-target"))
        require(status == 2 and payload.get("catalog_version") == CATALOG_VERSION and
                payload.get("problem", {}).get("code") == "argument_parse_error",
                "self_test_frozen_targets", "a weakening final-target override was accepted")
        checks += 1

        return emit_result(
            {
                **result_contract(),
                "catalog_digest": catalog_digest,
                "catalog_entries": len(catalog.entries),
                "catalog_version": CATALOG_VERSION,
                "checks": checks,
                "checks_interpretation": "nominal_cases_not_independent_proofs",
                "schema": SELF_TEST_SCHEMA,
                "verdict": "pass",
                "wrap_spellings_per_api": len(wrap_spellings),
            },
            0,
            stream,
        )
    except (AbsenceError, OSError) as error:
        code = getattr(error, "code", "self_test_io_error")
        return emit_result(
            {
                **result_contract(),
                "catalog_version": CATALOG_VERSION,
                "problem": {"code": code, "message": bounded_text(error)},
                "schema": SELF_TEST_SCHEMA,
                "verdict": "fail",
            },
            1,
            stream,
        )


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise AbsenceError("argument_parse_error", bounded_text(message))


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = JsonArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--build-ninja", type=Path)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--commands", type=Path)
    parser.add_argument("--source-root")
    parser.add_argument("--build-root")
    parser.add_argument("--max-evidence-bytes", type=int, default=DEFAULT_MAX_EVIDENCE_BYTES)
    return parser.parse_args(argv)


def result_contract() -> dict[str, object]:
    return {
        "claim": {
            "absence_scope": ABSENCE_SCOPE_CLAIM,
            "exit_zero_common_origin": EXIT_ZERO_COMMON_ORIGIN_CLAIM,
            "final_elf": FINAL_ELF_REQUIREMENT,
            "final_output_identity": FINAL_OUTPUT_IDENTITY_CLAIM,
            "gnu_wrap_absence": GNU_LD_WRAP_CLAIM,
            "gnu_wrap_shell_expansion": GNU_LD_WRAP_EXPANSION_CLAIM,
        },
        "final_link_model": {
            "containment": FINAL_OUTPUT_CONTAINMENT_CLAIM,
            "driver": R0_FINAL_LINK_DRIVER,
            "driver_claim": FINAL_LINK_DRIVER_CLAIM,
            "grammar": FINAL_LINK_GRAMMAR_VERSION,
            "scaffold": FINAL_LINK_SCAFFOLD_CLAIM,
        },
        "provenance": {
            "capture_origin": CAPTURE_ORIGIN_CLAIM,
            "content_identity": CONTENT_IDENTITY_CLAIM,
            "relative_final_outputs": RELATIVE_OUTPUT_CLAIM,
        },
        "root_model": {
            "domain": ROOT_DOMAIN_CLAIM,
            "grammar": ROOT_GRAMMAR_VERSION,
        },
        "wrap_model": {
            "expansion_constructed_spellings": GNU_LD_WRAP_EXPANSION_CLAIM,
            "expansion_example": GNU_LD_WRAP_EXPANSION_EXAMPLE,
            "grammar": GNU_LD_WRAP_GRAMMAR_VERSION,
            "reference_tool": GNU_LD_REFERENCE_VERSION,
        },
    }


def invalid_payload(error: AbsenceError | OSError, code: str | None = None) -> dict[str, object]:
    return {
        **result_contract(),
        "catalog_version": CATALOG_VERSION,
        "problem": {"code": code or getattr(error, "code", "io_error"),
                    "message": bounded_text(error)},
        "schema": SCHEMA,
        "verdict": "invalid",
    }


def run_gate(arguments: argparse.Namespace, stream: BinaryIO | None = None) -> int:
    if arguments.self_test:
        forbidden = (arguments.build_ninja, arguments.targets, arguments.commands,
                     arguments.source_root, arguments.build_root)
        if any(value is not None for value in forbidden):
            return emit_result(
                invalid_payload(AbsenceError(
                    "invalid_arguments", "--self-test cannot be combined with evidence inputs"
                )),
                2,
                stream,
            )
        return run_self_test(stream)

    try:
        require(arguments.build_ninja is not None, "missing_argument", "--build-ninja is required")
        require(arguments.targets is not None, "missing_argument", "--targets is required")
        require(arguments.commands is not None, "missing_argument", "--commands is required")
        require(arguments.source_root is not None, "missing_argument", "--source-root is required")
        require(arguments.build_root is not None, "missing_argument", "--build-root is required")
        require(0 < arguments.max_evidence_bytes <= HARD_MAX_EVIDENCE_BYTES,
                "invalid_limit",
                f"--max-evidence-bytes must be in 1..{HARD_MAX_EVIDENCE_BYTES}")

        source_root = normalized_root(arguments.source_root, "source_root")
        build_root = normalized_root(arguments.build_root, "build_root")
        require(source_root != build_root, "invalid_root",
                "source_root and build_root must identify different trees")
        catalog_entries = build_catalog()
        catalog_digest = validate_catalog_freeze(catalog_entries)
        catalog = Catalog(catalog_entries, (source_root, build_root))

        evidence: list[EvidenceSummary] = []
        reported: list[Match] = []
        total_matches = 0
        seen_identities: set[tuple[int, int]] = set()
        for role, path in (
            ("build_ninja", arguments.build_ninja),
            ("targets", arguments.targets),
            ("commands", arguments.commands),
        ):
            summary, matches, match_count = scan_regular_evidence(
                role,
                path,
                catalog,
                arguments.max_evidence_bytes,
                forbidden_identities=seen_identities,
                retain_limit=MAX_REPORTED_MATCHES - len(reported),
            )
            seen_identities.add((summary.device, summary.inode))
            evidence.append(summary)
            total_matches += match_count
            reported.extend(matches)

        verdict = "pass" if total_matches == 0 else "fail"
        return emit_result(
            {
                **result_contract(),
                "catalog_digest": catalog_digest,
                "catalog_entries": len(catalog.entries),
                "catalog_version": CATALOG_VERSION,
                "evidence": [item.as_dict() for item in evidence],
                "expected_final_targets": list(R0_FINAL_TARGETS),
                "match_count": total_matches,
                "matches": [item.as_dict() for item in reported],
                "matches_omitted": total_matches - len(reported),
                "roots": {"build": build_root, "source": source_root},
                "schema": SCHEMA,
                "verdict": verdict,
            },
            0 if total_matches == 0 else 1,
            stream,
        )
    except AbsenceError as error:
        return emit_result(invalid_payload(error), 2, stream)
    except OSError as error:
        return emit_result(invalid_payload(error, "io_error"), 2, stream)


def main(argv: Sequence[str] | None = None, stream: BinaryIO | None = None) -> int:
    try:
        arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    except AbsenceError as error:
        return emit_result(invalid_payload(error), 2, stream)
    return run_gate(arguments, stream)


if __name__ == "__main__":
    raise SystemExit(main())
