# SPDX-License-Identifier: LicenseRef-CSSL-1.0

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_executable(test_oai_memprof_wire test_oai_memprof_wire.c)
target_link_libraries(test_oai_memprof_wire PRIVATE oai_memprof_wire)
add_dependencies(tests test_oai_memprof_wire)

add_test(NAME oai_memprof_wire_c
         COMMAND test_oai_memprof_wire)

add_test(NAME oai_memprof_wire_cross_literal
         COMMAND ${CMAKE_COMMAND}
                 -DEMITTER=$<TARGET_FILE:test_oai_memprof_wire>
                 -DEXPECTED=${OPENAIR_DIR}/tools/profiling/memory/tests/event_record_v1_literal.hex
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/compare_literal.cmake)

add_test(NAME oai_memprof_wire_python
         COMMAND ${Python3_EXECUTABLE}
                 -B
                 -m unittest
                 -v
                 tools.profiling.memory.tests.test_oai_memprof_wire)
set_tests_properties(oai_memprof_wire_python PROPERTIES WORKING_DIRECTORY ${OPENAIR_DIR})

add_executable(test_oai_memprof_active_core test_oai_memprof_active_core.c)
target_compile_features(test_oai_memprof_active_core PRIVATE c_std_11)
target_compile_options(test_oai_memprof_active_core PRIVATE -Wall -Wextra -Werror -Wpedantic)
target_link_libraries(test_oai_memprof_active_core PRIVATE oai_memprof_active_core pthread)
add_dependencies(tests test_oai_memprof_active_core)

add_test(NAME oai_memprof_active_core_c
         COMMAND test_oai_memprof_active_core)
set_tests_properties(oai_memprof_active_core_c PROPERTIES
                     LABELS "memprof;active-core"
                     TIMEOUT 30)

add_executable(test_oai_memprof_active_wrappers test_oai_memprof_active_wrappers.c)
target_compile_features(test_oai_memprof_active_wrappers PRIVATE c_std_11)
target_compile_options(test_oai_memprof_active_wrappers PRIVATE
                       -Wall -Wextra -Werror -Wpedantic -fPIC)
target_link_libraries(test_oai_memprof_active_wrappers PRIVATE
                      oai_memprof_active_wrap_c
                      oai_memprof_active_runtime)
set_target_properties(test_oai_memprof_active_wrappers PROPERTIES
                      BUILD_RPATH "${CMAKE_BINARY_DIR}/memprof"
                      POSITION_INDEPENDENT_CODE ON)
target_link_options(test_oai_memprof_active_wrappers PRIVATE "LINKER:-pie")
add_dependencies(tests test_oai_memprof_active_wrappers)

add_test(NAME oai_memprof_active_wrappers_c
         COMMAND test_oai_memprof_active_wrappers)
set_tests_properties(oai_memprof_active_wrappers_c PROPERTIES
                     LABELS "memprof;active-wrappers"
                     TIMEOUT 30)

add_executable(test_oai_memprof_clock test_oai_memprof_clock.c)
target_compile_features(test_oai_memprof_clock PRIVATE c_std_11)
target_compile_options(test_oai_memprof_clock PRIVATE -Wall -Wextra -Werror -Wpedantic)
target_link_libraries(test_oai_memprof_clock PRIVATE oai_memprof_clock)
add_dependencies(tests test_oai_memprof_clock)

add_test(NAME oai_memprof_clock_c COMMAND test_oai_memprof_clock)
set_tests_properties(oai_memprof_clock_c PROPERTIES
                     LABELS "memprof;clock"
                     SKIP_RETURN_CODE 77
                     TIMEOUT 30)

add_executable(test_oai_memprof_stream_writer test_oai_memprof_stream_writer.c)
target_compile_features(test_oai_memprof_stream_writer PRIVATE c_std_11)
target_compile_options(test_oai_memprof_stream_writer PRIVATE -Wall -Wextra -Werror -Wpedantic)
target_link_libraries(test_oai_memprof_stream_writer PRIVATE oai_memprof_stream_writer)
target_link_options(test_oai_memprof_stream_writer PRIVATE
                    "LINKER:--wrap=write"
                    "LINKER:--wrap=pthread_join"
                    "LINKER:--wrap=fsync")
set_target_properties(test_oai_memprof_stream_writer PROPERTIES
                      BUILD_RPATH "${CMAKE_BINARY_DIR}/memprof")
add_dependencies(tests test_oai_memprof_stream_writer)

foreach(mode positive counters timer short failure join-failure)
  add_test(NAME oai_memprof_stream_writer_${mode}_c
           COMMAND test_oai_memprof_stream_writer
                   ${mode}
                   ${CMAKE_CURRENT_SOURCE_DIR}/active_stream_opening_v1.hex)
  set_tests_properties(oai_memprof_stream_writer_${mode}_c PROPERTIES
                       LABELS "memprof;stream-writer"
                       SKIP_RETURN_CODE 77
                       TIMEOUT 30)
endforeach()

add_executable(test_oai_memprof_stream_finalizer test_oai_memprof_stream_finalizer.c)
target_compile_features(test_oai_memprof_stream_finalizer PRIVATE c_std_11)
target_compile_options(test_oai_memprof_stream_finalizer PRIVATE -Wall -Wextra -Werror -Wpedantic -fPIC)
target_link_libraries(test_oai_memprof_stream_finalizer PRIVATE oai_memprof_stream_finalizer)
target_link_options(test_oai_memprof_stream_finalizer PRIVATE
                    "LINKER:--wrap=pwrite"
                    "LINKER:--wrap=oai_memprof_active_runtime_snapshot_v1"
                    "LINKER:--wrap=oai_memprof_active_runtime_complete_v1")
set_target_properties(test_oai_memprof_stream_finalizer PROPERTIES
                      BUILD_RPATH "${CMAKE_BINARY_DIR}/memprof"
                      POSITION_INDEPENDENT_CODE ON)
add_dependencies(tests test_oai_memprof_stream_finalizer)

foreach(mode positive offline short failure corrupt mismatch identity)
  add_test(NAME oai_memprof_stream_finalizer_${mode}_c
           COMMAND test_oai_memprof_stream_finalizer
                   ${mode}
                   ${Python3_EXECUTABLE}
                   ${CMAKE_CURRENT_SOURCE_DIR}/validate_oai_memprof_stream_finalizer.py)
  set_tests_properties(oai_memprof_stream_finalizer_${mode}_c PROPERTIES
                       ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
                       LABELS "memprof;stream-finalizer"
                       SKIP_RETURN_CODE 77
                       TIMEOUT 30
                       WORKING_DIRECTORY ${OPENAIR_DIR})
endforeach()

add_executable(test_oai_memprof_container_wire test_oai_memprof_container_wire.c)
target_link_libraries(test_oai_memprof_container_wire PRIVATE oai_memprof_container_wire)
add_dependencies(tests test_oai_memprof_container_wire)

add_test(NAME oai_memprof_container_wire_c
         COMMAND test_oai_memprof_container_wire)

add_test(NAME oai_memprof_container_wire_python
         COMMAND ${Python3_EXECUTABLE}
                 -B
                 -m unittest
                 -v
                 tools.profiling.memory.tests.test_oai_memprof_container_wire)
set_tests_properties(oai_memprof_container_wire_python PROPERTIES WORKING_DIRECTORY ${OPENAIR_DIR})

set(OAI_MEMPROF_CATALOG_V1_TESTS
    tools/profiling/memory/catalog_v1/callsite/tests/test_callsite_catalog_v1.py
    tools/profiling/memory/catalog_v1/config/tests/test_effective_config_v1.py
    tools/profiling/memory/catalog_v1/coverage/tests/test_coverage_catalog_v1.py
    tools/profiling/memory/catalog_v1/diagnostics/tests/test_diagnostic_instance_v1.py
    tools/profiling/memory/catalog_v1/integration/test_archive_semantic_verifier_v1.py
    tools/profiling/memory/catalog_v1/runtime/tests/test_runtime_catalog_v1.py
    tools/profiling/memory/catalog_v1/sampling/tests/test_selection_rule_v1.py
    tools/profiling/memory/catalog_v1/semantic/tests/test_event_classifier_v1.py
    tools/profiling/memory/catalog_v1/semantic/tests/test_semantic_catalog_v1.py
    tools/profiling/memory/catalog_v1/status/test_status_chain_v1.py)
add_test(NAME oai_memprof_catalog_v1_python
         COMMAND ${Python3_EXECUTABLE}
                 -B
                 -m unittest
                 -v
                 ${OAI_MEMPROF_CATALOG_V1_TESTS})
set_tests_properties(oai_memprof_catalog_v1_python PROPERTIES
                     ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
                     LABELS "memprof;catalog;archive-verifier"
                     TIMEOUT 90
                     WORKING_DIRECTORY ${OPENAIR_DIR})

add_executable(test_oai_memprof_process_handoff test_oai_memprof_process_handoff.c)
target_compile_features(test_oai_memprof_process_handoff PRIVATE c_std_11)
target_compile_options(test_oai_memprof_process_handoff PRIVATE -Wall -Wextra -Werror -Wpedantic)
target_link_libraries(test_oai_memprof_process_handoff PRIVATE oai_memprof_process_handoff)
set_target_properties(test_oai_memprof_process_handoff PROPERTIES BUILD_RPATH "${CMAKE_BINARY_DIR}/memprof")
add_dependencies(tests test_oai_memprof_process_handoff)

add_test(NAME oai_memprof_process_handoff_c
         COMMAND test_oai_memprof_process_handoff
                 ${CMAKE_CURRENT_SOURCE_DIR}/process_handoff_v1.hex)
set_tests_properties(oai_memprof_process_handoff_c PROPERTIES
                     LABELS "memprof;process-handoff"
                     TIMEOUT 30)

add_test(NAME oai_memprof_process_handoff_python
         COMMAND ${Python3_EXECUTABLE}
                 -B
                 -m unittest
                 -v
                 tools.profiling.memory.tests.test_oai_memprof_process_handoff)
set_tests_properties(oai_memprof_process_handoff_python PROPERTIES
                     ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
                     LABELS "memprof;process-handoff"
                     TIMEOUT 30
                     WORKING_DIRECTORY ${OPENAIR_DIR})

add_executable(test_oai_memprof_process_session test_oai_memprof_process_session.c)
target_compile_features(test_oai_memprof_process_session PRIVATE c_std_11)
target_compile_options(test_oai_memprof_process_session PRIVATE -Wall -Wextra -Werror -Wpedantic)
target_link_libraries(test_oai_memprof_process_session PRIVATE oai_memprof_process_session)
target_link_options(test_oai_memprof_process_session PRIVATE
                    "LINKER:--wrap=oai_memprof_stream_writer_finish_v1")
set_target_properties(test_oai_memprof_process_session PROPERTIES BUILD_RPATH "${CMAKE_BINARY_DIR}/memprof")
add_dependencies(tests test_oai_memprof_process_session)

foreach(mode positive existing writer-error)
  add_test(NAME oai_memprof_process_session_${mode}_c
           COMMAND test_oai_memprof_process_session ${mode})
  set_tests_properties(oai_memprof_process_session_${mode}_c PROPERTIES
                       LABELS "memprof;process-session"
                       SKIP_RETURN_CODE 77
                       TIMEOUT 30)
endforeach()

add_executable(test_oai_memprof_softmodem_session
               test_oai_memprof_softmodem_session.c)
target_compile_features(test_oai_memprof_softmodem_session PRIVATE c_std_11)
target_compile_options(test_oai_memprof_softmodem_session PRIVATE
                       -Wall -Wextra -Werror -Wpedantic)
target_link_libraries(test_oai_memprof_softmodem_session PRIVATE
                      oai_memprof_softmodem_session)
set_target_properties(test_oai_memprof_softmodem_session PROPERTIES
                      BUILD_RPATH "${CMAKE_BINARY_DIR}/memprof")
add_dependencies(tests test_oai_memprof_softmodem_session)

foreach(mode disabled partial legacy-path role-mismatch positive sampled fd-roots-replaced configuration-mismatch insecure-streams)
  add_test(NAME oai_memprof_softmodem_session_${mode}_c
           COMMAND test_oai_memprof_softmodem_session ${mode})
  set_tests_properties(oai_memprof_softmodem_session_${mode}_c PROPERTIES
                       LABELS "memprof;softmodem-session"
                       SKIP_RETURN_CODE 77
                       TIMEOUT 30)
endforeach()

add_test(NAME oai_memprof_softmodem_launcher_python
         COMMAND ${Python3_EXECUTABLE}
                 -B
                 ${OPENAIR_DIR}/tools/profiling/memory/tests/test_oai_memprof_softmodem_launcher.py
                 -v)
set_tests_properties(oai_memprof_softmodem_launcher_python PROPERTIES
                     ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
                     LABELS "memprof;softmodem-launcher;build-evidence"
                     TIMEOUT 30
                     WORKING_DIRECTORY ${OPENAIR_DIR})

add_test(NAME oai_memprof_trusted_release_authority_python
         COMMAND ${Python3_EXECUTABLE}
                 -B
                 ${OPENAIR_DIR}/tools/profiling/memory/tests/test_oai_memprof_trusted_release_authority.py
                 -v)
set_tests_properties(oai_memprof_trusted_release_authority_python PROPERTIES
                     ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
                     LABELS "memprof;trusted-release-authority;git"
                     TIMEOUT 30
                     WORKING_DIRECTORY ${OPENAIR_DIR})

add_test(NAME oai_memprof_build_evidence_python
         COMMAND ${Python3_EXECUTABLE}
                 -B
                 ${OPENAIR_DIR}/tools/profiling/memory/tests/test_oai_memprof_build_evidence.py
                 -v)
set_tests_properties(oai_memprof_build_evidence_python PROPERTIES
                     ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
                     LABELS "memprof;build-evidence"
                     TIMEOUT 60
                     WORKING_DIRECTORY ${OPENAIR_DIR})

add_dependencies(tests oai_memprof_archive_append)

add_executable(test_oai_memprof_archive_append
               test_oai_memprof_archive_append.c
               ../oai_memprof_stream_finalizer.c)
target_compile_features(test_oai_memprof_archive_append PRIVATE c_std_11)
target_include_directories(test_oai_memprof_archive_append PRIVATE ${OPENAIR_DIR})
target_compile_options(test_oai_memprof_archive_append PRIVATE
                       -Wall -Wextra -Werror -Wpedantic -fno-pie)
target_compile_definitions(test_oai_memprof_archive_append PRIVATE
                           OAI_MEMPROF_STREAM_FINALIZER_OFFLINE_ONLY=1)
target_link_libraries(test_oai_memprof_archive_append PRIVATE
                      oai_memprof_process_handoff
                      oai_memprof_container_wire
                      oai_memprof_wire)
target_link_options(test_oai_memprof_archive_append PRIVATE -static -no-pie)
set_target_properties(test_oai_memprof_archive_append PROPERTIES
                      BUILD_RPATH ""
                      INSTALL_RPATH ""
                      SKIP_BUILD_RPATH TRUE
                      SKIP_INSTALL_RPATH TRUE
                      INTERPROCEDURAL_OPTIMIZATION FALSE
                      POSITION_INDEPENDENT_CODE OFF)
add_dependencies(tests test_oai_memprof_archive_append)

add_test(NAME oai_memprof_archive_append_c
         COMMAND test_oai_memprof_archive_append)
set_tests_properties(oai_memprof_archive_append_c PROPERTIES
                     LABELS "memprof;archive-appender"
                     TIMEOUT 30)

set(OAI_MEMPROF_ARCHIVE_PRODUCER_MAP
    "${CMAKE_CURRENT_BINARY_DIR}/test_oai_memprof_archive_producer.map")
add_executable(test_oai_memprof_archive_producer test_oai_memprof_archive_producer.c)
target_compile_features(test_oai_memprof_archive_producer PRIVATE c_std_11)
target_compile_options(test_oai_memprof_archive_producer PRIVATE
                       -Wall -Wextra -Werror -Wpedantic
                       -fno-builtin-calloc -fno-builtin-free
                       -fno-builtin-malloc -fno-builtin-realloc)
target_link_libraries(test_oai_memprof_archive_producer PRIVATE
                      oai_memprof_process_session
                      oai_memprof_active_wrap_c
                      oai_memprof_active_runtime)
target_link_options(test_oai_memprof_archive_producer PRIVATE
                    "LINKER:--wrap=malloc"
                    "LINKER:--wrap=calloc"
                    "LINKER:--wrap=realloc"
                    "LINKER:--wrap=free"
                    "LINKER:-Map,${OAI_MEMPROF_ARCHIVE_PRODUCER_MAP}"
                    "LINKER:--cref")
set_target_properties(test_oai_memprof_archive_producer PROPERTIES
                      BUILD_RPATH "${CMAKE_BINARY_DIR}/memprof"
                      INTERPROCEDURAL_OPTIMIZATION FALSE)
add_dependencies(tests test_oai_memprof_archive_producer)

add_test(NAME oai_memprof_archive_composer_python
         COMMAND ${Python3_EXECUTABLE}
                 -B
                 ${OPENAIR_DIR}/tools/profiling/memory/tests/test_oai_memprof_archive_composer.py
                 -v)
set_tests_properties(oai_memprof_archive_composer_python PROPERTIES
                     ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
                     LABELS "memprof;archive-composer;build-evidence"
                     TIMEOUT 180
                     WORKING_DIRECTORY ${OPENAIR_DIR})

if(NOT CMAKE_GENERATOR STREQUAL "Ninja")
  message(FATAL_ERROR "The frozen R0 final-link gate requires the single-config Ninja generator")
endif()

if(NOT CMAKE_C_COMPILER_ID STREQUAL "GNU")
  message(FATAL_ERROR "The frozen R0 fixture gate requires the GNU C compiler and driver")
endif()

set(OAI_MEMPROF_R0_FIXTURE_COMPILE_OPTIONS
    -Wall
    -Wextra
    -Werror
    -Wpedantic
    -fno-builtin-calloc
    -fno-builtin-free
    -fno-builtin-malloc
    -fno-builtin-realloc
    ${OAI_MEMPROF_R0_PASSIVE_COMPILE_OPTIONS})

function(oai_memprof_r0_configure_c_target target)
  target_compile_features(${target} PRIVATE c_std_11)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${OPENAIR_DIR})
  target_compile_options(${target} PRIVATE ${OAI_MEMPROF_R0_FIXTURE_COMPILE_OPTIONS})
  set_target_properties(${target} PROPERTIES
                        C_COMPILER_LAUNCHER ""
                        C_VISIBILITY_PRESET hidden
                        INTERPROCEDURAL_OPTIMIZATION FALSE)
endfunction()

function(oai_memprof_r0_add_map target map_path)
  target_link_options(${target} PRIVATE
                      "LINKER:-Map,${map_path}"
                      "LINKER:--cref")
  set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
endfunction()

function(oai_memprof_r0_apply_wrap target)
  target_link_options(${target} PRIVATE
                      "LINKER:--wrap=malloc"
                      "LINKER:--wrap=calloc"
                      "LINKER:--wrap=realloc"
                      "LINKER:--wrap=free")
  target_link_libraries(${target} PRIVATE oai_memprof_wrap_c oai_memprof_runtime)
endfunction()

# The absolute scripted oracle uses no public allocator interposition. A00 calls
# the scripted __real_* functions directly; A01 calls the exact production
# wrappers, which resolve to those same scripted functions.
set(OAI_MEMPROF_R0_SCRIPTED_SOURCES
    r0_raw_emit.c
    r0_scripted_backend.c
    r0_scripted_oracle.c)

add_executable(test_oai_memprof_r0_scripted_a00 ${OAI_MEMPROF_R0_SCRIPTED_SOURCES})
oai_memprof_r0_configure_c_target(test_oai_memprof_r0_scripted_a00)
target_compile_definitions(test_oai_memprof_r0_scripted_a00 PRIVATE OAI_MEMPROF_R0_ORACLE_MODE=0)

add_executable(test_oai_memprof_r0_scripted_a01 ${OAI_MEMPROF_R0_SCRIPTED_SOURCES})
oai_memprof_r0_configure_c_target(test_oai_memprof_r0_scripted_a01)
target_compile_definitions(test_oai_memprof_r0_scripted_a01 PRIVATE OAI_MEMPROF_R0_ORACLE_MODE=1)
target_link_libraries(test_oai_memprof_r0_scripted_a01 PRIVATE oai_memprof_wrap_c oai_memprof_runtime)

set(OAI_MEMPROF_R0_MUTATION_SPECS
    "duplicate_real|duplicate-real|1"
    "operand|operand|2"
    "errno|errno|3"
    "result|result|4"
    "suppress_free_null|suppress-free-null|5"
    "context|context|6")
set(OAI_MEMPROF_R0_MUTATION_TARGETS)
set(OAI_MEMPROF_R0_MUTATION_TESTS)
foreach(specification IN LISTS OAI_MEMPROF_R0_MUTATION_SPECS)
  string(REPLACE "|" ";" fields "${specification}")
  list(GET fields 0 target_suffix)
  list(GET fields 1 oracle_name)
  list(GET fields 2 mutation_value)
  set(target "test_oai_memprof_r0_mutant_${target_suffix}")
  add_executable(${target}
                 ${OAI_MEMPROF_R0_SCRIPTED_SOURCES}
                 r0_scripted_passthrough.c)
  oai_memprof_r0_configure_c_target(${target})
  target_compile_definitions(${target} PRIVATE
                             OAI_MEMPROF_R0_ORACLE_MODE=1
                             OAI_MEMPROF_R0_MUTATION=${mutation_value})
  set(test_name "oai_memprof_r0_mutation_${target_suffix}")
  add_test(NAME ${test_name}
           COMMAND ${Python3_EXECUTABLE}
                   -B
                   ${CMAKE_CURRENT_SOURCE_DIR}/validate_r0_scripted_oracle.py
                   launch-mutation
                   ${oracle_name}
                   $<TARGET_FILE:${target}>)
  list(APPEND OAI_MEMPROF_R0_MUTATION_TARGETS ${target})
  list(APPEND OAI_MEMPROF_R0_MUTATION_TESTS ${test_name})
endforeach()

add_test(NAME oai_memprof_r0_scripted_pair
         COMMAND ${Python3_EXECUTABLE}
                 -B
                 ${CMAKE_CURRENT_SOURCE_DIR}/validate_r0_scripted_oracle.py
                 launch-pair
                 $<TARGET_FILE:test_oai_memprof_r0_scripted_a00>
                 $<TARGET_FILE:test_oai_memprof_r0_scripted_a01>)
add_test(NAME oai_memprof_r0_scripted_bounds
         COMMAND ${Python3_EXECUTABLE}
                 -B
                 ${CMAKE_CURRENT_SOURCE_DIR}/validate_r0_scripted_oracle.py
                 bounds-self-test)

# Compile each real-libc callsite object once and reuse it byte-for-byte in both
# final-link treatments. GNU wrapping is then applied independently to the
# executable and MODULE because it cannot cross an already linked ELF boundary.
add_library(oai_memprof_r0_actual_exe_object OBJECT r0_actual_fixture_exe.c)
add_library(oai_memprof_r0_actual_dso_object OBJECT r0_actual_fixture_dso.c)
oai_memprof_r0_configure_c_target(oai_memprof_r0_actual_exe_object)
oai_memprof_r0_configure_c_target(oai_memprof_r0_actual_dso_object)
set_target_properties(oai_memprof_r0_actual_exe_object
                      oai_memprof_r0_actual_dso_object
                      PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(oai_memprof_r0_actual_dso_a00 MODULE
            $<TARGET_OBJECTS:oai_memprof_r0_actual_dso_object>)
add_library(oai_memprof_r0_actual_dso_a01 MODULE
            $<TARGET_OBJECTS:oai_memprof_r0_actual_dso_object>)
add_executable(test_oai_memprof_r0_actual_a00
               $<TARGET_OBJECTS:oai_memprof_r0_actual_exe_object>)
add_executable(test_oai_memprof_r0_actual_a01
               $<TARGET_OBJECTS:oai_memprof_r0_actual_exe_object>)

set_target_properties(oai_memprof_r0_actual_dso_a00
                      oai_memprof_r0_actual_dso_a01
                      test_oai_memprof_r0_actual_a00
                      test_oai_memprof_r0_actual_a01
                      PROPERTIES
                      INTERPROCEDURAL_OPTIMIZATION FALSE
                      LINKER_LANGUAGE C
                      BUILD_RPATH "\$ORIGIN/.."
                      BUILD_WITH_INSTALL_RPATH TRUE
                      INSTALL_RPATH "\$ORIGIN/.."
                      SKIP_BUILD_RPATH FALSE
                      SKIP_INSTALL_RPATH FALSE)
set_target_properties(test_oai_memprof_r0_actual_a00
                      test_oai_memprof_r0_actual_a01
                      PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(oai_memprof_r0_actual_dso_a00 PRIVATE ${CMAKE_DL_LIBS})
target_link_libraries(oai_memprof_r0_actual_dso_a01 PRIVATE ${CMAKE_DL_LIBS})
target_link_libraries(test_oai_memprof_r0_actual_a00 PRIVATE ${CMAKE_DL_LIBS})
target_link_libraries(test_oai_memprof_r0_actual_a01 PRIVATE ${CMAKE_DL_LIBS})
target_link_options(oai_memprof_r0_actual_dso_a00 PRIVATE ${OAI_MEMPROF_R0_PASSIVE_COMPILE_OPTIONS})
target_link_options(oai_memprof_r0_actual_dso_a01 PRIVATE ${OAI_MEMPROF_R0_PASSIVE_COMPILE_OPTIONS})
target_link_options(test_oai_memprof_r0_actual_a00 PRIVATE ${OAI_MEMPROF_R0_PASSIVE_COMPILE_OPTIONS})
target_link_options(test_oai_memprof_r0_actual_a01 PRIVATE ${OAI_MEMPROF_R0_PASSIVE_COMPILE_OPTIONS})
target_link_options(oai_memprof_r0_actual_dso_a00 PRIVATE "LINKER:-z,defs")
target_link_options(oai_memprof_r0_actual_dso_a01 PRIVATE "LINKER:-z,defs")
target_link_options(test_oai_memprof_r0_actual_a00 PRIVATE "LINKER:-pie")
target_link_options(test_oai_memprof_r0_actual_a01 PRIVATE "LINKER:-pie")

set(OAI_MEMPROF_R0_A00_EXE_MAP "${CMAKE_CURRENT_BINARY_DIR}/oai_memprof_r0_a00_exe.map")
set(OAI_MEMPROF_R0_A01_EXE_MAP "${CMAKE_CURRENT_BINARY_DIR}/oai_memprof_r0_a01_exe.map")
set(OAI_MEMPROF_R0_A00_DSO_MAP "${CMAKE_CURRENT_BINARY_DIR}/oai_memprof_r0_a00_dso.map")
set(OAI_MEMPROF_R0_A01_DSO_MAP "${CMAKE_CURRENT_BINARY_DIR}/oai_memprof_r0_a01_dso.map")
oai_memprof_r0_add_map(test_oai_memprof_r0_actual_a00 "${OAI_MEMPROF_R0_A00_EXE_MAP}")
oai_memprof_r0_add_map(test_oai_memprof_r0_actual_a01 "${OAI_MEMPROF_R0_A01_EXE_MAP}")
oai_memprof_r0_add_map(oai_memprof_r0_actual_dso_a00 "${OAI_MEMPROF_R0_A00_DSO_MAP}")
oai_memprof_r0_add_map(oai_memprof_r0_actual_dso_a01 "${OAI_MEMPROF_R0_A01_DSO_MAP}")
oai_memprof_r0_apply_wrap(test_oai_memprof_r0_actual_a01)
oai_memprof_r0_apply_wrap(oai_memprof_r0_actual_dso_a01)

add_dependencies(test_oai_memprof_r0_actual_a00 oai_memprof_r0_actual_dso_a00)
add_dependencies(test_oai_memprof_r0_actual_a01 oai_memprof_r0_actual_dso_a01)

add_test(NAME oai_memprof_r0_actual_a00
         COMMAND $<TARGET_FILE:test_oai_memprof_r0_actual_a00>
                 --dso
                 $<TARGET_FILE:oai_memprof_r0_actual_dso_a00>
                 --runtime
                 absent)
add_test(NAME oai_memprof_r0_actual_a01
         COMMAND $<TARGET_FILE:test_oai_memprof_r0_actual_a01>
                 --dso
                 $<TARGET_FILE:oai_memprof_r0_actual_dso_a01>
                 --runtime
                 present-off
                 --runtime-path
                 $<TARGET_FILE:oai_memprof_runtime>)
add_test(NAME oai_memprof_r0_actual_differential
         COMMAND ${Python3_EXECUTABLE}
                 -B
                 ${CMAKE_CURRENT_SOURCE_DIR}/run_r0_actual_differential.py
                 --a00-exe
                 $<TARGET_FILE:test_oai_memprof_r0_actual_a00>
                 --a00-dso
                 $<TARGET_FILE:oai_memprof_r0_actual_dso_a00>
                 --a01-exe
                 $<TARGET_FILE:test_oai_memprof_r0_actual_a01>
                 --a01-dso
                 $<TARGET_FILE:oai_memprof_r0_actual_dso_a01>
                 --runtime-path
                 $<TARGET_FILE:oai_memprof_runtime>
                 --timeout-seconds
                 10
                 --max-output-bytes
                 4096)

add_test(NAME oai_memprof_r0_elf_validator_selftest
         COMMAND ${Python3_EXECUTABLE}
                 -B
                 ${CMAKE_CURRENT_SOURCE_DIR}/check_oai_memprof_r0_elf.py
                 --self-test)
add_test(NAME oai_memprof_r0_harness_selftest
         COMMAND ${Python3_EXECUTABLE}
                 -B
                 ${CMAKE_CURRENT_SOURCE_DIR}/test_r0_harness.py)
add_test(NAME oai_memprof_r0_absence_validator_selftest
         COMMAND ${Python3_EXECUTABLE}
                 -B
                 ${CMAKE_CURRENT_SOURCE_DIR}/check_oai_memprof_r0_absence.py
                 --self-test)
add_test(NAME oai_memprof_r0_elf
         COMMAND ${Python3_EXECUTABLE}
                 -B
                 ${CMAKE_CURRENT_SOURCE_DIR}/check_oai_memprof_r0_elf.py
                 --a00-exe
                 $<TARGET_FILE:test_oai_memprof_r0_actual_a00>
                 --a01-exe
                 $<TARGET_FILE:test_oai_memprof_r0_actual_a01>
                 --a00-dso
                 $<TARGET_FILE:oai_memprof_r0_actual_dso_a00>
                 --a01-dso
                 $<TARGET_FILE:oai_memprof_r0_actual_dso_a01>
                 --runtime
                 $<TARGET_FILE:oai_memprof_runtime>
                 --wrapper-archive
                 $<TARGET_FILE:oai_memprof_wrap_c>
                 --a00-exe-map
                 ${OAI_MEMPROF_R0_A00_EXE_MAP}
                 --a01-exe-map
                 ${OAI_MEMPROF_R0_A01_EXE_MAP}
                 --a00-dso-map
                 ${OAI_MEMPROF_R0_A00_DSO_MAP}
                 --a01-dso-map
                 ${OAI_MEMPROF_R0_A01_DSO_MAP}
                 --build-dir
                 ${CMAKE_BINARY_DIR}
                 --source-dir
                 ${OPENAIR_DIR}
                 --ninja-target
                 test_oai_memprof_r0_actual_a00
                 --ninja-target
                 test_oai_memprof_r0_actual_a01
                 --ninja-target
                 oai_memprof_r0_actual_dso_a00
                 --ninja-target
                 oai_memprof_r0_actual_dso_a01)

set(OAI_MEMPROF_R0_TESTS
    oai_memprof_r0_scripted_pair
    oai_memprof_r0_scripted_bounds
    oai_memprof_r0_actual_a00
    oai_memprof_r0_actual_a01
    oai_memprof_r0_actual_differential
    oai_memprof_r0_elf_validator_selftest
    oai_memprof_r0_harness_selftest
    oai_memprof_r0_absence_validator_selftest
    oai_memprof_r0_elf
    ${OAI_MEMPROF_R0_MUTATION_TESTS})
set_tests_properties(${OAI_MEMPROF_R0_TESTS} PROPERTIES
                     ENVIRONMENT "LC_ALL=C"
                     ENVIRONMENT_MODIFICATION
                     "LD_PRELOAD=unset:;LD_AUDIT=unset:;LD_DEBUG=unset:;LD_DEBUG_OUTPUT=unset:;LD_PROFILE=unset:;LD_PROFILE_OUTPUT=unset:;LD_LIBRARY_PATH=unset:;LD_ORIGIN_PATH=unset:;LD_POINTER_GUARD=unset:;LD_PREFER_MAP_32BIT_EXEC=unset:;LD_ASSUME_KERNEL=unset:;LD_BIND_NOW=unset:;LD_BIND_NOT=unset:;LD_DYNAMIC_WEAK=unset:;LD_HWCAP_MASK=unset:;LD_SHOW_AUXV=unset:;LD_TRACE_LOADED_OBJECTS=unset:;LD_TRACE_PRELINKING=unset:;LD_USE_LOAD_BIAS=unset:;LD_VERBOSE=unset:;LD_WARN=unset:;GLIBC_TUNABLES=unset:"
                     LABELS "memprof;r0"
                     TIMEOUT 120)
set_tests_properties(oai_memprof_r0_elf PROPERTIES TIMEOUT 900)

add_dependencies(tests
                 oai_memprof_runtime
                 oai_memprof_wrap_c
                 test_oai_memprof_r0_scripted_a00
                 test_oai_memprof_r0_scripted_a01
                 ${OAI_MEMPROF_R0_MUTATION_TARGETS}
                 oai_memprof_r0_actual_dso_a00
                 oai_memprof_r0_actual_dso_a01
                 test_oai_memprof_r0_actual_a00
                 test_oai_memprof_r0_actual_a01)
