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
