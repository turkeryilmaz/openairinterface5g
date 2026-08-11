# SPDX-License-Identifier: LicenseRef-CSSL-1.0

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "The R0 memory-profiler GNU-wrap gate currently admits Linux ELF targets only")
endif()

if(NOT CMAKE_C_COMPILER_ID STREQUAL "GNU")
  message(FATAL_ERROR "The frozen R0 memory-profiler gate requires GNU GCC")
endif()

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "The frozen R0 memory-profiler gate requires a 64-bit target")
endif()

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" OAI_MEMPROF_R0_SYSTEM_PROCESSOR)
if(NOT OAI_MEMPROF_R0_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|aarch64|arm64)$")
  message(FATAL_ERROR "The frozen R0 memory-profiler gate admits x86-64 and AArch64 only")
endif()

include(TestBigEndian)
test_big_endian(OAI_MEMPROF_R0_BIG_ENDIAN)
if(OAI_MEMPROF_R0_BIG_ENDIAN)
  message(FATAL_ERROR "The frozen R0 memory-profiler gate requires a little-endian target")
endif()

set(OAI_MEMPROF_R0_PASSIVE_COMPILE_OPTIONS
    -fplt
    -fno-instrument-functions
    -fno-lto
    -fno-profile-arcs
    -fno-sanitize=all
    -fno-stack-protector
    -fno-test-coverage)

add_library(oai_memprof_runtime SHARED oai_memprof_runtime.c)
target_compile_features(oai_memprof_runtime PRIVATE c_std_11)
target_include_directories(oai_memprof_runtime PRIVATE ${OPENAIR_DIR})
target_compile_options(oai_memprof_runtime PRIVATE ${OAI_MEMPROF_R0_PASSIVE_COMPILE_OPTIONS})
target_link_options(oai_memprof_runtime PRIVATE
                    -nostartfiles
                    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/oai_memprof_runtime.map"
                    "LINKER:--no-undefined"
                    "LINKER:-z,defs")
set_property(TARGET oai_memprof_runtime APPEND PROPERTY
             LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/oai_memprof_runtime.map)
set_target_properties(oai_memprof_runtime PROPERTIES
                      C_COMPILER_LAUNCHER ""
                      C_VISIBILITY_PRESET hidden
                      INTERPROCEDURAL_OPTIMIZATION FALSE
                      OUTPUT_NAME oai_memprof_runtime
                      POSITION_INDEPENDENT_CODE ON
                      BUILD_RPATH ""
                      INSTALL_RPATH ""
                      SKIP_BUILD_RPATH TRUE
                      SKIP_INSTALL_RPATH TRUE
                      SOVERSION 1
                      VERSION 1.0.0)

add_library(oai_memprof_wrap_c STATIC
            oai_memprof_wrap_malloc.c
            oai_memprof_wrap_calloc.c
            oai_memprof_wrap_realloc.c
            oai_memprof_wrap_free.c)
target_compile_features(oai_memprof_wrap_c PRIVATE c_std_11)
target_include_directories(oai_memprof_wrap_c PRIVATE ${OPENAIR_DIR})
target_compile_options(oai_memprof_wrap_c PRIVATE ${OAI_MEMPROF_R0_PASSIVE_COMPILE_OPTIONS})
set_target_properties(oai_memprof_wrap_c PROPERTIES
                      C_COMPILER_LAUNCHER ""
                      C_VISIBILITY_PRESET hidden
                      INTERPROCEDURAL_OPTIMIZATION FALSE
                      POSITION_INDEPENDENT_CODE ON)

add_library(oai_memprof_wire STATIC oai_memprof_wire.c)
target_compile_features(oai_memprof_wire PUBLIC c_std_11)
target_include_directories(oai_memprof_wire PUBLIC ${OPENAIR_DIR})

if(ENABLE_TESTS)
  add_subdirectory(tests)
endif()
