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

add_library(oai_memprof_active_runtime SHARED oai_memprof_active_runtime.c)
set(OAI_MEMPROF_ACTIVE_RUNTIME_LINK_MAP
    "${CMAKE_CURRENT_BINARY_DIR}/liboai_memprof_active_runtime.map")
target_compile_features(oai_memprof_active_runtime PRIVATE c_std_11)
target_include_directories(oai_memprof_active_runtime PUBLIC ${OPENAIR_DIR})
target_compile_options(oai_memprof_active_runtime PRIVATE -Wall -Wextra -Werror -Wpedantic)
target_link_libraries(oai_memprof_active_runtime PRIVATE oai_memprof_active_core pthread)
target_link_options(oai_memprof_active_runtime PRIVATE
                    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/oai_memprof_active_runtime.map"
                    "LINKER:--no-undefined"
                    "LINKER:-z,defs"
                    "LINKER:-Map,${OAI_MEMPROF_ACTIVE_RUNTIME_LINK_MAP}"
                    "LINKER:--cref")
set_property(TARGET oai_memprof_active_runtime APPEND PROPERTY
             LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/oai_memprof_active_runtime.map)
set_target_properties(oai_memprof_active_runtime PROPERTIES
                      C_COMPILER_LAUNCHER ""
                      C_VISIBILITY_PRESET hidden
                      INTERPROCEDURAL_OPTIMIZATION FALSE
                      OUTPUT_NAME oai_memprof_active_runtime
                      POSITION_INDEPENDENT_CODE ON
                      BUILD_RPATH ""
                      INSTALL_RPATH ""
                      SKIP_BUILD_RPATH TRUE
                      SKIP_INSTALL_RPATH TRUE
                      SOVERSION 1
                      VERSION 1.0.0)

add_library(oai_memprof_active_wrap_c STATIC
            oai_memprof_active_wrap_malloc.c
            oai_memprof_active_wrap_calloc.c
            oai_memprof_active_wrap_realloc.c
            oai_memprof_active_wrap_free.c
            oai_memprof_active_wrap_reallocarray.c
            oai_memprof_active_wrap_aligned_alloc.c
            oai_memprof_active_wrap_posix_memalign.c
            oai_memprof_active_wrap_memalign.c
            oai_memprof_active_wrap_valloc.c
            oai_memprof_active_wrap_pvalloc.c
            oai_memprof_active_wrap_strdup.c
            oai_memprof_active_wrap_strndup.c)
target_compile_features(oai_memprof_active_wrap_c PRIVATE c_std_11)
target_include_directories(oai_memprof_active_wrap_c PRIVATE ${OPENAIR_DIR})
target_compile_options(oai_memprof_active_wrap_c PRIVATE
                       ${OAI_MEMPROF_R0_PASSIVE_COMPILE_OPTIONS}
                       -Wall -Wextra -Werror -Wpedantic)
set_target_properties(oai_memprof_active_wrap_c PROPERTIES
                      C_COMPILER_LAUNCHER ""
                      C_VISIBILITY_PRESET hidden
                      INTERPROCEDURAL_OPTIMIZATION FALSE
                      POSITION_INDEPENDENT_CODE ON)

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

add_library(oai_memprof_active_core STATIC oai_memprof_active_core.c)
target_compile_features(oai_memprof_active_core PUBLIC c_std_11)
target_include_directories(oai_memprof_active_core PUBLIC ${OPENAIR_DIR})
target_compile_options(oai_memprof_active_core PRIVATE -Wall -Wextra -Werror -Wpedantic)
target_link_libraries(oai_memprof_active_core PUBLIC oai_memprof_wire PRIVATE pthread)
set_target_properties(oai_memprof_active_core PROPERTIES
                      C_COMPILER_LAUNCHER ""
                      INTERPROCEDURAL_OPTIMIZATION FALSE
                      POSITION_INDEPENDENT_CODE ON)

add_library(oai_memprof_wire STATIC oai_memprof_wire.c)
target_compile_features(oai_memprof_wire PUBLIC c_std_11)
target_include_directories(oai_memprof_wire PUBLIC ${OPENAIR_DIR})

add_library(oai_memprof_container_wire STATIC oai_memprof_container_wire.c)
target_compile_features(oai_memprof_container_wire PUBLIC c_std_11)
target_include_directories(oai_memprof_container_wire PUBLIC ${OPENAIR_DIR})

add_library(oai_memprof_clock STATIC oai_memprof_clock.c)
target_compile_features(oai_memprof_clock PUBLIC c_std_11)
target_include_directories(oai_memprof_clock PUBLIC ${OPENAIR_DIR})
target_compile_options(oai_memprof_clock PRIVATE -Wall -Wextra -Werror -Wpedantic)
set_target_properties(oai_memprof_clock PROPERTIES
                      C_COMPILER_LAUNCHER ""
                      INTERPROCEDURAL_OPTIMIZATION FALSE
                      POSITION_INDEPENDENT_CODE ON)

add_library(oai_memprof_stream_writer STATIC oai_memprof_stream_writer.c)
target_compile_features(oai_memprof_stream_writer PUBLIC c_std_11)
target_include_directories(oai_memprof_stream_writer PUBLIC ${OPENAIR_DIR})
target_compile_options(oai_memprof_stream_writer PRIVATE -Wall -Wextra -Werror -Wpedantic)
target_link_libraries(oai_memprof_stream_writer
                      PUBLIC oai_memprof_active_runtime
                      PRIVATE oai_memprof_clock oai_memprof_container_wire oai_memprof_wire pthread)
set_target_properties(oai_memprof_stream_writer PROPERTIES
                      C_COMPILER_LAUNCHER ""
                      INTERPROCEDURAL_OPTIMIZATION FALSE
                      POSITION_INDEPENDENT_CODE ON)

add_library(oai_memprof_stream_finalizer STATIC oai_memprof_stream_finalizer.c)
target_compile_features(oai_memprof_stream_finalizer PUBLIC c_std_11)
target_include_directories(oai_memprof_stream_finalizer PUBLIC ${OPENAIR_DIR})
target_compile_options(oai_memprof_stream_finalizer PRIVATE -Wall -Wextra -Werror -Wpedantic)
target_link_libraries(oai_memprof_stream_finalizer
                      PUBLIC oai_memprof_stream_writer
                      PRIVATE oai_memprof_container_wire oai_memprof_wire)
set_target_properties(oai_memprof_stream_finalizer PROPERTIES
                      C_COMPILER_LAUNCHER ""
                      INTERPROCEDURAL_OPTIMIZATION FALSE
                      POSITION_INDEPENDENT_CODE ON)

add_library(oai_memprof_process_handoff STATIC oai_memprof_process_handoff.c)
target_compile_features(oai_memprof_process_handoff PUBLIC c_std_11)
target_include_directories(oai_memprof_process_handoff PUBLIC ${OPENAIR_DIR})
target_compile_options(oai_memprof_process_handoff PRIVATE -Wall -Wextra -Werror -Wpedantic)
target_link_libraries(oai_memprof_process_handoff
                      PUBLIC oai_memprof_stream_writer
                      PRIVATE oai_memprof_container_wire)
set_target_properties(oai_memprof_process_handoff PROPERTIES
                      C_COMPILER_LAUNCHER ""
                      INTERPROCEDURAL_OPTIMIZATION FALSE
                      POSITION_INDEPENDENT_CODE ON)

add_library(oai_memprof_process_session STATIC oai_memprof_process_session.c)
target_compile_features(oai_memprof_process_session PUBLIC c_std_11)
target_include_directories(oai_memprof_process_session PUBLIC ${OPENAIR_DIR})
target_compile_options(oai_memprof_process_session PRIVATE -Wall -Wextra -Werror -Wpedantic)
target_link_libraries(oai_memprof_process_session PUBLIC oai_memprof_process_handoff)
set_target_properties(oai_memprof_process_session PROPERTIES
                      C_COMPILER_LAUNCHER ""
                      INTERPROCEDURAL_OPTIMIZATION FALSE
                      POSITION_INDEPENDENT_CODE ON)

add_library(oai_memprof_softmodem_session STATIC oai_memprof_softmodem_session.c)
target_compile_features(oai_memprof_softmodem_session PUBLIC c_std_11)
target_include_directories(oai_memprof_softmodem_session PUBLIC ${OPENAIR_DIR})
target_compile_options(oai_memprof_softmodem_session PRIVATE
                       -Wall -Wextra -Werror -Wpedantic)
target_link_libraries(oai_memprof_softmodem_session
                      PUBLIC oai_memprof_process_session
                      PRIVATE pthread)
set_target_properties(oai_memprof_softmodem_session PROPERTIES
                      C_COMPILER_LAUNCHER ""
                      INTERPROCEDURAL_OPTIMIZATION FALSE
                      POSITION_INDEPENDENT_CODE ON)

if(OAI_MEMPROF_ACTIVE)
  set_property(GLOBAL PROPERTY OAI_MEMPROF_ACTIVE_MAP_DIRECTORY
               "${CMAKE_CURRENT_BINARY_DIR}")

  function(oai_memprof_add_active_map target)
    if(NOT TARGET ${target})
      message(FATAL_ERROR "OAI_MEMPROF_ACTIVE requires target ${target}")
    endif()
    get_property(active_map_directory GLOBAL
                 PROPERTY OAI_MEMPROF_ACTIVE_MAP_DIRECTORY)
    target_link_options(${target} PRIVATE
                        "LINKER:-Map,${active_map_directory}/${target}.oai-memprof.map"
                        "LINKER:--cref")
  endfunction()

  function(oai_memprof_wrap_active_c target)
    oai_memprof_add_active_map(${target})
    foreach(api IN LISTS ARGN)
      target_compile_options(${target} PRIVATE "-fno-builtin-${api}")
      target_link_options(${target} PRIVATE "LINKER:--wrap=${api}")
    endforeach()
    target_link_libraries(${target} PRIVATE
                          oai_memprof_active_wrap_c
                          oai_memprof_active_runtime)
    get_property(active_map_directory GLOBAL
                 PROPERTY OAI_MEMPROF_ACTIVE_MAP_DIRECTORY)
    set_property(TARGET ${target} APPEND PROPERTY
                 BUILD_RPATH "${active_map_directory}")
    set_property(TARGET ${target} PROPERTY
                 INTERPROCEDURAL_OPTIMIZATION FALSE)
  endfunction()

  function(oai_memprof_configure_active_targets)
    foreach(softmodem_target nr-softmodem nr-uesoftmodem)
      if(NOT TARGET ${softmodem_target})
        message(FATAL_ERROR "OAI_MEMPROF_ACTIVE requires target ${softmodem_target}")
      endif()
      target_compile_definitions(${softmodem_target} PRIVATE
                                 OAI_MEMPROF_ACTIVE=1)
      target_link_libraries(${softmodem_target} PRIVATE
                            oai_memprof_softmodem_session)
    endforeach()
    oai_memprof_wrap_active_c(nr-softmodem
                              malloc calloc realloc free
                              aligned_alloc posix_memalign memalign strdup)
    oai_memprof_wrap_active_c(nr-uesoftmodem
                              malloc calloc realloc free
                              posix_memalign memalign strdup)

    oai_memprof_wrap_active_c(params_libconfig
                              malloc calloc realloc free strdup)

    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" OAI_MEMPROF_ACTIVE_SYSTEM_PROCESSOR)
    if(OAI_MEMPROF_ACTIVE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
      oai_memprof_wrap_active_c(dfts free aligned_alloc posix_memalign)
    elseif(OAI_MEMPROF_ACTIVE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
      oai_memprof_add_active_map(dfts)
    else()
      message(FATAL_ERROR
              "OAI_MEMPROF_ACTIVE supports only x86_64 and aarch64")
    endif()

    foreach(ldpc_target ldpc ldpc_orig)
      oai_memprof_add_active_map(${ldpc_target})
    endforeach()

    if(TARGET oai_usrpdevif)
      oai_memprof_wrap_active_c(oai_usrpdevif malloc calloc free)
    elseif(OAI_USRP)
      message(FATAL_ERROR
              "OAI_MEMPROF_ACTIVE with OAI_USRP requires target oai_usrpdevif")
    endif()

    if(TARGET rfsimulator)
      oai_memprof_wrap_active_c(rfsimulator malloc calloc free strdup)
    elseif(OAI_SIMU)
      message(FATAL_ERROR
              "OAI_MEMPROF_ACTIVE with OAI_SIMU requires target rfsimulator")
    endif()
  endfunction()

  cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}"
                 CALL oai_memprof_configure_active_targets)
endif()

if(ENABLE_TESTS)
  add_subdirectory(tests)
endif()
