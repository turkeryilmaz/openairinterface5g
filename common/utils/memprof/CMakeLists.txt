# SPDX-License-Identifier: LicenseRef-CSSL-1.0

add_library(oai_memprof_wire STATIC oai_memprof_wire.c)
target_compile_features(oai_memprof_wire PUBLIC c_std_11)
target_include_directories(oai_memprof_wire PUBLIC ${OPENAIR_DIR})

if(ENABLE_TESTS)
  add_subdirectory(tests)
endif()
