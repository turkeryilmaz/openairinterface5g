# SPDX-License-Identifier: LicenseRef-CSSL-1.0

add_library(radio_common STATIC common_lib.c record_player.c)
target_include_directories(radio_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(radio_common PUBLIC dl log_headers shlib_loader)

find_package(Threads REQUIRED)
add_library(radio_health STATIC radio_health.c)
target_include_directories(radio_health PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(radio_health PUBLIC Threads::Threads)

if(ENABLE_TESTS)
  add_subdirectory(tests)
endif()
