# SPDX-License-Identifier: LicenseRef-CSSL-1.0

find_package(Threads REQUIRED)
add_library(radio_gain STATIC radio_gain.c radio_gain_policy.c radio_gain_samples.c radio_tx_power.c)
target_include_directories(radio_gain PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(radio_gain PUBLIC Threads::Threads m)

add_library(radio_common STATIC common_lib.c record_player.c radio_gain_device.c)
target_include_directories(radio_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(radio_common PUBLIC dl log_headers shlib_loader radio_gain softmodem_common)

add_library(radio_health STATIC radio_health.c)
target_include_directories(radio_health PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(radio_health PUBLIC Threads::Threads)

if(ENABLE_TESTS)
  add_subdirectory(tests)
endif()
