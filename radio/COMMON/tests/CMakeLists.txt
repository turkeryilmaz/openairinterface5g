# SPDX-License-Identifier: LicenseRef-CSSL-1.0

add_executable(test_radio_health test_radio_health.c)
target_link_libraries(test_radio_health PRIVATE radio_health Threads::Threads)
add_dependencies(tests test_radio_health)
add_test(NAME test_radio_health
  COMMAND ./test_radio_health)
