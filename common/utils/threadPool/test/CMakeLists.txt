# SPDX-License-Identifier: LicenseRef-CSSL-1.0

add_executable(test_thread-pool
  test_thread-pool.c)
add_dependencies(tests test_thread-pool)
target_link_libraries(test_thread-pool PRIVATE thread-pool pthread LOG minimal_lib)
add_test(NAME test_thread-pool
  COMMAND ./test_thread-pool)

add_executable(test_thread-pool-profile-context
  test_thread-pool-profile-context.c)
add_dependencies(tests test_thread-pool-profile-context)
target_link_libraries(test_thread-pool-profile-context PRIVATE thread-pool pthread LOG minimal_lib)
add_test(NAME test_thread-pool-profile-context
  COMMAND ./test_thread-pool-profile-context)
