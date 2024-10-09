add_executable(test_ws-thread-pool test_thread_pool.c)
target_link_libraries(test_ws-thread-pool PRIVATE ws-thread-pool minimal_lib)
add_dependencies(tests test_ws-thread-pool)
add_test(NAME test_ws-thread-pool
  COMMAND ./test_ws-thread-pool)
