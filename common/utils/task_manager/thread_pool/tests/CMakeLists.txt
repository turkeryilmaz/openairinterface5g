add_executable(test_ws-thread-pool test_thread_pool.c)
target_link_libraries(test_ws-thread-pool PRIVATE ws-thread-pool minimal_lib)
add_dependencies(tests test_ws-thread-pool)
add_test(NAME test_ws-thread-pool
  COMMAND ./test_ws-thread-pool)

add_executable(test_marl test_marl.cpp)
target_link_libraries(test_marl PRIVATE marl)
add_dependencies(tests test_marl)
add_test(NAME test_marl
  COMMAND ./test_marl)

CPMAddPackage("gh:google/marl#690889fbb816a409b4fcd4bdf60e32a4deebeb08")
