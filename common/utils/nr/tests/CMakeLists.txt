add_executable(test_nr_common test_nr_common.cpp)
target_link_libraries(test_nr_common PRIVATE nr_common GTest::gtest minimal_lib)
