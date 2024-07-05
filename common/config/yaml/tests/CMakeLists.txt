add_executable(test_yaml_config test_yaml_config.cpp)
target_link_libraries(test_yaml_config PRIVATE GTest::gtest minimal_lib params_yaml)
add_dependencies(tests test_yaml_config)
add_test(NAME test_yaml_config
  COMMAND ./test_yaml_config)
configure_file(test1.yaml test1.yaml COPYONLY)
configure_file(test_recursion.yaml test_recursion.yaml COPYONLY)
configure_file(test_list.yaml test_list.yaml COPYONLY)
