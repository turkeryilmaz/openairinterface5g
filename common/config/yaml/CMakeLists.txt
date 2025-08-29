find_package(yaml-cpp)
if (NOT yaml-cpp_FOUND)
  CPMAddPackage("gh:jbeder/yaml-cpp#yaml-cpp-0.7.0@0.7.0")
else()
  if (NOT TARGET yaml-cpp::yaml-cpp)
    add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)
  endif()
endif()

add_library(params_yaml_static config_yaml.cpp)
target_link_libraries(params_yaml_static PUBLIC UTIL ${YAML_CPP_STATIC})
if (ENABLE_TESTS)
  add_subdirectory(tests)
endif()

add_library(params_yaml SHARED config_yaml.cpp)
target_link_libraries(params_yaml PUBLIC UTIL ${YAML_CPP_STATIC})
set_target_properties(params_yaml PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} VERSION 1.0.0 SOVERSION 1)
target_link_options(params_yaml PRIVATE -Wl,--hash-style=gnu)



add_custom_command(TARGET params_yaml
  COMMAND ${CMAKE_STRIP} --strip-unneeded $<TARGET_FILE:params_yaml>
  COMMENT "Stripping unneeded symbols from params_yaml"
  )
