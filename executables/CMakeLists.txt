# SPDX-License-Identifier: LicenseRef-CSSL-1.0

add_library(softmodem_common STATIC softmodem-common.c flight_options.c)
target_compile_definitions(softmodem_common PRIVATE OAI_FLIGHT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(softmodem_common PUBLIC shlib_loader)

if(ENABLE_TESTS)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_executable(flight_startup_fixture tests/flight_startup_fixture.c flight_options.c)
  target_compile_definitions(flight_startup_fixture PRIVATE OAI_FLIGHT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  target_link_libraries(flight_startup_fixture PRIVATE CONFIG_LIB LOG pthread)
  set_target_properties(flight_startup_fixture PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_test(NAME flight_startup COMMAND ${Python3_EXECUTABLE} -B
    ${CMAKE_SOURCE_DIR}/tools/flight_test/tests/test_startup.py $<TARGET_FILE:flight_startup_fixture>)
  set_tests_properties(flight_startup PROPERTIES TIMEOUT 60)
endif()
