# SPDX-License-Identifier: LicenseRef-CSSL-1.0

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_executable(test_oai_memprof_wire test_oai_memprof_wire.c)
target_link_libraries(test_oai_memprof_wire PRIVATE oai_memprof_wire)
add_dependencies(tests test_oai_memprof_wire)

add_test(NAME oai_memprof_wire_c
         COMMAND test_oai_memprof_wire)

add_test(NAME oai_memprof_wire_cross_literal
         COMMAND ${CMAKE_COMMAND}
                 -DEMITTER=$<TARGET_FILE:test_oai_memprof_wire>
                 -DEXPECTED=${OPENAIR_DIR}/tools/profiling/memory/tests/event_record_v1_literal.hex
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/compare_literal.cmake)

add_test(NAME oai_memprof_wire_python
         COMMAND ${Python3_EXECUTABLE}
                 -B
                 -m unittest
                 -v
                 tools.profiling.memory.tests.test_oai_memprof_wire)
set_tests_properties(oai_memprof_wire_python PROPERTIES WORKING_DIRECTORY ${OPENAIR_DIR})
