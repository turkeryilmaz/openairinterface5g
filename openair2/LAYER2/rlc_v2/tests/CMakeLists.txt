
foreach(TESTNUM RANGE 1 49)
    add_executable(rlc_v2_test_${TESTNUM} test.c)
    target_link_libraries(rlc_v2_test_${TESTNUM} PRIVATE rlc_v2_core pthread UTIL ${T_LIB} minimal_lib)
    target_compile_definitions(rlc_v2_test_${TESTNUM} PRIVATE TEST="test${TESTNUM}.h")
    #target_include_directories(nr_rlc_test_${TESTNUM} PRIVATE LOG)
    #target_compile_options(nr_rlc_test_${TESTNUM} PRIVATE -g --coverage -Wall)
    add_dependencies(tests rlc_v2_test_${TESTNUM})
endforeach()

add_test(NAME rlc_v2_tests
        COMMAND exec_rlc_v2_test.sh ${CMAKE_CURRENT_BINARY_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

