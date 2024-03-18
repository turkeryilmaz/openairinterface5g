#PARAM.request test
add_executable(nr_fapi_param_request_test nr_fapi_param_request_test.c)
target_link_libraries(nr_fapi_param_request_test PRIVATE nr_nfapi_p5 pthread UTIL ${T_LIB} minimal_lib)
target_compile_definitions(nr_fapi_param_request_test PRIVATE TEST="nr_fapi_test.h")
add_dependencies(tests nr_fapi_param_request_test)

add_test(NAME nr_fapi_param_request_test
        COMMAND nr_fapi_param_request_test ${CMAKE_CURRENT_BINARY_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

#PARAM.response test
add_executable(nr_fapi_param_response_test nr_fapi_param_response_test.c)
target_link_libraries(nr_fapi_param_response_test PRIVATE nr_nfapi_p5 pthread UTIL ${T_LIB} minimal_lib)
target_compile_definitions(nr_fapi_param_response_test PRIVATE TEST="nr_fapi_test.h")
add_dependencies(tests nr_fapi_param_response_test)

add_test(NAME nr_fapi_param_response_test
        COMMAND nr_fapi_param_response_test ${CMAKE_CURRENT_BINARY_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
