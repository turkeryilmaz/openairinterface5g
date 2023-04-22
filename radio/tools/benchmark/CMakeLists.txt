add_executable(oai_radio_latency_check
               benchmark.c
               ${SHLIB_LOADER_SOURCES})
target_link_libraries(oai_radio_latency_check CONFIG_LIB minimal_lib)
add_dependencies(oai_radio_latency_check oai_usrpdevif)
target_include_directories(oai_radio_latency_check PRIVATE ${CMAKE_CURRENT_DIR})