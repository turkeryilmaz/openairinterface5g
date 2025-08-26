add_library(sniffer MODULE sniffer.c frequency_offset.c)
set_target_properties(sniffer PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_dependencies(vrtsim generate_T)
