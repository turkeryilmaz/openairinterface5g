add_library(dummy_rf MODULE dummy_rf.c)
set_target_properties(dummy_rf PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_link_libraries(dummy_rf PRIVATE noise_device)
