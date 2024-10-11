add_library(rfsimulator_object OBJECT
  simulator.c
  apply_channelmod.c
  ../../openair1/PHY/TOOLS/signal_energy.c)
target_link_libraries(rfsimulator_object PUBLIC SIMU)

add_library(rfsimulator MODULE $<TARGET_OBJECTS:rfsimulator_object>)
target_link_libraries(rfsimulator PRIVATE SIMU HASHTABLE)
set_target_properties(rfsimulator PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(replay_node stored_node.c)
target_link_libraries(replay_node minimal_lib)

if (ENABLE_TESTS)
  add_library(rfsimulator_static STATIC $<TARGET_OBJECTS:rfsimulator_object>)
  target_include_directories(rfsimulator_static PUBLIC ./)
  target_link_libraries(rfsimulator_static PUBLIC SIMU)
  add_subdirectory(tests)
endif()
