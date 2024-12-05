set(rfsimulator_sources
    simulator.c
    apply_channelmod.c
    ../../openair1/PHY/TOOLS/signal_energy.c
)
if (ENABLE_RESTAPI)
  list(APPEND rfsimulator_sources simulator_rest_api.cpp)
endif()
add_library(rfsimulator MODULE ${rfsimulator_sources})

target_link_libraries(rfsimulator PRIVATE SIMU HASHTABLE)
if (ENABLE_RESTAPI)
  target_link_libraries(rfsimulator PRIVATE Crow::Crow nlohmann_json::nlohmann_json)
endif()
set_target_properties(rfsimulator PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(replay_node stored_node.c)
target_link_libraries (replay_node minimal_lib)
