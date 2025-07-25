add_library(rfsimulator SHARED
            simulator.c
            apply_channelmod.c
            ../../openair1/PHY/TOOLS/signal_energy.c
)
target_link_libraries(rfsimulator PRIVATE SIMU)
set_target_properties(rfsimulator PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} VERSION 1.0.0 SOVERSION 1)
target_link_options(rfsimulator PRIVATE -Wl,--hash-style=gnu)


add_custom_command(TARGET rfsimulator
    COMMAND ${CMAKE_STRIP} --strip-unneeded $<TARGET_FILE:rfsimulator>
    COMMENT "Stripping unneeded symbols from rfsimulator"
    )

add_executable(replay_node stored_node.c)
target_link_libraries (replay_node minimal_lib)
