add_library(vrtsim SHARED vrtsim.c noise_device.c)
set_target_properties(vrtsim PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} SOVERSION 1)
target_link_libraries(vrtsim PRIVATE SIMU shm_td_iq_channel actor)
  target_link_options(vrtsim PRIVATE -Wl,--hash-style=gnu)
add_dependencies(vrtsim generate_T)


add_custom_command(TARGET vrtsim
  COMMAND ${CMAKE_STRIP} --strip-unneeded $<TARGET_FILE:vrtsim>
  COMMENT "Stripping unneeded symbols from vrtsim"
  )
