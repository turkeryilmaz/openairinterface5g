#find_package(Boost REQUIRED)
find_package(UHD REQUIRED UHD)

add_library(oai_usrpdevif SHARED usrp_lib.cpp)
#target_include_directories(oai_usrpdevif PRIVATE Boost::boost)
target_link_libraries(oai_usrpdevif PRIVATE ${UHD_LIBRARIES})
target_include_directories(oai_usrpdevif PRIVATE ${UHD_INCLUDE_DIRS})
target_link_libraries(oai_usrpdevif PRIVATE UTIL)
target_link_options(oai_usrpdevif PRIVATE -Wl,--hash-style=gnu)
set_target_properties(oai_usrpdevif PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} VERSION 1.0.0 SOVERSION 1)
add_custom_command(TARGET oai_usrpdevif
                   COMMAND ${CMAKE_STRIP} --strip-unneeded $<TARGET_FILE:oai_usrpdevif>
                   COMMENT "Stripping unneeded symbols from oai_usrpdevif")  
add_custom_command(TARGET oai_usrpdevif POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E create_symlink liboai_usrpdevif.so liboai_device.so
                   WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
