cmake_minimum_required(VERSION 3.15)

project (TC_SM)
set(CMAKE_BUILD_TYPE Debug)
#set (CMAKE_C_FLAGS "-Wall -Wextra -std=c11 -g -ggdb -fsanitize=address -fno-omit-frame-pointer") 
set (CMAKE_C_FLAGS "-Wall -Wextra -std=c11 -g -fno-omit-frame-pointer") 



set(SM_ENCODING "PLAIN" CACHE STRING "The E2AP encoding to use")
set_property(CACHE SM_ENCODING PROPERTY STRINGS "PLAIN" "ASN" "FLATBUFFERS")
message(STATUS "Selected SM_ENCODING: ${SM_ENCODING}")


include_directories(${CMAKE_CURRENT_SOURCE_DIR} )
include_directories(${CMAKE_CURRENT_SOURCE_DIR}../ie/asn/ )
include_directories(${CMAKE_CURRENT_SOURCE_DIR}../ie/fb/ )
include_directories("/usr/local/share/asn1c/")


file(GLOB asn_sources "../ie/asn/*.c")

add_executable(test_tc_sm main.c 
                      ../../sm_proc_data.c 
                      ../tc_sm_agent.c 
                      ../tc_sm_ric.c 
                      ../enc/tc_enc_plain.c 
                      ../enc/tc_enc_fb.c 
                      ../enc/tc_enc_asn.c 
                      ../dec/tc_dec_plain.c 
                      ../dec/tc_dec_fb.c 
                      ../dec/tc_dec_asn.c 
                      ../ie/tc_data_ie.c 
                      ../../../util/alg_ds/alg/defer.c
                      ../../../util/alg_ds/alg/eq_float.c
                      ../../agent_if/read/sm_ag_if_rd.c
                      ${asn_sources} 
                      # This sucks...
                      ../../mac_sm/ie/mac_data_ie.c
                      ../../rlc_sm/ie/rlc_data_ie.c
                      ../../pdcp_sm/ie/pdcp_data_ie.c
                      ../../slice_sm/ie/slice_data_ie.c
                      )

target_compile_definitions(test_tc_sm PUBLIC ${SM_ENCODING})
target_compile_options(test_tc_sm PUBLIC "-DASN_DISABLE_OER_SUPPORT")
target_link_libraries(test_tc_sm PUBLIC -pthread)

