
add_executable(rrc_bearers_test rrc_bearers_test.c ../rrc_gNB_radio_bearers.c ${OPENAIR2_DIR}/LAYER2/nr_pdcp/asn1_utils.c)
add_dependencies(tests rrc_bearers_test)
add_test(NAME rrc_bearers_test COMMAND rrc_bearers_test)
target_link_libraries(rrc_bearers_test PRIVATE ds alg lte_rrc)
target_include_directories(rrc_bearers_test PRIVATE
                           .
                           ./common/utils/ds
                           ${CMAKE_BINARY_DIR}/openair2/E1AP/MESSAGES
                           ${CMAKE_BINARY_DIR}/openair2/RRC/NR/MESSAGES)

target_compile_definitions(rrc_bearers_test PRIVATE ENABLE_TESTS)
