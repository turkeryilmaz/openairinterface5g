add_executable(test_nr_ue_power_procedures
  test_nr_ue_power_procedures.cpp)
target_link_libraries(test_nr_ue_power_procedures PRIVATE NR_L2_UE asn1_nr_rrc_hdrs asn1_lte_rrc_hdrs gtest minimal_lib nr_l2_common UTIL)
add_test(NAME test_nr_ue_power_procedures
  COMMAND ./test_nr_ue_power_procedures
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
