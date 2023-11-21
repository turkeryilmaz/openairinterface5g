add_subdirectory(MESSAGES)

if(ENABLE_TESTS)
  add_executable(f1ap_ids_test f1ap_ids_test.c f1ap_ids.c)
  target_link_libraries(f1ap_ids_test UTIL HASHTABLE minimal_lib)
  add_dependencies(tests f1ap_ids_test)
  add_test(NAME F1AP_ID_test COMMAND f1ap_ids_test)
endif()

add_executable(f1ap_debug f1ap_debug.c)
target_link_libraries(f1ap_debug asn1_f1ap)
