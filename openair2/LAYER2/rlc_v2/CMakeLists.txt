add_library(rlc_v2_core OBJECT
        rlc_entity.c
        rlc_entity_am.c
        rlc_entity_um.c
        rlc_pdu.c
        rlc_sdu.c)
target_include_directories(rlc_v2_core PRIVATE ../../..)
target_link_libraries(rlc_v2_core PRIVATE ${T_LIB})
if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
