add_library(nr_nfapi_p5 OBJECT
        ../oai_integration/nfapi.c
        ../open-nFAPI/common/src/debug.c
        ../open-nFAPI/nfapi/src/nfapi.c
        ../open-nFAPI/nfapi/src/nfapi_p5.c
        ../open-nFAPI/fapi/src/nr_fapi_p5.c)

if(ENABLE_TESTS)
    add_subdirectory(p5)
#    add_subdirectory(p7)
endif()

