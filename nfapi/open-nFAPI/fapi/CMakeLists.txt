add_library(nr_fapi_common
        src/common/nr_fapi.c
)
target_include_directories(nr_fapi_common PUBLIC inc/common)
target_link_libraries(nr_fapi_common PUBLIC nfapi_common)

add_library(nr_fapi_p5
        src/p5/nr_fapi_p5.c
        src/p5/nr_fapi_p5_utils.c
)
target_include_directories(nr_fapi_p5 PUBLIC inc/p5)
target_link_libraries(nr_fapi_p5 PUBLIC nr_fapi_common nfapi_common)

add_library(nr_fapi_p7
        src/p7/nr_fapi_p7.c
)
target_include_directories(nr_fapi_p7 PUBLIC inc/p7)
# Note: Dependencies with NFAPI_LIB NFAPI_USER_LIB only necessary until last FAPI function is moved into new library, to be removed at a later stage
target_link_libraries(nr_fapi_p7 PUBLIC nr_fapi_common nfapi_common NFAPI_LIB NFAPI_USER_LIB)
