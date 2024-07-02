add_library(nr_fapi_p5
        src/p5/nr_fapi_p5.c
        src/p5/nr_fapi_p5_utils.c
)
target_include_directories(nr_fapi_p5 PUBLIC inc/shared inc/p5)
target_link_libraries(nr_fapi_p5 PUBLIC nfapi_common)

add_library(nr_fapi_p7
        src/p7/nr_fapi_p7.c
)
target_include_directories(nr_fapi_p7 PUBLIC inc/shared inc/p7)
# Note: Dependencies with NFAPI_LIB NFAPI_USER_LIB only necessary until last FAPI function is moved into new library, to be removed at a later stage
target_link_libraries(nr_fapi_p7 PUBLIC nfapi_common NFAPI_LIB NFAPI_USER_LIB)
