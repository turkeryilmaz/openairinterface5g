add_library(nr_fapi_p5
        src/p5/nr_fapi_p5.c
        src/p5/nr_fapi_p5_utils.c
)
target_include_directories(nr_fapi_p5 PUBLIC inc/shared inc/p5)
target_link_libraries(nr_fapi_p5 PUBLIC nfapi_common)
