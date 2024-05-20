add_library(nr_l2_common nr_common.c)
target_include_directories(nr_l2_common PUBLIC .)
target_compile_definitions(nr_l2_common PRIVATE -DTEST_L2)
