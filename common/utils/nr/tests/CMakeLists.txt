
add_executable(reverse_bits_test reverse_bits_test.c ../reverse_bits.c)
target_link_libraries(reverse_bits_test UTIL minimal_lib)
add_dependencies(tests reverse_bits_test)
add_test(NAME reverse_bits_test COMMAND reverse_bits_test)
