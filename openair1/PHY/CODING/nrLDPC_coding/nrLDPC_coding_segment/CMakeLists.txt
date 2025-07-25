add_library(ldpc_segment OBJECT
  nr_rate_matching.c
  nrLDPC_coding_segment_decoder.c
  nrLDPC_coding_segment_encoder.c
)
set_target_properties(ldpc PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_link_libraries(ldpc PUBLIC T_headers)

#ensure that the T header files are generated before targets depending on them
if (${T_TRACER})
  add_dependencies(ldpc_segment generate_T)
endif (${T_TRACER})
