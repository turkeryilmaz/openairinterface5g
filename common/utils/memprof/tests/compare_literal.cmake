# SPDX-License-Identifier: LicenseRef-CSSL-1.0

if(NOT DEFINED EMITTER OR NOT DEFINED EXPECTED)
  message(FATAL_ERROR "EMITTER and EXPECTED are required")
endif()

execute_process(
  COMMAND "${EMITTER}" --emit-literal-hex
  RESULT_VARIABLE emitter_status
  OUTPUT_VARIABLE actual_literal
  ERROR_VARIABLE emitter_stderr)

if(NOT "${emitter_status}" STREQUAL "0")
  message(FATAL_ERROR
          "wire literal emitter failed with status ${emitter_status}: ${emitter_stderr}")
endif()

file(READ "${EXPECTED}" expected_literal)
if(NOT actual_literal STREQUAL expected_literal)
  message(FATAL_ERROR "native C wire literal differs from the frozen Python fixture")
endif()
