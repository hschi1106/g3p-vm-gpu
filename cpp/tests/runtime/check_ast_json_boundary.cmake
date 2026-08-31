if(NOT DEFINED CLI OR NOT DEFINED CASES OR NOT DEFINED ASGP_CASES OR
   NOT DEFINED ASGP_SCALAR_CASES OR NOT DEFINED VALID_AST OR
   NOT DEFINED VALID_ASGP_AST OR NOT DEFINED VALID_DP1_AST OR
   NOT DEFINED VALID_DP2_AST OR NOT DEFINED INVALID_AST)
  message(FATAL_ERROR "all AST boundary fixtures are required")
endif()

foreach(dp_ast IN ITEMS "${VALID_DP1_AST}" "${VALID_DP2_AST}")
  execute_process(
    COMMAND "${CLI}" --cases "${ASGP_SCALAR_CASES}" --eval-ast-json "${dp_ast}"
    RESULT_VARIABLE dp_result
    OUTPUT_VARIABLE dp_stdout
    ERROR_VARIABLE dp_stderr
  )
  if(NOT dp_result EQUAL 0 OR NOT dp_stdout MATCHES "AST_EVAL fitness=0\\.000000")
    message(FATAL_ERROR
      "valid ASGP-DP AST boundary failed (${dp_result})\nstdout: ${dp_stdout}\nstderr: ${dp_stderr}")
  endif()
endforeach()

execute_process(
  COMMAND "${CLI}" --cases "${ASGP_CASES}" --eval-ast-json "${VALID_ASGP_AST}"
  RESULT_VARIABLE asgp_result
  OUTPUT_VARIABLE asgp_stdout
  ERROR_VARIABLE asgp_stderr
)
if(NOT asgp_result EQUAL 0 OR NOT asgp_stdout MATCHES "AST_EVAL fitness=0\\.000000")
  message(FATAL_ERROR
    "valid ASGP AST boundary failed (${asgp_result})\nstdout: ${asgp_stdout}\nstderr: ${asgp_stderr}")
endif()

execute_process(
  COMMAND "${CLI}" --cases "${CASES}" --eval-ast-json "${VALID_AST}"
  RESULT_VARIABLE valid_result
  OUTPUT_VARIABLE valid_stdout
  ERROR_VARIABLE valid_stderr
)
if(NOT valid_result EQUAL 0 OR NOT valid_stdout MATCHES "AST_EVAL fitness=0\\.000000")
  message(FATAL_ERROR
    "valid AST boundary failed (${valid_result})\nstdout: ${valid_stdout}\nstderr: ${valid_stderr}")
endif()

execute_process(
  COMMAND "${CLI}" --cases "${CASES}" --eval-ast-json "${INVALID_AST}"
  RESULT_VARIABLE invalid_result
  OUTPUT_VARIABLE invalid_stdout
  ERROR_VARIABLE invalid_stderr
)
if(invalid_result EQUAL 0)
  message(FATAL_ERROR "malformed AST unexpectedly passed: ${invalid_stdout}")
endif()
if(NOT invalid_stderr MATCHES "invalid AST \\(name_index_out_of_range\\)")
  message(FATAL_ERROR "malformed AST diagnostic was unstable: ${invalid_stderr}")
endif()
