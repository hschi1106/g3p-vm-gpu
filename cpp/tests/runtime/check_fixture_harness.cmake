execute_process(
  COMMAND "${HARNESS}" "${VALID_FIXTURE}"
  RESULT_VARIABLE valid_result
  OUTPUT_VARIABLE valid_stdout
  ERROR_VARIABLE valid_stderr
)
if(NOT valid_result EQUAL 0 OR NOT valid_stdout MATCHES "failed 0")
  message(FATAL_ERROR "valid fixture failed: ${valid_result}\n${valid_stdout}\n${valid_stderr}")
endif()

execute_process(
  COMMAND "${HARNESS}" "${MISMATCH_FIXTURE}"
  RESULT_VARIABLE mismatch_result
  OUTPUT_VARIABLE mismatch_stdout
  ERROR_VARIABLE mismatch_stderr
)
if(NOT mismatch_result EQUAL 1 OR NOT mismatch_stdout MATCHES "failed 1")
  message(FATAL_ERROR "fixture mismatch was not reported: ${mismatch_result}\n${mismatch_stdout}\n${mismatch_stderr}")
endif()

execute_process(
  COMMAND "${HARNESS}" "${OLD_FORMAT_FIXTURE}"
  RESULT_VARIABLE old_result
  OUTPUT_VARIABLE old_stdout
  ERROR_VARIABLE old_stderr
)
if(NOT old_result EQUAL 2 OR NOT old_stderr MATCHES "unsupported format_version")
  message(FATAL_ERROR "old fixture format was not rejected: ${old_result}\n${old_stdout}\n${old_stderr}")
endif()
