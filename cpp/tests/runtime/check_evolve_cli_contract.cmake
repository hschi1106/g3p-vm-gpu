if(NOT DEFINED CLI OR NOT DEFINED CASES OR NOT DEFINED OUT_JSON)
  message(FATAL_ERROR "CLI, CASES, and OUT_JSON are required")
endif()

execute_process(
  COMMAND "${CLI}"
  RESULT_VARIABLE no_args_result
  OUTPUT_VARIABLE no_args_stdout
  ERROR_VARIABLE no_args_stderr
)
if(NOT no_args_result EQUAL 2 OR NOT no_args_stdout STREQUAL "" OR
   NOT no_args_stderr STREQUAL "g3pvm_evolve_cli error: --cases is required\n")
  message(FATAL_ERROR "no-argument CLI contract changed: ${no_args_result}; ${no_args_stderr}")
endif()

execute_process(
  COMMAND "${CLI}" --help
  RESULT_VARIABLE help_result
  OUTPUT_VARIABLE help_stdout
  ERROR_VARIABLE help_stderr
)
if(NOT help_result EQUAL 2 OR NOT help_stdout STREQUAL "" OR
   NOT help_stderr STREQUAL "g3pvm_evolve_cli error: unknown argument: --help\n")
  message(FATAL_ERROR "--help CLI contract changed: ${help_result}; ${help_stderr}")
endif()

execute_process(
  COMMAND "${CLI}" --cases "${CASES}" --population-size 4 --generations 1
          --seed 7 --timing none --show-program none --out-json "${OUT_JSON}"
  RESULT_VARIABLE evolve_result
  OUTPUT_VARIABLE evolve_stdout
  ERROR_VARIABLE evolve_stderr
)
if(NOT evolve_result EQUAL 0 OR NOT evolve_stderr STREQUAL "")
  message(FATAL_ERROR "representative evolution failed: ${evolve_result}; ${evolve_stderr}")
endif()

foreach(required IN ITEMS
    "GEN 000 best=-1024.000000 mean=-1024.000000"
    "FINAL best=-1024.000000"
    "repro_backend=cpu"
    "selection=round_based_tournament"
    "crossover=typed_subtree")
  string(FIND "${evolve_stdout}" "${required}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "representative stdout lost '${required}': ${evolve_stdout}")
  endif()
endforeach()

file(READ "${OUT_JSON}" evolve_json)
foreach(required IN ITEMS
    "\"meta\""
    "\"history\""
    "\"timing\""
    "\"final\""
    "\"population_source\": \"generated\""
    "\"eval_engine\": \"cpu\""
    "\"reproduction_backend\": \"cpu\""
    "\"skipped\": false")
  string(FIND "${evolve_json}" "${required}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "representative JSON lost '${required}'")
  endif()
endforeach()

file(REMOVE "${OUT_JSON}")
