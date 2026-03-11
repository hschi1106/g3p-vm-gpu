#pragma once

#include <vector>

#include "g3pvm/core/bytecode.hpp"
#include "g3pvm/core/value.hpp"
#include "g3pvm/runtime/fitness_cpu.hpp"
#include "json.hpp"

namespace g3pvm::cli_detail {

Value decode_typed_value(const JsonValue& v);
BytecodeProgram decode_program(const JsonValue& bc);
CaseInputs decode_input_case(const JsonValue& v);
std::vector<CaseInputs> decode_cases(const JsonValue& v);
std::vector<Value> decode_shared_answer(const JsonValue& v);
std::vector<BytecodeProgram> decode_programs(const JsonValue& v);
void print_value(const Value& v);

}  // namespace g3pvm::cli_detail
