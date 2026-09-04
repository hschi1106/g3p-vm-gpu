#pragma once

#include <vector>

#include "gagp/core/bytecode.hpp"
#include "gagp/core/value.hpp"
#include "gagp/runtime/cpu/fitness_cpu.hpp"
#include "json.hpp"

namespace gagp::cli_detail {

Value decode_typed_value(const JsonValue& v);
BytecodeProgram decode_program(const JsonValue& bc);
CaseBindings decode_input_case(const JsonValue& v);
std::vector<CaseBindings> decode_cases(const JsonValue& v);
std::vector<Value> decode_shared_answer(const JsonValue& v);
std::vector<BytecodeProgram> decode_programs(const JsonValue& v);
void print_value(const Value& v);

}  // namespace gagp::cli_detail
