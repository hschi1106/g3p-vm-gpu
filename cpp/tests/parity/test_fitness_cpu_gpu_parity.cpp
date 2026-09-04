#include <climits>
#include <cmath>
#include <limits>
#include <iostream>
#include <vector>

#include "gagp/core/builtin.hpp"
#include "gagp/core/bytecode.hpp"
#include "gagp/core/value.hpp"
#include "gagp/evolution/compiler.hpp"
#include "gagp/evolution/genome.hpp"
#include "gagp/runtime/cpu/fitness_cpu.hpp"
#include "gagp/runtime/gpu/fitness_gpu.hpp"
#include "gagp/runtime/payload/payload.hpp"

namespace {

using gagp::BytecodeProgram;
using gagp::CaseBindings;
using gagp::InputBinding;
using gagp::Opcode;
using gagp::Value;

gagp::Instr ins(Opcode op) { return gagp::Instr{op, 0, 0, false, false}; }
gagp::Instr ins_a(Opcode op, int a) { return gagp::Instr{op, a, 0, true, false}; }
gagp::Instr ins_ab(Opcode op, int a, int b) { return gagp::Instr{op, a, b, true, true}; }

BytecodeProgram make_add_one_program() {
  BytecodeProgram p;
  p.n_locals = 1;
  p.consts = {Value::from_int(1)};
  p.code = {ins_a(Opcode::Load, 0), ins_a(Opcode::PushConst, 0), ins(Opcode::Add), ins(Opcode::Return)};
  return p;
}

BytecodeProgram make_type_error_program() {
  BytecodeProgram p;
  p.consts = {Value::from_bool(true)};
  p.code = {ins_a(Opcode::PushConst, 0), ins(Opcode::Neg)};
  return p;
}

BytecodeProgram make_timeout_program() {
  BytecodeProgram p;
  p.code = {ins_a(Opcode::Jmp, 0)};
  return p;
}

BytecodeProgram make_return_const_program(int v) {
  BytecodeProgram p;
  p.consts = {Value::from_int(v)};
  p.code = {ins_a(Opcode::PushConst, 0), ins(Opcode::Return)};
  return p;
}

BytecodeProgram make_return_bool_program(bool v) {
  BytecodeProgram p;
  p.consts = {Value::from_bool(v)};
  p.code = {ins_a(Opcode::PushConst, 0), ins(Opcode::Return)};
  return p;
}

BytecodeProgram make_return_huge_int_program() {
  BytecodeProgram p;
  p.consts = {Value::from_int(4617273859665047847LL)};
  p.code = {ins_a(Opcode::PushConst, 0), ins(Opcode::Return)};
  return p;
}

BytecodeProgram make_return_nan_program() {
  BytecodeProgram p;
  p.consts = {Value::from_float(std::numeric_limits<double>::quiet_NaN())};
  p.code = {ins_a(Opcode::PushConst, 0), ins(Opcode::Return)};
  return p;
}

BytecodeProgram make_return_string_program() {
  BytecodeProgram p;
  p.consts = {Value::from_string_hash_len(0x1234ULL, 3)};
  p.code = {ins_a(Opcode::PushConst, 0), ins(Opcode::Return)};
  return p;
}

BytecodeProgram make_index_fallback_program() {
  BytecodeProgram p;
  p.consts = {Value::from_string_hash_len(0x1234ULL, 3), Value::from_int(1)};
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::PushConst, 1),
      ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Index), 2),
      ins(Opcode::Return),
  };
  return p;
}

BytecodeProgram make_wrap_add_program() {
  BytecodeProgram p;
  p.consts = {Value::from_int(LLONG_MAX), Value::from_int(1)};
  p.code = {ins_a(Opcode::PushConst, 0), ins_a(Opcode::PushConst, 1), ins(Opcode::Add), ins(Opcode::Return)};
  return p;
}

BytecodeProgram make_float_mod_div_program() {
  BytecodeProgram p;
  p.n_locals = 1;
  p.consts = {Value::from_float(2.207), Value::from_float(3.0)};
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::PushConst, 1),
      ins_a(Opcode::Load, 0),
      ins(Opcode::Mod),
      ins(Opcode::Div),
      ins(Opcode::Return),
  };
  return p;
}

BytecodeProgram make_mixed_numeric_type_error_program() {
  BytecodeProgram p;
  p.consts = {Value::from_int(1), Value::from_float(1.5)};
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::PushConst, 1),
      ins(Opcode::Add),
      ins(Opcode::Return),
  };
  return p;
}

BytecodeProgram make_nested_exact_string_payload_program() {
  BytecodeProgram p;
  p.n_locals = 1;
  p.consts = {
      gagp::payload::make_string_value("wepw"),
      Value::from_int(-4),
      Value::from_int(-2),
      Value::from_int(1),
      Value::from_int(-4),
      gagp::payload::make_string_value("vww"),
      gagp::payload::make_string_value("cnwl"),
  };
  p.code = {
      ins_a(Opcode::Load, 0),
      ins_a(Opcode::Load, 0),
      ins(Opcode::Gt),
      ins_a(Opcode::JmpIfFalse, 6),
      ins_a(Opcode::Load, 0),
      ins_a(Opcode::Jmp, 7),
      ins_a(Opcode::Load, 0),
      ins(Opcode::Neg),
      ins(Opcode::Neg),
      ins(Opcode::Neg),
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::PushConst, 1),
      ins_a(Opcode::PushConst, 2),
      ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Slice), 3),
      ins_a(Opcode::PushConst, 3),
      ins_a(Opcode::PushConst, 4),
      ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Slice), 3),
      ins_a(Opcode::PushConst, 5),
      ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Concat), 2),
      ins_a(Opcode::PushConst, 6),
      ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Concat), 2),
      ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Len), 1),
      ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Min), 2),
      ins(Opcode::Return),
  };
  return p;
}

BytecodeProgram make_exact_string_index_program(const Value& s, int idx) {
  BytecodeProgram p;
  p.consts = {s, Value::from_int(idx)};
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::PushConst, 1),
      ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Index), 2),
      ins(Opcode::Return),
  };
  return p;
}

BytecodeProgram make_contains_string_const_program(const Value& needle) {
  BytecodeProgram p;
  p.n_locals = 1;
  p.consts = {needle};
  p.code = {
      ins_a(Opcode::Load, 0),
      ins_a(Opcode::PushConst, 0),
      ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Contains), 2),
      ins(Opcode::Return),
  };
  return p;
}

BytecodeProgram make_repeated_list_append_len_program() {
  BytecodeProgram p;
  p.consts = {
      gagp::payload::make_int_list_value({
          Value::from_int(1),
          Value::from_int(2),
          Value::from_int(3),
          Value::from_int(4),
      }),
      Value::from_int(0),
  };
  for (int i = 0; i < 40; ++i) {
    p.code.push_back(ins_a(Opcode::PushConst, 0));
    p.code.push_back(ins_a(Opcode::PushConst, 1));
    p.code.push_back(ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Append), 2));
    p.code.push_back(ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Len), 1));
  }
  p.code.push_back(ins(Opcode::Return));
  return p;
}

BytecodeProgram make_builtin_program(std::vector<Value> consts, gagp::BuiltinId builtin, int argc) {
  BytecodeProgram p;
  p.consts = std::move(consts);
  for (int i = 0; i < argc; ++i) {
    p.code.push_back(ins_a(Opcode::PushConst, i));
  }
  p.code.push_back(ins_ab(Opcode::CallBuiltin, static_cast<int>(builtin), argc));
  p.code.push_back(ins(Opcode::Return));
  return p;
}

BytecodeProgram make_binary_op_program(std::vector<Value> consts, Opcode op) {
  BytecodeProgram p;
  p.consts = std::move(consts);
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::PushConst, 1),
      ins(op),
      ins(Opcode::Return),
  };
  return p;
}

BytecodeProgram make_empty_list_program(int list_tag) {
  BytecodeProgram p;
  p.code = {
      ins_a(Opcode::EmptyList, list_tag),
      ins(Opcode::Return),
  };
  return p;
}

BytecodeProgram make_empty_list_like_program(const Value& source) {
  BytecodeProgram p;
  p.consts = {source};
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins(Opcode::EmptyListLike),
      ins(Opcode::Return),
  };
  return p;
}

BytecodeProgram make_check_program(const Value& source, Opcode op) {
  BytecodeProgram p;
  p.consts = {source};
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins(op),
      ins(Opcode::Return),
  };
  return p;
}

std::uint64_t prepend_list_hash48(std::uint8_t type_code, const Value& elem, const Value& src) {
  std::uint64_t h = Value::fnv1a_init();
  h = Value::fnv1a_mix_u8(h, type_code);
  h = Value::fnv1a_mix_u8(h, 0x70U);
  h = Value::fnv1a_mix_u64(h, Value::shallow_hash64(elem));
  h = Value::fnv1a_mix_u64(h, Value::container_hash48(src));
  h = Value::fnv1a_mix_u64(h, static_cast<std::uint64_t>(Value::container_len(src)));
  return (h & Value::k_container_hash_mask);
}

BytecodeProgram make_char_conversion_chain_program() {
  BytecodeProgram p;
  p.consts = {Value::from_int(97)};
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Chr), 1),
      ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::ToUpper), 1),
      ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::CharToString), 1),
      ins(Opcode::Return),
  };
  return p;
}

BytecodeProgram make_string_char_ord_chain_program() {
  BytecodeProgram p;
  p.consts = {gagp::payload::make_string_value("E")};
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::StringToChar), 1),
      ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::ToLower), 1),
      ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Ord), 1),
      ins(Opcode::Return),
  };
  return p;
}

BytecodeProgram make_compiled_for_range_sum_program_with_bound(const Value& bound) {
  using gagp::evo::AstNode;
  using gagp::evo::AstProgram;
  using gagp::evo::NodeKind;
  using gagp::evo::ProgramGenome;

  AstProgram program;
  program.names = {"x", "i"};
  program.consts = {
      Value::from_int(0),
      bound,
  };
  program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::ASSIGN, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::FOR_RANGE, 1, 0},
      AstNode{NodeKind::CALL_LEN, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::ASSIGN, 0, 0},
      AstNode{NodeKind::ADD, 0, 0},
      AstNode{NodeKind::VAR, 0, 0},
      AstNode{NodeKind::VAR, 1, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::VAR, 0, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  ProgramGenome genome;
  genome.ast = program;
  return gagp::evo::compile_for_eval(genome);
}

BytecodeProgram make_compiled_for_range_sum_program() {
  return make_compiled_for_range_sum_program_with_bound(gagp::payload::make_int_list_value({
      Value::from_int(10),
      Value::from_int(20),
      Value::from_int(30),
      Value::from_int(40),
  }));
}

BytecodeProgram make_compiled_for_range_sum_program_with_direct_bound(const Value& bound) {
  using gagp::evo::AstNode;
  using gagp::evo::AstProgram;
  using gagp::evo::NodeKind;
  using gagp::evo::ProgramGenome;

  AstProgram program;
  program.names = {"x", "i"};
  program.consts = {
      Value::from_int(0),
      bound,
  };
  program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::ASSIGN, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::FOR_RANGE, 1, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::ASSIGN, 0, 0},
      AstNode{NodeKind::ADD, 0, 0},
      AstNode{NodeKind::VAR, 0, 0},
      AstNode{NodeKind::VAR, 1, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::VAR, 0, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  ProgramGenome genome;
  genome.ast = program;
  return gagp::evo::compile_for_eval(genome);
}

BytecodeProgram make_compiled_if_stmt_branch_program(const Value& condition) {
  using gagp::evo::AstNode;
  using gagp::evo::AstProgram;
  using gagp::evo::NodeKind;
  using gagp::evo::ProgramGenome;

  AstProgram program;
  program.names = {"x"};
  program.consts = {
      Value::from_int(0),
      condition,
      Value::from_int(7),
      Value::from_int(11),
  };
  program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::ASSIGN, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::IF_STMT, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::ASSIGN, 0, 0},
      AstNode{NodeKind::CONST, 2, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::ASSIGN, 0, 0},
      AstNode{NodeKind::CONST, 3, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::VAR, 0, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  ProgramGenome genome;
  genome.ast = program;
  return gagp::evo::compile_for_eval(genome);
}

BytecodeProgram make_compiled_if_expr_program(const Value& condition) {
  using gagp::evo::AstNode;
  using gagp::evo::AstProgram;
  using gagp::evo::NodeKind;
  using gagp::evo::ProgramGenome;

  AstProgram program;
  program.consts = {
      condition,
      Value::from_int(7),
      Value::from_int(11),
  };
  program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::IF_EXPR, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::CONST, 2, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  ProgramGenome genome;
  genome.ast = program;
  return gagp::evo::compile_for_eval(genome);
}

BytecodeProgram make_compiled_binary_expr_program(gagp::evo::NodeKind op,
                                                  const Value& lhs,
                                                  const Value& rhs) {
  using gagp::evo::AstNode;
  using gagp::evo::AstProgram;
  using gagp::evo::NodeKind;
  using gagp::evo::ProgramGenome;

  AstProgram program;
  program.consts = {lhs, rhs};
  program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{op, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  ProgramGenome genome;
  genome.ast = program;
  return gagp::evo::compile_for_eval(genome);
}

BytecodeProgram make_compiled_bool_binary_program(gagp::evo::NodeKind op,
                                                  const Value& lhs,
                                                  const Value& rhs) {
  return make_compiled_binary_expr_program(op, lhs, rhs);
}

BytecodeProgram make_compiled_builtin_call_program(gagp::evo::NodeKind op, std::vector<Value> consts) {
  using gagp::evo::AstNode;
  using gagp::evo::AstProgram;
  using gagp::evo::NodeKind;
  using gagp::evo::ProgramGenome;

  AstProgram program;
  program.consts = std::move(consts);
  program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{op, 0, 0},
  };
  for (std::size_t i = 0; i < program.consts.size(); ++i) {
    program.nodes.push_back(AstNode{NodeKind::CONST, static_cast<int>(i), 0});
  }
  program.nodes.push_back(AstNode{NodeKind::BLOCK_NIL, 0, 0});
  ProgramGenome genome;
  genome.ast = program;
  return gagp::evo::compile_for_eval(genome);
}

BytecodeProgram make_compiled_unary_program(gagp::evo::NodeKind op, const Value& operand) {
  using gagp::evo::AstNode;
  using gagp::evo::AstProgram;
  using gagp::evo::NodeKind;
  using gagp::evo::ProgramGenome;

  AstProgram program;
  program.consts = {operand};
  program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{op, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  ProgramGenome genome;
  genome.ast = program;
  return gagp::evo::compile_for_eval(genome);
}

BytecodeProgram make_compiled_map_list_program() {
  using gagp::evo::AstNode;
  using gagp::evo::AstProgram;
  using gagp::evo::ListTypeTag;
  using gagp::evo::NodeKind;
  using gagp::evo::ProgramGenome;

  AstProgram program;
  program.names = {"u"};
  program.consts = {
      gagp::payload::make_int_list_value({Value::from_int(1), Value::from_int(2), Value::from_int(3)}),
      Value::from_int(2),
  };
  program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::MAP_LIST, 0, static_cast<int>(ListTypeTag::Int)},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::MUL, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  ProgramGenome genome;
  genome.ast = program;
  return gagp::evo::compile_for_eval(genome);
}

BytecodeProgram make_compiled_filter_list_program() {
  using gagp::evo::AstNode;
  using gagp::evo::AstProgram;
  using gagp::evo::NodeKind;
  using gagp::evo::ProgramGenome;

  AstProgram program;
  program.names = {"u"};
  program.consts = {
      gagp::payload::make_int_list_value({Value::from_int(3), Value::from_int(1), Value::from_int(4)}),
      Value::from_int(2),
  };
  program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::FILTER_LIST, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::GT, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  ProgramGenome genome;
  genome.ast = program;
  return gagp::evo::compile_for_eval(genome);
}

BytecodeProgram make_compiled_float_map_list_program() {
  using gagp::evo::AstNode;
  using gagp::evo::AstProgram;
  using gagp::evo::ListTypeTag;
  using gagp::evo::NodeKind;
  using gagp::evo::ProgramGenome;

  AstProgram program;
  program.names = {"u"};
  program.consts = {
      gagp::payload::make_float_list_value({Value::from_float(1.0), Value::from_float(2.5)}),
      Value::from_float(2.0),
  };
  program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::MAP_LIST, 0, static_cast<int>(ListTypeTag::Float)},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::MUL, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  ProgramGenome genome;
  genome.ast = program;
  return gagp::evo::compile_for_eval(genome);
}

BytecodeProgram make_compiled_string_filter_list_program() {
  using gagp::evo::AstNode;
  using gagp::evo::AstProgram;
  using gagp::evo::NodeKind;
  using gagp::evo::ProgramGenome;

  const Value aa = gagp::payload::make_string_value("aa");
  const Value b = gagp::payload::make_string_value("b");
  const Value ccc = gagp::payload::make_string_value("ccc");

  AstProgram program;
  program.names = {"u"};
  program.consts = {
      gagp::payload::make_string_list_value({aa, b, ccc}),
      Value::from_int(1),
  };
  program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::FILTER_LIST, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::GT, 0, 0},
      AstNode{NodeKind::CALL_LEN, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  ProgramGenome genome;
  genome.ast = program;
  return gagp::evo::compile_for_eval(genome);
}

BytecodeProgram make_compiled_empty_string_map_list_program() {
  using gagp::evo::AstNode;
  using gagp::evo::AstProgram;
  using gagp::evo::ListTypeTag;
  using gagp::evo::NodeKind;
  using gagp::evo::ProgramGenome;

  AstProgram program;
  program.names = {"u"};
  program.consts = {
      gagp::payload::make_string_list_value({}),
      gagp::payload::make_string_value("!"),
  };
  program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::MAP_LIST, 0, static_cast<int>(ListTypeTag::String)},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::CALL_CONCAT, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  ProgramGenome genome;
  genome.ast = program;
  return gagp::evo::compile_for_eval(genome);
}

BytecodeProgram make_compiled_linear_rec_program() {
  using gagp::evo::AstNode;
  using gagp::evo::AstProgram;
  using gagp::evo::LinearRecBinders;
  using gagp::evo::NodeKind;
  using gagp::evo::ProgramGenome;

  AstProgram program;
  program.names = {"u", "v", "i"};
  program.consts = {
      gagp::payload::make_int_list_value({Value::from_int(1), Value::from_int(2), Value::from_int(3)}),
      Value::from_int(4),
      Value::from_int(0),
      Value::from_int(10),
      Value::from_int(100),
  };
  program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::LINEAR_REC, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::CONST, 2, 0},
      AstNode{NodeKind::ADD, 0, 0},
      AstNode{NodeKind::MUL, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 1, 0},
      AstNode{NodeKind::CONST, 3, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::ADD, 0, 0},
      AstNode{NodeKind::MUL, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::CONST, 4, 0},
      AstNode{NodeKind::BOUND_VAR, 2, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  program.linear_rec_binders = {LinearRecBinders{3, 0, 1, 2}};
  ProgramGenome genome;
  genome.ast = program;
  return gagp::evo::compile_for_eval(genome);
}

BytecodeProgram make_compiled_float_linear_rec_program() {
  using gagp::evo::AstNode;
  using gagp::evo::AstProgram;
  using gagp::evo::LinearRecBinders;
  using gagp::evo::NodeKind;
  using gagp::evo::ProgramGenome;

  AstProgram program;
  program.names = {"u", "v", "i"};
  program.consts = {
      gagp::payload::make_float_list_value({Value::from_float(1.5), Value::from_float(2.5)}),
      Value::from_int(0),
      Value::from_float(0.0),
  };
  program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::LINEAR_REC, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::CONST, 2, 0},
      AstNode{NodeKind::ADD, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 1, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  program.linear_rec_binders = {LinearRecBinders{3, 0, 1, 2}};
  ProgramGenome genome;
  genome.ast = program;
  return gagp::evo::compile_for_eval(genome);
}

BytecodeProgram make_compiled_string_linear_rec_program() {
  using gagp::evo::AstNode;
  using gagp::evo::AstProgram;
  using gagp::evo::LinearRecBinders;
  using gagp::evo::NodeKind;
  using gagp::evo::ProgramGenome;

  const Value str_a = gagp::payload::make_string_value("a");
  const Value str_b = gagp::payload::make_string_value("b");
  const Value str_c = gagp::payload::make_string_value("c");

  AstProgram program;
  program.names = {"u", "v", "i"};
  program.consts = {
      gagp::payload::make_string_list_value({str_a, str_b, str_c}),
      Value::from_int(0),
      gagp::payload::make_string_value(""),
  };
  program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::LINEAR_REC, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::CONST, 2, 0},
      AstNode{NodeKind::CALL_CONCAT, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 1, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  program.linear_rec_binders = {LinearRecBinders{3, 0, 1, 2}};
  ProgramGenome genome;
  genome.ast = program;
  return gagp::evo::compile_for_eval(genome);
}

BytecodeProgram make_asgp_dc_sum_program() {
  gagp::PhaseProgram solve;
  solve.n_locals = 3;
  solve.consts = {Value::from_int(0)};
  solve.binder_locals = {{10, 0}, {11, 1}, {12, 2}};
  solve.code = {
      ins_a(Opcode::Load, 0),
      ins_a(Opcode::PushConst, 0),
      ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Index), 2),
      ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Singleton), 1),
      ins(Opcode::Return),
  };

  gagp::PhaseProgram divide;
  divide.n_locals = 1;
  divide.consts = {Value::from_int(1)};
  divide.binder_locals = {{13, 0}};
  divide.code = {
      ins_a(Opcode::PushConst, 0),
      ins(Opcode::Return),
  };

  gagp::PhaseProgram combine;
  combine.n_locals = 2;
  combine.binder_locals = {{14, 0}, {15, 1}};
  combine.code = {
      ins_a(Opcode::Load, 0),
      ins_a(Opcode::Load, 1),
      ins(Opcode::Add),
      ins(Opcode::Return),
  };

  gagp::AsgpDcSegment segment;
  segment.solve_xs_name = 10;
  segment.solve_n_name = 11;
  segment.solve_lo_name = 12;
  segment.divide_n_name = 13;
  segment.combine_left_name = 14;
  segment.combine_right_name = 15;
  segment.solve = solve;
  segment.divide = divide;
  segment.combine = combine;

  BytecodeProgram p;
  p.consts = {
      gagp::payload::make_int_list_value({
          Value::from_int(1),
          Value::from_int(2),
          Value::from_int(3),
          Value::from_int(4),
      }),
  };
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::AsgpDc, 0),
      ins(Opcode::Return),
  };
  p.asgp_dc_segments = {segment};
  return p;
}

BytecodeProgram make_asgp_dc_string_concat_program() {
  gagp::PhaseProgram solve;
  solve.n_locals = 3;
  solve.consts = {Value::from_int(0)};
  solve.binder_locals = {{20, 0}, {24, 1}, {25, 2}};
  solve.code = {
      ins_a(Opcode::Load, 0),
      ins_a(Opcode::PushConst, 0),
      ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Index), 2),
      ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Singleton), 1),
      ins(Opcode::Return),
  };

  gagp::PhaseProgram divide;
  divide.n_locals = 1;
  divide.consts = {Value::from_int(1)};
  divide.binder_locals = {{21, 0}};
  divide.code = {
      ins_a(Opcode::PushConst, 0),
      ins(Opcode::Return),
  };

  gagp::PhaseProgram combine;
  combine.n_locals = 2;
  combine.binder_locals = {{22, 0}, {23, 1}};
  combine.code = {
      ins_a(Opcode::Load, 0),
      ins_a(Opcode::Load, 1),
      ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Concat), 2),
      ins(Opcode::Return),
  };

  gagp::AsgpDcSegment segment;
  segment.solve_xs_name = 20;
  segment.solve_n_name = 24;
  segment.solve_lo_name = 25;
  segment.divide_n_name = 21;
  segment.combine_left_name = 22;
  segment.combine_right_name = 23;
  segment.solve = solve;
  segment.divide = divide;
  segment.combine = combine;

  BytecodeProgram p;
  p.consts = {
      gagp::payload::make_string_value("abc"),
  };
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::AsgpDc, 0),
      ins(Opcode::Return),
  };
  p.asgp_dc_segments = {segment};
  return p;
}

BytecodeProgram make_compiled_asgp_dc_program(bool as_float) {
  using gagp::evo::AsgpDcBinders;
  using gagp::evo::AstNode;
  using gagp::evo::AstProgram;
  using gagp::evo::NodeKind;
  using gagp::evo::ProgramGenome;

  AstProgram program;
  program.names = {"xs", "n", "lo", "dn", "l", "r"};
  if (as_float) {
    program.consts = {
        gagp::payload::make_float_list_value({
            Value::from_float(1.0),
            Value::from_float(2.0),
            Value::from_float(3.0),
        }),
        Value::from_int(0),
        Value::from_int(1),
    };
  } else {
    program.consts = {
        gagp::payload::make_int_list_value({
            Value::from_int(1),
            Value::from_int(2),
            Value::from_int(3),
        }),
        Value::from_int(0),
        Value::from_int(1),
    };
  }
  program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::ASGP_DC, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::CALL_INDEX, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::CONST, 2, 0},
      AstNode{NodeKind::ADD, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 4, 0},
      AstNode{NodeKind::BOUND_VAR, 5, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  program.asgp_dc_binders = {AsgpDcBinders{3, 0, 1, 2, 3, 4, 5}};
  ProgramGenome genome;
  genome.ast = program;
  return gagp::evo::compile_for_eval(genome);
}

BytecodeProgram make_compiled_asgp_dc_hidden_local_program() {
  using gagp::evo::AsgpDcBinders;
  using gagp::evo::AstNode;
  using gagp::evo::AstProgram;
  using gagp::evo::NodeKind;
  using gagp::evo::ProgramGenome;

  AstProgram program;
  program.names = {"hidden", "xs", "n", "lo", "dn", "l", "r"};
  program.consts = {
      Value::from_int(7),
      gagp::payload::make_int_list_value({Value::from_int(1)}),
      Value::from_int(1),
  };
  program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::ASSIGN, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::ASGP_DC, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::VAR, 0, 0},
      AstNode{NodeKind::CONST, 2, 0},
      AstNode{NodeKind::ADD, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 5, 0},
      AstNode{NodeKind::BOUND_VAR, 6, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  program.asgp_dc_binders = {AsgpDcBinders{6, 1, 2, 3, 4, 5, 6}};
  ProgramGenome genome;
  genome.ast = program;
  return gagp::evo::compile_for_eval(genome);
}

BytecodeProgram make_compiled_asgp_dc_divide_hidden_local_program() {
  using gagp::evo::AsgpDcBinders;
  using gagp::evo::AstNode;
  using gagp::evo::AstProgram;
  using gagp::evo::NodeKind;
  using gagp::evo::ProgramGenome;

  AstProgram program;
  program.names = {"hidden", "xs", "n", "lo", "dn", "l", "r"};
  program.consts = {
      Value::from_int(7),
      gagp::payload::make_int_list_value({Value::from_int(1), Value::from_int(2)}),
      Value::from_int(1),
  };
  program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::ASSIGN, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::ASGP_DC, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::CONST, 2, 0},
      AstNode{NodeKind::VAR, 0, 0},
      AstNode{NodeKind::ADD, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 5, 0},
      AstNode{NodeKind::BOUND_VAR, 6, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  program.asgp_dc_binders = {AsgpDcBinders{6, 1, 2, 3, 4, 5, 6}};
  ProgramGenome genome;
  genome.ast = program;
  return gagp::evo::compile_for_eval(genome);
}

BytecodeProgram make_asgp_dp1d_backward_sum_program() {
  gagp::PhaseProgram solve;
  solve.n_locals = 1;
  solve.consts = {Value::from_int(1)};
  solve.binder_locals = {{10, 0}};
  solve.code = {
      ins_a(Opcode::PushConst, 0),
      ins(Opcode::Return),
  };

  gagp::PhaseProgram transition;
  transition.n_locals = 2;
  transition.binder_locals = {{11, 0}, {12, 1}};
  transition.code = {
      ins_a(Opcode::Load, 1),
      ins_a(Opcode::Load, 0),
      ins(Opcode::Add),
      ins(Opcode::Return),
  };

  gagp::AsgpDp1dSegment segment;
  segment.lo = 0;
  segment.hi = 5;
  segment.base_state = 0;
  segment.boundary_value = Value::from_int(99);
  segment.dep_kind = -1;
  segment.dep_offsets = {1};
  segment.solve_state_name = 10;
  segment.transition_state_name = 11;
  segment.transition_dep_names = {12};
  segment.solve = solve;
  segment.transition = transition;

  BytecodeProgram p;
  p.consts = {Value::from_int(4)};
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::AsgpDp1d, 0),
      ins(Opcode::Return),
  };
  p.asgp_dp1d_segments = {segment};
  return p;
}

BytecodeProgram make_asgp_dp1d_fib_program() {
  gagp::PhaseProgram solve;
  solve.n_locals = 1;
  solve.consts = {Value::from_int(1)};
  solve.binder_locals = {{10, 0}};
  solve.code = {
      ins_a(Opcode::PushConst, 0),
      ins(Opcode::Return),
  };

  gagp::PhaseProgram transition;
  transition.n_locals = 3;
  transition.binder_locals = {{11, 0}, {12, 1}, {13, 2}};
  transition.code = {
      ins_a(Opcode::Load, 1),
      ins_a(Opcode::Load, 2),
      ins(Opcode::Add),
      ins(Opcode::Return),
  };

  gagp::AsgpDp1dSegment segment;
  segment.lo = 0;
  segment.hi = 20;
  segment.base_state = 0;
  segment.boundary_value = Value::from_int(0);
  segment.dep_kind = -1;
  segment.dep_offsets = {1, 2};
  segment.solve_state_name = 10;
  segment.transition_state_name = 11;
  segment.transition_dep_names = {12, 13};
  segment.solve = solve;
  segment.transition = transition;

  BytecodeProgram p;
  p.consts = {Value::from_int(8)};
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::AsgpDp1d, 0),
      ins(Opcode::Return),
  };
  p.asgp_dp1d_segments = {segment};
  return p;
}

BytecodeProgram make_asgp_dp1d_boundary_program() {
  gagp::PhaseProgram solve;
  solve.n_locals = 1;
  solve.consts = {Value::from_int(1)};
  solve.binder_locals = {{10, 0}};
  solve.code = {
      ins_a(Opcode::PushConst, 0),
      ins(Opcode::Return),
  };

  gagp::PhaseProgram transition;
  transition.n_locals = 2;
  transition.binder_locals = {{11, 0}, {12, 1}};
  transition.code = {
      ins_a(Opcode::Load, 1),
      ins(Opcode::Return),
  };

  gagp::AsgpDp1dSegment segment;
  segment.lo = 0;
  segment.hi = 5;
  segment.base_state = 0;
  segment.boundary_value = Value::from_int(123);
  segment.dep_kind = -1;
  segment.dep_offsets = {1};
  segment.solve_state_name = 10;
  segment.transition_state_name = 11;
  segment.transition_dep_names = {12};
  segment.solve = solve;
  segment.transition = transition;

  BytecodeProgram p;
  p.consts = {Value::from_int(-1)};
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::AsgpDp1d, 0),
      ins(Opcode::Return),
  };
  p.asgp_dp1d_segments = {segment};
  return p;
}

BytecodeProgram make_asgp_dp1d_dependency_type_error_program() {
  gagp::PhaseProgram solve;
  solve.n_locals = 1;
  solve.consts = {Value::from_int(1)};
  solve.binder_locals = {{10, 0}};
  solve.code = {
      ins_a(Opcode::PushConst, 0),
      ins(Opcode::Return),
  };

  gagp::PhaseProgram transition;
  transition.n_locals = 3;
  transition.binder_locals = {{11, 0}, {12, 1}, {13, 2}};
  transition.code = {
      ins_a(Opcode::Load, 1),
      ins(Opcode::Return),
  };

  gagp::AsgpDp1dSegment segment;
  segment.lo = 0;
  segment.hi = 5;
  segment.base_state = 0;
  segment.boundary_value = Value::from_bool(true);
  segment.dep_kind = -1;
  segment.dep_offsets = {1, 2};
  segment.solve_state_name = 10;
  segment.transition_state_name = 11;
  segment.transition_dep_names = {12, 13};
  segment.solve = solve;
  segment.transition = transition;

  BytecodeProgram p;
  p.consts = {Value::from_int(1)};
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::AsgpDp1d, 0),
      ins(Opcode::Return),
  };
  p.asgp_dp1d_segments = {segment};
  return p;
}

BytecodeProgram make_asgp_dp1d_transition_type_error_program() {
  gagp::PhaseProgram solve;
  solve.n_locals = 1;
  solve.consts = {Value::from_int(1)};
  solve.binder_locals = {{10, 0}};
  solve.code = {
      ins_a(Opcode::PushConst, 0),
      ins(Opcode::Return),
  };

  gagp::PhaseProgram transition;
  transition.n_locals = 2;
  transition.consts = {Value::from_bool(true)};
  transition.binder_locals = {{11, 0}, {12, 1}};
  transition.code = {
      ins_a(Opcode::PushConst, 0),
      ins(Opcode::Return),
  };

  gagp::AsgpDp1dSegment segment;
  segment.lo = 0;
  segment.hi = 5;
  segment.base_state = 0;
  segment.boundary_value = Value::from_int(0);
  segment.dep_kind = -1;
  segment.dep_offsets = {1};
  segment.solve_state_name = 10;
  segment.transition_state_name = 11;
  segment.transition_dep_names = {12};
  segment.solve = solve;
  segment.transition = transition;

  BytecodeProgram p;
  p.consts = {Value::from_int(1)};
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::AsgpDp1d, 0),
      ins(Opcode::Return),
  };
  p.asgp_dp1d_segments = {segment};
  return p;
}

BytecodeProgram make_asgp_dp1d_state_type_error_program() {
  BytecodeProgram p = make_asgp_dp1d_boundary_program();
  p.consts = {Value::from_bool(true)};
  return p;
}

BytecodeProgram make_asgp_dp2d_grid_program() {
  gagp::PhaseProgram solve;
  solve.n_locals = 2;
  solve.consts = {Value::from_int(1)};
  solve.binder_locals = {{10, 0}, {11, 1}};
  solve.code = {
      ins_a(Opcode::PushConst, 0),
      ins(Opcode::Return),
  };

  gagp::PhaseProgram transition;
  transition.n_locals = 4;
  transition.binder_locals = {{12, 0}, {13, 1}, {14, 2}, {15, 3}};
  transition.code = {
      ins_a(Opcode::Load, 2),
      ins_a(Opcode::Load, 3),
      ins(Opcode::Add),
      ins(Opcode::Return),
  };

  gagp::AsgpDp2dSegment segment;
  segment.i_lo = 0;
  segment.i_hi = 3;
  segment.j_lo = 0;
  segment.j_hi = 3;
  segment.base_i = 0;
  segment.base_j = 0;
  segment.boundary_value = Value::from_int(0);
  segment.dep_kind = 0;
  segment.solve_i_name = 10;
  segment.solve_j_name = 11;
  segment.transition_i_name = 12;
  segment.transition_j_name = 13;
  segment.transition_dep_names = {14, 15};
  segment.solve = solve;
  segment.transition = transition;

  BytecodeProgram p;
  p.consts = {Value::from_int(2), Value::from_int(2)};
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::PushConst, 1),
      ins_a(Opcode::AsgpDp2d, 0),
      ins(Opcode::Return),
  };
  p.asgp_dp2d_segments = {segment};
  return p;
}

BytecodeProgram make_asgp_dp2d_boundary_program() {
  gagp::PhaseProgram solve;
  solve.n_locals = 2;
  solve.consts = {Value::from_int(1)};
  solve.binder_locals = {{10, 0}, {11, 1}};
  solve.code = {
      ins_a(Opcode::PushConst, 0),
      ins(Opcode::Return),
  };

  gagp::PhaseProgram transition;
  transition.n_locals = 3;
  transition.binder_locals = {{12, 0}, {13, 1}, {14, 2}};
  transition.code = {
      ins_a(Opcode::Load, 2),
      ins(Opcode::Return),
  };

  gagp::AsgpDp2dSegment segment;
  segment.i_lo = 0;
  segment.i_hi = 3;
  segment.j_lo = 0;
  segment.j_hi = 3;
  segment.base_i = 0;
  segment.base_j = 0;
  segment.boundary_value = Value::from_int(321);
  segment.dep_kind = 2;
  segment.solve_i_name = 10;
  segment.solve_j_name = 11;
  segment.transition_i_name = 12;
  segment.transition_j_name = 13;
  segment.transition_dep_names = {14};
  segment.solve = solve;
  segment.transition = transition;

  BytecodeProgram p;
  p.consts = {Value::from_int(-1), Value::from_int(0)};
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::PushConst, 1),
      ins_a(Opcode::AsgpDp2d, 0),
      ins(Opcode::Return),
  };
  p.asgp_dp2d_segments = {segment};
  return p;
}

BytecodeProgram make_asgp_dp2d_dep_kind_program(int dep_kind,
                                                long long state_i,
                                                long long state_j,
                                                long long base_i,
                                                long long base_j) {
  const int dep_count = (dep_kind == 2 || dep_kind == 3) ? 1 : (dep_kind == 4 || dep_kind == 5) ? 3 : 2;

  gagp::PhaseProgram solve;
  solve.n_locals = 2;
  solve.consts = {Value::from_int(1)};
  solve.binder_locals = {{10, 0}, {11, 1}};
  solve.code = {
      ins_a(Opcode::PushConst, 0),
      ins(Opcode::Return),
  };

  gagp::PhaseProgram transition;
  transition.n_locals = 2 + dep_count;
  transition.binder_locals = {{12, 0}, {13, 1}};
  for (int i = 0; i < dep_count; ++i) {
    transition.binder_locals[14 + i] = 2 + i;
  }
  transition.code.push_back(ins_a(Opcode::Load, 2));
  for (int i = 1; i < dep_count; ++i) {
    transition.code.push_back(ins_a(Opcode::Load, 2 + i));
    transition.code.push_back(ins(Opcode::Add));
  }
  transition.code.push_back(ins(Opcode::Return));

  gagp::AsgpDp2dSegment segment;
  segment.i_lo = 0;
  segment.i_hi = 3;
  segment.j_lo = 0;
  segment.j_hi = 3;
  segment.base_i = base_i;
  segment.base_j = base_j;
  segment.boundary_value = Value::from_int(0);
  segment.dep_kind = dep_kind;
  segment.solve_i_name = 10;
  segment.solve_j_name = 11;
  segment.transition_i_name = 12;
  segment.transition_j_name = 13;
  for (int i = 0; i < dep_count; ++i) {
    segment.transition_dep_names.push_back(14 + i);
  }
  segment.solve = solve;
  segment.transition = transition;

  BytecodeProgram p;
  p.consts = {Value::from_int(state_i), Value::from_int(state_j)};
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::PushConst, 1),
      ins_a(Opcode::AsgpDp2d, 0),
      ins(Opcode::Return),
  };
  p.asgp_dp2d_segments = {segment};
  return p;
}

BytecodeProgram make_asgp_dp2d_dependency_type_error_program() {
  gagp::PhaseProgram solve;
  solve.n_locals = 2;
  solve.consts = {Value::from_int(1)};
  solve.binder_locals = {{10, 0}, {11, 1}};
  solve.code = {
      ins_a(Opcode::PushConst, 0),
      ins(Opcode::Return),
  };

  gagp::PhaseProgram transition;
  transition.n_locals = 4;
  transition.binder_locals = {{12, 0}, {13, 1}, {14, 2}, {15, 3}};
  transition.code = {
      ins_a(Opcode::Load, 2),
      ins(Opcode::Return),
  };

  gagp::AsgpDp2dSegment segment;
  segment.i_lo = 0;
  segment.i_hi = 3;
  segment.j_lo = 0;
  segment.j_hi = 3;
  segment.base_i = 0;
  segment.base_j = 0;
  segment.boundary_value = Value::from_bool(true);
  segment.dep_kind = 0;
  segment.solve_i_name = 10;
  segment.solve_j_name = 11;
  segment.transition_i_name = 12;
  segment.transition_j_name = 13;
  segment.transition_dep_names = {14, 15};
  segment.solve = solve;
  segment.transition = transition;

  BytecodeProgram p;
  p.consts = {Value::from_int(1), Value::from_int(0)};
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::PushConst, 1),
      ins_a(Opcode::AsgpDp2d, 0),
      ins(Opcode::Return),
  };
  p.asgp_dp2d_segments = {segment};
  return p;
}

BytecodeProgram make_asgp_dp2d_transition_type_error_program() {
  gagp::PhaseProgram solve;
  solve.n_locals = 2;
  solve.consts = {Value::from_int(1)};
  solve.binder_locals = {{10, 0}, {11, 1}};
  solve.code = {
      ins_a(Opcode::PushConst, 0),
      ins(Opcode::Return),
  };

  gagp::PhaseProgram transition;
  transition.n_locals = 3;
  transition.consts = {Value::from_bool(true)};
  transition.binder_locals = {{12, 0}, {13, 1}, {14, 2}};
  transition.code = {
      ins_a(Opcode::PushConst, 0),
      ins(Opcode::Return),
  };

  gagp::AsgpDp2dSegment segment;
  segment.i_lo = 0;
  segment.i_hi = 3;
  segment.j_lo = 0;
  segment.j_hi = 3;
  segment.base_i = 0;
  segment.base_j = 0;
  segment.boundary_value = Value::from_int(0);
  segment.dep_kind = 2;
  segment.solve_i_name = 10;
  segment.solve_j_name = 11;
  segment.transition_i_name = 12;
  segment.transition_j_name = 13;
  segment.transition_dep_names = {14};
  segment.solve = solve;
  segment.transition = transition;

  BytecodeProgram p;
  p.consts = {Value::from_int(1), Value::from_int(1)};
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::PushConst, 1),
      ins_a(Opcode::AsgpDp2d, 0),
      ins(Opcode::Return),
  };
  p.asgp_dp2d_segments = {segment};
  return p;
}

BytecodeProgram make_asgp_dp2d_state_type_error_program(bool invalid_i) {
  BytecodeProgram p = make_asgp_dp2d_boundary_program();
  p.consts = invalid_i ? std::vector<Value>{Value::from_bool(true), Value::from_int(0)}
                       : std::vector<Value>{Value::from_int(0), Value::from_bool(true)};
  return p;
}

BytecodeProgram make_compiled_asgp_dp1d_transition_hidden_local_program() {
  using gagp::evo::AsgpDp1dSpec;
  using gagp::evo::AstNode;
  using gagp::evo::AstProgram;
  using gagp::evo::NodeKind;
  using gagp::evo::ProgramGenome;

  AstProgram program;
  program.names = {"hidden", "s", "d1"};
  program.consts = {Value::from_int(7), Value::from_int(2), Value::from_int(1), Value::from_int(0)};
  program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::ASSIGN, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::ASGP_DP1D, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::CONST, 2, 0},
      AstNode{NodeKind::VAR, 0, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  program.asgp_dp1d_specs = {
      AsgpDp1dSpec{6, 0, 3, 0, 3, NodeKind::DP1_BACKWARD1, {1}, 1, 1, {2}},
  };
  ProgramGenome genome;
  genome.ast = program;
  return gagp::evo::compile_for_eval(genome);
}

BytecodeProgram make_compiled_asgp_dp1d_solve_hidden_local_program() {
  using gagp::evo::AsgpDp1dSpec;
  using gagp::evo::AstNode;
  using gagp::evo::AstProgram;
  using gagp::evo::NodeKind;
  using gagp::evo::ProgramGenome;

  AstProgram program;
  program.names = {"hidden", "s", "d1"};
  program.consts = {Value::from_int(7), Value::from_int(0), Value::from_int(1)};
  program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::ASSIGN, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::ASGP_DP1D, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::VAR, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 2, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  program.asgp_dp1d_specs = {
      AsgpDp1dSpec{6, 0, 3, 0, 1, NodeKind::DP1_BACKWARD1, {1}, 1, 1, {2}},
  };
  ProgramGenome genome;
  genome.ast = program;
  return gagp::evo::compile_for_eval(genome);
}

BytecodeProgram make_compiled_asgp_dp2d_transition_hidden_local_program() {
  using gagp::evo::AsgpDp2dSpec;
  using gagp::evo::AstNode;
  using gagp::evo::AstProgram;
  using gagp::evo::NodeKind;
  using gagp::evo::ProgramGenome;

  AstProgram program;
  program.names = {"hidden", "i", "j", "ti", "tj", "d1"};
  program.consts = {Value::from_int(7), Value::from_int(1), Value::from_int(1), Value::from_int(0)};
  program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::ASSIGN, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::ASGP_DP2D, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::CONST, 2, 0},
      AstNode{NodeKind::VAR, 0, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  program.asgp_dp2d_specs = {
      AsgpDp2dSpec{6, 0, 3, 0, 3, 0, 0, 3, NodeKind::DP2_DIAGONAL_BACKWARD, 1, 2, 3, 4, {5}},
  };
  ProgramGenome genome;
  genome.ast = program;
  return gagp::evo::compile_for_eval(genome);
}

BytecodeProgram make_compiled_asgp_dp2d_solve_hidden_local_program() {
  using gagp::evo::AsgpDp2dSpec;
  using gagp::evo::AstNode;
  using gagp::evo::AstProgram;
  using gagp::evo::NodeKind;
  using gagp::evo::ProgramGenome;

  AstProgram program;
  program.names = {"hidden", "i", "j", "d1"};
  program.consts = {Value::from_int(7), Value::from_int(0)};
  program.nodes = {
      AstNode{NodeKind::PROGRAM, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::ASSIGN, 0, 0},
      AstNode{NodeKind::CONST, 0, 0},
      AstNode{NodeKind::BLOCK_CONS, 0, 0},
      AstNode{NodeKind::RETURN, 0, 0},
      AstNode{NodeKind::ASGP_DP2D, 0, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::CONST, 1, 0},
      AstNode{NodeKind::VAR, 0, 0},
      AstNode{NodeKind::BOUND_VAR, 3, 0},
      AstNode{NodeKind::BLOCK_NIL, 0, 0},
  };
  program.asgp_dp2d_specs = {
      AsgpDp2dSpec{6, 0, 3, 0, 3, 0, 0, 1, NodeKind::DP2_DIAGONAL_BACKWARD, 1, 2, 1, 2, {3}},
  };
  ProgramGenome genome;
  genome.ast = program;
  return gagp::evo::compile_for_eval(genome);
}

BytecodeProgram make_asgp_dp1d_string_concat_program() {
  gagp::PhaseProgram solve;
  solve.n_locals = 1;
  solve.consts = {gagp::payload::make_string_value("a")};
  solve.binder_locals = {{10, 0}};
  solve.code = {
      ins_a(Opcode::PushConst, 0),
      ins(Opcode::Return),
  };

  gagp::PhaseProgram transition;
  transition.n_locals = 2;
  transition.consts = {gagp::payload::make_string_value("a")};
  transition.binder_locals = {{11, 0}, {12, 1}};
  transition.code = {
      ins_a(Opcode::Load, 1),
      ins_a(Opcode::PushConst, 0),
      ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Concat), 2),
      ins(Opcode::Return),
  };

  gagp::AsgpDp1dSegment segment;
  segment.lo = 0;
  segment.hi = 5;
  segment.base_state = 0;
  segment.boundary_value = gagp::payload::make_string_value("");
  segment.dep_kind = -1;
  segment.dep_offsets = {1};
  segment.solve_state_name = 10;
  segment.transition_state_name = 11;
  segment.transition_dep_names = {12};
  segment.solve = solve;
  segment.transition = transition;

  BytecodeProgram p;
  p.consts = {Value::from_int(4)};
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::AsgpDp1d, 0),
      ins(Opcode::Return),
  };
  p.asgp_dp1d_segments = {segment};
  return p;
}

BytecodeProgram make_asgp_dp2d_string_concat_program() {
  gagp::PhaseProgram solve;
  solve.n_locals = 2;
  solve.consts = {gagp::payload::make_string_value("a")};
  solve.binder_locals = {{10, 0}, {11, 1}};
  solve.code = {
      ins_a(Opcode::PushConst, 0),
      ins(Opcode::Return),
  };

  gagp::PhaseProgram transition;
  transition.n_locals = 4;
  transition.binder_locals = {{12, 0}, {13, 1}, {14, 2}, {15, 3}};
  transition.code = {
      ins_a(Opcode::Load, 2),
      ins_a(Opcode::Load, 3),
      ins_ab(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Concat), 2),
      ins(Opcode::Return),
  };

  gagp::AsgpDp2dSegment segment;
  segment.i_lo = 0;
  segment.i_hi = 3;
  segment.j_lo = 0;
  segment.j_hi = 3;
  segment.base_i = 0;
  segment.base_j = 0;
  segment.boundary_value = gagp::payload::make_string_value("");
  segment.dep_kind = 0;
  segment.solve_i_name = 10;
  segment.solve_j_name = 11;
  segment.transition_i_name = 12;
  segment.transition_j_name = 13;
  segment.transition_dep_names = {14, 15};
  segment.solve = solve;
  segment.transition = transition;

  BytecodeProgram p;
  p.consts = {Value::from_int(2), Value::from_int(2)};
  p.code = {
      ins_a(Opcode::PushConst, 0),
      ins_a(Opcode::PushConst, 1),
      ins_a(Opcode::AsgpDp2d, 0),
      ins(Opcode::Return),
  };
  p.asgp_dp2d_segments = {segment};
  return p;
}

bool approx(double a, double b) {
  return std::fabs(a - b) <= 1e-9;
}

bool exact(double a, double b) {
  return a == b;
}

gagp::FitnessEvalResult eval_gpu_via_session(const std::vector<BytecodeProgram>& programs,
                                              const std::vector<CaseBindings>& shared_cases,
                                              const std::vector<Value>& shared_answer,
                                              int fuel,
                                              int blocksize,
                                              double penalty) {
  gagp::FitnessSessionGpu session;
  gagp::FitnessSessionInitResult init = session.init(shared_cases, shared_answer, fuel, blocksize, penalty);
  if (!init.ok) {
    gagp::FitnessEvalResult out;
    out.ok = false;
    out.err = init.err;
    return out;
  }
  return session.eval_programs(programs);
}

bool check_single_cpu_gpu_exact(const BytecodeProgram& program,
                                const Value& expected,
                                double expected_fitness,
                                int fuel,
                                int blocksize,
                                double penalty,
                                const std::string& label) {
  const std::vector<BytecodeProgram> programs{program};
  const std::vector<CaseBindings> shared_cases(1);
  const std::vector<Value> shared_answer{expected};
  const std::vector<double> cpu_fit =
      gagp::eval_fitness_cpu(programs, shared_cases, shared_answer, fuel, penalty, blocksize);
  const gagp::FitnessEvalResult gpu_fit =
      eval_gpu_via_session(programs, shared_cases, shared_answer, fuel, blocksize, penalty);
  if (!gpu_fit.ok) {
    if (gpu_fit.err.message.find("cuda device unavailable") != std::string::npos) {
      std::cout << "gagp_test_fitness_cpu_gpu_parity: SKIP (" << gpu_fit.err.message << ")\n";
      return true;
    }
    std::cerr << "FAIL: gpu fitness run failed on " << label << ": " << gpu_fit.err.message << "\n";
    return false;
  }
  if (cpu_fit.size() != 1 || gpu_fit.fitness.size() != 1) {
    std::cerr << "FAIL: cpu/gpu fitness size mismatch on " << label << "\n";
    return false;
  }
  if (!exact(cpu_fit[0], gpu_fit.fitness[0])) {
    std::cerr << "FAIL: cpu/gpu fitness mismatch on " << label
              << " cpu=" << cpu_fit[0] << " gpu=" << gpu_fit.fitness[0] << "\n";
    return false;
  }
  if (!approx(cpu_fit[0], expected_fitness)) {
    std::cerr << "FAIL: exact expected fitness mismatch on " << label
              << " expected=" << expected_fitness << " actual=" << cpu_fit[0] << "\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  const double penalty = 1.0;
  constexpr int kParityBlocksize = 128;

  {
    std::vector<BytecodeProgram> programs;
    programs.push_back(make_add_one_program());
    programs.push_back(make_return_const_program(7));
    programs.push_back(make_return_huge_int_program());
    programs.push_back(make_return_nan_program());
    programs.push_back(make_return_string_program());
    programs.push_back(make_index_fallback_program());
    programs.push_back(make_type_error_program());
    programs.push_back(make_timeout_program());

    std::vector<CaseBindings> shared_cases;
    std::vector<Value> shared_answer;
    for (int i = 0; i < 64; ++i) {
      shared_cases.push_back(CaseBindings{InputBinding{0, Value::from_int(i)}});
      shared_answer.push_back(Value::from_int(i + 1));
    }

    const std::vector<double> cpu_fit =
        gagp::eval_fitness_cpu(programs, shared_cases, shared_answer, 64, penalty, kParityBlocksize);
    const gagp::FitnessEvalResult gpu_fit =
        eval_gpu_via_session(programs, shared_cases, shared_answer, 64, kParityBlocksize, penalty);

    if (!gpu_fit.ok) {
      if (gpu_fit.err.message.find("cuda device unavailable") != std::string::npos) {
        std::cout << "gagp_test_fitness_cpu_gpu_parity: SKIP (" << gpu_fit.err.message << ")\n";
        return 0;
      }
      std::cerr << "FAIL: gpu fitness run failed: " << gpu_fit.err.message << "\n";
      return 1;
    }

    if (cpu_fit.size() != gpu_fit.fitness.size()) {
      std::cerr << "FAIL: cpu/gpu fitness size mismatch on numeric cases\n";
      return 1;
    }
    for (std::size_t i = 0; i < cpu_fit.size(); ++i) {
      if (!exact(cpu_fit[i], gpu_fit.fitness[i])) {
        std::cerr << "FAIL: cpu/gpu fitness mismatch on numeric cases at " << i << "\n";
        return 1;
      }
    }
    if (!approx(cpu_fit[0], 0.0)) {
      std::cerr << "FAIL: exact numeric program should score 0 MAE\n";
      return 1;
    }
    if (!approx(cpu_fit[1], -63.0 * penalty)) {
      std::cerr << "FAIL: constant numeric program should clamp per-case numeric loss to penalty\n";
      return 1;
    }
    if (!approx(cpu_fit[2], -64.0 * penalty)) {
      std::cerr << "FAIL: huge numeric outlier should clamp per case to penalty\n";
      return 1;
    }
    if (!approx(cpu_fit[3], -64.0 * penalty)) {
      std::cerr << "FAIL: non-finite numeric actual should accumulate penalty per case\n";
      return 1;
    }
    if (!approx(cpu_fit[4], -64.0 * penalty)) {
      std::cerr << "FAIL: non-numeric actual against numeric expected should accumulate penalty\n";
      return 1;
    }
    if (!approx(cpu_fit[5], -64.0 * penalty)) {
      std::cerr << "FAIL: fallback token actual against numeric expected should accumulate penalty\n";
      return 1;
    }
    if (!approx(cpu_fit[6], -64.0 * penalty) || !approx(cpu_fit[7], -64.0 * penalty)) {
      std::cerr << "FAIL: runtime errors on numeric cases should accumulate penalty\n";
      return 1;
    }
  }

  {
    std::vector<BytecodeProgram> programs;
    programs.push_back(make_return_string_program());
    programs.push_back(make_return_const_program(7));
    programs.push_back(make_type_error_program());

    std::vector<CaseBindings> shared_cases(16);
    std::vector<Value> shared_answer(16, Value::from_string_hash_len(0x9999ULL, 3));

    const std::vector<double> cpu_fit =
        gagp::eval_fitness_cpu(programs, shared_cases, shared_answer, 64, penalty, kParityBlocksize);
    const gagp::FitnessEvalResult gpu_fit =
        eval_gpu_via_session(programs, shared_cases, shared_answer, 64, kParityBlocksize, penalty);

    if (!gpu_fit.ok) {
      if (gpu_fit.err.message.find("cuda device unavailable") != std::string::npos) {
        std::cout << "gagp_test_fitness_cpu_gpu_parity: SKIP (" << gpu_fit.err.message << ")\n";
        return 0;
      }
      std::cerr << "FAIL: gpu fitness run failed on binary cases: " << gpu_fit.err.message << "\n";
      return 1;
    }

    if (cpu_fit.size() != gpu_fit.fitness.size()) {
      std::cerr << "FAIL: cpu/gpu fitness size mismatch on binary cases\n";
      return 1;
    }
    for (std::size_t i = 0; i < cpu_fit.size(); ++i) {
      if (!exact(cpu_fit[i], gpu_fit.fitness[i])) {
        std::cerr << "FAIL: cpu/gpu fitness mismatch on binary cases at " << i << "\n";
        return 1;
      }
    }
    if (!approx(cpu_fit[0], 0.0)) {
      std::cerr << "FAIL: same-tag binary mismatch should score 0\n";
      return 1;
    }
    if (!approx(cpu_fit[1], -16.0 * penalty)) {
      std::cerr << "FAIL: binary type mismatch should accumulate penalty\n";
      return 1;
    }
    if (!approx(cpu_fit[2], -16.0 * penalty)) {
      std::cerr << "FAIL: runtime errors on binary cases should accumulate penalty\n";
      return 1;
    }
  }

  {
    std::vector<BytecodeProgram> programs;
    programs.push_back(make_mixed_numeric_type_error_program());

    std::vector<CaseBindings> shared_cases{CaseBindings{}};
    std::vector<Value> shared_answer{Value::from_float(2.5)};

    const std::vector<double> cpu_fit =
        gagp::eval_fitness_cpu(programs, shared_cases, shared_answer, 64, penalty, kParityBlocksize);
    const gagp::FitnessEvalResult gpu_fit =
        eval_gpu_via_session(programs, shared_cases, shared_answer, 64, kParityBlocksize, penalty);

    if (!gpu_fit.ok) {
      if (gpu_fit.err.message.find("cuda device unavailable") != std::string::npos) {
        std::cout << "gagp_test_fitness_cpu_gpu_parity: SKIP (" << gpu_fit.err.message << ")\n";
        return 0;
      }
      std::cerr << "FAIL: gpu fitness run failed on mixed numeric type-error case: "
                << gpu_fit.err.message << "\n";
      return 1;
    }

    if (cpu_fit.size() != gpu_fit.fitness.size()) {
      std::cerr << "FAIL: cpu/gpu fitness size mismatch on mixed numeric type-error case\n";
      return 1;
    }
    if (!exact(cpu_fit[0], gpu_fit.fitness[0]) || !exact(cpu_fit[0], -penalty)) {
      std::cerr << "FAIL: mixed numeric type-error case should score runtime penalty on CPU/GPU\n";
      return 1;
    }
  }

  {
    gagp::payload::clear();
    std::vector<BytecodeProgram> programs;
    programs.push_back(make_return_bool_program(false));
    programs.push_back(make_return_bool_program(true));
    programs.push_back(make_contains_string_const_program(gagp::payload::make_string_value("a")));

    std::vector<CaseBindings> shared_cases;
    std::vector<Value> shared_answer;
    shared_cases.reserve(64);
    shared_answer.reserve(64);
    for (int i = 0; i < 64; ++i) {
      shared_cases.push_back(CaseBindings{InputBinding{
          0, gagp::payload::make_string_value((i % 2 == 0) ? "a" : "b")}});
      shared_answer.push_back(Value::from_bool(i % 2 == 0));
    }

    const std::vector<double> cpu_fit =
        gagp::eval_fitness_cpu(programs, shared_cases, shared_answer, 64, penalty, 1024);
    const gagp::FitnessEvalResult gpu_fit =
        eval_gpu_via_session(programs, shared_cases, shared_answer, 64, 1024, penalty);

    if (!gpu_fit.ok) {
      if (gpu_fit.err.message.find("cuda device unavailable") != std::string::npos) {
        std::cout << "gagp_test_fitness_cpu_gpu_parity: SKIP (" << gpu_fit.err.message << ")\n";
        return 0;
      }
      std::cerr << "FAIL: gpu fitness run failed on bool exact string cases: " << gpu_fit.err.message << "\n";
      return 1;
    }

    if (cpu_fit.size() != gpu_fit.fitness.size()) {
      std::cerr << "FAIL: cpu/gpu fitness size mismatch on bool exact string cases\n";
      return 1;
    }
    for (std::size_t i = 0; i < cpu_fit.size(); ++i) {
      if (!exact(cpu_fit[i], gpu_fit.fitness[i])) {
        std::cerr << "FAIL: cpu/gpu fitness mismatch on bool exact string cases at " << i
                  << " cpu=" << cpu_fit[i] << " gpu=" << gpu_fit.fitness[i] << "\n";
        return 1;
      }
      if (gpu_fit.fitness[i] > static_cast<double>(shared_cases.size())) {
        std::cerr << "FAIL: bool exact string fitness exceeded case count at " << i << "\n";
        return 1;
      }
    }
    if (!approx(cpu_fit[0], 32.0) || !approx(cpu_fit[1], 32.0) || !approx(cpu_fit[2], 64.0)) {
      std::cerr << "FAIL: bool exact string fitness should be bounded by case count\n";
      return 1;
    }
  }

  {
    std::vector<BytecodeProgram> programs;
    programs.push_back(make_wrap_add_program());

    std::vector<CaseBindings> shared_cases(4);
    std::vector<Value> shared_answer(4, Value::from_int(LLONG_MIN));

    const std::vector<double> cpu_fit =
        gagp::eval_fitness_cpu(programs, shared_cases, shared_answer, 64, penalty, kParityBlocksize);
    const gagp::FitnessEvalResult gpu_fit =
        eval_gpu_via_session(programs, shared_cases, shared_answer, 64, kParityBlocksize, penalty);

    if (!gpu_fit.ok) {
      if (gpu_fit.err.message.find("cuda device unavailable") != std::string::npos) {
        std::cout << "gagp_test_fitness_cpu_gpu_parity: SKIP (" << gpu_fit.err.message << ")\n";
        return 0;
      }
      std::cerr << "FAIL: gpu fitness run failed on wrap cases: " << gpu_fit.err.message << "\n";
      return 1;
    }

    if (cpu_fit.size() != gpu_fit.fitness.size()) {
      std::cerr << "FAIL: cpu/gpu fitness size mismatch on wrap cases\n";
      return 1;
    }
    for (std::size_t i = 0; i < cpu_fit.size(); ++i) {
      if (!exact(cpu_fit[i], gpu_fit.fitness[i])) {
        std::cerr << "FAIL: cpu/gpu fitness mismatch on wrap cases at " << i << "\n";
        return 1;
      }
    }
    if (!approx(cpu_fit[0], 0.0)) {
      std::cerr << "FAIL: wrap add should match expected wrapped result\n";
      return 1;
    }
  }

  {
    std::vector<BytecodeProgram> programs;
    programs.push_back(make_float_mod_div_program());

    std::vector<CaseBindings> shared_cases{
        CaseBindings{InputBinding{0, Value::from_float(-0.008797653959)}},
    };
    std::vector<Value> shared_answer{Value::from_float(0.99124093216)};

    const std::vector<double> cpu_fit =
        gagp::eval_fitness_cpu(programs, shared_cases, shared_answer, 64, penalty, kParityBlocksize);
    const gagp::FitnessEvalResult gpu_fit =
        eval_gpu_via_session(programs, shared_cases, shared_answer, 64, kParityBlocksize, penalty);

    if (!gpu_fit.ok) {
      if (gpu_fit.err.message.find("cuda device unavailable") != std::string::npos) {
        std::cout << "gagp_test_fitness_cpu_gpu_parity: SKIP (" << gpu_fit.err.message << ")\n";
        return 0;
      }
      std::cerr << "FAIL: gpu fitness run failed on float mod/div case: " << gpu_fit.err.message << "\n";
      return 1;
    }

    if (cpu_fit.size() != gpu_fit.fitness.size()) {
      std::cerr << "FAIL: cpu/gpu fitness size mismatch on float mod/div case\n";
      return 1;
    }
    if (cpu_fit[0] != gpu_fit.fitness[0]) {
      std::cerr << "FAIL: cpu/gpu fitness mismatch on float mod/div case\n";
      return 1;
    }
  }

  {
    gagp::payload::clear();
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({
                                     gagp::payload::make_int_list_value({
                                         Value::from_int(2),
                                         Value::from_int(3),
                                     }),
                                     Value::from_int(1),
                                 },
                                 gagp::BuiltinId::Prepend,
                                 2),
            gagp::payload::make_int_list_value({
                Value::from_int(1),
                Value::from_int(2),
                Value::from_int(3),
            }),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current prepend int_list builtin")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({
                                     gagp::payload::make_string_value("abc"),
                                     Value::from_int(1),
                                 },
                                 gagp::BuiltinId::Index,
                                 2),
            Value::from_char('b'),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current index string to char builtin")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_char_conversion_chain_program(),
                                    gagp::payload::make_string_value("A"),
                                    1.0,
                                    128,
                                    kParityBlocksize,
                                    penalty,
                                    "current char conversion chain")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_string_char_ord_chain_program(),
                                    Value::from_int('e'),
                                    0.0,
                                    128,
                                    kParityBlocksize,
                                    penalty,
                                    "current string-char-ord chain")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({Value::from_char('7')}, gagp::BuiltinId::IsDigit, 1),
            Value::from_bool(true),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current char digit predicate")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({Value::from_int(123)}, gagp::BuiltinId::ToString, 1),
            gagp::payload::make_string_value("123"),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current to_string int builtin")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({Value::from_float(1.5)}, gagp::BuiltinId::ToString, 1),
            gagp::payload::make_string_value("1.5"),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current to_string float builtin")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({Value::from_float(2.0)}, gagp::BuiltinId::ToString, 1),
            gagp::payload::make_string_value("2"),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current to_string whole float builtin")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({Value::from_float(-0.0)}, gagp::BuiltinId::ToString, 1),
            gagp::payload::make_string_value("0"),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current to_string negative zero float builtin")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({Value::from_float(1.2345678)}, gagp::BuiltinId::ToString, 1),
            gagp::payload::make_string_value("1.234568"),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current to_string rounded float builtin")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({Value::from_float(std::numeric_limits<double>::quiet_NaN())},
                                 gagp::BuiltinId::ToString,
                                 1),
            gagp::payload::make_string_value("nan"),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current to_string nan builtin")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({Value::from_float(std::numeric_limits<double>::infinity())},
                                 gagp::BuiltinId::ToString,
                                 1),
            gagp::payload::make_string_value("inf"),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current to_string positive infinity builtin")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({Value::from_float(-std::numeric_limits<double>::infinity())},
                                 gagp::BuiltinId::ToString,
                                 1),
            gagp::payload::make_string_value("-inf"),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current to_string negative infinity builtin")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({Value::from_int(-7), Value::from_int(2)}, gagp::BuiltinId::IDiv0, 2),
            Value::from_int(-3),
            0.0,
            128,
            kParityBlocksize,
            penalty,
            "current idiv0 negative truncating builtin")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({Value::from_int(-7), Value::from_int(2)}, gagp::BuiltinId::IMod0, 2),
            Value::from_int(1),
            0.0,
            128,
            kParityBlocksize,
            penalty,
            "current imod0 negative dividend builtin")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({Value::from_int(7), Value::from_int(-2)}, gagp::BuiltinId::IMod0, 2),
            Value::from_int(-1),
            0.0,
            128,
            kParityBlocksize,
            penalty,
            "current imod0 negative divisor builtin")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({Value::from_char('A')}, gagp::BuiltinId::IsLetter, 1),
            Value::from_bool(true),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current char letter predicate")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({Value::from_char(' ')}, gagp::BuiltinId::IsSpace, 1),
            Value::from_bool(true),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current char space predicate")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({Value::from_char('E')}, gagp::BuiltinId::IsVowel, 1),
            Value::from_bool(true),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current char vowel predicate")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({Value::from_int(7)}, gagp::BuiltinId::Singleton, 1),
            gagp::payload::make_int_list_value({Value::from_int(7)}),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current singleton int builtin")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({Value::from_float(1.5)}, gagp::BuiltinId::Singleton, 1),
            gagp::payload::make_float_list_value({Value::from_float(1.5)}),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current singleton float builtin")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({Value::from_char('z')}, gagp::BuiltinId::Singleton, 1),
            gagp::payload::make_string_value("z"),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current singleton char builtin")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({gagp::payload::make_string_value("ab")}, gagp::BuiltinId::Singleton, 1),
            gagp::payload::make_string_list_value({gagp::payload::make_string_value("ab")}),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current singleton string builtin")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({gagp::payload::make_string_value("ab")},
                                 gagp::BuiltinId::StringToChar,
                                 1),
            Value::from_char('a'),
            -penalty,
            128,
            kParityBlocksize,
            penalty,
            "current string_to_char length ValueError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({Value::from_int(256)}, gagp::BuiltinId::Chr, 1),
            Value::from_char(0),
            -penalty,
            128,
            kParityBlocksize,
            penalty,
            "current chr invalid code point ValueError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({Value::from_char('q')}, gagp::BuiltinId::ToString, 1),
            gagp::payload::make_string_value("q"),
            -penalty,
            128,
            kParityBlocksize,
            penalty,
            "current to_string char TypeError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({Value::from_char(300)}, gagp::BuiltinId::CharToString, 1),
            gagp::payload::make_string_value("?"),
            -penalty,
            128,
            kParityBlocksize,
            penalty,
            "current char_to_string unsupported char ValueError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({Value::from_int(3)}, gagp::BuiltinId::Len, 1),
            Value::from_int(0),
            -penalty,
            128,
            kParityBlocksize,
            penalty,
            "current len non-container TypeError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({
                                     gagp::payload::make_string_value("abc"),
                                     Value::from_bool(false),
                                     Value::from_int(2),
                                 },
                                 gagp::BuiltinId::Slice,
                                 3),
            gagp::payload::make_string_value("ab"),
            -penalty,
            128,
            kParityBlocksize,
            penalty,
            "current slice non-int bound TypeError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({Value::from_int(3)}, gagp::BuiltinId::Reverse, 1),
            Value::from_int(3),
            -penalty,
            128,
            kParityBlocksize,
            penalty,
            "current reverse non-container TypeError parity")) {
      return 1;
    }
    const Value cmp_float_list = Value::from_float_list_hash_len(0x9191ULL, 3);
    if (!check_single_cpu_gpu_exact(
            make_binary_op_program({cmp_float_list, cmp_float_list}, Opcode::Eq),
            Value::from_bool(true),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current FloatList EQ fallback token parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_binary_op_program({
                                       gagp::payload::make_string_list_value({
                                           gagp::payload::make_string_value("a"),
                                       }),
                                       gagp::payload::make_string_list_value({
                                           gagp::payload::make_string_value("b"),
                                       }),
                                   },
                                   Opcode::Ne),
            Value::from_bool(true),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current StringList NE exact payload parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_binary_op_program({
                                       gagp::payload::make_string_value("a"),
                                       gagp::payload::make_string_value("b"),
                                   },
                                   Opcode::Lt),
            Value::from_bool(false),
            -penalty,
            128,
            kParityBlocksize,
            penalty,
            "current String ordering TypeError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_empty_list_program(1),
                                    gagp::payload::make_int_list_value({}),
                                    1.0,
                                    128,
                                    kParityBlocksize,
                                    penalty,
                                    "current EMPTY_LIST IntList exact parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_empty_list_program(2),
                                    gagp::payload::make_float_list_value({}),
                                    1.0,
                                    128,
                                    kParityBlocksize,
                                    penalty,
                                    "current EMPTY_LIST FloatList exact parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_empty_list_program(3),
                                    gagp::payload::make_string_list_value({}),
                                    1.0,
                                    128,
                                    kParityBlocksize,
                                    penalty,
                                    "current EMPTY_LIST StringList exact parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_empty_list_program(99),
                                    gagp::payload::make_int_list_value({}),
                                    -penalty,
                                    128,
                                    kParityBlocksize,
                                    penalty,
                                    "current EMPTY_LIST invalid tag TypeError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_empty_list_like_program(gagp::payload::make_string_list_value({
                gagp::payload::make_string_value("x"),
            })),
            gagp::payload::make_string_list_value({}),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current EMPTY_LIST_LIKE StringList exact parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_empty_list_like_program(gagp::payload::make_int_list_value({Value::from_int(1)})),
            gagp::payload::make_int_list_value({}),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current EMPTY_LIST_LIKE IntList exact parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_empty_list_like_program(gagp::payload::make_float_list_value({Value::from_float(1.25)})),
            gagp::payload::make_float_list_value({}),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current EMPTY_LIST_LIKE FloatList exact parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_empty_list_like_program(Value::from_int(7)),
                                    gagp::payload::make_int_list_value({}),
                                    -penalty,
                                    128,
                                    kParityBlocksize,
                                    penalty,
                                    "current EMPTY_LIST_LIKE non-list TypeError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_check_program(gagp::payload::make_string_list_value({
                                   gagp::payload::make_string_value("ok"),
                               }),
                               Opcode::CheckList),
            gagp::payload::make_string_list_value({
                gagp::payload::make_string_value("ok"),
            }),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current CHECK_LIST StringList pass-through parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_check_program(Value::from_int(42), Opcode::CheckInt),
                                    Value::from_int(42),
                                    0.0,
                                    128,
                                    kParityBlocksize,
                                    penalty,
                                    "current CHECK_INT Int pass-through parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_check_program(Value::from_int(3), Opcode::CheckList),
                                    Value::from_int(3),
                                    -penalty,
                                    128,
                                    kParityBlocksize,
                                    penalty,
                                    "current CHECK_LIST non-list TypeError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_check_program(Value::from_bool(true), Opcode::CheckInt),
                                    Value::from_bool(true),
                                    -penalty,
                                    128,
                                    kParityBlocksize,
                                    penalty,
                                    "current CHECK_INT non-int TypeError parity")) {
      return 1;
    }
  }

  {
    gagp::payload::clear();
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({
                                     gagp::payload::make_string_value("we"),
                                     Value::from_int(1),
                                     Value::from_int(-4),
                                 },
                                 gagp::BuiltinId::Slice,
                                 3),
            gagp::payload::make_string_value(""),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current empty string slice exact tag parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({
                                     gagp::payload::make_float_list_value({
                                         Value::from_float(1.0),
                                         Value::from_float(2.0),
                                     }),
                                     Value::from_int(2),
                                     Value::from_int(0),
                                 },
                                 gagp::BuiltinId::Slice,
                                 3),
            gagp::payload::make_float_list_value({}),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current empty FloatList slice exact tag parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({
                                     gagp::payload::make_string_list_value({
                                         gagp::payload::make_string_value("a"),
                                         gagp::payload::make_string_value("b"),
                                     }),
                                     Value::from_int(2),
                                     Value::from_int(0),
                                 },
                                 gagp::BuiltinId::Slice,
                                 3),
            gagp::payload::make_string_list_value({}),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current empty StringList slice exact tag parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({
                                     gagp::payload::make_float_list_value({
                                         Value::from_float(1.0),
                                         Value::from_float(2.5),
                                     }),
                                     Value::from_int(-1),
                                 },
                                 gagp::BuiltinId::Index,
                                 2),
            Value::from_float(2.5),
            0.0,
            128,
            kParityBlocksize,
            penalty,
            "current FloatList negative index exact parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({
                                     gagp::payload::make_string_list_value({
                                         gagp::payload::make_string_value("aa"),
                                         gagp::payload::make_string_value("bb"),
                                     }),
                                     Value::from_int(-1),
                                 },
                                 gagp::BuiltinId::Index,
                                 2),
            gagp::payload::make_string_value("bb"),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current StringList negative index exact parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({
                                     gagp::payload::make_int_list_value({
                                         Value::from_int(1),
                                         Value::from_int(2),
                                     }),
                                     Value::from_int(2),
                                 },
                                 gagp::BuiltinId::Index,
                                 2),
            Value::from_int(0),
            -penalty,
            128,
            kParityBlocksize,
            penalty,
            "current IntList index out-of-range ValueError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({
                                     gagp::payload::make_string_value("abracadabra"),
                                     gagp::payload::make_string_value("xyz"),
                                 },
                                 gagp::BuiltinId::Find,
                                 2),
            Value::from_int(-1),
            0.0,
            128,
            kParityBlocksize,
            penalty,
            "current find absent substring parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({
                                     gagp::payload::make_string_value("abracadabra"),
                                     gagp::payload::make_string_value(""),
                                 },
                                 gagp::BuiltinId::Find,
                                 2),
            Value::from_int(0),
            0.0,
            128,
            kParityBlocksize,
            penalty,
            "current find empty substring parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({
                                     gagp::payload::make_string_value("abracadabra"),
                                     gagp::payload::make_string_value("xyz"),
                                 },
                                 gagp::BuiltinId::Contains,
                                 2),
            Value::from_bool(false),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current contains absent substring parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({
                                     gagp::payload::make_string_value("abracadabra"),
                                     gagp::payload::make_string_value(""),
                                 },
                                 gagp::BuiltinId::Contains,
                                 2),
            Value::from_bool(true),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current contains empty substring parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({
                                     Value::from_string_hash_len(0x9876ULL, 3),
                                     gagp::payload::make_string_value("a"),
                                 },
                                 gagp::BuiltinId::Find,
                                 2),
            Value::from_int(0),
            -penalty,
            128,
            kParityBlocksize,
            penalty,
            "current find missing exact haystack payload ValueError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({
                                     gagp::payload::make_string_value("abc"),
                                     Value::from_string_hash_len(0x9877ULL, 1),
                                 },
                                 gagp::BuiltinId::Contains,
                                 2),
            Value::from_bool(false),
            -penalty,
            128,
            kParityBlocksize,
            penalty,
            "current contains missing exact needle payload ValueError parity")) {
      return 1;
    }

    if (!check_single_cpu_gpu_exact(
            make_builtin_program({
                                     gagp::payload::make_int_list_value({Value::from_int(1)}),
                                     gagp::payload::make_float_list_value({Value::from_float(2.0)}),
                                 },
                                 gagp::BuiltinId::Concat,
                                 2),
            gagp::payload::make_int_list_value({Value::from_int(1), Value::from_int(2)}),
            -penalty,
            128,
            kParityBlocksize,
            penalty,
            "current concat mixed numeric-list TypeError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({
                                     gagp::payload::make_float_list_value({Value::from_float(1.0)}),
                                     Value::from_int(2),
                                 },
                                 gagp::BuiltinId::Append,
                                 2),
            gagp::payload::make_float_list_value({Value::from_float(1.0), Value::from_float(2.0)}),
            -penalty,
            128,
            kParityBlocksize,
            penalty,
            "current append int into FloatList TypeError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({
                                     gagp::payload::make_int_list_value({Value::from_int(1)}),
                                     Value::from_float(2.0),
                                 },
                                 gagp::BuiltinId::Prepend,
                                 2),
            gagp::payload::make_int_list_value({Value::from_int(2), Value::from_int(1)}),
            -penalty,
            128,
            kParityBlocksize,
            penalty,
            "current prepend float into IntList TypeError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({
                                     gagp::payload::make_string_value("ab"),
                                     Value::from_char('c'),
                                 },
                                 gagp::BuiltinId::Append,
                                 2),
            gagp::payload::make_string_value("abc"),
            -penalty,
            128,
            kParityBlocksize,
            penalty,
            "current append char into String TypeError parity")) {
      return 1;
    }

    const Value int_list_a = Value::from_int_list_hash_len(0x1111ULL, 2);
    const Value int_list_b = Value::from_int_list_hash_len(0x2222ULL, 3);
    const Value int_concat_expected = Value::from_int_list_hash_len(
        Value::combine_container_hash48(2U, int_list_a, int_list_b),
        Value::saturating_len_add(Value::container_len(int_list_a), Value::container_len(int_list_b)));
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({int_list_a, int_list_b}, gagp::BuiltinId::Concat, 2),
            int_concat_expected,
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current int_list concat fallback token parity")) {
      return 1;
    }

    const Value float_list = Value::from_float_list_hash_len(0x3333ULL, 8);
    const Value float_slice_expected = Value::from_float_list_hash_len(
        Value::slice_container_hash48(3U, float_list, -5, -1),
        4);
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({float_list, Value::from_int(-5), Value::from_int(-1)},
                                 gagp::BuiltinId::Slice,
                                 3),
            float_slice_expected,
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current float_list slice fallback token parity")) {
      return 1;
    }

    const Value string_list = Value::from_string_list_hash_len(0x4444ULL, 2);
    const Value raw_string = Value::from_string_hash_len(0x5555ULL, 3);
    const Value append_expected = Value::from_string_list_hash_len(
        Value::append_list_hash48(4U, string_list, raw_string),
        Value::saturating_len_add(Value::container_len(string_list), 1U));
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({string_list, raw_string}, gagp::BuiltinId::Append, 2),
            append_expected,
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current string_list append fallback token parity")) {
      return 1;
    }

    const Value prepend_src = Value::from_int_list_hash_len(0x6666ULL, 4);
    const Value prepend_elem = Value::from_int(9);
    const Value prepend_expected = Value::from_int_list_hash_len(
        prepend_list_hash48(2U, prepend_elem, prepend_src),
        Value::saturating_len_add(Value::container_len(prepend_src), 1U));
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({prepend_src, prepend_elem}, gagp::BuiltinId::Prepend, 2),
            prepend_expected,
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current int_list prepend fallback token parity")) {
      return 1;
    }

    const Value reverse_src = Value::from_int_list_hash_len(0x7777ULL, 5);
    const Value reverse_expected = Value::from_int_list_hash_len(
        Value::reverse_container_hash48(2U, reverse_src),
        Value::container_len(reverse_src));
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({reverse_src}, gagp::BuiltinId::Reverse, 1),
            reverse_expected,
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current int_list reverse fallback token parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({
                                     gagp::payload::make_float_list_value({
                                         Value::from_float(1.0),
                                         Value::from_float(2.5),
                                         Value::from_float(-3.0),
                                     }),
                                 },
                                 gagp::BuiltinId::Reverse,
                                 1),
            gagp::payload::make_float_list_value({
                Value::from_float(-3.0),
                Value::from_float(2.5),
                Value::from_float(1.0),
            }),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current FloatList reverse exact payload parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(
            make_builtin_program({
                                     gagp::payload::make_string_list_value({
                                         gagp::payload::make_string_value("aa"),
                                         gagp::payload::make_string_value("bb"),
                                     }),
                                 },
                                 gagp::BuiltinId::Reverse,
                                 1),
            gagp::payload::make_string_list_value({
                gagp::payload::make_string_value("bb"),
                gagp::payload::make_string_value("aa"),
            }),
            1.0,
            128,
            kParityBlocksize,
            penalty,
            "current StringList reverse exact payload parity")) {
      return 1;
    }
  }

  {
    gagp::payload::clear();
    std::vector<BytecodeProgram> programs;
    programs.push_back(make_nested_exact_string_payload_program());
    programs.push_back(make_return_const_program(7));

    std::vector<CaseBindings> shared_cases;
    std::vector<Value> shared_answer;
    shared_cases.reserve(1024);
    shared_answer.reserve(1024);
    for (int i = 0; i < 1024; ++i) {
      const long long x = static_cast<long long>(i) - 512LL;
      shared_cases.push_back(CaseBindings{InputBinding{0, Value::from_int(x)}});
      shared_answer.push_back(Value::from_int(x + 1));
    }

    const std::vector<double> cpu_fit =
        gagp::eval_fitness_cpu(programs, shared_cases, shared_answer, 256, penalty, 256);
    const gagp::FitnessEvalResult gpu_fit =
        eval_gpu_via_session(programs, shared_cases, shared_answer, 256, 256, penalty);

    if (!gpu_fit.ok) {
      if (gpu_fit.err.message.find("cuda device unavailable") != std::string::npos) {
        std::cout << "gagp_test_fitness_cpu_gpu_parity: SKIP (" << gpu_fit.err.message << ")\n";
        return 0;
      }
      std::cerr << "FAIL: gpu fitness run failed on nested exact payload case: "
                << gpu_fit.err.message << "\n";
      return 1;
    }

    if (cpu_fit.size() != gpu_fit.fitness.size()) {
      std::cerr << "FAIL: cpu/gpu fitness size mismatch on nested exact payload case\n";
      return 1;
    }
    for (std::size_t i = 0; i < cpu_fit.size(); ++i) {
      if (!exact(cpu_fit[i], gpu_fit.fitness[i])) {
        std::cerr << "FAIL: cpu/gpu fitness mismatch on nested exact payload case at " << i << "\n";
        return 1;
      }
    }
  }

  {
    gagp::payload::clear();
    const Value expected_char = Value::from_char('x');
    std::vector<CaseBindings> shared_cases(8);
    std::vector<Value> shared_answer(8, expected_char);

    gagp::FitnessSessionGpu session;
    const gagp::FitnessSessionInitResult init = session.init(shared_cases, shared_answer, 64, kParityBlocksize, penalty);
    if (!init.ok) {
      if (init.err.message.find("cuda device unavailable") != std::string::npos) {
        std::cout << "gagp_test_fitness_cpu_gpu_parity: SKIP (" << init.err.message << ")\n";
        return 0;
      }
      std::cerr << "FAIL: gpu session init failed on late payload case: " << init.err.message << "\n";
      return 1;
    }

    const Value late_string = gagp::payload::make_string_value("xy");
    std::vector<BytecodeProgram> programs;
    programs.push_back(make_exact_string_index_program(late_string, 0));

    const std::vector<double> cpu_fit =
        gagp::eval_fitness_cpu(programs, shared_cases, shared_answer, 64, penalty, kParityBlocksize);
    const gagp::FitnessEvalResult gpu_fit = session.eval_programs(programs);

    if (!gpu_fit.ok) {
      std::cerr << "FAIL: gpu fitness run failed on late payload case: " << gpu_fit.err.message << "\n";
      return 1;
    }
    if (cpu_fit.size() != gpu_fit.fitness.size()) {
      std::cerr << "FAIL: cpu/gpu fitness size mismatch on late payload case\n";
      return 1;
    }
    for (std::size_t i = 0; i < cpu_fit.size(); ++i) {
      if (!exact(cpu_fit[i], gpu_fit.fitness[i])) {
        std::cerr << "FAIL: cpu/gpu fitness mismatch on late payload case at " << i << "\n";
        return 1;
      }
    }
    if (!approx(cpu_fit[0], 1.0 * static_cast<double>(shared_cases.size()))) {
      std::cerr << "FAIL: late payload exact string index should score exact char-match fitness\n";
      return 1;
    }
  }

  {
    gagp::payload::clear();
    std::vector<BytecodeProgram> programs;
    programs.push_back(make_repeated_list_append_len_program());

    std::vector<CaseBindings> shared_cases(1);
    std::vector<Value> shared_answer(1, Value::from_int(5));

    const std::vector<double> cpu_fit =
        gagp::eval_fitness_cpu(programs, shared_cases, shared_answer, 512, penalty, kParityBlocksize);
    const gagp::FitnessEvalResult gpu_fit =
        eval_gpu_via_session(programs, shared_cases, shared_answer, 512, kParityBlocksize, penalty);

    if (!gpu_fit.ok) {
      if (gpu_fit.err.message.find("cuda device unavailable") != std::string::npos) {
        std::cout << "gagp_test_fitness_cpu_gpu_parity: SKIP (" << gpu_fit.err.message << ")\n";
        return 0;
      }
      std::cerr << "FAIL: gpu fitness run failed on repeated append-len case: "
                << gpu_fit.err.message << "\n";
      return 1;
    }
    if (cpu_fit.size() != gpu_fit.fitness.size()) {
      std::cerr << "FAIL: cpu/gpu fitness size mismatch on repeated append-len case\n";
      return 1;
    }
    if (!approx(cpu_fit[0], 0.0)) {
      std::cerr << "FAIL: CPU repeated append-len should preserve host typed-list length semantics"
                << " cpu=" << cpu_fit[0] << "\n";
      return 1;
    }
    if (!approx(gpu_fit.fitness[0], cpu_fit[0])) {
      std::cerr << "FAIL: GPU repeated append-len should match CPU direct-list semantics"
                << " cpu=" << cpu_fit[0] << " gpu=" << gpu_fit.fitness[0] << "\n";
      return 1;
    }
  }

  {
    gagp::payload::clear();
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(gagp::evo::NodeKind::CALL_CHR,
                                                                       {Value::from_int(65)}),
                                    Value::from_char('A'),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_CHR int CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(gagp::evo::NodeKind::CALL_ORD,
                                                                       {Value::from_char('A')}),
                                    Value::from_int(65),
                                    0.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_ORD char CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(gagp::evo::NodeKind::CALL_STRING_TO_CHAR,
                                                                       {gagp::payload::make_string_value("z")}),
                                    Value::from_char('z'),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_STRING_TO_CHAR exact CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(gagp::evo::NodeKind::CALL_CHAR_TO_STRING,
                                                                       {Value::from_char('q')}),
                                    gagp::payload::make_string_value("q"),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_CHAR_TO_STRING exact CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(gagp::evo::NodeKind::CALL_TO_UPPER,
                                                                       {Value::from_char('m')}),
                                    Value::from_char('M'),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_TO_UPPER char CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(gagp::evo::NodeKind::CALL_TO_LOWER,
                                                                       {Value::from_char('Z')}),
                                    Value::from_char('z'),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_TO_LOWER char CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(gagp::evo::NodeKind::CALL_IS_LETTER,
                                                                       {Value::from_char('Q')}),
                                    Value::from_bool(true),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_IS_LETTER char CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(gagp::evo::NodeKind::CALL_IS_DIGIT,
                                                                       {Value::from_char('7')}),
                                    Value::from_bool(true),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_IS_DIGIT char CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(gagp::evo::NodeKind::CALL_IS_SPACE,
                                                                       {Value::from_char(' ')}),
                                    Value::from_bool(true),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_IS_SPACE char CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(gagp::evo::NodeKind::CALL_IS_VOWEL,
                                                                       {Value::from_char('E')}),
                                    Value::from_bool(true),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_IS_VOWEL char CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(gagp::evo::NodeKind::CALL_TO_STRING,
                                                                       {Value::from_int(123)}),
                                    gagp::payload::make_string_value("123"),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_TO_STRING int CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(gagp::evo::NodeKind::CALL_STRING_TO_CHAR,
                                                                       {gagp::payload::make_string_value("xy")}),
                                    Value::from_char(0),
                                    -penalty,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_STRING_TO_CHAR length ValueError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(gagp::evo::NodeKind::CALL_LEN,
                                                                       {gagp::payload::make_string_value("abc")}),
                                    Value::from_int(3),
                                    0.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_LEN string CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(gagp::evo::NodeKind::CALL_INDEX,
                                                                       {gagp::payload::make_string_value("abc"),
                                                                        Value::from_int(1)}),
                                    Value::from_char('b'),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_INDEX string CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(gagp::evo::NodeKind::CALL_CONCAT,
                                                                       {gagp::payload::make_string_value("ab"),
                                                                        gagp::payload::make_string_value("cd")}),
                                    gagp::payload::make_string_value("abcd"),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_CONCAT string CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(gagp::evo::NodeKind::CALL_SLICE,
                                                                       {gagp::payload::make_string_value("abcd"),
                                                                        Value::from_int(1),
                                                                        Value::from_int(3)}),
                                    gagp::payload::make_string_value("bc"),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_SLICE string CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(
                                        gagp::evo::NodeKind::CALL_APPEND,
                                        {gagp::payload::make_int_list_value({
                                             Value::from_int(1),
                                             Value::from_int(2),
                                         }),
                                         Value::from_int(3)}),
                                    gagp::payload::make_int_list_value({
                                        Value::from_int(1),
                                        Value::from_int(2),
                                        Value::from_int(3),
                                    }),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_APPEND IntList CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(
                                        gagp::evo::NodeKind::CALL_PREPEND,
                                        {gagp::payload::make_int_list_value({
                                             Value::from_int(2),
                                             Value::from_int(3),
                                         }),
                                         Value::from_int(1)}),
                                    gagp::payload::make_int_list_value({
                                        Value::from_int(1),
                                        Value::from_int(2),
                                        Value::from_int(3),
                                    }),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_PREPEND IntList CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(gagp::evo::NodeKind::CALL_SINGLETON,
                                                                       {Value::from_int(42)}),
                                    gagp::payload::make_int_list_value({Value::from_int(42)}),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_SINGLETON Int CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(
                                        gagp::evo::NodeKind::CALL_REVERSE,
                                        {gagp::payload::make_string_list_value({
                                            gagp::payload::make_string_value("aa"),
                                            gagp::payload::make_string_value("bb"),
                                        })}),
                                    gagp::payload::make_string_list_value({
                                        gagp::payload::make_string_value("bb"),
                                        gagp::payload::make_string_value("aa"),
                                    }),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_REVERSE StringList CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(gagp::evo::NodeKind::CALL_FIND,
                                                                       {gagp::payload::make_string_value("abracadabra"),
                                                                        gagp::payload::make_string_value("cad")}),
                                    Value::from_int(4),
                                    0.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_FIND string CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(gagp::evo::NodeKind::CALL_CONTAINS,
                                                                       {gagp::payload::make_string_value("abracadabra"),
                                                                        gagp::payload::make_string_value("xyz")}),
                                    Value::from_bool(false),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_CONTAINS string CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(gagp::evo::NodeKind::CALL_LEN,
                                                                       {Value::from_int(7)}),
                                    Value::from_int(0),
                                    -penalty,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_LEN non-container TypeError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(gagp::evo::NodeKind::CALL_ABS,
                                                                       {Value::from_int(-5)}),
                                    Value::from_int(5),
                                    0.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_ABS int CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(gagp::evo::NodeKind::CALL_MIN,
                                                                       {Value::from_int(9), Value::from_int(4)}),
                                    Value::from_int(4),
                                    0.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_MIN int CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(gagp::evo::NodeKind::CALL_MAX,
                                                                       {Value::from_int(9), Value::from_int(4)}),
                                    Value::from_int(9),
                                    0.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_MAX int CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(gagp::evo::NodeKind::CALL_CLIP,
                                                                       {Value::from_int(12),
                                                                        Value::from_int(0),
                                                                        Value::from_int(10)}),
                                    Value::from_int(10),
                                    0.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_CLIP int CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(gagp::evo::NodeKind::CALL_IDIV0,
                                                                       {Value::from_int(-7), Value::from_int(2)}),
                                    Value::from_int(-3),
                                    0.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_IDIV0 negative CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(gagp::evo::NodeKind::CALL_IMOD0,
                                                                       {Value::from_int(7), Value::from_int(-2)}),
                                    Value::from_int(-1),
                                    0.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_IMOD0 negative divisor CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_builtin_call_program(gagp::evo::NodeKind::CALL_ABS,
                                                                       {Value::from_bool(true)}),
                                    Value::from_int(0),
                                    -penalty,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled CALL_ABS non-numeric TypeError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_binary_expr_program(gagp::evo::NodeKind::ADD,
                                                                      Value::from_int(2),
                                                                      Value::from_int(3)),
                                    Value::from_int(5),
                                    0.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled ADD int CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_binary_expr_program(gagp::evo::NodeKind::SUB,
                                                                      Value::from_int(9),
                                                                      Value::from_int(4)),
                                    Value::from_int(5),
                                    0.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled SUB int CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_binary_expr_program(gagp::evo::NodeKind::MUL,
                                                                      Value::from_int(6),
                                                                      Value::from_int(7)),
                                    Value::from_int(42),
                                    0.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled MUL int CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_binary_expr_program(gagp::evo::NodeKind::DIV,
                                                                      Value::from_int(7),
                                                                      Value::from_int(2)),
                                    Value::from_float(3.5),
                                    0.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled DIV int-to-float CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_binary_expr_program(gagp::evo::NodeKind::MOD,
                                                                      Value::from_int(7),
                                                                      Value::from_int(3)),
                                    Value::from_int(1),
                                    0.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled MOD int CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_binary_expr_program(gagp::evo::NodeKind::LT,
                                                                      Value::from_int(2),
                                                                      Value::from_int(3)),
                                    Value::from_bool(true),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled LT int CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_binary_expr_program(gagp::evo::NodeKind::LE,
                                                                      Value::from_int(3),
                                                                      Value::from_int(3)),
                                    Value::from_bool(true),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled LE int CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_binary_expr_program(gagp::evo::NodeKind::GT,
                                                                      Value::from_int(5),
                                                                      Value::from_int(2)),
                                    Value::from_bool(true),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled GT int CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_binary_expr_program(gagp::evo::NodeKind::GE,
                                                                      Value::from_int(5),
                                                                      Value::from_int(5)),
                                    Value::from_bool(true),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled GE int CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_binary_expr_program(gagp::evo::NodeKind::EQ,
                                                                      Value::from_int(4),
                                                                      Value::from_int(4)),
                                    Value::from_bool(true),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled EQ int CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_binary_expr_program(gagp::evo::NodeKind::NE,
                                                                      Value::from_int(4),
                                                                      Value::from_int(5)),
                                    Value::from_bool(true),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled NE int CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_binary_expr_program(gagp::evo::NodeKind::ADD,
                                                                      Value::from_int(2),
                                                                      Value::from_bool(true)),
                                    Value::from_int(0),
                                    -penalty,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled ADD non-numeric TypeError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_binary_expr_program(gagp::evo::NodeKind::DIV,
                                                                      Value::from_int(7),
                                                                      Value::from_int(0)),
                                    Value::from_float(0.0),
                                    -penalty,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled DIV zero TypeError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_binary_expr_program(gagp::evo::NodeKind::LT,
                                                                      Value::from_bool(false),
                                                                      Value::from_bool(true)),
                                    Value::from_bool(false),
                                    -penalty,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled LT bool ordering TypeError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_unary_program(gagp::evo::NodeKind::NOT,
                                                                Value::from_bool(true)),
                                    Value::from_bool(false),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled NOT bool CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_unary_program(gagp::evo::NodeKind::NOT,
                                                                Value::from_int(1)),
                                    Value::from_bool(false),
                                    -penalty,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled NOT non-bool TypeError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_unary_program(gagp::evo::NodeKind::NEG,
                                                                Value::from_int(7)),
                                    Value::from_int(-7),
                                    0.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled NEG int CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_unary_program(gagp::evo::NodeKind::NEG,
                                                                Value::from_bool(true)),
                                    Value::from_int(0),
                                    -penalty,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled NEG non-numeric TypeError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_bool_binary_program(gagp::evo::NodeKind::AND,
                                                                      Value::from_bool(false),
                                                                      Value::from_int(1)),
                                    Value::from_bool(false),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled AND short-circuit false CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_bool_binary_program(gagp::evo::NodeKind::AND,
                                                                      Value::from_bool(true),
                                                                      Value::from_int(1)),
                                    Value::from_bool(false),
                                    -penalty,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled AND non-bool rhs TypeError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_bool_binary_program(gagp::evo::NodeKind::OR,
                                                                      Value::from_bool(true),
                                                                      Value::from_int(1)),
                                    Value::from_bool(true),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled OR short-circuit true CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_bool_binary_program(gagp::evo::NodeKind::OR,
                                                                      Value::from_bool(false),
                                                                      Value::from_int(1)),
                                    Value::from_bool(true),
                                    -penalty,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled OR non-bool rhs TypeError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_bool_binary_program(gagp::evo::NodeKind::AND,
                                                                      Value::from_int(1),
                                                                      Value::from_bool(true)),
                                    Value::from_bool(false),
                                    -penalty,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled AND non-bool lhs TypeError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_bool_binary_program(gagp::evo::NodeKind::OR,
                                                                      Value::from_int(1),
                                                                      Value::from_bool(false)),
                                    Value::from_bool(true),
                                    -penalty,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled OR non-bool lhs TypeError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_if_expr_program(Value::from_bool(true)),
                                    Value::from_int(7),
                                    0.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled IF_EXPR true branch CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_if_expr_program(Value::from_bool(false)),
                                    Value::from_int(11),
                                    0.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled IF_EXPR false branch CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_if_expr_program(Value::from_int(1)),
                                    Value::from_int(0),
                                    -penalty,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled IF_EXPR non-bool condition TypeError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_if_stmt_branch_program(Value::from_bool(true)),
                                    Value::from_int(7),
                                    0.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled IF_STMT true branch CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_if_stmt_branch_program(Value::from_bool(false)),
                                    Value::from_int(11),
                                    0.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled IF_STMT false branch CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_if_stmt_branch_program(Value::from_int(1)),
                                    Value::from_int(0),
                                    -penalty,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled IF_STMT non-bool condition TypeError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_for_range_sum_program(),
                                    Value::from_int(6),
                                    0.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled FOR_RANGE len-bound sum CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_for_range_sum_program_with_direct_bound(Value::from_int(0)),
                                    Value::from_int(0),
                                    0.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled FOR_RANGE zero bound CPU/GPU parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_for_range_sum_program_with_direct_bound(Value::from_int(-1)),
                                    Value::from_int(0),
                                    -penalty,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled FOR_RANGE negative bound TypeError parity")) {
      return 1;
    }
    if (!check_single_cpu_gpu_exact(make_compiled_for_range_sum_program_with_direct_bound(Value::from_bool(true)),
                                    Value::from_int(0),
                                    -penalty,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled FOR_RANGE bool bound TypeError parity")) {
      return 1;
    }
  }

  {
    gagp::payload::clear();
    std::vector<BytecodeProgram> programs;
    programs.push_back(make_compiled_map_list_program());
    programs.push_back(make_compiled_filter_list_program());
    programs.push_back(make_compiled_linear_rec_program());

    std::vector<CaseBindings> shared_cases(4);
    std::vector<Value> shared_answer = {
        gagp::payload::make_int_list_value({Value::from_int(2), Value::from_int(4), Value::from_int(6)}),
        gagp::payload::make_int_list_value({Value::from_int(3), Value::from_int(4)}),
        Value::from_int(30621),
        Value::from_int(30621),
    };
    shared_cases.resize(shared_answer.size());

    const std::vector<double> cpu_fit =
        gagp::eval_fitness_cpu(programs, shared_cases, shared_answer, 20000, penalty, kParityBlocksize);
    const gagp::FitnessEvalResult gpu_fit =
        eval_gpu_via_session(programs, shared_cases, shared_answer, 20000, kParityBlocksize, penalty);

    if (!gpu_fit.ok) {
      if (gpu_fit.err.message.find("cuda device unavailable") != std::string::npos) {
        std::cout << "gagp_test_fitness_cpu_gpu_parity: SKIP (" << gpu_fit.err.message << ")\n";
        return 0;
      }
      std::cerr << "FAIL: gpu fitness run failed on compiled structured-expression case: "
                << gpu_fit.err.message << "\n";
      return 1;
    }
    if (cpu_fit.size() != gpu_fit.fitness.size()) {
      std::cerr << "FAIL: cpu/gpu fitness size mismatch on compiled structured-expression case\n";
      return 1;
    }
    for (std::size_t i = 0; i < cpu_fit.size(); ++i) {
      if (!exact(cpu_fit[i], gpu_fit.fitness[i])) {
        std::cerr << "FAIL: cpu/gpu fitness mismatch on compiled structured-expression case at " << i
                  << " cpu=" << cpu_fit[i] << " gpu=" << gpu_fit.fitness[i] << "\n";
        return 1;
      }
    }
    if (!approx(cpu_fit[0], 1.0 - 2.0 * penalty)) {
      std::cerr << "FAIL: compiled MAP_LIST should match only its exact-list answer cases\n";
      return 1;
    }
    if (!approx(cpu_fit[1], 1.0 - 2.0 * penalty)) {
      std::cerr << "FAIL: compiled FILTER_LIST should match only its exact-list answer cases\n";
      return 1;
    }
    if (!approx(cpu_fit[2], -2.0 * penalty)) {
      std::cerr << "FAIL: compiled LINEAR_REC should match only its numeric answer cases\n";
      return 1;
    }
  }

  {
    gagp::payload::clear();
    if (!check_single_cpu_gpu_exact(
            make_compiled_float_map_list_program(),
            gagp::payload::make_float_list_value({Value::from_float(2.0), Value::from_float(5.0)}),
            1.0,
            20000,
            kParityBlocksize,
            penalty,
            "compiled MAP_LIST FloatList CPU/GPU parity")) {
      return 1;
    }

    const Value aa = gagp::payload::make_string_value("aa");
    const Value ccc = gagp::payload::make_string_value("ccc");
    if (!check_single_cpu_gpu_exact(
            make_compiled_string_filter_list_program(),
            gagp::payload::make_string_list_value({aa, ccc}),
            1.0,
            20000,
            kParityBlocksize,
            penalty,
            "compiled FILTER_LIST StringList CPU/GPU parity")) {
      return 1;
    }

    if (!check_single_cpu_gpu_exact(make_compiled_empty_string_map_list_program(),
                                    gagp::payload::make_string_list_value({}),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled MAP_LIST empty StringList CPU/GPU parity")) {
      return 1;
    }

    if (!check_single_cpu_gpu_exact(make_compiled_float_linear_rec_program(),
                                    Value::from_float(4.0),
                                    0.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled LINEAR_REC FloatList CPU/GPU parity")) {
      return 1;
    }

    if (!check_single_cpu_gpu_exact(make_compiled_string_linear_rec_program(),
                                    gagp::payload::make_string_value("abc"),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "compiled LINEAR_REC StringList CPU/GPU parity")) {
      return 1;
    }
  }

  {
    gagp::payload::clear();
    if (!check_single_cpu_gpu_exact(make_asgp_dc_string_concat_program(),
                                    gagp::payload::make_string_value("abc"),
                                    1.0,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "ASGP-DC string concat CPU/GPU parity")) {
      return 1;
    }
  }

  {
    gagp::payload::clear();
    if (!check_single_cpu_gpu_exact(make_compiled_asgp_dc_hidden_local_program(),
                                    Value::from_int(7),
                                    -penalty,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "ASGP-DC phase ordinary local isolation CPU/GPU parity")) {
      return 1;
    }
  }

  {
    gagp::payload::clear();
    if (!check_single_cpu_gpu_exact(make_compiled_asgp_dc_divide_hidden_local_program(),
                                    Value::from_int(7),
                                    -penalty,
                                    20000,
                                    kParityBlocksize,
                                    penalty,
                                    "ASGP-DC divide ordinary local isolation CPU/GPU parity")) {
      return 1;
    }
  }

  {
    gagp::payload::clear();
    std::vector<BytecodeProgram> programs;
    programs.push_back(make_asgp_dc_sum_program());
    programs.push_back(make_compiled_asgp_dc_program(false));
    programs.push_back(make_compiled_asgp_dc_program(true));

    std::vector<CaseBindings> shared_cases(8);
    std::vector<Value> shared_answer = {
        Value::from_int(10),
        Value::from_int(6),
        Value::from_float(6.0),
        Value::from_int(10),
        Value::from_int(6),
        Value::from_float(6.0),
        Value::from_int(10),
        Value::from_int(6),
        Value::from_float(6.0),
    };
    shared_cases.resize(shared_answer.size());

    const std::vector<double> cpu_fit =
        gagp::eval_fitness_cpu(programs, shared_cases, shared_answer, 20000, penalty, kParityBlocksize);
    const gagp::FitnessEvalResult gpu_fit =
        eval_gpu_via_session(programs, shared_cases, shared_answer, 20000, kParityBlocksize, penalty);

    if (!gpu_fit.ok) {
      if (gpu_fit.err.message.find("cuda device unavailable") != std::string::npos) {
        std::cout << "gagp_test_fitness_cpu_gpu_parity: SKIP (" << gpu_fit.err.message << ")\n";
        return 0;
      }
      std::cerr << "FAIL: gpu fitness run failed on ASGP-DC case: " << gpu_fit.err.message << "\n";
      return 1;
    }
    if (cpu_fit.size() != gpu_fit.fitness.size()) {
      std::cerr << "FAIL: cpu/gpu fitness size mismatch on ASGP-DC case\n";
      return 1;
    }
    for (std::size_t i = 0; i < cpu_fit.size(); ++i) {
      if (!exact(cpu_fit[i], gpu_fit.fitness[i])) {
        std::cerr << "FAIL: ASGP-DC CPU/GPU fitness mismatch at " << i
                  << " cpu=" << cpu_fit[i] << " gpu=" << gpu_fit.fitness[i] << "\n";
        return 1;
      }
    }
  }

  {
    gagp::payload::clear();
    std::vector<BytecodeProgram> programs;
    programs.push_back(make_asgp_dp1d_backward_sum_program());

    std::vector<CaseBindings> shared_cases(8);
    std::vector<Value> shared_answer(8, Value::from_int(11));

    const std::vector<double> cpu_fit =
        gagp::eval_fitness_cpu(programs, shared_cases, shared_answer, 20000, penalty, kParityBlocksize);
    const gagp::FitnessEvalResult gpu_fit =
        eval_gpu_via_session(programs, shared_cases, shared_answer, 20000, kParityBlocksize, penalty);

    if (!gpu_fit.ok) {
      if (gpu_fit.err.message.find("cuda device unavailable") != std::string::npos) {
        std::cout << "gagp_test_fitness_cpu_gpu_parity: SKIP (" << gpu_fit.err.message << ")\n";
        return 0;
      }
      std::cerr << "FAIL: gpu fitness run failed on ASGP-DP1D backward case: "
                << gpu_fit.err.message << "\n";
      return 1;
    }
    if (cpu_fit.size() != gpu_fit.fitness.size()) {
      std::cerr << "FAIL: cpu/gpu fitness size mismatch on ASGP-DP1D backward case\n";
      return 1;
    }
    if (!exact(cpu_fit[0], gpu_fit.fitness[0]) || !approx(cpu_fit[0], 0.0)) {
      std::cerr << "FAIL: ASGP-DP1D backward CPU/GPU fitness mismatch"
                << " cpu=" << cpu_fit[0] << " gpu=" << gpu_fit.fitness[0] << "\n";
      return 1;
    }
  }

  {
    gagp::payload::clear();
    std::vector<BytecodeProgram> programs;
    programs.push_back(make_asgp_dp1d_fib_program());

    std::vector<CaseBindings> shared_cases(8);
    std::vector<Value> shared_answer(8, Value::from_int(34));

    const std::vector<double> cpu_fit =
        gagp::eval_fitness_cpu(programs, shared_cases, shared_answer, 500, penalty, kParityBlocksize);
    const gagp::FitnessEvalResult gpu_fit =
        eval_gpu_via_session(programs, shared_cases, shared_answer, 500, kParityBlocksize, penalty);

    if (!gpu_fit.ok) {
      if (gpu_fit.err.message.find("cuda device unavailable") != std::string::npos) {
        std::cout << "gagp_test_fitness_cpu_gpu_parity: SKIP (" << gpu_fit.err.message << ")\n";
        return 0;
      }
      std::cerr << "FAIL: gpu fitness run failed on ASGP-DP1D memo case: "
                << gpu_fit.err.message << "\n";
      return 1;
    }
    if (cpu_fit.size() != gpu_fit.fitness.size()) {
      std::cerr << "FAIL: cpu/gpu fitness size mismatch on ASGP-DP1D memo case\n";
      return 1;
    }
    if (!exact(cpu_fit[0], gpu_fit.fitness[0]) || !approx(cpu_fit[0], 0.0)) {
      std::cerr << "FAIL: ASGP-DP1D memo CPU/GPU fitness mismatch"
                << " cpu=" << cpu_fit[0] << " gpu=" << gpu_fit.fitness[0] << "\n";
      return 1;
    }
  }

  {
    gagp::payload::clear();
    if (!check_single_cpu_gpu_exact(make_asgp_dp1d_string_concat_program(),
                                    gagp::payload::make_string_value("aaaaa"),
                                    1.0,
                                    500,
                                    kParityBlocksize,
                                    penalty,
                                    "ASGP-DP1D string concat CPU/GPU parity")) {
      return 1;
    }
  }

  {
    gagp::payload::clear();
    if (!check_single_cpu_gpu_exact(make_asgp_dp1d_boundary_program(),
                                    Value::from_int(123),
                                    0.0,
                                    500,
                                    kParityBlocksize,
                                    penalty,
                                    "ASGP-DP1D out-of-bounds boundary CPU/GPU parity")) {
      return 1;
    }
  }

  {
    gagp::payload::clear();
    if (!check_single_cpu_gpu_exact(make_asgp_dp1d_dependency_type_error_program(),
                                    Value::from_int(0),
                                    -penalty,
                                    500,
                                    kParityBlocksize,
                                    penalty,
                                    "ASGP-DP1D dependency type-error CPU/GPU parity")) {
      return 1;
    }
  }

  {
    gagp::payload::clear();
    if (!check_single_cpu_gpu_exact(make_asgp_dp1d_transition_type_error_program(),
                                    Value::from_int(0),
                                    -penalty,
                                    500,
                                    kParityBlocksize,
                                    penalty,
                                    "ASGP-DP1D transition type-error CPU/GPU parity")) {
      return 1;
    }
  }

  {
    gagp::payload::clear();
    if (!check_single_cpu_gpu_exact(make_asgp_dp1d_state_type_error_program(),
                                    Value::from_int(0),
                                    -penalty,
                                    500,
                                    kParityBlocksize,
                                    penalty,
                                    "ASGP-DP1D state type-error CPU/GPU parity")) {
      return 1;
    }
  }

  {
    gagp::payload::clear();
    if (!check_single_cpu_gpu_exact(make_asgp_dp1d_fib_program(),
                                    Value::from_int(0),
                                    -penalty,
                                    2,
                                    kParityBlocksize,
                                    penalty,
                                    "ASGP-DP1D timeout CPU/GPU parity")) {
      return 1;
    }
  }

  {
    gagp::payload::clear();
    if (!check_single_cpu_gpu_exact(make_compiled_asgp_dp1d_transition_hidden_local_program(),
                                    Value::from_int(7),
                                    -penalty,
                                    500,
                                    kParityBlocksize,
                                    penalty,
                                    "ASGP-DP1D transition ordinary local isolation CPU/GPU parity")) {
      return 1;
    }
  }

  {
    gagp::payload::clear();
    if (!check_single_cpu_gpu_exact(make_compiled_asgp_dp1d_solve_hidden_local_program(),
                                    Value::from_int(7),
                                    -penalty,
                                    500,
                                    kParityBlocksize,
                                    penalty,
                                    "ASGP-DP1D solve ordinary local isolation CPU/GPU parity")) {
      return 1;
    }
  }

  {
    gagp::payload::clear();
    std::vector<BytecodeProgram> programs;
    programs.push_back(make_asgp_dp2d_grid_program());

    std::vector<CaseBindings> shared_cases(8);
    std::vector<Value> shared_answer(8, Value::from_int(6));

    const std::vector<double> cpu_fit =
        gagp::eval_fitness_cpu(programs, shared_cases, shared_answer, 500, penalty, kParityBlocksize);
    const gagp::FitnessEvalResult gpu_fit =
        eval_gpu_via_session(programs, shared_cases, shared_answer, 500, kParityBlocksize, penalty);

    if (!gpu_fit.ok) {
      if (gpu_fit.err.message.find("cuda device unavailable") != std::string::npos) {
        std::cout << "gagp_test_fitness_cpu_gpu_parity: SKIP (" << gpu_fit.err.message << ")\n";
        return 0;
      }
      std::cerr << "FAIL: gpu fitness run failed on ASGP-DP2D grid case: "
                << gpu_fit.err.message << "\n";
      return 1;
    }
    if (cpu_fit.size() != gpu_fit.fitness.size()) {
      std::cerr << "FAIL: cpu/gpu fitness size mismatch on ASGP-DP2D grid case\n";
      return 1;
    }
    if (!exact(cpu_fit[0], gpu_fit.fitness[0]) || !approx(cpu_fit[0], 0.0)) {
      std::cerr << "FAIL: ASGP-DP2D grid CPU/GPU fitness mismatch"
                << " cpu=" << cpu_fit[0] << " gpu=" << gpu_fit.fitness[0] << "\n";
      return 1;
    }
  }

  {
    gagp::payload::clear();
    if (!check_single_cpu_gpu_exact(make_asgp_dp2d_dep_kind_program(1, 1, 2, 2, 2),
                                    Value::from_int(1),
                                    0.0,
                                    500,
                                    kParityBlocksize,
                                    penalty,
                                    "ASGP-DP2D dep-kind 1 forward-cross CPU/GPU parity")) {
      return 1;
    }
  }

  {
    gagp::payload::clear();
    if (!check_single_cpu_gpu_exact(make_asgp_dp2d_dep_kind_program(2, 1, 1, 0, 0),
                                    Value::from_int(1),
                                    0.0,
                                    500,
                                    kParityBlocksize,
                                    penalty,
                                    "ASGP-DP2D dep-kind 2 backward-diagonal CPU/GPU parity")) {
      return 1;
    }
  }

  {
    gagp::payload::clear();
    if (!check_single_cpu_gpu_exact(make_asgp_dp2d_dep_kind_program(3, 1, 1, 2, 2),
                                    Value::from_int(1),
                                    0.0,
                                    500,
                                    kParityBlocksize,
                                    penalty,
                                    "ASGP-DP2D dep-kind 3 forward-diagonal CPU/GPU parity")) {
      return 1;
    }
  }

  {
    gagp::payload::clear();
    if (!check_single_cpu_gpu_exact(make_asgp_dp2d_dep_kind_program(4, 1, 1, 0, 0),
                                    Value::from_int(3),
                                    0.0,
                                    500,
                                    kParityBlocksize,
                                    penalty,
                                    "ASGP-DP2D dep-kind 4 backward-neighborhood CPU/GPU parity")) {
      return 1;
    }
  }

  {
    gagp::payload::clear();
    if (!check_single_cpu_gpu_exact(make_asgp_dp2d_dep_kind_program(5, 1, 1, 2, 2),
                                    Value::from_int(3),
                                    0.0,
                                    500,
                                    kParityBlocksize,
                                    penalty,
                                    "ASGP-DP2D dep-kind 5 forward-neighborhood CPU/GPU parity")) {
      return 1;
    }
  }

  {
    gagp::payload::clear();
    if (!check_single_cpu_gpu_exact(make_asgp_dp2d_grid_program(),
                                    Value::from_int(0),
                                    -penalty,
                                    2,
                                    kParityBlocksize,
                                    penalty,
                                    "ASGP-DP2D timeout CPU/GPU parity")) {
      return 1;
    }
  }

  {
    gagp::payload::clear();
    if (!check_single_cpu_gpu_exact(make_asgp_dp2d_string_concat_program(),
                                    gagp::payload::make_string_value("aaaaaa"),
                                    1.0,
                                    500,
                                    kParityBlocksize,
                                    penalty,
                                    "ASGP-DP2D string concat CPU/GPU parity")) {
      return 1;
    }
  }

  {
    gagp::payload::clear();
    if (!check_single_cpu_gpu_exact(make_asgp_dp2d_boundary_program(),
                                    Value::from_int(321),
                                    0.0,
                                    500,
                                    kParityBlocksize,
                                    penalty,
                                    "ASGP-DP2D out-of-bounds boundary CPU/GPU parity")) {
      return 1;
    }
  }

  {
    gagp::payload::clear();
    if (!check_single_cpu_gpu_exact(make_asgp_dp2d_state_type_error_program(true),
                                    Value::from_int(0),
                                    -penalty,
                                    500,
                                    kParityBlocksize,
                                    penalty,
                                    "ASGP-DP2D state-i type-error CPU/GPU parity")) {
      return 1;
    }
  }

  {
    gagp::payload::clear();
    if (!check_single_cpu_gpu_exact(make_asgp_dp2d_state_type_error_program(false),
                                    Value::from_int(0),
                                    -penalty,
                                    500,
                                    kParityBlocksize,
                                    penalty,
                                    "ASGP-DP2D state-j type-error CPU/GPU parity")) {
      return 1;
    }
  }

  {
    gagp::payload::clear();
    if (!check_single_cpu_gpu_exact(make_asgp_dp2d_dependency_type_error_program(),
                                    Value::from_int(0),
                                    -penalty,
                                    500,
                                    kParityBlocksize,
                                    penalty,
                                    "ASGP-DP2D dependency type-error CPU/GPU parity")) {
      return 1;
    }
  }

  {
    gagp::payload::clear();
    if (!check_single_cpu_gpu_exact(make_asgp_dp2d_transition_type_error_program(),
                                    Value::from_int(0),
                                    -penalty,
                                    500,
                                    kParityBlocksize,
                                    penalty,
                                    "ASGP-DP2D transition type-error CPU/GPU parity")) {
      return 1;
    }
  }

  {
    gagp::payload::clear();
    if (!check_single_cpu_gpu_exact(make_compiled_asgp_dp2d_transition_hidden_local_program(),
                                    Value::from_int(7),
                                    -penalty,
                                    500,
                                    kParityBlocksize,
                                    penalty,
                                    "ASGP-DP2D transition ordinary local isolation CPU/GPU parity")) {
      return 1;
    }
  }

  {
    gagp::payload::clear();
    if (!check_single_cpu_gpu_exact(make_compiled_asgp_dp2d_solve_hidden_local_program(),
                                    Value::from_int(7),
                                    -penalty,
                                    500,
                                    kParityBlocksize,
                                    penalty,
                                    "ASGP-DP2D solve ordinary local isolation CPU/GPU parity")) {
      return 1;
    }
  }

  std::cout << "gagp_test_fitness_cpu_gpu_parity: OK\n";
  return 0;
}
