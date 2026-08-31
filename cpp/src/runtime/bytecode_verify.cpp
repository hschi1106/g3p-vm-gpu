#include "g3pvm/core/bytecode_verify.hpp"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "g3pvm/core/builtin.hpp"

namespace g3pvm {

const char* bytecode_verify_code_name(BytecodeVerifyCode code) noexcept {
  switch (code) {
    case BytecodeVerifyCode::Ok: return "ok";
    case BytecodeVerifyCode::InvalidLocalCount: return "invalid_local_count";
    case BytecodeVerifyCode::InvalidConstant: return "invalid_constant";
    case BytecodeVerifyCode::UnknownOpcode: return "unknown_opcode";
    case BytecodeVerifyCode::MissingOperand: return "missing_operand";
    case BytecodeVerifyCode::InvalidConstantIndex: return "invalid_constant_index";
    case BytecodeVerifyCode::InvalidLocalIndex: return "invalid_local_index";
    case BytecodeVerifyCode::InvalidBuiltinId: return "invalid_builtin_id";
    case BytecodeVerifyCode::InvalidBuiltinArity: return "invalid_builtin_arity";
    case BytecodeVerifyCode::InvalidJumpTarget: return "invalid_jump_target";
    case BytecodeVerifyCode::InvalidPrivateOpcode: return "invalid_private_opcode";
    case BytecodeVerifyCode::InvalidSegmentIndex: return "invalid_segment_index";
    case BytecodeVerifyCode::StackUnderflow: return "stack_underflow";
    case BytecodeVerifyCode::StackJoinMismatch: return "stack_join_mismatch";
    case BytecodeVerifyCode::InvalidFallthrough: return "invalid_fallthrough";
    case BytecodeVerifyCode::InvalidVarMapping: return "invalid_var_mapping";
    case BytecodeVerifyCode::InvalidBinderLocal: return "invalid_binder_local";
    case BytecodeVerifyCode::InvalidSegmentMetadata: return "invalid_segment_metadata";
    case BytecodeVerifyCode::ResourceLimit: return "resource_limit";
  }
  return "unknown";
}

namespace {

enum class AbstractType : std::uint8_t {
  Unknown,
  Int,
  Float,
  Bool,
  Char,
  String,
  IntList,
  FloatList,
  StringList,
};

struct AbstractState {
  std::vector<AbstractType> stack;
  std::unordered_map<int, AbstractType> known_locals;
};

struct CodeSummary {
  std::size_t max_stack_depth = 0;
  bool has_reachable_return = false;
};

bool is_public_value_tag(ValueTag tag) {
  return tag == ValueTag::Int || tag == ValueTag::Float || tag == ValueTag::Bool ||
         tag == ValueTag::Char || tag == ValueTag::String || tag == ValueTag::IntList ||
         tag == ValueTag::FloatList || tag == ValueTag::StringList;
}

bool is_valid_char(std::int64_t value) {
  return value >= 0 && value <= 0x10FFFF && !(value >= 0xD800 && value <= 0xDFFF);
}

AbstractType abstract_type(ValueTag tag) {
  switch (tag) {
    case ValueTag::Int: return AbstractType::Int;
    case ValueTag::Float: return AbstractType::Float;
    case ValueTag::Bool: return AbstractType::Bool;
    case ValueTag::Char: return AbstractType::Char;
    case ValueTag::String: return AbstractType::String;
    case ValueTag::IntList: return AbstractType::IntList;
    case ValueTag::FloatList: return AbstractType::FloatList;
    case ValueTag::StringList: return AbstractType::StringList;
    case ValueTag::FallbackToken:
    case ValueTag::Invalid: return AbstractType::Unknown;
  }
  return AbstractType::Unknown;
}

bool is_numeric(AbstractType type) {
  return type == AbstractType::Int || type == AbstractType::Float;
}

bool is_typed_list(AbstractType type) {
  return type == AbstractType::IntList || type == AbstractType::FloatList ||
         type == AbstractType::StringList;
}

bool is_container(AbstractType type) {
  return type == AbstractType::String || is_typed_list(type);
}

bool is_private_opcode(Opcode op) {
  return op == Opcode::CheckList || op == Opcode::CheckInt || op == Opcode::EmptyList ||
         op == Opcode::EmptyListLike || op == Opcode::AsgpDc || op == Opcode::AsgpDp1d ||
         op == Opcode::AsgpDp2d;
}

bool is_asgp_opcode(Opcode op) {
  return op == Opcode::AsgpDc || op == Opcode::AsgpDp1d || op == Opcode::AsgpDp2d;
}

int expected_builtin_arity(BuiltinId id) {
  switch (id) {
    case BuiltinId::Abs:
    case BuiltinId::Len:
    case BuiltinId::Reverse:
    case BuiltinId::IsInt:
    case BuiltinId::CharToString:
    case BuiltinId::StringToChar:
    case BuiltinId::Ord:
    case BuiltinId::Chr:
    case BuiltinId::IsLetter:
    case BuiltinId::IsDigit:
    case BuiltinId::IsSpace:
    case BuiltinId::IsVowel:
    case BuiltinId::ToLower:
    case BuiltinId::ToUpper:
    case BuiltinId::ToString:
    case BuiltinId::Singleton: return 1;
    case BuiltinId::Min:
    case BuiltinId::Max:
    case BuiltinId::Concat:
    case BuiltinId::Index:
    case BuiltinId::Append:
    case BuiltinId::Find:
    case BuiltinId::Contains:
    case BuiltinId::IDiv0:
    case BuiltinId::IMod0:
    case BuiltinId::Prepend: return 2;
    case BuiltinId::Clip:
    case BuiltinId::Slice: return 3;
  }
  return -1;
}

AbstractType builtin_result_type(BuiltinId id, const std::vector<AbstractType>& args) {
  switch (id) {
    case BuiltinId::Abs:
    case BuiltinId::Min:
    case BuiltinId::Max:
    case BuiltinId::Clip:
    case BuiltinId::Concat:
    case BuiltinId::Slice:
    case BuiltinId::Append:
    case BuiltinId::Prepend:
    case BuiltinId::Reverse:
      return args.empty() ? AbstractType::Unknown : args.front();
    case BuiltinId::Len:
    case BuiltinId::Find:
    case BuiltinId::Ord:
    case BuiltinId::IDiv0:
    case BuiltinId::IMod0: return AbstractType::Int;
    case BuiltinId::Contains:
    case BuiltinId::IsInt:
    case BuiltinId::IsLetter:
    case BuiltinId::IsDigit:
    case BuiltinId::IsSpace:
    case BuiltinId::IsVowel: return AbstractType::Bool;
    case BuiltinId::CharToString:
    case BuiltinId::ToString: return AbstractType::String;
    case BuiltinId::StringToChar:
    case BuiltinId::Chr:
    case BuiltinId::ToLower:
    case BuiltinId::ToUpper: return AbstractType::Char;
    case BuiltinId::Index:
      if (args.empty()) return AbstractType::Unknown;
      if (args.front() == AbstractType::String) return AbstractType::Char;
      if (args.front() == AbstractType::IntList) return AbstractType::Int;
      if (args.front() == AbstractType::FloatList) return AbstractType::Float;
      if (args.front() == AbstractType::StringList) return AbstractType::String;
      return AbstractType::Unknown;
    case BuiltinId::Singleton:
      if (args.empty()) return AbstractType::Unknown;
      if (args.front() == AbstractType::Char) return AbstractType::String;
      if (args.front() == AbstractType::Int) return AbstractType::IntList;
      if (args.front() == AbstractType::Float) return AbstractType::FloatList;
      if (args.front() == AbstractType::String) return AbstractType::StringList;
      return AbstractType::Unknown;
  }
  return AbstractType::Unknown;
}

class Verifier {
 public:
  Verifier(const BytecodeProgram& program, const BytecodeVerifyOptions& options)
      : program_(program), options_(options) {}

  BytecodeVerifyResult run() {
    if (!check_segment_limit()) return result_;
    CodeSummary main_summary;
    if (!verify_code(program_.consts, program_.code, program_.n_locals, program_.var2idx,
                     {}, "$.code", true, true, &main_summary)) {
      return result_;
    }
    result_.verified.max_stack_depth = main_summary.max_stack_depth;
    result_.verified.has_reachable_return = main_summary.has_reachable_return;
    if (!verify_segments()) return result_;
    result_.ok = true;
    result_.diagnostic = BytecodeVerifyDiagnostic{};
    return result_;
  }

 private:
  bool fail(BytecodeVerifyCode code, std::size_t instruction_index,
            std::string path, std::string message) {
    result_.ok = false;
    result_.diagnostic = BytecodeVerifyDiagnostic{
        code, instruction_index, std::move(path), std::move(message)};
    return false;
  }

  bool check_segment_limit() {
    const std::size_t segments = program_.asgp_dc_segments.size() +
                                 program_.asgp_dp1d_segments.size() +
                                 program_.asgp_dp2d_segments.size();
    if (options_.max_segments != 0 && segments > options_.max_segments) {
      return fail(BytecodeVerifyCode::ResourceLimit, 0, "$.segments",
                  "segment count exceeds configured limit");
    }
    return true;
  }

  bool validate_value(const Value& value, std::size_t index, const std::string& path) {
    if (!is_public_value_tag(value.tag)) {
      return fail(BytecodeVerifyCode::InvalidConstant, index, path,
                  "constant uses a non-public value tag");
    }
    if (value.tag == ValueTag::Char && !is_valid_char(value.i)) {
      return fail(BytecodeVerifyCode::InvalidConstant, index, path,
                  "Char constant is not a Unicode scalar value");
    }
    return true;
  }

  bool validate_maps(int n_locals,
                     const std::unordered_map<std::string, int>& var2idx,
                     const std::string& owner_path) {
    std::set<int> mapped_locals;
    for (const auto& item : var2idx) {
      if (item.second < 0 || item.second >= n_locals) {
        return fail(BytecodeVerifyCode::InvalidVarMapping, 0, owner_path + ".var2idx",
                    "variable mapping local is out of range");
      }
      if (!mapped_locals.insert(item.second).second) {
        return fail(BytecodeVerifyCode::InvalidVarMapping, 0, owner_path + ".var2idx",
                    "multiple variable names alias one local");
      }
    }
    return true;
  }

  bool validate_instruction(const Instr& ins, std::size_t ip,
                            std::size_t code_size, std::size_t const_count,
                            int n_locals, bool allow_asgp, const std::string& code_path) {
    const int raw_op = static_cast<int>(ins.op);
    if (raw_op < static_cast<int>(Opcode::PushConst) ||
        raw_op > static_cast<int>(Opcode::AsgpDp2d)) {
      return fail(BytecodeVerifyCode::UnknownOpcode, ip,
                  code_path + "[" + std::to_string(ip) + "].op", "unknown opcode value");
    }
    const Opcode op = ins.op;
    if (is_private_opcode(op) && !options_.allow_private_opcodes) {
      return fail(BytecodeVerifyCode::InvalidPrivateOpcode, ip,
                  code_path + "[" + std::to_string(ip) + "].op",
                  "private opcode is disabled for this verification profile");
    }
    if (is_asgp_opcode(op) && !allow_asgp) {
      return fail(BytecodeVerifyCode::InvalidPrivateOpcode, ip,
                  code_path + "[" + std::to_string(ip) + "].op",
                  "ASGP calls are forbidden inside phase programs");
    }

    const bool requires_a = op == Opcode::PushConst || op == Opcode::Load || op == Opcode::Store ||
                            op == Opcode::Jmp || op == Opcode::JmpIfFalse || op == Opcode::JmpIfTrue ||
                            op == Opcode::CallBuiltin || op == Opcode::EmptyList || is_asgp_opcode(op);
    if (requires_a && !ins.has_a) {
      return fail(BytecodeVerifyCode::MissingOperand, ip,
                  code_path + "[" + std::to_string(ip) + "].a",
                  std::string(opcode_name(op)) + " requires operand a");
    }
    if (op == Opcode::CallBuiltin && !ins.has_b) {
      return fail(BytecodeVerifyCode::MissingOperand, ip,
                  code_path + "[" + std::to_string(ip) + "].b",
                  "CALL_BUILTIN requires operand b");
    }
    if (op == Opcode::PushConst && (ins.a < 0 || static_cast<std::size_t>(ins.a) >= const_count)) {
      return fail(BytecodeVerifyCode::InvalidConstantIndex, ip,
                  code_path + "[" + std::to_string(ip) + "].a", "constant index is out of range");
    }
    if ((op == Opcode::Load || op == Opcode::Store) && (ins.a < 0 || ins.a >= n_locals)) {
      return fail(BytecodeVerifyCode::InvalidLocalIndex, ip,
                  code_path + "[" + std::to_string(ip) + "].a", "local index is out of range");
    }
    if ((op == Opcode::Jmp || op == Opcode::JmpIfFalse || op == Opcode::JmpIfTrue) &&
        (ins.a < 0 || static_cast<std::size_t>(ins.a) > code_size)) {
      return fail(BytecodeVerifyCode::InvalidJumpTarget, ip,
                  code_path + "[" + std::to_string(ip) + "].a", "jump target is out of range");
    }
    if (op == Opcode::CallBuiltin) {
      BuiltinId id = BuiltinId::Abs;
      if (!builtin_id_from_int(ins.a, id)) {
        return fail(BytecodeVerifyCode::InvalidBuiltinId, ip,
                    code_path + "[" + std::to_string(ip) + "].a", "unknown builtin id");
      }
      if (ins.b != expected_builtin_arity(id)) {
        return fail(BytecodeVerifyCode::InvalidBuiltinArity, ip,
                    code_path + "[" + std::to_string(ip) + "].b",
                    std::string(builtin_name(id)) + " has the wrong bytecode arity");
      }
    }
    if (op == Opcode::EmptyList && (ins.a < 1 || ins.a > 3)) {
      return fail(BytecodeVerifyCode::InvalidConstant, ip,
                  code_path + "[" + std::to_string(ip) + "].a", "invalid typed-list tag");
    }
    if (op == Opcode::AsgpDc &&
        (ins.a < 0 || static_cast<std::size_t>(ins.a) >= program_.asgp_dc_segments.size())) {
      return fail(BytecodeVerifyCode::InvalidSegmentIndex, ip,
                  code_path + "[" + std::to_string(ip) + "].a",
                  "ASGP-DC segment index is out of range");
    }
    if (op == Opcode::AsgpDp1d &&
        (ins.a < 0 || static_cast<std::size_t>(ins.a) >= program_.asgp_dp1d_segments.size())) {
      return fail(BytecodeVerifyCode::InvalidSegmentIndex, ip,
                  code_path + "[" + std::to_string(ip) + "].a",
                  "ASGP-DP1D segment index is out of range");
    }
    if (op == Opcode::AsgpDp2d &&
        (ins.a < 0 || static_cast<std::size_t>(ins.a) >= program_.asgp_dp2d_segments.size())) {
      return fail(BytecodeVerifyCode::InvalidSegmentIndex, ip,
                  code_path + "[" + std::to_string(ip) + "].a",
                  "ASGP-DP2D segment index is out of range");
    }
    return true;
  }

  bool merge_state(AbstractState& existing, const AbstractState& incoming,
                   std::size_t ip, const std::string& code_path, bool* changed) {
    if (existing.stack.size() != incoming.stack.size()) {
      return fail(BytecodeVerifyCode::StackJoinMismatch, ip,
                  code_path + "[" + std::to_string(ip) + "]",
                  "control-flow join has inconsistent stack depth");
    }
    *changed = false;
    for (std::size_t i = 0; i < existing.stack.size(); ++i) {
      AbstractType& dst = existing.stack[i];
      const AbstractType src = incoming.stack[i];
      if (dst == src || dst == AbstractType::Unknown) continue;
      if (src == AbstractType::Unknown) {
        dst = AbstractType::Unknown;
        *changed = true;
        continue;
      }
      return fail(BytecodeVerifyCode::StackJoinMismatch, ip,
                  code_path + "[" + std::to_string(ip) + "]",
                  "control-flow join has inconsistent stack types");
    }
    for (auto it = existing.known_locals.begin(); it != existing.known_locals.end();) {
      const auto incoming_it = incoming.known_locals.find(it->first);
      if (incoming_it == incoming.known_locals.end() || incoming_it->second != it->second) {
        it = existing.known_locals.erase(it);
        *changed = true;
      } else {
        ++it;
      }
    }
    return true;
  }

  bool enqueue(std::size_t target, const AbstractState& state,
               std::vector<AbstractState>& states, std::vector<bool>& seen,
               std::deque<std::size_t>& work, const std::string& code_path) {
    if (!seen[target]) {
      states[target] = state;
      seen[target] = true;
      work.push_back(target);
      return true;
    }
    bool changed = false;
    if (!merge_state(states[target], state, target, code_path, &changed)) return false;
    if (changed) work.push_back(target);
    return true;
  }

  bool verify_code(const std::vector<Value>& consts, const std::vector<Instr>& code,
                   int n_locals, const std::unordered_map<std::string, int>& var2idx,
                   const std::unordered_map<int, AbstractType>& initial_locals,
                   const std::string& code_path, bool require_return, bool allow_asgp,
                   CodeSummary* summary) {
    const std::string owner_path = code_path.substr(0, code_path.size() - 5);
    if (n_locals < 0) {
      return fail(BytecodeVerifyCode::InvalidLocalCount, 0, owner_path + ".n_locals",
                  "n_locals must be non-negative");
    }
    if ((options_.max_instructions_per_code != 0 && code.size() > options_.max_instructions_per_code) ||
        (options_.max_constants_per_code != 0 && consts.size() > options_.max_constants_per_code) ||
        (options_.max_locals_per_code != 0 &&
         static_cast<std::size_t>(n_locals) > options_.max_locals_per_code)) {
      return fail(BytecodeVerifyCode::ResourceLimit, 0, owner_path,
                  "bytecode resource count exceeds configured limit");
    }
    if (!validate_maps(n_locals, var2idx, owner_path)) return false;
    for (std::size_t i = 0; i < consts.size(); ++i) {
      if (!validate_value(consts[i], i, owner_path + ".consts[" + std::to_string(i) + "]")) return false;
    }
    for (std::size_t ip = 0; ip < code.size(); ++ip) {
      if (!validate_instruction(code[ip], ip, code.size(), consts.size(), n_locals,
                                allow_asgp, code_path)) return false;
    }

    std::vector<AbstractState> states(code.size() + 1);
    std::vector<bool> seen(code.size() + 1, false);
    std::deque<std::size_t> work;
    AbstractState start;
    for (const auto& item : initial_locals) {
      if (item.first < 0 || item.first >= n_locals) {
        return fail(BytecodeVerifyCode::InvalidBinderLocal, 0, owner_path + ".binder_locals",
                    "preset binder local is out of range");
      }
      if (item.second != AbstractType::Unknown) start.known_locals[item.first] = item.second;
    }
    states[0] = start;
    seen[0] = true;
    work.push_back(0);

    while (!work.empty()) {
      const std::size_t ip = work.front();
      work.pop_front();
      AbstractState state = states[ip];
      summary->max_stack_depth = std::max(summary->max_stack_depth, state.stack.size());
      result_.verified.max_stack_depth =
          std::max(result_.verified.max_stack_depth, summary->max_stack_depth);
      if (options_.max_stack_depth != 0 && state.stack.size() > options_.max_stack_depth) {
        return fail(BytecodeVerifyCode::ResourceLimit, ip, code_path,
                    "operand stack exceeds configured limit");
      }
      if (ip == code.size()) {
        if (require_return) {
          return fail(BytecodeVerifyCode::InvalidFallthrough, ip, code_path,
                      "main bytecode has a reachable path without RETURN");
        }
        if (state.stack.size() != 1) {
          return fail(BytecodeVerifyCode::InvalidFallthrough, ip, code_path,
                      "phase fallthrough must produce exactly one value");
        }
        continue;
      }

      const Instr& ins = code[ip];
      const Opcode op = ins.op;
      auto require_stack = [&](std::size_t count) -> bool {
        if (state.stack.size() >= count) return true;
        fail(BytecodeVerifyCode::StackUnderflow, ip,
             code_path + "[" + std::to_string(ip) + "]",
             std::string(opcode_name(op)) + " can underflow the operand stack");
        return false;
      };
      bool guaranteed_runtime_error = false;

      if (op == Opcode::PushConst) {
        state.stack.push_back(abstract_type(consts[static_cast<std::size_t>(ins.a)].tag));
      } else if (op == Opcode::Load) {
        const auto local = state.known_locals.find(ins.a);
        state.stack.push_back(local == state.known_locals.end()
                                  ? AbstractType::Unknown
                                  : local->second);
      } else if (op == Opcode::Store) {
        if (!require_stack(1)) return false;
        if (state.stack.back() == AbstractType::Unknown) {
          state.known_locals.erase(ins.a);
        } else {
          state.known_locals[ins.a] = state.stack.back();
        }
        state.stack.pop_back();
      } else if (op == Opcode::CheckList || op == Opcode::CheckInt) {
        if (!require_stack(1)) return false;
        const AbstractType type = state.stack.back();
        if (type != AbstractType::Unknown &&
            ((op == Opcode::CheckList && !is_typed_list(type)) ||
             (op == Opcode::CheckInt && type != AbstractType::Int))) {
          guaranteed_runtime_error = true;
        }
      } else if (op == Opcode::EmptyList) {
        state.stack.push_back(ins.a == 1 ? AbstractType::IntList
                                         : (ins.a == 2 ? AbstractType::FloatList
                                                       : AbstractType::StringList));
      } else if (op == Opcode::EmptyListLike) {
        if (!require_stack(1)) return false;
        const AbstractType source = state.stack.back();
        if (source != AbstractType::Unknown && !is_typed_list(source)) {
          guaranteed_runtime_error = true;
        } else if (source == AbstractType::Unknown) {
          state.stack.back() = AbstractType::Unknown;
        }
      } else if (op == Opcode::Neg || op == Opcode::Not) {
        if (!require_stack(1)) return false;
        const AbstractType operand = state.stack.back();
        if (operand != AbstractType::Unknown &&
            ((op == Opcode::Neg && !is_numeric(operand)) ||
             (op == Opcode::Not && operand != AbstractType::Bool))) {
          guaranteed_runtime_error = true;
        } else {
          state.stack.back() = op == Opcode::Not ? AbstractType::Bool : operand;
        }
      } else if (op == Opcode::Add || op == Opcode::Sub || op == Opcode::Mul ||
                 op == Opcode::Div || op == Opcode::Mod) {
        if (!require_stack(2)) return false;
        const AbstractType rhs = state.stack.back();
        state.stack.pop_back();
        const AbstractType lhs = state.stack.back();
        state.stack.pop_back();
        if (lhs != AbstractType::Unknown && rhs != AbstractType::Unknown &&
            (!is_numeric(lhs) || lhs != rhs)) {
          guaranteed_runtime_error = true;
        } else {
          AbstractType out = AbstractType::Unknown;
          if (lhs == rhs && is_numeric(lhs)) out = op == Opcode::Div ? AbstractType::Float : lhs;
          state.stack.push_back(out);
        }
      } else if (op == Opcode::Lt || op == Opcode::Le || op == Opcode::Gt ||
                 op == Opcode::Ge || op == Opcode::Eq || op == Opcode::Ne) {
        if (!require_stack(2)) return false;
        const AbstractType rhs = state.stack.back();
        state.stack.pop_back();
        const AbstractType lhs = state.stack.back();
        state.stack.pop_back();
        const bool ordering = op == Opcode::Lt || op == Opcode::Le || op == Opcode::Gt || op == Opcode::Ge;
        if (lhs != AbstractType::Unknown && rhs != AbstractType::Unknown &&
            (lhs != rhs || (ordering && !is_numeric(lhs)))) {
          guaranteed_runtime_error = true;
        } else {
          state.stack.push_back(AbstractType::Bool);
        }
      } else if (op == Opcode::JmpIfFalse || op == Opcode::JmpIfTrue) {
        if (!require_stack(1)) return false;
        const AbstractType cond = state.stack.back();
        state.stack.pop_back();
        if (cond != AbstractType::Unknown && cond != AbstractType::Bool) {
          guaranteed_runtime_error = true;
        }
      } else if (op == Opcode::CallBuiltin) {
        const std::size_t argc = static_cast<std::size_t>(ins.b);
        if (!require_stack(argc)) return false;
        std::vector<AbstractType> args(state.stack.end() - static_cast<std::ptrdiff_t>(argc),
                                       state.stack.end());
        state.stack.resize(state.stack.size() - argc);
        BuiltinId id = BuiltinId::Abs;
        (void)builtin_id_from_int(ins.a, id);
        state.stack.push_back(builtin_result_type(id, args));
      } else if (op == Opcode::AsgpDc || op == Opcode::AsgpDp1d || op == Opcode::AsgpDp2d) {
        const std::size_t argc = op == Opcode::AsgpDp2d ? 2U : 1U;
        if (!require_stack(argc)) return false;
        if (op == Opcode::AsgpDc) {
          const AbstractType source = state.stack.back();
          if (source != AbstractType::Unknown && !is_container(source)) guaranteed_runtime_error = true;
        } else {
          for (std::size_t i = 0; i < argc; ++i) {
            const AbstractType input = state.stack[state.stack.size() - argc + i];
            if (input != AbstractType::Unknown && input != AbstractType::Int) {
              guaranteed_runtime_error = true;
            }
          }
        }
        state.stack.resize(state.stack.size() - argc);
        if (!guaranteed_runtime_error) state.stack.push_back(AbstractType::Unknown);
      } else if (op == Opcode::Return) {
        if (!require_stack(1)) return false;
        summary->has_reachable_return = true;
        continue;
      }

      summary->max_stack_depth = std::max(summary->max_stack_depth, state.stack.size());
      if (guaranteed_runtime_error) continue;
      if (op == Opcode::Jmp) {
        if (!enqueue(static_cast<std::size_t>(ins.a), state, states, seen, work, code_path)) return false;
      } else if (op == Opcode::JmpIfFalse || op == Opcode::JmpIfTrue) {
        if (!enqueue(static_cast<std::size_t>(ins.a), state, states, seen, work, code_path) ||
            !enqueue(ip + 1, state, states, seen, work, code_path)) return false;
      } else {
        if (!enqueue(ip + 1, state, states, seen, work, code_path)) return false;
      }
    }
    return true;
  }

  bool validate_binder_map(const PhaseProgram& phase, const std::vector<int>& required_names,
                           const std::string& path,
                           const std::vector<AbstractType>& required_types,
                           std::unordered_map<int, AbstractType>* initial) {
    std::set<int> required_unique;
    std::set<int> local_unique;
    for (std::size_t i = 0; i < required_names.size(); ++i) {
      const int name = required_names[i];
      if (name < 0 || !required_unique.insert(name).second) {
        return fail(BytecodeVerifyCode::InvalidSegmentMetadata, 0, path,
                    "segment binder names must be non-negative and distinct");
      }
      const auto it = phase.binder_locals.find(name);
      if (it == phase.binder_locals.end()) {
        return fail(BytecodeVerifyCode::InvalidBinderLocal, 0, path + ".binder_locals",
                    "phase is missing a required binder-local mapping");
      }
      if (it->second < 0 || it->second >= phase.n_locals || !local_unique.insert(it->second).second) {
        return fail(BytecodeVerifyCode::InvalidBinderLocal, 0, path + ".binder_locals",
                    "phase binder local is out of range or aliases another binder");
      }
      (*initial)[it->second] = required_types[i];
    }
    for (const auto& item : phase.binder_locals) {
      if (item.first < 0 || item.second < 0 || item.second >= phase.n_locals) {
        return fail(BytecodeVerifyCode::InvalidBinderLocal, 0, path + ".binder_locals",
                    "binder-local mapping is out of range");
      }
    }
    return true;
  }

  bool verify_phase(const PhaseProgram& phase, const std::vector<int>& required_names,
                    const std::vector<AbstractType>& required_types, const std::string& path) {
    std::unordered_map<int, AbstractType> initial;
    if (!validate_binder_map(phase, required_names, path, required_types, &initial)) return false;
    CodeSummary summary;
    if (!verify_code(phase.consts, phase.code, phase.n_locals, phase.var2idx, initial,
                     path + ".code", false, false, &summary)) return false;
    result_.verified.phase_program_count += 1;
    return true;
  }

  bool validate_boundary(const Value& value, const std::string& path) {
    return validate_value(value, 0, path);
  }

  bool verify_segments() {
    for (std::size_t i = 0; i < program_.asgp_dc_segments.size(); ++i) {
      const AsgpDcSegment& segment = program_.asgp_dc_segments[i];
      const std::string path = "$.segments.asgp_dc[" + std::to_string(i) + "]";
      if (!verify_phase(segment.solve,
                        {segment.solve_xs_name, segment.solve_n_name, segment.solve_lo_name},
                        {AbstractType::Unknown, AbstractType::Int, AbstractType::Int}, path + ".solve") ||
          !verify_phase(segment.divide, {segment.divide_n_name}, {AbstractType::Int}, path + ".divide") ||
          !verify_phase(segment.combine, {segment.combine_left_name, segment.combine_right_name},
                        {AbstractType::Unknown, AbstractType::Unknown}, path + ".combine")) {
        return false;
      }
    }

    for (std::size_t i = 0; i < program_.asgp_dp1d_segments.size(); ++i) {
      const AsgpDp1dSegment& segment = program_.asgp_dp1d_segments[i];
      const std::string path = "$.segments.asgp_dp1d[" + std::to_string(i) + "]";
      if (segment.lo > segment.hi || segment.base_state < segment.lo || segment.base_state > segment.hi ||
          (segment.dep_kind != -1 && segment.dep_kind != 1) || segment.dep_offsets.empty() ||
          segment.dep_offsets.size() != segment.transition_dep_names.size() ||
          std::any_of(segment.dep_offsets.begin(), segment.dep_offsets.end(),
                      [](int offset) { return offset <= 0; })) {
        return fail(BytecodeVerifyCode::InvalidSegmentMetadata, 0, path,
                    "invalid ASGP-DP1D bounds, base state, dependency direction, or arity");
      }
      if (!validate_boundary(segment.boundary_value, path + ".boundary_value")) return false;
      const AbstractType cell_type = abstract_type(segment.boundary_value.tag);
      std::vector<AbstractType> transition_types(1 + segment.transition_dep_names.size(), cell_type);
      transition_types[0] = AbstractType::Int;
      std::vector<int> transition_names = {segment.transition_state_name};
      transition_names.insert(transition_names.end(), segment.transition_dep_names.begin(),
                              segment.transition_dep_names.end());
      if (!verify_phase(segment.solve, {segment.solve_state_name}, {AbstractType::Int}, path + ".solve") ||
          !verify_phase(segment.transition, transition_names, transition_types, path + ".transition")) {
        return false;
      }
    }

    for (std::size_t i = 0; i < program_.asgp_dp2d_segments.size(); ++i) {
      const AsgpDp2dSegment& segment = program_.asgp_dp2d_segments[i];
      const std::string path = "$.segments.asgp_dp2d[" + std::to_string(i) + "]";
      static constexpr int expected_arities[] = {2, 2, 1, 1, 3, 3};
      if (segment.i_lo > segment.i_hi || segment.j_lo > segment.j_hi ||
          segment.base_i < segment.i_lo || segment.base_i > segment.i_hi ||
          segment.base_j < segment.j_lo || segment.base_j > segment.j_hi ||
          segment.dep_kind < 0 || segment.dep_kind > 5 ||
          segment.transition_dep_names.size() !=
              static_cast<std::size_t>(expected_arities[segment.dep_kind])) {
        return fail(BytecodeVerifyCode::InvalidSegmentMetadata, 0, path,
                    "invalid ASGP-DP2D bounds, base cell, dependency kind, or arity");
      }
      if (!validate_boundary(segment.boundary_value, path + ".boundary_value")) return false;
      const AbstractType cell_type = abstract_type(segment.boundary_value.tag);
      std::vector<int> transition_names = {segment.transition_i_name, segment.transition_j_name};
      transition_names.insert(transition_names.end(), segment.transition_dep_names.begin(),
                              segment.transition_dep_names.end());
      std::vector<AbstractType> transition_types(transition_names.size(), cell_type);
      transition_types[0] = AbstractType::Int;
      transition_types[1] = AbstractType::Int;
      if (!verify_phase(segment.solve, {segment.solve_i_name, segment.solve_j_name},
                        {AbstractType::Int, AbstractType::Int}, path + ".solve") ||
          !verify_phase(segment.transition, transition_names, transition_types, path + ".transition")) {
        return false;
      }
    }
    return true;
  }

  const BytecodeProgram& program_;
  const BytecodeVerifyOptions& options_;
  BytecodeVerifyResult result_;
};

}  // namespace

BytecodeVerifyResult verify_bytecode(const BytecodeProgram& program,
                                     const BytecodeVerifyOptions& options) {
  return Verifier(program, options).run();
}

}  // namespace g3pvm
