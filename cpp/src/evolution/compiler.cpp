#include "g3pvm/evolution/genome.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "g3pvm/core/builtin.hpp"
#include "subtree_utils.hpp"

namespace g3pvm::evo {

namespace {

const char* op_name(NodeKind op) {
  switch (op) {
    case NodeKind::ADD: return "ADD";
    case NodeKind::SUB: return "SUB";
    case NodeKind::MUL: return "MUL";
    case NodeKind::DIV: return "DIV";
    case NodeKind::MOD: return "MOD";
    case NodeKind::LT: return "LT";
    case NodeKind::LE: return "LE";
    case NodeKind::GT: return "GT";
    case NodeKind::GE: return "GE";
    case NodeKind::EQ: return "EQ";
    case NodeKind::NE: return "NE";
    case NodeKind::AND: return "AND";
    case NodeKind::OR: return "OR";
    case NodeKind::NEG: return "NEG";
    case NodeKind::NOT: return "NOT";
    default: return "ADD";
  }
}

class Compiler {
 public:
  explicit Compiler(const std::vector<std::string>* preset_locals = nullptr) {
    if (preset_locals != nullptr) {
      for (const std::string& name : *preset_locals) {
        local(name);
      }
    }
  }

  BytecodeProgram build(const AstProgram& program) {
    if (program.version != "ast-prefix-v1") {
      throw std::runtime_error("unsupported ast prefix version");
    }
    if (program.nodes.empty() || program.nodes[0].kind != NodeKind::PROGRAM) {
      throw std::runtime_error("prefix compile: bad root");
    }
    const std::size_t end = compile_block_prefix(program, 1);
    if (end != program.nodes.size()) {
      throw std::runtime_error("prefix compile: trailing tokens");
    }
    patch_jumps();
    return finalize();
  }

 private:
  struct UnresolvedJump {
    int index = 0;
    std::string label;
  };

  BytecodeProgram finalize() {
    BytecodeProgram out;
    out.consts = consts_;
    out.code = code_;
    out.n_locals = static_cast<int>(var2idx_.size());
    out.var2idx = var2idx_;
    return out;
  }

  int add_const(const Value& value) {
    consts_.push_back(value);
    return static_cast<int>(consts_.size()) - 1;
  }

  int local(const std::string& name) {
    auto it = var2idx_.find(name);
    if (it != var2idx_.end()) {
      return it->second;
    }
    const int idx = static_cast<int>(var2idx_.size());
    var2idx_[name] = idx;
    return idx;
  }

  std::string new_label(const std::string& prefix) {
    return prefix + "_" + std::to_string(label_counter_++);
  }

  std::string new_temp() { return std::string("\x00for_i_") + std::to_string(tmp_counter_++); }

  void emit(const std::string& op, int a = 0, bool has_a = false, int b = 0, bool has_b = false) {
    code_.push_back(Instr{op, a, b, has_a, has_b});
  }

  void emit_jump(const std::string& op, const std::string& label) {
    emit(op, 0, true, 0, false);
    unresolved_.push_back({static_cast<int>(code_.size()) - 1, label});
  }

  void mark_label(const std::string& name) { labels_[name] = static_cast<int>(code_.size()); }

  void patch_jumps() {
    for (const UnresolvedJump& jump : unresolved_) {
      auto it = labels_.find(jump.label);
      if (it == labels_.end()) {
        throw std::runtime_error("undefined label");
      }
      code_[static_cast<std::size_t>(jump.index)].a = it->second;
      code_[static_cast<std::size_t>(jump.index)].has_a = true;
    }
  }

  const AstNode& node_at(const AstProgram& program, std::size_t idx) const {
    if (idx >= program.nodes.size()) {
      throw std::runtime_error("prefix compile: node index out of range");
    }
    return program.nodes[idx];
  }

  const std::string& name_at(const AstProgram& program, int idx) const {
    if (idx < 0 || static_cast<std::size_t>(idx) >= program.names.size()) {
      throw std::runtime_error("prefix compile: name index out of range");
    }
    return program.names[static_cast<std::size_t>(idx)];
  }

  const Value& const_at(const AstProgram& program, int idx) const {
    if (idx < 0 || static_cast<std::size_t>(idx) >= program.consts.size()) {
      throw std::runtime_error("prefix compile: const index out of range");
    }
    return program.consts[static_cast<std::size_t>(idx)];
  }

  std::size_t compile_expr_prefix(const AstProgram& program, std::size_t idx) {
    const AstNode& node = node_at(program, idx);
    switch (node.kind) {
      case NodeKind::CONST:
        emit("PUSH_CONST", add_const(const_at(program, node.i0)), true);
        return idx + 1;
      case NodeKind::VAR:
        emit("LOAD", local(name_at(program, node.i0)), true);
        return idx + 1;
      case NodeKind::NEG:
      case NodeKind::NOT: {
        const std::size_t next = compile_expr_prefix(program, idx + 1);
        emit(node.kind == NodeKind::NEG ? "NEG" : "NOT");
        return next;
      }
      case NodeKind::AND: {
        const std::string false_label = new_label("and_false");
        const std::string end_label = new_label("and_end");
        std::size_t next = compile_expr_prefix(program, idx + 1);
        emit_jump("JMP_IF_FALSE", false_label);
        next = compile_expr_prefix(program, next);
        emit("NOT");
        emit("NOT");
        emit_jump("JMP", end_label);
        mark_label(false_label);
        emit("PUSH_CONST", add_const(Value::from_bool(false)), true);
        mark_label(end_label);
        return next;
      }
      case NodeKind::OR: {
        const std::string true_label = new_label("or_true");
        const std::string end_label = new_label("or_end");
        std::size_t next = compile_expr_prefix(program, idx + 1);
        emit_jump("JMP_IF_TRUE", true_label);
        next = compile_expr_prefix(program, next);
        emit("NOT");
        emit("NOT");
        emit_jump("JMP", end_label);
        mark_label(true_label);
        emit("PUSH_CONST", add_const(Value::from_bool(true)), true);
        mark_label(end_label);
        return next;
      }
      case NodeKind::ADD:
      case NodeKind::SUB:
      case NodeKind::MUL:
      case NodeKind::DIV:
      case NodeKind::MOD:
      case NodeKind::LT:
      case NodeKind::LE:
      case NodeKind::GT:
      case NodeKind::GE:
      case NodeKind::EQ:
      case NodeKind::NE: {
        std::size_t next = compile_expr_prefix(program, idx + 1);
        next = compile_expr_prefix(program, next);
        emit(op_name(node.kind));
        return next;
      }
      case NodeKind::IF_EXPR: {
        const std::string else_label = new_label("ifexpr_else");
        const std::string end_label = new_label("ifexpr_end");
        std::size_t next = compile_expr_prefix(program, idx + 1);
        emit_jump("JMP_IF_FALSE", else_label);
        next = compile_expr_prefix(program, next);
        emit_jump("JMP", end_label);
        mark_label(else_label);
        next = compile_expr_prefix(program, next);
        mark_label(end_label);
        return next;
      }
      case NodeKind::CALL_ABS:
      case NodeKind::CALL_MIN:
      case NodeKind::CALL_MAX:
      case NodeKind::CALL_CLIP:
      case NodeKind::CALL_LEN:
      case NodeKind::CALL_CONCAT:
      case NodeKind::CALL_SLICE:
      case NodeKind::CALL_INDEX: {
        std::size_t next = idx + 1;
        const int argc = subtree::node_arity(node.kind);
        for (int i = 0; i < argc; ++i) {
          next = compile_expr_prefix(program, next);
        }
        g3pvm::BuiltinId builtin_id = g3pvm::BuiltinId::Index;
        if (node.kind == NodeKind::CALL_ABS) builtin_id = g3pvm::BuiltinId::Abs;
        else if (node.kind == NodeKind::CALL_MIN) builtin_id = g3pvm::BuiltinId::Min;
        else if (node.kind == NodeKind::CALL_MAX) builtin_id = g3pvm::BuiltinId::Max;
        else if (node.kind == NodeKind::CALL_CLIP) builtin_id = g3pvm::BuiltinId::Clip;
        else if (node.kind == NodeKind::CALL_LEN) builtin_id = g3pvm::BuiltinId::Len;
        else if (node.kind == NodeKind::CALL_CONCAT) builtin_id = g3pvm::BuiltinId::Concat;
        else if (node.kind == NodeKind::CALL_SLICE) builtin_id = g3pvm::BuiltinId::Slice;
        emit("CALL_BUILTIN", static_cast<int>(builtin_id), true, argc, true);
        return next;
      }
      default:
        throw std::runtime_error("prefix compile: expected expr node");
    }
  }

  std::size_t compile_block_prefix(const AstProgram& program, std::size_t idx) {
    const AstNode& node = node_at(program, idx);
    if (node.kind == NodeKind::BLOCK_NIL) {
      return idx + 1;
    }
    if (node.kind != NodeKind::BLOCK_CONS) {
      throw std::runtime_error("prefix compile: expected block node");
    }
    const std::size_t next = compile_stmt_prefix(program, idx + 1);
    return compile_block_prefix(program, next);
  }

  std::size_t compile_stmt_prefix(const AstProgram& program, std::size_t idx) {
    const AstNode& node = node_at(program, idx);
    if (node.kind == NodeKind::ASSIGN) {
      const std::size_t next = compile_expr_prefix(program, idx + 1);
      emit("STORE", local(name_at(program, node.i0)), true);
      return next;
    }
    if (node.kind == NodeKind::RETURN) {
      const std::size_t next = compile_expr_prefix(program, idx + 1);
      emit("RETURN");
      return next;
    }
    if (node.kind == NodeKind::IF_STMT) {
      const std::string else_label = new_label("if_else");
      const std::string end_label = new_label("if_end");
      std::size_t next = compile_expr_prefix(program, idx + 1);
      emit_jump("JMP_IF_FALSE", else_label);
      next = compile_block_prefix(program, next);
      emit_jump("JMP", end_label);
      mark_label(else_label);
      next = compile_block_prefix(program, next);
      mark_label(end_label);
      return next;
    }
    if (node.kind == NodeKind::FOR_RANGE) {
      if (node.i1 < 0) {
        emit("PUSH_CONST", add_const(Value::from_bool(true)), true);
        emit("NEG");
        return compile_block_prefix(program, idx + 1);
      }

      const int idx_k = add_const(Value::from_int(node.i1));
      const int idx_0 = add_const(Value::from_int(0));
      const int idx_1 = add_const(Value::from_int(1));
      const int counter_i = local(new_temp());
      const int user_i = local(name_at(program, node.i0));

      const std::string loop_label = new_label("for_loop");
      const std::string end_label = new_label("for_end");

      emit("PUSH_CONST", idx_0, true);
      emit("STORE", counter_i, true);

      mark_label(loop_label);
      emit("LOAD", counter_i, true);
      emit("PUSH_CONST", idx_k, true);
      emit("LT");
      emit_jump("JMP_IF_FALSE", end_label);

      emit("LOAD", counter_i, true);
      emit("STORE", user_i, true);

      const std::size_t next = compile_block_prefix(program, idx + 1);

      emit("LOAD", counter_i, true);
      emit("PUSH_CONST", idx_1, true);
      emit("ADD");
      emit("STORE", counter_i, true);
      emit_jump("JMP", loop_label);
      mark_label(end_label);
      return next;
    }
    throw std::runtime_error("prefix compile: expected stmt node");
  }

  std::vector<Value> consts_;
  std::vector<Instr> code_;
  std::vector<UnresolvedJump> unresolved_;
  std::unordered_map<std::string, int> labels_;
  std::unordered_map<std::string, int> var2idx_;
  int label_counter_ = 0;
  int tmp_counter_ = 0;
};

}  // namespace

BytecodeProgram compile_for_eval(const ProgramGenome& genome) {
  Compiler compiler;
  return compiler.build(genome.ast);
}

BytecodeProgram compile_for_eval_with_preset_locals(const ProgramGenome& genome,
                                                    const std::vector<std::string>& preset_locals) {
  Compiler compiler(&preset_locals);
  return compiler.build(genome.ast);
}

}  // namespace g3pvm::evo
