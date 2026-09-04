#include "gagp/evolution/compiler.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gagp/core/builtin.hpp"
#include "gagp/core/bytecode_verify.hpp"
#include "gagp/evolution/node_descriptor.hpp"
#include "subtree_utils.hpp"

namespace gagp::evo {

namespace {

Opcode op_name(NodeKind op) {
  switch (op) {
    case NodeKind::ADD: return Opcode::Add;
    case NodeKind::SUB: return Opcode::Sub;
    case NodeKind::MUL: return Opcode::Mul;
    case NodeKind::DIV: return Opcode::Div;
    case NodeKind::MOD: return Opcode::Mod;
    case NodeKind::LT: return Opcode::Lt;
    case NodeKind::LE: return Opcode::Le;
    case NodeKind::GT: return Opcode::Gt;
    case NodeKind::GE: return Opcode::Ge;
    case NodeKind::EQ: return Opcode::Eq;
    case NodeKind::NE: return Opcode::Ne;
    case NodeKind::NEG: return Opcode::Neg;
    case NodeKind::NOT: return Opcode::Not;
    default:
      throw std::runtime_error("prefix compile: unsupported opcode lowering for node kind");
  }
}

int asgp_dp1d_expected_arity(NodeKind kind) {
  const NodeDescriptor& descriptor = node_descriptor(kind);
  if (descriptor.dependency_family != DependencyFamily::Dp1d) {
    throw std::runtime_error("prefix compile: invalid ASGP-DP1D dependency kind");
  }
  return descriptor.dependency_arity;
}

int asgp_dp2d_expected_arity(NodeKind kind) {
  const NodeDescriptor& descriptor = node_descriptor(kind);
  if (descriptor.dependency_family != DependencyFamily::Dp2d) {
    throw std::runtime_error("prefix compile: invalid ASGP-DP2D dependency kind");
  }
  return descriptor.dependency_arity;
}

class Compiler {
 public:
  explicit Compiler(const std::vector<std::string>* preset_locals = nullptr,
                    const VerifiedAst* verified = nullptr)
      : verified_(verified) {
    if (preset_locals != nullptr) {
      for (const std::string& name : *preset_locals) {
        local(name);
      }
    }
  }

  BytecodeProgram build(const AstProgram& program) {
    if (program.version != k_ast_prefix_version_current) {
      throw std::runtime_error("unsupported ast prefix version");
    }
    if (program.nodes.empty() || program.nodes[0].kind != NodeKind::PROGRAM) {
      throw std::runtime_error("prefix compile: bad root");
    }
    if (verified_ != nullptr &&
        (verified_->subtree_end.size() != program.nodes.size() ||
         verified_->expression_types.size() != program.nodes.size() ||
         verified_->subtree_end[0] != program.nodes.size())) {
      throw std::invalid_argument("prefix compile: VerifiedAst does not match program shape");
    }
    const std::size_t end = compile_block_prefix(program, 1);
    if (end != program.nodes.size()) {
      throw std::runtime_error("prefix compile: trailing tokens");
    }
    patch_jumps();
    BytecodeProgram out = finalize();
#ifndef NDEBUG
    const BytecodeVerifyResult verified = verify_bytecode(out);
    if (!verified) {
      throw std::runtime_error(
          std::string("compiler produced invalid bytecode (") +
          bytecode_verify_code_name(verified.diagnostic.code) + ") at " +
          verified.diagnostic.path + ": " + verified.diagnostic.message);
    }
#endif
    return out;
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
    out.asgp_dc_segments = asgp_dc_segments_;
    out.asgp_dp1d_segments = asgp_dp1d_segments_;
    out.asgp_dp2d_segments = asgp_dp2d_segments_;
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

  void push_binder(int name_id, int local_idx) {
    binder_stack_[name_id].push_back(local_idx);
  }

  void pop_binder(int name_id) {
    auto it = binder_stack_.find(name_id);
    if (it == binder_stack_.end() || it->second.empty()) {
      throw std::runtime_error("prefix compile: binder stack underflow");
    }
    it->second.pop_back();
  }

  int bound_local(const AstProgram& program, int name_id) const {
    auto it = binder_stack_.find(name_id);
    if (it == binder_stack_.end() || it->second.empty()) {
      throw std::runtime_error("prefix compile: undefined binder " + name_at(program, name_id));
    }
    return it->second.back();
  }

  std::size_t expr_end_prefix(const AstProgram& program, std::size_t idx) const {
    (void)node_at(program, idx);
    if (verified_ != nullptr) {
      const std::size_t end = verified_->subtree_end[idx];
      if (end <= idx || end > program.nodes.size() ||
          verified_->expression_types[idx] == RType::Invalid) {
        throw std::invalid_argument(
            "prefix compile: VerifiedAst expression annotation is invalid");
      }
      return end;
    }
    std::size_t cur = idx + 1;
    for (int i = 0; i < subtree::node_arity(program.nodes[idx].kind); ++i) {
      cur = expr_end_prefix(program, cur);
    }
    return cur;
  }

  const LinearRecBinders& linear_rec_binders_for_node(const AstProgram& program, std::size_t idx) const {
    for (const LinearRecBinders& binders : program.linear_rec_binders) {
      if (binders.node_index == idx) return binders;
    }
    throw std::runtime_error("prefix compile: missing LinearRec binder metadata");
  }

  const AsgpDcBinders& asgp_dc_binders_for_node(const AstProgram& program, std::size_t idx) const {
    for (const AsgpDcBinders& binders : program.asgp_dc_binders) {
      if (binders.node_index == idx) return binders;
    }
    throw std::runtime_error("prefix compile: missing ASGP-DC binder metadata");
  }

  const AsgpDp1dSpec& asgp_dp1d_spec_for_node(const AstProgram& program, std::size_t idx) const {
    for (const AsgpDp1dSpec& spec : program.asgp_dp1d_specs) {
      if (spec.node_index == idx) return spec;
    }
    throw std::runtime_error("prefix compile: missing ASGP-DP1D metadata");
  }

  const AsgpDp2dSpec& asgp_dp2d_spec_for_node(const AstProgram& program, std::size_t idx) const {
    for (const AsgpDp2dSpec& spec : program.asgp_dp2d_specs) {
      if (spec.node_index == idx) return spec;
    }
    throw std::runtime_error("prefix compile: missing ASGP-DP2D metadata");
  }

  int asgp_dp2d_dep_kind_code(NodeKind kind) const {
    switch (kind) {
      case NodeKind::DP2_CROSS_BACKWARD:
        return 0;
      case NodeKind::DP2_CROSS_FORWARD:
        return 1;
      case NodeKind::DP2_DIAGONAL_BACKWARD:
        return 2;
      case NodeKind::DP2_DIAGONAL_FORWARD:
        return 3;
      case NodeKind::DP2_NEIGHBORHOOD_BACKWARD3:
        return 4;
      case NodeKind::DP2_NEIGHBORHOOD_FORWARD3:
        return 5;
      default:
        throw std::runtime_error("prefix compile: invalid ASGP-DP2D dependency kind");
    }
  }

  std::size_t compile_for_loop_body(const std::string& user_name, int bound_local, const AstProgram& program, std::size_t body_idx) {
    const int idx_0 = add_const(Value::from_int(0));
    const int idx_1 = add_const(Value::from_int(1));
    const int counter_i = local(new_temp());
    const int user_i = local(user_name);

    const std::string loop_label = new_label("for_loop");
    const std::string end_label = new_label("for_end");

    emit(Opcode::PushConst, idx_0, true);
    emit(Opcode::Store, counter_i, true);

    mark_label(loop_label);
    emit(Opcode::Load, counter_i, true);
    emit(Opcode::Load, bound_local, true);
    emit(Opcode::Lt);
    emit_jump(Opcode::JmpIfFalse, end_label);

    emit(Opcode::Load, counter_i, true);
    emit(Opcode::Store, user_i, true);

    const std::size_t next = compile_block_prefix(program, body_idx);

    emit(Opcode::Load, counter_i, true);
    emit(Opcode::PushConst, idx_1, true);
    emit(Opcode::Add);
    emit(Opcode::Store, counter_i, true);
    emit_jump(Opcode::Jmp, loop_label);
    mark_label(end_label);
    return next;
  }

  std::size_t compile_map_list_prefix(const AstProgram& program, std::size_t idx) {
    const AstNode& node = node_at(program, idx);
    const int xs_local = local(new_temp());
    const int out_local = local(new_temp());
    const int val_local = local(new_temp());
    const int counter_i = local(new_temp());
    const int binder_local = local(new_temp());
    const std::string loop_label = new_label("map_loop");
    const std::string end_label = new_label("map_end");

    const std::size_t body_idx = compile_expr_prefix(program, idx + 1);
    const std::size_t body_end = expr_end_prefix(program, body_idx);

    emit(Opcode::CheckList);
    emit(Opcode::Store, xs_local, true);
    emit(Opcode::EmptyList, node.i1, true);
    emit(Opcode::Store, out_local, true);
    emit(Opcode::PushConst, add_const(Value::from_int(0)), true);
    emit(Opcode::Store, counter_i, true);

    mark_label(loop_label);
    emit(Opcode::Load, counter_i, true);
    emit(Opcode::Load, xs_local, true);
    emit(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Len), true, 1, true);
    emit(Opcode::Lt);
    emit_jump(Opcode::JmpIfFalse, end_label);
    emit(Opcode::Load, xs_local, true);
    emit(Opcode::Load, counter_i, true);
    emit(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Index), true, 2, true);
    emit(Opcode::Store, binder_local, true);
    push_binder(node.i0, binder_local);
    const std::size_t compiled_body_end = compile_expr_prefix(program, body_idx);
    pop_binder(node.i0);
    if (compiled_body_end != body_end) {
      throw std::runtime_error("prefix compile: MapList body trailing tokens");
    }
    emit(Opcode::Store, val_local, true);
    emit(Opcode::Load, out_local, true);
    emit(Opcode::Load, val_local, true);
    emit(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Append), true, 2, true);
    emit(Opcode::Store, out_local, true);
    emit(Opcode::Load, counter_i, true);
    emit(Opcode::PushConst, add_const(Value::from_int(1)), true);
    emit(Opcode::Add);
    emit(Opcode::Store, counter_i, true);
    emit_jump(Opcode::Jmp, loop_label);
    mark_label(end_label);
    emit(Opcode::Load, out_local, true);
    return body_end;
  }

  std::size_t compile_filter_list_prefix(const AstProgram& program, std::size_t idx) {
    const AstNode& node = node_at(program, idx);
    const int xs_local = local(new_temp());
    const int out_local = local(new_temp());
    const int counter_i = local(new_temp());
    const int binder_local = local(new_temp());
    const std::string loop_label = new_label("filter_loop");
    const std::string skip_label = new_label("filter_skip");
    const std::string end_label = new_label("filter_end");

    const std::size_t pred_idx = compile_expr_prefix(program, idx + 1);
    const std::size_t pred_end = expr_end_prefix(program, pred_idx);

    emit(Opcode::CheckList);
    emit(Opcode::Store, xs_local, true);
    emit(Opcode::Load, xs_local, true);
    emit(Opcode::EmptyListLike);
    emit(Opcode::Store, out_local, true);
    emit(Opcode::PushConst, add_const(Value::from_int(0)), true);
    emit(Opcode::Store, counter_i, true);

    mark_label(loop_label);
    emit(Opcode::Load, counter_i, true);
    emit(Opcode::Load, xs_local, true);
    emit(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Len), true, 1, true);
    emit(Opcode::Lt);
    emit_jump(Opcode::JmpIfFalse, end_label);
    emit(Opcode::Load, xs_local, true);
    emit(Opcode::Load, counter_i, true);
    emit(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Index), true, 2, true);
    emit(Opcode::Store, binder_local, true);
    push_binder(node.i0, binder_local);
    const std::size_t compiled_pred_end = compile_expr_prefix(program, pred_idx);
    pop_binder(node.i0);
    if (compiled_pred_end != pred_end) {
      throw std::runtime_error("prefix compile: FilterList predicate trailing tokens");
    }
    emit_jump(Opcode::JmpIfFalse, skip_label);
    emit(Opcode::Load, out_local, true);
    emit(Opcode::Load, binder_local, true);
    emit(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Append), true, 2, true);
    emit(Opcode::Store, out_local, true);
    mark_label(skip_label);
    emit(Opcode::Load, counter_i, true);
    emit(Opcode::PushConst, add_const(Value::from_int(1)), true);
    emit(Opcode::Add);
    emit(Opcode::Store, counter_i, true);
    emit_jump(Opcode::Jmp, loop_label);
    mark_label(end_label);
    emit(Opcode::Load, out_local, true);
    return pred_end;
  }

  std::size_t compile_linear_rec_prefix(const AstProgram& program, std::size_t idx) {
    const LinearRecBinders& binders = linear_rec_binders_for_node(program, idx);
    const int xs_local = local(new_temp());
    const int start_local = local(new_temp());
    const int len_local = local(new_temp());
    const int counter_local = local(new_temp());
    const int elem_local = local(new_temp());
    const int accum_local = local(new_temp());
    const int index_local = local(new_temp());
    const std::string nonempty_label = new_label("linear_nonempty");
    const std::string step_check_label = new_label("linear_step_check");
    const std::string done_label = new_label("linear_done");
    const std::string end_label = new_label("linear_end");

    const std::size_t start_idx = compile_expr_prefix(program, idx + 1);
    const std::size_t empty_idx = compile_expr_prefix(program, start_idx);
    const std::size_t step_idx = expr_end_prefix(program, empty_idx);
    const std::size_t last_idx = expr_end_prefix(program, step_idx);
    const std::size_t linear_end = expr_end_prefix(program, last_idx);

    emit(Opcode::CheckInt);
    emit(Opcode::Store, start_local, true);
    emit(Opcode::CheckList);
    emit(Opcode::Store, xs_local, true);
    emit(Opcode::Load, xs_local, true);
    emit(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Len), true, 1, true);
    emit(Opcode::Store, len_local, true);
    emit(Opcode::Load, len_local, true);
    emit(Opcode::PushConst, add_const(Value::from_int(0)), true);
    emit(Opcode::Eq);
    emit_jump(Opcode::JmpIfFalse, nonempty_label);
    const std::size_t compiled_empty_end = compile_expr_prefix(program, empty_idx);
    if (compiled_empty_end != step_idx) {
      throw std::runtime_error("prefix compile: LinearRec empty case trailing tokens");
    }
    emit_jump(Opcode::Jmp, end_label);

    mark_label(nonempty_label);
    emit(Opcode::Load, len_local, true);
    emit(Opcode::PushConst, add_const(Value::from_int(1)), true);
    emit(Opcode::Sub);
    emit(Opcode::Store, counter_local, true);
    emit(Opcode::Load, xs_local, true);
    emit(Opcode::Load, counter_local, true);
    emit(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Index), true, 2, true);
    emit(Opcode::Store, elem_local, true);
    emit(Opcode::Load, start_local, true);
    emit(Opcode::Load, counter_local, true);
    emit(Opcode::Add);
    emit(Opcode::Store, index_local, true);
    push_binder(binders.elem_name, elem_local);
    push_binder(binders.index_name, index_local);
    const std::size_t compiled_last_end = compile_expr_prefix(program, last_idx);
    pop_binder(binders.index_name);
    pop_binder(binders.elem_name);
    if (compiled_last_end != linear_end) {
      throw std::runtime_error("prefix compile: LinearRec last body trailing tokens");
    }
    emit(Opcode::Store, accum_local, true);

    mark_label(step_check_label);
    emit(Opcode::Load, counter_local, true);
    emit(Opcode::PushConst, add_const(Value::from_int(0)), true);
    emit(Opcode::Gt);
    emit_jump(Opcode::JmpIfFalse, done_label);
    emit(Opcode::Load, counter_local, true);
    emit(Opcode::PushConst, add_const(Value::from_int(1)), true);
    emit(Opcode::Sub);
    emit(Opcode::Store, counter_local, true);
    emit(Opcode::Load, xs_local, true);
    emit(Opcode::Load, counter_local, true);
    emit(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::Index), true, 2, true);
    emit(Opcode::Store, elem_local, true);
    emit(Opcode::Load, start_local, true);
    emit(Opcode::Load, counter_local, true);
    emit(Opcode::Add);
    emit(Opcode::Store, index_local, true);
    push_binder(binders.elem_name, elem_local);
    push_binder(binders.accum_name, accum_local);
    push_binder(binders.index_name, index_local);
    const std::size_t compiled_step_end = compile_expr_prefix(program, step_idx);
    pop_binder(binders.index_name);
    pop_binder(binders.accum_name);
    pop_binder(binders.elem_name);
    if (compiled_step_end != last_idx) {
      throw std::runtime_error("prefix compile: LinearRec step body trailing tokens");
    }
    emit(Opcode::Store, accum_local, true);
    emit_jump(Opcode::Jmp, step_check_label);

    mark_label(done_label);
    emit(Opcode::Load, accum_local, true);
    mark_label(end_label);
    return linear_end;
  }

  bool subtree_contains_asgp(const AstProgram& program, std::size_t idx) const {
    const std::size_t end = expr_end_prefix(program, idx);
    for (std::size_t i = idx; i < end; ++i) {
      const NodeKind kind = program.nodes[i].kind;
      if (kind == NodeKind::ASGP_DC || kind == NodeKind::ASGP_DP1D || kind == NodeKind::ASGP_DP2D) {
        return true;
      }
    }
    return false;
  }

  PhaseProgram compile_phase_expr(const AstProgram& program,
                                  std::size_t expr_idx,
                                  std::size_t expected_end,
                                  const std::vector<int>& binder_names) {
    Compiler phase;
    std::unordered_map<int, int> binder_locals;
    for (std::size_t offset = 0; offset < binder_names.size(); ++offset) {
      const int name_id = binder_names[offset];
      const int local_idx =
          phase.local(std::string("\x00asgp_binder_") + std::to_string(offset) + "_" + std::to_string(name_id));
      binder_locals[name_id] = local_idx;
      phase.push_binder(name_id, local_idx);
    }
    const std::size_t end = phase.compile_expr_prefix(program, expr_idx);
    for (auto it = binder_names.rbegin(); it != binder_names.rend(); ++it) {
      phase.pop_binder(*it);
    }
    if (end != expected_end) {
      throw std::runtime_error("prefix compile: ASGP-DC phase trailing tokens");
    }
    phase.patch_jumps();
    PhaseProgram out;
    out.consts = phase.consts_;
    out.code = phase.code_;
    out.n_locals = static_cast<int>(phase.var2idx_.size());
    out.var2idx = phase.var2idx_;
    out.binder_locals = std::move(binder_locals);
    return out;
  }

  std::size_t compile_asgp_dc_prefix(const AstProgram& program, std::size_t idx) {
    const AsgpDcBinders& binders = asgp_dc_binders_for_node(program, idx);
    const std::size_t solve_idx = compile_expr_prefix(program, idx + 1);
    const std::size_t divide_idx = expr_end_prefix(program, solve_idx);
    const std::size_t combine_idx = expr_end_prefix(program, divide_idx);
    const std::size_t dc_end = expr_end_prefix(program, combine_idx);
    if (subtree_contains_asgp(program, solve_idx) || subtree_contains_asgp(program, divide_idx) ||
        subtree_contains_asgp(program, combine_idx)) {
      throw std::runtime_error("prefix compile: ASGP-DC phase bodies must not contain ASGP source forms");
    }

    AsgpDcSegment segment;
    segment.solve_xs_name = binders.solve_xs_name;
    segment.solve_n_name = binders.solve_n_name;
    segment.solve_lo_name = binders.solve_lo_name;
    segment.divide_n_name = binders.divide_n_name;
    segment.combine_left_name = binders.combine_left_name;
    segment.combine_right_name = binders.combine_right_name;
    segment.solve = compile_phase_expr(
        program, solve_idx, divide_idx, {binders.solve_xs_name, binders.solve_n_name, binders.solve_lo_name});
    segment.divide = compile_phase_expr(program, divide_idx, combine_idx, {binders.divide_n_name});
    segment.combine = compile_phase_expr(
        program, combine_idx, dc_end, {binders.combine_left_name, binders.combine_right_name});

    const int segment_idx = static_cast<int>(asgp_dc_segments_.size());
    asgp_dc_segments_.push_back(std::move(segment));
    emit(Opcode::AsgpDc, segment_idx, true);
    return dc_end;
  }

  std::size_t compile_asgp_dp1d_prefix(const AstProgram& program, std::size_t idx) {
    const AsgpDp1dSpec& spec = asgp_dp1d_spec_for_node(program, idx);
    const int expected_arity = asgp_dp1d_expected_arity(spec.dep_kind);
    if (spec.dep_offsets.size() != static_cast<std::size_t>(expected_arity) ||
        spec.transition_dep_names.size() != static_cast<std::size_t>(expected_arity)) {
      throw std::runtime_error("prefix compile: ASGP-DP1D dependency arity mismatch");
    }
    const std::size_t solve_idx = compile_expr_prefix(program, idx + 1);
    const std::size_t transition_idx = expr_end_prefix(program, solve_idx);
    const std::size_t dp_end = expr_end_prefix(program, transition_idx);
    if (subtree_contains_asgp(program, solve_idx) || subtree_contains_asgp(program, transition_idx)) {
      throw std::runtime_error("prefix compile: ASGP-DP1D phase bodies must not contain ASGP source forms");
    }

    std::vector<int> transition_binders;
    transition_binders.reserve(1 + spec.transition_dep_names.size());
    transition_binders.push_back(spec.transition_state_name);
    for (const int name_id : spec.transition_dep_names) {
      transition_binders.push_back(name_id);
    }

    AsgpDp1dSegment segment;
    segment.lo = spec.lo;
    segment.hi = spec.hi;
    segment.base_state = spec.base_state;
    segment.boundary_value = const_at(program, spec.boundary_const);
    segment.dep_kind = (spec.dep_kind == NodeKind::DP1_BACKWARD1 || spec.dep_kind == NodeKind::DP1_BACKWARD2 ||
                        spec.dep_kind == NodeKind::DP1_BACKWARD3)
                           ? -1
                           : 1;
    segment.dep_offsets = spec.dep_offsets;
    segment.solve_state_name = spec.solve_state_name;
    segment.transition_state_name = spec.transition_state_name;
    segment.transition_dep_names = spec.transition_dep_names;
    segment.solve = compile_phase_expr(program, solve_idx, transition_idx, {spec.solve_state_name});
    segment.transition = compile_phase_expr(program, transition_idx, dp_end, transition_binders);

    const int segment_idx = static_cast<int>(asgp_dp1d_segments_.size());
    asgp_dp1d_segments_.push_back(std::move(segment));
    emit(Opcode::AsgpDp1d, segment_idx, true);
    return dp_end;
  }

  std::size_t compile_asgp_dp2d_prefix(const AstProgram& program, std::size_t idx) {
    const AsgpDp2dSpec& spec = asgp_dp2d_spec_for_node(program, idx);
    const int expected_arity = asgp_dp2d_expected_arity(spec.dep_kind);
    if (spec.transition_dep_names.size() != static_cast<std::size_t>(expected_arity)) {
      throw std::runtime_error("prefix compile: ASGP-DP2D dependency arity mismatch");
    }
    const std::size_t state_j_idx = compile_expr_prefix(program, idx + 1);
    const std::size_t solve_idx = compile_expr_prefix(program, state_j_idx);
    const std::size_t transition_idx = expr_end_prefix(program, solve_idx);
    const std::size_t dp_end = expr_end_prefix(program, transition_idx);
    if (subtree_contains_asgp(program, solve_idx) || subtree_contains_asgp(program, transition_idx)) {
      throw std::runtime_error("prefix compile: ASGP-DP2D phase bodies must not contain ASGP source forms");
    }

    std::vector<int> transition_binders;
    transition_binders.reserve(2 + spec.transition_dep_names.size());
    transition_binders.push_back(spec.transition_i_name);
    transition_binders.push_back(spec.transition_j_name);
    for (const int name_id : spec.transition_dep_names) {
      transition_binders.push_back(name_id);
    }

    AsgpDp2dSegment segment;
    segment.i_lo = spec.i_lo;
    segment.i_hi = spec.i_hi;
    segment.j_lo = spec.j_lo;
    segment.j_hi = spec.j_hi;
    segment.base_i = spec.base_i;
    segment.base_j = spec.base_j;
    segment.boundary_value = const_at(program, spec.boundary_const);
    segment.dep_kind = asgp_dp2d_dep_kind_code(spec.dep_kind);
    segment.solve_i_name = spec.solve_i_name;
    segment.solve_j_name = spec.solve_j_name;
    segment.transition_i_name = spec.transition_i_name;
    segment.transition_j_name = spec.transition_j_name;
    segment.transition_dep_names = spec.transition_dep_names;
    segment.solve = compile_phase_expr(program, solve_idx, transition_idx, {spec.solve_i_name, spec.solve_j_name});
    segment.transition = compile_phase_expr(program, transition_idx, dp_end, transition_binders);

    const int segment_idx = static_cast<int>(asgp_dp2d_segments_.size());
    asgp_dp2d_segments_.push_back(std::move(segment));
    emit(Opcode::AsgpDp2d, segment_idx, true);
    return dp_end;
  }

  void emit(Opcode op, int a = 0, bool has_a = false, int b = 0, bool has_b = false) {
    code_.push_back(Instr{op, a, b, has_a, has_b});
  }

  void emit_jump(Opcode op, const std::string& label) {
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
        emit(Opcode::PushConst, add_const(const_at(program, node.i0)), true);
        return idx + 1;
      case NodeKind::VAR:
        emit(Opcode::Load, local(name_at(program, node.i0)), true);
        return idx + 1;
      case NodeKind::BOUND_VAR:
        emit(Opcode::Load, bound_local(program, node.i0), true);
        return idx + 1;
      case NodeKind::NEG:
      case NodeKind::NOT: {
        const std::size_t next = compile_expr_prefix(program, idx + 1);
        emit(node.kind == NodeKind::NEG ? Opcode::Neg : Opcode::Not);
        return next;
      }
      case NodeKind::AND: {
        const std::string false_label = new_label("and_false");
        const std::string end_label = new_label("and_end");
        std::size_t next = compile_expr_prefix(program, idx + 1);
        emit_jump(Opcode::JmpIfFalse, false_label);
        next = compile_expr_prefix(program, next);
        emit(Opcode::Not);
        emit(Opcode::Not);
        emit_jump(Opcode::Jmp, end_label);
        mark_label(false_label);
        emit(Opcode::PushConst, add_const(Value::from_bool(false)), true);
        mark_label(end_label);
        return next;
      }
      case NodeKind::OR: {
        const std::string true_label = new_label("or_true");
        const std::string end_label = new_label("or_end");
        std::size_t next = compile_expr_prefix(program, idx + 1);
        emit_jump(Opcode::JmpIfTrue, true_label);
        next = compile_expr_prefix(program, next);
        emit(Opcode::Not);
        emit(Opcode::Not);
        emit_jump(Opcode::Jmp, end_label);
        mark_label(true_label);
        emit(Opcode::PushConst, add_const(Value::from_bool(true)), true);
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
        emit_jump(Opcode::JmpIfFalse, else_label);
        next = compile_expr_prefix(program, next);
        emit_jump(Opcode::Jmp, end_label);
        mark_label(else_label);
        next = compile_expr_prefix(program, next);
        mark_label(end_label);
        return next;
      }
      case NodeKind::CALL_ABS:
      case NodeKind::CALL_MIN:
      case NodeKind::CALL_MAX:
      case NodeKind::CALL_CLIP:
      case NodeKind::CALL_IDIV0:
      case NodeKind::CALL_IMOD0:
      case NodeKind::CALL_LEN:
      case NodeKind::CALL_CONCAT:
      case NodeKind::CALL_SLICE:
      case NodeKind::CALL_INDEX:
      case NodeKind::CALL_APPEND:
      case NodeKind::CALL_PREPEND:
      case NodeKind::CALL_REVERSE:
      case NodeKind::CALL_FIND:
      case NodeKind::CALL_CONTAINS:
      case NodeKind::CALL_CHAR_TO_STRING:
      case NodeKind::CALL_STRING_TO_CHAR:
      case NodeKind::CALL_ORD:
      case NodeKind::CALL_CHR:
      case NodeKind::CALL_IS_LETTER:
      case NodeKind::CALL_IS_DIGIT:
      case NodeKind::CALL_IS_SPACE:
      case NodeKind::CALL_IS_VOWEL:
      case NodeKind::CALL_TO_LOWER:
      case NodeKind::CALL_TO_UPPER:
      case NodeKind::CALL_TO_STRING:
      case NodeKind::CALL_SINGLETON: {
        std::size_t next = idx + 1;
        const int argc = subtree::node_arity(node.kind);
        for (int i = 0; i < argc; ++i) {
          next = compile_expr_prefix(program, next);
        }
        const NodeDescriptor& descriptor = node_descriptor(node.kind);
        if (!descriptor.is_builtin() || descriptor.builtin_arity != argc) {
          throw std::runtime_error("prefix compile: invalid builtin descriptor");
        }
        emit(Opcode::CallBuiltin, descriptor.builtin_id, true, argc, true);
        return next;
      }
      case NodeKind::MAP_LIST:
        return compile_map_list_prefix(program, idx);
      case NodeKind::FILTER_LIST:
        return compile_filter_list_prefix(program, idx);
      case NodeKind::LINEAR_REC:
        return compile_linear_rec_prefix(program, idx);
      case NodeKind::ASGP_DC:
        return compile_asgp_dc_prefix(program, idx);
      case NodeKind::ASGP_DP1D:
        return compile_asgp_dp1d_prefix(program, idx);
      case NodeKind::ASGP_DP2D:
        return compile_asgp_dp2d_prefix(program, idx);
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
      emit(Opcode::Store, local(name_at(program, node.i0)), true);
      return next;
    }
    if (node.kind == NodeKind::RETURN) {
      const std::size_t next = compile_expr_prefix(program, idx + 1);
      emit(Opcode::Return);
      return next;
    }
    if (node.kind == NodeKind::IF_STMT) {
      const std::string else_label = new_label("if_else");
      const std::string end_label = new_label("if_end");
      std::size_t next = compile_expr_prefix(program, idx + 1);
      emit_jump(Opcode::JmpIfFalse, else_label);
      next = compile_block_prefix(program, next);
      emit_jump(Opcode::Jmp, end_label);
      mark_label(else_label);
      next = compile_block_prefix(program, next);
      mark_label(end_label);
      return next;
    }
    if (node.kind == NodeKind::FOR_RANGE) {
      const int bound_local = local(new_temp());
      const std::string valid_label = new_label("for_valid");
      const std::string bad_label = new_label("for_bad");
      std::size_t next = compile_expr_prefix(program, idx + 1);
      emit(Opcode::Store, bound_local, true);
      emit(Opcode::Load, bound_local, true);
      emit(Opcode::CallBuiltin, static_cast<int>(gagp::BuiltinId::IsInt), true, 1, true);
      emit_jump(Opcode::JmpIfFalse, bad_label);
      emit(Opcode::Load, bound_local, true);
      emit(Opcode::PushConst, add_const(Value::from_int(0)), true);
      emit(Opcode::Lt);
      emit_jump(Opcode::JmpIfFalse, valid_label);
      mark_label(bad_label);
      emit(Opcode::PushConst, add_const(Value::from_bool(true)), true);
      emit(Opcode::Neg);
      mark_label(valid_label);
      return compile_for_loop_body(name_at(program, node.i0), bound_local, program, next);
    }
    throw std::runtime_error("prefix compile: expected stmt node");
  }

  std::vector<Value> consts_;
  std::vector<Instr> code_;
  std::vector<UnresolvedJump> unresolved_;
  std::unordered_map<std::string, int> labels_;
  std::unordered_map<std::string, int> var2idx_;
  std::unordered_map<int, std::vector<int>> binder_stack_;
  std::vector<AsgpDcSegment> asgp_dc_segments_;
  std::vector<AsgpDp1dSegment> asgp_dp1d_segments_;
  std::vector<AsgpDp2dSegment> asgp_dp2d_segments_;
  int label_counter_ = 0;
  int tmp_counter_ = 0;
  const VerifiedAst* verified_ = nullptr;
};

}  // namespace

BytecodeProgram compile_for_eval(const ProgramGenome& genome,
                                 const std::vector<std::string>& preset_locals) {
  Compiler compiler(&preset_locals);
  return compiler.build(genome.ast);
}

BytecodeProgram compile_for_eval(const ProgramGenome& genome,
                                 const VerifiedAst& verified,
                                 const std::vector<std::string>& preset_locals) {
  Compiler compiler(&preset_locals, &verified);
  return compiler.build(genome.ast);
}

}  // namespace gagp::evo
