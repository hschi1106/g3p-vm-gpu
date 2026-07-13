#include "g3pvm/runtime/gpu/host_pack_gpu.hpp"

#include <cstdint>

#include "g3pvm/core/builtin.hpp"
#include "g3pvm/runtime/gpu/constants_gpu.hpp"
#include "opcode_map_gpu.hpp"

namespace g3pvm::gpu_detail {

namespace {

constexpr unsigned kPayloadMaskString = 1U << 0;
constexpr unsigned kPayloadMaskList = 1U << 1;

unsigned value_payload_mask(const Value& v) {
  if (v.tag == ValueTag::String) return kPayloadMaskString;
  if (v.tag == ValueTag::IntList || v.tag == ValueTag::FloatList || v.tag == ValueTag::StringList) {
    return kPayloadMaskList;
  }
  return 0U;
}

unsigned program_payload_const_mask(const BytecodeProgram& prog) {
  unsigned mask = 0U;
  for (const Value& v : prog.consts) {
    mask |= value_payload_mask(v);
  }
  return mask;
}

bool program_has_exact_payload_builtin(const BytecodeProgram& prog) {
  for (const Instr& ins : prog.code) {
    if (ins.op != Opcode::CallBuiltin || !ins.has_a) {
      continue;
    }
    BuiltinId bid = BuiltinId::Abs;
    if (!builtin_id_from_int(ins.a, bid)) {
      continue;
    }
    if (bid == BuiltinId::Concat || bid == BuiltinId::Slice || bid == BuiltinId::Index ||
        bid == BuiltinId::Append || bid == BuiltinId::Prepend || bid == BuiltinId::Reverse ||
        bid == BuiltinId::Find || bid == BuiltinId::Contains || bid == BuiltinId::CharToString ||
        bid == BuiltinId::StringToChar || bid == BuiltinId::Singleton) {
      return true;
    }
  }
  return false;
}

DPayloadFlavor classify_payload_flavor(const BytecodeProgram& prog, unsigned shared_input_payload_mask) {
  if (!program_has_exact_payload_builtin(prog)) {
    return DPayloadFlavor::None;
  }
  const unsigned payload_mask = shared_input_payload_mask | program_payload_const_mask(prog);
  if (payload_mask == 0U) {
    return DPayloadFlavor::None;
  }
  if (payload_mask == kPayloadMaskString) {
    return DPayloadFlavor::StringOnly;
  }
  if (payload_mask == kPayloadMaskList) {
    return DPayloadFlavor::ListOnly;
  }
  return DPayloadFlavor::Mixed;
}

DInstr pack_instr(const Instr& ins, bool* ok) {
  const int op = host_opcode(ins.op);
  if (op < 0) {
    *ok = false;
    return DInstr{};
  }
  DInstr di;
  di.op = static_cast<std::uint8_t>(op);
  di.flags = static_cast<std::uint8_t>((ins.has_a ? DINSTR_HAS_A : 0) | (ins.has_b ? DINSTR_HAS_B : 0));
  di.a = static_cast<std::int32_t>(ins.a);
  di.b = static_cast<std::int32_t>(ins.b);
  return di;
}

int phase_binder_local(const PhaseProgram& phase, int binder_name) {
  const auto it = phase.binder_locals.find(binder_name);
  if (it == phase.binder_locals.end()) {
    return -1;
  }
  return it->second;
}

DPhaseMeta append_phase(const PhaseProgram& phase,
                        std::vector<DInstr>* all_phase_code,
                        std::vector<Value>* all_phase_consts,
                        bool* ok) {
  DPhaseMeta meta;
  meta.code_offset = static_cast<int>(all_phase_code->size());
  meta.const_offset = static_cast<int>(all_phase_consts->size());
  meta.n_locals = phase.n_locals;
  all_phase_consts->insert(all_phase_consts->end(), phase.consts.begin(), phase.consts.end());
  meta.const_len = static_cast<int>(phase.consts.size());
  for (const Instr& ins : phase.code) {
    all_phase_code->push_back(pack_instr(ins, ok));
  }
  meta.code_len = static_cast<int>(all_phase_code->size()) - meta.code_offset;
  if (phase.n_locals < 0 || phase.n_locals > MAX_LOCALS) {
    *ok = false;
  }
  return meta;
}

}  // namespace

DPayloadFlavor classify_payload_flavor_for_program(const BytecodeProgram& prog, unsigned shared_input_payload_mask) {
  return classify_payload_flavor(prog, shared_input_payload_mask);
}

PackResult pack_programs_with_shared_case_count(const std::vector<BytecodeProgram>& programs,
                                                int shared_case_count,
                                                unsigned shared_input_payload_mask) {
  PackResult out;
  out.metas.resize(programs.size());

  for (std::size_t p = 0; p < programs.size(); ++p) {
    const BytecodeProgram& prog = programs[p];

    DProgramMeta meta;
    meta.code_offset = static_cast<int>(out.all_code.size());
    meta.const_offset = static_cast<int>(out.all_consts.size());
    meta.n_locals = prog.n_locals;
    meta.asgp_dc_offset = static_cast<int>(out.asgp_dc_segments.size());
    meta.asgp_dp1d_offset = static_cast<int>(out.asgp_dp1d_segments.size());
    meta.asgp_dp2d_offset = static_cast<int>(out.asgp_dp2d_segments.size());
    meta.case_offset = static_cast<int>(out.total_cases);
    meta.case_count = shared_case_count;
    meta.case_local_offset = 0;
    meta.is_valid = 1;
    meta.payload_flavor = classify_payload_flavor(prog, shared_input_payload_mask);
    meta.err_code = ErrCode::Value;

    out.all_consts.insert(out.all_consts.end(), prog.consts.begin(), prog.consts.end());
    meta.const_len = static_cast<int>(prog.consts.size());

    for (const Instr& ins : prog.code) {
      bool ok = true;
      out.all_code.push_back(pack_instr(ins, &ok));
      if (!ok) {
        meta.is_valid = 0;
        meta.err_code = ErrCode::Type;
      }
    }

    for (const AsgpDcSegment& segment : prog.asgp_dc_segments) {
      bool ok = true;
      DAsgpDcSegment packed_segment;
      packed_segment.solve_xs_local = phase_binder_local(segment.solve, segment.solve_xs_name);
      packed_segment.solve_n_local = phase_binder_local(segment.solve, segment.solve_n_name);
      packed_segment.solve_lo_local = phase_binder_local(segment.solve, segment.solve_lo_name);
      packed_segment.divide_n_local = phase_binder_local(segment.divide, segment.divide_n_name);
      packed_segment.combine_left_local = phase_binder_local(segment.combine, segment.combine_left_name);
      packed_segment.combine_right_local = phase_binder_local(segment.combine, segment.combine_right_name);
      packed_segment.solve = append_phase(segment.solve, &out.all_phase_code, &out.all_phase_consts, &ok);
      packed_segment.divide = append_phase(segment.divide, &out.all_phase_code, &out.all_phase_consts, &ok);
      packed_segment.combine = append_phase(segment.combine, &out.all_phase_code, &out.all_phase_consts, &ok);
      if (packed_segment.solve_xs_local < 0 || packed_segment.solve_n_local < 0 ||
          packed_segment.solve_lo_local < 0 || packed_segment.divide_n_local < 0 ||
          packed_segment.combine_left_local < 0 || packed_segment.combine_right_local < 0) {
        ok = false;
      }
      if (!ok) {
        meta.is_valid = 0;
        meta.err_code = ErrCode::Name;
      }
      out.asgp_dc_segments.push_back(packed_segment);
    }
    meta.asgp_dc_count = static_cast<int>(out.asgp_dc_segments.size()) - meta.asgp_dc_offset;

    for (const AsgpDp1dSegment& segment : prog.asgp_dp1d_segments) {
      bool ok = true;
      DAsgpDp1dSegment packed_segment;
      packed_segment.lo = segment.lo;
      packed_segment.hi = segment.hi;
      packed_segment.base_state = segment.base_state;
      packed_segment.boundary_value = segment.boundary_value;
      packed_segment.dep_kind = segment.dep_kind;
      if (segment.dep_offsets.size() > static_cast<std::size_t>(DMAX_ASGP_DP_DEPS) ||
          segment.transition_dep_names.size() > static_cast<std::size_t>(DMAX_ASGP_DP_DEPS)) {
        ok = false;
      }
      packed_segment.dep_offset_count = static_cast<int>(
          segment.dep_offsets.size() <= static_cast<std::size_t>(DMAX_ASGP_DP_DEPS)
              ? segment.dep_offsets.size()
              : static_cast<std::size_t>(DMAX_ASGP_DP_DEPS));
      for (int i = 0; i < packed_segment.dep_offset_count; ++i) {
        packed_segment.dep_offsets[i] = segment.dep_offsets[static_cast<std::size_t>(i)];
      }
      packed_segment.solve_state_local = phase_binder_local(segment.solve, segment.solve_state_name);
      packed_segment.transition_state_local =
          phase_binder_local(segment.transition, segment.transition_state_name);
      packed_segment.transition_dep_count = static_cast<int>(
          segment.transition_dep_names.size() <= static_cast<std::size_t>(DMAX_ASGP_DP_DEPS)
              ? segment.transition_dep_names.size()
              : static_cast<std::size_t>(DMAX_ASGP_DP_DEPS));
      for (int i = 0; i < packed_segment.transition_dep_count; ++i) {
        packed_segment.transition_dep_locals[i] =
            phase_binder_local(segment.transition, segment.transition_dep_names[static_cast<std::size_t>(i)]);
        if (packed_segment.transition_dep_locals[i] < 0) {
          ok = false;
        }
      }
      packed_segment.solve = append_phase(segment.solve, &out.all_phase_code, &out.all_phase_consts, &ok);
      packed_segment.transition =
          append_phase(segment.transition, &out.all_phase_code, &out.all_phase_consts, &ok);
      if (packed_segment.solve_state_local < 0 || packed_segment.transition_state_local < 0) {
        ok = false;
      }
      if (!ok) {
        meta.is_valid = 0;
        meta.err_code = ErrCode::Name;
      }
      out.asgp_dp1d_segments.push_back(packed_segment);
    }
    meta.asgp_dp1d_count = static_cast<int>(out.asgp_dp1d_segments.size()) - meta.asgp_dp1d_offset;

    for (const AsgpDp2dSegment& segment : prog.asgp_dp2d_segments) {
      bool ok = true;
      DAsgpDp2dSegment packed_segment;
      packed_segment.i_lo = segment.i_lo;
      packed_segment.i_hi = segment.i_hi;
      packed_segment.j_lo = segment.j_lo;
      packed_segment.j_hi = segment.j_hi;
      packed_segment.base_i = segment.base_i;
      packed_segment.base_j = segment.base_j;
      packed_segment.boundary_value = segment.boundary_value;
      packed_segment.dep_kind = segment.dep_kind;
      if (segment.transition_dep_names.size() > static_cast<std::size_t>(DMAX_ASGP_DP_DEPS)) {
        ok = false;
      }
      packed_segment.solve_i_local = phase_binder_local(segment.solve, segment.solve_i_name);
      packed_segment.solve_j_local = phase_binder_local(segment.solve, segment.solve_j_name);
      packed_segment.transition_i_local =
          phase_binder_local(segment.transition, segment.transition_i_name);
      packed_segment.transition_j_local =
          phase_binder_local(segment.transition, segment.transition_j_name);
      packed_segment.transition_dep_count = static_cast<int>(
          segment.transition_dep_names.size() <= static_cast<std::size_t>(DMAX_ASGP_DP_DEPS)
              ? segment.transition_dep_names.size()
              : static_cast<std::size_t>(DMAX_ASGP_DP_DEPS));
      for (int i = 0; i < packed_segment.transition_dep_count; ++i) {
        packed_segment.transition_dep_locals[i] =
            phase_binder_local(segment.transition, segment.transition_dep_names[static_cast<std::size_t>(i)]);
        if (packed_segment.transition_dep_locals[i] < 0) {
          ok = false;
        }
      }
      packed_segment.solve = append_phase(segment.solve, &out.all_phase_code, &out.all_phase_consts, &ok);
      packed_segment.transition =
          append_phase(segment.transition, &out.all_phase_code, &out.all_phase_consts, &ok);
      if (packed_segment.solve_i_local < 0 || packed_segment.solve_j_local < 0 ||
          packed_segment.transition_i_local < 0 || packed_segment.transition_j_local < 0) {
        ok = false;
      }
      if (!ok) {
        meta.is_valid = 0;
        meta.err_code = ErrCode::Name;
      }
      out.asgp_dp2d_segments.push_back(packed_segment);
    }
    meta.asgp_dp2d_count = static_cast<int>(out.asgp_dp2d_segments.size()) - meta.asgp_dp2d_offset;

    meta.code_len = static_cast<int>(out.all_code.size()) - meta.code_offset;
    if (static_cast<std::size_t>(meta.code_len) > out.max_code_len) {
      out.max_code_len = static_cast<std::size_t>(meta.code_len);
    }
    if (prog.n_locals < 0 || prog.n_locals > MAX_LOCALS) {
      meta.is_valid = 0;
      meta.err_code = ErrCode::Value;
    }

    out.total_cases += static_cast<std::size_t>(shared_case_count);
    out.metas[p] = meta;
  }

  return out;
}

void pack_shared_cases_only(const std::vector<CaseBindings>& shared_cases,
                            std::vector<Value>* packed_case_local_vals,
                            std::vector<unsigned char>* packed_case_local_set) {
  packed_case_local_vals->assign(shared_cases.size() * MAX_LOCALS, Value::invalid());
  packed_case_local_set->assign(shared_cases.size() * MAX_LOCALS, 0);
  for (std::size_t case_idx = 0; case_idx < shared_cases.size(); ++case_idx) {
    const std::size_t base = case_idx * MAX_LOCALS;
    for (const InputBinding& binding : shared_cases[case_idx]) {
      if (binding.idx >= 0 && binding.idx < MAX_LOCALS) {
        (*packed_case_local_vals)[base + static_cast<std::size_t>(binding.idx)] = binding.value;
        (*packed_case_local_set)[base + static_cast<std::size_t>(binding.idx)] = 1;
      }
    }
  }
}

DeviceArena::~DeviceArena() {
  if (d_consts) cudaFree(d_consts);
  if (d_code) cudaFree(d_code);
  if (d_phase_consts) cudaFree(d_phase_consts);
  if (d_phase_code) cudaFree(d_phase_code);
  if (d_asgp_dc_segments) cudaFree(d_asgp_dc_segments);
  if (d_asgp_dp1d_segments) cudaFree(d_asgp_dp1d_segments);
  if (d_asgp_dp2d_segments) cudaFree(d_asgp_dp2d_segments);
  if (d_metas) cudaFree(d_metas);
  if (d_shared_case_local_vals) cudaFree(d_shared_case_local_vals);
  if (d_shared_case_local_set) cudaFree(d_shared_case_local_set);
  if (d_out) cudaFree(d_out);
  if (d_expected) cudaFree(d_expected);
  if (d_fitness) cudaFree(d_fitness);
  if (d_string_payload_entries) cudaFree(d_string_payload_entries);
  if (d_string_payload_bytes) cudaFree(d_string_payload_bytes);
  if (d_list_payload_entries) cudaFree(d_list_payload_entries);
  if (d_list_payload_values) cudaFree(d_list_payload_values);
}

}  // namespace g3pvm::gpu_detail
