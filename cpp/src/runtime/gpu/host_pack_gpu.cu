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
  if (v.tag == ValueTag::NumList || v.tag == ValueTag::StringList) return kPayloadMaskList;
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
        bid == BuiltinId::Append || bid == BuiltinId::Reverse || bid == BuiltinId::Find ||
        bid == BuiltinId::Contains) {
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
    meta.case_offset = static_cast<int>(out.total_cases);
    meta.case_count = shared_case_count;
    meta.case_local_offset = 0;
    meta.is_valid = 1;
    meta.payload_flavor = classify_payload_flavor(prog, shared_input_payload_mask);
    meta.err_code = ErrCode::Value;

    out.all_consts.insert(out.all_consts.end(), prog.consts.begin(), prog.consts.end());
    meta.const_len = static_cast<int>(prog.consts.size());

    for (const Instr& ins : prog.code) {
      const int op = host_opcode(ins.op);
      if (op < 0) {
        meta.is_valid = 0;
        meta.err_code = ErrCode::Type;
        continue;
      }
      DInstr di;
      di.op = static_cast<std::uint8_t>(op);
      di.flags = static_cast<std::uint8_t>((ins.has_a ? DINSTR_HAS_A : 0) | (ins.has_b ? DINSTR_HAS_B : 0));
      di.a = static_cast<std::int32_t>(ins.a);
      di.b = static_cast<std::int32_t>(ins.b);
      out.all_code.push_back(di);
    }

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
  packed_case_local_vals->assign(shared_cases.size() * MAX_LOCALS, Value::none());
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
