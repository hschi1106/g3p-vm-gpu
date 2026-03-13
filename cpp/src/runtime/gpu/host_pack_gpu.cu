#include "g3pvm/runtime/gpu/host_pack_gpu.hpp"

#include <cstdint>

#include "g3pvm/core/builtin.hpp"
#include "g3pvm/runtime/gpu/constants_gpu.hpp"
#include "opcode_map_gpu.hpp"

namespace g3pvm::gpu_detail {

namespace {

bool value_has_payload(const Value& v) {
  return v.tag == ValueTag::String || v.tag == ValueTag::List;
}

bool program_has_payload_const(const BytecodeProgram& prog) {
  for (const Value& v : prog.consts) {
    if (value_has_payload(v)) {
      return true;
    }
  }
  return false;
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
    if (bid == BuiltinId::Concat || bid == BuiltinId::Slice || bid == BuiltinId::Index) {
      return true;
    }
  }
  return false;
}

bool program_needs_payload_fast_path(const BytecodeProgram& prog, bool shared_inputs_have_payload) {
  if (!program_has_exact_payload_builtin(prog)) {
    return false;
  }
  return shared_inputs_have_payload || program_has_payload_const(prog);
}

}  // namespace

PackResult pack_programs_with_shared_case_count(const std::vector<BytecodeProgram>& programs,
                                                int shared_case_count,
                                                bool shared_inputs_have_payload) {
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
    if (program_needs_payload_fast_path(prog, shared_inputs_have_payload)) {
      out.payload_program_indices.push_back(static_cast<int>(p));
      if (static_cast<std::size_t>(meta.code_len) > out.max_code_len_payload) {
        out.max_code_len_payload = static_cast<std::size_t>(meta.code_len);
      }
    } else {
      out.no_payload_program_indices.push_back(static_cast<int>(p));
      if (static_cast<std::size_t>(meta.code_len) > out.max_code_len_no_payload) {
        out.max_code_len_no_payload = static_cast<std::size_t>(meta.code_len);
      }
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
  if (d_no_payload_program_indices) cudaFree(d_no_payload_program_indices);
  if (d_payload_program_indices) cudaFree(d_payload_program_indices);
  if (d_string_payload_entries) cudaFree(d_string_payload_entries);
  if (d_string_payload_bytes) cudaFree(d_string_payload_bytes);
  if (d_list_payload_entries) cudaFree(d_list_payload_entries);
  if (d_list_payload_values) cudaFree(d_list_payload_values);
}

}  // namespace g3pvm::gpu_detail
