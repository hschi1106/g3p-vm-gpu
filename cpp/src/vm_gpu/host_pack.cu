#include "host_pack.hpp"

#include <cstdint>

#include "constants.hpp"
#include "opcode_map.hpp"

namespace g3pvm::gpu_detail {

PackResult pack_programs_and_shared_cases(const std::vector<BytecodeProgram>& programs,
                                          const std::vector<InputCase>& shared_cases) {
  PackResult out;
  out.metas.resize(programs.size());

  const int shared_case_count = static_cast<int>(shared_cases.size());
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
    meta.err_code = DERR_VALUE;

    out.all_consts.insert(out.all_consts.end(), prog.consts.begin(), prog.consts.end());
    meta.const_len = static_cast<int>(prog.consts.size());

    for (const Instr& ins : prog.code) {
      const int op = host_opcode(ins.op);
      if (op < 0) {
        meta.is_valid = 0;
        meta.err_code = DERR_TYPE;
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
      meta.err_code = DERR_VALUE;
    }

    out.total_cases += static_cast<std::size_t>(shared_case_count);
    out.metas[p] = meta;
  }

  out.packed_case_local_vals.assign(shared_cases.size() * MAX_LOCALS, Value::none());
  out.packed_case_local_set.assign(shared_cases.size() * MAX_LOCALS, 0);
  for (std::size_t case_idx = 0; case_idx < shared_cases.size(); ++case_idx) {
    const std::size_t base = case_idx * MAX_LOCALS;
    for (const LocalBinding& binding : shared_cases[case_idx]) {
      if (binding.idx >= 0 && binding.idx < MAX_LOCALS) {
        out.packed_case_local_vals[base + static_cast<std::size_t>(binding.idx)] = binding.value;
        out.packed_case_local_set[base + static_cast<std::size_t>(binding.idx)] = 1;
      }
    }
  }

  return out;
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
}

}  // namespace g3pvm::gpu_detail
