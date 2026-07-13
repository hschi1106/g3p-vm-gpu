#pragma once

#include <cstdint>

#include "g3pvm/core/builtin.hpp"
#include "builtins_device.cuh"
#include "g3pvm/core/value_semantics.hpp"
#include "g3pvm/runtime/gpu/device_types_gpu.hpp"

namespace g3pvm::gpu_detail {

__device__ inline void d_fail(DResult& out, ErrCode code) {
  out.is_error = 1;
  out.err_code = code;
}

__device__ inline ValueTag d_list_tag_from_private_code(int code) {
  if (code == 1) return ValueTag::IntList;
  if (code == 2) return ValueTag::FloatList;
  if (code == 3) return ValueTag::StringList;
  return ValueTag::Invalid;
}

template <typename State>
__device__ inline bool d_make_empty_list_for_tag(ValueTag tag, State& st, Value& out) {
  if (!d_is_typed_list_tag(tag)) {
    return false;
  }
  const std::uint64_t h = d_hash_list_payload(nullptr, 0);
  out = d_make_list_hash_len(tag, h, 0U);
  if constexpr (State::kMaxListEntries > 0) {
    const int off = st.list_values_used;
    (void)d_register_local_list(st, out, off, 0);
  }
  return true;
}

template <DPayloadFlavor Flavor>
struct DPayloadStateStorage {
  typename DPayloadFlavorTraits<Flavor>::State state{};

  __device__ typename DPayloadFlavorTraits<Flavor>::State& ref() { return state; }
};

struct DAsgpTables {
  const DInstr* phase_code = nullptr;
  const Value* phase_consts = nullptr;
  const DAsgpDcSegment* dc_segments = nullptr;
  int dc_segment_count = 0;
  const DAsgpDp1dSegment* dp1d_segments = nullptr;
  int dp1d_segment_count = 0;
  const DAsgpDp2dSegment* dp2d_segments = nullptr;
  int dp2d_segment_count = 0;
};

struct DCodeView {
  const DInstr* code = nullptr;
  int code_len = 0;
  const Value* consts = nullptr;
  int const_len = 0;
  int n_locals = 0;
  int asgp_dc_offset = 0;
  int asgp_dc_count = 0;
  int asgp_dp1d_offset = 0;
  int asgp_dp1d_count = 0;
  int asgp_dp2d_offset = 0;
  int asgp_dp2d_count = 0;
};

struct DLocalPreset {
  int local = -1;
  Value value = Value::invalid();
};

__device__ inline DResult d_ok(Value value) {
  DResult out;
  out.is_error = 0;
  out.err_code = ErrCode::Value;
  out.value = value;
  return out;
}

__device__ inline DResult d_error(ErrCode code) {
  DResult out;
  d_fail(out, code);
  return out;
}

__device__ inline bool d_is_asgp_dc_source(const Value& v) {
  return v.tag == ValueTag::String || v.tag == ValueTag::IntList ||
         v.tag == ValueTag::FloatList || v.tag == ValueTag::StringList;
}

__device__ inline bool d_is_payload_value(const Value& v) {
  return v.tag == ValueTag::String || v.tag == ValueTag::IntList ||
         v.tag == ValueTag::FloatList || v.tag == ValueTag::StringList ||
         v.tag == ValueTag::FallbackToken;
}

template <DPayloadFlavor Flavor, bool EnableAsgp>
__device__ __noinline__ DResult d_run_code_core(const DCodeView& view,
                                                const Value* shared_case_local_vals,
                                                const unsigned char* shared_case_local_set,
                                                int local_case,
                                                const DLocalPreset* presets,
                                                int preset_count,
                                                const DPayloadTables& payload_tables,
                                                typename DPayloadFlavorTraits<Flavor>::State& payload_state,
                                                const DAsgpTables& asgp_tables,
                                                int& fuel_left,
                                                bool require_return);

template <DPayloadFlavor Flavor>
__device__ inline DResult d_run_asgp_phase(const DPhaseMeta& phase,
                                           const DLocalPreset* presets,
                                           int preset_count,
                                           int asgp_dc_offset,
                                           int asgp_dc_count,
                                           const DPayloadTables& payload_tables,
                                           typename DPayloadFlavorTraits<Flavor>::State& payload_state,
                                           const DAsgpTables& asgp_tables,
                                           int& fuel_left) {
  const DCodeView view{
      asgp_tables.phase_code + phase.code_offset,
      phase.code_len,
      asgp_tables.phase_consts + phase.const_offset,
      phase.const_len,
      phase.n_locals,
      asgp_dc_offset,
      asgp_dc_count,
      0,
      0,
      0,
      0,
  };
  return d_run_code_core<Flavor, false>(view, nullptr, nullptr, 0, presets, preset_count,
                                        payload_tables, payload_state, asgp_tables, fuel_left, false);
}

struct DAsgpDcFrame {
  Value source = Value::invalid();
  long long lo = 0;
  int state = 0;
  int split = 0;
  Value left = Value::invalid();
};

struct DAsgpDp1dFrame {
  long long state = 0;
  int next_dep = 0;
  Value dep_values[DMAX_ASGP_DP_DEPS];
};

struct DAsgpDp2dCell {
  long long i = 0;
  long long j = 0;
};

struct DAsgpDp2dFrame {
  long long i = 0;
  long long j = 0;
  int next_dep = 0;
  Value dep_values[DMAX_ASGP_DP_DEPS];
};

__device__ inline int d_asgp_dp1d_memo_find(const long long* keys, const Value* values, int count, long long state) {
  for (int i = 0; i < count; ++i) {
    if (keys[i] == state) {
      return i;
    }
  }
  return -1;
}

__device__ inline long long d_asgp_dp1d_dep_state(const DAsgpDp1dSegment& segment, long long state, int dep_idx) {
  const long long offset = static_cast<long long>(segment.dep_offsets[dep_idx]);
  return (segment.dep_kind < 0) ? (state - offset) : (state + offset);
}

__device__ inline std::uint64_t d_asgp_dp2d_key(long long i, long long j) {
  const auto i32 = static_cast<std::uint32_t>(i);
  const auto j32 = static_cast<std::uint32_t>(j);
  return (static_cast<std::uint64_t>(i32) << 32) | static_cast<std::uint64_t>(j32);
}

__device__ inline int d_asgp_dp2d_memo_find(const std::uint64_t* keys,
                                            const Value* values,
                                            int count,
                                            std::uint64_t key) {
  for (int i = 0; i < count; ++i) {
    if (keys[i] == key) {
      return i;
    }
  }
  return -1;
}

__device__ inline int d_asgp_dp2d_deps(const DAsgpDp2dSegment& segment,
                                       long long i,
                                       long long j,
                                       DAsgpDp2dCell* out) {
  switch (segment.dep_kind) {
    case 0:
      out[0] = DAsgpDp2dCell{i - 1, j};
      out[1] = DAsgpDp2dCell{i, j - 1};
      return 2;
    case 1:
      out[0] = DAsgpDp2dCell{i + 1, j};
      out[1] = DAsgpDp2dCell{i, j + 1};
      return 2;
    case 2:
      out[0] = DAsgpDp2dCell{i - 1, j - 1};
      return 1;
    case 3:
      out[0] = DAsgpDp2dCell{i + 1, j + 1};
      return 1;
    case 4:
      out[0] = DAsgpDp2dCell{i - 1, j};
      out[1] = DAsgpDp2dCell{i, j - 1};
      out[2] = DAsgpDp2dCell{i - 1, j - 1};
      return 3;
    case 5:
      out[0] = DAsgpDp2dCell{i + 1, j};
      out[1] = DAsgpDp2dCell{i, j + 1};
      out[2] = DAsgpDp2dCell{i + 1, j + 1};
      return 3;
    default:
      return 0;
  }
}

template <DPayloadFlavor Flavor>
__device__ __noinline__ DResult d_eval_asgp_dc(const DAsgpDcSegment& segment,
                                               const Value& source,
                                               long long lo,
                                               int asgp_dc_offset,
                                               int asgp_dc_count,
                                               const DPayloadTables& payload_tables,
                                               typename DPayloadFlavorTraits<Flavor>::State& payload_state,
                                               const DAsgpTables& asgp_tables,
                                               int& fuel_left) {
  constexpr int kMaxAsgpDcFrames = 64;
  DAsgpDcFrame frames[kMaxAsgpDcFrames];
  int depth = 1;
  frames[0].source = source;
  frames[0].lo = lo;
  frames[0].state = 0;

  while (depth > 0) {
    DAsgpDcFrame& frame = frames[depth - 1];
    if (frame.state == 0) {
      if (fuel_left <= 0) {
        return d_error(ErrCode::Timeout);
      }
      fuel_left -= 1;
      if (!d_is_asgp_dc_source(frame.source)) {
        return d_error(ErrCode::Type);
      }
      const int n = static_cast<int>(Value::container_len(frame.source));
      if (n <= 1) {
        const DLocalPreset presets[3] = {
            DLocalPreset{segment.solve_xs_local, frame.source},
            DLocalPreset{segment.solve_n_local, Value::from_int(n)},
            DLocalPreset{segment.solve_lo_local, Value::from_int(frame.lo)},
        };
        DResult solved = d_run_asgp_phase<Flavor>(segment.solve, presets, 3, asgp_dc_offset,
                                                  asgp_dc_count, payload_tables, payload_state,
                                                  asgp_tables, fuel_left);
        if (solved.is_error) {
          return solved;
        }
        Value completed = solved.value;
        depth -= 1;
        while (true) {
          if (depth == 0) {
            return d_ok(completed);
          }
          DAsgpDcFrame& parent = frames[depth - 1];
          if (parent.state == 1) {
            parent.left = completed;
            parent.state = 2;
            const int parent_n = static_cast<int>(Value::container_len(parent.source));
            Value slice_args[3] = {parent.source, Value::from_int(parent.split), Value::from_int(parent_n)};
            Value right_source = Value::invalid();
            ErrCode derr = ErrCode::Type;
            if (!d_builtin_call<Flavor>(
                    BuiltinId::Slice, slice_args, 3, payload_tables, payload_state, right_source, derr)) {
              return d_error(derr);
            }
            if (depth >= kMaxAsgpDcFrames) {
              return d_error(ErrCode::Timeout);
            }
            frames[depth].source = right_source;
            frames[depth].lo = parent.lo + parent.split;
            frames[depth].state = 0;
            depth += 1;
            break;
          }
          if (parent.state != 2) {
            return d_error(ErrCode::Value);
          }
          if (parent.left.tag != completed.tag) {
            return d_error(ErrCode::Type);
          }
          const DLocalPreset combine_presets[2] = {
              DLocalPreset{segment.combine_left_local, parent.left},
              DLocalPreset{segment.combine_right_local, completed},
          };
          DResult combined = d_run_asgp_phase<Flavor>(segment.combine, combine_presets, 2,
                                                      asgp_dc_offset, asgp_dc_count, payload_tables,
                                                      payload_state, asgp_tables, fuel_left);
          if (combined.is_error) {
            return combined;
          }
          if (combined.value.tag != parent.left.tag) {
            return d_error(ErrCode::Type);
          }
          completed = combined.value;
          depth -= 1;
        }
        continue;
      }

      const DLocalPreset divide_presets[1] = {
          DLocalPreset{segment.divide_n_local, Value::from_int(n)},
      };
      DResult raw_split = d_run_asgp_phase<Flavor>(segment.divide, divide_presets, 1,
                                                   asgp_dc_offset, asgp_dc_count, payload_tables,
                                                   payload_state, asgp_tables, fuel_left);
      if (raw_split.is_error) {
        return raw_split;
      }
      if (raw_split.value.tag != ValueTag::Int) {
        return d_error(ErrCode::Type);
      }
      long long split_ll = raw_split.value.i;
      if (split_ll < 1) split_ll = 1;
      if (split_ll > n - 1) split_ll = n - 1;
      frame.split = static_cast<int>(split_ll);
      frame.state = 1;

      Value slice_args[3] = {frame.source, Value::from_int(0), Value::from_int(frame.split)};
      Value left_source = Value::invalid();
      ErrCode derr = ErrCode::Type;
      if (!d_builtin_call<Flavor>(BuiltinId::Slice, slice_args, 3, payload_tables, payload_state, left_source, derr)) {
        return d_error(derr);
      }
      if (depth >= kMaxAsgpDcFrames) {
        return d_error(ErrCode::Timeout);
      }
      frames[depth].source = left_source;
      frames[depth].lo = frame.lo;
      frames[depth].state = 0;
      depth += 1;
      continue;
    }
    return d_error(ErrCode::Value);
  }

  return d_error(ErrCode::Value);
}

template <DPayloadFlavor Flavor>
__device__ __noinline__ DResult d_eval_asgp_dp2d(const DAsgpDp2dSegment& segment,
                                                 long long root_i,
                                                 long long root_j,
                                                 const DPayloadTables& payload_tables,
                                                 typename DPayloadFlavorTraits<Flavor>::State& payload_state,
                                                 const DAsgpTables& asgp_tables,
                                                 int& fuel_left) {
  constexpr int kMaxFrames = 128;
  constexpr int kMaxMemo = 128;
  if (segment.transition_dep_count < 0 || segment.transition_dep_count > DMAX_ASGP_DP_DEPS) {
    return d_error(ErrCode::Value);
  }

  DAsgpDp2dFrame frames[kMaxFrames];
  int depth = 1;
  frames[0].i = root_i;
  frames[0].j = root_j;
  frames[0].next_dep = 0;
  std::uint64_t memo_keys[kMaxMemo];
  Value memo_values[kMaxMemo];
  int memo_count = 0;

  while (depth > 0) {
    DAsgpDp2dFrame& frame = frames[depth - 1];
    Value completed = Value::invalid();
    bool has_completed = false;

    if (frame.next_dep == 0) {
      if (fuel_left <= 0) {
        return d_error(ErrCode::Timeout);
      }
      fuel_left -= 1;

      if (frame.i < segment.i_lo || frame.i > segment.i_hi ||
          frame.j < segment.j_lo || frame.j > segment.j_hi) {
        completed = segment.boundary_value;
        has_completed = true;
      } else if (frame.i == segment.base_i && frame.j == segment.base_j) {
        const DLocalPreset presets[2] = {
            DLocalPreset{segment.solve_i_local, Value::from_int(frame.i)},
            DLocalPreset{segment.solve_j_local, Value::from_int(frame.j)},
        };
        DResult solved = d_run_asgp_phase<Flavor>(segment.solve, presets, 2, 0, 0,
                                                  payload_tables, payload_state, asgp_tables, fuel_left);
        if (solved.is_error) {
          return solved;
        }
        completed = solved.value;
        has_completed = true;
      } else {
        const std::uint64_t key = d_asgp_dp2d_key(frame.i, frame.j);
        const int cached = d_asgp_dp2d_memo_find(memo_keys, memo_values, memo_count, key);
        if (cached >= 0) {
          completed = memo_values[cached];
          has_completed = true;
        } else {
          DAsgpDp2dCell dep_cells[DMAX_ASGP_DP_DEPS];
          const int dep_count = d_asgp_dp2d_deps(segment, frame.i, frame.j, dep_cells);
          if (dep_count == 0) {
            const DLocalPreset presets[2] = {
                DLocalPreset{segment.transition_i_local, Value::from_int(frame.i)},
                DLocalPreset{segment.transition_j_local, Value::from_int(frame.j)},
            };
            DResult out = d_run_asgp_phase<Flavor>(segment.transition, presets, 2, 0, 0,
                                                   payload_tables, payload_state, asgp_tables, fuel_left);
            if (out.is_error) {
              return out;
            }
            if (memo_count >= kMaxMemo) {
              return d_error(ErrCode::Timeout);
            }
            memo_keys[memo_count] = key;
            memo_values[memo_count] = out.value;
            memo_count += 1;
            completed = out.value;
            has_completed = true;
          } else {
            if (depth >= kMaxFrames) {
              return d_error(ErrCode::Timeout);
            }
            frames[depth].i = dep_cells[0].i;
            frames[depth].j = dep_cells[0].j;
            frames[depth].next_dep = 0;
            depth += 1;
            continue;
          }
        }
      }
    } else {
      return d_error(ErrCode::Value);
    }

    if (!has_completed) {
      return d_error(ErrCode::Value);
    }

    depth -= 1;
    while (true) {
      if (depth == 0) {
        return d_ok(completed);
      }

      DAsgpDp2dFrame& parent = frames[depth - 1];
      DAsgpDp2dCell dep_cells[DMAX_ASGP_DP_DEPS];
      const int dep_count = d_asgp_dp2d_deps(segment, parent.i, parent.j, dep_cells);
      const int completed_dep_idx = parent.next_dep;
      if (completed_dep_idx < 0 || completed_dep_idx >= dep_count) {
        return d_error(ErrCode::Value);
      }
      if (completed_dep_idx > 0 && completed.tag != parent.dep_values[0].tag) {
        return d_error(ErrCode::Type);
      }
      parent.dep_values[completed_dep_idx] = completed;
      parent.next_dep += 1;

      if (parent.next_dep < dep_count) {
        if (depth >= kMaxFrames) {
          return d_error(ErrCode::Timeout);
        }
        frames[depth].i = dep_cells[parent.next_dep].i;
        frames[depth].j = dep_cells[parent.next_dep].j;
        frames[depth].next_dep = 0;
        depth += 1;
        break;
      }

      DLocalPreset presets[2 + DMAX_ASGP_DP_DEPS];
      presets[0] = DLocalPreset{segment.transition_i_local, Value::from_int(parent.i)};
      presets[1] = DLocalPreset{segment.transition_j_local, Value::from_int(parent.j)};
      const int bind_dep_count =
          (dep_count < segment.transition_dep_count) ? dep_count : segment.transition_dep_count;
      for (int dep_idx = 0; dep_idx < bind_dep_count; ++dep_idx) {
        presets[2 + dep_idx] =
            DLocalPreset{segment.transition_dep_locals[dep_idx], parent.dep_values[dep_idx]};
      }
      DResult out = d_run_asgp_phase<Flavor>(segment.transition, presets, 2 + bind_dep_count, 0, 0,
                                             payload_tables, payload_state, asgp_tables, fuel_left);
      if (out.is_error) {
        return out;
      }
      if (dep_count > 0 && out.value.tag != parent.dep_values[0].tag) {
        return d_error(ErrCode::Type);
      }
      if (memo_count >= kMaxMemo) {
        return d_error(ErrCode::Timeout);
      }
      memo_keys[memo_count] = d_asgp_dp2d_key(parent.i, parent.j);
      memo_values[memo_count] = out.value;
      memo_count += 1;
      completed = out.value;
      depth -= 1;
    }
  }

  return d_error(ErrCode::Value);
}

template <DPayloadFlavor Flavor>
__device__ __noinline__ DResult d_eval_asgp_dp1d(const DAsgpDp1dSegment& segment,
                                                 long long root_state,
                                                 const DPayloadTables& payload_tables,
                                                 typename DPayloadFlavorTraits<Flavor>::State& payload_state,
                                                 const DAsgpTables& asgp_tables,
                                                 int& fuel_left) {
  constexpr int kMaxFrames = 128;
  constexpr int kMaxMemo = 128;
  if (segment.dep_offset_count < 0 || segment.dep_offset_count > DMAX_ASGP_DP_DEPS ||
      segment.transition_dep_count < 0 || segment.transition_dep_count > DMAX_ASGP_DP_DEPS) {
    return d_error(ErrCode::Value);
  }

  DAsgpDp1dFrame frames[kMaxFrames];
  int depth = 1;
  frames[0].state = root_state;
  frames[0].next_dep = 0;
  long long memo_keys[kMaxMemo];
  Value memo_values[kMaxMemo];
  int memo_count = 0;

  while (depth > 0) {
    DAsgpDp1dFrame& frame = frames[depth - 1];
    if (frame.next_dep == 0) {
      if (fuel_left <= 0) {
        return d_error(ErrCode::Timeout);
      }
      fuel_left -= 1;
      if (frame.state < segment.lo || frame.state > segment.hi) {
        Value completed = segment.boundary_value;
        depth -= 1;
        while (true) {
          if (depth == 0) {
            return d_ok(completed);
          }
          DAsgpDp1dFrame& parent = frames[depth - 1];
          const int completed_dep_idx = parent.next_dep;
          if (completed_dep_idx < 0 || completed_dep_idx >= segment.dep_offset_count) {
            return d_error(ErrCode::Value);
          }
          if (completed_dep_idx > 0 && completed.tag != parent.dep_values[0].tag) {
            return d_error(ErrCode::Type);
          }
          parent.dep_values[completed_dep_idx] = completed;
          parent.next_dep += 1;
          if (parent.next_dep < segment.dep_offset_count) {
            if (depth >= kMaxFrames) {
              return d_error(ErrCode::Timeout);
            }
            frames[depth].state = d_asgp_dp1d_dep_state(segment, parent.state, parent.next_dep);
            frames[depth].next_dep = 0;
            depth += 1;
            break;
          }

          DLocalPreset presets[1 + DMAX_ASGP_DP_DEPS];
          presets[0] = DLocalPreset{segment.transition_state_local, Value::from_int(parent.state)};
          const int bind_dep_count =
              (segment.dep_offset_count < segment.transition_dep_count)
                  ? segment.dep_offset_count
                  : segment.transition_dep_count;
          for (int i = 0; i < bind_dep_count; ++i) {
            presets[1 + i] = DLocalPreset{segment.transition_dep_locals[i], parent.dep_values[i]};
          }
          DResult out = d_run_asgp_phase<Flavor>(
              segment.transition, presets, 1 + bind_dep_count, 0, 0,
              payload_tables, payload_state, asgp_tables, fuel_left);
          if (out.is_error) {
            return out;
          }
          if (segment.dep_offset_count > 0 && out.value.tag != parent.dep_values[0].tag) {
            return d_error(ErrCode::Type);
          }
          if (memo_count >= kMaxMemo) {
            return d_error(ErrCode::Timeout);
          }
          memo_keys[memo_count] = parent.state;
          memo_values[memo_count] = out.value;
          memo_count += 1;
          completed = out.value;
          depth -= 1;
        }
        continue;
      }

      if (frame.state == segment.base_state) {
        const DLocalPreset presets[1] = {
            DLocalPreset{segment.solve_state_local, Value::from_int(frame.state)},
        };
        DResult solved = d_run_asgp_phase<Flavor>(segment.solve, presets, 1, 0, 0,
                                                  payload_tables, payload_state, asgp_tables, fuel_left);
        if (solved.is_error) {
          return solved;
        }
        Value completed = solved.value;
        depth -= 1;
        while (true) {
          if (depth == 0) {
            return d_ok(completed);
          }
          DAsgpDp1dFrame& parent = frames[depth - 1];
          const int completed_dep_idx = parent.next_dep;
          if (completed_dep_idx < 0 || completed_dep_idx >= segment.dep_offset_count) {
            return d_error(ErrCode::Value);
          }
          if (completed_dep_idx > 0 && completed.tag != parent.dep_values[0].tag) {
            return d_error(ErrCode::Type);
          }
          parent.dep_values[completed_dep_idx] = completed;
          parent.next_dep += 1;
          if (parent.next_dep < segment.dep_offset_count) {
            if (depth >= kMaxFrames) {
              return d_error(ErrCode::Timeout);
            }
            frames[depth].state = d_asgp_dp1d_dep_state(segment, parent.state, parent.next_dep);
            frames[depth].next_dep = 0;
            depth += 1;
            break;
          }
          DLocalPreset transition_presets[1 + DMAX_ASGP_DP_DEPS];
          transition_presets[0] = DLocalPreset{segment.transition_state_local, Value::from_int(parent.state)};
          const int bind_dep_count =
              (segment.dep_offset_count < segment.transition_dep_count)
                  ? segment.dep_offset_count
                  : segment.transition_dep_count;
          for (int i = 0; i < bind_dep_count; ++i) {
            transition_presets[1 + i] =
                DLocalPreset{segment.transition_dep_locals[i], parent.dep_values[i]};
          }
          DResult out = d_run_asgp_phase<Flavor>(
              segment.transition, transition_presets, 1 + bind_dep_count, 0, 0,
              payload_tables, payload_state, asgp_tables, fuel_left);
          if (out.is_error) {
            return out;
          }
          if (segment.dep_offset_count > 0 && out.value.tag != parent.dep_values[0].tag) {
            return d_error(ErrCode::Type);
          }
          if (memo_count >= kMaxMemo) {
            return d_error(ErrCode::Timeout);
          }
          memo_keys[memo_count] = parent.state;
          memo_values[memo_count] = out.value;
          memo_count += 1;
          completed = out.value;
          depth -= 1;
        }
        continue;
      }

      const int cached = d_asgp_dp1d_memo_find(memo_keys, memo_values, memo_count, frame.state);
      if (cached >= 0) {
        Value completed = memo_values[cached];
        depth -= 1;
        while (true) {
          if (depth == 0) {
            return d_ok(completed);
          }
          DAsgpDp1dFrame& parent = frames[depth - 1];
          const int completed_dep_idx = parent.next_dep;
          if (completed_dep_idx < 0 || completed_dep_idx >= segment.dep_offset_count) {
            return d_error(ErrCode::Value);
          }
          if (completed_dep_idx > 0 && completed.tag != parent.dep_values[0].tag) {
            return d_error(ErrCode::Type);
          }
          parent.dep_values[completed_dep_idx] = completed;
          parent.next_dep += 1;
          if (parent.next_dep < segment.dep_offset_count) {
            if (depth >= kMaxFrames) {
              return d_error(ErrCode::Timeout);
            }
            frames[depth].state = d_asgp_dp1d_dep_state(segment, parent.state, parent.next_dep);
            frames[depth].next_dep = 0;
            depth += 1;
            break;
          }
          DLocalPreset transition_presets[1 + DMAX_ASGP_DP_DEPS];
          transition_presets[0] = DLocalPreset{segment.transition_state_local, Value::from_int(parent.state)};
          const int bind_dep_count =
              (segment.dep_offset_count < segment.transition_dep_count)
                  ? segment.dep_offset_count
                  : segment.transition_dep_count;
          for (int i = 0; i < bind_dep_count; ++i) {
            transition_presets[1 + i] =
                DLocalPreset{segment.transition_dep_locals[i], parent.dep_values[i]};
          }
          DResult out = d_run_asgp_phase<Flavor>(
              segment.transition, transition_presets, 1 + bind_dep_count, 0, 0,
              payload_tables, payload_state, asgp_tables, fuel_left);
          if (out.is_error) {
            return out;
          }
          if (segment.dep_offset_count > 0 && out.value.tag != parent.dep_values[0].tag) {
            return d_error(ErrCode::Type);
          }
          if (memo_count >= kMaxMemo) {
            return d_error(ErrCode::Timeout);
          }
          memo_keys[memo_count] = parent.state;
          memo_values[memo_count] = out.value;
          memo_count += 1;
          completed = out.value;
          depth -= 1;
        }
        continue;
      }

      if (segment.dep_offset_count == 0) {
        const DLocalPreset presets[1] = {
            DLocalPreset{segment.transition_state_local, Value::from_int(frame.state)},
        };
        DResult out = d_run_asgp_phase<Flavor>(segment.transition, presets, 1, 0, 0,
                                               payload_tables, payload_state, asgp_tables, fuel_left);
        if (out.is_error) {
          return out;
        }
        if (memo_count >= kMaxMemo) {
          return d_error(ErrCode::Timeout);
        }
        memo_keys[memo_count] = frame.state;
        memo_values[memo_count] = out.value;
        memo_count += 1;
        Value completed = out.value;
        depth -= 1;
        while (true) {
          if (depth == 0) {
            return d_ok(completed);
          }
          return d_error(ErrCode::Value);
        }
      }

      if (depth >= kMaxFrames) {
        return d_error(ErrCode::Timeout);
      }
      frames[depth].state = d_asgp_dp1d_dep_state(segment, frame.state, 0);
      frames[depth].next_dep = 0;
      depth += 1;
      continue;
    }

    return d_error(ErrCode::Value);
  }

  return d_error(ErrCode::Value);
}

template <DPayloadFlavor Flavor, bool EnableAsgp>
__device__ __noinline__ DResult d_run_code_core(const DCodeView& view,
                                                const Value* shared_case_local_vals,
                                                const unsigned char* shared_case_local_set,
                                                int local_case,
                                                const DLocalPreset* presets,
                                                int preset_count,
                                                const DPayloadTables& payload_tables,
                                                typename DPayloadFlavorTraits<Flavor>::State& payload_state,
                                                const DAsgpTables& asgp_tables,
                                                int& fuel_left,
                                                bool require_return) {
  DResult result;
  result.is_error = 0;
  result.err_code = ErrCode::Value;
  result.value = Value::invalid();

  Value stack[MAX_STACK];
  Value locals[MAX_LOCALS];
  static_assert(MAX_LOCALS <= 64, "local_set_mask requires MAX_LOCALS <= 64");
  std::uint64_t local_set_mask = 0;

  if (view.n_locals < 0 || view.n_locals > MAX_LOCALS) {
    return d_error(ErrCode::Value);
  }
  if (shared_case_local_vals != nullptr && shared_case_local_set != nullptr) {
    const int base = local_case * MAX_LOCALS;
    for (int i = 0; i < view.n_locals; ++i) {
      locals[i] = shared_case_local_vals[base + i];
      if (shared_case_local_set[base + i]) {
        local_set_mask |= (std::uint64_t{1} << i);
      }
    }
  }
  for (int i = 0; i < preset_count; ++i) {
    const int local = presets[i].local;
    if (local < 0 || local >= view.n_locals) {
      return d_error(ErrCode::Name);
    }
    locals[local] = presets[i].value;
    local_set_mask |= (std::uint64_t{1} << local);
  }

  int sp = 0;
  int ip = 0;
  bool returned = false;

  while (ip < view.code_len) {
    if (fuel_left <= 0) {
      d_fail(result, ErrCode::Timeout);
      break;
    }
    fuel_left -= 1;

    const DInstr ins = view.code[ip];
    ip += 1;

    if (ins.op == OP_PUSH_CONST) {
      if (!d_has_a(ins) || ins.a < 0 || ins.a >= view.const_len || sp >= MAX_STACK) {
        d_fail(result, ErrCode::Value);
        break;
      }
      stack[sp++] = view.consts[ins.a];
      continue;
    }

    if (ins.op == OP_LOAD) {
      if (!d_has_a(ins) || ins.a < 0 || ins.a >= view.n_locals || sp >= MAX_STACK) {
        d_fail(result, ErrCode::Name);
        break;
      }
      if ((local_set_mask & (std::uint64_t{1} << ins.a)) == 0) {
        d_fail(result, ErrCode::Name);
        break;
      }
      stack[sp++] = locals[ins.a];
      continue;
    }

    if (ins.op == OP_STORE) {
      if (!d_has_a(ins) || ins.a < 0 || ins.a >= view.n_locals) {
        d_fail(result, ErrCode::Name);
        break;
      }
      if (sp < 1) {
        d_fail(result, ErrCode::Value);
        break;
      }
      locals[ins.a] = stack[--sp];
      local_set_mask |= (std::uint64_t{1} << ins.a);
      continue;
    }

    if (ins.op == OP_NEG || ins.op == OP_NOT) {
      if (sp < 1) {
        d_fail(result, ErrCode::Value);
        break;
      }
      Value x = stack[--sp];
      if (ins.op == OP_NEG) {
        if (!d_is_num(x)) {
          d_fail(result, ErrCode::Type);
          break;
        }
        stack[sp++] = (x.tag == ValueTag::Float)
                          ? Value::from_float(vm_semantics::canonicalize_vm_float(-x.f))
                          : Value::from_int(vm_semantics::wrap_int_neg(x.i));
      } else {
        if (x.tag != ValueTag::Bool) {
          d_fail(result, ErrCode::Type);
          break;
        }
        stack[sp++] = Value::from_bool(!x.b);
      }
      continue;
    }

    if (ins.op == OP_ADD || ins.op == OP_SUB || ins.op == OP_MUL || ins.op == OP_DIV ||
        ins.op == OP_MOD) {
      if (sp < 2) {
        d_fail(result, ErrCode::Value);
        break;
      }
      Value b = stack[--sp];
      Value a = stack[--sp];
      double a_num = 0.0;
      double b_num = 0.0;
      bool any_float = false;
      if (!d_to_numeric_pair(a, b, a_num, b_num, any_float)) {
        d_fail(result, ErrCode::Type);
        break;
      }
      if ((ins.op == OP_DIV || ins.op == OP_MOD) && b_num == 0.0) {
        d_fail(result, ErrCode::ZeroDiv);
        break;
      }
      if (ins.op == OP_ADD) {
        stack[sp++] = any_float ? Value::from_float(vm_semantics::canonicalize_vm_float(a_num + b_num))
                                : Value::from_int(vm_semantics::wrap_int_add(
                                      static_cast<long long>(a_num), static_cast<long long>(b_num)));
      } else if (ins.op == OP_SUB) {
        stack[sp++] = any_float ? Value::from_float(vm_semantics::canonicalize_vm_float(a_num - b_num))
                                : Value::from_int(vm_semantics::wrap_int_sub(
                                      static_cast<long long>(a_num), static_cast<long long>(b_num)));
      } else if (ins.op == OP_MUL) {
        stack[sp++] = any_float ? Value::from_float(vm_semantics::canonicalize_vm_float(a_num * b_num))
                                : Value::from_int(vm_semantics::wrap_int_mul(
                                      static_cast<long long>(a_num), static_cast<long long>(b_num)));
      } else if (ins.op == OP_DIV) {
        stack[sp++] = Value::from_float(vm_semantics::canonicalize_vm_float(a_num / b_num));
      } else {
        stack[sp++] = any_float
                          ? Value::from_float(vm_semantics::canonicalize_vm_float(d_float_mod(a_num, b_num)))
                          : Value::from_int(d_int_mod(static_cast<long long>(a_num),
                                                      static_cast<long long>(b_num)));
      }
      continue;
    }

    if (ins.op == OP_LT || ins.op == OP_LE || ins.op == OP_GT || ins.op == OP_GE ||
        ins.op == OP_EQ || ins.op == OP_NE) {
      if (sp < 2) {
        d_fail(result, ErrCode::Value);
        break;
      }
      Value b = stack[--sp];
      Value a = stack[--sp];
      bool cmp = false;
      ErrCode derr = ErrCode::Type;
      if (!d_compare(ins.op, a, b, cmp, derr)) {
        d_fail(result, derr);
        break;
      }
      stack[sp++] = Value::from_bool(cmp);
      continue;
    }

    if (ins.op == OP_JMP) {
      if (!d_has_a(ins) || ins.a < 0 || ins.a > view.code_len) {
        d_fail(result, ErrCode::Value);
        break;
      }
      ip = ins.a;
      continue;
    }

    if (ins.op == OP_JMP_IF_FALSE || ins.op == OP_JMP_IF_TRUE) {
      if (sp < 1) {
        d_fail(result, ErrCode::Value);
        break;
      }
      if (!d_has_a(ins) || ins.a < 0 || ins.a > view.code_len) {
        d_fail(result, ErrCode::Value);
        break;
      }
      Value c = stack[--sp];
      if (c.tag != ValueTag::Bool) {
        d_fail(result, ErrCode::Type);
        break;
      }
      if (ins.op == OP_JMP_IF_FALSE && !c.b) ip = ins.a;
      if (ins.op == OP_JMP_IF_TRUE && c.b) ip = ins.a;
      continue;
    }

    if (ins.op == OP_CALL_BUILTIN) {
      BuiltinId bid = BuiltinId::Abs;
      const int argc = d_has_b(ins) ? ins.b : -1;
      if (!d_has_a(ins) || !builtin_id_from_int(ins.a, bid)) {
        d_fail(result, ErrCode::Name);
        break;
      }
      if (argc < 0 || sp < argc) {
        d_fail(result, (argc < 0) ? ErrCode::Type : ErrCode::Value);
        break;
      }
      Value args_buf[4];
      if (argc > 4) {
        d_fail(result, ErrCode::Type);
        break;
      }
      for (int i = 0; i < argc; ++i) {
        args_buf[i] = stack[sp - argc + i];
      }
      sp -= argc;
      ErrCode derr = ErrCode::Type;
      Value ret = Value::invalid();
      if (!d_builtin_call<Flavor>(
              bid, args_buf, argc, payload_tables, payload_state, ret, derr)) {
        d_fail(result, derr);
        break;
      }
      if (sp >= MAX_STACK) {
        d_fail(result, ErrCode::Value);
        break;
      }
      stack[sp++] = ret;
      continue;
    }

    if (ins.op == OP_CHECK_LIST) {
      if (sp < 1) {
        d_fail(result, ErrCode::Value);
        break;
      }
      if (!d_is_typed_list_tag(stack[sp - 1].tag)) {
        d_fail(result, ErrCode::Type);
        break;
      }
      continue;
    }

    if (ins.op == OP_CHECK_INT) {
      if (sp < 1) {
        d_fail(result, ErrCode::Value);
        break;
      }
      if (stack[sp - 1].tag != ValueTag::Int) {
        d_fail(result, ErrCode::Type);
        break;
      }
      continue;
    }

    if (ins.op == OP_EMPTY_LIST) {
      if (!d_has_a(ins) || sp >= MAX_STACK) {
        d_fail(result, ErrCode::Value);
        break;
      }
      Value out = Value::invalid();
      if (!d_make_empty_list_for_tag(d_list_tag_from_private_code(ins.a), payload_state, out)) {
        d_fail(result, ErrCode::Type);
        break;
      }
      stack[sp++] = out;
      continue;
    }

    if (ins.op == OP_EMPTY_LIST_LIKE) {
      if (sp < 1) {
        d_fail(result, ErrCode::Value);
        break;
      }
      const Value source = stack[--sp];
      Value out = Value::invalid();
      if (!d_make_empty_list_for_tag(source.tag, payload_state, out)) {
        d_fail(result, ErrCode::Type);
        break;
      }
      stack[sp++] = out;
      continue;
    }

    if constexpr (EnableAsgp) {
      if (ins.op == OP_ASGP_DC) {
        if (!d_has_a(ins) || ins.a < 0 || ins.a >= view.asgp_dc_count || sp < 1 || sp >= MAX_STACK) {
          d_fail(result, ErrCode::Value);
          break;
        }
        const int seg_idx = view.asgp_dc_offset + ins.a;
        if (seg_idx < 0 || seg_idx >= asgp_tables.dc_segment_count) {
          d_fail(result, ErrCode::Value);
          break;
        }
        const Value source = stack[--sp];
        const int saved_string_entry_count = payload_state.string_entry_count;
        const int saved_list_entry_count = payload_state.list_entry_count;
        const int saved_string_bytes_used = payload_state.string_bytes_used;
        const int saved_list_values_used = payload_state.list_values_used;
        DResult out = d_eval_asgp_dc<Flavor>(
            asgp_tables.dc_segments[seg_idx], source, 0, view.asgp_dc_offset, view.asgp_dc_count,
            payload_tables, payload_state, asgp_tables, fuel_left);
        if (out.is_error) {
          result = out;
          break;
        }
        if (!d_is_payload_value(out.value)) {
          payload_state.string_entry_count = saved_string_entry_count;
          payload_state.list_entry_count = saved_list_entry_count;
          payload_state.string_bytes_used = saved_string_bytes_used;
          payload_state.list_values_used = saved_list_values_used;
        }
        stack[sp++] = out.value;
        continue;
      }

      if (ins.op == OP_ASGP_DP1D) {
        if (!d_has_a(ins) || ins.a < 0 || ins.a >= view.asgp_dp1d_count || sp < 1 || sp >= MAX_STACK) {
          d_fail(result, ErrCode::Value);
          break;
        }
        const int seg_idx = view.asgp_dp1d_offset + ins.a;
        if (seg_idx < 0 || seg_idx >= asgp_tables.dp1d_segment_count) {
          d_fail(result, ErrCode::Value);
          break;
        }
        const Value state = stack[--sp];
        if (state.tag != ValueTag::Int) {
          d_fail(result, ErrCode::Type);
          break;
        }
        DResult out = d_eval_asgp_dp1d<Flavor>(
            asgp_tables.dp1d_segments[seg_idx], state.i, payload_tables, payload_state, asgp_tables, fuel_left);
        if (out.is_error) {
          result = out;
          break;
        }
        stack[sp++] = out.value;
        continue;
      }

      if (ins.op == OP_ASGP_DP2D) {
        if (!d_has_a(ins) || ins.a < 0 || ins.a >= view.asgp_dp2d_count || sp < 2 || sp >= MAX_STACK) {
          d_fail(result, ErrCode::Value);
          break;
        }
        const int seg_idx = view.asgp_dp2d_offset + ins.a;
        if (seg_idx < 0 || seg_idx >= asgp_tables.dp2d_segment_count) {
          d_fail(result, ErrCode::Value);
          break;
        }
        const Value state_j = stack[--sp];
        const Value state_i = stack[--sp];
        if (state_i.tag != ValueTag::Int || state_j.tag != ValueTag::Int) {
          d_fail(result, ErrCode::Type);
          break;
        }
        DResult out = d_eval_asgp_dp2d<Flavor>(
            asgp_tables.dp2d_segments[seg_idx], state_i.i, state_j.i,
            payload_tables, payload_state, asgp_tables, fuel_left);
        if (out.is_error) {
          result = out;
          break;
        }
        stack[sp++] = out.value;
        continue;
      }
    }

    if (ins.op == OP_RETURN) {
      if (sp < 1) {
        d_fail(result, ErrCode::Value);
        break;
      }
      result.is_error = 0;
      result.value = stack[sp - 1];
      returned = true;
      break;
    }

    d_fail(result, ErrCode::Type);
    break;
  }

  if (!returned && !result.is_error) {
    if (require_return || sp < 1) {
      d_fail(result, ErrCode::Value);
    } else {
      result = d_ok(stack[sp - 1]);
    }
  }
  return result;
}

template <DPayloadFlavor Flavor, bool EnableAsgp>
__device__ __noinline__ DResult d_execute_bytecode_impl(const DProgramMeta& meta,
                                                        const DInstr* shared_code,
                                                        const Value* all_consts,
                                                        const Value* shared_case_local_vals,
                                                        const unsigned char* shared_case_local_set,
                                                        const DPayloadTables& payload_tables,
                                                        const DAsgpTables& asgp_tables,
                                                        int local_case,
                                                        int fuel) {
  DResult result;
  result.is_error = 0;
  result.err_code = ErrCode::Value;
  result.value = Value::invalid();

  if (!meta.is_valid) {
    d_fail(result, meta.err_code);
    return result;
  }

  DPayloadStateStorage<Flavor> payload_state_storage;
  int fuel_left = fuel;
  const DCodeView view{
      shared_code,
      meta.code_len,
      all_consts + meta.const_offset,
      meta.const_len,
      meta.n_locals,
      meta.asgp_dc_offset,
      meta.asgp_dc_count,
      meta.asgp_dp1d_offset,
      meta.asgp_dp1d_count,
      meta.asgp_dp2d_offset,
      meta.asgp_dp2d_count,
  };
  return d_run_code_core<Flavor, EnableAsgp>(view, shared_case_local_vals, shared_case_local_set,
                                            local_case, nullptr, 0, payload_tables,
                                            payload_state_storage.ref(), asgp_tables, fuel_left, true);
}

}  // namespace g3pvm::gpu_detail
