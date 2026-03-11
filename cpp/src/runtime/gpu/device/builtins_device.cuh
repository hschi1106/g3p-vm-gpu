#pragma once

#include "arith_device.cuh"

namespace g3pvm::gpu_detail {

static constexpr int DMAX_THREAD_PAYLOAD_ENTRIES = 32;
static constexpr int DMAX_THREAD_STRING_BYTES = 512;
static constexpr int DMAX_THREAD_LIST_VALUES = 128;

struct DPayloadTables {
  const DStringPayloadEntry* string_entries = nullptr;
  int string_entry_count = 0;
  const char* string_bytes = nullptr;
  const DListPayloadEntry* list_entries = nullptr;
  int list_entry_count = 0;
  const Value* list_values = nullptr;
};

struct DThreadPayloadState {
  DStringPayloadEntry string_entries[DMAX_THREAD_PAYLOAD_ENTRIES];
  DListPayloadEntry list_entries[DMAX_THREAD_PAYLOAD_ENTRIES];
  int string_entry_count = 0;
  int list_entry_count = 0;
  int string_bytes_used = 0;
  int list_values_used = 0;
  char string_bytes[DMAX_THREAD_STRING_BYTES];
  Value list_values[DMAX_THREAD_LIST_VALUES];
};

__device__ inline long long d_norm_slice_idx(long long idx, long long n) {
  long long out = idx;
  if (out < 0) out += n;
  if (out < 0) out = 0;
  if (out > n) out = n;
  return out;
}

__device__ inline bool d_norm_index_idx(long long idx, long long n, long long& out) {
  long long j = idx;
  if (j < 0) j += n;
  if (j < 0 || j >= n) {
    return false;
  }
  out = j;
  return true;
}

__device__ inline std::uint64_t d_hash_bytes(const char* p, int n) {
  std::uint64_t h = Value::fnv1a_init();
  for (int i = 0; i < n; ++i) {
    h = Value::fnv1a_mix_u8(h, static_cast<std::uint8_t>(p[i]));
  }
  return h;
}

__device__ inline std::uint64_t d_hash_value_shallow(const Value& v) {
  std::uint64_t h = Value::fnv1a_init();
  h = Value::fnv1a_mix_u8(h, static_cast<std::uint8_t>(v.tag));
  if (v.tag == ValueTag::None) return h;
  if (v.tag == ValueTag::Bool) return Value::fnv1a_mix_u8(h, v.b ? 1U : 0U);
  if (v.tag == ValueTag::Float) {
    union {
      double d;
      std::uint64_t u;
    } bits{};
    bits.d = v.f;
    return Value::fnv1a_mix_u64(h, bits.u);
  }
  return Value::fnv1a_mix_u64(h, static_cast<std::uint64_t>(v.i));
}

__device__ inline std::uint64_t d_hash_list_payload(const Value* elems, int n) {
  std::uint64_t h = Value::fnv1a_init();
  h = Value::fnv1a_mix_u8(h, 0xA1U);
  h = Value::fnv1a_mix_u64(h, static_cast<std::uint64_t>(n));
  for (int i = 0; i < n; ++i) {
    h = Value::fnv1a_mix_u64(h, d_hash_value_shallow(elems[i]));
  }
  return h;
}

__device__ inline bool d_lookup_string_payload(const DPayloadTables& tables,
                                               const DThreadPayloadState& st,
                                               const Value& v,
                                               const char*& ptr,
                                               int& len) {
  if (v.tag != ValueTag::String) return false;
  for (int i = 0; i < st.string_entry_count; ++i) {
    if (st.string_entries[i].packed == v.i) {
      ptr = st.string_bytes + st.string_entries[i].offset;
      len = st.string_entries[i].len;
      return true;
    }
  }
  for (int i = 0; i < tables.string_entry_count; ++i) {
    if (tables.string_entries[i].packed == v.i) {
      ptr = tables.string_bytes + tables.string_entries[i].offset;
      len = tables.string_entries[i].len;
      return true;
    }
  }
  return false;
}

__device__ inline bool d_lookup_list_payload(const DPayloadTables& tables,
                                             const DThreadPayloadState& st,
                                             const Value& v,
                                             const Value*& ptr,
                                             int& len) {
  if (v.tag != ValueTag::List) return false;
  for (int i = 0; i < st.list_entry_count; ++i) {
    if (st.list_entries[i].packed == v.i) {
      ptr = st.list_values + st.list_entries[i].offset;
      len = st.list_entries[i].len;
      return true;
    }
  }
  for (int i = 0; i < tables.list_entry_count; ++i) {
    if (tables.list_entries[i].packed == v.i) {
      ptr = tables.list_values + tables.list_entries[i].offset;
      len = tables.list_entries[i].len;
      return true;
    }
  }
  return false;
}

__device__ inline bool d_register_local_string(DThreadPayloadState& st, const Value& v, int offset, int len) {
  if (st.string_entry_count >= DMAX_THREAD_PAYLOAD_ENTRIES) return false;
  st.string_entries[st.string_entry_count++] = DStringPayloadEntry{v.i, offset, len};
  return true;
}

__device__ inline bool d_register_local_list(DThreadPayloadState& st, const Value& v, int offset, int len) {
  if (st.list_entry_count >= DMAX_THREAD_PAYLOAD_ENTRIES) return false;
  st.list_entries[st.list_entry_count++] = DListPayloadEntry{v.i, offset, len};
  return true;
}

__device__ inline bool d_builtin_call(int bid,
                                      const Value* args,
                                      int argc,
                                      const DPayloadTables& tables,
                                      DThreadPayloadState& payload_state,
                                      Value& out,
                                      DeviceErrCode& err) {
  if (bid == 0) {
    if (argc != 1) {
      err = DERR_TYPE;
      return false;
    }
    const Value& x = args[0];
    if (!d_is_num(x)) {
      err = DERR_TYPE;
      return false;
    }
    out = (x.tag == ValueTag::Float)
              ? Value::from_float(vm_semantics::canonicalize_vm_float(x.f < 0 ? -x.f : x.f))
                                      : Value::from_int(x.i < 0 ? -x.i : x.i);
    return true;
  }

  if (bid == 1 || bid == 2) {
    if (argc != 2) {
      err = DERR_TYPE;
      return false;
    }
    double a = 0.0;
    double b = 0.0;
    bool any_float = false;
    if (!d_to_numeric_pair(args[0], args[1], a, b, any_float)) {
      err = DERR_TYPE;
      return false;
    }
    const double pick = (bid == 1) ? ((a <= b) ? a : b) : ((a >= b) ? a : b);
    out = any_float ? Value::from_float(vm_semantics::canonicalize_vm_float(pick))
                    : Value::from_int(static_cast<long long>(pick));
    return true;
  }

  if (bid == 3) {
    if (argc != 3) {
      err = DERR_TYPE;
      return false;
    }
    const Value& x = args[0];
    const Value& lo = args[1];
    const Value& hi = args[2];
    if (!d_is_num(x) || !d_is_num(lo) || !d_is_num(hi)) {
      err = DERR_TYPE;
      return false;
    }
    const bool any_float =
        (x.tag == ValueTag::Float) || (lo.tag == ValueTag::Float) || (hi.tag == ValueTag::Float);
    if (any_float) {
      const double x2 = (x.tag == ValueTag::Float) ? x.f : static_cast<double>(x.i);
      const double lo2 = (lo.tag == ValueTag::Float) ? lo.f : static_cast<double>(lo.i);
      const double hi2 = (hi.tag == ValueTag::Float) ? hi.f : static_cast<double>(hi.i);
      if (lo2 > hi2) {
        err = DERR_VALUE;
        return false;
      }
      out = (x2 < lo2) ? Value::from_float(vm_semantics::canonicalize_vm_float(lo2))
                       : ((x2 > hi2) ? Value::from_float(vm_semantics::canonicalize_vm_float(hi2))
                                     : Value::from_float(vm_semantics::canonicalize_vm_float(x2)));
      return true;
    }
    const long long x2 = x.i;
    const long long lo2 = lo.i;
    const long long hi2 = hi.i;
    if (lo2 > hi2) {
      err = DERR_VALUE;
      return false;
    }
    out = (x2 < lo2) ? Value::from_int(lo2)
                     : ((x2 > hi2) ? Value::from_int(hi2) : Value::from_int(x2));
    return true;
  }

  if (bid == 4) {
    if (argc != 1) {
      err = DERR_TYPE;
      return false;
    }
    const Value& x = args[0];
    if (x.tag != ValueTag::String && x.tag != ValueTag::List) {
      err = DERR_TYPE;
      return false;
    }
    out = Value::from_int(static_cast<long long>(Value::container_len(x)));
    return true;
  }

  if (bid == 5) {
    if (argc != 2) {
      err = DERR_TYPE;
      return false;
    }
    const Value& a = args[0];
    const Value& b = args[1];
    if (a.tag == ValueTag::String && b.tag == ValueTag::String) {
      const char* ap = nullptr;
      const char* bp = nullptr;
      int al = 0;
      int bl = 0;
      if (d_lookup_string_payload(tables, payload_state, a, ap, al) &&
          d_lookup_string_payload(tables, payload_state, b, bp, bl) &&
          al >= 0 && bl >= 0 &&
          payload_state.string_bytes_used + al + bl <= DMAX_THREAD_STRING_BYTES) {
        const int off = payload_state.string_bytes_used;
        for (int i = 0; i < al; ++i) payload_state.string_bytes[off + i] = ap[i];
        for (int i = 0; i < bl; ++i) payload_state.string_bytes[off + al + i] = bp[i];
        payload_state.string_bytes_used += al + bl;
        const std::uint64_t h = d_hash_bytes(payload_state.string_bytes + off, al + bl);
        out = Value::from_string_hash_len(h, static_cast<std::uint32_t>(al + bl));
        (void)d_register_local_string(payload_state, out, off, al + bl);
        return true;
      }
      const std::uint32_t len = Value::saturating_len_add(Value::container_len(a), Value::container_len(b));
      const std::uint64_t h = Value::combine_container_hash48(1U, a, b);
      out = Value::from_string_hash_len(h, len);
      return true;
    }
    if (a.tag == ValueTag::List && b.tag == ValueTag::List) {
      const Value* ap = nullptr;
      const Value* bp = nullptr;
      int al = 0;
      int bl = 0;
      if (d_lookup_list_payload(tables, payload_state, a, ap, al) &&
          d_lookup_list_payload(tables, payload_state, b, bp, bl) &&
          al >= 0 && bl >= 0 &&
          payload_state.list_values_used + al + bl <= DMAX_THREAD_LIST_VALUES) {
        const int off = payload_state.list_values_used;
        for (int i = 0; i < al; ++i) payload_state.list_values[off + i] = ap[i];
        for (int i = 0; i < bl; ++i) payload_state.list_values[off + al + i] = bp[i];
        payload_state.list_values_used += al + bl;
        const std::uint64_t h = d_hash_list_payload(payload_state.list_values + off, al + bl);
        out = Value::from_list_hash_len(h, static_cast<std::uint32_t>(al + bl));
        (void)d_register_local_list(payload_state, out, off, al + bl);
        return true;
      }
      const std::uint32_t len = Value::saturating_len_add(Value::container_len(a), Value::container_len(b));
      const std::uint64_t h = Value::combine_container_hash48(2U, a, b);
      out = Value::from_list_hash_len(h, len);
      return true;
    }
    err = DERR_TYPE;
    return false;
  }

  if (bid == 6) {
    if (argc != 3) {
      err = DERR_TYPE;
      return false;
    }
    const Value& x = args[0];
    const Value& lo = args[1];
    const Value& hi = args[2];
    if (!(x.tag == ValueTag::String || x.tag == ValueTag::List)) {
      err = DERR_TYPE;
      return false;
    }
    if (lo.tag != ValueTag::Int || hi.tag != ValueTag::Int) {
      err = DERR_TYPE;
      return false;
    }
    const long long n = static_cast<long long>(Value::container_len(x));
    const long long l = d_norm_slice_idx(lo.i, n);
    const long long h = d_norm_slice_idx(hi.i, n);
    const long long out_len_ll = (h > l) ? (h - l) : 0LL;
    const std::uint32_t out_len =
        static_cast<std::uint32_t>(out_len_ll > static_cast<long long>(Value::k_container_len_max)
                                       ? Value::k_container_len_max
                                       : out_len_ll);
    if (x.tag == ValueTag::String) {
      const char* xp = nullptr;
      int xl = 0;
      if (d_lookup_string_payload(tables, payload_state, x, xp, xl) &&
          l >= 0 && h >= l && h <= xl &&
          payload_state.string_bytes_used + static_cast<int>(out_len) <= DMAX_THREAD_STRING_BYTES) {
        const int off = payload_state.string_bytes_used;
        for (int i = 0; i < static_cast<int>(out_len); ++i) {
          payload_state.string_bytes[off + i] = xp[static_cast<int>(l) + i];
        }
        payload_state.string_bytes_used += static_cast<int>(out_len);
        const std::uint64_t out_h_exact = d_hash_bytes(payload_state.string_bytes + off, static_cast<int>(out_len));
        out = Value::from_string_hash_len(out_h_exact, out_len);
        (void)d_register_local_string(payload_state, out, off, static_cast<int>(out_len));
        return true;
      }
      const std::uint64_t out_h = Value::slice_container_hash48(3U, x, lo.i, hi.i);
      out = Value::from_string_hash_len(out_h, out_len);
      return true;
    }
    const Value* xp = nullptr;
    int xl = 0;
    if (d_lookup_list_payload(tables, payload_state, x, xp, xl) &&
        l >= 0 && h >= l && h <= xl &&
        payload_state.list_values_used + static_cast<int>(out_len) <= DMAX_THREAD_LIST_VALUES) {
      const int off = payload_state.list_values_used;
      for (int i = 0; i < static_cast<int>(out_len); ++i) {
        payload_state.list_values[off + i] = xp[static_cast<int>(l) + i];
      }
      payload_state.list_values_used += static_cast<int>(out_len);
      const std::uint64_t out_h_exact = d_hash_list_payload(payload_state.list_values + off, static_cast<int>(out_len));
      out = Value::from_list_hash_len(out_h_exact, out_len);
      (void)d_register_local_list(payload_state, out, off, static_cast<int>(out_len));
      return true;
    }
    const std::uint64_t out_h = Value::slice_container_hash48(4U, x, lo.i, hi.i);
    out = Value::from_list_hash_len(out_h, out_len);
    return true;
  }

  if (bid == 7) {
    if (argc != 2) {
      err = DERR_TYPE;
      return false;
    }
    const Value& x = args[0];
    const Value& i = args[1];
    if (!(x.tag == ValueTag::String || x.tag == ValueTag::List)) {
      err = DERR_TYPE;
      return false;
    }
    if (i.tag != ValueTag::Int) {
      err = DERR_TYPE;
      return false;
    }
    const long long n = static_cast<long long>(Value::container_len(x));
    long long j = 0;
    if (!d_norm_index_idx(i.i, n, j)) {
      err = DERR_VALUE;
      return false;
    }
    if (x.tag == ValueTag::String) {
      const char* xp = nullptr;
      int xl = 0;
      if (d_lookup_string_payload(tables, payload_state, x, xp, xl) && j < xl &&
          payload_state.string_bytes_used + 1 <= DMAX_THREAD_STRING_BYTES) {
        const int off = payload_state.string_bytes_used;
        payload_state.string_bytes[off] = xp[static_cast<int>(j)];
        payload_state.string_bytes_used += 1;
        const std::uint64_t h1 = d_hash_bytes(payload_state.string_bytes + off, 1);
        out = Value::from_string_hash_len(h1, 1U);
        (void)d_register_local_string(payload_state, out, off, 1);
        return true;
      }
      out = Value::from_int(Value::index_container_token64(5U, x, j));
      return true;
    }
    const Value* xp = nullptr;
    int xl = 0;
    if (d_lookup_list_payload(tables, payload_state, x, xp, xl) && j < xl) {
      out = xp[static_cast<int>(j)];
      return true;
    }
    out = Value::from_int(Value::index_container_token64(6U, x, j));
    return true;
  }

  err = DERR_NAME;
  return false;
}

}  // namespace g3pvm::gpu_detail
