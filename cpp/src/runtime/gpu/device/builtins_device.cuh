#pragma once

#include "g3pvm/core/builtin.hpp"
#include "arith_device.cuh"

namespace g3pvm::gpu_detail {

struct DPayloadTables {
  const DStringPayloadEntry* string_entries = nullptr;
  int string_entry_count = 0;
  const char* string_bytes = nullptr;
  const DListPayloadEntry* list_entries = nullptr;
  int list_entry_count = 0;
  const Value* list_values = nullptr;
};

template <int MaxStringEntries, int MaxListEntries, int MaxStringBytes, int MaxListValues>
struct DThreadPayloadStateT {
  static constexpr int kMaxStringEntries = MaxStringEntries;
  static constexpr int kMaxListEntries = MaxListEntries;
  static constexpr int kMaxStringBytes = MaxStringBytes;
  static constexpr int kMaxListValues = MaxListValues;

  DStringPayloadEntry string_entries[(MaxStringEntries > 0) ? MaxStringEntries : 1];
  DListPayloadEntry list_entries[(MaxListEntries > 0) ? MaxListEntries : 1];
  int string_entry_count = 0;
  int list_entry_count = 0;
  int string_bytes_used = 0;
  int list_values_used = 0;
  char string_bytes[(MaxStringBytes > 0) ? MaxStringBytes : 1];
  Value list_values[(MaxListValues > 0) ? MaxListValues : 1];
};

using DNoPayloadState = DThreadPayloadStateT<0, 0, 0, 0>;
using DStringPayloadState = DThreadPayloadStateT<DMAX_THREAD_PAYLOAD_ENTRIES, 0, DMAX_THREAD_STRING_BYTES, 0>;
using DListPayloadState = DThreadPayloadStateT<0, DMAX_THREAD_PAYLOAD_ENTRIES, 0, DMAX_THREAD_LIST_VALUES>;
using DMixedPayloadState =
    DThreadPayloadStateT<DMAX_THREAD_PAYLOAD_ENTRIES,
                         DMAX_THREAD_PAYLOAD_ENTRIES,
                         DMAX_THREAD_STRING_BYTES,
                         DMAX_THREAD_LIST_VALUES>;

template <DPayloadFlavor Flavor>
struct DPayloadFlavorTraits;

template <>
struct DPayloadFlavorTraits<DPayloadFlavor::None> {
  using State = DNoPayloadState;
  static constexpr bool kHasString = false;
  static constexpr bool kHasList = false;
};

template <>
struct DPayloadFlavorTraits<DPayloadFlavor::StringOnly> {
  using State = DStringPayloadState;
  static constexpr bool kHasString = true;
  static constexpr bool kHasList = false;
};

template <>
struct DPayloadFlavorTraits<DPayloadFlavor::ListOnly> {
  using State = DListPayloadState;
  static constexpr bool kHasString = false;
  static constexpr bool kHasList = true;
};

template <>
struct DPayloadFlavorTraits<DPayloadFlavor::Mixed> {
  using State = DMixedPayloadState;
  static constexpr bool kHasString = true;
  static constexpr bool kHasList = true;
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

__device__ inline int d_find_string_payload_entry(const DPayloadTables& tables, std::int64_t packed) {
  int lo = 0;
  int hi = tables.string_entry_count;
  while (lo < hi) {
    const int mid = lo + ((hi - lo) >> 1);
    const std::int64_t mid_packed = tables.string_entries[mid].packed;
    if (mid_packed < packed) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (lo >= tables.string_entry_count || tables.string_entries[lo].packed != packed) {
    return -1;
  }
  return lo;
}

__device__ inline int d_find_list_payload_entry(const DPayloadTables& tables, std::int64_t packed) {
  int lo = 0;
  int hi = tables.list_entry_count;
  while (lo < hi) {
    const int mid = lo + ((hi - lo) >> 1);
    const std::int64_t mid_packed = tables.list_entries[mid].packed;
    if (mid_packed < packed) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (lo >= tables.list_entry_count || tables.list_entries[lo].packed != packed) {
    return -1;
  }
  return lo;
}

template <typename State>
__device__ inline bool d_lookup_string_payload(const DPayloadTables& tables,
                                               const State& st,
                                               const Value& v,
                                               const char*& ptr,
                                               int& len) {
  if (v.tag != ValueTag::String) return false;
  if constexpr (State::kMaxStringEntries > 0) {
    for (int i = 0; i < st.string_entry_count; ++i) {
      if (st.string_entries[i].packed == v.i) {
        ptr = st.string_bytes + st.string_entries[i].offset;
        len = st.string_entries[i].len;
        return true;
      }
    }
  }
  const int idx = d_find_string_payload_entry(tables, v.i);
  if (idx >= 0) {
    ptr = tables.string_bytes + tables.string_entries[idx].offset;
    len = tables.string_entries[idx].len;
    return true;
  }
  return false;
}

template <typename State>
__device__ inline bool d_lookup_list_payload(const DPayloadTables& tables,
                                             const State& st,
                                             const Value& v,
                                             const Value*& ptr,
                                             int& len) {
  if (v.tag != ValueTag::List) return false;
  if constexpr (State::kMaxListEntries > 0) {
    for (int i = 0; i < st.list_entry_count; ++i) {
      if (st.list_entries[i].packed == v.i) {
        ptr = st.list_values + st.list_entries[i].offset;
        len = st.list_entries[i].len;
        return true;
      }
    }
  }
  const int idx = d_find_list_payload_entry(tables, v.i);
  if (idx >= 0) {
    ptr = tables.list_values + tables.list_entries[idx].offset;
    len = tables.list_entries[idx].len;
    return true;
  }
  return false;
}

template <typename State>
__device__ inline bool d_register_local_string(State& st, const Value& v, int offset, int len) {
  if constexpr (State::kMaxStringEntries <= 0) {
    (void)st;
    (void)v;
    (void)offset;
    (void)len;
    return false;
  }
  if (st.string_entry_count >= State::kMaxStringEntries) return false;
  st.string_entries[st.string_entry_count++] = DStringPayloadEntry{v.i, offset, len};
  return true;
}

template <typename State>
__device__ inline bool d_register_local_list(State& st, const Value& v, int offset, int len) {
  if constexpr (State::kMaxListEntries <= 0) {
    (void)st;
    (void)v;
    (void)offset;
    (void)len;
    return false;
  }
  if (st.list_entry_count >= State::kMaxListEntries) return false;
  st.list_entries[st.list_entry_count++] = DListPayloadEntry{v.i, offset, len};
  return true;
}

template <DPayloadFlavor Flavor>
__device__ inline bool d_builtin_call(BuiltinId bid,
                                      const Value* args,
                                      int argc,
                                      const DPayloadTables& tables,
                                      typename DPayloadFlavorTraits<Flavor>::State& payload_state,
                                      Value& out,
                                      ErrCode& err) {
  using PayloadTraits = DPayloadFlavorTraits<Flavor>;

  if (bid == BuiltinId::Abs) {
    if (argc != 1) {
      err = ErrCode::Type;
      return false;
    }
    const Value& x = args[0];
    if (!d_is_num(x)) {
      err = ErrCode::Type;
      return false;
    }
    out = (x.tag == ValueTag::Float)
              ? Value::from_float(vm_semantics::canonicalize_vm_float(x.f < 0 ? -x.f : x.f))
                                      : Value::from_int(x.i < 0 ? -x.i : x.i);
    return true;
  }

  if (bid == BuiltinId::Min || bid == BuiltinId::Max) {
    if (argc != 2) {
      err = ErrCode::Type;
      return false;
    }
    double a = 0.0;
    double b = 0.0;
    bool any_float = false;
    if (!d_to_numeric_pair(args[0], args[1], a, b, any_float)) {
      err = ErrCode::Type;
      return false;
    }
    const double pick =
        (bid == BuiltinId::Min) ? ((a <= b) ? a : b) : ((a >= b) ? a : b);
    out = any_float ? Value::from_float(vm_semantics::canonicalize_vm_float(pick))
                    : Value::from_int(static_cast<long long>(pick));
    return true;
  }

  if (bid == BuiltinId::Clip) {
    if (argc != 3) {
      err = ErrCode::Type;
      return false;
    }
    const Value& x = args[0];
    const Value& lo = args[1];
    const Value& hi = args[2];
    if (!d_is_num(x) || !d_is_num(lo) || !d_is_num(hi)) {
      err = ErrCode::Type;
      return false;
    }
    const bool any_float =
        (x.tag == ValueTag::Float) || (lo.tag == ValueTag::Float) || (hi.tag == ValueTag::Float);
    if (any_float) {
      const double x2 = (x.tag == ValueTag::Float) ? x.f : static_cast<double>(x.i);
      const double lo2 = (lo.tag == ValueTag::Float) ? lo.f : static_cast<double>(lo.i);
      const double hi2 = (hi.tag == ValueTag::Float) ? hi.f : static_cast<double>(hi.i);
      if (lo2 > hi2) {
        err = ErrCode::Value;
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
      err = ErrCode::Value;
      return false;
    }
    out = (x2 < lo2) ? Value::from_int(lo2)
                     : ((x2 > hi2) ? Value::from_int(hi2) : Value::from_int(x2));
    return true;
  }

  if (bid == BuiltinId::Len) {
    if (argc != 1) {
      err = ErrCode::Type;
      return false;
    }
    const Value& x = args[0];
    if (x.tag != ValueTag::String && x.tag != ValueTag::List) {
      err = ErrCode::Type;
      return false;
    }
    out = Value::from_int(static_cast<long long>(Value::container_len(x)));
    return true;
  }

  if (bid == BuiltinId::Concat) {
    if (argc != 2) {
      err = ErrCode::Type;
      return false;
    }
    const Value& a = args[0];
    const Value& b = args[1];
    if (a.tag == ValueTag::String && b.tag == ValueTag::String) {
      if constexpr (PayloadTraits::kHasString) {
        auto& st = payload_state;
        const char* ap = nullptr;
        const char* bp = nullptr;
        int al = 0;
        int bl = 0;
        if (d_lookup_string_payload(tables, st, a, ap, al) &&
            d_lookup_string_payload(tables, st, b, bp, bl) &&
            al >= 0 && bl >= 0 &&
            st.string_bytes_used + al + bl <= st.kMaxStringBytes) {
          const int off = st.string_bytes_used;
          for (int i = 0; i < al; ++i) st.string_bytes[off + i] = ap[i];
          for (int i = 0; i < bl; ++i) st.string_bytes[off + al + i] = bp[i];
          st.string_bytes_used += al + bl;
          const std::uint64_t h = d_hash_bytes(st.string_bytes + off, al + bl);
          out = Value::from_string_hash_len(h, static_cast<std::uint32_t>(al + bl));
          (void)d_register_local_string(st, out, off, al + bl);
          return true;
        }
      }
      const std::uint32_t len = Value::saturating_len_add(Value::container_len(a), Value::container_len(b));
      const std::uint64_t h = Value::combine_container_hash48(1U, a, b);
      out = Value::from_fallback_token(Value::pack_container_payload(h, len));
      return true;
    }
    if (a.tag == ValueTag::List && b.tag == ValueTag::List) {
      if constexpr (PayloadTraits::kHasList) {
        auto& st = payload_state;
        const Value* ap = nullptr;
        const Value* bp = nullptr;
        int al = 0;
        int bl = 0;
        if (d_lookup_list_payload(tables, st, a, ap, al) &&
            d_lookup_list_payload(tables, st, b, bp, bl) &&
            al >= 0 && bl >= 0 &&
            st.list_values_used + al + bl <= st.kMaxListValues) {
          const int off = st.list_values_used;
          for (int i = 0; i < al; ++i) st.list_values[off + i] = ap[i];
          for (int i = 0; i < bl; ++i) st.list_values[off + al + i] = bp[i];
          st.list_values_used += al + bl;
          const std::uint64_t h = d_hash_list_payload(st.list_values + off, al + bl);
          out = Value::from_list_hash_len(h, static_cast<std::uint32_t>(al + bl));
          (void)d_register_local_list(st, out, off, al + bl);
          return true;
        }
      }
      const std::uint32_t len = Value::saturating_len_add(Value::container_len(a), Value::container_len(b));
      const std::uint64_t h = Value::combine_container_hash48(2U, a, b);
      out = Value::from_fallback_token(Value::pack_container_payload(h, len));
      return true;
    }
    err = ErrCode::Type;
    return false;
  }

  if (bid == BuiltinId::Slice) {
    if (argc != 3) {
      err = ErrCode::Type;
      return false;
    }
    const Value& x = args[0];
    const Value& lo = args[1];
    const Value& hi = args[2];
    if (!(x.tag == ValueTag::String || x.tag == ValueTag::List)) {
      err = ErrCode::Type;
      return false;
    }
    if (lo.tag != ValueTag::Int || hi.tag != ValueTag::Int) {
      err = ErrCode::Type;
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
      if constexpr (PayloadTraits::kHasString) {
        auto& st = payload_state;
        const char* xp = nullptr;
        int xl = 0;
        if (d_lookup_string_payload(tables, st, x, xp, xl) &&
            l >= 0 && l <= xl && h >= 0 && h <= xl &&
            st.string_bytes_used + static_cast<int>(out_len) <= st.kMaxStringBytes) {
          const int off = st.string_bytes_used;
          for (int i = 0; i < static_cast<int>(out_len); ++i) {
            st.string_bytes[off + i] = xp[static_cast<int>(l) + i];
          }
          st.string_bytes_used += static_cast<int>(out_len);
          const std::uint64_t out_h_exact = d_hash_bytes(st.string_bytes + off, static_cast<int>(out_len));
          out = Value::from_string_hash_len(out_h_exact, out_len);
          (void)d_register_local_string(st, out, off, static_cast<int>(out_len));
          return true;
        }
      }
      const std::uint64_t out_h = Value::slice_container_hash48(3U, x, lo.i, hi.i);
      out = Value::from_fallback_token(Value::pack_container_payload(out_h, out_len));
      return true;
    }
    if constexpr (PayloadTraits::kHasList) {
      auto& st = payload_state;
      const Value* xp = nullptr;
      int xl = 0;
      if (d_lookup_list_payload(tables, st, x, xp, xl) &&
          l >= 0 && l <= xl && h >= 0 && h <= xl &&
          st.list_values_used + static_cast<int>(out_len) <= st.kMaxListValues) {
        const int off = st.list_values_used;
        for (int i = 0; i < static_cast<int>(out_len); ++i) {
          st.list_values[off + i] = xp[static_cast<int>(l) + i];
        }
        st.list_values_used += static_cast<int>(out_len);
        const std::uint64_t out_h_exact = d_hash_list_payload(st.list_values + off, static_cast<int>(out_len));
        out = Value::from_list_hash_len(out_h_exact, out_len);
        (void)d_register_local_list(st, out, off, static_cast<int>(out_len));
        return true;
      }
    }
    const std::uint64_t out_h = Value::slice_container_hash48(4U, x, lo.i, hi.i);
    out = Value::from_fallback_token(Value::pack_container_payload(out_h, out_len));
    return true;
  }

  if (bid == BuiltinId::Index) {
    if (argc != 2) {
      err = ErrCode::Type;
      return false;
    }
    const Value& x = args[0];
    const Value& i = args[1];
    if (!(x.tag == ValueTag::String || x.tag == ValueTag::List)) {
      err = ErrCode::Type;
      return false;
    }
    if (i.tag != ValueTag::Int) {
      err = ErrCode::Type;
      return false;
    }
    const long long n = static_cast<long long>(Value::container_len(x));
    long long j = 0;
    if (!d_norm_index_idx(i.i, n, j)) {
      err = ErrCode::Value;
      return false;
    }
    if (x.tag == ValueTag::String) {
      if constexpr (PayloadTraits::kHasString) {
        auto& st = payload_state;
        const char* xp = nullptr;
        int xl = 0;
        if (d_lookup_string_payload(tables, st, x, xp, xl) && j < xl &&
            st.string_bytes_used + 1 <= st.kMaxStringBytes) {
          const int off = st.string_bytes_used;
          st.string_bytes[off] = xp[static_cast<int>(j)];
          st.string_bytes_used += 1;
          const std::uint64_t h1 = d_hash_bytes(st.string_bytes + off, 1);
          out = Value::from_string_hash_len(h1, 1U);
          (void)d_register_local_string(st, out, off, 1);
          return true;
        }
      }
      out = Value::from_fallback_token(Value::index_container_token64(5U, x, j));
      return true;
    }
    if constexpr (PayloadTraits::kHasList) {
      auto& st = payload_state;
      const Value* xp = nullptr;
      int xl = 0;
      if (d_lookup_list_payload(tables, st, x, xp, xl) && j < xl) {
        out = xp[static_cast<int>(j)];
        return true;
      }
    }
    out = Value::from_fallback_token(Value::index_container_token64(6U, x, j));
    return true;
  }

  if (bid == BuiltinId::IsInt) {
    if (argc != 1) {
      err = ErrCode::Type;
      return false;
    }
    out = Value::from_bool(args[0].tag == ValueTag::Int);
    return true;
  }

  err = ErrCode::Name;
  return false;
}

}  // namespace g3pvm::gpu_detail
