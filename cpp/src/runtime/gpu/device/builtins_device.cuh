#pragma once

#include "gagp/core/builtin.hpp"
#include "arith_device.cuh"

namespace gagp::gpu_detail {

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
  if (v.tag == ValueTag::Invalid) return h;
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

__device__ inline bool d_is_typed_list_tag(ValueTag tag) {
  return tag == ValueTag::IntList || tag == ValueTag::FloatList || tag == ValueTag::StringList;
}

__device__ inline std::uint8_t d_list_type_code(const Value& v) {
  if (v.tag == ValueTag::IntList) return 2U;
  if (v.tag == ValueTag::FloatList) return 3U;
  return 4U;
}

__device__ inline Value d_make_list_hash_len(ValueTag tag, std::uint64_t h, std::uint32_t len) {
  if (tag == ValueTag::IntList) return Value::from_int_list_hash_len(h, len);
  if (tag == ValueTag::FloatList) return Value::from_float_list_hash_len(h, len);
  return Value::from_string_list_hash_len(h, len);
}

__device__ inline std::uint64_t d_prepend_list_hash48(std::uint8_t type_code,
                                                      const Value& elem,
                                                      const Value& src) {
  std::uint64_t h = Value::fnv1a_init();
  h = Value::fnv1a_mix_u8(h, type_code);
  h = Value::fnv1a_mix_u8(h, 0x70U);
  h = Value::fnv1a_mix_u64(h, Value::shallow_hash64(elem));
  h = Value::fnv1a_mix_u64(h, Value::container_hash48(src));
  h = Value::fnv1a_mix_u64(h, static_cast<std::uint64_t>(Value::container_len(src)));
  return (h & Value::k_container_hash_mask);
}

__device__ inline bool d_list_accepts_elem(ValueTag list_tag, const Value& elem) {
  if (list_tag == ValueTag::IntList) return elem.tag == ValueTag::Int;
  if (list_tag == ValueTag::FloatList) return elem.tag == ValueTag::Float;
  if (list_tag == ValueTag::StringList) return elem.tag == ValueTag::String;
  return false;
}

__device__ inline bool d_is_ascii_letter(long long c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

__device__ inline bool d_is_ascii_digit(long long c) {
  return c >= '0' && c <= '9';
}

__device__ inline bool d_is_ascii_space(long long c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

__device__ inline bool d_is_ascii_vowel(long long c) {
  if (c >= 'A' && c <= 'Z') c += ('a' - 'A');
  return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
}

__device__ inline long long d_ascii_to_lower(long long c) {
  return (c >= 'A' && c <= 'Z') ? c + ('a' - 'A') : c;
}

__device__ inline long long d_ascii_to_upper(long long c) {
  return (c >= 'a' && c <= 'z') ? c - ('a' - 'A') : c;
}

__device__ inline int d_append_int_decimal(char* out, long long value) {
  int n = 0;
  unsigned long long mag = 0;
  if (value < 0) {
    out[n++] = '-';
    mag = static_cast<unsigned long long>(-(value + 1)) + 1ULL;
  } else {
    mag = static_cast<unsigned long long>(value);
  }
  char digits[32];
  int digit_count = 0;
  do {
    digits[digit_count++] = static_cast<char>('0' + (mag % 10ULL));
    mag /= 10ULL;
  } while (mag != 0ULL);
  for (int i = digit_count - 1; i >= 0; --i) {
    out[n++] = digits[i];
  }
  return n;
}

__device__ inline int d_append_literal(char* out, const char* value) {
  int n = 0;
  while (value[n] != '\0') {
    out[n] = value[n];
    ++n;
  }
  return n;
}

__device__ inline int d_append_float_decimal(char* out, double value) {
  if (value != value) {
    return d_append_literal(out, "nan");
  }
  if (value > 1.7976931348623157e308) {
    return d_append_literal(out, "inf");
  }
  if (value < -1.7976931348623157e308) {
    return d_append_literal(out, "-inf");
  }
  int n = 0;
  if (value < 0.0) {
    out[n++] = '-';
    value = -value;
  }
  long long whole = static_cast<long long>(value);
  double frac = value - static_cast<double>(whole);
  long long frac_scaled = static_cast<long long>(frac * 1000000.0 + 0.5);
  if (frac_scaled >= 1000000LL) {
    ++whole;
    frac_scaled -= 1000000LL;
  }
  n += d_append_int_decimal(out + n, whole);
  if (frac_scaled == 0LL) {
    if (n == 2 && out[0] == '-' && out[1] == '0') {
      out[0] = '0';
      return 1;
    }
    return n;
  }
  out[n++] = '.';
  int frac_start = n;
  char digits[6];
  for (int i = 5; i >= 0; --i) {
    digits[i] = static_cast<char>('0' + (frac_scaled % 10LL));
    frac_scaled /= 10LL;
  }
  for (int i = 0; i < 6; ++i) {
    out[n++] = digits[i];
  }
  while (n > frac_start && out[n - 1] == '0') {
    --n;
  }
  if (n == frac_start) {
    --n;
  }
  return n;
}

__device__ inline int d_find_list_payload_entry(const DPayloadTables& tables, ValueTag tag, std::int64_t packed) {
  int lo = 0;
  int hi = tables.list_entry_count;
  while (lo < hi) {
    const int mid = lo + ((hi - lo) >> 1);
    const DListPayloadEntry mid_entry = tables.list_entries[mid];
    const bool go_right =
        (static_cast<int>(mid_entry.tag) < static_cast<int>(tag)) ||
        (mid_entry.tag == tag && mid_entry.packed < packed);
    if (go_right) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (lo >= tables.list_entry_count || tables.list_entries[lo].tag != tag || tables.list_entries[lo].packed != packed) {
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
  if (!d_is_typed_list_tag(v.tag)) return false;
  if constexpr (State::kMaxListEntries > 0) {
    for (int i = 0; i < st.list_entry_count; ++i) {
      if (st.list_entries[i].tag == v.tag && st.list_entries[i].packed == v.i) {
        ptr = st.list_values + st.list_entries[i].offset;
        len = st.list_entries[i].len;
        return true;
      }
    }
  }
  const int idx = d_find_list_payload_entry(tables, v.tag, v.i);
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
  st.list_entries[st.list_entry_count++] = DListPayloadEntry{v.tag, v.i, offset, len};
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
    if (x.tag != lo.tag || x.tag != hi.tag) {
      err = ErrCode::Type;
      return false;
    }
    const bool any_float =
        (x.tag == ValueTag::Float) || (lo.tag == ValueTag::Float) || (hi.tag == ValueTag::Float);
    if (any_float) {
      const double x2 = (x.tag == ValueTag::Float) ? x.f : static_cast<double>(x.i);
      const double lo2 = (lo.tag == ValueTag::Float) ? lo.f : static_cast<double>(lo.i);
      const double hi2 = (hi.tag == ValueTag::Float) ? hi.f : static_cast<double>(hi.i);
      const double lower = (x2 > lo2) ? x2 : lo2;
      const double picked = (lower < hi2) ? lower : hi2;
      out = Value::from_float(vm_semantics::canonicalize_vm_float(picked));
      return true;
    }
    const long long x2 = x.i;
    const long long lo2 = lo.i;
    const long long hi2 = hi.i;
    const long long lower = (x2 > lo2) ? x2 : lo2;
    out = Value::from_int((lower < hi2) ? lower : hi2);
    return true;
  }

  if (bid == BuiltinId::IDiv0 || bid == BuiltinId::IMod0) {
    if (argc != 2) {
      err = ErrCode::Type;
      return false;
    }
    const Value& a = args[0];
    const Value& b = args[1];
    if (a.tag != ValueTag::Int || b.tag != ValueTag::Int) {
      err = ErrCode::Type;
      return false;
    }
    if (b.i == 0) {
      out = Value::from_int(0);
    } else if (bid == BuiltinId::IDiv0) {
      out = Value::from_int(a.i / b.i);
    } else {
      out = Value::from_int(d_int_mod(a.i, b.i));
    }
    return true;
  }

  if (bid == BuiltinId::Len) {
    if (argc != 1) {
      err = ErrCode::Type;
      return false;
    }
    const Value& x = args[0];
    if (!(x.tag == ValueTag::String || d_is_typed_list_tag(x.tag))) {
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
    if (d_is_typed_list_tag(a.tag) && a.tag == b.tag) {
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
          out = d_make_list_hash_len(a.tag, h, static_cast<std::uint32_t>(al + bl));
          (void)d_register_local_list(st, out, off, al + bl);
          return true;
        }
      }
      const std::uint32_t len = Value::saturating_len_add(Value::container_len(a), Value::container_len(b));
      const std::uint64_t h = Value::combine_container_hash48(d_list_type_code(a), a, b);
      out = d_make_list_hash_len(a.tag, h, len);
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
    if (!(x.tag == ValueTag::String || d_is_typed_list_tag(x.tag))) {
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
      const std::uint64_t out_h = Value::slice_container_hash48(1U, x, lo.i, hi.i);
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
        out = d_make_list_hash_len(x.tag, out_h_exact, out_len);
        (void)d_register_local_list(st, out, off, static_cast<int>(out_len));
        return true;
      }
    }
    const std::uint64_t out_h = Value::slice_container_hash48(d_list_type_code(x), x, lo.i, hi.i);
    out = d_make_list_hash_len(x.tag, out_h, out_len);
    return true;
  }

  if (bid == BuiltinId::Index) {
    if (argc != 2) {
      err = ErrCode::Type;
      return false;
    }
    const Value& x = args[0];
    const Value& i = args[1];
    if (!(x.tag == ValueTag::String || d_is_typed_list_tag(x.tag))) {
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
        if (d_lookup_string_payload(tables, st, x, xp, xl) && j < xl) {
          out = Value::from_char(static_cast<unsigned char>(xp[static_cast<int>(j)]));
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
    out = Value::from_fallback_token(Value::index_container_token64(d_list_type_code(x), x, j));
    return true;
  }

  if (bid == BuiltinId::Append) {
    if (argc != 2) {
      err = ErrCode::Type;
      return false;
    }
    const Value& xs = args[0];
    const Value& elem = args[1];
    if (!d_is_typed_list_tag(xs.tag)) {
      err = ErrCode::Type;
      return false;
    }
    if (!d_list_accepts_elem(xs.tag, elem)) {
      err = ErrCode::Type;
      return false;
    }
    if constexpr (PayloadTraits::kHasList) {
      auto& st = payload_state;
      const Value* xp = nullptr;
      int xl = 0;
      if (d_lookup_list_payload(tables, st, xs, xp, xl) && st.list_values_used + xl + 1 <= st.kMaxListValues) {
        const int off = st.list_values_used;
        for (int i = 0; i < xl; ++i) st.list_values[off + i] = xp[i];
        st.list_values[off + xl] = elem;
        st.list_values_used += xl + 1;
        const std::uint64_t h = d_hash_list_payload(st.list_values + off, xl + 1);
        out = d_make_list_hash_len(xs.tag, h, static_cast<std::uint32_t>(xl + 1));
        (void)d_register_local_list(st, out, off, xl + 1);
        return true;
      }
    }
    const std::uint32_t len = Value::saturating_len_add(Value::container_len(xs), 1U);
    const std::uint64_t h = Value::append_list_hash48(d_list_type_code(xs), xs, elem);
    out = d_make_list_hash_len(xs.tag, h, len);
    return true;
  }

  if (bid == BuiltinId::Prepend) {
    if (argc != 2) {
      err = ErrCode::Type;
      return false;
    }
    const Value& xs = args[0];
    const Value& elem = args[1];
    if (!d_is_typed_list_tag(xs.tag) || !d_list_accepts_elem(xs.tag, elem)) {
      err = ErrCode::Type;
      return false;
    }
    if constexpr (PayloadTraits::kHasList) {
      auto& st = payload_state;
      const Value* xp = nullptr;
      int xl = 0;
      if (d_lookup_list_payload(tables, st, xs, xp, xl) && st.list_values_used + xl + 1 <= st.kMaxListValues) {
        const int off = st.list_values_used;
        st.list_values[off] = elem;
        for (int i = 0; i < xl; ++i) st.list_values[off + 1 + i] = xp[i];
        st.list_values_used += xl + 1;
        const std::uint64_t h = d_hash_list_payload(st.list_values + off, xl + 1);
        out = d_make_list_hash_len(xs.tag, h, static_cast<std::uint32_t>(xl + 1));
        (void)d_register_local_list(st, out, off, xl + 1);
        return true;
      }
    }
    const std::uint32_t len = Value::saturating_len_add(Value::container_len(xs), 1U);
    const std::uint64_t h = d_prepend_list_hash48(d_list_type_code(xs), elem, xs);
    out = d_make_list_hash_len(xs.tag, h, len);
    return true;
  }

  if (bid == BuiltinId::Reverse) {
    if (argc != 1) {
      err = ErrCode::Type;
      return false;
    }
    const Value& x = args[0];
    if (x.tag == ValueTag::String) {
      if constexpr (PayloadTraits::kHasString) {
        auto& st = payload_state;
        const char* xp = nullptr;
        int xl = 0;
        if (d_lookup_string_payload(tables, st, x, xp, xl) && st.string_bytes_used + xl <= st.kMaxStringBytes) {
          const int off = st.string_bytes_used;
          for (int i = 0; i < xl; ++i) {
            st.string_bytes[off + i] = xp[xl - 1 - i];
          }
          st.string_bytes_used += xl;
          const std::uint64_t h = d_hash_bytes(st.string_bytes + off, xl);
          out = Value::from_string_hash_len(h, static_cast<std::uint32_t>(xl));
          (void)d_register_local_string(st, out, off, xl);
          return true;
        }
      }
      out = Value::from_fallback_token(Value::pack_container_payload(Value::reverse_container_hash48(7U, x),
                                                                     Value::container_len(x)));
      return true;
    }
    if (!d_is_typed_list_tag(x.tag)) {
      err = ErrCode::Type;
      return false;
    }
    if constexpr (PayloadTraits::kHasList) {
      auto& st = payload_state;
      const Value* xp = nullptr;
      int xl = 0;
      if (d_lookup_list_payload(tables, st, x, xp, xl) && st.list_values_used + xl <= st.kMaxListValues) {
        const int off = st.list_values_used;
        for (int i = 0; i < xl; ++i) {
          st.list_values[off + i] = xp[xl - 1 - i];
        }
        st.list_values_used += xl;
        const std::uint64_t h = d_hash_list_payload(st.list_values + off, xl);
        out = d_make_list_hash_len(x.tag, h, static_cast<std::uint32_t>(xl));
        (void)d_register_local_list(st, out, off, xl);
        return true;
      }
    }
    out = d_make_list_hash_len(x.tag, Value::reverse_container_hash48(d_list_type_code(x), x), Value::container_len(x));
    return true;
  }

  if (bid == BuiltinId::Find || bid == BuiltinId::Contains) {
    if (argc != 2) {
      err = ErrCode::Type;
      return false;
    }
    const Value& haystack = args[0];
    const Value& needle = args[1];
    if (haystack.tag != ValueTag::String || needle.tag != ValueTag::String) {
      err = ErrCode::Type;
      return false;
    }
    if constexpr (PayloadTraits::kHasString) {
      auto& st = payload_state;
      const char* hp = nullptr;
      const char* np = nullptr;
      int hl = 0;
      int nl = 0;
      if (d_lookup_string_payload(tables, st, haystack, hp, hl) &&
          d_lookup_string_payload(tables, st, needle, np, nl)) {
        int found = -1;
        if (nl == 0) {
          found = 0;
        } else if (nl <= hl) {
          for (int i = 0; i <= hl - nl; ++i) {
            bool match = true;
            for (int j = 0; j < nl; ++j) {
              if (hp[i + j] != np[j]) {
                match = false;
                break;
              }
            }
            if (match) {
              found = i;
              break;
            }
          }
        }
        out = (bid == BuiltinId::Find) ? Value::from_int(found) : Value::from_bool(found >= 0);
        return true;
      }
    }
    err = ErrCode::Value;
    return false;
  }

  if (bid == BuiltinId::IsInt) {
    if (argc != 1) {
      err = ErrCode::Type;
      return false;
    }
    out = Value::from_bool(args[0].tag == ValueTag::Int);
    return true;
  }

  if (bid == BuiltinId::CharToString) {
    if (argc != 1 || args[0].tag != ValueTag::Char || args[0].i < 0 || args[0].i > 255) {
      err = (argc == 1 && args[0].tag == ValueTag::Char) ? ErrCode::Value : ErrCode::Type;
      return false;
    }
    if constexpr (PayloadTraits::kHasString) {
      auto& st = payload_state;
      if (st.string_bytes_used + 1 <= st.kMaxStringBytes) {
        const int off = st.string_bytes_used;
        st.string_bytes[off] = static_cast<char>(args[0].i);
        st.string_bytes_used += 1;
        out = Value::from_string_hash_len(d_hash_bytes(st.string_bytes + off, 1), 1U);
        (void)d_register_local_string(st, out, off, 1);
        return true;
      }
    }
    const char c = static_cast<char>(args[0].i);
    out = Value::from_string_hash_len(d_hash_bytes(&c, 1), 1U);
    return true;
  }

  if (bid == BuiltinId::StringToChar) {
    if (argc != 1 || args[0].tag != ValueTag::String) {
      err = ErrCode::Type;
      return false;
    }
    if (Value::container_len(args[0]) != 1U) {
      err = ErrCode::Value;
      return false;
    }
    if constexpr (PayloadTraits::kHasString) {
      auto& st = payload_state;
      const char* sp = nullptr;
      int sl = 0;
      if (d_lookup_string_payload(tables, st, args[0], sp, sl) && sl == 1) {
        out = Value::from_char(static_cast<unsigned char>(sp[0]));
        return true;
      }
    }
    err = ErrCode::Value;
    return false;
  }

  if (bid == BuiltinId::Ord) {
    if (argc != 1 || args[0].tag != ValueTag::Char) {
      err = ErrCode::Type;
      return false;
    }
    out = Value::from_int(args[0].i);
    return true;
  }

  if (bid == BuiltinId::Chr) {
    if (argc != 1 || args[0].tag != ValueTag::Int) {
      err = ErrCode::Type;
      return false;
    }
    if (args[0].i < 0 || args[0].i > 255) {
      err = ErrCode::Value;
      return false;
    }
    out = Value::from_char(args[0].i);
    return true;
  }

  if (bid == BuiltinId::IsLetter || bid == BuiltinId::IsDigit || bid == BuiltinId::IsSpace ||
      bid == BuiltinId::IsVowel || bid == BuiltinId::ToLower || bid == BuiltinId::ToUpper) {
    if (argc != 1 || args[0].tag != ValueTag::Char) {
      err = ErrCode::Type;
      return false;
    }
    const long long c = args[0].i;
    if (bid == BuiltinId::IsLetter) out = Value::from_bool(d_is_ascii_letter(c));
    else if (bid == BuiltinId::IsDigit) out = Value::from_bool(d_is_ascii_digit(c));
    else if (bid == BuiltinId::IsSpace) out = Value::from_bool(d_is_ascii_space(c));
    else if (bid == BuiltinId::IsVowel) out = Value::from_bool(d_is_ascii_vowel(c));
    else if (bid == BuiltinId::ToLower) out = Value::from_char(d_ascii_to_lower(c));
    else out = Value::from_char(d_ascii_to_upper(c));
    return true;
  }

  if (bid == BuiltinId::ToString) {
    if (argc != 1 || (args[0].tag != ValueTag::Int && args[0].tag != ValueTag::Float)) {
      err = ErrCode::Type;
      return false;
    }
    char tmp[64];
    const int len = args[0].tag == ValueTag::Int
                        ? d_append_int_decimal(tmp, args[0].i)
                        : d_append_float_decimal(tmp, args[0].f);
    if constexpr (PayloadTraits::kHasString) {
      auto& st = payload_state;
      if (st.string_bytes_used + len <= st.kMaxStringBytes) {
        const int off = st.string_bytes_used;
        for (int i = 0; i < len; ++i) {
          st.string_bytes[off + i] = tmp[i];
        }
        st.string_bytes_used += len;
        out = Value::from_string_hash_len(d_hash_bytes(st.string_bytes + off, len),
                                          static_cast<std::uint32_t>(len));
        (void)d_register_local_string(st, out, off, len);
        return true;
      }
    }
    out = Value::from_string_hash_len(d_hash_bytes(tmp, len), static_cast<std::uint32_t>(len));
    return true;
  }

  if (bid == BuiltinId::Singleton) {
    if (argc != 1) {
      err = ErrCode::Type;
      return false;
    }
    const Value& x = args[0];
    if (x.tag == ValueTag::Char) {
      if (x.i < 0 || x.i > 255) {
        err = ErrCode::Value;
        return false;
      }
      if constexpr (PayloadTraits::kHasString) {
        auto& st = payload_state;
        if (st.string_bytes_used + 1 <= st.kMaxStringBytes) {
          const int off = st.string_bytes_used;
          st.string_bytes[off] = static_cast<char>(x.i);
          st.string_bytes_used += 1;
          out = Value::from_string_hash_len(d_hash_bytes(st.string_bytes + off, 1), 1U);
          (void)d_register_local_string(st, out, off, 1);
          return true;
        }
      }
      const char c = static_cast<char>(x.i);
      out = Value::from_string_hash_len(d_hash_bytes(&c, 1), 1U);
      return true;
    }
    if (x.tag == ValueTag::Int || x.tag == ValueTag::Float || x.tag == ValueTag::String) {
      const ValueTag list_tag = (x.tag == ValueTag::Int)
                                    ? ValueTag::IntList
                                    : ((x.tag == ValueTag::Float) ? ValueTag::FloatList : ValueTag::StringList);
      if constexpr (PayloadTraits::kHasList) {
        auto& st = payload_state;
        if (st.list_values_used + 1 <= st.kMaxListValues) {
          const int off = st.list_values_used;
          st.list_values[off] = x;
          st.list_values_used += 1;
          out = d_make_list_hash_len(list_tag, d_hash_list_payload(st.list_values + off, 1), 1U);
          (void)d_register_local_list(st, out, off, 1);
          return true;
        }
      }
      std::uint64_t h = Value::fnv1a_init();
      h = Value::fnv1a_mix_u8(h, 0xA1U);
      h = Value::fnv1a_mix_u64(h, 1U);
      h = Value::fnv1a_mix_u64(h, d_hash_value_shallow(x));
      out = d_make_list_hash_len(list_tag, h, 1U);
      return true;
    }
    err = ErrCode::Type;
    return false;
  }

  err = ErrCode::Name;
  return false;
}

}  // namespace gagp::gpu_detail
