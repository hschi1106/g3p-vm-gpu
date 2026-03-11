#pragma once

#include <cstdint>
#include <type_traits>

namespace g3pvm {

#ifdef __CUDACC__
#define G3PVM_HD __host__ __device__
#else
#define G3PVM_HD
#endif

enum class ValueTag : std::uint8_t {
  Int,
  Float,
  Bool,
  None,
  String,
  List,
};

struct Value {
  union {
    std::int64_t i;
    double f;
  };
  bool b = false;
  ValueTag tag = ValueTag::None;

  G3PVM_HD static Value from_int(std::int64_t v) {
    Value out;
    out.i = 0;
    out.tag = ValueTag::Int;
    out.i = v;
    return out;
  }

  G3PVM_HD static Value from_float(double v) {
    Value out;
    out.i = 0;
    out.tag = ValueTag::Float;
    out.f = v;
    return out;
  }

  G3PVM_HD static Value from_bool(bool v) {
    Value out;
    out.i = 0;
    out.tag = ValueTag::Bool;
    out.b = v;
    return out;
  }

  G3PVM_HD static Value none() {
    Value out;
    out.i = 0;
    out.b = false;
    out.tag = ValueTag::None;
    return out;
  }

  // String/List use a compact packed token in the base Value representation:
  // upper 16 bits store a saturated length, lower 48 bits store a deterministic hash.
  // Full container payloads live in higher runtime layers keyed by this packed token.
  static constexpr std::uint64_t k_container_hash_mask = (1ULL << 48) - 1ULL;
  static constexpr std::uint32_t k_container_len_max = 0xFFFFU;

  G3PVM_HD static std::uint64_t fnv1a_init() { return 1469598103934665603ULL; }

  G3PVM_HD static std::uint64_t fnv1a_mix_u8(std::uint64_t h, std::uint8_t b) {
    h ^= static_cast<std::uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
  }

  G3PVM_HD static std::uint64_t fnv1a_mix_u64(std::uint64_t h, std::uint64_t x) {
    for (int i = 0; i < 8; ++i) {
      const std::uint8_t b = static_cast<std::uint8_t>((x >> (8 * i)) & 0xFFULL);
      h = fnv1a_mix_u8(h, b);
    }
    return h;
  }

  G3PVM_HD static std::uint32_t saturating_len_add(std::uint32_t a, std::uint32_t b) {
    const std::uint32_t s = a + b;
    if (s < a || s > k_container_len_max) return k_container_len_max;
    return s;
  }

  // Combine two container payloads of the same tag into a deterministic hash48.
  // type_code: 1 for string, 2 for list.
  G3PVM_HD static std::uint64_t combine_container_hash48(std::uint8_t type_code, const Value& a, const Value& b) {
    std::uint64_t h = fnv1a_init();
    h = fnv1a_mix_u8(h, type_code);
    h = fnv1a_mix_u64(h, container_hash48(a));
    h = fnv1a_mix_u64(h, static_cast<std::uint64_t>(container_len(a)));
    h = fnv1a_mix_u64(h, container_hash48(b));
    h = fnv1a_mix_u64(h, static_cast<std::uint64_t>(container_len(b)));
    return (h & k_container_hash_mask);
  }

  // Build a deterministic hash48 for slice(container, lo, hi).
  G3PVM_HD static std::uint64_t slice_container_hash48(std::uint8_t type_code, const Value& src, std::int64_t lo, std::int64_t hi) {
    std::uint64_t h = fnv1a_init();
    h = fnv1a_mix_u8(h, type_code);
    h = fnv1a_mix_u64(h, container_hash48(src));
    h = fnv1a_mix_u64(h, static_cast<std::uint64_t>(container_len(src)));
    h = fnv1a_mix_u64(h, static_cast<std::uint64_t>(lo));
    h = fnv1a_mix_u64(h, static_cast<std::uint64_t>(hi));
    return (h & k_container_hash_mask);
  }

  // Build a deterministic int token for index(container, i) in hash+len transport mode.
  G3PVM_HD static std::int64_t index_container_token64(std::uint8_t type_code, const Value& src, std::int64_t idx) {
    std::uint64_t h = fnv1a_init();
    h = fnv1a_mix_u8(h, type_code);
    h = fnv1a_mix_u64(h, container_hash48(src));
    h = fnv1a_mix_u64(h, static_cast<std::uint64_t>(container_len(src)));
    h = fnv1a_mix_u64(h, static_cast<std::uint64_t>(idx));
    return static_cast<std::int64_t>(h);
  }

  G3PVM_HD static std::int64_t pack_container_payload(std::uint64_t h, std::uint32_t len) {
    const std::uint64_t packed = ((static_cast<std::uint64_t>(len) & 0xFFFFULL) << 48) | (h & k_container_hash_mask);
    return static_cast<std::int64_t>(packed);
  }

  G3PVM_HD static std::uint32_t container_len(const Value& v) {
    const std::uint64_t u = static_cast<std::uint64_t>(v.i);
    return static_cast<std::uint32_t>((u >> 48) & 0xFFFFULL);
  }

  G3PVM_HD static std::uint64_t container_hash48(const Value& v) {
    const std::uint64_t u = static_cast<std::uint64_t>(v.i);
    return (u & k_container_hash_mask);
  }

  G3PVM_HD static Value from_string_hash_len(std::uint64_t h, std::uint32_t len) {
    Value out;
    out.i = pack_container_payload(h, len);
    out.b = false;
    out.tag = ValueTag::String;
    return out;
  }

  G3PVM_HD static Value from_list_hash_len(std::uint64_t h, std::uint32_t len) {
    Value out;
    out.i = pack_container_payload(h, len);
    out.b = false;
    out.tag = ValueTag::List;
    return out;
  }
};

G3PVM_HD inline bool is_numeric(const Value& v) {
  return v.tag == ValueTag::Int || v.tag == ValueTag::Float;
}

static_assert(std::is_trivially_copyable<Value>::value, "Value must be trivially copyable");
static_assert(sizeof(Value) <= 16, "Value should remain compact");

#undef G3PVM_HD

}  // namespace g3pvm
