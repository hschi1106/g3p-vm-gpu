#include "g3pvm/payload.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace g3pvm::payload {

namespace {

struct PayloadKey {
  ValueTag tag = ValueTag::None;
  std::int64_t packed = 0;

  bool operator==(const PayloadKey& other) const {
    return tag == other.tag && packed == other.packed;
  }
};

struct PayloadKeyHash {
  std::size_t operator()(const PayloadKey& k) const {
    const std::uint64_t a = static_cast<std::uint64_t>(k.packed);
    return static_cast<std::size_t>((a * 11400714819323198485ULL) ^
                                    static_cast<std::uint64_t>(k.tag));
  }
};

std::mutex g_mu;
std::unordered_map<PayloadKey, std::string, PayloadKeyHash> g_strings;
std::unordered_map<PayloadKey, std::vector<Value>, PayloadKeyHash> g_lists;

PayloadKey key_of(const Value& v) {
  return PayloadKey{v.tag, v.i};
}

std::uint64_t hash_bytes(const unsigned char* p, std::size_t n) {
  std::uint64_t h = Value::fnv1a_init();
  for (std::size_t i = 0; i < n; ++i) {
    h = Value::fnv1a_mix_u8(h, p[i]);
  }
  return h;
}

std::uint64_t hash_value_shallow(const Value& v) {
  std::uint64_t h = Value::fnv1a_init();
  h = Value::fnv1a_mix_u8(h, static_cast<std::uint8_t>(v.tag));
  if (v.tag == ValueTag::None) return h;
  if (v.tag == ValueTag::Bool) return Value::fnv1a_mix_u8(h, v.b ? 1U : 0U);
  if (v.tag == ValueTag::Int || v.tag == ValueTag::String || v.tag == ValueTag::List) {
    return Value::fnv1a_mix_u64(h, static_cast<std::uint64_t>(v.i));
  }
  union {
    double d;
    std::uint64_t u;
  } bits{};
  bits.d = v.f;
  return Value::fnv1a_mix_u64(h, bits.u);
}

std::uint64_t hash_list_payload(const std::vector<Value>& elems) {
  std::uint64_t h = Value::fnv1a_init();
  h = Value::fnv1a_mix_u8(h, 0xA1U);  // list domain-separator
  h = Value::fnv1a_mix_u64(h, static_cast<std::uint64_t>(elems.size()));
  for (const Value& e : elems) {
    h = Value::fnv1a_mix_u64(h, hash_value_shallow(e));
  }
  return h;
}

}  // namespace

void clear() {
  std::lock_guard<std::mutex> lock(g_mu);
  g_strings.clear();
  g_lists.clear();
}

void register_string(const Value& key, const std::string& s) {
  if (key.tag != ValueTag::String) return;
  std::lock_guard<std::mutex> lock(g_mu);
  g_strings[key_of(key)] = s;
}

void register_list(const Value& key, const std::vector<Value>& elems) {
  if (key.tag != ValueTag::List) return;
  std::lock_guard<std::mutex> lock(g_mu);
  g_lists[key_of(key)] = elems;
}

bool lookup_string(const Value& key, std::string* out) {
  if (key.tag != ValueTag::String || out == nullptr) return false;
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_strings.find(key_of(key));
  if (it == g_strings.end()) return false;
  *out = it->second;
  return true;
}

bool lookup_list(const Value& key, std::vector<Value>* out) {
  if (key.tag != ValueTag::List || out == nullptr) return false;
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_lists.find(key_of(key));
  if (it == g_lists.end()) return false;
  *out = it->second;
  return true;
}

Value make_string_value(const std::string& s) {
  const std::uint64_t h = hash_bytes(reinterpret_cast<const unsigned char*>(s.data()), s.size());
  const std::uint32_t len =
      static_cast<std::uint32_t>(s.size() > static_cast<std::size_t>(Value::k_container_len_max)
                                     ? Value::k_container_len_max
                                     : s.size());
  const Value out = Value::from_string_hash_len(h, len);
  register_string(out, s);
  return out;
}

Value make_list_value(const std::vector<Value>& elems) {
  const std::uint64_t h = hash_list_payload(elems);
  const std::uint32_t len =
      static_cast<std::uint32_t>(elems.size() > static_cast<std::size_t>(Value::k_container_len_max)
                                     ? Value::k_container_len_max
                                     : elems.size());
  const Value out = Value::from_list_hash_len(h, len);
  register_list(out, elems);
  return out;
}

std::vector<StringSnapshot> snapshot_strings() {
  std::lock_guard<std::mutex> lock(g_mu);
  std::vector<StringSnapshot> out;
  out.reserve(g_strings.size());
  for (const auto& kv : g_strings) {
    StringSnapshot s;
    s.key.tag = kv.first.tag;
    s.key.i = kv.first.packed;
    s.data = kv.second;
    out.push_back(std::move(s));
  }
  return out;
}

std::vector<ListSnapshot> snapshot_lists() {
  std::lock_guard<std::mutex> lock(g_mu);
  std::vector<ListSnapshot> out;
  out.reserve(g_lists.size());
  for (const auto& kv : g_lists) {
    ListSnapshot s;
    s.key.tag = kv.first.tag;
    s.key.i = kv.first.packed;
    s.elems = kv.second;
    out.push_back(std::move(s));
  }
  return out;
}

}  // namespace g3pvm::payload
