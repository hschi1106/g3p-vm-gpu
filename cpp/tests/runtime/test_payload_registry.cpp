#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "g3pvm/core/value.hpp"
#include "g3pvm/runtime/payload/payload.hpp"

namespace {

using g3pvm::Value;

bool check(bool condition, const std::string& message) {
  if (!condition) std::cerr << "FAIL: " << message << "\n";
  return condition;
}

bool test_retain_only_keeps_live_closure() {
  g3pvm::payload::clear();
  const Value keep_a = g3pvm::payload::make_string_value("keep-a");
  const Value keep_b = g3pvm::payload::make_string_value("keep-b");
  const Value drop_s = g3pvm::payload::make_string_value("drop-s");
  const Value keep_list = g3pvm::payload::make_string_list_value({keep_a, keep_b});
  const Value drop_list =
      g3pvm::payload::make_int_list_value({Value::from_int(7), Value::from_int(9)});

  g3pvm::payload::retain_only({keep_list});

  std::string exact;
  if (!check(g3pvm::payload::lookup_string(keep_a, &exact) && exact == "keep-a",
             "retain_only should keep transitive string payloads")) return false;
  if (!check(g3pvm::payload::lookup_string(keep_b, &exact) && exact == "keep-b",
             "retain_only should keep all transitive string payloads")) return false;
  if (!check(!g3pvm::payload::lookup_string(drop_s, &exact),
             "retain_only should drop unreferenced string payloads")) return false;

  std::vector<Value> elems;
  if (!check(g3pvm::payload::lookup_list(keep_list, &elems) && elems.size() == 2,
             "retain_only should keep live list payloads")) return false;
  if (!check(!g3pvm::payload::lookup_list(drop_list, &elems),
             "retain_only should drop unreferenced list payloads")) return false;

  const g3pvm::payload::PayloadStats stats = g3pvm::payload::stats();
  return check(stats.string_entries == 2 && stats.list_entries == 1,
               "retain_only registry statistics mismatch");
}

bool test_typed_list_construction_rejects_wrong_tags() {
  g3pvm::payload::clear();
  bool rejected = false;
  try {
    (void)g3pvm::payload::make_int_list_value({Value::from_float(1.0)});
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  if (!check(rejected, "IntList should reject Float elements")) return false;

  rejected = false;
  try {
    (void)g3pvm::payload::make_float_list_value({Value::from_int(1)});
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  if (!check(rejected, "FloatList should reject Int elements")) return false;

  rejected = false;
  try {
    (void)g3pvm::payload::make_string_list_value({Value::from_char('x')});
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  return check(rejected, "StringList should reject Char elements");
}

bool test_packed_lookup_uses_exact_typed_list_tag() {
  g3pvm::payload::clear();
  constexpr std::uint64_t hash = 0x12345ULL;
  constexpr std::uint32_t length = 2U;
  const Value int_key = Value::from_int_list_hash_len(hash, length);
  const Value float_key = Value::from_float_list_hash_len(hash, length);
  const Value string_key = Value::from_string_list_hash_len(hash, length);

  g3pvm::payload::register_list(int_key, {Value::from_int(10), Value::from_int(20)});
  std::vector<Value> elems;
  if (!check(g3pvm::payload::lookup_list_packed(g3pvm::ValueTag::IntList, int_key.i,
                                                &elems),
             "matching IntList token should resolve")) return false;
  if (!check(!g3pvm::payload::lookup_list_packed(g3pvm::ValueTag::FloatList, int_key.i,
                                                 &elems),
             "FloatList must not alias an IntList token")) return false;

  g3pvm::payload::register_list(float_key,
                                {Value::from_float(1.5), Value::from_float(2.5)});
  const Value s0 = g3pvm::payload::make_string_value("a");
  const Value s1 = g3pvm::payload::make_string_value("b");
  g3pvm::payload::register_list(string_key, {s0, s1});
  if (!check(g3pvm::payload::lookup_list_packed(g3pvm::ValueTag::FloatList, float_key.i,
                                                &elems) &&
                 elems[0].tag == g3pvm::ValueTag::Float,
             "matching FloatList token should resolve")) return false;
  if (!check(g3pvm::payload::lookup_list_packed(g3pvm::ValueTag::StringList, string_key.i,
                                                &elems) &&
                 elems[0].tag == g3pvm::ValueTag::String,
             "matching StringList token should resolve")) return false;
  return check(g3pvm::payload::stats().list_entries == 3,
               "registry should retain one entry per typed-list tag");
}

}  // namespace

int main() {
  if (!test_retain_only_keeps_live_closure()) return 1;
  if (!test_typed_list_construction_rejects_wrong_tags()) return 1;
  if (!test_packed_lookup_uses_exact_typed_list_tag()) return 1;
  std::cout << "g3pvm_test_payload_registry: OK\n";
  return 0;
}
