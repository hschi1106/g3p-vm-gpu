#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "g3pvm/cli/codec.hpp"
#include "g3pvm/cli/json.hpp"

namespace {

bool check(bool cond, const std::string& msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << "\n";
    return false;
  }
  return true;
}

bool test_subnormal_number_is_accepted() {
  const g3pvm::cli_detail::JsonValue root =
      g3pvm::cli_detail::JsonParser("{\"x\":3.340886621450795e-309}").parse();
  const auto it = root.object_v.find("x");
  if (!check(it != root.object_v.end(), "subnormal field missing")) {
    return false;
  }
  if (!check(it->second.kind == g3pvm::cli_detail::JsonValue::Kind::Number,
             "subnormal field should parse as number")) {
    return false;
  }
  if (!check(it->second.number_v > 0.0, "subnormal number should not be rounded to zero")) {
    return false;
  }
  return true;
}

bool test_overflow_number_is_rejected() {
  try {
    (void)g3pvm::cli_detail::JsonParser("{\"x\":1e9999}").parse();
  } catch (const std::runtime_error& err) {
    return std::string(err.what()).find("out of range") != std::string::npos;
  }
  std::cerr << "FAIL: overflowing JSON number should be rejected\n";
  return false;
}

bool decode_programs_rejects(const std::string& json, const std::string& needle) {
  try {
    const g3pvm::cli_detail::JsonValue root = g3pvm::cli_detail::JsonParser(json).parse();
    (void)g3pvm::cli_detail::decode_programs(root);
  } catch (const std::runtime_error& err) {
    return std::string(err.what()).find(needle) != std::string::npos;
  }
  std::cerr << "FAIL: decode_programs should reject " << needle << "\n";
  return false;
}

bool test_bytecode_asgp_dp_segment_arity_is_validated() {
  const std::string phase = R"({"n_locals":0,"consts":[],"code":[]})";

  const std::string bad_dp1 = R"([
    {
      "n_locals": 0,
      "consts": [],
      "code": [],
      "segments": {
        "asgp_dp1d": [
          {
            "lo": 0,
            "hi": 3,
            "base_state": 0,
            "boundary_value": {"type":"int","value":0},
            "dep_kind": -1,
            "dep_offsets": [1, 2],
            "solve_state_name": 0,
            "transition_state_name": 1,
            "transition_dep_names": [2],
            "solve": )" + phase + R"(,
            "transition": )" + phase + R"(
          }
        ]
      }
    }
  ])";
  if (!decode_programs_rejects(bad_dp1, "ASGP-DP1D dependency arity mismatch")) {
    return false;
  }

  const std::string bad_dp2 = R"([
    {
      "n_locals": 0,
      "consts": [],
      "code": [],
      "segments": {
        "asgp_dp2d": [
          {
            "i_lo": 0,
            "i_hi": 3,
            "j_lo": 0,
            "j_hi": 3,
            "base_i": 0,
            "base_j": 0,
            "boundary_value": {"type":"int","value":0},
            "dep_kind": 0,
            "solve_i_name": 0,
            "solve_j_name": 1,
            "transition_i_name": 2,
            "transition_j_name": 3,
            "transition_dep_names": [4],
            "solve": )" + phase + R"(,
            "transition": )" + phase + R"(
          }
        ]
      }
    }
  ])";
  if (!decode_programs_rejects(bad_dp2, "ASGP-DP2D dependency arity mismatch")) {
    return false;
  }

  return true;
}

bool test_bytecode_verifier_runs_at_decode_boundary() {
  const std::string valid = R"([
    {
      "n_locals": 0,
      "consts": [{"type":"int","value":1}],
      "code": [{"op":"PUSH_CONST","a":0},{"op":"RETURN"}]
    }
  ])";
  try {
    const auto root = g3pvm::cli_detail::JsonParser(valid).parse();
    if (!check(g3pvm::cli_detail::decode_programs(root).size() == 1,
               "valid bytecode should decode")) return false;
  } catch (const std::runtime_error& err) {
    std::cerr << "FAIL: valid bytecode decode failed: " << err.what() << "\n";
    return false;
  }

  const std::string underflow = R"([
    {"n_locals":0,"consts":[],"code":[{"op":"RETURN"}]}
  ])";
  if (!decode_programs_rejects(underflow, "stack_underflow")) return false;

  const std::string fallthrough = R"([
    {
      "n_locals":0,
      "consts":[{"type":"int","value":1}],
      "code":[{"op":"PUSH_CONST","a":0}]
    }
  ])";
  if (!decode_programs_rejects(fallthrough, "invalid_fallthrough")) return false;

  const std::string runtime_type_error = R"([
    {
      "n_locals":0,
      "consts":[{"type":"bool","value":true}],
      "code":[{"op":"PUSH_CONST","a":0},{"op":"NEG"}]
    }
  ])";
  try {
    const auto root = g3pvm::cli_detail::JsonParser(runtime_type_error).parse();
    if (!check(g3pvm::cli_detail::decode_programs(root).size() == 1,
               "runtime TypeError bytecode should remain decodable")) return false;
  } catch (const std::runtime_error& err) {
    std::cerr << "FAIL: runtime error bytecode was treated as malformed: " << err.what() << "\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!test_subnormal_number_is_accepted()) return 1;
  if (!test_overflow_number_is_rejected()) return 1;
  if (!test_bytecode_asgp_dp_segment_arity_is_validated()) return 1;
  if (!test_bytecode_verifier_runs_at_decode_boundary()) return 1;
  std::cout << "g3pvm_test_cli_json: OK\n";
  return 0;
}
