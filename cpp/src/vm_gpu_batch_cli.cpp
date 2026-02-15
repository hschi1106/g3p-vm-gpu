#include <cctype>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "g3pvm/bytecode.hpp"
#include "g3pvm/errors.hpp"
#include "g3pvm/value.hpp"
#include "g3pvm/vm_gpu.hpp"

namespace {

using g3pvm::BytecodeProgram;
using g3pvm::InputCase;
using g3pvm::Instr;
using g3pvm::LocalBinding;
using g3pvm::Value;
using g3pvm::ValueTag;

struct JsonValue {
  enum class Kind { Null, Bool, Number, String, Array, Object };
  Kind kind = Kind::Null;
  bool bool_v = false;
  double number_v = 0.0;
  std::string string_v;
  std::vector<JsonValue> array_v;
  std::map<std::string, JsonValue> object_v;
};

class JsonParser {
 public:
  explicit JsonParser(std::string text) : text_(std::move(text)) {}

  JsonValue parse() {
    skip_ws();
    JsonValue v = parse_value();
    skip_ws();
    if (pos_ != text_.size()) {
      throw std::runtime_error("trailing characters in JSON");
    }
    return v;
  }

 private:
  JsonValue parse_value() {
    if (pos_ >= text_.size()) throw std::runtime_error("unexpected end of JSON");
    const char c = text_[pos_];
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == '"') return parse_string();
    if (c == 't') return parse_true();
    if (c == 'f') return parse_false();
    if (c == 'n') return parse_null();
    if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_number();
    throw std::runtime_error("invalid JSON token");
  }

  JsonValue parse_object() {
    expect('{');
    JsonValue out;
    out.kind = JsonValue::Kind::Object;
    skip_ws();
    if (peek('}')) {
      expect('}');
      return out;
    }
    while (true) {
      skip_ws();
      JsonValue key = parse_string();
      skip_ws();
      expect(':');
      skip_ws();
      JsonValue val = parse_value();
      out.object_v.emplace(key.string_v, std::move(val));
      skip_ws();
      if (peek('}')) {
        expect('}');
        break;
      }
      expect(',');
    }
    return out;
  }

  JsonValue parse_array() {
    expect('[');
    JsonValue out;
    out.kind = JsonValue::Kind::Array;
    skip_ws();
    if (peek(']')) {
      expect(']');
      return out;
    }
    while (true) {
      skip_ws();
      out.array_v.push_back(parse_value());
      skip_ws();
      if (peek(']')) {
        expect(']');
        break;
      }
      expect(',');
    }
    return out;
  }

  JsonValue parse_string() {
    expect('"');
    JsonValue out;
    out.kind = JsonValue::Kind::String;
    while (pos_ < text_.size()) {
      char c = text_[pos_++];
      if (c == '"') return out;
      if (c == '\\') {
        if (pos_ >= text_.size()) throw std::runtime_error("invalid JSON escape");
        char e = text_[pos_++];
        if (e == '"' || e == '\\' || e == '/') out.string_v.push_back(e);
        else if (e == 'b') out.string_v.push_back('\b');
        else if (e == 'f') out.string_v.push_back('\f');
        else if (e == 'n') out.string_v.push_back('\n');
        else if (e == 'r') out.string_v.push_back('\r');
        else if (e == 't') out.string_v.push_back('\t');
        else throw std::runtime_error("unsupported JSON escape");
      } else {
        out.string_v.push_back(c);
      }
    }
    throw std::runtime_error("unterminated JSON string");
  }

  JsonValue parse_number() {
    std::size_t start = pos_;
    if (peek('-')) pos_++;
    if (peek('0')) {
      pos_++;
    } else {
      if (pos_ >= text_.size() || !std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        throw std::runtime_error("invalid JSON number");
      }
      while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) pos_++;
    }
    if (peek('.')) {
      pos_++;
      if (pos_ >= text_.size() || !std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        throw std::runtime_error("invalid JSON number fraction");
      }
      while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) pos_++;
    }
    if (peek('e') || peek('E')) {
      pos_++;
      if (peek('+') || peek('-')) pos_++;
      if (pos_ >= text_.size() || !std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        throw std::runtime_error("invalid JSON number exponent");
      }
      while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) pos_++;
    }
    JsonValue out;
    out.kind = JsonValue::Kind::Number;
    out.number_v = std::stod(text_.substr(start, pos_ - start));
    return out;
  }

  JsonValue parse_true() {
    expect_word("true");
    JsonValue out;
    out.kind = JsonValue::Kind::Bool;
    out.bool_v = true;
    return out;
  }

  JsonValue parse_false() {
    expect_word("false");
    JsonValue out;
    out.kind = JsonValue::Kind::Bool;
    out.bool_v = false;
    return out;
  }

  JsonValue parse_null() {
    expect_word("null");
    return JsonValue{};
  }

  void expect(char c) {
    if (pos_ >= text_.size() || text_[pos_] != c) throw std::runtime_error("unexpected JSON character");
    pos_++;
  }

  void expect_word(const char* word) {
    while (*word) {
      expect(*word);
      word++;
    }
  }

  bool peek(char c) const { return pos_ < text_.size() && text_[pos_] == c; }

  void skip_ws() {
    while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) pos_++;
  }

  std::string text_;
  std::size_t pos_ = 0;
};

const JsonValue& require_object_field(const JsonValue& obj, const char* key) {
  if (obj.kind != JsonValue::Kind::Object) throw std::runtime_error("expected object");
  auto it = obj.object_v.find(key);
  if (it == obj.object_v.end()) throw std::runtime_error(std::string("missing field: ") + key);
  return it->second;
}

int require_int(const JsonValue& v, const char* field_name) {
  if (v.kind != JsonValue::Kind::Number) {
    throw std::runtime_error(std::string("expected number field: ") + field_name);
  }
  const long long i = static_cast<long long>(v.number_v);
  if (static_cast<double>(i) != v.number_v) {
    throw std::runtime_error(std::string("expected integer number field: ") + field_name);
  }
  return static_cast<int>(i);
}

std::string require_string(const JsonValue& v, const char* field_name) {
  if (v.kind != JsonValue::Kind::String) {
    throw std::runtime_error(std::string("expected string field: ") + field_name);
  }
  return v.string_v;
}

Value decode_typed_value(const JsonValue& v) {
  const std::string t = require_string(require_object_field(v, "type"), "type");
  if (t == "none") return Value::none();
  if (t == "bool") {
    const JsonValue& raw = require_object_field(v, "value");
    if (raw.kind != JsonValue::Kind::Bool) throw std::runtime_error("bool typed value requires bool payload");
    return Value::from_bool(raw.bool_v);
  }
  if (t == "int") {
    const JsonValue& raw = require_object_field(v, "value");
    if (raw.kind != JsonValue::Kind::Number) throw std::runtime_error("int typed value requires numeric payload");
    const long long i = static_cast<long long>(raw.number_v);
    if (static_cast<double>(i) != raw.number_v) throw std::runtime_error("int typed value must be integral");
    return Value::from_int(i);
  }
  if (t == "float") {
    const JsonValue& raw = require_object_field(v, "value");
    if (raw.kind != JsonValue::Kind::Number) throw std::runtime_error("float typed value requires numeric payload");
    return Value::from_float(raw.number_v);
  }
  throw std::runtime_error("unknown typed value type");
}

BytecodeProgram decode_program(const JsonValue& bc) {
  BytecodeProgram program;
  program.n_locals = require_int(require_object_field(bc, "n_locals"), "n_locals");

  const JsonValue& consts = require_object_field(bc, "consts");
  if (consts.kind != JsonValue::Kind::Array) throw std::runtime_error("bytecode.consts must be array");
  for (const JsonValue& c : consts.array_v) {
    program.consts.push_back(decode_typed_value(c));
  }

  const JsonValue& code = require_object_field(bc, "code");
  if (code.kind != JsonValue::Kind::Array) throw std::runtime_error("bytecode.code must be array");
  for (const JsonValue& ci : code.array_v) {
    Instr ins;
    ins.op = require_string(require_object_field(ci, "op"), "op");

    auto a_it = ci.object_v.find("a");
    if (a_it != ci.object_v.end() && a_it->second.kind != JsonValue::Kind::Null) {
      ins.a = require_int(a_it->second, "a");
      ins.has_a = true;
    }
    auto b_it = ci.object_v.find("b");
    if (b_it != ci.object_v.end() && b_it->second.kind != JsonValue::Kind::Null) {
      ins.b = require_int(b_it->second, "b");
      ins.has_b = true;
    }
    program.code.push_back(ins);
  }
  return program;
}

std::vector<InputCase> decode_cases(const JsonValue& v) {
  if (v.kind != JsonValue::Kind::Array) throw std::runtime_error("cases must be array");
  std::vector<InputCase> out;
  out.reserve(v.array_v.size());
  for (const JsonValue& case_node : v.array_v) {
    if (case_node.kind != JsonValue::Kind::Array) throw std::runtime_error("case must be array");
    InputCase one;
    one.reserve(case_node.array_v.size());
    for (const JsonValue& item : case_node.array_v) {
      int idx = require_int(require_object_field(item, "idx"), "idx");
      Value value = decode_typed_value(require_object_field(item, "value"));
      one.push_back(LocalBinding{idx, value});
    }
    out.push_back(std::move(one));
  }
  return out;
}

std::vector<BytecodeProgram> decode_programs(const JsonValue& v) {
  if (v.kind != JsonValue::Kind::Array) throw std::runtime_error("programs must be array");
  std::vector<BytecodeProgram> out;
  out.reserve(v.array_v.size());
  for (const JsonValue& p : v.array_v) {
    out.push_back(decode_program(p));
  }
  return out;
}

std::vector<std::vector<InputCase>> decode_cases_by_program(const JsonValue& v) {
  if (v.kind != JsonValue::Kind::Array) throw std::runtime_error("cases_by_program must be array");
  std::vector<std::vector<InputCase>> out;
  out.reserve(v.array_v.size());
  for (const JsonValue& cases_node : v.array_v) {
    out.push_back(decode_cases(cases_node));
  }
  return out;
}

bool value_equal(const Value& a, const Value& b) {
  if (a.tag != b.tag) return false;
  if (a.tag == ValueTag::Int) return a.i == b.i;
  if (a.tag == ValueTag::Float) return a.f == b.f;
  if (a.tag == ValueTag::Bool) return a.b == b.b;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);

  try {
    std::string text;
    if (argc > 1) {
      std::ifstream ifs(argv[1]);
      if (!ifs) return 2;
      std::stringstream buf;
      buf << ifs.rdbuf();
      text = buf.str();
    } else {
      std::stringstream buf;
      buf << std::cin.rdbuf();
      text = buf.str();
    }

    if (text.empty()) return 2;

    JsonParser parser(text);
    JsonValue root = parser.parse();
    if (root.kind != JsonValue::Kind::Object) throw std::runtime_error("top-level JSON must be object");

    const JsonValue* req = nullptr;
    bool is_multi = false;
    auto mbr_it = root.object_v.find("multi_batch_request");
    if (mbr_it != root.object_v.end()) {
      req = &mbr_it->second;
      is_multi = true;
    }
    if (req == nullptr) {
      req = &root;
    }

    auto br_it = root.object_v.find("batch_request");
    if (!is_multi && br_it != root.object_v.end()) req = &br_it->second;

    if (req->kind != JsonValue::Kind::Object) {
      if (is_multi) throw std::runtime_error("multi_batch_request must be object");
      throw std::runtime_error("batch_request must be object");
    }

    auto fv_it = req->object_v.find("format_version");
    if (fv_it != req->object_v.end()) {
      const std::string format = require_string(fv_it->second, "format_version");
      if (format != "bytecode-json-v0.1") throw std::runtime_error("unsupported format_version");
    }

    const int fuel = require_int(require_object_field(*req, "fuel"), "fuel");
    int blocksize = 256;
    auto bs_it = req->object_v.find("blocksize");
    if (bs_it != req->object_v.end()) {
      blocksize = require_int(bs_it->second, "blocksize");
    }

    int mismatches = 0;
    auto exp_it = root.object_v.find("expected");

    if (is_multi) {
      std::vector<BytecodeProgram> programs = decode_programs(require_object_field(*req, "programs"));
      std::vector<std::vector<InputCase>> cases_by_program =
          decode_cases_by_program(require_object_field(*req, "cases_by_program"));
      std::vector<std::vector<g3pvm::VMResult>> out =
          g3pvm::run_bytecode_gpu_multi_batch(programs, cases_by_program, fuel, blocksize);

      int total_cases = 0;
      int ok_count = 0;
      int err_count = 0;
      for (const auto& prog_out : out) {
        total_cases += static_cast<int>(prog_out.size());
        for (const auto& r : prog_out) {
          if (r.is_error) err_count++;
          else ok_count++;
        }
      }

      if (exp_it != root.object_v.end()) {
        if (exp_it->second.kind != JsonValue::Kind::Array) throw std::runtime_error("expected must be array");
        if (exp_it->second.array_v.size() != out.size()) throw std::runtime_error("expected size mismatch");
        for (std::size_t pi = 0; pi < out.size(); ++pi) {
          const JsonValue& e_prog = exp_it->second.array_v[pi];
          if (e_prog.kind != JsonValue::Kind::Array) throw std::runtime_error("expected program entry must be array");
          if (e_prog.array_v.size() != out[pi].size()) throw std::runtime_error("expected case size mismatch");
          for (std::size_t ci = 0; ci < out[pi].size(); ++ci) {
            const JsonValue& e = e_prog.array_v[ci];
            const std::string kind = require_string(require_object_field(e, "kind"), "kind");
            if (kind == "return") {
              const Value expected_value = decode_typed_value(require_object_field(e, "value"));
              if (out[pi][ci].is_error || !value_equal(out[pi][ci].value, expected_value)) mismatches++;
              continue;
            }
            if (kind == "error") {
              const std::string code = require_string(require_object_field(e, "code"), "code");
              if (!out[pi][ci].is_error || code != g3pvm::err_code_name(out[pi][ci].err.code)) mismatches++;
              continue;
            }
            throw std::runtime_error("expected.kind must be return/error");
          }
        }
      }

      std::cout << "OK programs=" << out.size() << " cases=" << total_cases << " return=" << ok_count
                << " error=" << err_count << " blocksize=" << blocksize << "\n";
      if (exp_it != root.object_v.end()) {
        std::cout << "CHECK mismatches=" << mismatches << "\n";
        return mismatches == 0 ? 0 : 1;
      }
      return 0;
    } else {
      BytecodeProgram program = decode_program(require_object_field(*req, "bytecode"));
      std::vector<InputCase> cases = decode_cases(require_object_field(*req, "cases"));

      std::vector<g3pvm::VMResult> out = g3pvm::run_bytecode_gpu_batch(program, cases, fuel, blocksize);

      int ok_count = 0;
      int err_count = 0;
      for (const auto& r : out) {
        if (r.is_error) err_count++;
        else ok_count++;
      }

      if (exp_it != root.object_v.end()) {
        if (exp_it->second.kind != JsonValue::Kind::Array) throw std::runtime_error("expected must be array");
        if (exp_it->second.array_v.size() != out.size()) {
          throw std::runtime_error("expected size mismatch");
        }

        for (std::size_t i = 0; i < out.size(); ++i) {
          const JsonValue& e = exp_it->second.array_v[i];
          const std::string kind = require_string(require_object_field(e, "kind"), "kind");
          if (kind == "return") {
            const Value expected_value = decode_typed_value(require_object_field(e, "value"));
            if (out[i].is_error || !value_equal(out[i].value, expected_value)) {
              mismatches++;
            }
            continue;
          }
          if (kind == "error") {
            const std::string code = require_string(require_object_field(e, "code"), "code");
            if (!out[i].is_error || code != g3pvm::err_code_name(out[i].err.code)) {
              mismatches++;
            }
            continue;
          }
          throw std::runtime_error("expected.kind must be return/error");
        }
      }

      std::cout << "OK cases=" << out.size() << " return=" << ok_count << " error=" << err_count
                << " blocksize=" << blocksize << "\n";
      if (exp_it != root.object_v.end()) {
        std::cout << "CHECK mismatches=" << mismatches << "\n";
        return mismatches == 0 ? 0 : 1;
      }
      return 0;
    }
  } catch (const std::exception& e) {
    std::cerr << "parse/run error: " << e.what() << "\n";
    return 2;
  }
}
