#include <cctype>
#include <cstdlib>
#include <iomanip>
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
#include "g3pvm/vm_cpu.hpp"

namespace {

using g3pvm::BytecodeProgram;
using g3pvm::Instr;
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
    if (pos_ >= text_.size()) {
      throw std::runtime_error("unexpected end of JSON");
    }
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
      if (c == '"') {
        return out;
      }
      if (c == '\\') {
        if (pos_ >= text_.size()) {
          throw std::runtime_error("invalid JSON escape");
        }
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
    if (pos_ >= text_.size() || text_[pos_] != c) {
      throw std::runtime_error("unexpected JSON character");
    }
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
  if (obj.kind != JsonValue::Kind::Object) {
    throw std::runtime_error("expected object");
  }
  auto it = obj.object_v.find(key);
  if (it == obj.object_v.end()) {
    throw std::runtime_error(std::string("missing field: ") + key);
  }
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
  const JsonValue& t_node = require_object_field(v, "type");
  const std::string t = require_string(t_node, "type");
  if (t == "none") {
    return Value::none();
  }
  if (t == "bool") {
    const JsonValue& raw = require_object_field(v, "value");
    if (raw.kind != JsonValue::Kind::Bool) {
      throw std::runtime_error("bool typed value requires bool payload");
    }
    return Value::from_bool(raw.bool_v);
  }
  if (t == "int") {
    const JsonValue& raw = require_object_field(v, "value");
    if (raw.kind != JsonValue::Kind::Number) {
      throw std::runtime_error("int typed value requires numeric payload");
    }
    const long long i = static_cast<long long>(raw.number_v);
    if (static_cast<double>(i) != raw.number_v) {
      throw std::runtime_error("int typed value must be integral");
    }
    return Value::from_int(i);
  }
  if (t == "float") {
    const JsonValue& raw = require_object_field(v, "value");
    if (raw.kind != JsonValue::Kind::Number) {
      throw std::runtime_error("float typed value requires numeric payload");
    }
    return Value::from_float(raw.number_v);
  }
  throw std::runtime_error("unknown typed value type");
}

void print_value(const Value& v) {
  if (v.tag == ValueTag::Int) {
    std::cout << "int " << v.i << "\n";
    return;
  }
  if (v.tag == ValueTag::Float) {
    std::cout << "float " << std::setprecision(17) << v.f << "\n";
    return;
  }
  if (v.tag == ValueTag::Bool) {
    std::cout << "bool " << (v.b ? 1 : 0) << "\n";
    return;
  }
  std::cout << "none\n";
}

}  // namespace

int main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);

  try {
    std::stringstream buf;
    buf << std::cin.rdbuf();
    const std::string text = buf.str();
    if (text.empty()) {
      return 2;
    }

    JsonParser parser(text);
    JsonValue root = parser.parse();

    const JsonValue* req = &root;
    if (root.kind == JsonValue::Kind::Object) {
      auto it = root.object_v.find("request");
      if (it != root.object_v.end()) {
        req = &it->second;
      }
    }

    if (req->kind != JsonValue::Kind::Object) {
      throw std::runtime_error("top-level JSON must be object");
    }

    auto fv_it = req->object_v.find("format_version");
    if (fv_it != req->object_v.end()) {
      const std::string format = require_string(fv_it->second, "format_version");
      if (format != "bytecode-json-v0.1") {
        throw std::runtime_error("unsupported format_version");
      }
    }

    const int fuel = require_int(require_object_field(*req, "fuel"), "fuel");
    const JsonValue& bc = require_object_field(*req, "bytecode");

    BytecodeProgram program;
    program.n_locals = require_int(require_object_field(bc, "n_locals"), "n_locals");

    const JsonValue& consts = require_object_field(bc, "consts");
    if (consts.kind != JsonValue::Kind::Array) {
      throw std::runtime_error("bytecode.consts must be array");
    }
    for (const JsonValue& c : consts.array_v) {
      program.consts.push_back(decode_typed_value(c));
    }

    const JsonValue& code = require_object_field(bc, "code");
    if (code.kind != JsonValue::Kind::Array) {
      throw std::runtime_error("bytecode.code must be array");
    }
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

    std::vector<std::pair<int, Value>> inputs;
    auto in_it = req->object_v.find("inputs");
    if (in_it != req->object_v.end()) {
      if (in_it->second.kind != JsonValue::Kind::Array) {
        throw std::runtime_error("inputs must be array");
      }
      for (const JsonValue& item : in_it->second.array_v) {
        const int idx = require_int(require_object_field(item, "idx"), "idx");
        const Value v = decode_typed_value(require_object_field(item, "value"));
        inputs.push_back({idx, v});
      }
    }

    g3pvm::VMResult result = g3pvm::run_bytecode(program, inputs, fuel);
    if (result.is_error) {
      std::cout << "ERR " << g3pvm::err_code_name(result.err.code) << "\n";
      return 0;
    }

    std::cout << "OK ";
    print_value(result.value);
    return 0;
  } catch (const std::exception&) {
    return 2;
  }
}
