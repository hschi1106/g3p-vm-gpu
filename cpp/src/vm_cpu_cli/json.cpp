#include "json.hpp"

#include <cctype>
#include <stdexcept>
#include <utility>

namespace g3pvm::cli_detail {

JsonParser::JsonParser(std::string text) : text_(std::move(text)) {}

JsonValue JsonParser::parse() {
  skip_ws();
  JsonValue v = parse_value();
  skip_ws();
  if (pos_ != text_.size()) {
    throw std::runtime_error("trailing characters in JSON");
  }
  return v;
}

JsonValue JsonParser::parse_value() {
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

JsonValue JsonParser::parse_object() {
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

JsonValue JsonParser::parse_array() {
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

JsonValue JsonParser::parse_string() {
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

JsonValue JsonParser::parse_number() {
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

JsonValue JsonParser::parse_true() {
  expect_word("true");
  JsonValue out;
  out.kind = JsonValue::Kind::Bool;
  out.bool_v = true;
  return out;
}

JsonValue JsonParser::parse_false() {
  expect_word("false");
  JsonValue out;
  out.kind = JsonValue::Kind::Bool;
  out.bool_v = false;
  return out;
}

JsonValue JsonParser::parse_null() {
  expect_word("null");
  return JsonValue{};
}

void JsonParser::expect(char c) {
  if (pos_ >= text_.size() || text_[pos_] != c) {
    throw std::runtime_error("unexpected JSON character");
  }
  pos_++;
}

void JsonParser::expect_word(const char* word) {
  while (*word) {
    expect(*word);
    word++;
  }
}

bool JsonParser::peek(char c) const { return pos_ < text_.size() && text_[pos_] == c; }

void JsonParser::skip_ws() {
  while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) pos_++;
}

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

}  // namespace g3pvm::cli_detail
