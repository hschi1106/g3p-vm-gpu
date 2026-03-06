#pragma once

#include <map>
#include <string>
#include <vector>

namespace g3pvm::cli_detail {

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
  explicit JsonParser(std::string text);

  JsonValue parse();

 private:
  JsonValue parse_value();
  JsonValue parse_object();
  JsonValue parse_array();
  JsonValue parse_string();
  JsonValue parse_number();
  JsonValue parse_true();
  JsonValue parse_false();
  JsonValue parse_null();

  void expect(char c);
  void expect_word(const char* word);
  bool peek(char c) const;
  void skip_ws();

  std::string text_;
  std::size_t pos_ = 0;
};

const JsonValue& require_object_field(const JsonValue& obj, const char* key);
int require_int(const JsonValue& v, const char* field_name);
std::string require_string(const JsonValue& v, const char* field_name);

}  // namespace g3pvm::cli_detail
