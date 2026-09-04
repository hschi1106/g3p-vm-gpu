#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gagp/core/value.hpp"

namespace gagp::payload {

struct PayloadStats {
  std::size_t string_entries = 0;
  std::size_t list_entries = 0;
  std::size_t string_bytes = 0;
  std::size_t list_value_count = 0;
};

struct StringSnapshot {
  Value key = Value::invalid();
  std::string data;
};

struct ListSnapshot {
  Value key = Value::invalid();
  std::vector<Value> elems;
};

void clear();
void retain_only(const std::vector<Value>& roots);
PayloadStats stats();

void register_string(const Value& key, const std::string& s);
void register_list(const Value& key, const std::vector<Value>& elems);

bool lookup_string(const Value& key, std::string* out);
bool lookup_list(const Value& key, std::vector<Value>* out);
bool lookup_string_packed(std::int64_t packed, std::string* out);
bool lookup_list_packed(ValueTag tag, std::int64_t packed, std::vector<Value>* out);

Value make_string_value(const std::string& s);
Value make_int_list_value(const std::vector<Value>& elems);
Value make_float_list_value(const std::vector<Value>& elems);
Value make_string_list_value(const std::vector<Value>& elems);

std::vector<StringSnapshot> snapshot_strings();
std::vector<ListSnapshot> snapshot_lists();

}  // namespace gagp::payload
