#pragma once

#include <string>
#include <vector>

#include "g3pvm/core/value.hpp"

namespace g3pvm::payload {

struct StringSnapshot {
  Value key = Value::none();
  std::string data;
};

struct ListSnapshot {
  Value key = Value::none();
  std::vector<Value> elems;
};

void clear();

void register_string(const Value& key, const std::string& s);
void register_list(const Value& key, const std::vector<Value>& elems);

bool lookup_string(const Value& key, std::string* out);
bool lookup_list(const Value& key, std::vector<Value>* out);

Value make_string_value(const std::string& s);
Value make_list_value(const std::vector<Value>& elems);

std::vector<StringSnapshot> snapshot_strings();
std::vector<ListSnapshot> snapshot_lists();

}  // namespace g3pvm::payload
