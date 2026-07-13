#pragma once

#include <cstdint>

namespace g3pvm {

#if defined(__CUDACC__)
#define G3PVM_BUILTIN_HD __host__ __device__
#else
#define G3PVM_BUILTIN_HD
#endif

enum class BuiltinId : std::int32_t {
  Abs = 0,
  Min = 1,
  Max = 2,
  Clip = 3,
  Len = 4,
  Concat = 5,
  Slice = 6,
  Index = 7,
  Append = 8,
  Reverse = 9,
  Find = 10,
  Contains = 11,
  IsInt = 12,
  IDiv0 = 13,
  IMod0 = 14,
  Prepend = 15,
  CharToString = 16,
  StringToChar = 17,
  Ord = 18,
  Chr = 19,
  IsLetter = 20,
  IsDigit = 21,
  IsSpace = 22,
  IsVowel = 23,
  ToLower = 24,
  ToUpper = 25,
  ToString = 26,
  Singleton = 27,
};

G3PVM_BUILTIN_HD inline const char* builtin_name(BuiltinId id) {
  switch (id) {
    case BuiltinId::Abs:
      return "abs";
    case BuiltinId::Min:
      return "min";
    case BuiltinId::Max:
      return "max";
    case BuiltinId::Clip:
      return "clip";
    case BuiltinId::Len:
      return "len";
    case BuiltinId::Concat:
      return "concat";
    case BuiltinId::Slice:
      return "slice";
    case BuiltinId::Index:
      return "index";
    case BuiltinId::Append:
      return "append";
    case BuiltinId::Reverse:
      return "reverse";
    case BuiltinId::Find:
      return "find";
    case BuiltinId::Contains:
      return "contains";
    case BuiltinId::IsInt:
      return "is_int";
    case BuiltinId::IDiv0:
      return "idiv0";
    case BuiltinId::IMod0:
      return "imod0";
    case BuiltinId::Prepend:
      return "prepend";
    case BuiltinId::CharToString:
      return "char_to_string";
    case BuiltinId::StringToChar:
      return "string_to_char";
    case BuiltinId::Ord:
      return "ord";
    case BuiltinId::Chr:
      return "chr";
    case BuiltinId::IsLetter:
      return "is_letter";
    case BuiltinId::IsDigit:
      return "is_digit";
    case BuiltinId::IsSpace:
      return "is_space";
    case BuiltinId::IsVowel:
      return "is_vowel";
    case BuiltinId::ToLower:
      return "to_lower";
    case BuiltinId::ToUpper:
      return "to_upper";
    case BuiltinId::ToString:
      return "to_string";
    case BuiltinId::Singleton:
      return "singleton";
  }
  return "";
}

G3PVM_BUILTIN_HD inline bool builtin_id_from_int(int value, BuiltinId& out) {
  if (value < static_cast<int>(BuiltinId::Abs) || value > static_cast<int>(BuiltinId::Singleton)) {
    return false;
  }
  out = static_cast<BuiltinId>(value);
  return true;
}

}  // namespace g3pvm
