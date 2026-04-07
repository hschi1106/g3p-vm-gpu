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
  }
  return "";
}

G3PVM_BUILTIN_HD inline bool builtin_id_from_int(int value, BuiltinId& out) {
  if (value < static_cast<int>(BuiltinId::Abs) || value > static_cast<int>(BuiltinId::IsInt)) {
    return false;
  }
  out = static_cast<BuiltinId>(value);
  return true;
}

}  // namespace g3pvm
