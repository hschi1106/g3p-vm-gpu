#pragma once

#include <string>

namespace g3pvm {

enum class ErrCode {
  Name,
  Type,
  ZeroDiv,
  Value,
  Timeout,
};

inline const char* err_code_name(ErrCode code) {
  switch (code) {
    case ErrCode::Name:
      return "NameError";
    case ErrCode::Type:
      return "TypeError";
    case ErrCode::ZeroDiv:
      return "ZeroDivisionError";
    case ErrCode::Value:
      return "ValueError";
    case ErrCode::Timeout:
      return "Timeout";
  }
  return "ValueError";
}

struct Err {
  ErrCode code;
  std::string message;
};

}  // namespace g3pvm
