#pragma once

#include <string>

namespace gagp {

enum class ErrCode : int {
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

}  // namespace gagp
