#include "gagp/runtime/cpu/builtins_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

#include "gagp/core/value_semantics.hpp"
#include "gagp/runtime/payload/payload.hpp"

namespace gagp {

namespace {

bool to_numeric_pair(const Value& a, const Value& b, double& a_out, double& b_out,
                     bool& any_float) {
  if (!is_numeric(a) || !is_numeric(b)) {
    return false;
  }
  if (a.tag != b.tag) {
    return false;
  }
  any_float = (a.tag == ValueTag::Float) || (b.tag == ValueTag::Float);
  a_out = (a.tag == ValueTag::Float) ? a.f : static_cast<double>(a.i);
  b_out = (b.tag == ValueTag::Float) ? b.f : static_cast<double>(b.i);
  return true;
}

BuiltinResult fail(ErrCode code, const std::string& message) {
  BuiltinResult out;
  out.is_error = true;
  out.err = Err{code, message};
  return out;
}

long long normalize_slice_idx(long long idx, long long n) {
  long long out = idx;
  if (out < 0) out += n;
  if (out < 0) out = 0;
  if (out > n) out = n;
  return out;
}

bool normalize_index_idx(long long idx, long long n, long long& out) {
  long long j = idx;
  if (j < 0) j += n;
  if (j < 0 || j >= n) {
    return false;
  }
  out = j;
  return true;
}

bool is_list_tag(ValueTag tag) {
  return tag == ValueTag::IntList || tag == ValueTag::FloatList || tag == ValueTag::StringList;
}

std::string format_float_current(double value) {
  if (std::isnan(value)) return "nan";
  if (std::isinf(value)) return value < 0.0 ? "-inf" : "inf";
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6) << value;
  std::string out = oss.str();
  while (!out.empty() && out.back() == '0') {
    out.pop_back();
  }
  if (!out.empty() && out.back() == '.') {
    out.pop_back();
  }
  if (out == "-0") return "0";
  return out;
}

std::uint8_t list_type_code(const Value& v) {
  if (v.tag == ValueTag::IntList) return 2U;
  if (v.tag == ValueTag::FloatList) return 3U;
  return 4U;
}

Value make_list_hash_len(const Value& src, std::uint64_t h, std::uint32_t len) {
  if (src.tag == ValueTag::IntList) return Value::from_int_list_hash_len(h, len);
  if (src.tag == ValueTag::FloatList) return Value::from_float_list_hash_len(h, len);
  return Value::from_string_list_hash_len(h, len);
}

std::uint64_t prepend_list_hash48(std::uint8_t type_code, const Value& elem, const Value& src) {
  std::uint64_t h = Value::fnv1a_init();
  h = Value::fnv1a_mix_u8(h, type_code);
  h = Value::fnv1a_mix_u8(h, 0x70U);
  h = Value::fnv1a_mix_u64(h, Value::shallow_hash64(elem));
  h = Value::fnv1a_mix_u64(h, Value::container_hash48(src));
  h = Value::fnv1a_mix_u64(h, static_cast<std::uint64_t>(Value::container_len(src)));
  return (h & Value::k_container_hash_mask);
}

BuiltinResult make_exact_list(ValueTag tag, const std::vector<Value>& elems) {
  if (tag == ValueTag::IntList) {
    return BuiltinResult{false, payload::make_int_list_value(elems), Err{ErrCode::Value, ""}};
  }
  if (tag == ValueTag::FloatList) {
    return BuiltinResult{false, payload::make_float_list_value(elems), Err{ErrCode::Value, ""}};
  }
  return BuiltinResult{false, payload::make_string_list_value(elems), Err{ErrCode::Value, ""}};
}

bool is_ascii_letter(long long c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

bool is_ascii_digit(long long c) {
  return c >= '0' && c <= '9';
}

bool is_ascii_space(long long c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

bool is_ascii_vowel(long long c) {
  if (c >= 'A' && c <= 'Z') c += ('a' - 'A');
  return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
}

long long ascii_to_lower(long long c) {
  return (c >= 'A' && c <= 'Z') ? c + ('a' - 'A') : c;
}

long long ascii_to_upper(long long c) {
  return (c >= 'a' && c <= 'z') ? c - ('a' - 'A') : c;
}

}  // namespace

BuiltinResult builtin_call(BuiltinId id, const std::vector<Value>& args) {
  if (id == BuiltinId::Abs) {
    if (args.size() != 1) {
      return fail(ErrCode::Type, "abs expects 1 argument");
    }
    const Value& x = args[0];
    if (!is_numeric(x)) {
      return fail(ErrCode::Type, "abs expects a numeric argument");
    }
    BuiltinResult out;
    out.value = (x.tag == ValueTag::Float)
                    ? Value::from_float(vm_semantics::canonicalize_vm_float(x.f < 0 ? -x.f : x.f))
                                            : Value::from_int(x.i < 0 ? -x.i : x.i);
    return out;
  }

  if (id == BuiltinId::Min || id == BuiltinId::Max) {
    if (args.size() != 2) {
      return fail(ErrCode::Type, std::string(builtin_name(id)) + " expects 2 arguments");
    }
    double a = 0.0;
    double b = 0.0;
    bool any_float = false;
    if (!to_numeric_pair(args[0], args[1], a, b, any_float)) {
      return fail(ErrCode::Type, std::string(builtin_name(id)) + " expects numeric arguments");
    }
    const double pick = (id == BuiltinId::Min) ? (a <= b ? a : b) : (a >= b ? a : b);
    BuiltinResult out;
    out.value = any_float ? Value::from_float(vm_semantics::canonicalize_vm_float(pick))
                          : Value::from_int(static_cast<long long>(pick));
    return out;
  }

  if (id == BuiltinId::Clip) {
    if (args.size() != 3) {
      return fail(ErrCode::Type, "clip expects 3 arguments: clip(x, lo, hi)");
    }
    const Value& x = args[0];
    const Value& lo = args[1];
    const Value& hi = args[2];
    if (!is_numeric(x) || !is_numeric(lo) || !is_numeric(hi)) {
      return fail(ErrCode::Type, "clip expects numeric arguments");
    }
    if (x.tag != lo.tag || x.tag != hi.tag) {
      return fail(ErrCode::Type, "clip expects matching numeric argument types");
    }
    const bool any_float =
        (x.tag == ValueTag::Float) || (lo.tag == ValueTag::Float) || (hi.tag == ValueTag::Float);
    if (any_float) {
      const double x2 = (x.tag == ValueTag::Float) ? x.f : static_cast<double>(x.i);
      const double lo2 = (lo.tag == ValueTag::Float) ? lo.f : static_cast<double>(lo.i);
      const double hi2 = (hi.tag == ValueTag::Float) ? hi.f : static_cast<double>(hi.i);
      BuiltinResult out;
      const double picked = std::min(std::max(x2, lo2), hi2);
      out.value = Value::from_float(vm_semantics::canonicalize_vm_float(picked));
      return out;
    }

    const long long x2 = x.i;
    const long long lo2 = lo.i;
    const long long hi2 = hi.i;
    BuiltinResult out;
    out.value = Value::from_int(std::min(std::max(x2, lo2), hi2));
    return out;
  }

  if (id == BuiltinId::IDiv0 || id == BuiltinId::IMod0) {
    if (args.size() != 2) {
      return fail(ErrCode::Type, std::string(builtin_name(id)) + " expects 2 arguments");
    }
    const Value& a = args[0];
    const Value& b = args[1];
    if (a.tag != ValueTag::Int || b.tag != ValueTag::Int) {
      return fail(ErrCode::Type, std::string(builtin_name(id)) + " expects integer arguments");
    }
    BuiltinResult out;
    if (b.i == 0) {
      out.value = Value::from_int(0);
    } else if (id == BuiltinId::IDiv0) {
      out.value = Value::from_int(a.i / b.i);
    } else {
      out.value = Value::from_int(vm_semantics::py_int_mod(a.i, b.i));
    }
    return out;
  }

  if (id == BuiltinId::Len) {
    if (args.size() != 1) {
      return fail(ErrCode::Type, "len expects 1 argument");
    }
    const Value& x = args[0];
    if (!is_container(x)) {
      return fail(ErrCode::Type, "len expects string/typed-list argument");
    }
    BuiltinResult out;
    out.value = Value::from_int(static_cast<long long>(Value::container_len(x)));
    return out;
  }

  if (id == BuiltinId::Concat) {
    if (args.size() != 2) {
      return fail(ErrCode::Type, "concat expects 2 arguments");
    }
    const Value& a = args[0];
    const Value& b = args[1];
    if (a.tag == ValueTag::String && b.tag == ValueTag::String) {
      std::string sa;
      std::string sb;
      if (payload::lookup_string(a, &sa) && payload::lookup_string(b, &sb)) {
        return BuiltinResult{false, payload::make_string_value(sa + sb), Err{ErrCode::Value, ""}};
      }
      const std::uint32_t len = Value::saturating_len_add(Value::container_len(a), Value::container_len(b));
      const std::uint64_t h = Value::combine_container_hash48(1U, a, b);
      BuiltinResult out;
      out.value = Value::from_fallback_token(Value::pack_container_payload(h, len));
      return out;
    }
    if (is_list_tag(a.tag) && a.tag == b.tag) {
      std::vector<Value> la;
      std::vector<Value> lb;
      if (payload::lookup_list(a, &la) && payload::lookup_list(b, &lb)) {
        std::vector<Value> out_elems;
        out_elems.reserve(la.size() + lb.size());
        out_elems.insert(out_elems.end(), la.begin(), la.end());
        out_elems.insert(out_elems.end(), lb.begin(), lb.end());
        return make_exact_list(a.tag, out_elems);
      }
      const std::uint32_t len = Value::saturating_len_add(Value::container_len(a), Value::container_len(b));
      const std::uint64_t h = Value::combine_container_hash48(list_type_code(a), a, b);
      BuiltinResult out;
      out.value = make_list_hash_len(a, h, len);
      return out;
    }
    return fail(ErrCode::Type, "concat expects matching string/typed-list arguments");
  }

  if (id == BuiltinId::Slice) {
    if (args.size() != 3) {
      return fail(ErrCode::Type, "slice expects 3 arguments: slice(x, lo, hi)");
    }
    const Value& x = args[0];
    const Value& lo = args[1];
    const Value& hi = args[2];
    if (!is_container(x)) {
      return fail(ErrCode::Type, "slice expects string/typed-list as first argument");
    }
    if (lo.tag != ValueTag::Int || hi.tag != ValueTag::Int) {
      return fail(ErrCode::Type, "slice expects integer lo/hi");
    }
    const long long n = static_cast<long long>(Value::container_len(x));
    const long long l = normalize_slice_idx(lo.i, n);
    const long long h = normalize_slice_idx(hi.i, n);
    const long long out_len_ll = (h > l) ? (h - l) : 0LL;
    const std::uint32_t out_len =
        static_cast<std::uint32_t>(out_len_ll > static_cast<long long>(Value::k_container_len_max)
                                       ? Value::k_container_len_max
                                       : out_len_ll);
    if (x.tag == ValueTag::String) {
      std::string sx;
      if (payload::lookup_string(x, &sx)) {
        const std::size_t ls = static_cast<std::size_t>(l);
        const std::size_t hs = static_cast<std::size_t>(h);
        const std::size_t count = (hs > ls) ? (hs - ls) : 0U;
        return BuiltinResult{false, payload::make_string_value(sx.substr(ls, count)), Err{ErrCode::Value, ""}};
      }
      const std::uint64_t out_h = Value::slice_container_hash48(1U, x, lo.i, hi.i);
      BuiltinResult out;
      out.value = Value::from_fallback_token(Value::pack_container_payload(out_h, out_len));
      return out;
    }
    if (is_list_tag(x.tag)) {
      std::vector<Value> lx;
      if (payload::lookup_list(x, &lx)) {
        const std::size_t ls = static_cast<std::size_t>(l);
        const std::size_t hs = static_cast<std::size_t>(h);
        std::vector<Value> out_elems;
        out_elems.reserve(hs > ls ? (hs - ls) : 0U);
        for (std::size_t k = ls; k < hs; ++k) {
          out_elems.push_back(lx[k]);
        }
        return make_exact_list(x.tag, out_elems);
      }
      const std::uint64_t out_h = Value::slice_container_hash48(list_type_code(x), x, lo.i, hi.i);
      BuiltinResult out;
      out.value = make_list_hash_len(x, out_h, out_len);
      return out;
    }
    return fail(ErrCode::Type, "slice expects string/typed-list as first argument");
  }

  if (id == BuiltinId::Index) {
    if (args.size() != 2) {
      return fail(ErrCode::Type, "index expects 2 arguments: index(x, i)");
    }
    const Value& x = args[0];
    const Value& i = args[1];
    if (!is_container(x)) {
      return fail(ErrCode::Type, "index expects string/typed-list as first argument");
    }
    if (i.tag != ValueTag::Int) {
      return fail(ErrCode::Type, "index expects integer index");
    }
    const long long n = static_cast<long long>(Value::container_len(x));
    long long j = 0;
    if (!normalize_index_idx(i.i, n, j)) {
      return fail(ErrCode::Value, "index out of range");
    }
    BuiltinResult out;
    if (x.tag == ValueTag::String) {
      std::string sx;
      if (payload::lookup_string(x, &sx)) {
        const char ch = sx[static_cast<std::size_t>(j)];
        out.value = Value::from_char(static_cast<unsigned char>(ch));
        return out;
      }
      out.value = Value::from_fallback_token(Value::index_container_token64(5U, x, j));
      return out;
    }
    if (is_list_tag(x.tag)) {
      std::vector<Value> lx;
      if (payload::lookup_list(x, &lx)) {
        out.value = lx[static_cast<std::size_t>(j)];
        return out;
      }
      out.value = Value::from_fallback_token(Value::index_container_token64(list_type_code(x), x, j));
      return out;
    }
    return fail(ErrCode::Type, "index expects string/typed-list as first argument");
  }

  if (id == BuiltinId::Append) {
    if (args.size() != 2) {
      return fail(ErrCode::Type, "append expects 2 arguments");
    }
    const Value& xs = args[0];
    const Value& elem = args[1];
    if (!is_list_tag(xs.tag)) {
      return fail(ErrCode::Type, "append expects typed-list as first argument");
    }
    if (xs.tag == ValueTag::IntList && elem.tag != ValueTag::Int) {
      return fail(ErrCode::Type, "append expects int element for IntList");
    }
    if (xs.tag == ValueTag::FloatList && elem.tag != ValueTag::Float) {
      return fail(ErrCode::Type, "append expects float element for FloatList");
    }
    if (xs.tag == ValueTag::StringList && elem.tag != ValueTag::String) {
      return fail(ErrCode::Type, "append expects string element for StringList");
    }
    std::vector<Value> lx;
    if (payload::lookup_list(xs, &lx)) {
      lx.push_back(elem);
      return make_exact_list(xs.tag, lx);
    }
    const std::uint32_t len = Value::saturating_len_add(Value::container_len(xs), 1U);
    const std::uint64_t h = Value::append_list_hash48(list_type_code(xs), xs, elem);
    BuiltinResult out;
    out.value = make_list_hash_len(xs, h, len);
    return out;
  }

  if (id == BuiltinId::Prepend) {
    if (args.size() != 2) {
      return fail(ErrCode::Type, "prepend expects 2 arguments");
    }
    const Value& xs = args[0];
    const Value& elem = args[1];
    if (!is_list_tag(xs.tag)) {
      return fail(ErrCode::Type, "prepend expects typed-list as first argument");
    }
    if (xs.tag == ValueTag::IntList && elem.tag != ValueTag::Int) {
      return fail(ErrCode::Type, "prepend expects int element for IntList");
    }
    if (xs.tag == ValueTag::FloatList && elem.tag != ValueTag::Float) {
      return fail(ErrCode::Type, "prepend expects float element for FloatList");
    }
    if (xs.tag == ValueTag::StringList && elem.tag != ValueTag::String) {
      return fail(ErrCode::Type, "prepend expects string element for StringList");
    }
    std::vector<Value> lx;
    if (payload::lookup_list(xs, &lx)) {
      lx.insert(lx.begin(), elem);
      return make_exact_list(xs.tag, lx);
    }
    const std::uint32_t len = Value::saturating_len_add(Value::container_len(xs), 1U);
    const std::uint64_t h = prepend_list_hash48(list_type_code(xs), elem, xs);
    BuiltinResult out;
    out.value = make_list_hash_len(xs, h, len);
    return out;
  }

  if (id == BuiltinId::Reverse) {
    if (args.size() != 1) {
      return fail(ErrCode::Type, "reverse expects 1 argument");
    }
    const Value& x = args[0];
    if (x.tag == ValueTag::String) {
      std::string sx;
      if (payload::lookup_string(x, &sx)) {
        std::reverse(sx.begin(), sx.end());
        return BuiltinResult{false, payload::make_string_value(sx), Err{ErrCode::Value, ""}};
      }
      const std::uint64_t h = Value::reverse_container_hash48(7U, x);
      BuiltinResult out;
      out.value = Value::from_fallback_token(Value::pack_container_payload(h, Value::container_len(x)));
      return out;
    }
    if (!is_list_tag(x.tag)) {
      return fail(ErrCode::Type, "reverse expects string/typed-list argument");
    }
    std::vector<Value> lx;
    if (payload::lookup_list(x, &lx)) {
      std::reverse(lx.begin(), lx.end());
      return make_exact_list(x.tag, lx);
    }
    const std::uint64_t h = Value::reverse_container_hash48(list_type_code(x), x);
    BuiltinResult out;
    out.value = make_list_hash_len(x, h, Value::container_len(x));
    return out;
  }

  if (id == BuiltinId::Find || id == BuiltinId::Contains) {
    if (args.size() != 2) {
      return fail(ErrCode::Type, std::string(builtin_name(id)) + " expects 2 arguments");
    }
    const Value& haystack = args[0];
    const Value& needle = args[1];
    if (haystack.tag != ValueTag::String || needle.tag != ValueTag::String) {
      return fail(ErrCode::Type, std::string(builtin_name(id)) + " expects (string,string)");
    }
    std::string hs;
    std::string nd;
    if (!payload::lookup_string(haystack, &hs) || !payload::lookup_string(needle, &nd)) {
      return fail(ErrCode::Value, std::string(builtin_name(id)) + " requires exact string payload");
    }
    const std::size_t pos = hs.find(nd);
    BuiltinResult out;
    if (id == BuiltinId::Find) {
      out.value = Value::from_int(pos == std::string::npos ? -1LL : static_cast<long long>(pos));
    } else {
      out.value = Value::from_bool(pos != std::string::npos);
    }
    return out;
  }

  if (id == BuiltinId::IsInt) {
    if (args.size() != 1) {
      return fail(ErrCode::Type, "is_int expects 1 argument");
    }
    BuiltinResult out;
    out.value = Value::from_bool(args[0].tag == ValueTag::Int);
    return out;
  }

  if (id == BuiltinId::CharToString) {
    if (args.size() != 1 || args[0].tag != ValueTag::Char) {
      return fail(ErrCode::Type, "char_to_string expects char");
    }
    const long long c = args[0].i;
    if (c < 0 || c > 255) {
      return fail(ErrCode::Value, "char_to_string supports byte-sized char in CPU slice");
    }
    return BuiltinResult{false, payload::make_string_value(std::string(1, static_cast<char>(c))), Err{ErrCode::Value, ""}};
  }

  if (id == BuiltinId::StringToChar) {
    if (args.size() != 1 || args[0].tag != ValueTag::String) {
      return fail(ErrCode::Type, "string_to_char expects string");
    }
    std::string s;
    if (!payload::lookup_string(args[0], &s)) {
      return fail(ErrCode::Value, "string_to_char requires exact string payload");
    }
    if (s.size() != 1U) {
      return fail(ErrCode::Value, "string_to_char expects length-1 string");
    }
    return BuiltinResult{false, Value::from_char(static_cast<unsigned char>(s[0])), Err{ErrCode::Value, ""}};
  }

  if (id == BuiltinId::Ord) {
    if (args.size() != 1 || args[0].tag != ValueTag::Char) {
      return fail(ErrCode::Type, "ord expects char");
    }
    return BuiltinResult{false, Value::from_int(args[0].i), Err{ErrCode::Value, ""}};
  }

  if (id == BuiltinId::Chr) {
    if (args.size() != 1 || args[0].tag != ValueTag::Int) {
      return fail(ErrCode::Type, "chr expects int");
    }
    if (args[0].i < 0 || args[0].i > 255) {
      return fail(ErrCode::Value, "invalid character code point");
    }
    return BuiltinResult{false, Value::from_char(args[0].i), Err{ErrCode::Value, ""}};
  }

  if (id == BuiltinId::IsLetter || id == BuiltinId::IsDigit || id == BuiltinId::IsSpace ||
      id == BuiltinId::IsVowel || id == BuiltinId::ToLower || id == BuiltinId::ToUpper) {
    if (args.size() != 1 || args[0].tag != ValueTag::Char) {
      return fail(ErrCode::Type, std::string(builtin_name(id)) + " expects char");
    }
    const long long c = args[0].i;
    BuiltinResult out;
    if (id == BuiltinId::IsLetter) out.value = Value::from_bool(is_ascii_letter(c));
    else if (id == BuiltinId::IsDigit) out.value = Value::from_bool(is_ascii_digit(c));
    else if (id == BuiltinId::IsSpace) out.value = Value::from_bool(is_ascii_space(c));
    else if (id == BuiltinId::IsVowel) out.value = Value::from_bool(is_ascii_vowel(c));
    else if (id == BuiltinId::ToLower) out.value = Value::from_char(ascii_to_lower(c));
    else out.value = Value::from_char(ascii_to_upper(c));
    return out;
  }

  if (id == BuiltinId::ToString) {
    if (args.size() != 1 || !is_numeric(args[0])) {
      return fail(ErrCode::Type, "to_string expects int or float");
    }
    if (args[0].tag == ValueTag::Int) {
      return BuiltinResult{false, payload::make_string_value(std::to_string(args[0].i)), Err{ErrCode::Value, ""}};
    }
    return BuiltinResult{false, payload::make_string_value(format_float_current(args[0].f)), Err{ErrCode::Value, ""}};
  }

  if (id == BuiltinId::Singleton) {
    if (args.size() != 1) {
      return fail(ErrCode::Type, "singleton expects 1 argument");
    }
    const Value& x = args[0];
    if (x.tag == ValueTag::Char) {
      if (x.i < 0 || x.i > 255) {
        return fail(ErrCode::Value, "singleton supports byte-sized char in CPU slice");
      }
      return BuiltinResult{false, payload::make_string_value(std::string(1, static_cast<char>(x.i))), Err{ErrCode::Value, ""}};
    }
    if (x.tag == ValueTag::Int) {
      return BuiltinResult{false, payload::make_int_list_value({x}), Err{ErrCode::Value, ""}};
    }
    if (x.tag == ValueTag::Float) {
      return BuiltinResult{false, payload::make_float_list_value({x}), Err{ErrCode::Value, ""}};
    }
    if (x.tag == ValueTag::String) {
      return BuiltinResult{false, payload::make_string_list_value({x}), Err{ErrCode::Value, ""}};
    }
    return fail(ErrCode::Type, "singleton expects int, float, string, or char");
  }

  return fail(ErrCode::Name, "unknown builtin");
}

}  // namespace gagp
