#include "g3pvm/runtime/cpu/builtins_cpu.hpp"

#include <algorithm>

#include "g3pvm/core/value_semantics.hpp"
#include "g3pvm/runtime/payload/payload.hpp"

namespace g3pvm {

namespace {

bool to_numeric_pair(const Value& a, const Value& b, double& a_out, double& b_out,
                     bool& any_float) {
  if (!is_numeric(a) || !is_numeric(b)) {
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
  return tag == ValueTag::NumList || tag == ValueTag::StringList;
}

std::uint8_t list_type_code(const Value& v) {
  return v.tag == ValueTag::NumList ? 2U : 3U;
}

Value make_list_hash_len(const Value& src, std::uint64_t h, std::uint32_t len) {
  return src.tag == ValueTag::NumList ? Value::from_num_list_hash_len(h, len)
                                      : Value::from_string_list_hash_len(h, len);
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
    const bool any_float =
        (x.tag == ValueTag::Float) || (lo.tag == ValueTag::Float) || (hi.tag == ValueTag::Float);
    if (any_float) {
      const double x2 = (x.tag == ValueTag::Float) ? x.f : static_cast<double>(x.i);
      const double lo2 = (lo.tag == ValueTag::Float) ? lo.f : static_cast<double>(lo.i);
      const double hi2 = (hi.tag == ValueTag::Float) ? hi.f : static_cast<double>(hi.i);
      if (lo2 > hi2) {
        return fail(ErrCode::Value, "clip requires lo <= hi");
      }
      BuiltinResult out;
      if (x2 < lo2) {
        out.value = Value::from_float(vm_semantics::canonicalize_vm_float(lo2));
      } else if (x2 > hi2) {
        out.value = Value::from_float(vm_semantics::canonicalize_vm_float(hi2));
      } else {
        out.value = Value::from_float(vm_semantics::canonicalize_vm_float(x2));
      }
      return out;
    }

    const long long x2 = x.i;
    const long long lo2 = lo.i;
    const long long hi2 = hi.i;
    if (lo2 > hi2) {
      return fail(ErrCode::Value, "clip requires lo <= hi");
    }
    BuiltinResult out;
    if (x2 < lo2) {
      out.value = Value::from_int(lo2);
    } else if (x2 > hi2) {
      out.value = Value::from_int(hi2);
    } else {
      out.value = Value::from_int(x2);
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
        if (a.tag == ValueTag::NumList) {
          return BuiltinResult{false, payload::make_num_list_value(out_elems), Err{ErrCode::Value, ""}};
        }
        return BuiltinResult{false, payload::make_string_list_value(out_elems), Err{ErrCode::Value, ""}};
      }
      const std::uint32_t len = Value::saturating_len_add(Value::container_len(a), Value::container_len(b));
      const std::uint64_t h = Value::combine_container_hash48(list_type_code(a), a, b);
      BuiltinResult out;
      out.value = Value::from_fallback_token(Value::pack_container_payload(h, len));
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
      const std::uint64_t out_h = Value::slice_container_hash48(3U, x, lo.i, hi.i);
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
        if (x.tag == ValueTag::NumList) {
          return BuiltinResult{false, payload::make_num_list_value(out_elems), Err{ErrCode::Value, ""}};
        }
        return BuiltinResult{false, payload::make_string_list_value(out_elems), Err{ErrCode::Value, ""}};
      }
      const std::uint64_t out_h = Value::slice_container_hash48(list_type_code(x), x, lo.i, hi.i);
      BuiltinResult out;
      out.value = Value::from_fallback_token(Value::pack_container_payload(out_h, out_len));
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
        out.value = payload::make_string_value(std::string(1, ch));
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
    if (xs.tag == ValueTag::NumList && !is_numeric(elem)) {
      return fail(ErrCode::Type, "append expects numeric element for NumList");
    }
    if (xs.tag == ValueTag::StringList && elem.tag != ValueTag::String) {
      return fail(ErrCode::Type, "append expects string element for StringList");
    }
    std::vector<Value> lx;
    if (payload::lookup_list(xs, &lx)) {
      lx.push_back(elem);
      if (xs.tag == ValueTag::NumList) {
        return BuiltinResult{false, payload::make_num_list_value(lx), Err{ErrCode::Value, ""}};
      }
      return BuiltinResult{false, payload::make_string_list_value(lx), Err{ErrCode::Value, ""}};
    }
    const std::uint32_t len = Value::saturating_len_add(Value::container_len(xs), 1U);
    const std::uint64_t h = Value::append_list_hash48(list_type_code(xs), xs, elem);
    BuiltinResult out;
    out.value = Value::from_fallback_token(Value::pack_container_payload(h, len));
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
      if (x.tag == ValueTag::NumList) {
        return BuiltinResult{false, payload::make_num_list_value(lx), Err{ErrCode::Value, ""}};
      }
      return BuiltinResult{false, payload::make_string_list_value(lx), Err{ErrCode::Value, ""}};
    }
    const std::uint64_t h = Value::reverse_container_hash48(list_type_code(x), x);
    BuiltinResult out;
    out.value = Value::from_fallback_token(Value::pack_container_payload(h, Value::container_len(x)));
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

  return fail(ErrCode::Name, "unknown builtin");
}

}  // namespace g3pvm
