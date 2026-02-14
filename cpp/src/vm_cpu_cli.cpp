#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "g3pvm/bytecode.hpp"
#include "g3pvm/errors.hpp"
#include "g3pvm/value.hpp"
#include "g3pvm/vm_cpu.hpp"

namespace {

using g3pvm::BytecodeProgram;
using g3pvm::Instr;
using g3pvm::Value;
using g3pvm::ValueTag;

bool parse_value(std::istream& in, Value& out) {
  std::string type;
  if (!(in >> type)) return false;
  if (type == "int") {
    long long v = 0;
    if (!(in >> v)) return false;
    out = Value::from_int(v);
    return true;
  }
  if (type == "float") {
    double v = 0.0;
    if (!(in >> v)) return false;
    out = Value::from_float(v);
    return true;
  }
  if (type == "bool") {
    int b = 0;
    if (!(in >> b)) return false;
    out = Value::from_bool(b != 0);
    return true;
  }
  if (type == "none") {
    out = Value::none();
    return true;
  }
  return false;
}

bool parse_opt_int(const std::string& tok, int& out, bool& has_value) {
  if (tok == "x") {
    has_value = false;
    return true;
  }
  has_value = true;
  out = std::stoi(tok);
  return true;
}

void print_value(const Value& v) {
  if (v.tag == ValueTag::Int) {
    std::cout << "int " << v.i << "\n";
    return;
  }
  if (v.tag == ValueTag::Float) {
    std::cout << "float " << std::setprecision(17) << v.f << "\n";
    return;
  }
  if (v.tag == ValueTag::Bool) {
    std::cout << "bool " << (v.b ? 1 : 0) << "\n";
    return;
  }
  std::cout << "none\n";
}

}  // namespace

int main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);

  std::string token;
  int fuel = 0;
  int n_locals = 0;
  int n_consts = 0;
  int n_code = 0;
  int n_inputs = 0;

  BytecodeProgram program;
  std::vector<std::pair<int, Value>> inputs;

  if (!(std::cin >> token) || token != "FUEL") return 2;
  if (!(std::cin >> fuel)) return 2;

  if (!(std::cin >> token) || token != "N_LOCALS") return 2;
  if (!(std::cin >> n_locals)) return 2;
  program.n_locals = n_locals;

  if (!(std::cin >> token) || token != "N_CONSTS") return 2;
  if (!(std::cin >> n_consts)) return 2;
  for (int i = 0; i < n_consts; ++i) {
    if (!(std::cin >> token) || token != "CONST") return 2;
    Value v;
    if (!parse_value(std::cin, v)) return 2;
    program.consts.push_back(v);
  }

  if (!(std::cin >> token) || token != "N_CODE") return 2;
  if (!(std::cin >> n_code)) return 2;
  for (int i = 0; i < n_code; ++i) {
    if (!(std::cin >> token) || token != "INS") return 2;
    Instr ins;
    std::string a_tok;
    std::string b_tok;
    if (!(std::cin >> ins.op >> a_tok >> b_tok)) return 2;
    if (!parse_opt_int(a_tok, ins.a, ins.has_a)) return 2;
    if (!parse_opt_int(b_tok, ins.b, ins.has_b)) return 2;
    program.code.push_back(ins);
  }

  if (!(std::cin >> token) || token != "N_INPUTS") return 2;
  if (!(std::cin >> n_inputs)) return 2;
  for (int i = 0; i < n_inputs; ++i) {
    if (!(std::cin >> token) || token != "INPUT") return 2;
    int idx = -1;
    if (!(std::cin >> idx)) return 2;
    Value v;
    if (!parse_value(std::cin, v)) return 2;
    inputs.push_back({idx, v});
  }

  g3pvm::VMResult result = g3pvm::run_bytecode(program, inputs, fuel);
  if (result.is_error) {
    std::cout << "ERR " << g3pvm::err_code_name(result.err.code) << "\n";
    return 0;
  }

  std::cout << "OK ";
  print_value(result.value);
  return 0;
}
