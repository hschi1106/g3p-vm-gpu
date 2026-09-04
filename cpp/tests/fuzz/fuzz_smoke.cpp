#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

#include "fuzz_entrypoints.hpp"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: gagp_test_decode_fuzz_smoke CORPUS_DIR\n";
    return 2;
  }
  int corpus_cases = 0;
  for (const auto& entry : std::filesystem::directory_iterator(argv[1])) {
    if (!entry.is_regular_file()) continue;
    std::ifstream input(entry.path(), std::ios::binary);
    const std::vector<std::uint8_t> bytes(
        (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    fuzz_bytecode_json(bytes.data(), bytes.size());
    fuzz_ast_verify(bytes.data(), bytes.size());
    ++corpus_cases;
  }
  if (corpus_cases == 0) {
    std::cerr << "fuzz smoke corpus is empty\n";
    return 1;
  }

  std::mt19937_64 rng(0x5eedULL);
  std::uniform_int_distribution<int> length_dist(0, 256);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  for (int iteration = 0; iteration < 1000; ++iteration) {
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length_dist(rng)));
    for (std::uint8_t& byte : bytes) byte = static_cast<std::uint8_t>(byte_dist(rng));
    fuzz_bytecode_json(bytes.data(), bytes.size());
    fuzz_ast_verify(bytes.data(), bytes.size());
  }
  std::cout << "gagp_test_decode_fuzz_smoke: OK corpus " << corpus_cases
            << " random 1000\n";
  return 0;
}
