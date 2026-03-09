#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_DIR="${ROOT_DIR}/exp"
CPP_BUILD_DIR="${CPP_BUILD_DIR:-${ROOT_DIR}/cpp/build}"
OUT_BIN="${OUT_BIN:-${EXP_DIR}/bin/g3pvm_repro_proto_bench}"
CUDA_ARCH="${CUDA_ARCH:-89}"
NVCC_BIN="${NVCC_BIN:-/usr/local/cuda/bin/nvcc}"

if [[ ! -f "${CPP_BUILD_DIR}/libg3pvm_cpu.a" ]]; then
  echo "missing ${CPP_BUILD_DIR}/libg3pvm_cpu.a" >&2
  exit 1
fi

if [[ ! -f "${CPP_BUILD_DIR}/libg3pvm_gpu.a" ]]; then
  echo "missing ${CPP_BUILD_DIR}/libg3pvm_gpu.a" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUT_BIN}")"

"${NVCC_BIN}" \
  -DG3PVM_HAS_CUDA=1 \
  -I"${ROOT_DIR}/cpp/include" \
  -g \
  --generate-code="arch=compute_${CUDA_ARCH},code=[compute_${CUDA_ARCH},sm_${CUDA_ARCH}]" \
  -std=c++17 \
  "${EXP_DIR}/repro_proto_bench.cu" \
  -L"${CPP_BUILD_DIR}" \
  -L/usr/local/cuda/targets/x86_64-linux/lib/stubs \
  -L/usr/local/cuda/targets/x86_64-linux/lib \
  -lg3pvm_cpu \
  -lg3pvm_gpu \
  -lcudadevrt \
  -lcudart_static \
  -lrt \
  -lpthread \
  -ldl \
  -o "${OUT_BIN}"

echo "built ${OUT_BIN}"
