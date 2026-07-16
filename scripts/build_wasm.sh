#!/usr/bin/env bash
# Build the parakeet.cpp WebAssembly module with Emscripten.
#
# Produces examples/wasm/dist/parakeet.js + parakeet.wasm, the module the demo
# page (examples/wasm/index.html) and the JS API wrapper (examples/wasm/parakeet.js)
# load.
#
# Requirements:
#   * The Emscripten SDK on PATH (emcmake/emcc). Install with:
#       git clone https://github.com/emscripten-core/emsdk
#       cd emsdk && ./emsdk install latest && ./emsdk activate latest
#       source ./emsdk_env.sh
#   * The ggml submodule checked out:
#       git submodule update --init --recursive
#
# Usage:
#   scripts/build_wasm.sh            # single-threaded (runs from any static host)
#   PARAKEET_WASM_THREADS=4 scripts/build_wasm.sh   # pthreads (needs COOP/COEP)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/build-wasm"
DIST_DIR="${ROOT}/examples/wasm/dist"
THREADS="${PARAKEET_WASM_THREADS:-0}"

if ! command -v emcmake >/dev/null 2>&1; then
    echo "error: emcmake not found on PATH." >&2
    echo "       Install the Emscripten SDK and 'source emsdk_env.sh' first." >&2
    exit 1
fi

if [ ! -e "${ROOT}/third_party/ggml/CMakeLists.txt" ]; then
    echo "error: third_party/ggml is empty; run:" >&2
    echo "       git submodule update --init --recursive" >&2
    exit 1
fi

echo ">> configuring (threads=${THREADS})"
emcmake cmake -S "${ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPARAKEET_BUILD_WASM=ON \
    -DPARAKEET_BUILD_CLI=OFF \
    -DPARAKEET_BUILD_SERVER=OFF \
    -DPARAKEET_BUILD_TESTS=OFF \
    -DPARAKEET_WASM_THREADS="${THREADS}" \
    -DGGML_NATIVE=OFF \
    -DGGML_LLAMAFILE=OFF \
    -DGGML_OPENMP=OFF \
    -DGGML_WASM_SINGLE_FILE=OFF

echo ">> building"
cmake --build "${BUILD_DIR}" -j"$(nproc 2>/dev/null || echo 4)" --target parakeet-wasm

echo ">> collecting artifacts into ${DIST_DIR}"
mkdir -p "${DIST_DIR}"
OUT_DIR="${BUILD_DIR}/examples/wasm"
cp "${OUT_DIR}/parakeet.mjs" "${DIST_DIR}/"
cp "${OUT_DIR}/parakeet.wasm" "${DIST_DIR}/"
if [ -f "${OUT_DIR}/parakeet.worker.mjs" ]; then
    cp "${OUT_DIR}/parakeet.worker.mjs" "${DIST_DIR}/"
fi

echo ">> done:"
ls -la "${DIST_DIR}"
