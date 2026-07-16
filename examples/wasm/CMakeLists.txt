# WebAssembly (Emscripten) module for parakeet.cpp.
#
# Produces bin/parakeet.js (+ parakeet.wasm) — a MODULARIZE'd factory named
# createParakeetModule that the hand-written examples/wasm/parakeet.js API
# wrapper drives. Only built under an Emscripten toolchain; configure with
#
#   emcmake cmake -B build-wasm -DPARAKEET_BUILD_WASM=ON \
#       -DPARAKEET_BUILD_CLI=OFF -DPARAKEET_BUILD_SERVER=OFF \
#       -DGGML_NATIVE=OFF -DGGML_LLAMAFILE=OFF -DGGML_OPENMP=OFF
#
# See scripts/build_wasm.sh for the canonical invocation.

if(NOT EMSCRIPTEN)
    message(WARNING
        "PARAKEET_BUILD_WASM is ON but the compiler is not Emscripten; "
        "skipping the wasm target. Configure with `emcmake cmake ...`.")
    return()
endif()

add_executable(parakeet-wasm parakeet_wasm.cpp)
target_link_libraries(parakeet-wasm PRIVATE parakeet)
# Emit an ES6 module (parakeet.mjs with `export default createParakeetModule`)
# so the same file loads via import() in the browser and in Node.
set_target_properties(parakeet-wasm PROPERTIES OUTPUT_NAME "parakeet" SUFFIX ".mjs")

# The flat C-API entry points the JS wrapper marshals, plus our two wasm
# helpers and the allocator. Everything not on this list is dead-code-eliminated.
set(PARAKEET_WASM_EXPORTS
    "_parakeet_capi_abi_version"
    "_parakeet_capi_load"
    "_parakeet_capi_free"
    "_parakeet_capi_transcribe_pcm"
    "_parakeet_capi_transcribe_pcm_lang"
    "_parakeet_capi_transcribe_path"
    "_parakeet_capi_transcribe_path_json"
    "_parakeet_capi_stream_begin"
    "_parakeet_capi_stream_begin_lang"
    "_parakeet_capi_stream_feed"
    "_parakeet_capi_stream_feed_json"
    "_parakeet_capi_stream_finalize"
    "_parakeet_capi_stream_finalize_json"
    "_parakeet_capi_stream_free"
    "_parakeet_capi_last_error"
    "_parakeet_capi_free_string"
    "_pk_wasm_transcribe_pcm_json"
    "_pk_wasm_set_threads"
    "_pk_wasm_version"
    "_malloc"
    "_free")
string(REPLACE ";" "," PARAKEET_WASM_EXPORTS_JOINED "${PARAKEET_WASM_EXPORTS}")

set(PARAKEET_WASM_RUNTIME_METHODS
    "ccall,cwrap,FS,HEAPF32,HEAPU8,UTF8ToString,stringToUTF8,lengthBytesUTF8,getValue,setValue")

target_link_options(parakeet-wasm PRIVATE
    "-sMODULARIZE=1"
    "-sEXPORT_ES6=1"
    "-sEXPORT_NAME=createParakeetModule"
    "-sEXPORTED_FUNCTIONS=[${PARAKEET_WASM_EXPORTS_JOINED}]"
    "-sEXPORTED_RUNTIME_METHODS=[${PARAKEET_WASM_RUNTIME_METHODS}]"
    "-sSTACK_SIZE=1MB"
    "-sFORCE_FILESYSTEM=1"
    "-sENVIRONMENT=web,worker,node"
    "-sEXIT_RUNTIME=0"
    "-sALLOW_TABLE_GROWTH=1"
    "-O3")

# Optional pthreads build. ggml multithreading in the browser needs
# SharedArrayBuffer (cross-origin isolation / COOP+COEP headers), and the module
# must run INSIDE a Web Worker — ggml's compute blocks the calling thread waiting
# on its worker pool, and the browser main thread is forbidden from blocking
# (Atomics.wait throws there), which is what froze the UI. Off by default so the
# single-threaded module still runs from any static host and file://; enable with
# -DPARAKEET_WASM_THREADS=<N>.
#
# IMPORTANT: the threaded build uses FIXED memory (no ALLOW_MEMORY_GROWTH).
# -pthread + growable memory routes every heap access through a slow bounds-checked
# path (Emscripten warns about it) and measured several times slower; a fixed
# SharedArrayBuffer is required for fast threaded compute. Size it for the model
# via -DPARAKEET_WASM_MEMORY_MB (default 1536, enough for the <=0.6B models).
set(PARAKEET_WASM_THREADS "0" CACHE STRING
    "Emscripten pthread pool size for the wasm build (0 = single-threaded)")
set(PARAKEET_WASM_MEMORY_MB "1536" CACHE STRING
    "Fixed wasm heap size (MB) for the threaded build")
if(PARAKEET_WASM_THREADS STREQUAL "0")
    # Single-threaded: growable heap, runs anywhere.
    target_link_options(parakeet-wasm PRIVATE
        "-sALLOW_MEMORY_GROWTH=1"
        "-sINITIAL_MEMORY=64MB"
        "-sMAXIMUM_MEMORY=2GB")
else()
    # Threaded: fixed heap (SharedArrayBuffer) + pre-spawned worker pool.
    target_compile_options(parakeet-wasm PRIVATE "-pthread")
    target_link_options(parakeet-wasm PRIVATE
        "-pthread"
        "-sPTHREAD_POOL_SIZE=${PARAKEET_WASM_THREADS}"
        "-sINITIAL_MEMORY=${PARAKEET_WASM_MEMORY_MB}MB")
endif()
