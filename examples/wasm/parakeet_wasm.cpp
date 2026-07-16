// Emscripten/WASM glue for parakeet.cpp.
//
// This is a thin layer over the stable flat C-API (include/parakeet_capi.h).
// Everything the browser needs — loading a GGUF model, transcribing in-memory
// 16 kHz mono float PCM to text or to a per-word/-token JSON document, and
// cache-aware streaming — is already exposed there and never lets a C++
// exception cross the boundary. We only add:
//
//   * pk_wasm_transcribe_pcm_json: a single-clip wrapper over the batch-JSON
//     entry point so the JS side gets one object (not a one-element array).
//   * pk_wasm_version: the library version string.
//
// The remaining parakeet_capi_* symbols are exported directly via the linker's
// EXPORTED_FUNCTIONS list (see CMakeLists.txt), and marshalled from JS in
// parakeet.js. Model bytes are written into Emscripten's in-memory filesystem
// (MEMFS) by the JS wrapper, then loaded here by path through the normal
// gguf_init_from_file path.

#include "parakeet_capi.h"

#include <cstdlib>
#include <cstring>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define PK_WASM_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define PK_WASM_EXPORT
#endif

// C++-linkage declaration of the thread-count override in libparakeet
// (src/ggml_graph.cpp). Declared OUTSIDE extern "C" so it resolves to the
// mangled pk::set_num_threads symbol.
namespace pk { void set_num_threads(int n); }

extern "C" {

// Transcribe one in-memory 16 kHz (or arbitrary — it is resampled) mono float
// PCM clip and return the rich timestamp JSON document as a single object
// (shape identical to parakeet_capi_transcribe_path_json). Delegates to the
// batched JSON entry point with n_clips == 1 and unwraps the surrounding array
// so the JS caller can JSON.parse straight into an object.
//
// `target_lang` selects the language prompt for multilingual (nemotron) models;
// NULL or "" uses the model default. Ignored by non-prompt models.
//
// Returns a malloc'd, NUL-terminated UTF-8 string (free with
// parakeet_capi_free_string) or NULL on error (see parakeet_capi_last_error).
PK_WASM_EXPORT
char* pk_wasm_transcribe_pcm_json(parakeet_ctx* ctx, const float* samples,
                                  int n_samples, int sample_rate, int decoder,
                                  const char* target_lang) {
    int ns = n_samples;
    char* arr = parakeet_capi_transcribe_pcm_batch_json_lang(
        ctx, samples, &ns, /*n_clips=*/1, sample_rate, decoder, target_lang);
    if (!arr) return nullptr;

    // The batch document is a JSON array of exactly one object: "[{...}]".
    // Strip the outer brackets so the JS side receives the bare object. Be
    // defensive — if the shape is unexpected, hand back the raw document.
    size_t len = std::strlen(arr);
    size_t b = 0;
    while (b < len && (arr[b] == ' ' || arr[b] == '\n' || arr[b] == '\t' ||
                       arr[b] == '\r'))
        ++b;
    size_t e = len;
    while (e > b && (arr[e - 1] == ' ' || arr[e - 1] == '\n' ||
                     arr[e - 1] == '\t' || arr[e - 1] == '\r'))
        --e;
    if (b < e && arr[b] == '[' && arr[e - 1] == ']') {
        size_t inner_len = (e - 1) - (b + 1);
        char* obj = (char*)std::malloc(inner_len + 1);
        if (obj) {
            std::memcpy(obj, arr + b + 1, inner_len);
            obj[inner_len] = '\0';
            parakeet_capi_free_string(arr);
            return obj;
        }
    }
    return arr;  // fall back to the raw array document
}

// Set the ggml compute thread count for every subsequent graph computation
// (encoder is the bulk). In the pthreads build the JS side pins this to the
// PTHREAD_POOL_SIZE so ggml never tries to spawn a worker on demand (which would
// deadlock the module's blocked worker thread). No-op-safe on the
// single-threaded build (ggml just runs on one thread regardless).
PK_WASM_EXPORT
void pk_wasm_set_threads(int n) { pk::set_num_threads(n); }

// The parakeet.cpp version string (e.g. "0.0.1"). malloc'd; free with
// parakeet_capi_free_string.
PK_WASM_EXPORT
char* pk_wasm_version(void) {
    extern const char* parakeet_version(void);
    const char* v = parakeet_version();
    size_t n = std::strlen(v);
    char* out = (char*)std::malloc(n + 1);
    if (out) std::memcpy(out, v, n + 1);
    return out;
}

}  // extern "C"
