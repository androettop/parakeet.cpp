# parakeet.cpp in the browser (WebAssembly)

Run NVIDIA Parakeet speech recognition **entirely client-side** — the model and
audio never leave the page. parakeet.cpp is compiled to WebAssembly with
Emscripten on the ggml CPU backend and exposed through a small promise-based
JavaScript API.

This is the same idea as `whisper.cpp`'s WASM build: a static `.wasm` + `.mjs`
pair plus a thin JS wrapper you can drop into any page.

```
examples/wasm/
├── parakeet_wasm.cpp   # C glue over the flat C-API (include/parakeet_capi.h)
├── CMakeLists.txt      # the Emscripten target (parakeet.mjs + parakeet.wasm)
├── parakeet.js         # hand-written JS API wrapper (the Parakeet class)
├── index.html          # drag-and-drop demo page
├── serve.py            # dev static server (correct MIME + COOP/COEP headers)
├── test_node.mjs       # Node smoke test
└── dist/               # build output (git-ignored): parakeet.mjs + parakeet.wasm
```

## Build

You need the [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html)
on your `PATH` and the ggml submodule checked out:

```sh
git submodule update --init --recursive

# Install + activate emsdk (once):
git clone https://github.com/emscripten-core/emsdk
cd emsdk && ./emsdk install latest && ./emsdk activate latest && source ./emsdk_env.sh
cd -

# Build (single-threaded, runs from any static host):
scripts/build_wasm.sh
```

This produces `examples/wasm/dist/parakeet.mjs` and `parakeet.wasm`
(~1 MB wasm; the model is downloaded separately at runtime).

## Try the demo

```sh
python3 examples/wasm/serve.py           # http://localhost:8000
```

Open the page, click **Load model** (defaults to the ~125 MB
`tdt_ctc-110m-q4_k` English model on HuggingFace, or drop your own `.gguf`),
drop in an audio file, and hit **Transcribe**. Everything runs locally in the
tab.

## JavaScript API

```js
import { Parakeet, Decoder } from './parakeet.js';

// Load the WASM module (base dir holding parakeet.mjs/.wasm) and a GGUF model
// (a URL, ArrayBuffer, or Uint8Array).
const pk = await Parakeet.load('./dist/',
  'https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/tdt_ctc-110m-q4_k.gguf');

// Decode any browser-supported audio file to mono Float32Array PCM.
const { pcm, sampleRate } = await Parakeet.decodeAudio(await file.arrayBuffer());

// Plain transcript:
const text = pk.transcribe(pcm, sampleRate);

// With per-word / per-token timestamps + confidence:
const doc = pk.transcribeWithTimestamps(pcm, sampleRate);
//   doc = { text, frame_sec, words: [{w,start,end,conf}], tokens: [{id,t,conf}] }

// Multilingual (nemotron) models take a language prompt:
const es = pk.transcribe(pcm, sampleRate, { lang: 'es' });

pk.free();  // release the model when done
```

`pcm` is mono `Float32Array` in `[-1, 1]`. Any sample rate is accepted and
linearly resampled to 16 kHz inside the library, so you can pass the raw output
of `AudioContext.decodeAudioData` directly.

### Model input

`Parakeet.load(base, model)` accepts the model as:

- a **URL string** — fetched with `fetch()` (HuggingFace, your own CDN, a
  `blob:`/`data:` URL, or a same-origin path);
- an **`ArrayBuffer` / `Uint8Array`** — e.g. from a drag-and-dropped `File`;
- in **Node**, a filesystem path string.

The bytes are written into the WASM in-memory filesystem once and loaded by the
native GGUF loader.

## Node

The module also runs under Node (used for CI verification):

```sh
node examples/wasm/test_node.mjs path/to/model.gguf path/to/audio.wav
```

## Performance & threading

The default build is **single-threaded** for maximum compatibility: it runs
from any static host and from `file://`, with no special headers. On the small
110M model, transcription runs several times faster than real time on a modern
laptop.

For a faster multi-threaded build, pass a pthread pool size:

```sh
PARAKEET_WASM_THREADS=4 scripts/build_wasm.sh
```

pthreads use `SharedArrayBuffer`, which the browser only exposes on
**cross-origin isolated** pages. Serve with the COOP/COEP headers (the included
`serve.py` already sends them):

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

## Notes / limitations

- **CPU backend only.** WebGPU is not wired up; ggml's CPU backend (with WASM
  SIMD) does the compute.
- **Memory.** The model is held in WASM memory (growable up to 2 GB). The 110M
  and 0.6B models are comfortable; the 1.1B models need a browser/tab with
  enough memory.
- **Streaming** (cache-aware EOU / nemotron streaming) C-API entry points are
  compiled in and exported, but the JS wrapper currently ships the offline
  `transcribe` / `transcribeWithTimestamps` surface. The streaming functions
  (`parakeet_capi_stream_*`) can be `ccall`'d directly from JS.
