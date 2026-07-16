# parakeet.cpp in the browser (WebAssembly)

Run NVIDIA Parakeet speech recognition **entirely client-side** — the model and
audio never leave the page. parakeet.cpp is compiled to WebAssembly with
Emscripten on the ggml CPU backend and exposed through a small promise-based
JavaScript API.

This is the same idea as `whisper.cpp`'s WASM build: a static `.wasm` + `.mjs`
pair plus a thin JS wrapper you can drop into any page.

```
examples/wasm/
├── parakeet_wasm.cpp        # C glue over the flat C-API (include/parakeet_capi.h)
├── CMakeLists.txt           # the Emscripten target (parakeet.mjs + parakeet.wasm)
├── parakeet.js              # single-thread JS API (Parakeet + streaming, sync)
│
│   # single-thread demos (simplest; run from any static host):
├── index.html               #   offline file transcription
├── mic.html                 #   live microphone (cache-aware streaming)
│
│   # multi-thread demos (Web Worker + pthread pool; faster, UI never blocks):
├── parakeet-worker.js       #   Web Worker hosting the threaded module
├── parakeet-threaded.js     #   main-thread async API (ParakeetThreaded)
├── index-threaded.html      #   offline file transcription, multi-core
├── mic-threaded.html        #   live microphone, multi-core
│
├── serve.py                 # dev static server (MIME + COOP/COEP headers)
├── test_node.mjs            # Node smoke test
├── dist/                    # single-thread build output: parakeet.mjs + .wasm
└── dist-threaded/           # multi-thread build output: parakeet.mjs + .wasm
```

Two flavors are prebuilt and committed:

- **Single-thread** (`dist/`, used by `index.html` / `mic.html`) — simplest,
  runs from any static host and `file://`. The WASM compute runs on the calling
  thread.
- **Multi-thread** (`dist-threaded/`, used by `index-threaded.html` /
  `mic-threaded.html`) — the module runs inside a **Web Worker** with a ggml
  **pthread pool**, so it uses multiple CPU cores *and* the browser UI never
  freezes (the compute is off the main thread). Needs a cross-origin-isolated
  page (COOP/COEP headers — `serve.py` sends them) for `SharedArrayBuffer`.

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

The prebuilt `dist/parakeet.mjs` + `dist/parakeet.wasm` are **committed to the
repo**, so you can try the demos without installing Emscripten — just serve the
folder (see below). Rebuild them with `scripts/build_wasm.sh` when you change
the C++.

## Try the demos

```sh
python3 examples/wasm/serve.py           # http://localhost:8000
```

Then open one of:

- **`index.html`** — offline file transcription. Click **Load model** (defaults
  to the ~125 MB `tdt_ctc-110m-q4_k` English model on HuggingFace, or drop your
  own `.gguf`), drop in an audio file, and hit **Transcribe**.
- **`mic.html`** — **live microphone** transcription. Click **Load model**
  (defaults to the streaming `realtime_eou_120m-v1` model), then **Start
  speaking** and watch the transcript build up in real time, with `EOU` /`EOB`
  markers where the model detects end-of-utterance / backchannel.

Everything runs locally in the tab; audio and model never leave the page. The
microphone demo needs `localhost` or HTTPS (a browser requirement for mic
access) — `serve.py` on `localhost` satisfies it.

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

### Live streaming (microphone)

With a **streaming** model (`realtime_eou_120m-v1`, or a nemotron streaming
model) you can feed audio as it arrives and get incremental text plus
end-of-utterance events:

```js
import { Parakeet, Resampler } from './parakeet.js';

const pk = await Parakeet.load('./dist/',
  'https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/realtime_eou_120m-v1-q4_k.gguf');

const stream = pk.stream();                 // begin a streaming session
const resampler = new Resampler(micRate, 16000); // e.g. 48000 -> 16000

// For each block of mic PCM (mono Float32Array at the mic's sample rate):
const r = stream.feed(resampler.process(block));
//   r = { text, eou, eob, frame_sec, events: [{type,frame,t}], words: [...] }
if (r.text) appendToTranscript(r.text);
for (const e of r.events) mark(e.type /* "eou" | "eob" */, e.t);

// When the audio ends, flush the tail:
const tail = stream.finalize();
stream.free();
```

`stream.feed()` expects **16 kHz** mono PCM (unlike the offline calls it does
not resample), which is exactly what the included `Resampler` produces from
arbitrary mic rates. See `mic.html` for the full AudioWorklet capture pipeline.

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

Two builds are provided:

- **Single-thread** (`dist/`, `scripts/build_wasm.sh`) — growable heap, runs
  anywhere including `file://`, no special headers. Compute runs on the calling
  thread; if you call it on the browser main thread the UI blocks while it runs.
- **Multi-thread** (`dist-threaded/`, `PARAKEET_WASM_THREADS=N scripts/build_wasm.sh`)
  — a ggml pthread pool over a **fixed** `SharedArrayBuffer` heap, meant to run
  **inside a Web Worker**. This is what the `*-threaded.html` demos use, and it
  is both faster (multi-core) and keeps the UI responsive.

```sh
scripts/build_wasm.sh                         # single-thread -> dist/
PARAKEET_WASM_THREADS=4 scripts/build_wasm.sh # multi-thread  -> dist-threaded/
```

Two things are essential for the threaded build and are handled here:

1. **Run it in a Web Worker.** ggml's `compute` blocks the thread it runs on
   while its pool works, and the browser main thread is forbidden from blocking
   (`Atomics.wait` throws there) — that is exactly what froze the UI. The module
   runs in `parakeet-worker.js`, so the blocking happens off the main thread.
2. **Fixed memory, not growable.** `-pthread` + `ALLOW_MEMORY_GROWTH` routes
   every heap access through a slow bounds-checked path (Emscripten warns), which
   made an early threaded build several times *slower*. The threaded build uses a
   fixed heap (`-DPARAKEET_WASM_MEMORY_MB`, default 1536, enough for the ≤0.6B
   models); ggml is pinned to the pre-spawned pool size so it never tries to
   spawn a worker on demand (which would deadlock the blocked worker thread).

**Pick a thread count that matches your PHYSICAL cores.** ggml spin-waits
between graph nodes, so oversubscribing (more threads than physical cores) is
badly slower, not just flat — e.g. 8 threads on a 4-core machine measured ~7×
slower than 4 threads. `navigator.hardwareConcurrency` reports *logical* cores
(hyperthreads), so the auto default can overshoot; the `*-threaded.html` demos
expose a **Threads** selector (applies live via `ParakeetThreaded#setThreads`)
so you can find the sweet spot on your machine. The build pre-spawns a pool of 8
(`PARAKEET_WASM_THREADS`), and the JS caps requests to that — asking for more
than the pool would deadlock.

pthreads need `SharedArrayBuffer`, exposed only on **cross-origin isolated**
pages — serve with the COOP/COEP headers (`serve.py` already sends them):

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

### Multi-thread JS API

Same shape as the single-thread API but async (everything crosses to the
worker), and with an explicit `terminate()`:

```js
import { ParakeetThreaded } from './parakeet-threaded.js';

const pk = await ParakeetThreaded.load('./', modelUrl);   // opts.threads caps the pool
const text = await pk.transcribe(pcm, sampleRate);        // offline
const doc  = await pk.transcribeWithTimestamps(pcm, sampleRate);

const st = await pk.stream();                             // realtime (streaming model)
const r  = await st.feed(pcm16k);   // { text, events, words, ... }
await st.finalize(); await st.free();

pk.terminate();
```

## Notes / limitations

- **CPU backend only.** WebGPU is not wired up; ggml's CPU backend (with WASM
  SIMD) does the compute.
- **Memory.** The model is held in WASM memory (growable up to 2 GB). The 110M
  and 0.6B models are comfortable; the 1.1B models need a browser/tab with
  enough memory.
- **Streaming** (cache-aware EOU / nemotron streaming) is exposed via
  `pk.stream()` / `ParakeetStream` and demonstrated in `mic.html`. It needs a
  streaming model; offline models throw from `pk.stream()`.
