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
├── parakeet.js         # hand-written JS API wrapper (Parakeet + streaming)
├── index.html          # drag-and-drop file demo (offline transcription)
├── mic.html            # LIVE microphone demo (cache-aware streaming)
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

The default build is **single-threaded**, which is both the most compatible
(runs from any static host and from `file://`, no special headers) and, in
practice here, the fastest. On the small 110M model transcription runs faster
than real time on a modern laptop (~1.3× real-time in a headless Chromium
measurement on this repo's `speech.wav`).

An experimental pthreads build is wired in behind `PARAKEET_WASM_THREADS=N`:

```sh
PARAKEET_WASM_THREADS=8 scripts/build_wasm.sh
```

It is **not currently recommended** — Emscripten's `-pthread` combined with
`ALLOW_MEMORY_GROWTH` routes heap access through a slower path, and for
parakeet's many small graph computes that overhead outweighed the parallelism
in testing (it came out several times *slower* than single-threaded). Making it
a real win needs a fixed (non-growable) memory build, which trades off support
for the larger models. If you do use it, pthreads need `SharedArrayBuffer`,
which the browser only exposes on **cross-origin isolated** pages — serve with
the COOP/COEP headers (the included `serve.py` already sends them):

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
- **Streaming** (cache-aware EOU / nemotron streaming) is exposed via
  `pk.stream()` / `ParakeetStream` and demonstrated in `mic.html`. It needs a
  streaming model; offline models throw from `pk.stream()`.
