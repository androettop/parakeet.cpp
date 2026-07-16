// parakeet.js — a small, promise-based JavaScript API over the parakeet.cpp
// WebAssembly module (dist/parakeet.js, the createParakeetModule factory).
//
// Usage (browser, ES module):
//
//   import { Parakeet } from './parakeet.js';
//
//   const pk = await Parakeet.load('./dist/', 'https://.../tdt_ctc-110m-q4_k.gguf');
//   const pcm = await Parakeet.decodeAudio(await file.arrayBuffer()); // Float32Array @16k
//   const text = pk.transcribe(pcm, 16000);
//   const doc  = pk.transcribeWithTimestamps(pcm, 16000); // {text, words, tokens, ...}
//   pk.free();
//
// Usage (node, CommonJS-ish via dynamic import): see examples/wasm/test_node.mjs.
//
// The module is CPU-only and single-threaded by default (see scripts/build_wasm.sh).
// Model bytes are written into the WASM in-memory filesystem (MEMFS) once, then
// loaded by the native gguf_init_from_file path.

// Decoder selector matching the C-API: 0 = default (by arch), 1 = force CTC,
// 2 = force TDT/RNN-T head.
export const Decoder = Object.freeze({ DEFAULT: 0, CTC: 1, TDT: 2 });

// Resolve the module factory across environments. When bundled with the
// Emscripten output the caller passes an explicit factory or a base URL from
// which we import ./parakeet.js.
async function resolveFactory(moduleOrBase) {
  if (typeof moduleOrBase === 'function') return moduleOrBase;
  if (moduleOrBase && typeof moduleOrBase.default === 'function') {
    return moduleOrBase.default;
  }
  // Treat as a base directory/URL containing the Emscripten dist/parakeet.mjs.
  let base = typeof moduleOrBase === 'string' ? moduleOrBase : './';
  if (!base.endsWith('/')) base += '/';
  const url = base + 'parakeet.mjs';
  const mod = await import(/* @vite-ignore */ url);
  if (typeof mod.default !== 'function') {
    throw new Error(`parakeet: ${url} did not export the createParakeetModule factory`);
  }
  return mod.default;
}

async function fetchModelBytes(model) {
  if (model instanceof Uint8Array) return model;
  if (model instanceof ArrayBuffer) return new Uint8Array(model);
  if (typeof model === 'string') {
    // Node with a bare filesystem path (no URL scheme) reads from disk; the
    // browser (or an explicit http/blob/data URL) fetches over the network.
    const isNode = typeof process !== 'undefined' &&
                   !!(process.versions && process.versions.node) &&
                   typeof window === 'undefined';
    const looksLikeUrl = /^(https?|blob|data):/.test(model);
    if (isNode && !looksLikeUrl) {
      const fs = await import('node:fs/promises');
      return new Uint8Array(await fs.readFile(model));
    }
    const resp = await fetch(model);
    if (!resp.ok) throw new Error(`parakeet: failed to fetch model ${model}: ${resp.status}`);
    return new Uint8Array(await resp.arrayBuffer());
  }
  throw new Error('parakeet: model must be a URL string, ArrayBuffer, or Uint8Array');
}

export class Parakeet {
  constructor(module, ctxPtr) {
    this._m = module;
    this._ctx = ctxPtr;
    this._freed = false;
  }

  // Load the WASM module (from `moduleSource`: a base URL/dir, the imported
  // module namespace, or the factory function) and a GGUF model (`model`: a
  // URL string, ArrayBuffer, Uint8Array, or — in node — a filesystem path).
  // Returns a ready Parakeet instance.
  static async load(moduleSource, model, opts = {}) {
    const factory = await resolveFactory(moduleSource);
    const moduleArg = {};
    // When moduleSource is a base URL, tell Emscripten where to find the .wasm.
    if (typeof moduleSource === 'string') {
      let base = moduleSource.endsWith('/') ? moduleSource : moduleSource + '/';
      moduleArg.locateFile = (path, prefix) =>
        path.endsWith('.wasm') ? base + path : prefix + path;
    }
    if (opts.print) moduleArg.print = opts.print;
    if (opts.printErr) moduleArg.printErr = opts.printErr;
    const m = await factory(moduleArg);

    const bytes = await fetchModelBytes(model);
    const path = opts.modelPath || '/model.gguf';
    m.FS.writeFile(path, bytes);

    const ctx = m.ccall('parakeet_capi_load', 'number', ['string'], [path]);
    if (!ctx) {
      throw new Error('parakeet: failed to load model (see console for details)');
    }
    // The MEMFS copy is no longer needed once the model is resident in the
    // loader's own ggml context; drop it to reclaim ~model-size bytes.
    try { m.FS.unlink(path); } catch (_) { /* ignore */ }

    return new Parakeet(m, ctx);
  }

  _assertLive() {
    if (this._freed) throw new Error('parakeet: instance already freed');
  }

  // Copy a Float32Array of mono PCM into the WASM heap. Returns [ptr, length].
  _pushPcm(pcm) {
    const f32 = pcm instanceof Float32Array ? pcm : Float32Array.from(pcm);
    const bytes = f32.length * 4;
    const ptr = this._m._malloc(bytes || 4);
    this._m.HEAPF32.set(f32, ptr >>> 2);
    return [ptr, f32.length];
  }

  _takeString(ptr) {
    if (!ptr) return null;
    const s = this._m.UTF8ToString(ptr);
    this._m.ccall('parakeet_capi_free_string', null, ['number'], [ptr]);
    return s;
  }

  _lastError() {
    const p = this._m.ccall('parakeet_capi_last_error', 'number', ['number'], [this._ctx]);
    return p ? this._m.UTF8ToString(p) : '';
  }

  // Transcribe mono float PCM to plain text. `sampleRate` defaults to 16000;
  // any other rate is linearly resampled to 16 kHz inside the library.
  // `opts`: { decoder: Decoder.*, lang: 'en'|'de'|'auto'|... }.
  transcribe(pcm, sampleRate = 16000, opts = {}) {
    this._assertLive();
    const decoder = opts.decoder ?? Decoder.DEFAULT;
    const lang = opts.lang ?? '';
    const [ptr, n] = this._pushPcm(pcm);
    try {
      const outPtr = this._m.ccall(
        'parakeet_capi_transcribe_pcm_lang', 'number',
        ['number', 'number', 'number', 'number', 'number', 'string'],
        [this._ctx, ptr, n, sampleRate, decoder, lang]);
      const text = this._takeString(outPtr);
      if (text === null) throw new Error('parakeet: transcribe failed: ' + this._lastError());
      return text;
    } finally {
      this._m._free(ptr);
    }
  }

  // Transcribe mono float PCM and return the rich document
  //   { text, frame_sec, words: [{w,start,end,conf}], tokens: [{id,t,conf}] }
  // with per-word and per-token timestamps + confidence.
  transcribeWithTimestamps(pcm, sampleRate = 16000, opts = {}) {
    this._assertLive();
    const decoder = opts.decoder ?? Decoder.DEFAULT;
    const lang = opts.lang ?? '';
    const [ptr, n] = this._pushPcm(pcm);
    try {
      const outPtr = this._m.ccall(
        'pk_wasm_transcribe_pcm_json', 'number',
        ['number', 'number', 'number', 'number', 'number', 'string'],
        [this._ctx, ptr, n, sampleRate, decoder, lang]);
      const json = this._takeString(outPtr);
      if (json === null) throw new Error('parakeet: transcribe failed: ' + this._lastError());
      return JSON.parse(json);
    } finally {
      this._m._free(ptr);
    }
  }

  // Begin a cache-aware streaming session for live transcription. Requires a
  // streaming model (e.g. parakeet_realtime_eou_120m-v1, or a nemotron
  // streaming model). Returns a ParakeetStream. `opts.lang` picks the language
  // prompt for multilingual streaming models. Throws if the model is not a
  // streaming model.
  stream(opts = {}) {
    this._assertLive();
    const lang = opts.lang ?? '';
    const s = this._m.ccall('parakeet_capi_stream_begin_lang', 'number',
                            ['number', 'string'], [this._ctx, lang]);
    if (!s) {
      throw new Error('parakeet: could not begin streaming (not a streaming model?): ' +
                      this._lastError());
    }
    return new ParakeetStream(this._m, s);
  }

  // The parakeet.cpp C-API ABI version compiled into this module.
  abiVersion() {
    return this._m.ccall('parakeet_capi_abi_version', 'number', [], []);
  }

  // Free the model/context. The instance is unusable afterwards.
  free() {
    if (this._freed) return;
    this._m.ccall('parakeet_capi_free', null, ['number'], [this._ctx]);
    this._ctx = 0;
    this._freed = true;
  }

  // ---- static helpers -----------------------------------------------------

  // Decode compressed/uncompressed audio bytes (WAV/MP3/OGG/FLAC/... — anything
  // the browser's Web Audio API understands) into a mono Float32Array. Returns
  // { pcm, sampleRate }. Downmixes multi-channel to mono. Browser-only (uses
  // OfflineAudioContext / AudioContext). Pass the raw ArrayBuffer of the file.
  static async decodeAudio(arrayBuffer, targetSampleRate = 16000) {
    const AC = globalThis.AudioContext || globalThis.webkitAudioContext ||
               globalThis.OfflineAudioContext;
    if (!AC) throw new Error('parakeet: Web Audio API not available in this environment');
    // decodeAudioData needs a real (or offline) context. Use an OfflineAudioContext
    // so we can also request a specific sample rate on browsers that honor it.
    let ctx;
    if (globalThis.OfflineAudioContext) {
      ctx = new OfflineAudioContext(1, 1, targetSampleRate);
    } else {
      ctx = new AC();
    }
    const buf = await ctx.decodeAudioData(arrayBuffer.slice(0));
    const ch = buf.numberOfChannels;
    const n = buf.length;
    const mono = new Float32Array(n);
    for (let c = 0; c < ch; c++) {
      const data = buf.getChannelData(c);
      for (let i = 0; i < n; i++) mono[i] += data[i];
    }
    if (ch > 1) for (let i = 0; i < n; i++) mono[i] /= ch;
    return { pcm: mono, sampleRate: buf.sampleRate };
  }
}

// A live cache-aware streaming session. Feed 16 kHz mono float PCM as it
// arrives; each feed returns the text finalized since the last feed plus any
// end-of-utterance (<EOU>) / backchannel (<EOB>) events and per-word timestamps.
// Create via Parakeet#stream(). Feed order matters — the session carries encoder
// and decoder state across calls.
export class ParakeetStream {
  constructor(module, streamPtr) {
    this._m = module;
    this._s = streamPtr;
    this._freed = false;
  }

  _pushPcm(pcm) {
    const f32 = pcm instanceof Float32Array ? pcm : Float32Array.from(pcm);
    const ptr = this._m._malloc((f32.length * 4) || 4);
    this._m.HEAPF32.set(f32, ptr >>> 2);
    return [ptr, f32.length];
  }

  _takeJson(ptr) {
    if (!ptr) throw new Error('parakeet: streaming call failed');
    const s = this._m.UTF8ToString(ptr);
    this._m.ccall('parakeet_capi_free_string', null, ['number'], [ptr]);
    return JSON.parse(s);
  }

  // Feed a block of 16 kHz MONO float PCM. Returns
  //   { text, eou, eob, frame_sec, events: [{type,frame,t}], words: [{w,start,end,conf}] }
  // where `text` is the newly-finalized text since the last feed ("" if none),
  // `eou`/`eob` are 1 when that event fired during this feed.
  feed(pcm16k) {
    if (this._freed) throw new Error('parakeet: stream already freed');
    const [ptr, n] = this._pushPcm(pcm16k);
    try {
      const out = this._m.ccall('parakeet_capi_stream_feed_json', 'number',
                                ['number', 'number', 'number'], [this._s, ptr, n]);
      return this._takeJson(out);
    } finally {
      this._m._free(ptr);
    }
  }

  // Flush the end-of-stream tail. Returns the same shape as feed(); call once
  // when the audio ends. The running transcript is complete afterwards.
  finalize() {
    if (this._freed) throw new Error('parakeet: stream already freed');
    const out = this._m.ccall('parakeet_capi_stream_finalize_json', 'number',
                              ['number'], [this._s]);
    return this._takeJson(out);
  }

  // Release the streaming session.
  free() {
    if (this._freed) return;
    this._m.ccall('parakeet_capi_stream_free', null, ['number'], [this._s]);
    this._s = 0;
    this._freed = true;
  }
}

// Continuous linear resampler: converts a stream of float PCM blocks from
// `inRate` to `outRate` (default 16 kHz), preserving the fractional read
// position across blocks so there are no clicks at block boundaries. Use it to
// turn microphone audio (typically 44.1/48 kHz) into the 16 kHz mono the
// streaming model expects.
export class Resampler {
  constructor(inRate, outRate = 16000) {
    this.ratio = outRate / inRate;      // output samples per input sample
    this._tail = 0;                     // last sample of the previous block
    this._havePrev = false;
    this._pos = 0;                      // fractional position within the input stream
  }

  // Feed one mono Float32Array block; returns a Float32Array of ~block*ratio
  // resampled samples (16 kHz).
  process(block) {
    if (this.ratio === 1) return block.slice();
    const out = [];
    // Reconstruct a virtual input array [prevTail, ...block] so `_pos` (relative
    // to prevTail at index 0) can interpolate across the boundary.
    const prev = this._havePrev ? this._tail : (block.length ? block[0] : 0);
    let pos = this._pos;                 // in units of input samples, 0 == prev
    const N = block.length;
    const sampleAt = (idx) => (idx <= 0 ? prev : (idx - 1 < N ? block[idx - 1] : block[N - 1]));
    // Emit while the right neighbor is available (idx+1 <= N).
    while (pos + 1 <= N) {
      const i0 = Math.floor(pos);
      const frac = pos - i0;
      const a = sampleAt(i0);
      const b = sampleAt(i0 + 1);
      out.push(a + (b - a) * frac);
      pos += 1 / this.ratio;
    }
    // Carry the remainder into the next block: shift the origin to the block end.
    this._pos = pos - N;
    this._tail = N ? block[N - 1] : prev;
    this._havePrev = true;
    return Float32Array.from(out);
  }
}

export default Parakeet;
