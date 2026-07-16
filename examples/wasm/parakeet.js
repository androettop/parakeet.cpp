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

export default Parakeet;
