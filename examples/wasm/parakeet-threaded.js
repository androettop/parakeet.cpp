// parakeet-threaded.js — main-thread API for the MULTI-THREADED, Web-Worker
// hosted parakeet.cpp. Everything is async (returns Promises) because the actual
// WASM compute runs in parakeet-worker.js off the main thread, so the browser UI
// never freezes and ggml's pthread pool uses multiple CPU cores.
//
// Mirrors the synchronous single-thread API in parakeet.js, but async:
//
//   import { ParakeetThreaded, Resampler } from './parakeet-threaded.js';
//   const pk = await ParakeetThreaded.load('./', modelUrl, { threads: 4 });
//   const text = await pk.transcribe(pcm, sampleRate);          // offline
//   const doc  = await pk.transcribeWithTimestamps(pcm, sampleRate);
//   const st   = await pk.stream();                             // realtime
//   const r    = await st.feed(pcm16k);   // { text, events, words, ... }
//   await st.finalize(); await st.free();
//   pk.terminate();
//
// Requires the threaded build (examples/wasm/dist-threaded/) and a
// cross-origin-isolated page (COOP/COEP headers — serve.py sends them) so
// SharedArrayBuffer is available.

export { Resampler } from './parakeet.js';  // reuse the same streaming resampler

export const Decoder = Object.freeze({ DEFAULT: 0, CTC: 1, TDT: 2 });

// MUST equal the threaded build's PTHREAD_POOL_SIZE (PARAKEET_WASM_THREADS).
// ggml is pinned to at most this many threads; requesting more than the
// pre-spawned pool would make it spawn a worker on demand from the blocked
// worker thread, which DEADLOCKS (the symptom is transcription that never
// finishes). Keep this in sync with scripts/build_wasm.sh's default.
const MAX_THREADS = 8;

// Turn a Float32Array into a transferable ArrayBuffer (copy so the caller keeps
// its data), for zero-copy postMessage into the worker.
function toTransfer(pcm) {
  const f32 = pcm instanceof Float32Array ? pcm : Float32Array.from(pcm);
  const copy = f32.slice();
  return copy.buffer;
}

class Rpc {
  constructor(worker) {
    this._w = worker;
    this._id = 0;
    this._pending = new Map();
    worker.onmessage = (e) => {
      const { id, ok, data, error } = e.data;
      const p = this._pending.get(id);
      if (!p) return;
      this._pending.delete(id);
      ok ? p.resolve(data) : p.reject(new Error(error));
    };
    worker.onerror = (e) => {
      const err = new Error('worker error: ' + (e.message || e));
      for (const p of this._pending.values()) p.reject(err);
      this._pending.clear();
    };
  }
  call(type, payload, transfer) {
    const id = ++this._id;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this._w.postMessage({ id, type, ...payload }, transfer || []);
    });
  }
}

export class ParakeetThreaded {
  constructor(worker, rpc, info) {
    this._w = worker;
    this._rpc = rpc;
    this.abi = info.abi;
    this.threads = info.threads;
  }

  // base: directory containing parakeet-worker.js (and dist-threaded/). model: a
  // URL string (fetched in the worker) or an ArrayBuffer/Uint8Array. opts.threads
  // caps the pool (defaults to the build's pool size); opts.workerUrl overrides
  // the worker script location.
  static async load(base = './', model, opts = {}) {
    let b = base.endsWith('/') ? base : base + '/';
    const workerUrl = opts.workerUrl || new URL(b + 'parakeet-worker.js', document.baseURI).href;
    const worker = new Worker(workerUrl, { type: 'module' });
    const rpc = new Rpc(worker);

    // Never request more than the pre-spawned pool (see MAX_THREADS) — doing so
    // deadlocks. Default to the machine's core count, capped to the pool.
    const want = opts.threads || navigator.hardwareConcurrency || 4;
    const threads = Math.max(1, Math.min(want, MAX_THREADS));
    let payload, transfer;
    if (typeof model === 'string') {
      payload = { modelUrl: new URL(model, document.baseURI).href, threads };
    } else {
      const buf = model instanceof Uint8Array ? model.buffer : model;
      payload = { modelBytes: buf, threads };
      transfer = [buf];
    }
    const info = await rpc.call('load', payload, transfer);
    return new ParakeetThreaded(worker, rpc, info);
  }

  async transcribe(pcm, sampleRate = 16000, opts = {}) {
    const buf = toTransfer(pcm);
    return this._rpc.call('transcribe', {
      pcm: buf, sampleRate, decoder: opts.decoder ?? Decoder.DEFAULT,
      lang: opts.lang ?? '', json: false,
    }, [buf]);
  }

  async transcribeWithTimestamps(pcm, sampleRate = 16000, opts = {}) {
    const buf = toTransfer(pcm);
    return this._rpc.call('transcribe', {
      pcm: buf, sampleRate, decoder: opts.decoder ?? Decoder.DEFAULT,
      lang: opts.lang ?? '', json: true,
    }, [buf]);
  }

  // Begin a live streaming session (requires a streaming model). Returns a
  // ParakeetThreadedStream whose feed()/finalize() are async.
  async stream(opts = {}) {
    const { sid } = await this._rpc.call('stream-begin', { lang: opts.lang ?? '' });
    return new ParakeetThreadedStream(this._rpc, sid);
  }

  // Change the ggml thread count live (no model reload). Capped to the pool
  // (MAX_THREADS) — asking for more than the pre-spawned pool deadlocks. Note
  // that using more threads than the machine's PHYSICAL cores usually makes it
  // slower (ggml spin-waits), so more is not always better.
  async setThreads(n) {
    const t = Math.max(1, Math.min(n | 0, MAX_THREADS));
    const r = await this._rpc.call('set-threads', { threads: t });
    this.threads = r.threads;
    return this.threads;
  }

  // Tear down the worker (and its pthread pool).
  terminate() {
    this._w.terminate();
  }
}

export class ParakeetThreadedStream {
  constructor(rpc, sid) { this._rpc = rpc; this._sid = sid; this._freed = false; }

  // Feed 16 kHz mono float PCM; resolves to
  //   { text, eou, eob, frame_sec, events:[{type,frame,t}], words:[...] }
  async feed(pcm16k) {
    if (this._freed) throw new Error('stream freed');
    const buf = toTransfer(pcm16k);
    return this._rpc.call('stream-feed', { sid: this._sid, pcm: buf }, [buf]);
  }
  async finalize() {
    if (this._freed) throw new Error('stream freed');
    return this._rpc.call('stream-finalize', { sid: this._sid });
  }
  async free() {
    if (this._freed) return;
    this._freed = true;
    await this._rpc.call('stream-free', { sid: this._sid });
  }
}

export default ParakeetThreaded;
