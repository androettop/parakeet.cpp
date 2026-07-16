// parakeet.cpp Web Worker — hosts the MULTI-THREADED WASM module off the main
// thread. This is what keeps the browser UI responsive: ggml's compute blocks
// the thread it runs on while its pthread pool works, and the browser main
// thread is forbidden from blocking (Atomics.wait throws there). Running the
// module inside this worker means the blocking happens here, not on the UI
// thread, and the pthread pool actually uses multiple CPU cores.
//
// Loaded as a module worker (new Worker(url, { type: 'module' })) by
// parakeet-threaded.js, which speaks the request/response protocol below.

import createParakeetModule from './dist-threaded/parakeet.mjs';

let m = null;         // the Emscripten module
let ctx = 0;          // parakeet_ctx*
const streams = {};   // sid -> parakeet_stream*
let nextSid = 1;

const reply = (id, data, transfer) =>
  postMessage({ id, ok: true, data }, transfer || []);
const fail = (id, err) =>
  postMessage({ id, ok: false, error: String(err && err.message ? err.message : err) });

function lastError() {
  const p = m.ccall('parakeet_capi_last_error', 'number', ['number'], [ctx]);
  return p ? m.UTF8ToString(p) : '';
}

// Copy a Float32Array (from a transferred ArrayBuffer) into the wasm heap.
function pushPcm(pcm) {
  const ptr = m._malloc((pcm.length * 4) || 4);
  m.HEAPF32.set(pcm, ptr >>> 2);
  return ptr;
}
function takeString(ptr) {
  if (!ptr) return null;
  const s = m.UTF8ToString(ptr);
  m.ccall('parakeet_capi_free_string', null, ['number'], [ptr]);
  return s;
}

self.onmessage = async (e) => {
  const { id, type } = e.data;
  try {
    switch (type) {
      case 'load': {
        m = await createParakeetModule({ printErr: (s) => console.warn('[parakeet]', s) });
        let bytes;
        if (e.data.modelBytes) {
          bytes = new Uint8Array(e.data.modelBytes);
        } else {
          const r = await fetch(e.data.modelUrl);
          if (!r.ok) throw new Error(`fetch model ${e.data.modelUrl}: ${r.status}`);
          bytes = new Uint8Array(await r.arrayBuffer());
        }
        m.FS.writeFile('/model.gguf', bytes);
        ctx = m.ccall('parakeet_capi_load', 'number', ['string'], ['/model.gguf']);
        try { m.FS.unlink('/model.gguf'); } catch (_) {}
        if (!ctx) throw new Error('failed to load model');
        // Pin ggml to the pre-spawned pool so it never spawns a worker on demand
        // (which would deadlock this blocked worker thread).
        const threads = e.data.threads || 4;
        m.ccall('pk_wasm_set_threads', null, ['number'], [threads]);
        reply(id, { abi: m.ccall('parakeet_capi_abi_version', 'number', [], []), threads });
        break;
      }
      case 'set-threads': {
        const n = Math.max(1, e.data.threads | 0);
        m.ccall('pk_wasm_set_threads', null, ['number'], [n]);
        reply(id, { threads: n });
        break;
      }
      case 'transcribe': {
        const pcm = new Float32Array(e.data.pcm);
        const ptr = pushPcm(pcm);
        try {
          let out;
          if (e.data.json) {
            out = m.ccall('pk_wasm_transcribe_pcm_json', 'number',
              ['number', 'number', 'number', 'number', 'number', 'string'],
              [ctx, ptr, pcm.length, e.data.sampleRate, e.data.decoder | 0, e.data.lang || '']);
            const s = takeString(out);
            if (s === null) throw new Error('transcribe failed: ' + lastError());
            reply(id, JSON.parse(s));
          } else {
            out = m.ccall('parakeet_capi_transcribe_pcm_lang', 'number',
              ['number', 'number', 'number', 'number', 'number', 'string'],
              [ctx, ptr, pcm.length, e.data.sampleRate, e.data.decoder | 0, e.data.lang || '']);
            const s = takeString(out);
            if (s === null) throw new Error('transcribe failed: ' + lastError());
            reply(id, s);
          }
        } finally { m._free(ptr); }
        break;
      }
      case 'stream-begin': {
        const s = m.ccall('parakeet_capi_stream_begin_lang', 'number',
          ['number', 'string'], [ctx, e.data.lang || '']);
        if (!s) throw new Error('could not begin streaming (not a streaming model?): ' + lastError());
        const sid = nextSid++;
        streams[sid] = s;
        reply(id, { sid });
        break;
      }
      case 'stream-feed': {
        const s = streams[e.data.sid];
        if (!s) throw new Error('unknown stream ' + e.data.sid);
        const pcm = new Float32Array(e.data.pcm);
        const ptr = pushPcm(pcm);
        try {
          const out = m.ccall('parakeet_capi_stream_feed_json', 'number',
            ['number', 'number', 'number'], [s, ptr, pcm.length]);
          const str = takeString(out);
          if (str === null) throw new Error('stream feed failed');
          reply(id, JSON.parse(str));
        } finally { m._free(ptr); }
        break;
      }
      case 'stream-finalize': {
        const s = streams[e.data.sid];
        if (!s) throw new Error('unknown stream ' + e.data.sid);
        const out = m.ccall('parakeet_capi_stream_finalize_json', 'number', ['number'], [s]);
        const str = takeString(out);
        if (str === null) throw new Error('stream finalize failed');
        reply(id, JSON.parse(str));
        break;
      }
      case 'stream-free': {
        const s = streams[e.data.sid];
        if (s) { m.ccall('parakeet_capi_stream_free', null, ['number'], [s]); delete streams[e.data.sid]; }
        reply(id, {});
        break;
      }
      default:
        throw new Error('unknown message type ' + type);
    }
  } catch (err) {
    fail(id, err);
  }
};
