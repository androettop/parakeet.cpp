// Node smoke test for the parakeet.cpp WASM build.
//
//   node examples/wasm/test_node.mjs <model.gguf> <audio.(wav|mp3|...)>
//
// Loads the WASM module + model, decodes the audio to 16 kHz mono PCM, and
// prints the transcript and the per-word timestamp document. Audio decoding
// here uses a tiny built-in WAV reader (node has no Web Audio API); the browser
// path uses Parakeet.decodeAudio instead. Exercised in CI-style verification.

import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { Parakeet, Decoder } from './parakeet.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Minimal WAV → mono Float32Array reader (PCM 16/32-bit int, 32-bit float).
function decodeWav(buf) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  if (dv.getUint32(0, false) !== 0x52494646 /*RIFF*/ ||
      dv.getUint32(8, false) !== 0x57415645 /*WAVE*/) {
    throw new Error('not a RIFF/WAVE file');
  }
  let off = 12, fmt = null, dataOff = -1, dataLen = 0;
  while (off + 8 <= dv.byteLength) {
    const id = dv.getUint32(off, false);
    const sz = dv.getUint32(off + 4, true);
    const body = off + 8;
    if (id === 0x666d7420 /*fmt */) {
      fmt = {
        audioFormat: dv.getUint16(body, true),
        channels: dv.getUint16(body + 2, true),
        sampleRate: dv.getUint32(body + 4, true),
        bitsPerSample: dv.getUint16(body + 14, true),
      };
    } else if (id === 0x64617461 /*data*/) {
      dataOff = body; dataLen = sz;
    }
    off = body + sz + (sz & 1);
  }
  if (!fmt || dataOff < 0) throw new Error('missing fmt/data chunk');
  const { channels, bitsPerSample, sampleRate, audioFormat } = fmt;
  const frames = Math.floor(dataLen / (channels * (bitsPerSample >> 3)));
  const mono = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    let acc = 0;
    for (let c = 0; c < channels; c++) {
      const p = dataOff + (i * channels + c) * (bitsPerSample >> 3);
      let s;
      if (audioFormat === 3 && bitsPerSample === 32) s = dv.getFloat32(p, true);
      else if (bitsPerSample === 16) s = dv.getInt16(p, true) / 32768;
      else if (bitsPerSample === 32) s = dv.getInt32(p, true) / 2147483648;
      else if (bitsPerSample === 8) s = (dv.getUint8(p) - 128) / 128;
      else throw new Error('unsupported bitsPerSample ' + bitsPerSample);
      acc += s;
    }
    mono[i] = acc / channels;
  }
  return { pcm: mono, sampleRate };
}

async function main() {
  const [modelPath, audioPath] = process.argv.slice(2);
  if (!modelPath || !audioPath) {
    console.error('usage: node test_node.mjs <model.gguf> <audio.wav>');
    process.exit(2);
  }

  console.log('loading module + model:', modelPath);
  const t0 = Date.now();
  const pk = await Parakeet.load(__dirname + '/dist/', modelPath, {
    printErr: (s) => process.stderr.write(s + '\n'),
  });
  console.log('loaded in', Date.now() - t0, 'ms; ABI', pk.abiVersion());

  const wav = decodeWav(await readFile(audioPath));
  console.log('audio:', wav.pcm.length, 'samples @', wav.sampleRate, 'Hz',
              '(' + (wav.pcm.length / wav.sampleRate).toFixed(2) + 's)');

  const t1 = Date.now();
  const text = pk.transcribe(wav.pcm, wav.sampleRate);
  const dt = Date.now() - t1;
  console.log('\n=== transcript (' + dt + ' ms) ===');
  console.log(text);

  const doc = pk.transcribeWithTimestamps(wav.pcm, wav.sampleRate);
  console.log('\n=== first words with timestamps ===');
  for (const w of (doc.words || []).slice(0, 8)) {
    console.log(`  ${w.start.toFixed(2)}-${w.end.toFixed(2)}  ${w.w}  (${w.conf.toFixed(2)})`);
  }

  pk.free();
  if (!text || !text.trim()) { console.error('\nFAIL: empty transcript'); process.exit(1); }
  console.log('\nOK');
}

main().catch((e) => { console.error(e); process.exit(1); });
