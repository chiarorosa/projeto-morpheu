// Decode any browser-supported audio (mp3, m4a, webm from the recorder, ...) and re-encode
// it as 16-bit mono WAV, so the backend only ever has to read WAV.

export async function toWav(input: Blob): Promise<{ wav: Blob; seconds: number }> {
  const ctx = new AudioContext();
  try {
    const buffer = await ctx.decodeAudioData(await input.arrayBuffer());
    const mono = mixDown(buffer);
    return { wav: encodeWav(mono, buffer.sampleRate), seconds: buffer.duration };
  } finally {
    void ctx.close();
  }
}

function mixDown(buffer: AudioBuffer): Float32Array {
  const out = new Float32Array(buffer.length);
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const data = buffer.getChannelData(c);
    for (let i = 0; i < data.length; i++) out[i] += data[i] / buffer.numberOfChannels;
  }
  return out;
}

function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const view = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const str = (off: number, s: string) => [...s].forEach((ch, i) => view.setUint8(off + i, ch.charCodeAt(0)));
  str(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  str(8, "WAVE");
  str(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  str(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([view], { type: "audio/wav" });
}
