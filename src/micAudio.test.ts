import { describe, expect, it } from "vitest";
import { MicFilterbank, dbToUnit, defaultMicAnalysisOptions, linearToDb } from "./micAudio";

const SAMPLE_RATE = 48_000;
const WINDOW = 2048;

function sine(freqHz: number, amplitude: number, n = WINDOW, sampleRate = SAMPLE_RATE): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = amplitude * Math.sin((2 * Math.PI * freqHz * i) / sampleRate);
  return out;
}

// Settle the one-pole filter state on a couple of windows before asserting, the
// way the live mic stream would have been running continuously. Windows must be
// phase-continuous slices of one signal — restarting the sine at each window
// boundary would inject a broadband click the peak detector picks up.
function settledFrame(freqHz: number, amplitude: number): ReturnType<MicFilterbank["analyzeWindow"]> {
  const bank = new MicFilterbank(SAMPLE_RATE, defaultMicAnalysisOptions);
  const signal = sine(freqHz, amplitude, WINDOW * 3);
  let frame = bank.analyzeWindow(signal.subarray(0, WINDOW), 1, 0);
  for (let i = 2; i <= 3; i++) {
    frame = bank.analyzeWindow(signal.subarray((i - 1) * WINDOW, i * WINDOW), i, (i - 1) * (WINDOW / SAMPLE_RATE));
  }
  return frame;
}

describe("dbToUnit / linearToDb (parity with the Rust analyzer)", () => {
  it("maps floor→0, ceiling→1, midpoint→0.5 and clamps outside", () => {
    expect(dbToUnit(-72, -72, -12)).toBe(0);
    expect(dbToUnit(-12, -72, -12)).toBe(1);
    expect(dbToUnit(-42, -72, -12)).toBeCloseTo(0.5, 5);
    expect(dbToUnit(-100, -72, -12)).toBe(0);
    expect(dbToUnit(0, -72, -12)).toBe(1);
  });

  it("linearToDb clamps to the Rust MIN_DB_MAGNITUDE floor", () => {
    expect(linearToDb(1)).toBeCloseTo(0, 5);
    expect(linearToDb(0.1)).toBeCloseTo(-20, 4);
    expect(linearToDb(0)).toBeCloseTo(-180, 1);
  });
});

// The one-pole filterbank (mirroring crates/audio_analysis Filterbank mode) is gentle —
// 6 dB/oct skirts leak across bands by design, and the presets' bucket gain/exponent
// shaping compensates. What must hold is the ORDERING: the band containing the tone
// reads strictly hotter than its neighbors, in raw dB terms.

describe("MicFilterbank — band semantics", () => {
  it("a 100 Hz tone reads hottest in low, then mid, then high", () => {
    const frame = settledFrame(100, 0.5);
    expect(frame.bands.low.rawDb!).toBeGreaterThan(frame.bands.mid.rawDb! + 3);
    expect(frame.bands.mid.rawDb!).toBeGreaterThan(frame.bands.high.rawDb! + 3);
  });

  it("semantic delta: moving the same tone to 8 kHz flips the ordering", () => {
    const low = settledFrame(100, 0.5);
    const high = settledFrame(8000, 0.5);
    expect(low.bands.low.rawDb!).toBeGreaterThan(low.bands.high.rawDb!);
    expect(high.bands.high.rawDb!).toBeGreaterThan(high.bands.low.rawDb!);
    expect(high.bands.high.rawDb!).toBeGreaterThan(low.bands.high.rawDb! + 6);
    expect(low.bands.low.rawDb!).toBeGreaterThan(high.bands.low.rawDb! + 6);
  });

  it("a 700 Hz tone reads hottest in mid", () => {
    const frame = settledFrame(700, 0.5);
    expect(frame.bands.mid.rawDb!).toBeGreaterThan(frame.bands.low.rawDb! + 3);
    expect(frame.bands.mid.rawDb!).toBeGreaterThan(frame.bands.high.rawDb! + 3);
  });

  it("silence produces a fully dark frame", () => {
    const bank = new MicFilterbank(SAMPLE_RATE, defaultMicAnalysisOptions);
    const frame = bank.analyzeWindow(new Float32Array(WINDOW), 1, 0);
    expect(frame.rms).toBe(0);
    expect(frame.peak).toBe(0);
    expect(frame.bands.low.value).toBe(0);
    expect(frame.bands.mid.value).toBe(0);
    expect(frame.bands.high.value).toBe(0);
  });

  it("louder input raises the band value (level actually matters)", () => {
    const quiet = settledFrame(700, 0.01);
    const loud = settledFrame(700, 0.5);
    expect(loud.bands.mid.value).toBeGreaterThan(quiet.bands.mid.value + 0.2);
  });

  it("accumulators reset between windows — a loud window does not haunt a silent one", () => {
    const bank = new MicFilterbank(SAMPLE_RATE, defaultMicAnalysisOptions);
    bank.analyzeWindow(sine(700, 0.5), 1, 0);
    // A couple of silent windows to drain the filter state.
    bank.analyzeWindow(new Float32Array(WINDOW), 2, 0.04);
    const silent = bank.analyzeWindow(new Float32Array(WINDOW), 3, 0.08);
    expect(silent.bands.mid.value).toBeLessThan(0.2);
    expect(silent.rms).toBeLessThan(0.05);
  });
});

describe("MicFilterbank — frame schema and levels", () => {
  it("emits the Rust AudioAnalysisFrame shape", () => {
    const bank = new MicFilterbank(SAMPLE_RATE, defaultMicAnalysisOptions);
    const frame = bank.analyzeWindow(sine(1000, 0.1), 7, 1.25);
    expect(frame.version).toBe(1);
    expect(frame.sequence).toBe(7);
    expect(frame.timestampSec).toBe(1.25);
    expect(frame.sampleRate).toBe(SAMPLE_RATE);
    for (const name of ["low", "mid", "high"]) {
      const band = frame.bands[name];
      expect(band).toBeDefined();
      expect(typeof band.value).toBe("number");
      expect(typeof band.rawDb).toBe("number");
      expect(typeof band.rawMagnitude).toBe("number");
    }
  });

  it("rms/peak match the analytic values for a sine", () => {
    const amplitude = 0.1;
    const bank = new MicFilterbank(SAMPLE_RATE, defaultMicAnalysisOptions);
    const frame = bank.analyzeWindow(sine(1000, amplitude), 1, 0);
    const rmsDb = 20 * Math.log10(amplitude / Math.SQRT2);
    const peakDb = 20 * Math.log10(amplitude);
    expect(frame.rmsDb).toBeCloseTo(rmsDb, 1);
    expect(frame.peakDb).toBeCloseTo(peakDb, 1);
    expect(frame.rms).toBeCloseTo(dbToUnit(rmsDb, -72, -12), 2);
    expect(frame.peak).toBeCloseTo(dbToUnit(peakDb, -72, -6), 2);
  });
});
