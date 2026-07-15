import type { AudioAnalysisFrame } from "./audioReactive";

// In-browser microphone analyzer for the website embed: replicates the native Rust
// backend's default Filterbank mode (crates/audio_analysis) so the same presets drive
// the same slider modulation without the helper download. Per band: one-pole
// highpass (input minus low-passed input) into a one-pole lowpass, magnitude =
// max(peak, rms) of the filtered window, mapped through the same dB floor/ceiling.

const MIN_DB_MAGNITUDE = 1e-9;

export type MicBandConfig = { name: string; minHz: number; maxHz: number };

export type MicAnalysisOptions = {
  floorDb: number;
  ceilingDb: number;
  peakFloorDb: number;
  peakCeilingDb: number;
  bands: MicBandConfig[];
};

export const defaultMicAnalysisOptions: MicAnalysisOptions = {
  floorDb: -72,
  ceilingDb: -12,
  peakFloorDb: -72,
  peakCeilingDb: -6,
  bands: [
    { name: "low", minHz: 25, maxHz: 180 },
    { name: "mid", minHz: 180, maxHz: 2000 },
    { name: "high", minHz: 2000, maxHz: 12000 }
  ]
};

export function linearToDb(value: number): number {
  return 20 * Math.log10(Math.max(value, MIN_DB_MAGNITUDE));
}

export function dbToUnit(db: number, floorDb: number, ceilingDb: number): number {
  if (!(Number.isFinite(db) && Number.isFinite(floorDb) && Number.isFinite(ceilingDb) && floorDb < ceilingDb)) return 0;
  const unit = (db - floorDb) / (ceilingDb - floorDb);
  return Math.min(1, Math.max(0, unit));
}

class OnePoleLowPass {
  private alpha: number;
  private value = 0;

  constructor(cutoffHz: number, sampleRate: number) {
    const nyquist = sampleRate * 0.5;
    const clamped = Math.min(Math.max(cutoffHz, 1), Math.max(nyquist * 0.98, 1));
    const alpha = 1 - Math.exp((-2 * Math.PI * clamped) / sampleRate);
    this.alpha = Math.min(Math.max(alpha, 0), 1);
  }

  process(sample: number): number {
    this.value += (sample - this.value) * this.alpha;
    return this.value;
  }
}

type BandState = {
  config: MicBandConfig;
  highpassSource: OnePoleLowPass | null;
  lowpass: OnePoleLowPass | null;
};

export class MicFilterbank {
  private readonly sampleRate: number;
  private readonly options: MicAnalysisOptions;
  private readonly bands: BandState[];

  constructor(sampleRate: number, options: MicAnalysisOptions = defaultMicAnalysisOptions) {
    this.sampleRate = sampleRate;
    this.options = options;
    const nyquist = sampleRate * 0.5;
    this.bands = options.bands.map((config) => ({
      config,
      highpassSource: config.minHz > 0 ? new OnePoleLowPass(config.minHz, sampleRate) : null,
      lowpass: config.maxHz < nyquist * 0.98 ? new OnePoleLowPass(config.maxHz, sampleRate) : null
    }));
  }

  // Analyze one contiguous window of samples. Filter state persists across calls
  // (the mic stream is continuous); per-window accumulators reset every call.
  analyzeWindow(samples: Float32Array, sequence: number, timestampSec: number): AudioAnalysisFrame {
    const count = Math.max(samples.length, 1);
    let inputSumSquares = 0;
    let inputPeak = 0;
    const sums = new Float64Array(this.bands.length);
    const peaks = new Float64Array(this.bands.length);

    for (let i = 0; i < samples.length; i++) {
      const raw = samples[i];
      const sample = Number.isFinite(raw) ? raw : 0;
      inputSumSquares += sample * sample;
      const mag = Math.abs(sample);
      if (mag > inputPeak) inputPeak = mag;
      for (let b = 0; b < this.bands.length; b++) {
        const band = this.bands[b];
        let value = sample;
        if (band.highpassSource) value -= band.highpassSource.process(sample);
        if (band.lowpass) value = band.lowpass.process(value);
        sums[b] += value * value;
        const filteredMag = Math.abs(value);
        if (filteredMag > peaks[b]) peaks[b] = filteredMag;
      }
    }

    const rmsDb = linearToDb(Math.sqrt(inputSumSquares / count));
    const peakDb = linearToDb(inputPeak);

    const bands: AudioAnalysisFrame["bands"] = {};
    for (let b = 0; b < this.bands.length; b++) {
      const magnitude = Math.max(peaks[b], Math.sqrt(sums[b] / count));
      const rawDb = linearToDb(magnitude);
      bands[this.bands[b].config.name] = {
        value: dbToUnit(rawDb, this.options.floorDb, this.options.ceilingDb),
        rawDb,
        rawMagnitude: magnitude
      };
    }

    return {
      version: 1,
      sequence,
      timestampSec,
      sampleRate: this.sampleRate,
      rms: dbToUnit(rmsDb, this.options.floorDb, this.options.ceilingDb),
      peak: dbToUnit(peakDb, this.options.peakFloorDb, this.options.peakCeilingDb),
      rmsDb,
      peakDb,
      bands
    };
  }
}

export type MicAudioStatus = "starting" | "running" | "denied" | "error";

export type MicAudioController = {
  stop: () => void;
  // The live input MediaStream (null until getUserMedia resolves, or after stop). Exposed so a
  // performance recorder can capture the exact audio that drives reactivity — see
  // performanceAudioRecorder.ts.
  getStream: () => MediaStream | null;
};

const MIC_WINDOW_SIZE = 2048;
const MIC_EMIT_INTERVAL_MS = 33;

export function startMicAudio(handlers: {
  onFrame: (frame: AudioAnalysisFrame, dtSec: number) => void;
  onStatus?: (status: MicAudioStatus, detail?: string) => void;
  options?: MicAnalysisOptions;
}): MicAudioController {
  const options = handlers.options ?? defaultMicAnalysisOptions;
  let stopped = false;
  let stream: MediaStream | null = null;
  let context: AudioContext | null = null;
  let timer: number | null = null;
  let removeResumeListeners: (() => void) | null = null;

  handlers.onStatus?.("starting");

  (async () => {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        // Music-reactive use wants the raw signal: browser voice processing would
        // squash the dynamics the band mapping feeds on.
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false }
      });
    } catch (error) {
      handlers.onStatus?.("denied", error instanceof Error ? error.message : String(error));
      return;
    }
    if (stopped) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }
    try {
      context = new AudioContext();
      const source = context.createMediaStreamSource(stream);
      const analyser = context.createAnalyser();
      analyser.fftSize = MIC_WINDOW_SIZE;
      source.connect(analyser);

      // Autoplay policy can leave a no-gesture AudioContext suspended; resume on
      // the first real interaction.
      if (context.state === "suspended") {
        void context.resume();
        const resume = () => { void context?.resume(); };
        window.addEventListener("pointerdown", resume);
        window.addEventListener("keydown", resume);
        removeResumeListeners = () => {
          window.removeEventListener("pointerdown", resume);
          window.removeEventListener("keydown", resume);
        };
      }

      const filterbank = new MicFilterbank(context.sampleRate, options);
      const buffer = new Float32Array(MIC_WINDOW_SIZE);
      let sequence = 0;
      let lastEmitMs = performance.now();
      const startedMs = lastEmitMs;
      handlers.onStatus?.("running");
      timer = window.setInterval(() => {
        if (!context || context.state !== "running") return;
        analyser.getFloatTimeDomainData(buffer);
        const nowMs = performance.now();
        const dtSec = Math.max((nowMs - lastEmitMs) / 1000, 0.001);
        lastEmitMs = nowMs;
        sequence += 1;
        handlers.onFrame(filterbank.analyzeWindow(buffer, sequence, (nowMs - startedMs) / 1000), dtSec);
      }, MIC_EMIT_INTERVAL_MS);
    } catch (error) {
      handlers.onStatus?.("error", error instanceof Error ? error.message : String(error));
    }
  })();

  return {
    stop: () => {
      stopped = true;
      if (timer !== null) window.clearInterval(timer);
      removeResumeListeners?.();
      stream?.getTracks().forEach((track) => track.stop());
      void context?.close();
      stream = null;
    },
    getStream: () => stream
  };
}
