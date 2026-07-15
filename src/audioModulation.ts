import type { AudioAnalysisFrame } from "./audioReactive";
import {
  defaultSliderMidiMapping,
  sanitizeSliderMidiMapping,
  type SliderMidiMapping
} from "./midiMapping";

export const audioBuckets = ["low", "mid", "high"] as const;
export type AudioBucket = (typeof audioBuckets)[number];

export type AudioBucketControls = {
  gain: number;
  exponent: number;
  attackMs: number;
  decayMs: number;
};

export type AudioPanelState = {
  enabled: boolean;
  buckets: Record<AudioBucket, AudioBucketControls>;
};

export type SliderBucketModulation = {
  enabled: boolean;
  gain: number;
};

export type SliderModulationConfig = {
  min: number;
  max: number;
  buckets: Record<AudioBucket, SliderBucketModulation>;
  midi?: SliderMidiMapping;
};

export type SliderModulationSettings = Record<string, SliderModulationConfig>;

export type AudioModulationRuntime = {
  meters: Record<AudioBucket, number>;
  bucketValues: Record<string, Partial<Record<AudioBucket, number>>>;
};

export const defaultAudioPanelState: AudioPanelState = {
  enabled: true,
  buckets: {
    low: { gain: 1, exponent: 1, attackMs: 1, decayMs: 1 },
    mid: { gain: 1, exponent: 1, attackMs: 1, decayMs: 1 },
    high: { gain: 1, exponent: 1, attackMs: 1, decayMs: 1 }
  }
};

export function createAudioModulationRuntime(): AudioModulationRuntime {
  return {
    meters: { low: 0, mid: 0, high: 0 },
    bucketValues: {}
  };
}

export function defaultSliderModulation(min: number, max: number): SliderModulationConfig {
  return {
    min,
    max,
    buckets: {
      low: { enabled: false, gain: 1 },
      mid: { enabled: false, gain: 1 },
      high: { enabled: false, gain: 1 }
    },
    midi: defaultSliderMidiMapping(min, max)
  };
}

export function getSliderModulation(
  settings: SliderModulationSettings,
  target: string,
  min: number,
  max: number
): SliderModulationConfig {
  return settings[target] ?? defaultSliderModulation(min, max);
}

export function hasEnabledSliderModulation(config: SliderModulationConfig): boolean {
  return audioBuckets.some((bucket) => config.buckets[bucket].enabled);
}

// The visible slider range is just a default. Once a slider is driven by audio activity or a
// MIDI mapping, those mappings define their own output range and are allowed to disregard the
// slider's pre-baked min/max entirely. This widens the effective range to cover whatever the
// active mappings ask for (plus the current value) so the slider track and any clamping reflect
// the real reachable range rather than fighting the mapping.
export function effectiveSliderRange(
  baseMin: number,
  baseMax: number,
  value: number,
  config: SliderModulationConfig | null | undefined
): { min: number; max: number } {
  const candidates = [baseMin, baseMax];
  if (Number.isFinite(value)) candidates.push(value);
  if (config) {
    const midi = config.midi;
    if (midi?.control) {
      if (Number.isFinite(midi.min)) candidates.push(midi.min);
      if (Number.isFinite(midi.max)) candidates.push(midi.max);
    }
    if (hasEnabledSliderModulation(config)) {
      if (Number.isFinite(config.min)) candidates.push(config.min);
      if (Number.isFinite(config.max)) candidates.push(config.max);
    }
  }
  return { min: Math.min(...candidates), max: Math.max(...candidates) };
}

export function audioBucketSignals(frame: AudioAnalysisFrame | null): Record<AudioBucket, number> {
  if (!frame) return { low: 0, mid: 0, high: 0 };
  return {
    low: Math.max(signal(frame, "low"), signal(frame, "sub"), signal(frame, "bass")),
    mid: Math.max(signal(frame, "mid"), signal(frame, "lowMid")),
    high: Math.max(signal(frame, "high"), signal(frame, "presence"))
  };
}

export function updateAudioMeters(
  runtime: AudioModulationRuntime,
  frame: AudioAnalysisFrame | null,
  panel: AudioPanelState,
  dtSec: number
): Record<AudioBucket, number> {
  const signals = audioBucketSignals(frame);
  const next = { ...runtime.meters };
  for (const bucket of audioBuckets) {
    const controls = panel.buckets[bucket];
    const target = unit(shapeUnit(signals[bucket], controls.exponent) * controls.gain);
    next[bucket] = smoothUnit(runtime.meters[bucket], target, controls.attackMs, controls.decayMs, dtSec);
  }
  runtime.meters = next;
  return next;
}

export function applyAudioModulation(
  runtime: AudioModulationRuntime,
  frame: AudioAnalysisFrame | null,
  panel: AudioPanelState,
  settings: SliderModulationSettings,
  baseValues: Record<string, number>,
  dtSec: number
): Record<string, number> {
  if (!panel.enabled || !frame) return {};
  const signals = audioBucketSignals(frame);
  const patches: Record<string, number> = {};
  for (const [target, config] of Object.entries(settings)) {
    if (!hasEnabledSliderModulation(config)) continue;
    const targetBucketValues = runtime.bucketValues[target] ?? {};
    let activity = 0;
    for (const bucket of audioBuckets) {
      const bucketConfig = config.buckets[bucket];
      if (!bucketConfig.enabled) continue;
      const controls = panel.buckets[bucket];
      const previous = targetBucketValues[bucket] ?? 0;
      const targetValue = unit(shapeUnit(signals[bucket], controls.exponent) * controls.gain * bucketConfig.gain);
      const smoothed = smoothUnit(
        previous,
        targetValue,
        controls.attackMs,
        controls.decayMs,
        dtSec
      );
      targetBucketValues[bucket] = smoothed;
      activity = Math.max(activity, smoothed);
    }
    runtime.bucketValues[target] = targetBucketValues;
    patches[target] = lerp(config.min, config.max, activity, baseValues[target]);
  }
  return patches;
}

export function sanitizeAudioPanelState(value: unknown): AudioPanelState {
  const source = isRecord(value) ? value : {};
  const bucketsSource = isRecord(source.buckets) ? source.buckets : {};
  return {
    enabled: typeof source.enabled === "boolean" ? source.enabled : defaultAudioPanelState.enabled,
    buckets: {
      low: sanitizeBucketControls(bucketsSource.low, defaultAudioPanelState.buckets.low),
      mid: sanitizeBucketControls(bucketsSource.mid, defaultAudioPanelState.buckets.mid),
      high: sanitizeBucketControls(bucketsSource.high, defaultAudioPanelState.buckets.high)
    }
  };
}

export function sanitizeSliderModulationSettings(value: unknown): SliderModulationSettings {
  if (!isRecord(value)) return {};
  const settings: SliderModulationSettings = {};
  for (const [target, config] of Object.entries(value)) {
    if (!target || !isRecord(config)) continue;
    settings[target] = sanitizeSliderModulationConfig(config);
  }
  return settings;
}

export function sanitizeSliderModulationConfig(value: unknown): SliderModulationConfig {
  const source = isRecord(value) ? value : {};
  const fallback = defaultSliderModulation(0, 1);
  const bucketsSource = isRecord(source.buckets) ? source.buckets : {};
  const min = finiteNumber(source.min, fallback.min);
  const max = finiteNumber(source.max, fallback.max);
  return {
    min,
    max,
    buckets: {
      low: sanitizeSliderBucket(bucketsSource.low),
      mid: sanitizeSliderBucket(bucketsSource.mid),
      high: sanitizeSliderBucket(bucketsSource.high)
    },
    midi: sanitizeSliderMidiMapping(source.midi, min, max)
  };
}

function sanitizeBucketControls(value: unknown, fallback: AudioBucketControls): AudioBucketControls {
  const source = isRecord(value) ? value : {};
  return {
    gain: clamp(finiteNumber(source.gain, fallback.gain), 0, 8),
    exponent: clamp(finiteNumber(source.exponent, fallback.exponent), 0.1, 8),
    attackMs: clamp(
      finiteNumber(source.attackMs, finiteNumber(source.smoothingMs, fallback.attackMs)),
      1,
      2000
    ),
    decayMs: clamp(
      finiteNumber(source.decayMs, finiteNumber(source.smoothingMs, fallback.decayMs)),
      1,
      2000
    )
  };
}

function sanitizeSliderBucket(value: unknown): SliderBucketModulation {
  const source = isRecord(value) ? value : {};
  return {
    enabled: typeof source.enabled === "boolean" ? source.enabled : false,
    gain: clamp(finiteNumber(source.gain, finiteNumber(source.smoothing, 1)), 0, 16)
  };
}

function signal(frame: AudioAnalysisFrame, key: string): number {
  return unit(frame.bands[key]?.value ?? 0);
}

// Power-curve shaper on the normalized 0..1 band signal. exponent === 1 is linear (pass-through);
// exponent > 1 pushes mid-level values toward 0 so only loud peaks survive (spikier); exponent < 1
// lifts quiet signal (more sensitive). pow of a unit value by a positive exponent stays in 0..1.
function shapeUnit(value: number, exponent: number): number {
  const v = unit(value);
  if (!Number.isFinite(exponent) || exponent === 1 || v <= 0) return v;
  return Math.pow(v, exponent);
}

function smoothUnit(previous: number, target: number, attackMs: number, decayMs: number, dtSec: number): number {
  const smoothing = target >= previous ? attackMs : decayMs;
  if (!Number.isFinite(dtSec) || dtSec <= 0 || !Number.isFinite(smoothing) || smoothing <= 1) {
    return unit(target);
  }
  const alpha = 1 - Math.exp(-dtSec / (smoothing / 1000));
  return unit(previous + (target - previous) * unit(alpha));
}

function lerp(min: number, max: number, value: number, fallback: number | undefined): number {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return fallback ?? 0;
  return min + (max - min) * unit(value);
}

function unit(value: number): number {
  return Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function finiteNumber(value: unknown, fallback: number): number {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : fallback;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
