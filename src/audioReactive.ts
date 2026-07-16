export type AudioBandFrame = {
  value: number;
  rawDb?: number;
  rawMagnitude?: number;
};

export type AudioAnalysisFrame = {
  version: 1;
  sequence: number;
  timestampSec: number;
  sampleRate: number;
  rms: number;
  peak: number;
  rmsDb?: number;
  peakDb?: number;
  bands: Record<string, AudioBandFrame>;
};

export const maxAudioBands = 64;
const maxAudioLabelChars = 256;

export function parseAudioAnalysisFrame(value: unknown): AudioAnalysisFrame | null {
  if (!isRecord(value)) return null;
  const version = finiteNumber(value.version, NaN);
  const sequence = finiteNumber(value.sequence, NaN);
  const timestampSec = finiteNumber(value.timestampSec, NaN);
  const sampleRate = finiteNumber(value.sampleRate, NaN);
  const rms = unit(finiteNumber(value.rms, 0));
  const peak = unit(finiteNumber(value.peak, 0));
  if (version !== 1 || !Number.isFinite(sequence) || !Number.isFinite(timestampSec) || !Number.isFinite(sampleRate)) {
    return null;
  }
  const bands = Object.create(null) as Record<string, AudioBandFrame>;
  if (isRecord(value.bands)) {
    let bandCount = 0;
    for (const name of Object.keys(value.bands)) {
      if (bandCount >= maxAudioBands) break;
      const bandValue = value.bands[name];
      if (!isRecord(bandValue)) continue;
      const boundedName = boundedText(name, maxAudioLabelChars);
      if (!boundedName) continue;
      bands[boundedName] = {
        value: unit(finiteNumber(bandValue.value, 0)),
        rawDb: optionalFiniteNumber(bandValue.rawDb),
        rawMagnitude: optionalFiniteNumber(bandValue.rawMagnitude)
      };
      bandCount += 1;
    }
  }
  return {
    version: 1,
    sequence,
    timestampSec,
    sampleRate,
    rms,
    peak,
    rmsDb: optionalFiniteNumber(value.rmsDb),
    peakDb: optionalFiniteNumber(value.peakDb),
    bands
  };
}

function unit(value: number): number {
  return Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0;
}

function finiteNumber(value: unknown, fallback: number): number {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : fallback;
}

function optionalFiniteNumber(value: unknown): number | undefined {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : undefined;
}

function boundedText(value: string, maxChars: number): string {
  return Array.from(value)
    .slice(0, maxChars)
    .join("")
    .replace(/[\u0000-\u001f\u007f-\u009f\u202a-\u202e\u2066-\u2069]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
