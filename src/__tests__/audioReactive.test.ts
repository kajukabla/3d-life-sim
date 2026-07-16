import { describe, expect, it } from "vitest";
import { maxAudioBands, parseAudioAnalysisFrame } from "../audioReactive";

describe("audio analysis frame parsing", () => {
  it("normalizes levels and rejects invalid frame versions", () => {
    const parsed = parseAudioAnalysisFrame({
      version: 1,
      sequence: 7,
      timestampSec: 1.25,
      sampleRate: 48_000,
      rms: -2,
      peak: 3,
      bands: { low: { value: 0.4 } }
    });

    expect(parsed).toMatchObject({ version: 1, sequence: 7, rms: 0, peak: 1 });
    expect(parsed?.bands.low.value).toBe(0.4);
    expect(parseAudioAnalysisFrame({ ...parsed, version: 2 })).toBeNull();
  });

  it("bounds and safely stores externally supplied band maps", () => {
    const parsedFrame = parseAudioAnalysisFrame({
      version: 1,
      sequence: 1,
      timestampSec: 1,
      sampleRate: 48_000,
      rms: 0.5,
      peak: 0.5,
      bands: Object.fromEntries(Array.from({ length: maxAudioBands + 20 }, (_, index) => [
        `band-${index}`,
        { value: 0.5 }
      ]))
    });
    expect(Object.keys(parsedFrame?.bands ?? {})).toHaveLength(maxAudioBands);

    const prototypeKeyFrame = parseAudioAnalysisFrame(JSON.parse(`{
      "version": 1,
      "sequence": 1,
      "timestampSec": 1,
      "sampleRate": 48000,
      "rms": 0.5,
      "peak": 0.5,
      "bands": { "__proto__": { "value": 0.75 } }
    }`));
    expect(Object.getPrototypeOf(prototypeKeyFrame?.bands)).toBe(null);
    expect(prototypeKeyFrame?.bands["__proto__"]?.value).toBe(0.75);
  });
});
