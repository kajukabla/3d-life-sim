import { describe, expect, it } from "vitest";
import { sanitizeLightingState } from "../lighting/lightingStorage";
import { defaultLightingState } from "../lighting/types";

describe("sanitizeLightingState", () => {
  it("returns defaults for junk input", () => {
    expect(sanitizeLightingState(null)).toEqual(defaultLightingState());
    expect(sanitizeLightingState("nope")).toEqual(defaultLightingState());
    expect(sanitizeLightingState({})).toEqual(defaultLightingState());
  });

  it("clamps values into range", () => {
    const s = sanitizeLightingState({
      scrub: 5,
      scrubRateHz: 99,
      phaseSpread: -3,
      freqSpread: 100,
      master: 2,
      gamma: -1,
      strobe: { enabled: true, hz: 999, duty: 2 }
    });
    expect(s.scrub).toBe(1);
    expect(s.phaseSpread).toBe(0);
    expect(s.master).toBe(1);
    expect(s.gamma).toBeGreaterThan(0);
    expect(s.strobe.enabled).toBe(true);
    expect(s.strobe.duty).toBeLessThanOrEqual(1);
    expect(s.strobe.hz).toBeLessThanOrEqual(60);
  });

  it("round-trips a valid state", () => {
    const original = {
      scrub: 0.42,
      scrubRateHz: 0.25,
      phaseSpread: 0.5,
      freqSpread: 1.3,
      locked: false,
      loopMode: "repeat" as const,
      master: 0.8,
      gamma: 1.6,
      brightness: 0.6,
      strobe: { enabled: true, hz: 10, duty: 0.3 }
    };
    expect(sanitizeLightingState(original)).toEqual(original);
  });

  it("preserves partial fields and fills the rest", () => {
    const s = sanitizeLightingState({ scrub: 0.7 });
    expect(s.scrub).toBe(0.7);
    expect(s.master).toBe(defaultLightingState().master);
    expect(s.strobe).toEqual(defaultLightingState().strobe);
  });
});
