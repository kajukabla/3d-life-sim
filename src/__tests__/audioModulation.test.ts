import { describe, expect, it } from "vitest";
import type { AudioAnalysisFrame } from "../audioReactive";
import {
  applyAudioModulation,
  audioBucketSignals,
  createAudioModulationRuntime,
  defaultAudioPanelState,
  defaultSliderModulation,
  effectiveSliderRange,
  getSliderModulation,
  sanitizeAudioPanelState,
  sanitizeSliderModulationSettings,
  updateAudioMeters,
  type SliderModulationConfig,
  type SliderModulationSettings
} from "../audioModulation";

function frame(values: Partial<Record<string, number>>): AudioAnalysisFrame {
  return {
    version: 1,
    sequence: 1,
    timestampSec: 1,
    sampleRate: 48_000,
    rms: 0,
    peak: 0,
    bands: Object.fromEntries(Object.entries(values).flatMap(([key, value]) => (value === undefined ? [] : [[key, { value }]])))
  };
}

describe("audio modulation model", () => {
  it("derives low mid high buckets from backend bands", () => {
    expect(audioBucketSignals(frame({ sub: 0.2, bass: 0.7, lowMid: 0.1, mid: 0.4, presence: 0.3, high: 0.9 }))).toEqual({
      low: 0.7,
      mid: 0.4,
      high: 0.9
    });
    expect(audioBucketSignals(frame({ low: 0.8, mid: 0.3, high: 0.2 }))).toEqual({
      low: 0.8,
      mid: 0.3,
      high: 0.2
    });
  });

  it("treats one millisecond attack and decay as raw meter values", () => {
    const runtime = createAudioModulationRuntime();
    const panel = sanitizeAudioPanelState({
      buckets: {
        low: { gain: 2, attackMs: 1, decayMs: 1 },
        mid: { gain: 1, attackMs: 1, decayMs: 1 },
        high: { gain: 1, attackMs: 1, decayMs: 1 }
      }
    });

    const meters = updateAudioMeters(runtime, frame({ bass: 0.4, mid: 0.25, high: 0.1 }), panel, 1 / 750);

    expect(meters.low).toBeCloseTo(0.8, 3);
    expect(meters.mid).toBeCloseTo(0.25, 3);
    expect(meters.high).toBeCloseTo(0.1, 3);

    const decay = updateAudioMeters(runtime, frame({ bass: 0, mid: 0, high: 0 }), panel, 1 / 750);
    expect(decay.low).toBe(0);
    expect(decay.mid).toBe(0);
    expect(decay.high).toBe(0);
  });

  it("shapes a band with a power-curve exponent before gain", () => {
    const runtime = createAudioModulationRuntime();
    const panel = sanitizeAudioPanelState({
      buckets: {
        low: { gain: 1, exponent: 2, attackMs: 1, decayMs: 1 },
        mid: { gain: 1, exponent: 0.5, attackMs: 1, decayMs: 1 },
        high: { gain: 1, exponent: 1, attackMs: 1, decayMs: 1 }
      }
    });

    const meters = updateAudioMeters(runtime, frame({ bass: 0.5, mid: 0.25, high: 0.5 }), panel, 1 / 750);

    expect(meters.low).toBeCloseTo(0.25, 3); // 0.5^2 — peaks survive, mids pushed down (spikier)
    expect(meters.mid).toBeCloseTo(0.5, 3); // 0.25^0.5 — quiet signal lifted (more sensitive)
    expect(meters.high).toBeCloseTo(0.5, 3); // exponent 1 is linear pass-through
  });

  it("drives multiple params from the same audio band independently", () => {
    const runtime = createAudioModulationRuntime();
    const panel = sanitizeAudioPanelState({ buckets: { low: { gain: 1, attackMs: 1, decayMs: 1 } } });
    // Two distinct targets both listening to the SAME low band, with different output ranges/gains.
    const settings = sanitizeSliderModulationSettings({
      "render.bloomStrength": { min: 0, max: 1, buckets: { low: { enabled: true, gain: 1 } } },
      "live.depositMass": { min: 0, max: 10, buckets: { low: { enabled: true, gain: 0.5 } } }
    });

    const patches = applyAudioModulation(runtime, frame({ bass: 0.8 }), panel, settings, {}, 1 / 60);

    // Both params get driven from the one band — neither shadows the other.
    expect(patches["render.bloomStrength"]).toBeGreaterThan(0);
    expect(patches["live.depositMass"]).toBeGreaterThan(0);
    // ...and each respects its own range/gain (depositMass spans 0..10 at half gain).
    expect(patches["live.depositMass"]).toBeCloseTo(0.8 * 0.5 * 10, 3);
    expect(patches["render.bloomStrength"]).toBeCloseTo(0.8, 3);
  });

  it("round-trips all band controls (incl. exponent) through preset sanitization", () => {
    const saved = {
      enabled: false,
      buckets: {
        low: { gain: 1.5, exponent: 2.5, attackMs: 40, decayMs: 220 },
        mid: { gain: 0.75, exponent: 0.4, attackMs: 10, decayMs: 90 },
        high: { gain: 3, exponent: 1, attackMs: 5, decayMs: 5 }
      }
    };
    // sanitizeAudioPanelState is exactly what preset save (buildSavedAudioState) and
    // load (settingsPresetFromUnknown) route through, so this is the persisted shape.
    expect(sanitizeAudioPanelState(saved)).toEqual(saved);
  });

  it("defaults exponent to 1 and clamps it to 0.1..8", () => {
    expect(defaultAudioPanelState.buckets.low.exponent).toBe(1);
    const panel = sanitizeAudioPanelState({
      buckets: {
        low: { exponent: 100 },
        mid: { exponent: 0 },
        high: { exponent: "nope" }
      }
    });
    expect(panel.buckets.low.exponent).toBe(8);
    expect(panel.buckets.mid.exponent).toBe(0.1);
    expect(panel.buckets.high.exponent).toBe(1);
  });

  it("maps enabled buckets onto slider min and max", () => {
    const runtime = createAudioModulationRuntime();
    const settings: SliderModulationSettings = {
      "render.bloomStrength": {
        min: 0.2,
        max: 1.2,
        buckets: {
          low: { enabled: true, gain: 1 },
          mid: { enabled: false, gain: 1 },
          high: { enabled: false, gain: 1 }
        }
      }
    };

    const patches = applyAudioModulation(
      runtime,
      frame({ bass: 0.5 }),
      { ...defaultAudioPanelState, buckets: { ...defaultAudioPanelState.buckets, low: { gain: 1, exponent: 1, attackMs: 1, decayMs: 1 } } },
      settings,
      { "render.bloomStrength": 0.5 },
      1 / 60
    );

    expect(patches["render.bloomStrength"]).toBeCloseTo(0.7, 3);
  });

  it("treats one millisecond attack and decay as raw slider modulation", () => {
    const runtime = createAudioModulationRuntime();
    const settings: SliderModulationSettings = {
      "render.bloomStrength": {
        min: 0,
        max: 1,
        buckets: {
          low: { enabled: true, gain: 1 },
          mid: { enabled: false, gain: 1 },
          high: { enabled: false, gain: 1 }
        }
      }
    };
    const panel = sanitizeAudioPanelState({
      buckets: {
        low: { gain: 1, attackMs: 1, decayMs: 1 }
      }
    });

    const attack = applyAudioModulation(runtime, frame({ low: 1 }), panel, settings, {}, 1 / 750);
    const decay = applyAudioModulation(runtime, frame({ low: 0 }), panel, settings, {}, 1 / 750);

    expect(attack["render.bloomStrength"]).toBe(1);
    expect(decay["render.bloomStrength"]).toBe(0);
  });

  it("uses per-slider bucket gains as signal multipliers", () => {
    const quietRuntime = createAudioModulationRuntime();
    const loudRuntime = createAudioModulationRuntime();
    const quietSettings = sanitizeSliderModulationSettings({
      "live.depositMass": {
        min: 0,
        max: 1,
        buckets: { low: { enabled: true, gain: 0.5 } }
      }
    });
    const loudSettings = sanitizeSliderModulationSettings({
      "live.depositMass": {
        min: 0,
        max: 1,
        buckets: { low: { enabled: true, gain: 2 } }
      }
    });
    const panel = sanitizeAudioPanelState({ buckets: { low: { gain: 1, attackMs: 1, decayMs: 1 } } });

    const quiet = applyAudioModulation(quietRuntime, frame({ bass: 0.4 }), panel, quietSettings, {}, 1 / 60);
    const loud = applyAudioModulation(loudRuntime, frame({ bass: 0.4 }), panel, loudSettings, {}, 1 / 60);

    expect(loud["live.depositMass"]).toBeGreaterThan(quiet["live.depositMass"]);
  });

  it("uses attack for rising values and decay for falling values", () => {
    const runtime = createAudioModulationRuntime();
    const panel = sanitizeAudioPanelState({
      buckets: {
        low: { gain: 1, attackMs: 1, decayMs: 400 }
      }
    });

    const attack = updateAudioMeters(runtime, frame({ bass: 1 }), panel, 1 / 60);
    const decay = updateAudioMeters(runtime, frame({ bass: 0 }), panel, 1 / 60);

    expect(attack.low).toBe(1);
    expect(decay.low).toBeGreaterThan(0.9);
  });

  it("returns slider defaults without mutating saved settings", () => {
    const settings: SliderModulationSettings = {};
    const config = getSliderModulation(settings, "render.fogBrightness", 0.1, 8);

    expect(config.min).toBe(0.1);
    expect(config.max).toBe(8);
    expect(config.buckets.low.gain).toBe(1);
    expect(settings).toEqual({});
  });

  it("profiles a full panel of slider modulation cheaply", () => {
    const runtime = createAudioModulationRuntime();
    const settings: SliderModulationSettings = {};
    const baseValues: Record<string, number> = {};
    for (let index = 0; index < 72; index += 1) {
      const key = `slider-${index}`;
      settings[key] = sanitizeSliderModulationSettings({
        [key]: {
          min: 0,
          max: 1,
          buckets: {
            low: { enabled: index % 3 === 0, gain: 0.75 },
            mid: { enabled: index % 3 === 1, gain: 1 },
            high: { enabled: index % 3 === 2, gain: 1.4 }
          }
        }
      })[key];
      baseValues[key] = 0;
    }
    const panel = sanitizeAudioPanelState({
      buckets: {
        low: { gain: 1.2, attackMs: 80, decayMs: 160 },
        mid: { gain: 1.1, attackMs: 100, decayMs: 180 },
        high: { gain: 1.3, attackMs: 55, decayMs: 120 }
      }
    });
    const start = performance.now();
    let patches: Record<string, number> = {};
    for (let frameIndex = 0; frameIndex < 600; frameIndex += 1) {
      patches = applyAudioModulation(runtime, frame({ bass: 0.7, mid: 0.45, high: 0.9 }), panel, settings, baseValues, 1 / 120);
    }
    const elapsedMs = performance.now() - start;

    expect(Object.keys(patches)).toHaveLength(72);
    expect(elapsedMs).toBeLessThan(1000);
  });
});

describe("effective slider range overrides", () => {
  function modConfig(patch: Partial<SliderModulationConfig>): SliderModulationConfig {
    return { ...defaultSliderModulation(0, 8), ...patch };
  }

  it("keeps the base range when no mapping is active", () => {
    expect(effectiveSliderRange(0, 8, 4, null)).toEqual({ min: 0, max: 8 });
    expect(effectiveSliderRange(0, 8, 4, defaultSliderModulation(0, 8))).toEqual({ min: 0, max: 8 });
  });

  it("widens to an enabled audio activity range that exceeds the slider range", () => {
    const config = modConfig({
      min: -5,
      max: 20,
      buckets: { low: { enabled: true, gain: 1 }, mid: { enabled: false, gain: 1 }, high: { enabled: false, gain: 1 } }
    });
    expect(effectiveSliderRange(0, 8, 4, config)).toEqual({ min: -5, max: 20 });
  });

  it("ignores an audio range when no bucket is enabled", () => {
    const config = modConfig({ min: -5, max: 20 });
    expect(effectiveSliderRange(0, 8, 4, config)).toEqual({ min: 0, max: 8 });
  });

  it("widens to a mapped MIDI range that exceeds the slider range", () => {
    const config = modConfig({
      midi: {
        enabled: true,
        min: -10,
        max: 50,
        control: { inputId: "x", inputName: "x", messageType: "cc", channel: 1, controller: 7 }
      }
    });
    expect(effectiveSliderRange(0, 8, 4, config)).toEqual({ min: -10, max: 50 });
  });

  it("ignores a MIDI range with no mapped control", () => {
    const config = modConfig({ midi: { enabled: false, min: -10, max: 50, control: null } });
    expect(effectiveSliderRange(0, 8, 4, config)).toEqual({ min: 0, max: 8 });
  });

  it("always includes the current value so a driven value never falls outside the range", () => {
    expect(effectiveSliderRange(0, 8, 25, null)).toEqual({ min: 0, max: 25 });
    expect(effectiveSliderRange(0, 8, -3, null)).toEqual({ min: -3, max: 8 });
  });
});
