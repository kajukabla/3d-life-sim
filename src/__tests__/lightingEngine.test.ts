import { describe, expect, it } from "vitest";
import { computeLightFrame, foldPhase, lightPhase } from "../lighting/lightingEngine";
import { sampleLightingGradient } from "../lighting/gradient";
import { defaultLightingState, demoLightingState, type LightConfig, type LightingColorSource, type LightingState } from "../lighting/types";
import { gradientColor, particleGradientKeyForMode, type ParticleGradientKey } from "../particleGradients";

function lights(n: number): LightConfig[] {
  return Array.from({ length: n }, (_, i) => ({ channel: i, name: `L${i}`, enabled: true }));
}

function source(mode: LightingColorSource["mode"]): LightingColorSource {
  return { mode, palette: "aurora", tint: "#ffffff" };
}

function state(overrides: Partial<LightingState> = {}): LightingState {
  return { ...defaultLightingState(), ...overrides };
}

const equalRgb = (a: [number, number, number], b: [number, number, number]) =>
  a[0] === b[0] && a[1] === b[1] && a[2] === b[2];

describe("sampleLightingGradient", () => {
  it("changing the color mode changes the sampled color (semantic delta, not a constant)", () => {
    const t = 0.4;
    const inferno = sampleLightingGradient(source("gradient-inferno"), t);
    const viridis = sampleLightingGradient(source("gradient-viridis"), t);
    expect(equalRgb(inferno, viridis)).toBe(false);
  });

  it("depends on the scrub position (advancing t changes the color)", () => {
    const a = sampleLightingGradient(source("gradient-viridis"), 0.1);
    const b = sampleLightingGradient(source("gradient-viridis"), 0.7);
    expect(equalRgb(a, b)).toBe(false);
  });

  it("solid mode returns the tint color", () => {
    const rgb = sampleLightingGradient({ mode: "solid", palette: "aurora", tint: "#ff8000" }, 0.5);
    expect(rgb).toEqual([255, 128, 0]);
  });

  it("resolves every particle color mode to the particle shader's concrete gradient", () => {
    const expected: Array<[LightingColorSource["mode"], ParticleGradientKey]> = [
      ["solid", "solid"],
      ["gradient-inferno", "gradient-inferno"],
      ["gradient-magma", "gradient-magma"],
      ["gradient-viridis", "gradient-viridis"],
      ["gradient-turbo", "gradient-turbo"],
      ["gradient-rainbow", "gradient-rainbow"],
      ["gradient-spectral", "gradient-spectral"],
      ["gradient-plasma", "gradient-plasma"],
      ["gradient-cosmic", "gradient-cosmic"],
      ["gradient-ice", "gradient-ice"],
      ["gradient-ember", "gradient-ember"],
      ["velocity", "velocity"],
      ["velocity-inferno", "gradient-inferno"],
      ["velocity-viridis", "gradient-viridis"],
      ["velocity-spectral", "gradient-spectral"],
      ["velocity-cosmic", "gradient-cosmic"],
      ["velocity-ice", "gradient-ice"],
      ["audio-magma", "gradient-magma"],
      ["audio-viridis", "gradient-viridis"],
      ["audio-turbo", "gradient-turbo"],
      ["audio-cosmic", "gradient-cosmic"],
      ["audio-ice", "gradient-ice"],
      ["audio-ember", "gradient-ember"],
      ["audio-plasma", "gradient-plasma"],
      ["cohort", "cohort"],
      ["audio", "audio"]
    ];
    for (const [mode, key] of expected) {
      expect(particleGradientKeyForMode(mode)).toBe(key);
      if (key !== "solid") {
        expect(sampleLightingGradient(source(mode), 0.42)).toEqual(gradientColor(key, 0.42));
      }
    }
  });
});

describe("lightPhase", () => {
  it("both spreads 0 -> identical phase for every channel at any time", () => {
    const s = state({ scrub: 0.2, scrubRateHz: 0.3, phaseSpread: 0, freqSpread: 0 });
    for (const t of [0, 1.3, 7.9]) {
      const p0 = lightPhase(s, 0, t);
      for (const ch of [1, 5, 19, 123]) {
        expect(lightPhase(s, ch, t)).toBeCloseTo(p0, 10);
      }
    }
  });

  it("phaseSpread > 0 -> channels get distinct static offsets (when unlocked)", () => {
    const s = state({ locked: false, phaseSpread: 0.5 });
    const phases = new Set([0, 1, 2, 3, 4].map((ch) => lightPhase(s, ch, 0).toFixed(6)));
    expect(phases.size).toBeGreaterThan(1);
  });

  it("freqSpread only diverges over time (identical at t=0, differ later)", () => {
    const s = state({ locked: false, freqSpread: 2, phaseSpread: 0 });
    const atZero = new Set([0, 1, 2, 3].map((ch) => lightPhase(s, ch, 0).toFixed(6)));
    expect(atZero.size).toBe(1);
    const atOne = new Set([0, 1, 2, 3].map((ch) => lightPhase(s, ch, 1).toFixed(6)));
    expect(atOne.size).toBeGreaterThan(1);
  });
});

describe("demoLightingState (idle preset)", () => {
  it("is slow, fully out of phase, ping-pong, no strobe", () => {
    const d = demoLightingState(defaultLightingState());
    expect(d.phaseSpread).toBe(1);
    expect(d.scrubRateHz).toBeGreaterThan(0);
    expect(d.scrubRateHz).toBeLessThan(0.05);
    expect(d.loopMode).toBe("pingpong");
    expect(d.strobe.enabled).toBe(false);
  });

  it("spreads distinct colors across the lights", () => {
    const frame = computeLightFrame(demoLightingState(defaultLightingState()), source("gradient-turbo"), lights(6), 0);
    expect(new Set(frame.map((c) => c.join(","))).size).toBeGreaterThan(1);
  });

  it("preserves brightness", () => {
    const prev = { ...defaultLightingState(), master: 0.7 };
    expect(demoLightingState(prev).master).toBe(0.7);
  });
});

describe("foldPhase (loop modes)", () => {
  it("ping-pong is continuous across the loop boundary; repeat cuts", () => {
    const ppBelow = foldPhase(0.98, "pingpong");
    const ppAbove = foldPhase(1.02, "pingpong");
    expect(Math.abs(ppAbove - ppBelow)).toBeLessThan(0.1); // no cut
    const rBelow = foldPhase(0.98, "repeat");
    const rAbove = foldPhase(1.02, "repeat");
    expect(Math.abs(rAbove - rBelow)).toBeGreaterThan(0.8); // hard cut
  });

  it("ping-pong is a triangle: 0 -> end -> back", () => {
    expect(foldPhase(0, "pingpong")).toBeCloseTo(0, 6);
    expect(foldPhase(1, "pingpong")).toBeCloseTo(1, 6);
    expect(foldPhase(2, "pingpong")).toBeCloseTo(0, 6);
    expect(foldPhase(0.5, "pingpong")).toBeCloseTo(0.5, 6);
  });

  it("repeat wraps 0..1 (sawtooth)", () => {
    expect(foldPhase(0.25, "repeat")).toBeCloseTo(0.25, 6);
    expect(foldPhase(1.25, "repeat")).toBeCloseTo(0.25, 6);
  });

  it("ping-pong keeps a non-cyclic gradient continuous at the loop point", () => {
    const src = source("gradient-inferno");
    // Just before and just after the loop point sample nearly the same color...
    const before = sampleLightingGradient(src, foldPhase(0.99, "pingpong"));
    const after = sampleLightingGradient(src, foldPhase(1.01, "pingpong"));
    expect(Math.abs(after[0] - before[0]) + Math.abs(after[2] - before[2])).toBeLessThan(30);
    // ...whereas repeat jumps from the bright end back to the dark start.
    const repBefore = sampleLightingGradient(src, foldPhase(0.99, "repeat"));
    const repAfter = sampleLightingGradient(src, foldPhase(1.01, "repeat"));
    expect(Math.abs(repAfter[0] - repBefore[0]) + Math.abs(repAfter[2] - repBefore[2])).toBeGreaterThan(80);
  });
});

describe("computeLightFrame", () => {
  it("synced (both spreads 0) -> all enabled channels identical", () => {
    const frame = computeLightFrame(state({ scrub: 0.35 }), source("gradient-turbo"), lights(6), 2.5);
    for (const rgb of frame) expect(equalRgb(rgb, frame[0])).toBe(true);
  });

  it("phaseSpread > 0 -> channels differ (when unlocked)", () => {
    const frame = computeLightFrame(state({ locked: false, phaseSpread: 0.6 }), source("gradient-turbo"), lights(6), 0);
    const distinct = new Set(frame.map((c) => c.join(",")));
    expect(distinct.size).toBeGreaterThan(1);
  });

  it("locked forces every light to the identical color, even with spreads set (robust sync)", () => {
    // Big spreads + a nonzero time that would normally desync them...
    const s = state({ locked: true, phaseSpread: 1, freqSpread: 4, scrub: 0.3 });
    const frame = computeLightFrame(s, source("gradient-turbo"), lights(8), 7.91);
    for (const rgb of frame) expect(rgb).toEqual(frame[0]);
    // ...and unlocking with the same spreads breaks them apart.
    const unlocked = computeLightFrame({ ...s, locked: false }, source("gradient-turbo"), lights(8), 7.91);
    expect(new Set(unlocked.map((c) => c.join(","))).size).toBeGreaterThan(1);
  });

  it("disabled light is black", () => {
    const cfg = lights(2);
    cfg[1].enabled = false;
    const frame = computeLightFrame(state(), source("gradient-viridis"), cfg, 0);
    expect(frame[1]).toEqual([0, 0, 0]);
  });

  it("master brightness scales output; master 0 -> black", () => {
    const src = source("gradient-viridis");
    const full = computeLightFrame(state({ master: 1, gamma: 1 }), src, lights(1), 0.4)[0];
    const half = computeLightFrame(state({ master: 0.5, gamma: 1 }), src, lights(1), 0.4)[0];
    const off = computeLightFrame(state({ master: 0 }), src, lights(1), 0.4)[0];
    expect(off).toEqual([0, 0, 0]);
    expect(half[0]).toBe(Math.round(full[0] * 0.5));
    expect(half[1]).toBe(Math.round(full[1] * 0.5));
  });

  it("final brightness dims every bulb; brightness 0 -> black", () => {
    const src = source("gradient-viridis");
    const full = computeLightFrame(state({ brightness: 1 }), src, lights(1), 0.4)[0];
    const dim = computeLightFrame(state({ brightness: 0.5 }), src, lights(1), 0.4)[0];
    const off = computeLightFrame(state({ brightness: 0 }), src, lights(1), 0.4)[0];
    expect(off).toEqual([0, 0, 0]);
    expect(dim[0]).toBe(Math.round(full[0] * 0.5));
    expect(dim[1]).toBe(Math.round(full[1] * 0.5));
  });

  it("strobe: on-window shows color, off-window is black, at the set rate", () => {
    const src = source("gradient-inferno");
    const s = state({ scrub: 0.4, strobe: { enabled: true, hz: 5, duty: 0.5 } });
    // cycle = 0.2s. t=0.02 -> frac(t*5)=0.1 < 0.5 -> ON. t=0.15 -> frac=0.75 >= 0.5 -> OFF.
    const on = computeLightFrame(s, src, lights(1), 0.02)[0];
    const offFrame = computeLightFrame(s, src, lights(1), 0.15)[0];
    expect(on).not.toEqual([0, 0, 0]);
    expect(offFrame).toEqual([0, 0, 0]);
  });

  it("would fail if the engine ignored the gradient (trivial-replacement guard)", () => {
    const s = state({ scrub: 0.15 });
    const a = computeLightFrame(s, source("gradient-inferno"), lights(1), 0)[0];
    const b = computeLightFrame({ ...s, scrub: 0.85 }, source("gradient-inferno"), lights(1), 0)[0];
    // real gradient dependence: different scrub -> different color
    expect(equalRgb(a, b)).toBe(false);
  });
});
