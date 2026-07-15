import { describe, expect, it } from "vitest";
import { createState, defaultControls, metrics, stepState } from "../sim";

describe("deterministic cockpit simulation", () => {
  it("replays the same seed exactly", () => {
    let a = createState(defaultControls);
    let b = createState(defaultControls);
    for (let i = 0; i < 12; i += 1) {
      a = stepState(a, defaultControls);
      b = stepState(b, defaultControls);
    }
    expect(Array.from(a.field)).toEqual(Array.from(b.field));
    expect(a.particles).toEqual(b.particles);
  });

  it("generates nonblank field metrics", () => {
    let state = createState(defaultControls);
    for (let i = 0; i < 4; i += 1) state = stepState(state, defaultControls);
    const result = metrics(state);
    expect(result.trailIntensitySum).toBeGreaterThan(1);
    expect(result.nonzeroCells).toBeGreaterThan(10);
    expect(Number.isFinite(result.averageVelocity)).toBe(true);
    expect(result.particleCount).toBe(defaultControls.particleCount);
  });
});
