import { describe, expect, it } from "vitest";
import { sliderPosToValue, sliderValueToPos } from "../sliderCurve";

describe("non-linear slider curve", () => {
  it("maps endpoints exactly (pos 0 -> min, pos 1 -> max) for any curve", () => {
    for (const curve of [1, 2, 3]) {
      expect(sliderPosToValue(0, 0, 0.05, curve)).toBeCloseTo(0, 10);
      expect(sliderPosToValue(1, 0, 0.05, curve)).toBeCloseTo(0.05, 10);
    }
  });

  it("round-trips value <-> position", () => {
    for (const v of [0.0005, 0.002, 0.01, 0.025, 0.05]) {
      const pos = sliderValueToPos(v, 0, 0.05, 3);
      expect(sliderPosToValue(pos, 0, 0.05, 3)).toBeCloseTo(v, 9);
    }
  });

  it("front-loads resolution: small values occupy MORE slider travel than linear", () => {
    // Useful cutoff ~0.0005..0.014. On a linear 0-0.05 slider that's the leftmost ~28%.
    const linearTravel = sliderValueToPos(0.014, 0, 0.05, 1) - sliderValueToPos(0.0005, 0, 0.05, 1);
    const curvedTravel = sliderValueToPos(0.014, 0, 0.05, 3) - sliderValueToPos(0.0005, 0, 0.05, 3);
    expect(curvedTravel).toBeGreaterThan(linearTravel * 1.5); // curve spreads the useful band out
  });

  it("is monotonic in position", () => {
    let prev = -1;
    for (let p = 0; p <= 1.0001; p += 0.05) {
      const v = sliderPosToValue(p, 0, 0.05, 3);
      expect(v).toBeGreaterThan(prev);
      prev = v;
    }
  });
});
