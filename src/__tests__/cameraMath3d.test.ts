import { describe, it, expect } from "vitest";
import { DOF_DEPTH_K, focalToFocus, focusToFocal } from "../cameraMath3d";

describe("DOF focal-distance conversion", () => {
  it("focus plane sits at DOF_DEPTH_K / focusDistance in world units", () => {
    // The default focusDistance 0.54 should mean a focal distance of 1.85/0.54 ≈ 3.43,
    // i.e. roughly the default camera distance — a sensible 'focus on the cloud' default.
    expect(focusToFocal(0.54)).toBeCloseTo(DOF_DEPTH_K / 0.54, 5);
    expect(focusToFocal(0.54)).toBeGreaterThan(3.3);
    expect(focusToFocal(0.54)).toBeLessThan(3.6);
  });

  it("round-trips world distance <-> stored focusDistance", () => {
    for (const d of [2, 3.15, 5, 10]) {
      expect(focusToFocal(focalToFocus(d))).toBeCloseTo(d, 5);
    }
  });

  it("is monotonic: larger focal distance -> smaller stored focusDistance (it was inverted)", () => {
    expect(focalToFocus(2)).toBeGreaterThan(focalToFocus(10));
  });
});
