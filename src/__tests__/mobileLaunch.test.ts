import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { liveParticleStep, minLiveParticles } from "../particleLimits";
import { mobileParticleCount } from "../mobileLaunch";

describe("mobileParticleCount", () => {
  it("boots the curated default preset at a tenth of its desktop budget", () => {
    const preset = JSON.parse(readFileSync(new URL("../../Presets/default.json", import.meta.url), "utf8"));
    const desktopCount = preset.liveConfig.particleCount as number;
    expect(desktopCount).toBe(3399680);
    expect(mobileParticleCount(desktopCount)).toBe(339968);
  });

  it("changes the budget for every large preset count (semantic delta)", () => {
    for (const count of [3399680, 2399232, 505856, 16384]) {
      const scaled = mobileParticleCount(count);
      expect(scaled).not.toBe(count);
      expect(scaled).toBeGreaterThanOrEqual(count / 10 - liveParticleStep);
      expect(scaled).toBeLessThanOrEqual(count / 10 + liveParticleStep);
    }
  });

  it("snaps to the particle step", () => {
    expect(mobileParticleCount(3399680) % liveParticleStep).toBe(0);
    expect(mobileParticleCount(2399232) % liveParticleStep).toBe(0);
  });

  it("never drops below the minimum live particle count", () => {
    expect(mobileParticleCount(8)).toBe(minLiveParticles);
    expect(mobileParticleCount(minLiveParticles)).toBe(minLiveParticles);
    expect(mobileParticleCount(1024)).toBe(minLiveParticles);
  });
});
