import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { liveParticleStep, minLiveParticles } from "../particleLimits";
import { mobileMaxParticles, mobileParticleCount } from "../mobileLaunch";

describe("mobileParticleCount", () => {
  it("caps the curated default preset at ~200K on mobile", () => {
    const preset = JSON.parse(readFileSync(new URL("../../Presets/default.json", import.meta.url), "utf8"));
    const desktopCount = preset.liveConfig.particleCount as number;
    expect(desktopCount).toBe(3399680);
    expect(mobileParticleCount(desktopCount)).toBe(199680);
    expect(mobileMaxParticles).toBe(199680);
  });

  it("caps every preset above the mobile budget (semantic delta)", () => {
    for (const count of [3399680, 2399232, 505856]) {
      expect(mobileParticleCount(count)).toBe(mobileMaxParticles);
      expect(mobileParticleCount(count)).not.toBe(count);
    }
  });

  it("leaves smaller presets untouched", () => {
    expect(mobileParticleCount(16384)).toBe(16384);
    expect(mobileParticleCount(102400)).toBe(102400);
  });

  it("snaps to the particle step and respects the minimum", () => {
    expect(mobileParticleCount(3399680) % liveParticleStep).toBe(0);
    expect(mobileParticleCount(8)).toBe(minLiveParticles);
    expect(mobileParticleCount(minLiveParticles)).toBe(minLiveParticles);
  });
});
