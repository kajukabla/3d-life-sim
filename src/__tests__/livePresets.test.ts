import { describe, expect, it } from "vitest";
import { getLivePreset, livePresets } from "../livePresets";

describe("curated live presets", () => {
  it("keeps curated preset ids unique", () => {
    const ids = livePresets.map((preset) => preset.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("uses Default1 as the default live preset", () => {
    const preset = getLivePreset("shimmer-orb");

    expect(livePresets[0]).toBe(preset);
    expect(preset.name).toBe("Default1");
    expect(preset.source).toBe("3d-life-sim/built-in/Default1");
    expect(preset.config.simulationSpeed).toBe(0.38);
    expect(preset.config.particleCount).toBe(505856);
    expect(preset.config.width).toBe(96);
    expect(preset.config.cohorts).toBe(25);
    expect(preset.renderControls).toMatchObject({
      renderLayer: "particles",
      rayResolution: 2160,
      colorSaturation: 1.16,
      colorContrast: 1.08,
      particleBlendMode: "additive",
      particleColorMode: "gradient-turbo",
      particleTint: "#2b0fff",
      particleBrightness: 1.15,
      particleOpacity: 0.02,
      particleVelocityStretch: true,
      particleStretch: 1.2,
      particleSupportMask: 1,
      particleSupportRadius: 0.35,
      particleSupportNeighbors: 12,
      particleDensityCutoff: 0,
      particleDensityRadius: 0,
      cameraYaw: 0.2865388222983949,
      cameraDistance: 0.781352472543831,
      dofEnabled: false
    });
  });

  it("only exposes presets compatible with particle rendering", () => {
    for (const preset of livePresets) {
      expect(preset.renderControls?.renderLayer ?? "particles", preset.id).toBe("particles");
    }
    expect(livePresets.map((preset) => preset.id)).not.toEqual(expect.arrayContaining([
      "volume-tendrils",
      "volume-filament-bloom",
      "accumulation-filaments"
    ]));
  });
});
