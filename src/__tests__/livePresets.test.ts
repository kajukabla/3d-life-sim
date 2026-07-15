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
      raySteps: 64,
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

  it("exposes the volume-density tendrils render preset", () => {
    const listedPreset = livePresets.find((preset) => preset.id === "volume-tendrils");
    const preset = getLivePreset("volume-tendrils");

    expect(listedPreset).toBeTruthy();
    expect(preset).toBe(listedPreset);
    expect(preset.id).toBe("volume-tendrils");
    expect(preset.config.width).toBe(96);
    expect(preset.config.height).toBe(96);
    expect(preset.config.depth).toBe(96);
    expect(preset.config.particleCount).toBe(262144);
    expect(preset.renderControls).toMatchObject({
      renderLayer: "volume-density",
      particleColorMode: "gradient-turbo",
      particleTint: "#ffffff",
      particleOpacity: 0.42,
      particleDensityCutoff: 0.002,
      particleDensityRadius: 0.04,
      densityPassStrength: 4.2,
      densitySmallScale: 2.2,
      densityLargeScale: 12,
      densityLargeThreshold: 0.18,
      densityContrastGain: 9,
      densityContrastBalance: 0.7,
      densityEmissionPower: 2.4,
      densityOcclusion: 0.35,
      raySteps: 64,
      rayResolution: 2160,
      fogStepScale: 0.6,
      emptySpaceSkipping: true,
      emptySpaceThreshold: 0.035,
      emptySpaceStride: 4,
      bloomStrength: 0.8,
      bloomThreshold: 0.2,
      bloomRadius: 2.2
    });
  });

  it("exposes a volume-density filament bloom showcase preset", () => {
    const preset = getLivePreset("volume-filament-bloom");

    expect(preset.id).toBe("volume-filament-bloom");
    expect(preset.name).toBe("Volume Filament Bloom");
    expect(preset.config.width).toBe(96);
    expect(preset.config.height).toBe(96);
    expect(preset.config.depth).toBe(96);
    expect(preset.config.particleCount).toBe(262144);
    expect(preset.config.cohorts).toBe(6);
    expect(preset.renderControls).toMatchObject({
      renderLayer: "volume-density",
      particleColorMode: "gradient-rainbow",
      particleTint: "#ffffff",
      particleBrightness: 0.82,
      particleOpacity: 0.14,
      particleDensityCutoff: 0.0012,
      particleDensityRadius: 0.04,
      densityPassStrength: 5.8,
      densitySmallScale: 0.75,
      densityLargeScale: 5.2,
      densityLargeThreshold: 0.15,
      densityContrastGain: 12.5,
      densityContrastBalance: 1.05,
      densityEmissionPower: 2.05,
      densityOcclusion: 0.12,
      raySteps: 64,
      rayResolution: 2160,
      fogStepScale: 0.52,
      emptySpaceSkipping: true,
      emptySpaceThreshold: 0.035,
      emptySpaceStride: 4,
      bloomStrength: 0.18,
      bloomThreshold: 0.72,
      bloomRadius: 1.05
    });
  });

  it("exposes an accumulation filaments showcase preset", () => {
    const listedPreset = livePresets.find((preset) => preset.id === "accumulation-filaments");
    const preset = getLivePreset("accumulation-filaments");

    expect(listedPreset).toBeTruthy();
    expect(preset).toBe(listedPreset);
    expect(preset.id).toBe("accumulation-filaments");
    expect(preset.name).toBe("Accumulation Filaments");
    expect(preset.config.width).toBe(96);
    expect(preset.config.height).toBe(96);
    expect(preset.config.depth).toBe(96);
    expect(preset.config.particleCount).toBe(262144);
    expect(preset.config.colorByCohort).toBe(true);
    expect(preset.renderControls).toMatchObject({
      renderLayer: "accumulation",
      particleColorMode: "gradient-rainbow",
      particleTint: "#ffffff",
      particleBrightness: 1.4,
      particleOpacity: 0.2,
      particleDensityCutoff: 0.0085,
      particleDensityRadius: 0.046,
      accumulationStrength: 12,
      accumulationRadius: 0.86,
      accumulationCurve: 5,
      accumulationMemory: 0.88,
      accumulationNoiseReject: 0.82,
      raySteps: 64,
      rayResolution: 2160,
      bloomStrength: 0.42,
      bloomThreshold: 0.55,
      bloomRadius: 1.1,
      filament: 1
    });
  });
});
