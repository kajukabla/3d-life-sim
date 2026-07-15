import { describe, expect, it } from "vitest";
import { defaultLiveGpu3dConfig, liveShaderSources } from "../realtimeGpuSim3d";

// Static guards for the emergent-behavior knobs. The runtime byte-identity + delta proof lives in
// the Playwright spec (tests/emergentBehaviors.spec.ts); these cheap CPU checks pin the two invariants
// that make that guarantee hold: every master strength defaults to 0 (off), and every behavior's
// shader code is gated by `if (config.<x> > 0.0)` so the off-path is the classic computation graph.

describe("emergent behaviors", () => {
  it("every master strength defaults to 0 (off => byte-identical to the classic engine)", () => {
    for (const key of [
      "mips",
      "anisoFollow",
      "flockAlign",
      "flockSeparate",
      "chemotaxis",
      "quorumStrength",
      "leniaStrength",
      "speciesForce",
      "predator",
      "alarm",
      "grayScott",
      "energy"
    ] as const) {
      expect(defaultLiveGpu3dConfig[key], `${key} defaults to 0`).toBe(0);
    }
  });

  it("particle-update behaviors are gated so 0 skips the code path", () => {
    const update = liveShaderSources.particleUpdate;
    expect(update).toContain("config.mips > 0.0");
    expect(update).toContain("config.aniso_follow > 0.0");
    expect(update).toContain("config.flock_align > 0.0");
    expect(update).toContain("config.flock_separate > 0.0");
    expect(update).toContain("abs(config.chemotaxis) > 0.0");
    expect(update).toContain("config.quorum_strength > 0.0");
    expect(update).toContain("config.energy > 0.0");
  });

  it("field behaviors (Lenia + reaction-diffusion) are gated in the field update", () => {
    const field = liveShaderSources.field;
    expect(field).toContain("config.lenia_strength > 0.0 || config.gray_scott > 0.0");
    expect(field).toContain("config.lenia_strength > 0.0");
    expect(field).toContain("config.gray_scott > 0.0");
  });

  it("ecology behaviors (species/predator/alarm) are gated only in the ecology shader variant", () => {
    // The default particle update must NOT reference the ecology field or its forces, so the classic
    // path is byte-identical; the gated steering lives only in the ecology variant.
    const base = liveShaderSources.particleUpdate;
    expect(base).not.toContain("sample_ecology");
    expect(base).toContain("__ECOLOGY_FORCES__");

    const eco = liveShaderSources.particleUpdateEcology;
    expect(eco).toContain("config.species_force > 0.0 || config.predator > 0.0 || config.alarm > 0.0");
    expect(eco).toContain("fn sample_ecology");
    expect(eco).toContain("@group(0) @binding(6) var<storage, read> eco_field");
    expect(eco).not.toContain("__ECOLOGY_FORCES__");
  });
});
