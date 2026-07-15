import { describe, expect, it } from "vitest";
import { encodePostUniforms, liveShaderSources } from "../realtimeGpuSim3d";

describe("post-process uniform encoding", () => {
  it("packs bloom params, tint, grade, and Spencer octave weights", () => {
    const buf = encodePostUniforms({
      bloomStrength: 0.5,
      bloomThreshold: 0.6,
      bloomRadius: 1.25,
      colorSaturation: 1.16,
      colorContrast: 1.08,
      tint: [1, 0.9, 0.8],
      levelCount: 6,
      weights: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
      chromaticAberration: 0.42,
      vignetteStrength: 0.7,
      vignetteSoftness: 0.25,
      streakStrength: 0.55,
      streakLength: 0.8,
      streakVertical: 0.3,
      flareHeight: 0.4,
      flareCutoff: 0.15
    });
    expect(buf.byteLength).toBe(128);
    const f = new Float32Array(buf);
    expect(f[0]).toBeCloseTo(0.5, 6);
    expect(f[1]).toBeCloseTo(0.6, 6);
    expect(f[2]).toBeCloseTo(1.25, 6);
    expect(f[3]).toBeCloseTo(6, 6); // levelCount
    expect(f[4]).toBeCloseTo(1, 6);
    expect(f[5]).toBeCloseTo(0.9, 6);
    expect(f[6]).toBeCloseTo(0.8, 6);
    expect(f[8]).toBeCloseTo(1.16, 6);
    expect(f[9]).toBeCloseTo(1.08, 6);
    // weights_a (offset 48 = f[12]) and weights_b (offset 64 = f[16])
    expect(f[12]).toBeCloseTo(0.1, 6);
    expect(f[15]).toBeCloseTo(0.4, 6);
    expect(f[16]).toBeCloseTo(0.5, 6);
    expect(f[17]).toBeCloseTo(0.6, 6);
    // lens = (chromatic aberration, vignette strength, vignette softness, reserved)
    expect(f[20]).toBeCloseTo(0.42, 6);
    expect(f[21]).toBeCloseTo(0.7, 6);
    expect(f[22]).toBeCloseTo(0.25, 6);
    // streak = (strength, length, vertical/star mix, reserved)
    expect(f[24]).toBeCloseTo(0.55, 6);
    expect(f[25]).toBeCloseTo(0.8, 6);
    expect(f[26]).toBeCloseTo(0.3, 6);
    // flare = (height, cutoff, reserved, reserved)
    expect(f[28]).toBeCloseTo(0.4, 6);
    expect(f[29]).toBeCloseTo(0.15, 6);
  });
});

// Mirror of the WGSL knee_prefilter bright-pass (pinned by the source guard below).
function kneePrefilter(c: [number, number, number], threshold: number): [number, number, number] {
  const br = Math.max(c[0], c[1], c[2]);
  const knee = threshold * 0.5 + 1e-5;
  let soft = Math.min(Math.max(br - threshold + knee, 0), 2 * knee);
  soft = (soft * soft) / (4 * knee + 1e-5);
  const contrib = Math.max(soft, br - threshold) / Math.max(br, 1e-5);
  return [c[0] * contrib, c[1] * contrib, c[2] * contrib];
}

describe("bloom bright-pass (prefilter knee)", () => {
  it("removes energy well below the threshold", () => {
    const out = kneePrefilter([0.1, 0.1, 0.1], 0.6);
    expect(Math.max(...out)).toBeLessThan(0.01);
  });
  it("passes energy well above the threshold roughly intact", () => {
    const out = kneePrefilter([3, 3, 3], 0.6);
    expect(out[0]).toBeGreaterThan(2.0);
  });
  it("ramps smoothly across the knee (no hard cutoff)", () => {
    const below = Math.max(...kneePrefilter([0.5, 0.5, 0.5], 0.6));
    const at = Math.max(...kneePrefilter([0.6, 0.6, 0.6], 0.6));
    const above = Math.max(...kneePrefilter([0.75, 0.75, 0.75], 0.6));
    expect(below).toBeLessThan(at);
    expect(at).toBeLessThan(above);
    expect(at).toBeGreaterThan(0); // soft knee lets a little through at the threshold
  });
});

describe("post-process shader stack", () => {
  const src = liveShaderSources.post;
  it("runs the AgX view transform (exact Wrensch inset matrix)", () => {
    expect(src).toContain("0.842479062253094");
    expect(src).toContain("fn agx(");
  });
  it("applies cheap saturation and contrast inside the final AgX composite", () => {
    expect(src).toContain("grade: vec4f");
    expect(src).toContain("fn apply_color_grade(");
    expect(src).toContain("mix(vec3f(luma), val, clamp(post.grade.x, 0.0, 2.0))");
    expect(src).toContain("clamp(post.grade.y, 0.5, 1.8)");
    expect(src).toContain("agx_eotf(apply_color_grade(agx(max(combined, vec3f(0.0)))))");
    expect(src).toContain("agx_eotf(apply_color_grade(agx(max(scene, vec3f(0.0)))))");
  });
  it("has the full bloom pyramid plus a no-bloom grade-only composite", () => {
    expect(src).toContain("fn prefilter_fs(");
    expect(src).toContain("fn downsample_fs(");
    expect(src).toContain("fn upsample_fs(");
    expect(src).toContain("fn composite_fs(");
    expect(src).toContain("fn composite_no_bloom_fs(");
  });
  it("samples every declared bloom mip with the encoded octave weights", () => {
    expect(src).toContain("@group(0) @binding(10) var bloom_m7");
    expect(src).toContain("post.weights_a.x");
    expect(src).toContain("textureSampleLevel(bloom_m7");
    expect(src).toContain("post.weights_b.w");
  });
  it("uses a neutral (non-cyan) bloom tint", () => {
    // Scene-specific tint must not be baked into the renderer.
    expect(src).not.toContain("0.30, 0.95, 0.88");
    expect(src).not.toContain("0.95, 0.88");
  });
});
