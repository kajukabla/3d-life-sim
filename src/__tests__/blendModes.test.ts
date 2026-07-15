import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const source = readFileSync(new URL("../realtimeGpuSim3d.ts", import.meta.url), "utf8");

// All three particle blend modes (additive / alpha / opaque) should share ONE premultiplied
// convention: the fragment emits premultiplied color = color * intensity with alpha = intensity,
// and the blend state adds/composites that once. Previously additive used srcFactor "src-alpha"
// which multiplied by intensity a SECOND time (intensity^2), and opaque used a separate ad-hoc
// brightness formula that ignored particle_opacity.
describe("particle blend modes share one premultiplied convention", () => {
  it("additive adds premultiplied color once (no src-alpha double multiply)", () => {
    expect(source).not.toContain('srcFactor: "src-alpha" as GPUBlendFactor');
  });

  it("the additive blend is a clean add of the premultiplied source (srcFactor one, dstFactor one)", () => {
    const block = source.slice(source.indexOf("const additiveBlend"), source.indexOf("const alphaBlend"));
    expect(block).toContain('srcFactor: "one"');
    expect(block).toContain('dstFactor: "one"');
  });

  it("opaque no longer uses its own ad-hoc brightness curve", () => {
    expect(source).not.toContain("0.35 + core * 0.85");
    expect(source).not.toContain("smoothstep(0.08, 0.92, glow)");
  });

  it("intensity (glow * opacity * gain) is computed before any mode branch so every mode uses it", () => {
    const fs = source.slice(source.indexOf("fn splat_fs"), source.indexOf("fn splat_fs") + 1300);
    const intensityAt = fs.indexOf("let intensity = particle_intensity(glow, in.alpha, in.bright_mul);");
    const opaqueAt = fs.indexOf("particle_blend_mode == 2u");
    expect(intensityAt).toBeGreaterThan(0);
    expect(opaqueAt).toBeGreaterThan(0);
    expect(intensityAt).toBeLessThan(opaqueAt); // intensity computed before the opaque branch
  });
});
