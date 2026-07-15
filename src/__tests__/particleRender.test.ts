import { describe, expect, it } from "vitest";
import { liveShaderSources } from "../realtimeGpuSim3d";

// Mirror of the WGSL particle projection/sizing. The exact expressions are pinned in the
// shader by source guards (see gpuSim3d.test.ts and the constants check below), so these
// properties are a faithful proof of how the real renderer behaves.
const DEPTH_METRIC_FOCAL = 1.85;
const DEFAULT_FOCAL_K = 1.702;
const PARTICLE_NEAR = 0.05;
const PARTICLE_SIZE_REF_DEPTH = 3.15;

// view_depth = camera distance + the point's depth along the view axis.
function isBehindCamera(viewDepth: number): boolean {
  return viewDepth <= PARTICLE_NEAR;
}

// Perspective billboard: no hidden screen-space min/max clamp. Old P Min Px values from saved
// settings must not turn live particles back into screen-space sprites.
function splatSizePx(sizePx: number, viewDepth: number, focal = DEFAULT_FOCAL_K): number {
  const perspective = DEPTH_METRIC_FOCAL / Math.max(PARTICLE_NEAR, viewDepth);
  return Math.max(0, sizePx) * perspective * (PARTICLE_SIZE_REF_DEPTH / DEPTH_METRIC_FOCAL) * (focal / DEFAULT_FOCAL_K);
}

function splatFootprintPx(sizePx: number, viewDepth: number, focusDistance: number, aperture: number, dofBlur: number, dofEnabled = true): number {
  return splatSizePx(sizePx, viewDepth) + particleDofBlurPx(focusDepthForViewDepth(viewDepth), focusDistance, aperture, dofBlur, dofEnabled);
}

function focusDepthForViewDepth(viewDepth: number): number {
  return DEPTH_METRIC_FOCAL / Math.max(PARTICLE_NEAR, viewDepth);
}

function particleDofBlurPx(depth: number, focusDistance: number, aperture: number, dofBlur: number, dofEnabled = true): number {
  return particleDofBlurPxFromDefocus(particleFocusDefocus(depth, focusDistance, aperture), dofBlur, dofEnabled);
}

function particleFocusDefocus(depth: number, focusDistance: number, aperture: number): number {
  return Math.min(1, Math.max(0, Math.abs(depth - focusDistance) * Math.max(0, aperture) * 8));
}

function particleDofBlurPxFromDefocus(defocus: number, dofBlur: number, dofEnabled = true): number {
  if (!dofEnabled) return 0;
  return Math.min(1, Math.max(0, defocus)) * Math.max(0, dofBlur) * 8;
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function particleSpriteBlurAmount(focusBlurPx: number, baseSizePx: number): number {
  const relativeBlur = focusBlurPx / Math.max(0.75, baseSizePx * 0.75);
  return smoothstep(0.08, 1.0, relativeBlur);
}

function particleSpriteProfile(radius: number, defocus: number, aa = 0.008): number {
  const amount = Math.min(1, Math.max(0, defocus));
  const sharp = 1 - smoothstep(1 - aa * 2, 1 + aa, radius);
  if (amount <= 0.001) return Math.min(1, Math.max(0, sharp));
  const softInner = 0.86 * (1 - amount) + 0.05 * amount;
  const softDisc = 1 - smoothstep(softInner, 1, radius);
  const center = 1 - smoothstep(0, 1, radius);
  const soft = softDisc * (0.8 + center * 0.2);
  return Math.min(1, Math.max(0, sharp * (1 - smoothstep(0, 1, amount)) + soft * smoothstep(0, 1, amount)));
}

describe("particle splat size follows 3D perspective (bug #1)", () => {
  it("renders particle_size_px at the reference (default) depth", () => {
    expect(splatSizePx(6, PARTICLE_SIZE_REF_DEPTH)).toBeCloseTo(6, 5);
  });

  it("shrinks as the particle gets farther (zoom out) and grows as it gets closer", () => {
    const near = splatSizePx(6, 1.0);
    const mid = splatSizePx(6, 3.15);
    const far = splatSizePx(6, 9.0);
    const farther = splatSizePx(6, 18.0);
    expect(near).toBeGreaterThan(mid);
    expect(mid).toBeGreaterThan(far);
    expect(far).toBeGreaterThan(farther);
  });

  it("is strictly monotonic decreasing in view depth across the visible range", () => {
    let previous = Infinity;
    for (let depth = 0.2; depth <= 30; depth += 0.2) {
      const size = splatSizePx(6, depth);
      expect(size).toBeLessThan(previous);
      previous = size;
    }
  });

  it("scales inversely with depth — doubling the distance halves the size", () => {
    expect(splatSizePx(6, 4) / splatSizePx(6, 8)).toBeCloseTo(2, 5);
  });

  it("scales with focal length exactly like projected 3D geometry", () => {
    const narrow = focalK(40);
    const wide = focalK(80);
    expect(splatSizePx(6, 3.15, narrow) / splatSizePx(6, 3.15, wide)).toBeCloseTo(narrow / wide, 5);
  });

  it("combines camera distance and FOV as one perspective model", () => {
    const size = splatSizePx(6, 7.4, focalK(72));
    const expected = 6 * (PARTICLE_SIZE_REF_DEPTH / 7.4) * (focalK(72) / DEFAULT_FOCAL_K);
    expect(size).toBeCloseTo(expected, 5);
  });

  it("has NO screen-space size cap — a close particle keeps growing without limit", () => {
    // The old max-px clamp made close particles stop growing (they appeared to shrink as the
    // cloud spread). Now halving the distance keeps doubling the size, all the way in.
    expect(splatSizePx(6, 1) / splatSizePx(6, 0.5)).toBeCloseTo(0.5, 5);
    expect(splatSizePx(6, 0.5)).toBeGreaterThan(35); // very close -> very large, not capped
    expect(splatSizePx(6, 0.1)).toBeGreaterThan(180);
  });

  it("keeps shrinking with distance without a screen-space floor", () => {
    expect(splatSizePx(6, 100000)).toBeCloseTo(0.000189, 6);
  });

  it("does not preserve the old P Min Px screen-space floor behavior", () => {
    expect(liveShaderSources.splatCommon).not.toContain("floor_px");
    expect(liveShaderSources.splatCommon).not.toContain("max(floor_px");
  });
});

describe("fake particle sprite DOF", () => {
  it("adds no sprite blur at the focal plane and grows blur with focus error", () => {
    const focus = focusDepthForViewDepth(3.15);
    expect(particleDofBlurPx(focus, focus, 0.8, 4)).toBe(0);

    const nearDepth = focusDepthForViewDepth(1.2);
    const farDepth = focusDepthForViewDepth(9.0);
    expect(particleDofBlurPx(nearDepth, focus, 0.8, 4)).toBeGreaterThan(0);
    expect(particleDofBlurPx(farDepth, focus, 0.8, 4)).toBeGreaterThan(0);
  });

  it("uses aperture for the defocus mask and DOF Blur for maximum extra blur radius", () => {
    const depth = 0.95;
    const focus = 0.25;
    const defocus = particleFocusDefocus(depth, focus, 0.9);
    expect(defocus).toBeGreaterThan(0);
    expect(particleDofBlurPx(depth, focus, 0.9, 0)).toBe(0);
    expect(particleDofBlurPx(depth, focus, 0.9, 4)).toBeCloseTo(defocus * 32, 5);
    expect(particleDofBlurPx(depth, focus, 0.9, 4)).toBeLessThanOrEqual(32);
  });

  it("keeps particle footprints at their base perspective size when DOF is disabled", () => {
    const focus = focusDepthForViewDepth(3.15);
    const farDepth = 12;
    expect(particleDofBlurPx(focusDepthForViewDepth(farDepth), focus, 0.9, 4, false)).toBe(0);
    expect(splatFootprintPx(6, farDepth, focus, 0.9, 4, false)).toBeCloseTo(splatSizePx(6, farDepth), 6);
  });

  it("keeps focused particles sharp and expands only defocused footprints", () => {
    const focus = focusDepthForViewDepth(3.15);
    const farDepth = 12;
    const focused = splatFootprintPx(6, 3.15, focus, 0.9, 4);
    const base = splatFootprintPx(6, farDepth, focus, 0.9, 0);
    const blurred = splatFootprintPx(6, farDepth, focus, 0.9, 4);
    expect(focused).toBeCloseTo(splatSizePx(6, 3.15), 6);
    expect(base).toBeCloseTo(splatSizePx(6, farDepth), 6);
    expect(blurred).toBeGreaterThan(base);
  });

  it("turns blur pixels into per-particle softness relative to the base sprite size", () => {
    const base = splatSizePx(6, 3.15);
    expect(particleSpriteBlurAmount(0, base)).toBe(0);
    expect(particleSpriteBlurAmount(base, base)).toBeGreaterThan(0.9);
  });

  it("is a sharp circle in focus and a softer disc when defocused", () => {
    expect(particleSpriteProfile(0.5, 0)).toBeCloseTo(1, 5);
    expect(particleSpriteProfile(0.95, 0)).toBeGreaterThan(0.95);
    expect(particleSpriteProfile(0.5, 1)).toBeLessThan(particleSpriteProfile(0.5, 0));
    expect(particleSpriteProfile(0.9, 1)).toBeLessThan(particleSpriteProfile(0.5, 1));
    expect(particleSpriteProfile(1.04, 1)).toBeLessThan(0.01);
  });

  it("wires the live WGSL particle path to perspective size and DOF softness", () => {
    expect(liveShaderSources.splatCommon).toContain("fn particle_dof_blur_px(depth: f32) -> f32");
    expect(liveShaderSources.splatCommon).toContain("fn particle_focus_defocus(depth: f32) -> f32");
    expect(liveShaderSources.splatCommon).toContain("fn particle_dof_blur_px_from_defocus(defocus: f32) -> f32");
    expect(liveShaderSources.splatCommon).toContain("if (uniforms.dof_enabled == 0u)");
    expect(liveShaderSources.splatCommon).toContain("fn particle_sprite_blur_amount(focus_blur_px: f32, base_size_px: f32) -> f32");
    expect(liveShaderSources.splatCommon).toContain("fn particle_dof_splat_radius_ndc(perspective: f32, focus_blur_px: f32) -> vec2f");
    expect(liveShaderSources.particleRender).toContain("fn particle_sprite_profile(local: vec2f, defocus: f32) -> f32");
    expect(liveShaderSources.particleRender).toContain("let focus_defocus = particle_focus_defocus(projected.z);");
    expect(liveShaderSources.particleRender).toContain("let focus_blur_px = particle_dof_blur_px_from_defocus(focus_defocus);");
    expect(liveShaderSources.particleRender).toContain("let splat_radius = particle_splat_radius_for_size_px(sized_px + max(0.0, focus_blur_px));");
    expect(liveShaderSources.particleRender).toContain("out.defocus = focus_defocus;");
    expect(liveShaderSources.particleRender).toContain("out.sprite_blur = particle_sprite_blur_amount(focus_blur_px, sized_px);");
    expect(liveShaderSources.particleRender).toContain("let glow = particle_sprite_profile(in.local, in.sprite_blur);");
    expect(liveShaderSources.particleRender).toContain("out.alpha = uniforms.particle_opacity * clamp(visual_color.w / 0.045, 0.0, 2.0) * particle_fade * large_motion_fade * variation.opacity_mul;");
    expect(liveShaderSources.particleRender).not.toContain("/ (1.0 + focus_blur_px");
  });
});

describe("particles at or behind the camera are clipped (bug #2)", () => {
  it("culls anything within the near plane or behind the viewer", () => {
    expect(isBehindCamera(-2)).toBe(true); // well behind the camera
    expect(isBehindCamera(0)).toBe(true); // exactly at the eye
    expect(isBehindCamera(PARTICLE_NEAR)).toBe(true); // on the near plane
  });

  it("keeps particles that are genuinely in front", () => {
    expect(isBehindCamera(PARTICLE_NEAR + 0.01)).toBe(false);
    expect(isBehindCamera(3.15)).toBe(false);
  });

  // The old projection clamped depth to 0.62 and never culled, so a point behind the camera
  // got pinned to a fixed positive perspective (~2.98) and "piled up" on screen. Guard against
  // that exact regression.
  it("does not pin behind-camera points to a fixed perspective", () => {
    expect(liveShaderSources.splatCommon).not.toContain("max(0.62,");
  });
});

// Mirror of the FOV->focal mapping used by the live renderer: focal_k = 1/tan(fov/2).
function focalK(fovDeg: number): number {
  return 1 / Math.tan((fovDeg * Math.PI) / 360);
}

describe("FOV drives the live renderer focal length", () => {
  it("matches the legacy 1.85*0.92 focal at the default FOV", () => {
    const defaultFov = (2 * Math.atan(1 / (1.85 * 0.92)) * 180) / Math.PI;
    expect(focalK(defaultFov)).toBeCloseTo(1.85 * 0.92, 5);
  });

  it("widening the FOV shortens the focal length (objects get smaller)", () => {
    expect(focalK(40)).toBeGreaterThan(focalK(61));
    expect(focalK(61)).toBeGreaterThan(focalK(100));
  });

  it("focal_k is the exact inverse of the vertical FOV relation", () => {
    // The renderer focal_k must satisfy fov = 2*atan(1/focal_k).
    for (const fov of [30, 61, 90, 110]) {
      const recovered = (2 * Math.atan(1 / focalK(fov)) * 180) / Math.PI;
      expect(recovered).toBeCloseTo(fov, 5);
    }
  });
});

function dofDebugTint(defocus: number): [number, number, number] {
  const blurT = smoothstep(0, 1, Math.min(1, Math.max(0, defocus)));
  const focus: [number, number, number] = [0.0, 1.0, 0.12];
  const blur: [number, number, number] = [1.0, 0.02, 0.0];
  return [0, 1, 2].map((i) => focus[i] * (1 - blurT) + blur[i] * blurT) as [number, number, number];
}

function dofTint(depth: number, focus: number, aperture: number, dofBlur: number, baseSizePx = 4.5): [number, number, number] {
  void dofBlur;
  void baseSizePx;
  return dofDebugTint(particleFocusDefocus(depth, focus, aperture));
}

describe("DOF debug tint visualizes the focal plane", () => {
  it("is green exactly at the focal plane", () => {
    const [r, g, b] = dofTint(0.5, 0.5, 0.6, 1);
    expect(g).toBeGreaterThan(r);
    expect(g).toBeGreaterThan(b);
  });
  it("tints red for particles nearer than focus when they are blurred", () => {
    const [r, g] = dofTint(0.95, 0.5, 0.9, 4); // depth > focus = nearer
    expect(r).toBeGreaterThan(g);
  });
  it("tints red for particles farther than focus when they are blurred", () => {
    const [r, g] = dofTint(0.05, 0.5, 0.9, 4); // depth < focus = farther
    expect(r).toBeGreaterThan(g);
  });
  it("saturates with aperture, not the DOF Blur pixel amount", () => {
    const [mildRed, mildGreen] = dofTint(0.6, 0.5, 0.3, 0);
    const [strongRed, strongGreen] = dofTint(0.6, 0.5, 0.9, 0);
    const [sameRed, sameGreen] = dofTint(0.6, 0.5, 0.9, 4);
    expect(strongRed).toBeGreaterThan(mildRed);
    expect(strongGreen).toBeLessThan(mildGreen);
    expect(sameRed).toBeCloseTo(strongRed, 5);
    expect(sameGreen).toBeCloseTo(strongGreen, 5);
  });
});

describe("shader constants match the verified perspective model", () => {
  it("declares the same near plane and reference depth the proof relies on", () => {
    expect(liveShaderSources.splatCommon).toContain("const PARTICLE_NEAR = 0.05;");
    expect(liveShaderSources.splatCommon).toContain("const DEFAULT_FOCAL_K = 1.702;");
    expect(liveShaderSources.splatCommon).toContain("const PARTICLE_SIZE_REF_DEPTH = 3.15;");
  });
});
