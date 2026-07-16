import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { projectWorldToNdc, rayForNdc, rayPointForProjectedWorld, worldToCamera } from "../cameraMath3d";
import { defaultGpuSim3dConfig, runCpuReference3d } from "../gpuSim3d";
import { computeRuleEvidence, defaultLiveGpu3dConfig, liveShaderSources } from "../realtimeGpuSim3d";
import { hdrCanvasColorSpace, hdrCanvasFormat, hdrCanvasToneMappingMode, wideGamut2dSettings } from "../webgpuCanvas";

describe("3D WebGPU compute reference", () => {
  it("keeps the CPU oracle deterministic for the compute conformance gate", () => {
    const config = { ...defaultGpuSim3dConfig, particleCount: 32, steps: 3 };
    const a = runCpuReference3d(config);
    const b = runCpuReference3d(config);
    expect(Array.from(a.field)).toEqual(Array.from(b.field));
    expect(Array.from(a.particles)).toEqual(Array.from(b.particles));
  });

  it("produces nonblank 3D field data before browser GPU execution", () => {
    const config = { ...defaultGpuSim3dConfig, particleCount: 40, steps: 2 };
    const state = runCpuReference3d(config);
    const fieldSum = state.field.reduce((total, value) => total + value, 0);
    const nonzero = state.field.filter((value) => value > 0.0001).length;
    expect(fieldSum).toBeGreaterThan(1);
    expect(nonzero).toBeGreaterThan(10);
  });

  it("generates stable cohort-specific Fluoddity Fourier rules for the live WGSL port", () => {
    const evidence = computeRuleEvidence(defaultLiveGpu3dConfig);
    expect(evidence.nonzero).toBe(true);
    expect(evidence.cohortDependent).toBe(true);
    expect(evidence.stable).toBe(true);
  });

  it("keeps particle projection and volume raymarch rays in the same camera space", () => {
    const camera = { yaw: 0.63, pitch: -0.41, distance: 3.15, aspect: 960 / 640 };
    const points: Array<readonly [number, number, number]> = [
      [-0.72, -0.45, 0],
      [-0.72, 0, 0],
      [0.72, -0.25, 0],
      [0.4, 0.3, -0.35],
      [-0.1, 0.7, 0.55]
    ];

    for (const point of points) {
      const projected = projectWorldToNdc(point, camera);
      const ray = rayForNdc([projected[0], projected[1]], camera);
      const reconstructed = rayPointForProjectedWorld(point, camera);
      const cameraPoint = worldToCamera(point, camera);

      expect(projected[0]).toBeGreaterThan(-1);
      expect(projected[0]).toBeLessThan(1);
      expect(projected[1]).toBeGreaterThan(-1);
      expect(projected[1]).toBeLessThan(1);
      expect(ray.origin.every(Number.isFinite)).toBe(true);
      expect(ray.direction.every(Number.isFinite)).toBe(true);
      expect(cameraPoint[2] + camera.distance).toBeGreaterThan(0);
      expect(distance(point, reconstructed)).toBeLessThan(1e-12);
    }
  });

  it("keeps fog controls wired without changing trail color by layer", () => {
    const source = readFileSync(new URL("../realtimeGpuSim3d.ts", import.meta.url), "utf8");
    const appSource = readFileSync(new URL("../App.tsx", import.meta.url), "utf8");

    expect(source).toContain('const useFogPipeline = controls.renderLayer === "both" || controls.renderLayer === "trails";');
    expect(source).toContain("const bloomEnabled = controls.bloomStrength > 0.0001;");
    expect(source).toContain("const mips = bloomEnabled ? this.bloomTextures : [];");
    expect(source).toContain("if (bloomEnabled && mips.length > 0) {");
    expect(source).not.toContain("particle-splats-high-density");
    expect(source).toContain('controls.renderLayer === "particles" ? "particle-splats"');
    expect(source).not.toContain("highDensityRenderControls");
    expect(source).not.toContain("isHighDensityParticleCount");
    expect(source).toContain("const fogScale = useFogPipeline ? clampNumber(controls.fogRenderScale, 0.25, 1) : 1;");
    expect(source).toContain("const effectiveFogSteps = useFogPipeline ? clampRayStepCount(controls.raySteps * controls.fogStepScale) : controls.raySteps;");
    expect(source).toContain("const useTemporalHistory = useFogPipeline && controls.fogTemporal && controls.fogTemporalBlend > 0;");
    expect(source).toContain("const useFieldTexture = controls.fieldTextureSampling && useFogPipeline && !visualFieldSmoothing;");
    expect(source).toContain('const usesFieldAlignedParticlePhase = controls.renderLayer !== "particles";');
    expect(source).toContain("usesFieldAlignedParticlePhase ? (visualFieldSmoothing ? this.renderLerpT : 0) : this.renderLerpT");
    expect(source).toContain("private shouldUseVisualFieldSmoothing(controls: RenderControls): boolean");
    expect(source).toContain('if (controls.renderLayer !== "particles")');
    expect(source).toContain("return controls.particleDensityCutoff > 0 || controls.particleDensityNormalize > 0.0001;");
    expect(source).toContain("const visualFieldPhaseReset = visualFieldSmoothing && simStepsThisFrame > 0;");
    expect(source).toContain("visualFieldSourceBufferIndexForFrame(this.readIndex, this.writeIndex, visualFieldPhaseReset)");
    expect(source).toContain("const visualFieldCorrection = 0;");
    expect(source).toContain("const particleSupportSourceBufferIndex = particleSourceBufferIndexForLerp(this.readIndex, particleRenderLerpT);");
    expect(source).not.toContain("previous_values");
    expect(source).not.toContain("field_lerp_t");
    expect(source).toContain("const fogNoiseFrame = controls.fogBlueNoise && effectiveFogTemporalBlend > 0 ? this.renderFrame : 0;");
    expect(source).toContain("fogFrame: fogNoiseFrame,");
    expect(source).toContain("fogBlueNoise: controls.fogBlueNoise ? 1 : 0,");
    expect(source).toContain("emptySpaceSkip: controls.emptySpaceSkipping ? 1 : 0,");
    expect(source).toContain("dofEnabled: controls.dofEnabled ? 1 : 0,");
    expect(source).toContain("let focus_defocus = particle_focus_defocus(projected.z);");
    expect(source).toContain("let focus_blur_px = particle_dof_blur_px_from_defocus(focus_defocus);");
    expect(source).toContain("let dof_focus_gain = 1.0 / (1.0 + abs(normalized_depth - uniforms.focus_distance) * uniforms.aperture * 4.0);");
    expect(source).toContain("let focus_gain = select(1.0, dof_focus_gain, uniforms.dof_enabled == 1u);");
    expect(source).toContain("let palette_anchor = palette(uniforms.palette, density_signal, confidence, 0.42 + confidence * 0.28);");
    expect(source).toContain("let visible_signal = smoothstep(0.018, 1.0, signal);");
    expect(source).toContain("fn saturate_color(color: vec3f, amount: f32) -> vec3f");
    expect(source).toContain("fn trail_thermal_color(raw: vec4f, signal: f32) -> vec3f");
    expect(source).toContain("fn trail_tint_color(signal: f32, flow_energy: f32) -> vec3f");
    expect(source).toContain("return apply_tint(color, particle_tint_color(), 0.28);");
    expect(source).toContain("fn particle_gradient_color(t: f32, mode_in: u32) -> vec3f");
    expect(source).toContain("fn scene_gain() -> f32");
    expect(source).toContain("return uniforms.exposure * scene_gain() * max(0.0, uniforms.particle_brightness);");
    expect(source).toContain("return uniforms.exposure * scene_gain() * max(0.0, uniforms.fog_brightness) * 3.0;");
    expect(source).toContain("f32[42] = uniforms.fogTint[0];");
    expect(source).toContain("f32[45] = uniforms.particleTint[0];");
    expect(source).toContain("f32[48] = uniforms.sceneBrightness;");
    expect(source).toContain("u32[51] = uniforms.particleColorMode;");
    expect(source).toContain("u32[52] = uniforms.particleVelocityStretch;");
    expect(source).toContain("f32[54] = uniforms.particleGradientSensitivity;");
    expect(source).toContain("u32[59] = uniforms.dofEnabled;");
    expect(source).toContain('particleSizeModel: "perspective-3d";');
    expect(source).toContain("fn particle_splat_base_size_px(perspective: f32) -> f32");
    expect(source).toContain("fn particle_splat_radius_ndc(perspective: f32) -> vec2f");
    expect(source).toContain("fn particle_dof_blur_px(depth: f32) -> f32");
    expect(source).toContain("fn particle_focus_defocus(depth: f32) -> f32");
    expect(source).toContain("return clamp(focus_error * max(0.0, uniforms.aperture) * 8.0, 0.0, 1.0);");
    expect(source).toContain("if (uniforms.dof_enabled == 0u)");
    expect(source).toContain("return clamp(defocus, 0.0, 1.0) * max(0.0, uniforms.dof_blur) * 8.0;");
    expect(source).toContain("fn particle_sprite_profile(local: vec2f, defocus: f32) -> f32");
    const splatRadiusBody = source.slice(source.indexOf("fn particle_splat_radius_ndc"), source.indexOf("fn particle_dof_blur_px"));
    // Splat base size is pure 1/view_depth perspective scaling with no screen-space min/max clamp;
    // particle DOF adds a bounded per-particle blur radius on top of that base.
    expect(source).toContain("return max(0.0, uniforms.particle_size_px) * perspective * (PARTICLE_SIZE_REF_DEPTH / 1.85) * (uniforms.focal_k / DEFAULT_FOCAL_K);");
    expect(source).not.toContain("uniforms.aperture * (8.0 + uniforms.dof_blur * 10.0)");
    expect(source).toContain("fn particle_dof_splat_radius_ndc(perspective: f32, focus_blur_px: f32) -> vec2f");
    expect(source).toContain("let splat_radius = particle_splat_radius_for_size_px(sized_px + max(0.0, focus_blur_px));");
    expect(source).toContain("let glow = particle_sprite_profile(in.local, in.sprite_blur);");
    expect(source).not.toContain("let size_px = clamp(scaled_px, min_px, max_px);");
    expect(splatRadiusBody).not.toContain("floor_px");
    expect(splatRadiusBody).not.toContain("particle_min_px");
    expect(splatRadiusBody).not.toContain("particle_max_px");
    expect(splatRadiusBody).not.toContain("clamp(");
    expect(appSource).not.toContain('displayMode === "cache"');
    expect(appSource).not.toContain('<Slider label="P Min Px"');
    expect(appSource).not.toContain('<Slider label="P Max Px"');
    // Particles at/behind the camera are clipped, not pinned to the viewport.
    expect(source).toContain("if (view_depth <= PARTICLE_NEAR) {");
    expect(source).toContain("let inv_depth = 1.0 / max(PARTICLE_NEAR, uniforms.distance + camera.z);");
    // FOV-driven focal: projection x,y scale with focal_k; size scales with focal_k too.
    expect(source).toContain("camera.x * uniforms.focal_k * inv_depth / aspect");
    expect(source).toContain("(uniforms.focal_k / DEFAULT_FOCAL_K)");
    // DOF debug tint is a pure visualization gated on the dof_debug uniform.
    expect(source).toContain("if (uniforms.dof_debug == 1u) {");
    expect(source).toContain("fn particle_velocity_screen_axis(velocity: vec3f) -> vec2f");
    expect(source).toContain("let axis = camera_velocity.xy * vec2f(aspect, 1.0);");
    expect(source).toContain("fn particle_velocity_stretch_amount(velocity: vec3f) -> f32");
    expect(source).toContain("let speed_t = smoothstep(0.0, max(0.000001, uniforms.particle_stretch_speed), length(velocity));");
    expect(source).toContain("fn particle_gradient_coordinate(velocity: vec3f, base: f32) -> f32");
    expect(source).toContain("render_local = direction * local.x * (1.0 + stretch) + perpendicular * local.y;");
    expect(source).toContain("f32[69] = uniforms.particleSlowCutoff;");
    expect(source).toContain("f32[85] = uniforms.particleStretchMin;");
    expect(source).toContain("f32[86] = uniforms.particleStretchSpeed;");
    expect(source).toContain("f32[87] = uniforms.particleSpeedCutoff;");
    expect(appSource).toContain('<Slider label="Stretch Min"');
    expect(appSource).toContain('<Slider label="Stretch Max"');
    expect(appSource).toContain('<Slider label="Stretch Speed"');
    expect(appSource).toContain('<Slider label="P Speed Cut"');
    expect(appSource).toContain('<Slider label="P Slow Cut"');
    expect(source).not.toContain("particle_velocity_stretch_strength");
    expect(source).not.toContain("perspective-scaled-pixels");
    expect(source).not.toContain("let projected_forward = project(position + velocity_step);");
    expect(source).not.toContain("let projected_backward = project(position - velocity_step);");
    expect(source).not.toContain("color = color / (vec3f(1.0) + color);");
    expect(source).not.toContain("PARTICLE_REFERENCE_PERSPECTIVE");
    expect(source).not.toContain("perspective_scale");
    expect(source).not.toContain("let neutral = vec3f");
    expect(source).not.toContain("trail_signal_boost");
    expect(source).not.toContain("trail_layer_boost");
    expect(source).not.toContain("trail_color_boost");
  });

  it("wires the per-particle variation + fractal noise system end to end", () => {
    const source = readFileSync(new URL("../realtimeGpuSim3d.ts", import.meta.url), "utf8");
    const appSource = readFileSync(new URL("../App.tsx", import.meta.url), "utf8");
    const controlsSource = readFileSync(new URL("../renderControls.ts", import.meta.url), "utf8");

    // Uniform buffer holds the variation block (88..111), domainShape (112), audio (113..115),
    // the color cross-fade fields (116..117), render-optimization toggles (118 = cutoff
    // prepass; 119..123 reserved), and particle haze/contrast controls (124..127).
    expect(source).toContain("const renderUniformByteLength = 512;");
    expect(source).toContain("u32[118] = uniforms.particleCutoffPrepass;");
    expect(source).toContain("f32[124] = Number.isFinite(uniforms.particleExponent) ? uniforms.particleExponent : 1;");
    expect(source).toContain("f32[125] = Number.isFinite(uniforms.particleBrightnessBoost) ? uniforms.particleBrightnessBoost : 1;");
    expect(source).toContain("f32[126] = Number.isFinite(uniforms.particleSupportSmoothing) ? uniforms.particleSupportSmoothing : 0;");
    expect(source).toContain("f32[127] = Number.isFinite(uniforms.particleHazeCull) ? uniforms.particleHazeCull : 0;");
    expect(source).toContain("f32[88] = uniforms.variationMaster;");
    expect(source).toContain("f32[89] = uniforms.variationTime;");
    expect(source).toContain("u32[93] = Math.max(1, Math.round(uniforms.variationOctaves));");
    expect(source).toContain("f32[111] = uniforms.variationColorMax;");
    expect(source).toContain("u32[112] = uniforms.domainShape;");
    // Time drift is fed from the smooth render clock, not the steppy sim timestep.
    expect(source).toContain("variationTime: this.renderFrame / 60,");

    // Noise lib + variation helper live in the shared splat chunk so both vertex shaders share them.
    const splatCommon = source.slice(source.indexOf("const liveSplatRenderCommon"), source.indexOf("const liveSplatFragmentShader"));
    expect(splatCommon).toContain("fn var_value_noise3(p: vec3f) -> f32");
    expect(splatCommon).toContain("fn var_fbm3(p_in: vec3f, octaves: u32, gain: f32, lacunarity: f32) -> f32");
    expect(splatCommon).toContain("const VARIATION_MAX_OCTAVES: u32 = 4u;");
    expect(splatCommon).toContain("fn compute_particle_variation(index: u32, pos: vec3f) -> ParticleVariation");
    expect(splatCommon).toContain("fn particle_variation_sized_px(base_size_px: f32, size_mul: f32) -> f32");
    // Disabled = neutral early-out (master <= 0 or no active amount), so the feature is ~free when off.
    expect(splatCommon).toContain("if (master <= 0.0 || any_amount <= 0.0) {");
    // Brightness rides a dedicated interpolant into the fragment stage.
    expect(splatCommon).toContain("@location(10) bright_mul: f32,");
    expect(source).toContain("fn particle_intensity(glow: f32, alpha: f32, bright_mul: f32) -> f32");
    expect(source).toContain("return glow * alpha * particle_gain() * bright_mul;");

    // Both vertex shaders compute the variation and consume every stream.
    const particleShader = source.slice(source.indexOf("const liveParticleVsCommon"), source.indexOf("const liveDensitySplatShader"));
    const variationCalls = particleShader.match(/let variation = compute_particle_variation\(particle_persistent_ids\[index\], interp_pos\);/g) ?? [];
    expect(variationCalls.length).toBe(3); // particle_vs, particle_fast_vs, and prepare_splats (variation keyed by persistent id, sort-invariant)
    expect(particleShader).toContain("out.bright_mul = variation.bright_mul * variation.val_mul;");
    expect(particleShader).toContain("let varied_hue = fract(particle_gradient_coordinate(visual_velocity, base_gradient) + variation.hue_off);");

    // Controls exist on the type, default to a no-op, and surface in the Variation panel.
    expect(controlsSource).toContain("variationMaster: number;");
    expect(controlsSource).toContain("variationColorMax: number;");
    expect(appSource).toContain("variationMaster: 1,");
    expect(appSource).toContain('<ControlGroup title="Variation">');
    expect(appSource).toContain('<Slider label="Var Master"');
    expect(appSource).toContain('<Slider label="Size Curve"');
    expect(appSource).toContain('"variation-size-amount-slider": "variationSizeAmount",');
  });

  it("keeps the failed particle filter controls and shader paths removed", () => {
    const source = readFileSync(new URL("../realtimeGpuSim3d.ts", import.meta.url), "utf8");
    const appSource = readFileSync(new URL("../App.tsx", import.meta.url), "utf8");
    const particleShader = source.slice(source.indexOf("const liveParticleVsCommon"), source.indexOf("const liveDensitySplatShader"));

    expect(source).toContain("const PARTICLE_CUTOFF_FADE");
    expect(source).toContain("density_fade = smoothstep(cutoff, cutoff * PARTICLE_CUTOFF_FADE, local_density);");
    expect(particleShader).toContain("let focus_defocus = particle_focus_defocus(projected.z);");
    expect(particleShader).toContain("let focus_blur_px = particle_dof_blur_px_from_defocus(focus_defocus);");
    expect(particleShader).toContain("let base_size_px = particle_splat_base_size_px(projected.z);");
    expect(particleShader).toContain("let splat_radius = particle_splat_radius_for_size_px(sized_px + max(0.0, focus_blur_px));");
    expect(particleShader).toContain("out.alpha = uniforms.particle_opacity * clamp(visual_color.w / 0.045, 0.0, 2.0) * particle_fade * large_motion_fade * variation.opacity_mul;");
    expect(particleShader).not.toContain("/ (1.0 + focus_blur_px");
    expect(particleShader).toContain("out.defocus = focus_defocus;");
    expect(particleShader).toContain("out.sprite_blur = particle_sprite_blur_amount(focus_blur_px, sized_px);");
    expect(particleShader).not.toContain("let core_stats = particle_density_stats");
    expect(particleShader).not.toContain("let haze_stats = particle_density_stats");
    expect(particleShader).not.toContain("let cutoff_stats = particle_density_stats");
    expect(source).not.toContain("select(0.0, max(0.0, uniforms.particle_structure_radius)");
    expect(source).not.toContain("particleSpeedFade");
    expect(source).not.toContain("particleStructureMask");
    expect(source).not.toContain("particleStructureRadius");
    expect(source).not.toContain("particleStructurePower");
    expect(source).not.toContain("particleSpeedCap");
    expect(source).not.toContain("particle_structure_mask");
    expect(source).not.toContain("particle_speed_fade");
    expect(source).not.toContain("particle_speed_cap");
    expect(source).not.toContain("liveParticleStructureMaskShader");
    expect(source).not.toContain("maskBuffer");
    expect(source).not.toContain("particleOccupancy");
    expect(source).not.toContain("structure_fade");
    expect(source).not.toContain("speed_fade");
    expect(appSource).toContain("particleDensityCutoff: { min: 0, max: 0.03, step: 0.0001 }");
    expect(appSource).toContain('<Slider label="P Cutoff"');
    expect(appSource).toContain('<Slider label="P Radius"');
    expect(appSource).toContain('<Slider label="P Keep"');
    expect(appSource).toContain('<Slider label="P Soft"');
    expect(appSource).toContain('<Slider label="P Speed Cut"');
    expect(appSource).toContain('<Slider label="P Slow Cut"');
    expect(appSource).toContain('<Slider label="Support"');
    expect(appSource).toContain('<Slider label="Supp Rad"');
    expect(appSource).toContain('<Slider label="Neighbors"');
    expect(appSource).toContain('<Slider label="Flow Agree"');
    expect(appSource).toContain('testId="particle-density-cutoff-slider"');
    expect(appSource).toContain('testId="particle-density-radius-slider"');
    expect(appSource).toContain('testId="particle-density-normalize-slider"');
    expect(appSource).toContain('testId="particle-density-softness-slider"');
    expect(appSource).toContain('testId="particle-speed-cutoff-slider"');
    expect(appSource).toContain('testId="particle-slow-cutoff-slider"');
    expect(appSource).toContain('testId="particle-support-mask-slider"');
    expect(appSource).toContain('testId="particle-support-radius-slider"');
    expect(appSource).toContain('testId="particle-support-neighbors-slider"');
    expect(appSource).toContain('testId="particle-support-flow-slider"');
    expect(appSource).not.toContain("particleSpeedFade");
    expect(appSource).not.toContain("particleStructureMask");
    expect(appSource).not.toContain("particleStructureRadius");
    expect(appSource).not.toContain("particleStructurePower");
    expect(appSource).not.toContain("particleSpeedCap");
    expect(appSource).not.toContain('<Slider label="Core Mask"');
    expect(appSource).not.toContain('<Slider label="Core Scale"');
    expect(appSource).not.toContain('<Slider label="Haze Reject"');
    expect(appSource).not.toContain('<Slider label="P Spd Fade"');
    expect(appSource).not.toContain('<Slider label="Speed Cap"');
    expect(appSource).not.toContain('testId="particle-structure-mask-slider"');
    expect(appSource).not.toContain('testId="particle-structure-radius-slider"');
    expect(appSource).not.toContain('testId="particle-structure-power-slider"');
    expect(appSource).not.toContain('testId="particle-speed-fade-slider"');
    expect(appSource).not.toContain('testId="particle-speed-cap-slider"');
    expect(appSource).not.toContain('const particleStructureMask = params.has("structureMask")');
  });

  it("adds render-only particle speed cutoffs with zero as the disabled value", () => {
    const source = readFileSync(new URL("../realtimeGpuSim3d.ts", import.meta.url), "utf8");
    const appSource = readFileSync(new URL("../App.tsx", import.meta.url), "utf8");
    const particleShader = source.slice(source.indexOf("const liveParticleVsCommon"), source.indexOf("const liveDensitySplatShader"));
    const densityShader = source.slice(source.indexOf("const liveDensitySplatShader"), source.indexOf("const liveDensityCompositeShader"));
    const accumulationShader = source.slice(source.indexOf("const liveAccumulationSplatShader"), source.indexOf("const liveAccumulationCompositeShader"));
    const volumeDensityShader = source.slice(source.indexOf("const liveVolumeDensityComputeShader"), source.indexOf("const liveVolumeDensityRenderShader"));

    expect(appSource).toContain("particleSpeedCutoff: { min: 0, max: 0.12, step: 0.001 }");
    expect(appSource).toContain("particleSlowCutoff: { min: 0, max: 0.12, step: 0.001 }");
    expect(appSource).toContain("particleSpeedCutoff: 0");
    expect(appSource).toContain("particleSlowCutoff: 0");
    expect(appSource).toContain('const particleSpeedCutoff = params.has("particleSpeedCutoff")');
    expect(appSource).toContain('const particleSlowCutoff = params.has("particleSlowCutoff")');
    expect(appSource).toContain('testId="particle-speed-cutoff-slider"');
    expect(appSource).toContain('testId="particle-slow-cutoff-slider"');
    expect(source).toContain("particleSpeedCutoff: controls.particleSpeedCutoff");
    expect(source).toContain("particleSlowCutoff: controls.particleSlowCutoff");
    expect(source).toContain("particleSpeedCutoff: number;");
    expect(source).toContain("particleSlowCutoff: number;");
    expect(source).toContain("particle_speed_cutoff: f32");
    expect(source).toContain("particle_slow_cutoff: f32");
    expect(source).toContain("controls.particleSpeedCutoff.toFixed(5)");
    expect(source).toContain("controls.particleSlowCutoff.toFixed(5)");
    expect(source).toContain("f32[69] = uniforms.particleSlowCutoff;");
    expect(source).toContain("f32[87] = uniforms.particleSpeedCutoff;");
    expect(source).toContain("fn particle_speed_cutoff_visibility(velocity: vec3f) -> f32");
    expect(source).toContain("let upper_cutoff = max(0.0, uniforms.particle_speed_cutoff);");
    expect(source).toContain("let lower_cutoff = max(0.0, uniforms.particle_slow_cutoff);");
    expect(source).toContain("if (upper_cutoff <= 0.0 && lower_cutoff <= 0.0) {");
    expect(source).toContain("let upper_visibility = 1.0 - smoothstep(upper_cutoff, upper_cutoff + upper_edge_width, speed);");
    expect(source).toContain("let lower_visibility = smoothstep(max(0.0, lower_cutoff - lower_edge_width), lower_cutoff, speed);");
    expect(particleShader).toContain("let speed_cutoff_visibility = particle_speed_cutoff_visibility(visual_velocity);");
    expect(particleShader).toContain("let particle_fade = density_fade * normalized_density_fade * support_fade * speed_cutoff_visibility * particle_haze_fade(interp_pos);");
    expect(densityShader).toContain("out.weight = density_gate * speed_cutoff_visibility * area_norm * large_motion_fade;");
    expect(accumulationShader).toContain("out.weight = max(0.0, uniforms.particle_opacity) * clamp(visual_color.w / 0.045, 0.0, 2.0) * accumulation_gate * speed_cutoff_visibility * area_norm * large_motion_fade;");
    expect(volumeDensityShader).toContain("let density_weight = max(0.0, uniforms.particle_opacity) * alpha_weight * density_fade * speed_cutoff_visibility * large_motion_fade");
    expect(source).not.toContain("particle_speed_cap");
    expect(appSource).not.toContain("particleSpeedCap");
  });

  it("restores live particle density cutoff and radius without the old core mask stack", () => {
    const source = readFileSync(new URL("../realtimeGpuSim3d.ts", import.meta.url), "utf8");
    const particleShader = source.slice(source.indexOf("const liveParticleVsCommon"), source.indexOf("const liveDensitySplatShader"));

    expect(particleShader).toContain("if (uniforms.particle_density_cutoff > 0.0) {");
    expect(source).toContain("const PARTICLE_CUTOFF_FADE = 2.25;");
    expect(source).toContain("fn particle_density_gate_radius() -> f32");
    expect(source).toContain("let voxel_radius = 2.0 / max(1.0, max_dim);");
    expect(source).toContain("return max(configured, voxel_radius * 1.5);");
    expect(particleShader).toContain("fn particle_cutoff_support_signal(position: vec3f, radius: f32) -> f32");
    expect(particleShader).toContain("let diagonal_step = radius * 0.57735027;");
    expect(particleShader).toContain("return center * 0.38 + axis * 0.28 + diagonal * 0.34;");
    expect(particleShader).toContain("local_density = particle_cutoff_support_signal(interp_pos, particle_density_gate_radius());");
    expect(particleShader).toContain("density_fade = smoothstep(cutoff, cutoff * PARTICLE_CUTOFF_FADE, local_density);");
    expect(particleShader).toContain("let particle_fade = density_fade * normalized_density_fade * support_fade * speed_cutoff_visibility * particle_haze_fade(interp_pos);");
    expect(particleShader).toContain("out.alpha = uniforms.particle_opacity * clamp(visual_color.w / 0.045, 0.0, 2.0) * particle_fade * large_motion_fade * variation.opacity_mul;");
    expect(particleShader).not.toContain("particle_structure_masks");
    expect(particleShader).not.toContain("particle_speed_fade_for_velocity");
  });

  it("adds a normalized particle density mask from the current field snapshot", () => {
    const source = readFileSync(new URL("../realtimeGpuSim3d.ts", import.meta.url), "utf8");
    const appSource = readFileSync(new URL("../App.tsx", import.meta.url), "utf8");
    const particleShader = source.slice(source.indexOf("const liveParticleVsCommon"), source.indexOf("const liveDensitySplatShader"));

    expect(appSource).toContain("particleDensityNormalize: { min: 0, max: 1, step: 0.01 }");
    expect(appSource).toContain("particleDensitySoftness: { min: 0.02, max: 1, step: 0.01 }");
    expect(appSource).toContain('<Slider label="P Keep"');
    expect(appSource).toContain('<Slider label="P Soft"');
    expect(appSource).toContain('const particleDensityNormalize = params.has("particleDensityNormalize")');
    expect(appSource).toContain('const particleDensitySoftness = params.has("particleDensitySoftness")');

    expect(source).toContain("maxDensitySignal: number;");
    expect(source).toContain("densitySignalMean: number;");
    expect(source).toContain("densitySignalP90: number;");
    expect(source).toContain("function densitySignalPercentile");
    expect(source).toContain("function particleDensityReferenceForStats");
    expect(source).toContain("const particleDensityReference = this.advanceParticleDensityReference();");
    expect(source).toContain("this.scheduleFieldStatsRead();");
    expect(source).not.toContain("await this.readFieldStats()");
    expect(source).toContain("particleDensityReferenceTarget: this.targetParticleDensityReference");
    expect(source).toContain("statsReadPending: this.fieldStatsReadPending");
    expect(source).toContain("f32[77] = uniforms.particleDensityReference;");
    expect(source).toContain("f32[78] = uniforms.particleDensityNormalize;");
    expect(source).toContain("f32[79] = uniforms.particleDensitySoftness;");
    expect(source).toContain("particle_density_reference: f32");
    expect(source).toContain("particle_density_normalize: f32");
    expect(source).toContain("particle_density_softness: f32");

    expect(particleShader).toContain("fn particle_normalized_density_fade(local_density: f32) -> f32");
    expect(particleShader).toContain("if (amount <= 0.0001 || uniforms.particle_density_reference <= 0.000001) {");
    expect(particleShader).toContain("let normalized_density = local_density / max(0.000001, uniforms.particle_density_reference);");
    expect(particleShader).toContain("return smoothstep(threshold, threshold + softness, normalized_density);");
    expect(particleShader).toContain("if (uniforms.particle_density_cutoff > 0.0 || uniforms.particle_density_normalize > 0.0001) {");
    expect(particleShader).toContain("let normalized_density_fade = particle_normalized_density_fade(local_density);");
    expect(particleShader).toContain("let particle_fade = density_fade * normalized_density_fade * support_fade * speed_cutoff_visibility * particle_haze_fade(interp_pos);");
  });

  it("adds a render-only particle support mask from a high-resolution occupancy grid", () => {
    const source = readFileSync(new URL("../realtimeGpuSim3d.ts", import.meta.url), "utf8");
    const appSource = readFileSync(new URL("../App.tsx", import.meta.url), "utf8");
    const particleShader = source.slice(source.indexOf("const liveParticleVsCommon"), source.indexOf("const liveDensitySplatShader"));
    const supportShader = source.slice(source.indexOf("const liveParticleSupportComputeShader"), source.indexOf("const liveSplatFragmentShader"));

    expect(appSource).toContain("particleSupportMask: { min: 0, max: 1, step: 0.01 }");
    expect(appSource).toContain("particleSupportRadius: { min: 0.35, max: 1.75, step: 0.05 }");
    expect(appSource).toContain("particleSupportNeighbors: { min: 1, max: 24, step: 1 }");
    expect(appSource).toContain("particleSupportFlow: { min: 0, max: 1, step: 0.01 }");
    expect(appSource).toContain("fastParticleRender: true");
    expect(appSource).toContain("fastNoBloomPost: true");
    expect(appSource).toContain('const fastParticleRender = parseBooleanParam(params.get("fastParticles") ?? params.get("fastParticleRender")) ?? baseControls.fastParticleRender;');
    expect(appSource).toContain('const fastNoBloomPost = parseBooleanParam(params.get("fastNoBloomPost") ?? params.get("fastPost")) ?? baseControls.fastNoBloomPost;');
    expect(appSource).toContain('const particleSupportMask = params.has("particleSupport")');
    expect(appSource).toContain('const particleSupportRadius = params.has("particleSupportRadius")');

    expect(source).toContain("const defaultParticleSupportGridSize = 128;");
    expect(source).toContain("function particleSupportGridSizeForLimits");
    expect(source).toContain("supportBuffer: GPUBuffer;");
    expect(source).toContain("activeIndexBuffer: GPUBuffer;");
    expect(source).toContain("activeDrawBuffer: GPUBuffer;");
    expect(source).toContain("private particleSupportGridBuffer?: GPUBuffer;");
    expect(source).toContain("private particleFastRenderPipelines: Partial<Record<ParticleBlendMode, GPURenderPipeline>> = {};");
    expect(source).toContain("const particleSupportIndirectDrawReset = new Uint32Array([4, 0, 0, 0]);");
    // Pipelines are built as deferred tasks (serial by default, Promise.all when
    // parallelPipelineCompile is on) rather than inline `await` assignments.
    expect(source).toContain("this.particleSupportClearPipeline = p;");
    expect(source).toContain("if (this.parallelPipelineCompile) {");
    expect(source).toContain("await Promise.all(tasks.map((task) => task()));");
    expect(source).toContain("entryPoint: \"clear_particle_support_grid\"");
    expect(source).toContain("entryPoint: \"build_particle_support_grid\"");
    expect(source).toContain("entryPoint: \"resolve_particle_support\"");
    expect(source).toContain("entryPoint: \"particle_fast_vs\"");
    expect(source).toContain('entryPoint: "composite_no_bloom_fs"');
    expect(source).toContain("fastParticleRender: this.usesFastParticlePipeline(controls)");
    expect(source).toContain("fastNoBloomPost: useNoBloomPost");
    expect(source).toContain("const useNoBloomPost = !bloomEnabled && controls.fastNoBloomPost;");
    expect(source).toContain("return controls.fastParticleRender &&");
    expect(source).toContain("primitive: { topology: \"triangle-strip\" }");
    expect(source).toContain("this.encodeParticleSupportMask(\n      encoder,");
    expect(source).toContain("useTimestamps && useParticleSupportTimings ? supportTimestampStart : undefined");
    expect(source).toContain("return controls.particleSupportMask > 0.0001 && this.usesParticleSplatLayer(controls) && !!this.particleSupportGridBuffer;");
    expect(source).toContain("pass.drawIndirect(chunk.activeDrawBuffer, 0);");
    expect(source).toContain("pass.draw(4, chunk.count);");
    expect(source).toContain("particleSupportClearMs: number;");
    expect(source).toContain("particleSupportBuildMs: number;");
    expect(source).toContain("particleSupportResolveMs: number;");
    expect(source).toContain("this.createParticleSupportBuildBindGroup(chunkIndex, particleSourceBufferIndex)");
    expect(source).toContain("this.createParticleSupportResolveBindGroup(chunkIndex, particleSourceBufferIndex)");
    expect(source).toContain("f32[80] = uniforms.particleSupportMask;");
    expect(source).toContain("f32[81] = uniforms.particleSupportRadius;");
    expect(source).toContain("f32[82] = uniforms.particleSupportNeighbors;");
    expect(source).toContain("f32[83] = uniforms.particleSupportFlow;");
    expect(source).toContain("u32[84] = Math.max(1, Math.round(uniforms.particleSupportGridSize));");

    expect(supportShader).toContain("@group(0) @binding(2) var<storage, read_write> particle_support_grid: array<atomic<i32>>;");
    expect(supportShader).toContain("@group(0) @binding(4) var<storage, read_write> particle_active_indices: array<u32>;");
    expect(supportShader).toContain("@group(0) @binding(5) var<storage, read_write> particle_active_draw: array<atomic<u32>>;");
    expect(supportShader).toContain("atomicAdd(&particle_support_grid[base + 0u], 1);");
    expect(supportShader).toContain("let self_count = select(0, 1, dx == 0i && dy == 0i && dz == 0i);");
    expect(supportShader).toContain("let neighbor_score = smoothstep(min_neighbors * 0.35, min_neighbors, neighbor_signal);");
    expect(supportShader).toContain("let flow_gate = mix(1.0, alignment_score, clamp(uniforms.particle_support_flow, 0.0, 1.0));");
    expect(supportShader).toContain("let slot = atomicAdd(&particle_active_draw[1], 1u);");
    expect(supportShader).toContain("particle_active_indices[slot] = i;");

    expect(particleShader).toContain("@group(0) @binding(4) var<storage, read> particle_support_values: array<f32>;");
    expect(particleShader).toContain("@group(0) @binding(5) var<storage, read> particle_active_indices: array<u32>;");
    expect(particleShader).toContain("fn particle_support_fade(index: u32, position: vec3f) -> f32");
    expect(particleShader).toContain("fn particle_instance_index(instance_index: u32) -> u32");
    expect(particleShader).toContain("return particle_active_indices[instance_index];");
    expect(particleShader).toContain("let cutoff = mix(0.05, 0.74, amount);");
    expect(particleShader).toContain("let softness = mix(0.34, 0.18, amount);");
    expect(particleShader).toContain("let support_fade = particle_support_fade(index, interp_pos);");
    expect(particleShader).toContain("let particle_fade = density_fade * normalized_density_fade * support_fade * speed_cutoff_visibility * particle_haze_fade(interp_pos);");
    expect(particleShader).toContain("fn particle_fast_vs");
  });

  it("keeps the obsolete particle cohesion mask removed from the render path", () => {
    const source = readFileSync(new URL("../realtimeGpuSim3d.ts", import.meta.url), "utf8");
    const appSource = readFileSync(new URL("../App.tsx", import.meta.url), "utf8");
    const particleShader = source.slice(source.indexOf("const liveParticleVsCommon"), source.indexOf("const liveDensitySplatShader"));
    const splatPrepareBindGroup = source.slice(source.indexOf("private createSplatPrepareBindGroup"), source.indexOf("private createSplatDrawBindGroup"));

    expect(appSource).not.toContain("particleCohesionMask");
    expect(appSource).not.toContain("particleCohesionRadius");
    expect(appSource).not.toContain("particleCohesionAlignment");
    expect(appSource).not.toContain("particleCohesionMemory");
    expect(appSource).not.toContain('<Slider label="Coh Size"');
    expect(appSource).not.toContain('testId="particle-cohesion-mask-slider"');
    expect(appSource).not.toContain('const particleCohesionMask = params.has("particleCohesion")');

    expect(source).not.toContain("particleCohesionMask");
    expect(source).not.toContain("particleCohesionRadius");
    expect(source).not.toContain("particleCohesionAlignment");
    expect(source).not.toContain("particleCohesionMemory");
    expect(source).not.toContain("particle_cohesion_mask");
    expect(source).not.toContain("particle_cohesion_radius");
    expect(source).not.toContain("particle_cohesion_fade");
    expect(source).not.toContain("cohesion_fade");
    expect(source).toContain("{ binding: 3, resource: { buffer: this.fieldBuffers[this.readIndex] } }");
    expect(splatPrepareBindGroup).toContain("{ binding: 3, resource: { buffer: this.fieldBuffers[this.readIndex] } }");
    expect(particleShader).toContain("@group(0) @binding(3) var<storage, read> particle_density_field_values: array<vec4f>;");
    expect(particleShader).toContain("let particle_fade = density_fade * normalized_density_fade * support_fade * speed_cutoff_visibility * particle_haze_fade(interp_pos);");
    expect(particleShader).toContain("out.alpha = uniforms.particle_opacity * clamp(visual_color.w / 0.045, 0.0, 2.0) * particle_fade * large_motion_fade * variation.opacity_mul;");
  });

  it("normalizes particle and density gradient colors by the live cohort count", () => {
    const source = readFileSync(new URL("../realtimeGpuSim3d.ts", import.meta.url), "utf8");

    expect(source).toContain("cohorts: number;");
    expect(source).toContain("cohorts: config.cohorts,");
    expect(source).toContain("u32[68] = Math.max(1, Math.round(uniforms.cohorts));");
    expect(source).toContain("cohorts: u32,");
    expect(source).toContain("let cohort_count = max(1.0, f32(uniforms.cohorts));");
    expect(source).toContain("let base_gradient = fract(cohort / max(1.0, f32(uniforms.cohorts)) + visual_color.x * 0.17);");
    expect(source).toContain("out.color = density_particle_color(visual_velocity, particle.pos_cohort.w, visual_color);");
    expect(source).not.toContain("f32(6u)");
    expect(source).not.toContain("max(1.0, 6.0)");
  });

  it("recycles sub-cutoff particles back into the pool without a sim speed cap", () => {
    const source = readFileSync(new URL("../realtimeGpuSim3d.ts", import.meta.url), "utf8");
    const appSource = readFileSync(new URL("../App.tsx", import.meta.url), "utf8");
    // Deterministic sim params live in SimConfig (so timeline replay stays in lockstep).
    expect(source).toContain("visual_lerp_t: f32");
    expect(source).toContain("recycle_cutoff: f32");
    expect(source).toContain("recycle_enabled: u32");
    // Recycle: particles in sub-cutoff (empty) regions reset back to the spawn distribution,
    // concentrating density in the active core instead of wandering the fringe.
    expect(source).toContain("if (config.recycle_enabled == 1u && config.recycle_cutoff > 0.0) {");
    expect(source).toContain("if (max(0.0, here.w + length(here.xyz) * 0.25) < config.recycle_cutoff) {");
    expect(source).not.toContain("if (config.particle_speed_cap > 0.0) {");
    expect(source).not.toContain("velocity = velocity * (config.particle_speed_cap / spd);");
    expect(source).toContain("f32[33] = clampNumber(visualLerpT, 0, 1);");
    expect(source).toContain("f32[34] = config.recycleCutoff;");
    expect(source).toContain("u32[35] = config.recycleEnabled ? 1 : 0;");
    expect(appSource).not.toContain("updateLiveConfig({ recycleCutoff: particleDensityCutoff })");
  });

  it("chunks live particle storage buffers instead of imposing a WebGPU particle cap", () => {
    const source = readFileSync(new URL("../realtimeGpuSim3d.ts", import.meta.url), "utf8");
    const limits = readFileSync(new URL("../particleLimits.ts", import.meta.url), "utf8");

    expect(limits).toContain("Number.MAX_SAFE_INTEGER");
    expect(source).toContain("type ParticleBufferChunk");
    expect(source).toContain("private particleChunks: ParticleBufferChunk[] = []");
    expect(source).toContain("function maxParticlesPerStorageBuffer");
    expect(source).toContain("const layout = this.particleRenderBindGroupLayout ?? pipeline?.getBindGroupLayout(0);");
    expect(source).toContain("particle_offset: u32");
    expect(source).toContain("active_particle_count: u32");
    expect(source).toContain("for (let chunkIndex = 0; chunkIndex < this.particleChunks.length; chunkIndex += 1)");
    expect(source).not.toContain("requiredDeviceLimitsForParticleBuffers");
    expect(source).not.toContain("deviceLimitsSupportParticleBuffers");
  });

  it("interpolates fractional-speed particles through the wrapped domain", () => {
    const source = readFileSync(new URL("../realtimeGpuSim3d.ts", import.meta.url), "utf8");
    expect(source).toContain("const PARTICLE_INTERP_SNAP_DISTANCE");
    expect(source).toContain("fn shortest_particle_delta(previous: vec3f, current: vec3f) -> vec3f");
    expect(source).toContain("fn particle_wrap_crossed(previous: vec3f, current: vec3f) -> bool");
    expect(source).toContain("delta.x = delta.x - 2.0;");
    expect(source).toContain("delta.x = delta.x + 2.0;");
    expect(source).toContain("fn interpolate_particle_position(previous: vec3f, current: vec3f, t: f32) -> vec3f");
    expect(source).toContain("fn particle_velocity_blend(previous: Particle, current: Particle, t: f32) -> vec3f");
    expect(source).toContain("fn particle_large_motion_fade(previous: vec3f, current: vec3f, requested_t: f32) -> f32");
    expect(source).toContain("return smoothstep(0.05, 0.85, t);");
    expect(source).toContain("if (particle_wrap_crossed(previous, current)) {");
    expect(source).toContain("motion_length <= PARTICLE_INTERP_SNAP_DISTANCE && !particle_wrap_crossed(previous, current)");
    expect(source).toContain("motion_length > PARTICLE_INTERP_SNAP_DISTANCE || particle_wrap_crossed(previous.pos_cohort.xyz, current.pos_cohort.xyz)");
    expect(source).toContain("return velocity;");
    expect(source).toContain("let particle_lerp_t = particle_lerp_amount(previous_particle.pos_cohort.xyz, particle.pos_cohort.xyz, uniforms.render_lerp_t);");
    expect(source).toContain("let large_motion_fade = particle_large_motion_fade(previous_particle.pos_cohort.xyz, particle.pos_cohort.xyz, uniforms.render_lerp_t);");
    expect(source).toContain("fn particle_fast_vs");
    expect(source).toContain("let interp_pos = interpolate_particle_position(previous_particle.pos_cohort.xyz, particle.pos_cohort.xyz, particle_lerp_t);");
    expect(source).not.toContain("let interp_pos = mix(particles_prev[index].pos_cohort.xyz, particle.pos_cohort.xyz, uniforms.render_lerp_t);");
  });

  it("uses the same snap rule for the render-only visual density field", () => {
    const source = liveShaderSources.visualDeposit;

    expect(source).toContain("const VISUAL_INTERP_SNAP_DISTANCE = 0.35;");
    expect(source).toContain("fn visual_wrap_crossed(previous: vec3f, current: vec3f) -> bool");
    expect(source).toContain("fn visual_lerp_amount(previous: vec3f, current: vec3f, requested_t: f32) -> f32");
    expect(source).toContain("if (visual_wrap_crossed(previous, current))");
    expect(source).toContain("length(visual_shortest_delta(previous, current)) > VISUAL_INTERP_SNAP_DISTANCE");
    expect(source).toContain("let visual_t = visual_lerp_amount(previous_particle.pos_cohort.xyz, particle.pos_cohort.xyz, config.visual_lerp_t);");
    expect(source).not.toContain("let visual_t = clamp(config.visual_lerp_t, 0.0, 1.0);");
  });

  it("samples the cutoff density trilinearly so the fade doesn't quantize into voxel cubes", () => {
    const source = readFileSync(new URL("../realtimeGpuSim3d.ts", import.meta.url), "utf8");
    // The cutoff sampler must blend the 8 surrounding voxels (like sample_field), not point-sample
    // one cell — otherwise the fade snaps at 64^3 grid boundaries and looks like cubes.
    expect(source).not.toContain("return particle_density_signal(particle_field_values[particle_field_index(position)]);");
    const body = source.slice(source.indexOf("fn density_particle_field_at"), source.indexOf("fn density_particle_field_at") + 1100);
    expect(body).toContain("let t = fract(grid);");
    expect(body).toContain("mix(1.0 - t.x, t.x, f32(dx)) * mix(1.0 - t.y, t.y, f32(dy)) * mix(1.0 - t.z, t.z, f32(dz))");
    expect(body).toContain("density_particle_field_signal(accum)");
  });

  it("renders a screen-space multi-scale density layer before bloom", () => {
    const source = readFileSync(new URL("../realtimeGpuSim3d.ts", import.meta.url), "utf8");
    const appSource = readFileSync(new URL("../App.tsx", import.meta.url), "utf8");

    expect(source).toContain("private densitySmallTexture?: GPUTexture;");
    expect(source).toContain("private densityLargeTexture?: GPUTexture;");
    expect(source).toContain("private densitySmallPipeline?: GPURenderPipeline;");
    expect(source).toContain("private densityLargePipeline?: GPURenderPipeline;");
    expect(source).toContain("private densityCompositePipeline?: GPURenderPipeline;");
    expect(source).toContain("private densitySplatBindGroups: GPUBindGroup[][][] = [];");
    expect(source).toContain("private densityCompositeBindGroup?: GPUBindGroup;");
    expect(source).toContain("densitySmallMs: number;");
    expect(source).toContain("densityLargeMs: number;");
    expect(source).toContain("densityCompositeMs: number;");
    expect(source).toContain("postMs: number;");
    expect(source).toContain('const useScreenDensityPipeline = controls.renderLayer === "both" || controls.renderLayer === "density";');
    expect(source).toContain("const useScreenDensityTimings = controls.densityPassStrength > 0 && useScreenDensityPipeline;");
    expect(source).toContain("const screenDensityQueryCount = useScreenDensityTimings ? 6 : 0;");
    expect(source).toContain("const supportQueryCount = useParticleSupportTimings ? 6 : 0;");
    expect(source).toContain("const postQueryCount = 2;");
    expect(source).toContain("const postTimestampStart = renderTimestampStart + renderQueryCount;");
    expect(source).toContain("const baseTimestampQueryCount = (stepSimulation ? 8 : 0) + supportQueryCount + renderQueryCount + postQueryCount;");
    expect(source).toContain("const fogCompositeMs = temporalFog && temporalHistory ? deltaMs(renderBase + 2, renderBase + 3) : 0;");
    expect(source).toContain("const fogPresentMs = temporalFog ? deltaMs(renderBase + (temporalHistory ? 4 : 2), renderBase + (temporalHistory ? 5 : 3)) : 0;");
    expect(source).toContain("const density = densityTimings(renderBase + (temporalFog ? (temporalHistory ? 6 : 4) : 2));");
    expect(source).toContain("const postMs = deltaMs(postBase, postBase + 1);");
    expect(source).toContain("timestampWrites: this.timestampWrites(postTimestampStart, postTimestampStart + 1)");
    expect(source).toContain("totalMeasuredMs: clearBrushMs + depositMs + particleUpdateMs + fieldUpdateMs + supportMs + renderMs + postMs");
    expect(source).toContain("this.renderDensityLayer(");
    expect(source.indexOf("this.renderDensityLayer(")).toBeLessThan(source.indexOf("const bloomEnabled = controls.bloomStrength > 0.0001;"));
    expect(source).toContain("targets: [{ format: densityTextureFormat, blend: additiveBlend }]");
    expect(source).toContain("entryPoint: \"density_small_vs\"");
    expect(source).toContain("entryPoint: \"density_large_vs\"");
    expect(source).toContain("entryPoint: \"density_composite_fs\"");
    expect(source).toContain('controls.renderLayer !== "particles" && controls.renderLayer !== "density" && controls.renderLayer !== "volume-density"');
    expect(source).toContain('!force && controls.renderLayer !== "both" && controls.renderLayer !== "density"');
    expect(source).toContain("let density_gate = smoothstep(density_gate_min, density_gate_max, local_density);");
    expect(source).toContain("return vec4f(in.color * glow, glow);");
    expect(source).toContain("let ridge_field = log2(1.0 + local_peak * 14.0 + edge_energy * 5.5);");
    expect(source).toContain("let support = smoothstep(uniforms.density_large_threshold, uniforms.density_large_threshold + 0.24, large_field);");
    expect(source).toContain("let filament = pow(1.0 - exp(-density_response), max(0.25, uniforms.density_emission_power));");
    expect(source).toContain("let emission = exp2(filament * emission_gain * 0.42) - 1.0;");
    expect(source).toContain("let volume_detail_scale = select(1.0, 0.72, uniforms.render_layer == 5u);");
    expect(source).toContain("let signal = emission * max(0.0, uniforms.density_pass_strength) * occlusion * volume_detail_scale;");
    expect(source).toContain("let density_color = max(vec3f(0.0), (small_sample.rgb + large_sample.rgb * 0.18) / color_weight);");
    expect(source).toContain("fn preserve_density_chroma");
    expect(source).toContain("let color = preserve_density_chroma(density_color, 0.32 + filament * 0.38);");
    expect(source).toContain("let occlusion = mix(1.0, 1.0 - fog.a * 0.72, clamp(uniforms.density_occlusion, 0.0, 1.0));");
    expect(source).toContain("this.densitySampler ??= this.device!.createSampler");
    expect(source).toContain("chunkBindGroups[this.readIndex] ??= this.device!.createBindGroup");
    expect(source).toContain("if (this.densityCompositeBindGroup && this.densityCompositeFogTexture === fogTexture)");
    expect(source).toContain("private renderDensityLayer(\n    encoder: GPUCommandEncoder,");
    expect(source).toContain("fieldBuffer = this.fieldBuffers[this.readIndex]");
    expect(source).toContain("this.drawDensitySplatPass(encoder, this.densitySmallPipeline!, this.densitySmallTexture!, timestampBase, fieldBuffer);");
    expect(source).toContain("this.drawDensitySplatPass(encoder, this.densityLargePipeline!, this.densityLargeTexture!, timestampBase !== undefined ? timestampBase + 2 : undefined, fieldBuffer);");
    expect(source).toContain("timestampWrites: this.timestampWrites(timestampBase + 4, timestampBase + 5)");

    expect(appSource).not.toContain('testId="density-small-scale-slider"');
    expect(appSource).not.toContain('<option value="density">Density</option>');
    expect(appSource).not.toContain('testId="density-large-scale-slider"');
    expect(appSource).not.toContain('testId="density-contrast-gain-slider"');
    expect(appSource).not.toContain('testId="density-emission-power-slider"');
    expect(appSource).not.toContain('testId="density-occlusion-slider"');
  });

  it("renders a color-preserving 3D volume-density mode before bloom", () => {
    const source = readFileSync(new URL("../realtimeGpuSim3d.ts", import.meta.url), "utf8");
    const appSource = readFileSync(new URL("../App.tsx", import.meta.url), "utf8");

    expect(source).toContain('const useVolumeDensityPipeline = controls.renderLayer === "volume-density";');
    expect(source).toContain("private volumeDensityAccumBuffer?: GPUBuffer;");
    expect(source).toContain("private volumeDensitySmallTexture?: GPUTexture;");
    expect(source).toContain("private volumeDensityLargeTexture?: GPUTexture;");
    expect(source).toContain("private volumeDensityDepositBindGroups: GPUBindGroup[][] = [];");
    expect(source).toContain('private volumeDensityPreparedKey = "";');
    expect(source).toContain("private volumeDensityPrepared = false;");
    expect(source).toContain("this.volumeDensityRenderBindGroup ??= this.device!.createBindGroup");
    expect(source).toContain("volumeDensityClearMs: number;");
    expect(source).toContain("volumeDensityRaymarchMs: number;");
    expect(source).toContain("postMs: number;");
    expect(source).toContain("const timestampQueryCount = 32;");
    expect(source).toContain("const useVolumeDensityTimings = useVolumeDensityPipeline && controls.densityPassStrength > 0;");
    expect(source).toContain("const rebuildVolumeDensity = useVolumeDensityPipeline && controls.densityPassStrength > 0 && this.shouldRebuildVolumeDensity(controls, stepSimulation);");
    expect(source).toContain("? (stepSimulation ? 14 : rebuildVolumeDensity ? 14 : 2)");
    expect(source).toContain("const renderTimestampStart = supportTimestampStart + supportQueryCount;");
    expect(source).toContain("const baseTimestampQueryCount = (stepSimulation ? 8 : 0) + supportQueryCount + renderQueryCount + postQueryCount;");
    expect(source).toContain("const volumeDensityBlurMs = deltaMs(renderBase + 6, renderBase + 7) + deltaMs(renderBase + 8, renderBase + 9) + deltaMs(renderBase + 10, renderBase + 11);");
    expect(source).toContain("private shouldRebuildVolumeDensity(controls: RenderControls, stepSimulation: boolean): boolean");
    expect(source).toContain("return this.volumeDensityPreparedKey !== this.volumeDensityCacheKey(controls);");
    expect(source).toContain("private volumeDensityCacheKey(controls: RenderControls): string");
    expect(source).toContain("this.volumeDensityPreparedKey = this.volumeDensityCacheKey(controls);");
    expect(source).toContain("this.volumeDensityPrepared = true;");
    expect(source).toContain("this.renderDensityLayer(encoder, controls, undefined, undefined, true);");
    expect(source).toContain("if (!volumeDensityRebuilt)");
    expect(source).toContain("entryPoint: \"deposit_volume_density\"");
    expect(source).toContain("entryPoint: \"resolve_volume_density\"");
    expect(source).toContain("entryPoint: \"blur_volume_density_x\"");
    expect(source).toContain("entryPoint: \"volume_density_fs\"");
    expect(source).toContain("texture_storage_3d<rgba16float, write>");
    expect(source).toContain("fn volume_particle_color");
    expect(source).toContain("@group(0) @binding(10) var<storage, read> volume_particle_field_values: array<vec4f>;");
    expect(source).toContain("{ binding: 10, resource: { buffer: this.fieldBuffers[this.readIndex] } }");
    expect(source).toContain("fn volume_particle_density_neighborhood");
    expect(source).toContain("density_fade = smoothstep(cutoff, cutoff * PARTICLE_CUTOFF_FADE, local_density);");
    expect(source).toContain("* density_fade * speed_cutoff_visibility * large_motion_fade * max(0.04, uniforms.density_small_scale)");
    expect(source).toContain("fn add_volume_density_trilinear");
    expect(source).toContain("const VOLUME_DENSITY_PARTICLE_BUDGET = 262144u;");
    expect(source).toContain("fn volume_atomic_add_saturating(slot: ptr<storage, atomic<i32>, read_write>, delta: i32)");
    expect(source).toContain("atomicCompareExchangeWeak(slot, old_value, next_value)");
    expect(source).toContain("return clamp(value, vec4f(0.0), vec4f(VOLUME_RESOLVE_MAX));");
    expect(source).toContain("let stride = volume_density_particle_stride();");
    expect(source).toContain("let source_i = i * stride + (chunk_config.particle_offset % stride);");
    expect(source).toContain("* count_normalizer * f32(stride) * 0.22");
    expect(source).toContain("let tri_weight = mix(1.0 - t.x, t.x, f32(dx)) * mix(1.0 - t.y, t.y, f32(dy)) * mix(1.0 - t.z, t.z, f32(dz));");
    expect(source).toContain("add_volume_density_flat(volume_flat_clamped(base.x + dx, base.y + dy, base.z + dz), color, weight * tri_weight);");
    expect(source).toContain("add_volume_density_trilinear(interp_pos, color, density_weight);");
    expect(source).toContain("textureSampleLevel(volume_small_texture, volume_density_sampler, uvw, 0.0)");
    expect(source).toContain("let large_color_mix = mix(0.04, 0.014, clamp(filament, 0.0, 1.0));");
    expect(source).toContain("let contrast_gate = smoothstep(0.045, 0.28 + ridge_mix * 0.18, core_peak);");
    expect(source).toContain("let structural_gate = contrast_gate * mix(0.72, 1.0, core_mask);");
    expect(source).toContain("body_field * 0.002");
    expect(source).toContain("let glow_gate = smoothstep(mix(0.08, 0.20, ridge_amount), mix(0.28, 0.56, ridge_amount), filament);");
    expect(source).toContain("let emission_drive = clamp(filament * max(0.0, uniforms.density_contrast_gain) * (0.30 + core_mask * 0.32), 0.0, 10.0);");
    expect(source).toContain("let curved_emission = exp2(emission_drive * mix(0.32, 0.48, core_mask)) - 1.0;");
    expect(source).toContain("let exponential_emission = curved_emission * glow_gate * (0.72 + core_mask * 0.28);");
    expect(source).toContain("let filament_signal = filament * (0.08 + core_mask * 0.30) * glow_gate;");
    expect(source).toContain("var dominant_color = vec3f(0.0);");
    expect(source).toContain("if (sample_signal > dominant_signal)");
    expect(source).toContain("let dominant_hue = dominant_color / dominant_peak;");
    expect(source).toContain("let final_radiance = mix(radiance, dominant_hue * radiance_peak, chroma_mix) * 0.08;");
    expect(source).toContain("return saturate_color(mix(tinted, hue_color, 0.58), 1.78);");
    expect(source).toContain("let color_sum = sample.small.rgb + sample.large.rgb * large_color_mix;");
    expect(source).toContain("let hue_color = color / peak;");
    expect(source).toContain("return saturate_color(chroma, 1.10 + filament * 0.46);");
    expect(source).toContain('controls.renderLayer === "volume-density" ? "volume-density-raymarch"');
    expect(source.indexOf("this.renderVolumeDensityLayer(")).toBeLessThan(source.indexOf("const bloomEnabled = controls.bloomStrength > 0.0001;"));

    expect(appSource).not.toContain('<option value="volume-density">Volume Density</option>');
    expect(appSource).not.toContain('function renderLayerForDisplayMode');
    expect(appSource).toContain('return { ...controls, renderLayer: "particles", ribbonFraction: 0 };');
    expect(appSource).toContain("function liveWebGpuConformance(live: LiveGpu3dDiagnostics | null): boolean");
    expect(appSource).toContain('live.renderMode === "volume-density-raymarch"');
    expect(appSource).toContain('const fieldStatsOk = live.renderMode === "particle-splats" || live.fieldStats.nonzeroVoxels > 0;');
  });

  it("renders a color-preserving accumulation mode as a distinct live path", () => {
    const source = readFileSync(new URL("../realtimeGpuSim3d.ts", import.meta.url), "utf8");
    const appSource = readFileSync(new URL("../App.tsx", import.meta.url), "utf8");

    expect(source).toContain('const useAccumulationPipeline = controls.renderLayer === "accumulation";');
    expect(source).toContain("const useAccumulationTimings = useAccumulationPipeline;");
    expect(source).toContain("private accumulationCurrentTexture?: GPUTexture;");
    expect(source).toContain("private accumulationHistoryTextures: GPUTexture[] = [];");
    expect(source).toContain("private accumulationHistoryPrepared = false;");
    expect(source).toContain("accumulationSplatMs: number;");
    expect(source).toContain("accumulationCompositeMs: number;");
    expect(source).toContain('entryPoint: "accumulation_vs"');
    expect(source).toContain('entryPoint: "accumulation_splat_fs"');
    expect(source).toContain('entryPoint: "accumulation_composite_fs"');
    expect(source).toContain("targets: [{ format: accumulationTextureFormat, blend: additiveBlend }]");
    expect(source).toContain("{ format: accumulationTextureFormat }");
    expect(source).toContain("this.renderAccumulationLayer(");
    expect(source).toContain("private renderAccumulationLayer(");
    expect(source).toContain("private drawAccumulationSplatPass(");
    expect(source).toContain("private createAccumulationSplatBindGroup(chunkIndex: number): GPUBindGroup");
    expect(source).toContain("private createAccumulationCompositeBindGroup(historyTexture: GPUTexture): GPUBindGroup");
    expect(source).toContain("private shouldResetAccumulationHistory(controls: RenderControls, width: number, height: number): boolean");
    expect(source).toContain("private accumulationHistoryCacheKey(controls: RenderControls, width: number, height: number): string");
    expect(source).toContain("const liveAccumulationSplatShader");
    expect(source).toContain("const liveAccumulationCompositeShader");
    expect(source).toContain("fn accumulation_particle_stats(position: vec3f, radius: f32) -> ParticleDensityStats");
    expect(source).toContain("let field_confidence = smoothstep(0.0002, 0.006, max(density_stats.peak, density_stats.mean));");
    expect(source).toContain("let density_gate = mix(mix(0.20, 0.10, reject), density_gate_raw, field_confidence);");
    expect(source).toContain("let accumulation_gate = density_gate * mix(0.45, support_term, 0.65) * ridge_term;");
    expect(source).toContain("out.weight = max(0.0, uniforms.particle_opacity) * clamp(visual_color.w / 0.045, 0.0, 2.0) * accumulation_gate * speed_cutoff_visibility * area_norm * large_motion_fade;");
    expect(source).toContain("return vec4f(in.color * glow, glow);");
    expect(source).toContain("fn accumulation_sample(uv: vec2f, memory: f32) -> vec4f");
    expect(source).toContain("let wide = wide_cardinal * 0.65 + wide_diagonal * 0.35;");
    expect(source).toContain("let support_gate = smoothstep(mix(0.10, 0.28, reject), mix(0.34, 0.70, reject), support_ratio) *");
    expect(source).toContain("let filament_excess = max(0.0, neighborhood.a - wide.a * mix(0.30, 0.56, reject));");
    expect(source).toContain("let coherent_gate = max(support_gate * ridge_gate, support_gate * smoothstep(0.001, 0.032, filament_excess) * 0.42);");
    expect(source).toContain("let coherent_energy = (filament_excess + neighborhood.a * coherent_gate * 0.24) * coherent_gate;");
    expect(source).toContain("let average_color = neighborhood.rgb / max(0.0001, neighborhood.a);");
    expect(source).toContain("let normalized = 1.0 - exp(-coherent_energy * max(0.0, uniforms.accumulation_strength) * 0.22);");
    expect(source).toContain("let hot_core = exp2(shaped * max(0.0, uniforms.accumulation_strength) * 0.46) - 1.0;");
    expect(source).toContain("fn preserve_accumulation_chroma");
    expect(source).toContain("let history_gate = clamp(max(coherent_gate, smoothstep(0.02, 0.24, coherent_energy) * 0.65), 0.0, 1.0);");
    expect(source).toContain("out.history = vec4f(clamp(neighborhood.rgb * history_gate, vec3f(0.0), vec3f(512.0)), clamp(neighborhood.a * history_gate, 0.0, 512.0));");
    expect(source).toContain('controls.renderLayer === "accumulation" ? "accumulation"');
    expect(source).toContain("f32[72] = uniforms.accumulationStrength;");
    expect(source).toContain("f32[73] = uniforms.accumulationRadius;");
    expect(source).toContain("f32[74] = uniforms.accumulationCurve;");
    expect(source).toContain("f32[75] = uniforms.accumulationMemory;");
    expect(source).toContain("f32[76] = uniforms.accumulationNoiseReject;");
    expect(source).toContain("accumulationSplat: liveAccumulationSplatShader");
    expect(source).toContain("accumulationComposite: liveAccumulationCompositeShader");
    expect(source.indexOf("this.renderAccumulationLayer(")).toBeLessThan(source.indexOf("const bloomEnabled = controls.bloomStrength > 0.0001;"));

    expect(appSource).not.toContain('<option value="accumulation">Accumulation</option>');
    expect(appSource).not.toContain('testId="accumulation-strength-slider"');
    expect(appSource).not.toContain('testId="accumulation-radius-slider"');
    expect(appSource).not.toContain('testId="accumulation-curve-slider"');
    expect(appSource).not.toContain('testId="accumulation-memory-slider"');
    expect(appSource).not.toContain('testId="accumulation-noise-reject-slider"');
    expect(appSource).toContain('const accumulationStrength = params.has("accumulationStrength")');
    expect(appSource).toContain('const accumulationNoiseReject = params.has("accumulationReject")');
  });

  it("restores every render control and live config knob through settings JSON and full capture manifests", () => {
    const appSource = readFileSync(new URL("../App.tsx", import.meta.url), "utf8");
    const controlsSource = readFileSync(new URL("../renderControls.ts", import.meta.url), "utf8");
    const liveSource = readFileSync(new URL("../realtimeGpuSim3d.ts", import.meta.url), "utf8");
    const controlsType = controlsSource.slice(
      controlsSource.indexOf("export type RenderControls = {"),
      controlsSource.indexOf("};", controlsSource.indexOf("export type RenderControls = {"))
    );
    const liveConfigType = liveSource.slice(
      liveSource.indexOf("export type LiveGpu3dConfig = {"),
      liveSource.indexOf("};", liveSource.indexOf("export type LiveGpu3dConfig = {"))
    );
    const controlKeys = Array.from(controlsType.matchAll(/^  ([A-Za-z0-9_]+):/gm), (match) => match[1]);
    const liveConfigKeys = Array.from(liveConfigType.matchAll(/^  ([A-Za-z0-9_]+):/gm), (match) => match[1]);
    const sanitizer = appSource.slice(
      appSource.indexOf("function sanitizeRenderControls"),
      appSource.indexOf("function sanitizeLiveConfig")
    );
    const liveConfigSanitizer = appSource.slice(
      appSource.indexOf("function sanitizeLiveConfig"),
      appSource.indexOf("function finiteNumber")
    );

    expect(controlKeys.length).toBeGreaterThan(0);
    for (const key of controlKeys) {
      if (key === "renderLayer") {
        expect(sanitizer).toContain('renderLayer: "particles"');
        continue;
      }
      expect(sanitizer, `sanitizeRenderControls should preserve ${key}`).toContain(`source.${key}`);
    }
    expect(liveConfigKeys.length).toBeGreaterThan(0);
    for (const key of liveConfigKeys) {
      expect(liveConfigSanitizer, `sanitizeLiveConfig should preserve ${key}`).toContain(`source.${key}`);
    }
    expect(appSource).toContain("const normalizedControls = normalizeRenderControlsForDisplayMode(");
    expect(appSource).toContain("renderControlsWithModulationRangeOverrides(sanitizeRenderControls(controls), controls, audio.sliders)");
    expect(appSource).toContain("controls: normalizedControls");
    expect(appSource).toContain("renderControlsWithModulationRangeOverrides(sanitizeRenderControls(controls), controls, currentSavedAudio.sliders)");
    expect(appSource).toContain("liveConfig: sanitizeLiveConfig(liveConfig)");
    expect(appSource).toContain('const displayMode: DisplayMode = "live";');
    expect(appSource).toContain("renderControls: sanitizeRenderControls(rendered)");
    expect(appSource).toContain('const bloomStrength = params.has("bloom")');
    expect(appSource).toContain('const bloomThreshold = params.has("bloomThreshold")');
    expect(appSource).toContain('const bloomRadius = params.has("bloomRadius")');
  });

  it("requests HDR canvas output without rendering a viewport frame-rate HUD", () => {
    const realtimeSource = readFileSync(new URL("../realtimeGpuSim3d.ts", import.meta.url), "utf8");
    const appSource = readFileSync(new URL("../App.tsx", import.meta.url), "utf8");
    const styles = readFileSync(new URL("../styles.css", import.meta.url), "utf8");

    expect(hdrCanvasFormat).toBe("rgba16float");
    expect(hdrCanvasColorSpace).toBe("display-p3");
    expect(hdrCanvasToneMappingMode).toBe("extended");
    expect(wideGamut2dSettings).toEqual({ alpha: false, colorSpace: "display-p3" });
    expect(realtimeSource).toContain("configureHdrWebGpuCanvas(this.context, this.device)");
    expect(appSource).not.toContain("viewport-hud");
    expect(appSource).not.toContain("fps-counter");
    expect(appSource).not.toContain("parseParticleColorMode");
    // Live stepping is now driven by the deterministic timeline transport (1 frame === 1
    // sim timestep): forward = advanceSteps, backward = reset + replay (planSeek).
    expect(appSource).toContain("const seek = planSeek(sim.currentTimestep, targetFrame);");
    expect(appSource).toContain("await sim.advanceSteps(canvasRef.current, currentConfig, seek.steps);");
    expect(appSource).toContain('testId="sim-speed-slider"');
    expect(appSource).toContain("const minCameraDistance = 0;");
    expect(appSource).toContain("const cameraDistanceSliderRange: SliderRange = { min: 0, max: 25, step: 0.001 };");
    expect(appSource).toContain("const maxCameraDistance = 100000;");
    expect(appSource).toContain("const zoomBase = camera.targetDistance <= minCameraDistance && delta > 0");
    expect(appSource).toContain("const nextDistance = clampCameraDistance(patch.distance ?? camera.targetDistance);");
    expect(realtimeSource).toContain("f32[8] = config.dt * scale;");
    expect(realtimeSource).toContain("f32[32] = scale;");
    expect(realtimeSource).toContain("fn sim_time_scale(config: SimConfig) -> f32");
    expect(realtimeSource).toContain("fn velocity_drag_factor(drag: f32, time_scale: f32) -> f32");
    expect(realtimeSource).toContain("let drag_factor = velocity_drag_factor(config.drag, time_scale);");
    expect(realtimeSource).toContain("var velocity = particle.vel_id.xyz * drag_factor + force * force_factor;");
    expect(realtimeSource).toContain("let persistence = pow(clamp(config.trail_persistence, 0.0, 0.999), time_scale);");
    expect(styles).not.toContain(".viewport-hud");
    expect(styles).toContain(".app-shell:fullscreen .viewport-stats");
    expect(styles).toContain(".app-shell:fullscreen .control-panel");
  });
});

function distance(a: readonly [number, number, number], b: readonly [number, number, number]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
