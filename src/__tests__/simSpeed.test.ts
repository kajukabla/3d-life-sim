import { describe, expect, it } from "vitest";
import {
  liveShaderSources,
  MAX_SIMULATION_SPEED,
  particleSourceBufferIndexForLerp,
  planSimulationSteps,
  renderLerpTForSimulationSpeed,
  smoothParticleDensityReference,
  visualFieldSourceBufferIndexForFrame
} from "../realtimeGpuSim3d";

// Run the scheduler across `frames` frames at a constant speed and report how many
// authored whole simulation ticks fired in total.
function totalWholeStepTime(speed: number, frames: number): number {
  let accumulator = 0;
  let time = 0;
  for (let i = 0; i < frames; i += 1) {
    const plan = planSimulationSteps(accumulator, speed);
    accumulator = plan.accumulator;
    time += plan.timeAdvance;
  }
  return time;
}

function functionBody(source: string, signature: string): string {
  const start = source.indexOf(signature);
  expect(start, `${signature} should exist`).toBeGreaterThanOrEqual(0);
  const next = source.indexOf("\nfn ", start + signature.length);
  return source.slice(start, next === -1 ? undefined : next);
}

describe("simulation speed time scaling", () => {
  it("caps the speed range at 4x", () => {
    expect(MAX_SIMULATION_SPEED).toBe(4);
  });

  it("accumulates fractional speeds into whole authored physics ticks", () => {
    expect(planSimulationSteps(0, 0)).toEqual({ steps: 0, accumulator: 0, stepScales: [], timeAdvance: 0, renderLerpT: 1 });
    expect(planSimulationSteps(0, 0.05)).toEqual({ steps: 0, accumulator: 0.05, stepScales: [], timeAdvance: 0, renderLerpT: 0.05 });
    expect(planSimulationSteps(0.95, 0.05)).toEqual({ steps: 1, accumulator: 0, stepScales: [1], timeAdvance: 1, renderLerpT: 0 });
    expect(planSimulationSteps(0, 0.5)).toEqual({ steps: 0, accumulator: 0.5, stepScales: [], timeAdvance: 0, renderLerpT: 0.5 });
    expect(planSimulationSteps(0.5, 0.5)).toEqual({ steps: 1, accumulator: 0, stepScales: [1], timeAdvance: 1, renderLerpT: 0 });
    expect(planSimulationSteps(0, 1)).toEqual({ steps: 1, accumulator: 0, stepScales: [1], timeAdvance: 1, renderLerpT: 1 });
  });

  it("uses only whole steps above 1x and carries the fractional remainder", () => {
    expect(planSimulationSteps(0, 2)).toEqual({ steps: 2, accumulator: 0, stepScales: [1, 1], timeAdvance: 2, renderLerpT: 1 });
    expect(planSimulationSteps(0, 2.5)).toEqual({ steps: 2, accumulator: 0.5, stepScales: [1, 1], timeAdvance: 2, renderLerpT: 1 });
    expect(planSimulationSteps(0.5, 2.5)).toEqual({ steps: 3, accumulator: 0, stepScales: [1, 1, 1], timeAdvance: 3, renderLerpT: 1 });
    expect(planSimulationSteps(0, 4)).toEqual({ steps: 4, accumulator: 0, stepScales: [1, 1, 1, 1], timeAdvance: 4, renderLerpT: 1 });
  });

  it("never sends fractional step scales into the physics shader", () => {
    for (const speed of [0.1, 0.5, 0.82, 1, 2.5, 3.9, 4]) {
      for (let acc = 0; acc < 1; acc += 0.137) {
        const plan = planSimulationSteps(acc, speed);
        expect(plan.stepScales.every((scale) => scale === 1)).toBe(true);
        expect(plan.accumulator).toBeGreaterThanOrEqual(0);
        expect(plan.accumulator).toBeLessThan(1);
      }
    }
  });

  it("clamps speed to the maximum", () => {
    expect(planSimulationSteps(0, 9).steps).toBe(MAX_SIMULATION_SPEED);
    expect(planSimulationSteps(0, 9).timeAdvance).toBe(MAX_SIMULATION_SPEED);
    expect(planSimulationSteps(0, -3).steps).toBe(0);
  });

  it("averages whole-step time equal to the clamped speed", () => {
    const frames = 2000;
    for (const speed of [0, 0.25, 0.82, 1, 1.4, 2.5, 3.9, 4, 6]) {
      const expected = Math.min(MAX_SIMULATION_SPEED, Math.max(0, speed));
      expect(totalWholeStepTime(speed, frames) / frames).toBeCloseTo(expected, 2);
    }
  });

  it("uses low-speed accumulator phase for particle-only render interpolation", () => {
    expect(renderLerpTForSimulationSpeed(0, 0)).toBe(1);
    expect(renderLerpTForSimulationSpeed(-1, 0.25)).toBe(1);
    expect(renderLerpTForSimulationSpeed(0.5, 0.25)).toBe(0.25);
    expect(renderLerpTForSimulationSpeed(0.5, 1.25)).toBe(1);
    expect(renderLerpTForSimulationSpeed(1.5, 0.25)).toBe(1);
  });

  it("chooses the whole-step particle buffer that matches the rendered phase", () => {
    expect(particleSourceBufferIndexForLerp(0, 0)).toBe(1);
    expect(particleSourceBufferIndexForLerp(1, 0)).toBe(0);
    expect(particleSourceBufferIndexForLerp(0, 1)).toBe(0);
    expect(particleSourceBufferIndexForLerp(1, 1)).toBe(1);
  });

  it("resets low-speed visual fields from the previous sim field on step boundaries", () => {
    expect(visualFieldSourceBufferIndexForFrame(0, 1, false)).toBe(0);
    expect(visualFieldSourceBufferIndexForFrame(1, 0, false)).toBe(1);
    expect(visualFieldSourceBufferIndexForFrame(0, 1, true)).toBe(1);
    expect(visualFieldSourceBufferIndexForFrame(1, 0, true)).toBe(0);
  });

  it("smooths particle density reference updates instead of snapping brightness", () => {
    expect(smoothParticleDensityReference(0, 0.002)).toBe(0.002);
    expect(smoothParticleDensityReference(0.002, 0.004)).toBeCloseTo(0.00208, 8);
    expect(smoothParticleDensityReference(0.004, 0.002)).toBeCloseTo(0.00392, 8);
  });

});

describe("trail deposition is time-integrated once", () => {
  it("does not double-scale emission before field persistence applies time", () => {
    const deposit = functionBody(liveShaderSources.deposit, "fn deposit_particles");
    // Pulse adds a stateless brightness multiplier (* pulse); the guard below is the real intent:
    // emission must not be time-scaled here (field persistence applies time once, downstream).
    expect(deposit).toContain("let emission = config.deposit_mass * deposit_scale(config) * pulse;");
    expect(deposit).not.toContain("* sim_time_scale(config)");

    const field = functionBody(liveShaderSources.field, "fn update_field");
    expect(field).toContain("let persistence = pow(clamp(config.trail_persistence, 0.0, 0.999), time_scale);");
    expect(field).toContain("let mixed = canvas_color * persistence + brush_value * (1.0 - persistence);");
  });
});

describe("fractional velocity steps compose like full-speed physics", () => {
  it("uses exponential drag and the matching force integral for positive drag", () => {
    const update = functionBody(liveShaderSources.particleUpdate, "fn update_particles");
    expect(liveShaderSources.particleUpdate).toContain("fn velocity_drag_factor(drag: f32, time_scale: f32) -> f32");
    expect(liveShaderSources.particleUpdate).toContain("return pow(clamp(drag, 0.0, 1.0), time_scale);");
    expect(liveShaderSources.particleUpdate).toContain("return (1.0 - drag_factor) / max(0.000001, 1.0 - drag);");
    expect(update).toContain("let drag_factor = velocity_drag_factor(config.drag, time_scale);");
    expect(update).toContain("let force_factor = velocity_force_factor(config.drag, drag_factor, time_scale);");
    expect(update).toContain("var velocity = particle.vel_id.xyz * drag_factor + force * force_factor;");
    // The drag integration must not lerp velocity. (The initial-conditions morph legitimately uses
    // mix(particle.vel_id.xyz, target...) on a separate early-return path, so guard the assignment.)
    expect(update).not.toContain("velocity = mix(particle.vel_id.xyz");
  });
});

describe("particle coloring is view-independent", () => {
  it("derives the gradient from world-space velocity, not the camera projection", () => {
    const coordinate = functionBody(liveShaderSources.splatCommon, "fn particle_gradient_coordinate");
    // The color must not depend on how the camera is oriented. This is the airtight,
    // non-flaky proof of view-independence: the color path references no camera state.
    expect(coordinate).not.toContain("world_vector_to_camera");
    expect(coordinate).not.toContain("particle_velocity_screen_axis");
    expect(coordinate).not.toContain("camera");
    expect(coordinate).not.toContain("yaw_");
    expect(coordinate).not.toContain("pitch_");
    expect(coordinate).not.toContain("resolution");
    // It should map the raw world velocity direction to a hue.
    expect(coordinate).toContain("atan2(dir.z, dir.x)");
    expect(coordinate).toContain("let speed_band = smoothstep(0.00001, 0.035, speed);");
    expect(coordinate).toContain("let velocity_mix = smoothstep(0.0, 1.0, abs(sensitivity));");
  });

  it("feeds visual world velocity into the gradient at the call site", () => {
    expect(liveShaderSources.particleRender).toContain("let visual_velocity = particle_visual_velocity(previous_particle, particle, particle_lerp_t);");
    expect(liveShaderSources.particleRender).toContain("particle_gradient_coordinate(visual_velocity");
  });

  it("keeps the sprite stretch screen-aligned (orientation, not color)", () => {
    expect(liveShaderSources.particleRender).toContain("particle_velocity_screen_axis(visual_velocity)");
    expect(liveShaderSources.particleRender).toContain("let stretch = particle_velocity_stretch_amount(visual_velocity);");
    expect(liveShaderSources.particleRender).toContain("render_local = direction * local.x * (1.0 + stretch) + perpendicular * local.y;");
  });

  it("lets cohort coloring override behavior hue like upstream Fluoddity", () => {
    const update = functionBody(liveShaderSources.particleUpdate, "fn update_particles");
    expect(update).toContain("var hue = config.hue_sensitivity * color_params.x;");
    expect(update).toContain("hue = hash(vec2f(floor(cohort), floor(cohort)));");
    expect(update).not.toContain("cohort_hue + config.hue_sensitivity");
    expect(update).toContain("let hue_blend = clamp(time_scale * 0.08, 0.0, 1.0);");
    // Hue is packed into vel_id.w (the struct was shrunk 48->32B by dropping the color field).
    // `var` (not `let`) so the quorum-ignition emergent behavior can shift it toward the ignite hue.
    expect(update).toContain("var new_hue = mix(particle.vel_id.w, hue, hue_blend);");
  });
});

describe("3D sensor frame stability", () => {
  it("uses a continuous basis instead of a hard reference-axis swap", () => {
    const frame = functionBody(liveShaderSources.particleUpdate, "fn build_frame");
    expect(frame).toContain("let sign = select(-1.0, 1.0, forward.z >= 0.0);");
    expect(frame).toContain("let a = -1.0 / (sign + forward.z);");
    expect(frame).not.toContain("abs(dot(forward, reference)) > 0.88");
    expect(frame).not.toContain("reference = vec3f(1.0, 0.0, 0.0)");
  });

  it("prevents negative axial force from flipping velocity every step", () => {
    const guard = functionBody(liveShaderSources.particleUpdate, "fn guard_velocity_reversal");
    const update = functionBody(liveShaderSources.particleUpdate, "fn update_particles");
    expect(guard).toContain("let forward_speed = dot(candidate_velocity, previous_dir);");
    expect(guard).toContain("if (forward_speed >= 0.0)");
    expect(guard).toContain("sideways_velocity + previous_dir");
    expect(update).toContain("velocity = guard_velocity_reversal(particle.vel_id.xyz, velocity);");
    expect(update).toContain("let displacement = guard_velocity_reversal(");
    expect(update).toContain("particle.vel_id.xyz * config.dt");
    expect(update).toContain("velocity * config.dt + strafe * config.strafe_power * config.dt");
    expect(update).toContain("var position = particle.pos_cohort.xyz + displacement;");
    expect(update).not.toContain("var position = particle.pos_cohort.xyz + velocity * config.dt + strafe * config.strafe_power * config.dt;");
  });
});
