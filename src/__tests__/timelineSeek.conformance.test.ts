import { describe, expect, it } from "vitest";
import { defaultGpuSim3dConfig, runCpuReference3d } from "../gpuSim3d";
import { planSeek } from "../timeline";

const base = { ...defaultGpuSim3dConfig, particleCount: 48 };

describe("deterministic seek conformance", () => {
  it("re-running to the same step is bit-identical (replay determinism)", () => {
    const a = runCpuReference3d({ ...base, steps: 25 });
    const b = runCpuReference3d({ ...base, steps: 25 });
    expect(Array.from(a.particles)).toEqual(Array.from(b.particles));
    expect(Array.from(a.field)).toEqual(Array.from(b.field));
  });

  it("semantic delta: state(N) differs from state(M) for N != M", () => {
    const early = runCpuReference3d({ ...base, steps: 8 });
    const late = runCpuReference3d({ ...base, steps: 32 });
    expect(Array.from(early.particles)).not.toEqual(Array.from(late.particles));
  });

  it("forward seek from a checkpoint equals a fresh replay to the target", () => {
    const plan = planSeek(8, 32);
    expect(plan).toEqual({ needsReset: false, steps: 24 });
    const fresh = runCpuReference3d({ ...base, steps: 32 });
    const replay = runCpuReference3d({ ...base, steps: 8 + plan.steps });
    expect(Array.from(replay.particles)).toEqual(Array.from(fresh.particles));
  });

  it("backward seek resets and replays from zero to the target", () => {
    const plan = planSeek(32, 8);
    expect(plan).toEqual({ needsReset: true, steps: 8 });
    const target = runCpuReference3d({ ...base, steps: plan.steps });
    const fresh = runCpuReference3d({ ...base, steps: 8 });
    expect(Array.from(target.particles)).toEqual(Array.from(fresh.particles));
  });
});
