import { describe, expect, it } from "vitest";
import { analyzeParticleOscillation } from "../particleOscillationProbe";

const stride = 12;

function frame(positions: Array<readonly [number, number, number]>, cohorts: readonly number[] = []): Float32Array {
  const values = new Float32Array(positions.length * stride);
  for (let i = 0; i < positions.length; i += 1) {
    const base = i * stride;
    values[base + 0] = positions[i][0];
    values[base + 1] = positions[i][1];
    values[base + 2] = positions[i][2];
    values[base + 3] = cohorts[i] ?? 0;
    values[base + 7] = i;
    values[base + 11] = 0.045;
  }
  return values;
}

describe("particle oscillation probe", () => {
  it("ranks a two-state particle above smooth one-way motion", () => {
    const frames = [
      frame([[0, 0, 0], [-0.4, 0, 0]]),
      frame([[0.08, 0, 0], [-0.36, 0, 0]]),
      frame([[0, 0, 0], [-0.32, 0, 0]]),
      frame([[0.08, 0, 0], [-0.28, 0, 0]]),
      frame([[0, 0, 0], [-0.24, 0, 0]]),
      frame([[0.08, 0, 0], [-0.2, 0, 0]])
    ];

    const report = analyzeParticleOscillation(frames, { particleCount: 2, topK: 2, minStepDistance: 0 });

    expect(report.top[0].index).toBe(0);
    expect(report.top[0].score).toBeGreaterThan(1000);
    expect(report.top.every((track) => track.index !== 1)).toBe(true);
  });

  it("handles wrapped domain motion using the shortest torus delta", () => {
    const frames = [
      frame([[0.98, 0, 0]]),
      frame([[-0.98, 0, 0]]),
      frame([[0.98, 0, 0]]),
      frame([[-0.98, 0, 0]])
    ];

    const report = analyzeParticleOscillation(frames, { particleCount: 1, topK: 1, minStepDistance: 0 });

    expect(report.top[0].index).toBe(0);
    expect(report.top[0].stepDistanceMean).toBeCloseTo(0.04, 5);
    expect(report.top[0].twoBackDistanceMean).toBeCloseTo(0, 5);
    expect(report.top[0].wrapCrossingFraction).toBe(1);
    expect(report.cohorts[0].wrapCrossingCount).toBe(3);
  });

  it("summarizes snap-risk motion by color cohort", () => {
    const cohorts = [20, 22, 3];
    const frames = [
      frame([[0, 0, 0], [0, 0, 0], [0, 0, 0]], cohorts),
      frame([[0.4, 0, 0], [0.1, 0, 0], [0.02, 0, 0]], cohorts),
      frame([[0.8, 0, 0], [0.2, 0, 0], [0.04, 0, 0]], cohorts)
    ];

    const report = analyzeParticleOscillation(frames, { particleCount: 3, topK: 3, minStepDistance: 0, snapDistance: 0.35 });
    const cohort20 = report.cohorts.find((cohort) => cohort.cohort === 20);
    const cohort22 = report.cohorts.find((cohort) => cohort.cohort === 22);

    expect(report.snapDistance).toBe(0.35);
    expect(report.cohorts[0].cohort).toBe(20);
    expect(cohort20?.snapRiskCount).toBe(2);
    expect(cohort20?.snapRiskFraction).toBe(1);
    expect(cohort20?.maxStepDistance).toBeCloseTo(0.4, 5);
    expect(cohort22?.snapRiskFraction).toBe(0);
    expect(cohort22?.wrapCrossingFraction).toBe(0);
  });
});
