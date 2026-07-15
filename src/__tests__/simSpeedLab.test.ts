import { describe, expect, it } from "vitest";
import { analyzeSimSpeedSamples, type SimSpeedLabSample } from "../simSpeedLab";

function sample(frame: number, patch: Partial<SimSpeedLabSample> = {}): SimSpeedLabSample {
  return {
    frame,
    timestep: frame,
    simulationTime: frame,
    simulationTimeAdvance: 1,
    renderLerpT: 1,
    statsReadMs: 0,
    statsReadPending: false,
    frameTimeMs: 1,
    cpuFrameTimeMs: 1,
    renderErrorEvents: 0,
    meanLuma: frame * 0.01,
    lumaDelta: 0.01,
    meanChannelDelta: 0.2,
    maxChannelDelta: 1,
    stepScales: [1],
    ...patch
  };
}

describe("sim speed debug lab analysis", () => {
  it("summarizes low-speed whole ticks and render phases", () => {
    const samples = [
      sample(0, { timestep: 0, simulationTimeAdvance: 0, renderLerpT: 0.05, stepScales: [] }),
      sample(1, { timestep: 0, simulationTimeAdvance: 0, renderLerpT: 0.1, stepScales: [] }),
      sample(2, { timestep: 1, simulationTimeAdvance: 1, renderLerpT: 0, stepScales: [1] })
    ];

    const report = analyzeSimSpeedSamples(samples);

    expect(report.sampleCount).toBe(3);
    expect(report.frameSpan).toBe(2);
    expect(report.tickSpan).toBe(1);
    expect(report.zeroAdvanceFrames).toBe(2);
    expect(report.wholeAdvanceFrames).toBe(1);
    expect(report.uniqueRenderLerpT).toEqual([0.05, 0.1, 0]);
    expect(report.uniqueStepScales).toEqual([1]);
    expect(report.phaseResetEvents).toBe(1);
  });

  it("flags luminance spikes and estimates repeated cadence", () => {
    const samples = Array.from({ length: 24 }, (_, frame) => sample(frame, {
      timestep: frame,
      meanLuma: frame,
      lumaDelta: frame === 8 || frame === 13 || frame === 18 ? 8 : 0.05,
      maxChannelDelta: frame === 8 || frame === 13 || frame === 18 ? 28 : 1
    }));

    const report = analyzeSimSpeedSamples(samples);

    expect(report.flickerEvents).toBeGreaterThanOrEqual(3);
    expect(report.maxAbsLumaDelta).toBe(8);
    expect(report.maxChannelDelta).toBe(28);
    expect(report.likelyCadenceFrames).toBe(5);
  });

  it("tracks readback and frame-time events separately from flicker", () => {
    const samples = [
      sample(0),
      sample(1, { statsReadMs: 0.2, statsReadPending: true }),
      sample(2, { frameTimeMs: 40 })
    ];

    const report = analyzeSimSpeedSamples(samples);

    expect(report.statsReadEvents).toBe(1);
    expect(report.renderErrorEvents).toBe(0);
    expect(report.longFrameEvents).toBe(1);
    expect(report.events.map((event) => event.kind)).toContain("stats-read");
    expect(report.events.map((event) => event.kind)).toContain("long-frame");
  });
});
