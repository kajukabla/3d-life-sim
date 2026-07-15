export type SimSpeedLabSample = {
  frame: number;
  timestep: number;
  simulationTime: number;
  simulationTimeAdvance: number;
  renderLerpT: number;
  statsReadMs: number;
  statsReadPending: boolean;
  frameTimeMs: number;
  cpuFrameTimeMs: number;
  renderErrorEvents: number;
  meanLuma: number;
  lumaDelta: number;
  meanChannelDelta: number;
  maxChannelDelta: number;
  stepScales: number[];
};

export type SimSpeedLabEvent = {
  frame: number;
  timestep: number;
  kind: "flicker" | "stats-read" | "long-frame" | "render-error";
  value: number;
  renderLerpT: number;
};

export type SimSpeedLabReport = {
  sampleCount: number;
  frameSpan: number;
  tickSpan: number;
  zeroAdvanceFrames: number;
  wholeAdvanceFrames: number;
  statsReadEvents: number;
  renderErrorEvents: number;
  flickerEvents: number;
  longFrameEvents: number;
  phaseResetEvents: number;
  maxAbsLumaDelta: number;
  p95AbsLumaDelta: number;
  maxChannelDelta: number;
  avgFrameTimeMs: number;
  p95FrameTimeMs: number;
  maxFrameTimeMs: number;
  uniqueRenderLerpT: number[];
  uniqueStepScales: number[];
  likelyCadenceFrames: number | null;
  events: SimSpeedLabEvent[];
};

export type SimSpeedLabSnapshot = {
  speed: number;
  scene: string;
  layer: string;
  latest?: SimSpeedLabSample;
  report: SimSpeedLabReport;
  samples: SimSpeedLabSample[];
};

const emptyReport: SimSpeedLabReport = {
  sampleCount: 0,
  frameSpan: 0,
  tickSpan: 0,
  zeroAdvanceFrames: 0,
  wholeAdvanceFrames: 0,
  statsReadEvents: 0,
  renderErrorEvents: 0,
  flickerEvents: 0,
  longFrameEvents: 0,
  phaseResetEvents: 0,
  maxAbsLumaDelta: 0,
  p95AbsLumaDelta: 0,
  maxChannelDelta: 0,
  avgFrameTimeMs: 0,
  p95FrameTimeMs: 0,
  maxFrameTimeMs: 0,
  uniqueRenderLerpT: [],
  uniqueStepScales: [],
  likelyCadenceFrames: null,
  events: []
};

const warmupSampleCount = 6;

export function analyzeSimSpeedSamples(samples: readonly SimSpeedLabSample[]): SimSpeedLabReport {
  if (samples.length === 0) return emptyReport;
  const first = samples[0];
  const last = samples[samples.length - 1];
  const analysisStartIndex = samples.length > warmupSampleCount * 2 ? warmupSampleCount : 1;
  const steadySamples = samples.slice(analysisStartIndex);
  const absLumaDeltas = steadySamples.map((sample) => Math.abs(sample.lumaDelta));
  const frameTimes = steadySamples.map((sample) => sample.frameTimeMs);
  const maxAbsLumaDelta = maxValue(absLumaDeltas);
  const p95AbsLumaDelta = percentile(absLumaDeltas, 0.95);
  const maxFrameTimeMs = maxValue(frameTimes);
  const p95FrameTimeMs = percentile(frameTimes, 0.95);
  const avgFrameTimeMs = frameTimes.reduce((sum, value) => sum + value, 0) / Math.max(1, frameTimes.length);
  const baselineLumaDelta = Math.max(percentile(absLumaDeltas, 0.5) * 8, percentile(absLumaDeltas, 0.75) * 4);
  const flickerThreshold = Math.max(0.85, baselineLumaDelta);
  const longFrameThreshold = Math.max(12, p95FrameTimeMs * 2.5);
  const events: SimSpeedLabEvent[] = [];
  let phaseResetEvents = 0;

  for (let i = 1; i < samples.length; i += 1) {
    const sample = samples[i];
    const previous = samples[i - 1];
    const isWarmup = i < analysisStartIndex;
    const absDelta = Math.abs(sample.lumaDelta);
    if (sample.statsReadMs > 0 || sample.statsReadPending) {
      events.push(eventFor(sample, "stats-read", sample.statsReadMs));
    }
    if (!isWarmup && absDelta >= flickerThreshold) {
      events.push(eventFor(sample, "flicker", absDelta));
    }
    if (!isWarmup && sample.frameTimeMs >= longFrameThreshold) {
      events.push(eventFor(sample, "long-frame", sample.frameTimeMs));
    }
    if (sample.renderLerpT + 0.0001 < previous.renderLerpT && sample.simulationTimeAdvance > 0) {
      phaseResetEvents += 1;
    }
  }

  const flickerFrames = events.filter((event) => event.kind === "flicker").map((event) => event.frame);
  return {
    sampleCount: samples.length,
    frameSpan: last.frame - first.frame,
    tickSpan: last.timestep - first.timestep,
    zeroAdvanceFrames: samples.filter((sample) => sample.simulationTimeAdvance === 0).length,
    wholeAdvanceFrames: samples.filter((sample) => sample.simulationTimeAdvance >= 1).length,
    statsReadEvents: events.filter((event) => event.kind === "stats-read").length,
    renderErrorEvents: samples.reduce((sum, sample) => sum + sample.renderErrorEvents, 0),
    flickerEvents: events.filter((event) => event.kind === "flicker").length,
    longFrameEvents: events.filter((event) => event.kind === "long-frame").length,
    phaseResetEvents,
    maxAbsLumaDelta,
    p95AbsLumaDelta,
    maxChannelDelta: maxValue(samples.map((sample) => sample.maxChannelDelta)),
    avgFrameTimeMs,
    p95FrameTimeMs,
    maxFrameTimeMs,
    uniqueRenderLerpT: uniqueRounded(samples.map((sample) => sample.renderLerpT), 3).slice(0, 48),
    uniqueStepScales: uniqueRounded(samples.flatMap((sample) => sample.stepScales), 3),
    likelyCadenceFrames: likelyCadence(flickerFrames),
    events: events.slice(-64)
  };
}

function eventFor(sample: SimSpeedLabSample, kind: SimSpeedLabEvent["kind"], value: number): SimSpeedLabEvent {
  return {
    frame: sample.frame,
    timestep: sample.timestep,
    kind,
    value,
    renderLerpT: sample.renderLerpT
  };
}

function maxValue(values: readonly number[]): number {
  return values.length > 0 ? Math.max(...values) : 0;
}

function percentile(values: readonly number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * p)))];
}

function uniqueRounded(values: readonly number[], decimals: number): number[] {
  const scale = 10 ** decimals;
  return [...new Set(values.map((value) => Math.round(value * scale) / scale))];
}

function likelyCadence(frames: readonly number[]): number | null {
  if (frames.length < 3) return null;
  const gaps = frames.slice(1).map((frame, index) => frame - frames[index]).filter((gap) => gap > 0);
  if (gaps.length < 2) return null;
  const counts = new Map<number, number>();
  for (const gap of gaps) {
    const rounded = Math.max(1, Math.round(gap));
    counts.set(rounded, (counts.get(rounded) ?? 0) + 1);
  }
  let bestGap = 0;
  let bestCount = 0;
  for (const [gap, count] of counts) {
    if (count > bestCount) {
      bestGap = gap;
      bestCount = count;
    }
  }
  return bestCount >= 2 ? bestGap : null;
}
