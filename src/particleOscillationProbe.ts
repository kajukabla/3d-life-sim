export type ParticleOscillationSample = {
  step: number;
  position: [number, number, number];
};

export type ParticleOscillationTrack = {
  index: number;
  id: number;
  cohort: number;
  score: number;
  reversalMean: number;
  stepDistanceMean: number;
  maxStepDistance: number;
  snapRiskFraction: number;
  wrapCrossingFraction: number;
  twoBackDistanceMean: number;
  netToPathRatio: number;
  samples: ParticleOscillationSample[];
};

export type ParticleCohortMotionSummary = {
  cohort: number;
  particleCount: number;
  stepDistanceMean: number;
  maxStepDistance: number;
  snapRiskFraction: number;
  snapRiskCount: number;
  wrapCrossingFraction: number;
  wrapCrossingCount: number;
};

export type ParticleOscillationReport = {
  frameCount: number;
  particleCount: number;
  analyzedParticles: number;
  snapDistance: number;
  top: ParticleOscillationTrack[];
  cohorts: ParticleCohortMotionSummary[];
};

export type ParticleOscillationAnalysisOptions = {
  particleCount?: number;
  stride?: number;
  topK?: number;
  minStepDistance?: number;
  sampleCount?: number;
  snapDistance?: number;
};

const defaultStride = 12;
const epsilon = 1e-9;

function finitePosition(frame: Float32Array, base: number): boolean {
  return Number.isFinite(frame[base]) && Number.isFinite(frame[base + 1]) && Number.isFinite(frame[base + 2]);
}

function shortestDelta(a: readonly [number, number, number], b: readonly [number, number, number]): [number, number, number] {
  let dx = b[0] - a[0];
  let dy = b[1] - a[1];
  let dz = b[2] - a[2];
  if (dx > 1) dx -= 2;
  else if (dx < -1) dx += 2;
  if (dy > 1) dy -= 2;
  else if (dy < -1) dy += 2;
  if (dz > 1) dz -= 2;
  else if (dz < -1) dz += 2;
  return [dx, dy, dz];
}

function crossesDomainWrap(a: readonly [number, number, number], b: readonly [number, number, number]): boolean {
  return Math.abs(b[0] - a[0]) > 1 || Math.abs(b[1] - a[1]) > 1 || Math.abs(b[2] - a[2]) > 1;
}

function length3(v: readonly [number, number, number]): number {
  return Math.hypot(v[0], v[1], v[2]);
}

function dot3(a: readonly [number, number, number], b: readonly [number, number, number]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function positionAt(frame: Float32Array, base: number): [number, number, number] {
  return [frame[base], frame[base + 1], frame[base + 2]];
}

function samplePositions(frames: readonly Float32Array[], particleIndex: number, stride: number, sampleCount: number): ParticleOscillationSample[] {
  const samples: ParticleOscillationSample[] = [];
  if (frames.length === 0) return samples;
  const interval = Math.max(1, Math.floor((frames.length - 1) / Math.max(1, sampleCount - 1)));
  for (let step = 0; step < frames.length; step += interval) {
    const base = particleIndex * stride;
    samples.push({ step, position: positionAt(frames[step], base) });
  }
  const lastStep = frames.length - 1;
  const lastBase = particleIndex * stride;
  if (samples.length === 0 || samples[samples.length - 1].step !== lastStep) {
    samples.push({ step: lastStep, position: positionAt(frames[lastStep], lastBase) });
  }
  return samples;
}

export function analyzeParticleOscillation(
  frames: readonly Float32Array[],
  options: ParticleOscillationAnalysisOptions = {}
): ParticleOscillationReport {
  const stride = Math.max(4, Math.floor(options.stride ?? defaultStride));
  const frameCount = frames.length;
  const frameParticleCounts = frames.map((frame) => Math.floor(frame.length / stride));
  const particleCount = Math.max(0, Math.min(options.particleCount ?? Infinity, ...frameParticleCounts));
  const topK = Math.max(1, Math.floor(options.topK ?? 12));
  const minStepDistance = Math.max(0, options.minStepDistance ?? 0.00002);
  const sampleCount = Math.max(2, Math.floor(options.sampleCount ?? 10));
  const snapDistance = Math.max(0, options.snapDistance ?? 0.35);
  const tracks: ParticleOscillationTrack[] = [];
  const cohortBuckets = new Map<number, {
    particleCount: number;
    stepDistanceSum: number;
    stepCount: number;
    maxStepDistance: number;
    snapRiskCount: number;
    wrapCrossingCount: number;
  }>();

  if (frameCount < 3 || particleCount <= 0) {
    return { frameCount, particleCount, analyzedParticles: 0, snapDistance, top: [], cohorts: [] };
  }

  for (let particleIndex = 0; particleIndex < particleCount; particleIndex += 1) {
    const base = particleIndex * stride;
    if (!frames.every((frame) => finitePosition(frame, base))) continue;

    let stepDistanceSum = 0;
    let twoBackDistanceSum = 0;
    let reversalSum = 0;
    let reversalCount = 0;
    let pathLength = 0;
    let usableSteps = 0;
    let maxStepDistance = 0;
    let snapRiskCount = 0;
    let wrapCrossingCount = 0;

    const first = positionAt(frames[0], base);
    const last = positionAt(frames[frameCount - 1], base);
    const cohort = Math.floor(frames[frameCount - 1][base + 3]);

    for (let step = 1; step < frameCount; step += 1) {
      const previous = positionAt(frames[step - 1], base);
      const current = positionAt(frames[step], base);
      const delta = shortestDelta(previous, current);
      if (crossesDomainWrap(previous, current)) wrapCrossingCount += 1;
      const stepDistance = length3(delta);
      stepDistanceSum += stepDistance;
      pathLength += stepDistance;
      usableSteps += 1;
      maxStepDistance = Math.max(maxStepDistance, stepDistance);
      if (stepDistance > snapDistance) snapRiskCount += 1;

      if (step >= 2) {
        const twoBack = positionAt(frames[step - 2], base);
        twoBackDistanceSum += length3(shortestDelta(twoBack, current));
        const priorDelta = shortestDelta(twoBack, previous);
        const priorLength = length3(priorDelta);
        if (stepDistance > epsilon && priorLength > epsilon) {
          reversalSum += Math.max(0, -dot3(delta, priorDelta) / (stepDistance * priorLength));
          reversalCount += 1;
        }
      }
    }

    const stepDistanceMean = stepDistanceSum / Math.max(1, usableSteps);
    const bucket = cohortBuckets.get(cohort) ?? {
      particleCount: 0,
      stepDistanceSum: 0,
      stepCount: 0,
      maxStepDistance: 0,
      snapRiskCount: 0,
      wrapCrossingCount: 0
    };
    bucket.particleCount += 1;
    bucket.stepDistanceSum += stepDistanceSum;
    bucket.stepCount += usableSteps;
    bucket.maxStepDistance = Math.max(bucket.maxStepDistance, maxStepDistance);
    bucket.snapRiskCount += snapRiskCount;
    bucket.wrapCrossingCount += wrapCrossingCount;
    cohortBuckets.set(cohort, bucket);
    if (stepDistanceMean < minStepDistance) continue;

    const twoBackDistanceMean = twoBackDistanceSum / Math.max(1, frameCount - 2);
    const reversalMean = reversalSum / Math.max(1, reversalCount);
    const netDistance = length3(shortestDelta(first, last));
    const netToPathRatio = netDistance / Math.max(epsilon, pathLength);
    const returnStrength = stepDistanceMean / Math.max(epsilon, twoBackDistanceMean);
    const score = reversalMean * returnStrength * (1 - Math.min(1, netToPathRatio));
    if (score <= 0) continue;

    tracks.push({
      index: particleIndex,
      id: frames[frameCount - 1][base + 7],
      cohort,
      score,
      reversalMean,
      stepDistanceMean,
      maxStepDistance,
      snapRiskFraction: snapRiskCount / Math.max(1, usableSteps),
      wrapCrossingFraction: wrapCrossingCount / Math.max(1, usableSteps),
      twoBackDistanceMean,
      netToPathRatio,
      samples: samplePositions(frames, particleIndex, stride, sampleCount)
    });
  }

  tracks.sort((a, b) => b.score - a.score);
  const cohorts = [...cohortBuckets.entries()].map(([cohort, bucket]) => ({
    cohort,
    particleCount: bucket.particleCount,
    stepDistanceMean: bucket.stepDistanceSum / Math.max(1, bucket.stepCount),
    maxStepDistance: bucket.maxStepDistance,
    snapRiskFraction: bucket.snapRiskCount / Math.max(1, bucket.stepCount),
    snapRiskCount: bucket.snapRiskCount,
    wrapCrossingFraction: bucket.wrapCrossingCount / Math.max(1, bucket.stepCount),
    wrapCrossingCount: bucket.wrapCrossingCount
  })).sort((a, b) => b.snapRiskFraction - a.snapRiskFraction || b.wrapCrossingFraction - a.wrapCrossingFraction || b.maxStepDistance - a.maxStepDistance);
  return {
    frameCount,
    particleCount,
    analyzedParticles: tracks.length,
    snapDistance,
    top: tracks.slice(0, topK),
    cohorts
  };
}
