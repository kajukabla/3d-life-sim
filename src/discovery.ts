import type { RenderControls } from "./renderControls";
import { getLivePreset, livePresets } from "./livePresets";
import { defaultLiveGpu3dConfig, type LiveGpu3dConfig, type LiveGpu3dDiagnostics } from "./realtimeGpuSim3d";
import { maxRayResolution } from "./renderTarget";

export type DiscoveryTemperature = "cold" | "workable" | "hot";
export type DiscoveryStage = "survey" | "mutate" | "harvest";

export type DiscoveryImageMetrics = {
  width: number;
  height: number;
  pixel_mean: number;
  pixel_variance: number;
  non_black_ratio: number;
  edge_sharpness: number;
  bounding_box?: [number, number, number, number] | null;
};

export type DiscoveryTemporalMetrics = {
  mean_absolute_error: number;
  ssim: number;
  alignment_error: number;
};

export type DiscoveryEvidence = {
  image?: DiscoveryImageMetrics | null;
  temporal?: DiscoveryTemporalMetrics | null;
  live?: Partial<LiveGpu3dDiagnostics> | null;
  fps?: number | null;
};

export type DiscoverySignals = {
  activity: number;
  persistence: number;
  novelty: number;
  cohesion: number;
  diversity: number;
  containment: number;
  recoverability: number;
};

export type DiscoveryScore = {
  interestingness: number;
  temperature: DiscoveryTemperature;
  signals: DiscoverySignals;
  reason: string;
};

export type DiscoveryCandidate = {
  version: 1;
  id: string;
  name: string;
  createdAt: string;
  stage: DiscoveryStage;
  generation: number;
  index: number;
  seed: number;
  parentId: string | null;
  presetId: string;
  mutationRate: number;
  controls: RenderControls;
  liveConfig: LiveGpu3dConfig;
  tags: string[];
  score: DiscoveryScore;
};

export type DiscoveryBatchOptions = {
  seed: number;
  count?: number;
  stage?: DiscoveryStage;
  generation?: number;
  basePresetId?: string;
  parent?: DiscoveryCandidate | null;
};

export type DiscoveryRunSummary = {
  count: number;
  workable: number;
  cold: number;
  hot: number;
  best: DiscoveryCandidate | null;
};

export const defaultDiscoverySeed = 20260528;
export const defaultDiscoveryBatchSize = 9;

export const defaultDiscoveryControls: RenderControls = {
  bloomStrength: 0.05,
  bloomThreshold: 0.6,
  bloomRadius: 1,
  colorSaturation: 1.16,
  colorContrast: 1.08,
  chromaticAberration: 0,
  vignetteStrength: 0,
  vignetteSoftness: 0.5,
  streakStrength: 0,
  streakLength: 0.6,
  streakVertical: 0,
  flareHeight: 0.3,
  flareCutoff: 0,
  ribbonFraction: 0,
  ribbonWidth: 0.6,
  ribbonTaper: 0.7,
  ribbonLength: 16,
  ribbonJoints: 16,
  ribbonFadeStart: 0.5,
  ribbonEdgeFade: 0.5,
  density: 1.08,
  exposure: 1.08,
  fov: (2 * Math.atan(1 / (1.85 * 0.92)) * 180) / Math.PI,
  aperture: 0.34,
  focusDistance: 0.54,
  dofBlur: 1,
  dofEnabled: false,
  dofDebug: false,
  sceneBrightness: 1,
  raySteps: 64,
  rayResolution: maxRayResolution,
  fogTemporal: true,
  fogRenderScale: 0.5,
  fogStepScale: 0.42,
  fogTemporalBlend: 0.78,
  fogBlueNoise: true,
  fieldTextureSampling: false,
  emptySpaceSkipping: false,
  emptySpaceThreshold: 0.035,
  emptySpaceStride: 4,
  particleSizePx: 3.2,
  particleMinPx: 0.45,
  particleMaxPx: 6.2,
  particleOpacity: 0.38,
  particleBrightness: 1,
  particleColorMode: "gradient-viridis",
  particleVelocityStretch: false,
  particleStretch: 1.4,
  particleStretchMin: 0,
  particleStretchSpeed: 0.035,
  particleSpeedCutoff: 0,
  particleSlowCutoff: 0,
  particleGlowCore: 0,
  particleHotCore: 0,
  particleExponent: 1,
  particleBrightnessBoost: 1,
  particleSupportSmoothing: 0,
  particleHazeCull: 0,
  particleDespeckle: 0,
  particleBlendMode: "additive",
  particleDensityCutoff: 0,
  particleDensityRadius: 0.04,
  particleDensityNormalize: 0,
  particleDensitySoftness: 0.22,
  particleSupportMask: 0,
  particleSupportRadius: 1.15,
  particleSupportNeighbors: 3,
  particleSupportFlow: 0.35,
  fastParticleRender: true,
  fastNoBloomPost: true,
  particleCutoffPrepass: false,
  densityLargeHalfRes: false,
  densityPassStrength: 0.85,
  densitySmallScale: 1.15,
  densityLargeScale: 7.5,
  densityLargeThreshold: 0.16,
  densityContrastGain: 3.2,
  densityContrastBalance: 0.34,
  densityEmissionPower: 1.55,
  densityOcclusion: 0.55,
  accumulationStrength: 3.6,
  accumulationRadius: 1.05,
  accumulationCurve: 2.2,
  accumulationMemory: 0.72,
  accumulationNoiseReject: 0.35,
  trailOpacity: 1.22,
  fogBrightness: 1,
  trailThreshold: 0,
  trailColorMode: "flow",
  fogTint: "#ffffff",
  particleTint: "#ffffff",
  renderLayer: "both",
  palette: "aurora",
  filament: 0.68,
  cameraYaw: 0.42,
  cameraPitch: -0.32,
  cameraDistance: 3.15,
  cameraPanX: 0,
  cameraPanY: 0,
  variationMaster: 0,
  variationDrift: 0.3,
  variationNoiseMix: 0.5,
  variationFreq: 2.5,
  variationOctaves: 2,
  variationGain: 0.5,
  variationLacunarity: 2,
  variationSizeAmount: 0,
  variationSizeCurve: 2.5,
  variationSizeMin: 0.3,
  variationSizeMax: 1.8,
  variationBrightAmount: 0,
  variationBrightCurve: 2,
  variationBrightMin: 0.4,
  variationBrightMax: 1.6,
  variationOpacityAmount: 0,
  variationOpacityCurve: 1.5,
  variationOpacityMin: 0.5,
  variationOpacityMax: 1.3,
  variationColorAmount: 0,
  variationColorCurve: 1,
  variationColorMin: -0.12,
  variationColorMax: 0.12
};

export function createDiscoveryBatch(options: DiscoveryBatchOptions): DiscoveryCandidate[] {
  const count = Math.max(1, Math.min(64, Math.round(options.count ?? defaultDiscoveryBatchSize)));
  const stage = options.stage ?? (options.parent ? "mutate" : "survey");
  const generation = options.generation ?? (options.parent ? options.parent.generation + 1 : 0);
  const basePresetId = options.parent?.presetId ?? options.basePresetId ?? livePresets[0].id;
  const rng = seededRng(options.seed + generation * 4099 + stageOffset(stage));
  const createdAt = new Date(0).toISOString();

  return Array.from({ length: count }, (_, index) => {
    const localSeed = hashNumber(options.seed, generation, index, Math.floor(rng() * 1_000_000));
    const localRng = seededRng(localSeed);
    const parent = options.parent ?? null;
    const mutationRate = parent
      ? clamp01(parent.mutationRate * randomRange(localRng, 0.72, 1.34) + randomRange(localRng, -0.025, 0.035))
      : randomRange(localRng, 0.035, 0.28);
    const preset = getLivePreset(basePresetId);
    const liveConfig = parent
      ? mutateLiveConfig(parent.liveConfig, mutationRate, localRng)
      : surveyLiveConfig(preset.config, localRng, count);
    const controls = parent
      ? mutateControls(parent.controls, mutationRate, localRng)
      : surveyControls(localRng);
    const score = scoreDiscoveryCandidate({ controls, liveConfig, presetId: preset.id, mutationRate });
    return {
      version: 1,
      id: candidateId(options.seed, generation, index, localSeed),
      name: `${stage === "survey" ? "Survey" : stage === "harvest" ? "Harvest" : "Branch"} ${generation}.${index + 1}`,
      createdAt,
      stage,
      generation,
      index,
      seed: localSeed,
      parentId: parent?.id ?? null,
      presetId: preset.id,
      mutationRate,
      controls,
      liveConfig,
      tags: tagsForScore(score),
      score
    };
  });
}

export function scoreDiscoveryCandidate(
  candidate: Pick<DiscoveryCandidate, "controls" | "liveConfig" | "presetId" | "mutationRate">,
  evidence: DiscoveryEvidence = {}
): DiscoveryScore {
  const config = candidate.liveConfig;
  const controls = candidate.controls;
  const fieldStats = evidence.live?.fieldStats;
  const voxelCount = evidence.live?.voxelCount ?? config.width * config.height * config.depth;
  const nonzeroRatio = fieldStats ? fieldStats.nonzeroVoxels / Math.max(1, voxelCount) : null;
  const flowPerParticle = fieldStats && evidence.live?.particleCount
    ? fieldStats.flowSum / Math.max(1, evidence.live.particleCount)
    : null;
  const image = evidence.image ?? null;
  const temporal = evidence.temporal ?? null;

  const staticActivity = clamp01(
    Math.abs(config.lateralForce) * 0.8 +
    Math.abs(config.axialForce) * 0.28 +
    config.globalForceMult * 0.36 +
    config.sensorGain * 0.055 +
    config.strafePower * 1.25 +
    config.mutationScale * 0.8 -
    Math.max(0, config.drag) * 0.1
  );
  const visualActivity = temporal
    ? clamp01(temporal.mean_absolute_error * 9.5 + Math.min(80, temporal.alignment_error) / 180)
    : null;
  const fieldActivity = flowPerParticle === null ? null : clamp01(flowPerParticle * 1100);
  const activity = blendSignals(staticActivity, visualActivity, fieldActivity);

  const persistence = clamp01(
    remap(config.trailPersistence, 0.88, 0.996) * 0.62 +
    remap(1 - config.trailDiffusion, 0, 1) * 0.18 +
    remap(controls.trailOpacity, 0.3, 2.2) * 0.2
  );
  const visualDensity = image ? image.non_black_ratio : nonzeroRatio;
  const cohesion = clamp01(
    (visualDensity === null || visualDensity === undefined ? 0.48 : bell(visualDensity, 0.04, 0.42)) * 0.48 +
    (image ? clamp01(image.edge_sharpness * 18) : 0.35) * 0.28 +
    remap(config.depositRadius, 0.004, 0.1) * 0.12 +
    clamp01(1 - Math.abs(config.axialForce) * 1.2) * 0.12
  );
  const diversity = clamp01(
    remap(config.cohorts, 1, 64) * 0.42 +
    remap(candidate.mutationRate, 0.02, 0.3) * 0.34 +
    (config.colorByCohort ? 0.12 : 0.04) +
    Math.min(0.12, Math.abs(config.hueSensitivity) * 0.16)
  );
  const containment = clamp01(
    0.92 -
    Math.max(0, Math.abs(config.axialForce) - Math.abs(config.lateralForce) * 0.55) * 0.7 -
    Math.max(0, config.globalForceMult - 1.05) * 0.35 -
    Math.max(0, config.hazardRate - 0.018) * 4 +
    Math.max(0, config.drag) * 0.06
  );
  const novelty = clamp01(
    diversity * 0.4 +
    cohesion * 0.22 +
    remap(Math.abs(config.sensorAngle), 0.04, 0.9) * 0.16 +
    remap(config.sensorDistance, 0.35, 2.4) * 0.12 +
    (image ? clamp01(image.pixel_variance * 34) * 0.1 : 0.05)
  );
  const recoverability = clamp01(
    persistence * 0.34 +
    containment * 0.24 +
    clamp01(1 - Math.abs(candidate.mutationRate - 0.12) * 3.4) * 0.24 +
    clamp01(1 - Math.max(0, config.hazardRate - 0.01) * 16) * 0.18
  );

  let temperature: DiscoveryTemperature = "workable";
  let reason = "balanced motion and inspectable structure";
  const tooBlank = image ? image.non_black_ratio < 0.012 || image.pixel_mean < 0.003 : false;
  const tooFull = image ? image.non_black_ratio > 0.88 || (image.non_black_ratio > 0.76 && image.pixel_variance < 0.032) : false;
  if (activity < 0.18 || tooBlank) {
    temperature = "cold";
    reason = "low motion or sparse visual signal";
  } else if ((activity > 0.86 && (cohesion < 0.38 || containment < 0.38)) || (tooFull && cohesion < 0.58)) {
    temperature = "hot";
    reason = "high motion with weak containment or diffuse structure";
  }

  const signals = { activity, persistence, novelty, cohesion, diversity, containment, recoverability };
  const interestingness = clamp01(
    activity * 0.18 +
    persistence * 0.15 +
    novelty * 0.2 +
    cohesion * 0.17 +
    diversity * 0.13 +
    containment * 0.1 +
    recoverability * 0.07
  );
  return {
    interestingness: temperature === "workable" ? interestingness : interestingness * 0.74,
    temperature,
    signals,
    reason
  };
}

export function summarizeDiscoveryRun(candidates: DiscoveryCandidate[]): DiscoveryRunSummary {
  const best = candidates.reduce<DiscoveryCandidate | null>((current, candidate) => {
    if (!current || candidate.score.interestingness > current.score.interestingness) return candidate;
    return current;
  }, null);
  return {
    count: candidates.length,
    workable: candidates.filter((candidate) => candidate.score.temperature === "workable").length,
    cold: candidates.filter((candidate) => candidate.score.temperature === "cold").length,
    hot: candidates.filter((candidate) => candidate.score.temperature === "hot").length,
    best
  };
}

export function discoveryCandidateSettingsName(candidate: DiscoveryCandidate): string {
  return `${candidate.name} ${Math.round(candidate.score.interestingness * 100)}`;
}

function surveyLiveConfig(base: LiveGpu3dConfig, rng: () => number, cohortBudget: number): LiveGpu3dConfig {
  const particleCount = cohortBudget >= 49 ? 65536 : 16384;
  const cohorts = randomChoice(rng, cohortBudget >= 49 ? [32, 48, 64] : [6, 9, 16, 25, 36, 64]);
  const config: LiveGpu3dConfig = {
    ...defaultLiveGpu3dConfig,
    ...base,
    seed: Math.floor(randomRange(rng, 1, 0xffffffff)),
    simulationSpeed: 1,
    width: 96,
    height: 96,
    depth: 96,
    particleCount,
    dt: 1,
    depositMass: randomRange(rng, 0.35, 1.65),
    depositRadius: randomRange(rng, 0.012, 0.085),
    trailPersistence: randomRange(rng, 0.9, 0.992),
    trailDiffusion: randomRange(rng, 0.18, 1),
    sensorGain: randomRange(rng, 0.45, 6.6),
    sensorAngle: randomRange(rng, -0.72, 0.72),
    sensorDistance: randomRange(rng, 0.45, 2.45),
    mutationScale: randomRange(rng, 0.02, 0.3),
    drag: randomRange(rng, -0.18, 0.68),
    axialForce: randomRange(rng, -0.08, 0.34),
    lateralForce: randomRange(rng, -0.86, 0.28),
    globalForceMult: randomRange(rng, 0.22, 1.35),
    strafePower: randomRange(rng, 0, 0.34),
    hazardRate: randomRange(rng, 0, 0.018),
    cohorts,
    ruleSeed: rng(),
    hueSensitivity: randomRange(rng, -0.72, 0.72),
    colorByCohort: rng() > 0.18,
    symmetryAxes: Math.floor(rng() * 8),
    absoluteOrientation: randomChoice(rng, [0, 1, 2, 3, 4, 5]) as LiveGpu3dConfig["absoluteOrientation"],
    orientationMix: randomRange(rng, 0, 1),
    initialConditions: randomChoice(rng, [0, 1, 2, 4, 5, 6, 7, 8]) as LiveGpu3dConfig["initialConditions"],
    boundaryMode: randomChoice(rng, [0, 2, 3, 5, 6]) as LiveGpu3dConfig["boundaryMode"]
  };
  return normalizeDiscoveryConfig(config);
}

function mutateLiveConfig(parent: LiveGpu3dConfig, amount: number, rng: () => number): LiveGpu3dConfig {
  const mutate = (value: number, spread: number, min: number, max: number) => clamp(value + randomRange(rng, -spread, spread) * (0.35 + amount * 2.8), min, max);
  const config: LiveGpu3dConfig = {
    ...parent,
    seed: Math.floor(randomRange(rng, 1, 0xffffffff)),
    depositMass: mutate(parent.depositMass, 0.28, 0.05, 2),
    depositRadius: mutate(parent.depositRadius, 0.018, 0.0015, 0.16),
    trailPersistence: mutate(parent.trailPersistence, 0.022, 0.85, 0.999),
    trailDiffusion: mutate(parent.trailDiffusion, 0.18, 0, 1),
    sensorGain: mutate(parent.sensorGain, 0.95, 0, 10),
    sensorAngle: mutate(parent.sensorAngle, 0.18, -1, 1),
    sensorDistance: mutate(parent.sensorDistance, 0.34, 0, 3),
    mutationScale: mutate(parent.mutationScale, 0.04, 0, 0.5),
    drag: mutate(parent.drag, 0.16, -0.5, 1),
    axialForce: mutate(parent.axialForce, 0.16, -1, 1),
    lateralForce: mutate(parent.lateralForce, 0.22, -1, 1),
    globalForceMult: mutate(parent.globalForceMult, 0.24, 0, 2),
    strafePower: mutate(parent.strafePower, 0.06, 0, 0.5),
    hazardRate: mutate(parent.hazardRate, 0.006, 0, 0.05),
    cohorts: Math.round(clamp(parent.cohorts + randomRange(rng, -5, 5), 1, 128)),
    ruleSeed: rng() > 0.72 ? rng() : parent.ruleSeed + randomRange(rng, -0.035, 0.035),
    hueSensitivity: mutate(parent.hueSensitivity, 0.16, -1, 1),
    orientationMix: mutate(parent.orientationMix, 0.18, 0, 1),
    colorByCohort: rng() > 0.08 ? parent.colorByCohort : !parent.colorByCohort,
    symmetryAxes: rng() > 0.12 ? parent.symmetryAxes : Math.floor(rng() * 8)
  };
  return normalizeDiscoveryConfig(config);
}

function surveyControls(rng: () => number): RenderControls {
  const particleStretch = randomRange(rng, 0.45, 3.8);
  const particleStretchMin = randomRange(rng, 0, Math.min(1.1, particleStretch * 0.45));
  return {
    ...defaultDiscoveryControls,
    density: randomRange(rng, 0.75, 1.55),
    exposure: randomRange(rng, 0.85, 1.55),
    sceneBrightness: randomRange(rng, 0.72, 1.65),
    aperture: randomRange(rng, 0.2, 0.62),
    particleSizePx: randomRange(rng, 0.6, 5.8),
    particleOpacity: randomRange(rng, 0.16, 0.58),
    particleBrightness: randomRange(rng, 0.65, 2.15),
    particleColorMode: randomChoice(rng, ["solid", "gradient-inferno", "gradient-magma", "gradient-viridis", "gradient-turbo", "gradient-rainbow", "gradient-spectral"] as const),
    particleVelocityStretch: rng() > 0.58,
    particleStretch,
    particleStretchMin,
    particleStretchSpeed: randomRange(rng, 0.012, 0.075),
    particleDensityCutoff: randomRange(rng, 0, 0.012),
    particleDensityRadius: randomRange(rng, 0.02, 0.08),
    particleDensityNormalize: randomRange(rng, 0, 0.8),
    particleDensitySoftness: randomRange(rng, 0.08, 0.45),
    particleSupportMask: randomRange(rng, 0, 0.85),
    particleSupportRadius: randomRange(rng, 0.55, 1.55),
    particleSupportNeighbors: Math.round(randomRange(rng, 2, 12)),
    particleSupportFlow: randomRange(rng, 0, 0.8),
    densityPassStrength: randomRange(rng, 0.35, 4.8),
    densitySmallScale: randomRange(rng, 0.65, 2.4),
    densityLargeScale: randomRange(rng, 4.5, 24),
    densityLargeThreshold: randomRange(rng, 0.05, 2.4),
    densityContrastGain: randomRange(rng, 0.1, 8),
    densityContrastBalance: randomRange(rng, 0.18, 0.72),
    densityEmissionPower: randomRange(rng, 0.85, 2.7),
    densityOcclusion: randomRange(rng, 0.2, 0.85),
    accumulationStrength: randomRange(rng, 1.6, 7.5),
    accumulationRadius: randomRange(rng, 0.45, 2.2),
    accumulationCurve: randomRange(rng, 1.2, 4.2),
    accumulationMemory: randomRange(rng, 0.25, 0.88),
    accumulationNoiseReject: randomRange(rng, 0.08, 0.72),
    trailOpacity: randomRange(rng, 0.75, 1.9),
    fogBrightness: randomRange(rng, 0.65, 2.1),
    trailThreshold: randomRange(rng, 0, 0.035),
    trailColorMode: randomChoice(rng, ["stable", "flow", "thermal", "tint"]),
    renderLayer: randomChoice(rng, ["both", "density", "volume-density", "accumulation"] as const),
    palette: randomChoice(rng, ["aurora", "ember", "spectral"]),
    filament: randomRange(rng, 0.35, 0.92),
    cameraYaw: randomRange(rng, -Math.PI, Math.PI),
    cameraPitch: randomRange(rng, -0.72, 0.28),
    cameraDistance: randomRange(rng, 2.15, 4.2)
  };
}

function mutateControls(parent: RenderControls, amount: number, rng: () => number): RenderControls {
  const mutate = (value: number, spread: number, min: number, max: number) => clamp(value + randomRange(rng, -spread, spread) * (0.35 + amount * 2.5), min, max);
  const particleStretch = mutate(parent.particleStretch, 0.55, 0, 6);
  const particleStretchMin = Math.min(particleStretch, mutate(parent.particleStretchMin, 0.18, 0, 6));
  return {
    ...parent,
    density: mutate(parent.density, 0.18, 0.25, 2.8),
    exposure: mutate(parent.exposure, 0.16, 0.4, 2.6),
    sceneBrightness: mutate(parent.sceneBrightness, 0.18, 0, 2.6),
    aperture: mutate(parent.aperture, 0.08, 0.05, 0.9),
    particleSizePx: mutate(parent.particleSizePx, 0.8, 0.2, 24),
    particleOpacity: mutate(parent.particleOpacity, 0.08, 0, 1),
    particleBrightness: mutate(parent.particleBrightness, 0.24, 0, 8),
    particleVelocityStretch: rng() > 0.12 ? parent.particleVelocityStretch : !parent.particleVelocityStretch,
    particleStretch,
    particleStretchMin,
    particleStretchSpeed: mutate(parent.particleStretchSpeed, 0.008, 0.004, 0.12),
    particleDensityCutoff: mutate(parent.particleDensityCutoff, 0.003, 0, 0.012),
    particleDensityRadius: mutate(parent.particleDensityRadius, 0.012, 0, 0.18),
    particleDensityNormalize: mutate(parent.particleDensityNormalize, 0.14, 0, 1),
    particleDensitySoftness: mutate(parent.particleDensitySoftness, 0.08, 0.02, 1),
    particleSupportMask: mutate(parent.particleSupportMask, 0.16, 0, 1),
    particleSupportRadius: mutate(parent.particleSupportRadius, 0.18, 0.35, 1.75),
    particleSupportNeighbors: Math.round(mutate(parent.particleSupportNeighbors, 2.2, 1, 24)),
    particleSupportFlow: mutate(parent.particleSupportFlow, 0.16, 0, 1),
    densityPassStrength: mutate(parent.densityPassStrength, 0.7, 0, 8),
    densitySmallScale: mutate(parent.densitySmallScale, 0.32, 0.25, 5),
    densityLargeScale: mutate(parent.densityLargeScale, 2.2, 1, 36),
    densityLargeThreshold: mutate(parent.densityLargeThreshold, 0.35, 0, 4),
    densityContrastGain: mutate(parent.densityContrastGain, 1.2, 0, 20),
    densityContrastBalance: mutate(parent.densityContrastBalance, 0.12, 0, 1.5),
    densityEmissionPower: mutate(parent.densityEmissionPower, 0.28, 0.25, 5),
    densityOcclusion: mutate(parent.densityOcclusion, 0.12, 0, 1),
    accumulationStrength: mutate(parent.accumulationStrength, 0.9, 0, 12),
    accumulationRadius: mutate(parent.accumulationRadius, 0.28, 0.2, 4),
    accumulationCurve: mutate(parent.accumulationCurve, 0.45, 0.5, 6),
    accumulationMemory: mutate(parent.accumulationMemory, 0.12, 0, 0.96),
    accumulationNoiseReject: mutate(parent.accumulationNoiseReject, 0.14, 0, 1),
    trailOpacity: mutate(parent.trailOpacity, 0.25, 0, 2.5),
    fogBrightness: mutate(parent.fogBrightness, 0.24, 0, 8),
    trailThreshold: mutate(parent.trailThreshold, 0.01, 0, 0.28),
    filament: mutate(parent.filament, 0.12, 0, 1),
    renderLayer: rng() > 0.12 ? parent.renderLayer : randomChoice(rng, ["both", "particles", "trails", "density", "volume-density", "accumulation"] as const),
    cameraYaw: parent.cameraYaw + randomRange(rng, -0.42, 0.42),
    cameraPitch: mutate(parent.cameraPitch, 0.14, -1.32, 1.32),
    cameraDistance: mutate(parent.cameraDistance, 0.32, 0.9, 10.5),
    particleColorMode: rng() > 0.14 ? parent.particleColorMode : randomChoice(rng, ["solid", "gradient-inferno", "gradient-magma", "gradient-viridis", "gradient-turbo", "gradient-rainbow", "gradient-spectral"] as const),
    trailColorMode: rng() > 0.16 ? parent.trailColorMode : randomChoice(rng, ["stable", "flow", "thermal", "tint"]),
    palette: rng() > 0.16 ? parent.palette : randomChoice(rng, ["aurora", "ember", "spectral"])
  };
}

function normalizeDiscoveryConfig(config: LiveGpu3dConfig): LiveGpu3dConfig {
  const tapRadius = config.particleCount >= 262144 ? 1 : config.particleCount >= 65536 ? 2 : config.particleCount >= 16384 ? 3 : 5;
  const maxRadius = Math.min(0.28, (tapRadius + 0.5) * (2 / Math.max(1, config.width)));
  const depositRadius = clamp(config.depositRadius, 0.0015, maxRadius);
  return {
    ...config,
    width: Math.round(clamp(config.width, 32, 96) / 8) * 8,
    height: Math.round(clamp(config.height, 32, 96) / 8) * 8,
    depth: Math.round(clamp(config.depth, 32, 96) / 8) * 8,
    particleCount: Math.round(clamp(config.particleCount, 128, 1_048_576) / 1024) * 1024,
    depositRadius,
    sigma: Math.max(0.00075, depositRadius * 0.34),
    depositTapRadius: tapRadius,
    trailPersistence: clamp(config.trailPersistence, 0.85, 0.999),
    trailDiffusion: clamp(config.trailDiffusion, 0, 1),
    sensorDistance: clamp(config.sensorDistance, 0, 3),
    sensorAngle: clamp(config.sensorAngle, -1, 1),
    mutationScale: clamp(config.mutationScale, 0, 0.5),
    globalForceMult: clamp(config.globalForceMult, 0, 2),
    drag: clamp(config.drag, -0.5, 1),
    strafePower: clamp(config.strafePower, 0, 0.5),
    axialForce: clamp(config.axialForce, -1, 1),
    lateralForce: clamp(config.lateralForce, -1, 1),
    hazardRate: clamp(config.hazardRate, 0, 0.05),
    simulationSpeed: clamp(config.simulationSpeed, 0, 1),
    cohorts: Math.round(clamp(config.cohorts, 1, 128)),
    ruleSeed: finiteModulo(config.ruleSeed, 1),
    hueSensitivity: clamp(config.hueSensitivity, -1, 1),
    orientationMix: clamp(config.orientationMix, 0, 1)
  };
}

function tagsForScore(score: DiscoveryScore): string[] {
  const tags: string[] = [score.temperature];
  if (score.signals.cohesion > 0.62) tags.push("cohesive");
  if (score.signals.diversity > 0.62) tags.push("diverse");
  if (score.signals.persistence > 0.72) tags.push("persistent");
  if (score.signals.novelty > 0.62) tags.push("novel");
  return tags;
}

function blendSignals(...values: Array<number | null | undefined>): number {
  const valid = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  if (!valid.length) return 0;
  return clamp01(valid.reduce((total, value) => total + value, 0) / valid.length);
}

function bell(value: number, low: number, high: number): number {
  if (value < low) return remap(value, 0, low);
  if (value > high) return 1 - remap(value, high, 1);
  return 1;
}

function remap(value: number, min: number, max: number): number {
  return clamp01((value - min) / Math.max(0.000001, max - min));
}

function randomRange(rng: () => number, min: number, max: number): number {
  return min + (max - min) * rng();
}

function randomChoice<T>(rng: () => number, values: readonly T[]): T {
  return values[Math.min(values.length - 1, Math.floor(rng() * values.length))];
}

function clamp01(value: number): number {
  return clamp(value, 0, 1);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, Number.isFinite(value) ? value : min));
}

function finiteModulo(value: number, modulo: number): number {
  if (!Number.isFinite(value)) return 0;
  return ((value % modulo) + modulo) % modulo;
}

function candidateId(seed: number, generation: number, index: number, localSeed: number): string {
  return `discovery-${Math.abs(seed).toString(36)}-g${generation}-${index + 1}-${Math.abs(localSeed).toString(36).slice(0, 6)}`;
}

function stageOffset(stage: DiscoveryStage): number {
  return stage === "survey" ? 101 : stage === "mutate" ? 503 : 907;
}

function hashNumber(...values: number[]): number {
  let hash = 2166136261;
  for (const value of values) {
    hash ^= Math.floor(value) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function seededRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
