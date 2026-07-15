import type { ParticleBlendMode, ParticleColorMode, RenderControls, TrailColorMode } from "./cacheRenderer";
import { maxRayResolution, resizeCanvasToDisplayResolution } from "./renderTarget";
import { requestHighPerformanceWebGpuAdapter } from "./webgpu";
import { configureHdrWebGpuCanvas, type WebGpuCanvasColor } from "./webgpuCanvas";

export type LiveGpu3dConfig = {
  seed: number;
  simulationSpeed: number;
  width: number;
  height: number;
  depth: number;
  particleCount: number;
  dt: number;
  sensorGain: number;
  sensorAngle: number;
  sensorDistance: number;
  mutationScale: number;
  globalForceMult: number;
  drag: number;
  strafePower: number;
  strafeMomentum: number;
  axialForce: number;
  lateralForce: number;
  hazardRate: number;
  trailPersistence: number;
  trailDiffusion: number;
  depositRadius: number;
  depositTapRadius: number;
  depositMass: number;
  sigma: number;
  pulseDepth: number;
  pulseRate: number;
  restlessness: number;
  cohorts: number;
  ruleSeed: number;
  hueSensitivity: number;
  colorByCohort: boolean;
  symmetryAxes: number;
  absoluteOrientation: 0 | 1 | 2 | 3 | 4 | 5;
  orientationMix: number;
  initialConditions: 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8;
  boundaryMode: 0 | 1 | 2 | 3 | 4 | 5 | 6;
  domainShape: 0 | 1 | 2 | 3;
  rule: number[];
  recycleCutoff: number;
  recycleEnabled: boolean;
  // --- Emergent behaviors. All 0 = off; with every one at 0 the engine is byte-identical to before.
  mips: number;            // motility-induced phase separation: crowd-slowdown -> spontaneous droplets
  anisoFollow: number;     // follow the deposited current direction -> one-way lanes / vortex rings
  flockAlign: number;      // Couzin/Vicsek velocity alignment to local flow -> murmurations
  flockSeparate: number;   // steer away from density gradient -> spacing / repulsion
  chemotaxis: number;      // climb (+) or descend (-) the density gradient
  quorumStrength: number;  // ignition: above local-density threshold, boost emission + shift hue (contagious)
  quorumThreshold: number; // local density at which a particle "ignites"
  leniaStrength: number;   // Lenia continuous-CA growth term on the density field
  leniaCenter: number;     // Lenia growth kernel center (mu)
  leniaWidth: number;      // Lenia growth kernel width (sigma)
  speciesForce: number;    // inter-species (cohort) attract/repel via the ecology field -> particle-life clusters
  predator: number;        // predator cohorts climb prey-species gradient -> hunting fronts
  alarm: number;           // prey flee the alarm channel -> escape waves
  grayScott: number;       // Gray-Scott reaction-diffusion coupled into the density field -> Turing skins
  gsFeed: number;          // Gray-Scott feed rate F
  gsKill: number;          // Gray-Scott kill rate k
  energy: number;          // per-particle metabolism: drain by motion, refuel from trail, starve -> reset
  energyDrain: number;     // metabolism drain rate
};

export type LiveFieldStats = {
  densitySum: number;
  flowSum: number;
  nonzeroVoxels: number;
  maxDensity: number;
  maxDensitySignal: number;
  densitySignalMean: number;
  densitySignalP90: number;
};

export type LiveRuleEvidence = {
  nonzero: boolean;
  cohortDependent: boolean;
  stable: boolean;
  firstSignature: number;
  secondSignature: number;
};

export type LiveGpu3dDiagnostics = {
  renderer: "webgpu-live-fluoddity-3d";
  webgpu: true;
  rulePort: "fluoddity-core-fourier";
  fieldKind: "vector-volume";
  adaptation: "two-plane-3d";
  renderMode: "volume-raymarch+particle-splats" | "particle-splats" | "debug-voxel-splats" | "screen-space-density" | "volume-density-raymarch" | "accumulation";
  depositMode: "particle-atomic-fixed-point";
  particleColorModel: ParticleColorMode;
  particleSizeModel: "perspective-3d";
  trailColorMode: TrailColorMode;
  sceneBrightness: number;
  particleBrightness: number;
  fogBrightness: number;
  particleVelocityStretch: boolean;
  particleStretch: number;
  particleStretchMin: number;
  particleStretchSpeed: number;
  particleSpeedCutoff: number;
  particleSlowCutoff: number;
  particleGradientSensitivity: number;
  simulationSpeed: number;
  simulationTime: number;
  simulationTimeAdvance: number;
  simulationStepScales: number[];
  depositTaps: number;
  depositScale: number;
  sensorModel: "fluoddity-core-two-sensor-two-plane-3d";
  pheromoneModel: "vector-current-density-volume";
  particleCount: number;
  voxelCount: number;
  particleBufferBytes: number;
  fieldBufferBytes: number;
  requestedRenderLayer: RenderControls["renderLayer"];
  effectiveRenderLayer: RenderControls["renderLayer"];
  deviceLimits: {
    maxBufferSize: number;
    maxStorageBufferBindingSize: number;
  };
  particleBlendMode: ParticleBlendMode;
  particleDensityCutoff: number;
  particleDensityRadius: number;
  particleDensityNormalize: number;
  particleDensitySoftness: number;
  particleDensityReference: number;
  particleDensityReferenceTarget: number;
  statsReadPending: boolean;
  particleSupportMask: number;
  particleSupportRadius: number;
  particleSupportNeighbors: number;
  particleSupportFlow: number;
  particleSupportGridSize: number;
  particleSupportSourceBufferIndex: number;
  fastParticleRender: boolean;
  cutoffPrepassActive: boolean;
  densityLargeHalfResActive: boolean;
  splatPath: "compute" | "prepared" | "classic";
  lastSortTimestep: number;
  fastNoBloomPost: boolean;
  timestep: number;
  renderFrame: number;
  renderLerpT: number;
  visualFieldSmoothing: boolean;
  visualFieldStepScale: number;
  visualFieldCorrection: number;
  visualFieldPhaseReset: boolean;
  visualFieldSourceBufferIndex: number;
  fogNoiseFrame: number;
  frameTimeMs: number;
  cpuFrameTimeMs: number;
  profileGpu: boolean;
  timestampQueries: boolean;
  timings: LiveRenderTimings;
  renderResolution: [number, number];
  particleResolutionScale: number;
  canvasColor: WebGpuCanvasColor;
  raySteps: number;
  effectiveFogSteps: number;
  fogTemporal: boolean;
  fogRenderScale: number;
  fogTemporalBlend: number;
  fogHistoryReset: boolean;
  fieldTextureSampling: boolean;
  emptySpaceSkipping: boolean;
  emptySpaceThreshold: number;
  emptySpaceStride: number;
  dimensions: [number, number, number];
  fieldStats: LiveFieldStats;
  ruleEvidence: LiveRuleEvidence;
  camera: {
    yaw: number;
    pitch: number;
    distance: number;
    panX: number;
    panY: number;
  };
};

export type LiveRenderTimings = {
  resizeMs: number;
  ensureMs: number;
  encodeStepMs: number;
  clearBrushEncodeMs: number;
  depositEncodeMs: number;
  particleUpdateEncodeMs: number;
  fieldUpdateEncodeMs: number;
  visualSmoothEncodeMs: number;
  renderPassEncodeMs: number;
  submitMs: number;
  gpuQueueWaitMs: number;
  timestampReadMs: number;
  gpuPassMs: LiveGpuPassTimings | null;
  statsReadMs: number;
};

export type LiveGpuPassTimings = {
  clearBrushMs: number;
  depositMs: number;
  particleUpdateMs: number;
  fieldUpdateMs: number;
  particleSupportClearMs: number;
  particleSupportBuildMs: number;
  particleSupportResolveMs: number;
  densitySmallMs: number;
  densityLargeMs: number;
  densityCompositeMs: number;
  accumulationSplatMs: number;
  accumulationCompositeMs: number;
  volumeDensityClearMs: number;
  volumeDensityDepositMs: number;
  volumeDensityResolveMs: number;
  volumeDensityBlurMs: number;
  volumeDensityRaymarchMs: number;
  renderMs: number;
  fogCompositeMs: number;
  fogPresentMs: number;
  postMs: number;
  totalMeasuredMs: number;
};

type RenderUniforms = {
  resolution: [number, number];
  dimensions: [number, number, number];
  particleCount: number;
  voxelCount: number;
  timestep: number;
  density: number;
  exposure: number;
  focusDistance: number;
  aperture: number;
  focalK: number;
  dofBlur: number;
  dofDebug: number;
  dofEnabled: number;
  renderLerpT: number;
  sceneBrightness: number;
  overlay: number;
  palette: number;
  filament: number;
  yaw: number;
  pitch: number;
  yawCos: number;
  yawSin: number;
  pitchCos: number;
  pitchSin: number;
  distance: number;
  raySteps: number;
  fogStepScale: number;
  fogTemporalBlend: number;
  fogBlueNoise: number;
  fogFrame: number;
  fieldTextureMode: number;
  emptySpaceSkip: number;
  emptySpaceThreshold: number;
  emptySpaceStride: number;
  particleSizePx: number;
  particleMinPx: number;
  particleMaxPx: number;
  particleOpacity: number;
  particleBrightness: number;
  particleColorMode: number;
  particleColorModeFrom: number;
  particleColorBlend: number;
  particleCutoffPrepass: number;
  particleVelocityStretch: number;
  particleStretch: number;
  particleStretchMin: number;
  particleStretchSpeed: number;
  particleSpeedCutoff: number;
  particleSlowCutoff: number;
  particleGlowCore: number;
  particleHotCore: number;
  particleExponent: number;
  particleBrightnessBoost: number;
  particleSupportSmoothing: number;
  particleHazeCull: number;
  particleGradientSensitivity: number;
  particleBlendMode: number;
  particleDensityCutoff: number;
  particleDensityRadius: number;
  particleDensityNormalize: number;
  particleDensitySoftness: number;
  particleDensityReference: number;
  particleSupportMask: number;
  particleSupportRadius: number;
  particleSupportNeighbors: number;
  particleSupportFlow: number;
  particleSupportGridSize: number;
  trailOpacity: number;
  fogBrightness: number;
  trailThreshold: number;
  renderLayer: number;
  trailColorMode: number;
  cameraPanX: number;
  cameraPanY: number;
  fogTint: [number, number, number];
  particleTint: [number, number, number];
  densityPassStrength: number;
  densitySmallScale: number;
  densityLargeScale: number;
  densityLargeThreshold: number;
  densityContrastGain: number;
  densityContrastBalance: number;
  densityEmissionPower: number;
  densityOcclusion: number;
  accumulationStrength: number;
  accumulationRadius: number;
  accumulationCurve: number;
  accumulationMemory: number;
  accumulationNoiseReject: number;
  cohorts: number;
  variationMaster: number;
  variationTime: number;
  variationDrift: number;
  variationNoiseMix: number;
  variationFreq: number;
  variationOctaves: number;
  variationGain: number;
  variationLacunarity: number;
  variationSizeAmount: number;
  variationSizeCurve: number;
  variationSizeMin: number;
  variationSizeMax: number;
  variationBrightAmount: number;
  variationBrightCurve: number;
  variationBrightMin: number;
  variationBrightMax: number;
  variationOpacityAmount: number;
  variationOpacityCurve: number;
  variationOpacityMin: number;
  variationOpacityMax: number;
  variationColorAmount: number;
  variationColorCurve: number;
  variationColorMin: number;
  variationColorMax: number;
  domainShape: number;
  audioLow: number;
  audioMid: number;
  audioHigh: number;
};

type FogHistoryView = {
  width: number;
  height: number;
  steps: number;
  layer: number;
  fieldTextureMode: number;
  emptySpaceSkip: number;
  emptySpaceThreshold: number;
  emptySpaceStride: number;
  density: number;
  exposure: number;
  sceneBrightness: number;
  trailOpacity: number;
  fogBrightness: number;
  trailThreshold: number;
  trailColorMode: number;
  palette: number;
  filament: number;
  fogTint: [number, number, number];
  yaw: number;
  pitch: number;
  distance: number;
  panX: number;
  panY: number;
};

export const defaultLiveGpu3dConfig: LiveGpu3dConfig = {
  seed: 3405691582,
  simulationSpeed: 1,
  width: 64,
  height: 64,
  depth: 64,
  particleCount: 8192,
  dt: 1,
  sensorGain: 1.369,
  sensorAngle: -0.141,
  sensorDistance: 1.013,
  mutationScale: 0.095,
  globalForceMult: 0.94,
  drag: 0.125,
  strafePower: 0.169,
  strafeMomentum: 0,
  axialForce: 0.022,
  lateralForce: -0.186,
  hazardRate: 0,
  trailPersistence: 0.943,
  trailDiffusion: 1,
  depositRadius: 0.07,
  depositTapRadius: 2,
  depositMass: 0.78,
  sigma: 0.024,
  pulseDepth: 0,
  pulseRate: 1,
  restlessness: 0,
  cohorts: 6,
  ruleSeed: 0.6633918167161669,
  hueSensitivity: 0.5,
  colorByCohort: true,
  symmetryAxes: 2,
  absoluteOrientation: 1,
  orientationMix: 0,
  initialConditions: 0,
  boundaryMode: 2,
  domainShape: 0,
  rule: [],
  recycleCutoff: 0,
  recycleEnabled: false,
  // Emergent behaviors — every master strength defaults to 0 (off). Secondary shape params
  // (threshold/center/width/feed/kill/drain) carry sensible non-zero defaults but are read by
  // no shader while their master strength is 0, so sim output is byte-identical to the classic engine.
  mips: 0,
  anisoFollow: 0,
  flockAlign: 0,
  flockSeparate: 0,
  chemotaxis: 0,
  quorumStrength: 0,
  quorumThreshold: 0.5,
  leniaStrength: 0,
  leniaCenter: 0.15,
  leniaWidth: 0.015,
  speciesForce: 0,
  predator: 0,
  alarm: 0,
  grayScott: 0,
  gsFeed: 0.037,
  gsKill: 0.06,
  energy: 0,
  energyDrain: 0.5
};

const emptyStats: LiveFieldStats = {
  densitySum: 0,
  flowSum: 0,
  nonzeroVoxels: 0,
  maxDensity: 0,
  maxDensitySignal: 0,
  densitySignalMean: 0,
  densitySignalP90: 0
};

const particleFloatCount = 8;
// Sim uniform buffer size in bytes. The classic engine used 176 (44 f32/u32 slots, indices 0..43).
// The emergent-behavior knobs append at slots 43+, so the buffer grows to 256 (64 slots). When every
// emergent knob is 0 the new slots are written but read by no shader branch, so sim output stays
// byte-identical to the classic engine. Must equal the WGSL `struct SimConfig` size (rounded to 16).
const SIM_UNIFORM_BYTES = 256;
const timestampQueryCount = 32;
const timestampQueryBufferSize = 256;
const fogTextureFormat: GPUTextureFormat = "rgba16float";
const densityTextureFormat: GPUTextureFormat = "rgba16float";
const accumulationTextureFormat: GPUTextureFormat = "rgba16float";
const defaultParticleSupportGridSize = 128;
// Ribbon trails: fixed-length history ring per ribbon, indexed by ribbon index (= particleId/stride).
// The trail buffer is a single fixed-size allocation (MAX_RIBBONS * TRAIL_LENGTH * vec4) so changing
// the ribbon fraction never reallocates — it just changes the stride/draw count.
// History ring holds up to RIBBON_TRAIL_MAX frames per ribbon; the ribbonLength control selects how
// many recent frames to span and ribbonJoints how many spine points to render across that span.
const RIBBON_TRAIL_MAX = 32;
// Capped so the single trail storage buffer (RIBBON_MAX * TRAIL_MAX * vec4) stays under the default
// 128MB maxStorageBufferBindingSize: 240k * 32 * 16B = 123MB. Higher fractions cap at this count.
const RIBBON_MAX = 240_000;
const ribbonUniformByteLength = 48; // 5 u32 (stride,max,frame,length,joints) + 4 f32 (width,fade,fade_start,edge_fade), padded
// 480B held slots 0-119. Grown to 496B (slots 0-123) for the render-side optimization toggles:
// slot 118 = particle_cutoff_prepass (A); slots 119-123 stay reserved for volume-density toggles.
// Grown to 512B (slots 124-127) for the particle haze/contrast controls.
const renderUniformByteLength = 512;
// Duration of the automatic color-mode cross-fade (ms). Short/"quick" so switching feels responsive.
const COLOR_BLEND_MS = 260;
const particleSupportIndirectDrawReset = new Uint32Array([4, 0, 0, 0]);

// IEEE 754 half-float (rgba16float) -> float32 decode.
function halfToFloat(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

export type CapturedSimState = {
  field: Float32Array;
  particles: Float32Array;
  voxelCount: number;
  particleCount: number;
  width: number;
  height: number;
  depth: number;
};

type ParticleBufferChunk = {
  offset: number;
  count: number;
  buffers: [GPUBuffer, GPUBuffer];
  supportBuffer: GPUBuffer;
  // One f32 per particle: the density-cutoff signal precomputed by the cutoff prepass when
  // controls.particleCutoffPrepass is on (read by particle_vs at binding 6).
  cutoffSignalBuffer: GPUBuffer;
  activeIndexBuffer: GPUBuffer;
  activeDrawBuffer: GPUBuffer;
  // 32 bytes per particle, written by prepare_splats and read by particle_record_vs.
  splatRecordBuffer: GPUBuffer;
  // Persistent global particle ids (survive the spatial sort's permutations).
  sortIdBuffer: GPUBuffer;
  sortScratchBuffer: GPUBuffer;
  sortScratchIdBuffer: GPUBuffer;
  sortMapBuffer: GPUBuffer;
};

export class RealtimeGpuSim3d {
  private device?: GPUDevice;
  private context?: GPUCanvasContext;
  private format?: GPUTextureFormat;
  private canvasColor?: WebGpuCanvasColor;
  private config = defaultLiveGpu3dConfig;
  // Latest audio band levels (0..1) for the audio-reactive color mode; fed each frame by the host.
  private audioLevels = { low: 0, mid: 0, high: 0 };
  // Time-based color-mode cross-fade. When the color mode changes we ease `colorBlend` 0->1 over
  // COLOR_BLEND_MS, blending the previous mode into the new one. A swipe through several modes keeps
  // the original `from` and retargets `to` to the latest pick, so it eases into wherever you land.
  private colorBlendFromMode = 0;
  private colorBlendToMode = 0;
  private colorBlend = 1;
  private colorBlendLastMs = 0;
  private particleChunks: ParticleBufferChunk[] = [];
  private fieldBuffers: GPUBuffer[] = [];
  private visualFieldBuffers: GPUBuffer[] = [];
  private brushBuffer?: GPUBuffer;
  private ruleBuffer?: GPUBuffer;
  private simUniformBuffer?: GPUBuffer;
  private chunkSimUniformBuffers: GPUBuffer[] = [];
  private renderUniformBuffer?: GPUBuffer;
  private fogRenderUniformBuffer?: GPUBuffer;
  private particleSupportGridBuffer?: GPUBuffer;
  private particleSupportGridSize = defaultParticleSupportGridSize;
  private clearBrushPipeline?: GPUComputePipeline;
  private initParticlePipeline?: GPUComputePipeline;
  private depositPipeline?: GPUComputePipeline;
  private visualDepositPipeline?: GPUComputePipeline;
  private updatePipeline?: GPUComputePipeline;
  // Ribbon trails (lazy: created the first time ribbonFraction > 0; default off = zero cost/VRAM).
  private trailBuffer?: GPUBuffer;
  private ribbonUniformBuffer?: GPUBuffer;
  private trailAppendPipeline?: GPUComputePipeline;
  private ribbonRenderPipeline?: GPURenderPipeline;
  private ribbonFrame = 0;
  private fieldPipeline?: GPUComputePipeline;
  private fieldTexturePipeline?: GPUComputePipeline;
  private particleSupportClearPipeline?: GPUComputePipeline;
  private particleSupportBuildPipeline?: GPUComputePipeline;
  private particleSupportResolvePipeline?: GPUComputePipeline;
  private cutoffPrepassPipeline?: GPUComputePipeline;
  private fieldRenderPipeline?: GPURenderPipeline;
  private fieldFogRenderPipeline?: GPURenderPipeline;
  private fogCompositePipeline?: GPURenderPipeline;
  private postPipeline?: GPURenderPipeline;
  private postNoBloomPipeline?: GPURenderPipeline;
  private bloomPrefilterPipeline?: GPURenderPipeline;
  private bloomDownsamplePipeline?: GPURenderPipeline;
  private bloomUpsamplePipeline?: GPURenderPipeline;
  private densitySmallPipeline?: GPURenderPipeline;
  private densityLargePipeline?: GPURenderPipeline;
  private densityCompositePipeline?: GPURenderPipeline;
  private accumulationSplatPipeline?: GPURenderPipeline;
  private accumulationCompositePipeline?: GPURenderPipeline;
  private volumeDensityClearPipeline?: GPUComputePipeline;
  private volumeDensityDepositPipeline?: GPUComputePipeline;
  private volumeDensityResolvePipeline?: GPUComputePipeline;
  private volumeDensityBlurXPipeline?: GPUComputePipeline;
  private volumeDensityBlurYPipeline?: GPUComputePipeline;
  private volumeDensityBlurZPipeline?: GPUComputePipeline;
  // Optimization C: half-resolution large-channel passes (downsample + half-res separable blur).
  private volumeDensityDownsamplePipeline?: GPUComputePipeline;
  private volumeDensityBlurXHalfPipeline?: GPUComputePipeline;
  private volumeDensityBlurYHalfPipeline?: GPUComputePipeline;
  private volumeDensityBlurZHalfPipeline?: GPUComputePipeline;
  private volumeDensityRenderPipeline?: GPURenderPipeline;
  private bloomTextures: GPUTexture[] = [];
  private postUniformBuffer?: GPUBuffer;
  private sceneTexture?: GPUTexture;
  private sceneTextureSize: [number, number] = [0, 0];
  // Persistent copy of the post-processed swapchain frame (bloom/vignette/grade applied), kept so an
  // offline export can read the FINAL HDR image — the swapchain itself is gone once presented, and
  // sceneTexture is pre-post. Only maintained while captureFinalHDR is set (replay), so live is untouched.
  private finalTexture?: GPUTexture;
  private finalTextureSize: [number, number] = [0, 0];
  captureFinalHDR = false;
  // HDR-capture highlight headroom (post.grade.w): when capturing, the composite renders the EXACT
  // live AGX look and lifts only AGX-clipped highlights up to (1 + headroom)×paper-white. The export
  // maps shader 1.0 -> paper-white nits via --peak, so headroom 4 with --peak 203 => ~1000-nit cores.
  // 0/unset while not capturing keeps the live SDR look bit-identical. Tunable from the export.
  hdrCaptureHeadroom = 3.0;
  // Forces the render aspect (width / height) instead of deriving it from the canvas CSS box. Set by
  // the offline HDR exporter (via the replay hooks) to render an exact 9:16 portrait independent of the
  // replay browser window; 0/undefined = follow the canvas (normal live behavior).
  aspectOverride = 0;
  private postSampler?: GPUSampler;
  private densitySmallTexture?: GPUTexture;
  private densityLargeTexture?: GPUTexture;
  private densityEmptyFogTexture?: GPUTexture;
  private densityTextureSize: [number, number] = [0, 0];
  private densitySampler?: GPUSampler;
  private densitySplatBindGroups: GPUBindGroup[][][] = [];
  private densityCompositeBindGroup?: GPUBindGroup;
  private densityCompositeFogTexture?: GPUTexture;
  private accumulationCurrentTexture?: GPUTexture;
  private accumulationHistoryTextures: GPUTexture[] = [];
  private accumulationTextureSize: [number, number] = [0, 0];
  private accumulationSampler?: GPUSampler;
  private accumulationSplatBindGroups: GPUBindGroup[][] = [];
  private accumulationHistoryReadIndex = 0;
  private accumulationHistoryKey = "";
  private accumulationHistoryPrepared = false;
  private volumeDensityAccumBuffer?: GPUBuffer;
  private volumeDensitySmallBuffer?: GPUBuffer;
  private volumeDensityBlurTempBuffer?: GPUBuffer;
  private volumeDensityLargeBuffer?: GPUBuffer;
  private volumeDensitySmallTexture?: GPUTexture;
  private volumeDensityLargeTexture?: GPUTexture;
  // Optimization C half-resolution large-channel resources (allocated only when the toggle is on).
  private volumeDensitySmallHalfBuffer?: GPUBuffer;
  private volumeDensityBlurTempHalfBuffer?: GPUBuffer;
  private volumeDensityLargeHalfBuffer?: GPUBuffer;
  private volumeDensityLargeTextureHalf?: GPUTexture;
  private volumeDensitySampler?: GPUSampler;
  private volumeDensityResourceSize: [number, number, number] = [0, 0, 0];
  private volumeDensityClearBindGroup?: GPUBindGroup;
  private volumeDensityDepositBindGroups: GPUBindGroup[][] = [];
  private volumeDensityResolveBindGroup?: GPUBindGroup;
  private volumeDensityBlurXBindGroup?: GPUBindGroup;
  private volumeDensityBlurYBindGroup?: GPUBindGroup;
  private volumeDensityBlurZBindGroup?: GPUBindGroup;
  private volumeDensityDownsampleBindGroup?: GPUBindGroup;
  private volumeDensityBlurXHalfBindGroup?: GPUBindGroup;
  private volumeDensityBlurYHalfBindGroup?: GPUBindGroup;
  private volumeDensityBlurZHalfBindGroup?: GPUBindGroup;
  private volumeDensityRenderBindGroup?: GPUBindGroup;
  private volumeDensityRenderHalfBindGroup?: GPUBindGroup;
  private volumeDensityPreparedKey = "";
  private volumeDensityPrepared = false;
  private particleRenderPipelines: Partial<Record<ParticleBlendMode, GPURenderPipeline>> = {};
  private particleFastRenderPipelines: Partial<Record<ParticleBlendMode, GPURenderPipeline>> = {};
  private splatPreparePipeline?: GPUComputePipeline;
  private splatDrawPipelines: Partial<Record<ParticleBlendMode, GPURenderPipeline>> = {};
  private splatPrepareBindGroups: GPUBindGroup[][] = [];
  private splatDrawBindGroups: Partial<Record<ParticleBlendMode, GPUBindGroup[]>> = {};
  private splatAccumPipeline?: GPUComputePipeline;
  private splatResolvePipeline?: GPURenderPipeline;
  private splatAccumBuffer?: GPUBuffer;
  private splatAccumBindGroups: GPUBindGroup[] = [];
  private splatResolveBindGroup?: GPUBindGroup;
  private updateTexturePipeline?: GPUComputePipeline;
  private updateTextureBindGroups: GPUBindGroup[][] = [];
  // Ecology field (species/predator/alarm). Lazily created the first time one of those behaviors is
  // active; fully torn down with the other buffers on reset. Default off => never allocated.
  private ecologyFieldBuffer?: GPUBuffer;
  private ecologyBrushBuffer?: GPUBuffer;
  private ecologyDepositPipeline?: GPUComputePipeline;
  private ecologyUpdatePipeline?: GPUComputePipeline;
  private ecologyParticlePipeline?: GPUComputePipeline;
  private ecologyClearBindGroup?: GPUBindGroup;
  private ecologyUpdateBindGroup?: GPUBindGroup;
  private ecologyDepositBindGroups: GPUBindGroup[][] = [];
  private ecologyParticleBindGroups: GPUBindGroup[][] = [];
  private fieldSimSampler?: GPUSampler;
  private visualFieldSmoothPipeline?: GPUComputePipeline;
  private float32FilterableSupported = false;
  private sortHistogramPipeline?: GPUComputePipeline;
  private sortScanPipeline?: GPUComputePipeline;
  private sortScatterPipeline?: GPUComputePipeline;
  private sortApplyPipeline?: GPUComputePipeline;
  private sortHistBuffer?: GPUBuffer;
  private sortHistBindGroups: GPUBindGroup[][] = [];
  private sortScanBindGroup?: GPUBindGroup;
  private sortScatterBindGroups: GPUBindGroup[][] = [];
  private sortApplyBindGroups: GPUBindGroup[][] = [];
  // Render-side knowledge cached for the step path: ribbons select particles by buffer slot,
  // so the spatial sort must stand down while they are active.
  private ribbonsActive = false;
  // Set when the field buffers were written directly (seed/cache/zero); the sensing texture
  // must be re-synced before the next particle update reads it.
  private fieldTextureDirty = false;
  private particleRenderBindGroupLayout?: GPUBindGroupLayout;
  private overlayPipeline?: GPURenderPipeline;
  private timestampQueriesSupported = false;
  private timestampQuerySet?: GPUQuerySet;
  private timestampResolveBuffer?: GPUBuffer;
  private timestampReadBuffer?: GPUBuffer;
  private timestampFrame = 0;
  private lastGpuPassTimings: LiveGpuPassTimings | null = null;
  private clearBindGroup?: GPUBindGroup;
  private particleInitializationPending = false;
  private depositBindGroups: GPUBindGroup[][] = [];
  private visualDepositBindGroups: GPUBindGroup[][] = [];
  private updateBindGroups: GPUBindGroup[][] = [];
  private particleSupportClearBindGroup?: GPUBindGroup;
  private particleSupportBuildBindGroups: GPUBindGroup[][] = [];
  private particleSupportResolveBindGroups: GPUBindGroup[][] = [];
  private fieldBindGroups: GPUBindGroup[] = [];
  private visualFieldBindGroups: GPUBindGroup[] = [];
  private fieldTextureBindGroups: GPUBindGroup[] = [];
  private fieldRenderBindGroups: GPUBindGroup[] = [];
  private particleRenderBindGroups: Partial<Record<ParticleBlendMode, GPUBindGroup[][]>> = {};
  private overlayBindGroup?: GPUBindGroup;
  private fieldTexture?: GPUTexture;
  private fieldTextureSampler?: GPUSampler;
  private fogCurrentTexture?: GPUTexture;
  private fogHistoryTextures: GPUTexture[] = [];
  private fogSampler?: GPUSampler;
  private fogTextureSize: [number, number] = [0, 0];
  private fogHistoryReadIndex = 0;
  private lastFogHistoryView?: FogHistoryView;
  private fogHistoryResetFrames = 0;
  private readIndex = 0;
  private writeIndex = 1;
  private visualFieldReadIndex = 0;
  private timestep = 0;
  private simulationTime = 0;
  private simStepAccumulator = 0;
  private renderLerpT = 1;
  private visualFieldLerpT = 0;
  private visualFieldPrepared = false;
  private visualFieldActiveLastFrame = false;
  private renderFrame = 0;
  private failed = false;
  private renderValidationChecked = false;
  private lastFieldStats: LiveFieldStats = emptyStats;
  private targetParticleDensityReference = 0;
  private smoothedParticleDensityReference = 0;
  private fieldStatsReadPending = false;
  private fieldStatsReadGeneration = 0;
  private ruleEvidence = computeRuleEvidence(defaultLiveGpu3dConfig);

  async reset(): Promise<void> {
    await this.waitForSubmittedWorkDone();
    this.destroyBuffers();
    this.timestep = 0;
    this.simulationTime = 0;
    this.simStepAccumulator = 0;
    // A leftover live-morph transition must not leak into the fresh run, or replaying
    // from frame 0 consumes a varying number of residual morph steps and the timeline
    // loses its bit-identical replay guarantee.
    this.morphStepsRemaining = 0;
    this.renderLerpT = 1;
    this.visualFieldLerpT = 0;
    this.renderFrame = 0;
    this.readIndex = 0;
    this.writeIndex = 1;
    this.visualFieldReadIndex = 0;
    this.visualFieldPrepared = false;
    this.visualFieldActiveLastFrame = false;
    this.renderValidationChecked = false;
    this.lastFieldStats = emptyStats;
    this.targetParticleDensityReference = 0;
    this.smoothedParticleDensityReference = 0;
    this.fieldStatsReadPending = false;
    this.fieldStatsReadGeneration += 1;
    this.timestampFrame = 0;
    this.lastGpuPassTimings = null;
    this.lastFogHistoryView = undefined;
    this.fogHistoryResetFrames = 0;
  }

  private async waitForSubmittedWorkDone(): Promise<void> {
    const queue = this.device?.queue;
    if (!queue) return;
    try {
      await queue.onSubmittedWorkDone();
    } catch {
      // Device loss/recovery will rebuild resources; reset should still clear JS-side state.
    }
  }

  recover(): void {
    this.destroyDeviceResources();
    this.context = undefined;
    this.format = undefined;
    this.canvasColor = undefined;
    this.failed = false;
    this.timestep = 0;
    this.simulationTime = 0;
    this.simStepAccumulator = 0;
    this.renderLerpT = 1;
    this.renderFrame = 0;
    this.readIndex = 0;
    this.writeIndex = 1;
    this.renderValidationChecked = false;
    this.lastFieldStats = emptyStats;
    this.targetParticleDensityReference = 0;
    this.smoothedParticleDensityReference = 0;
    this.fieldStatsReadPending = false;
    this.fieldStatsReadGeneration += 1;
    this.timestampFrame = 0;
    this.lastGpuPassTimings = null;
    this.lastFogHistoryView = undefined;
    this.fogHistoryResetFrames = 0;
  }

  get currentTimestep(): number {
    return this.timestep;
  }

  // Step-only fast path for deterministic timeline seeking. Encodes `count` full
  // simulation steps (stepTimeScale = 1, no accumulator) without a render pass.
  // Additive: reuses encodeStepPasses; does not touch the render/post-process path.
  async advanceSteps(canvas: HTMLCanvasElement, config: LiveGpu3dConfig, count: number): Promise<void> {
    if (count <= 0) return;
    await this.ensureInitialized(canvas, config);
    await this.ensureEcologyResources();
    const device = this.device!;
    for (let i = 0; i < count; i += 1) {
      const encoder = device.createCommandEncoder();
      if (this.shouldSortAtCurrentStep()) {
        this.encodeSortPass(encoder);
      }
      this.encodeStepPasses(encoder, false, 1);
      device.queue.submit([encoder.finish()]);
      [this.readIndex, this.writeIndex] = [this.writeIndex, this.readIndex];
      this.timestep += 1;
      this.simulationTime += 1;
    }
    this.simStepAccumulator = 0;
    this.renderLerpT = 1;
    await device.queue.onSubmittedWorkDone();
  }

  private resetPipelines(): void {
    this.clearBrushPipeline = undefined;
    this.initParticlePipeline = undefined;
    this.depositPipeline = undefined;
    this.visualDepositPipeline = undefined;
    this.updatePipeline = undefined;
    this.ecologyDepositPipeline = undefined;
    this.ecologyUpdatePipeline = undefined;
    this.ecologyParticlePipeline = undefined;
    // Bind groups are built from `layout:"auto"` pipeline layouts; the format-change path calls
    // resetPipelines() WITHOUT destroyBuffers(), so the ecology caches must be invalidated here too
    // or the recreated pipelines would be fed stale, layout-incompatible bind groups.
    this.ecologyClearBindGroup = undefined;
    this.ecologyUpdateBindGroup = undefined;
    this.ecologyDepositBindGroups = [];
    this.ecologyParticleBindGroups = [];
    this.trailAppendPipeline = undefined;
    this.ribbonRenderPipeline = undefined;
    this.fieldPipeline = undefined;
    this.fieldTexturePipeline = undefined;
    this.particleSupportClearPipeline = undefined;
    this.particleSupportBuildPipeline = undefined;
    this.particleSupportResolvePipeline = undefined;
    this.cutoffPrepassPipeline = undefined;
    this.fieldRenderPipeline = undefined;
    this.fieldFogRenderPipeline = undefined;
    this.fogCompositePipeline = undefined;
    this.postNoBloomPipeline = undefined;
    this.bloomPrefilterPipeline = undefined;
    this.bloomDownsamplePipeline = undefined;
    this.bloomUpsamplePipeline = undefined;
    this.densitySmallPipeline = undefined;
    this.densityLargePipeline = undefined;
    this.densityCompositePipeline = undefined;
    this.accumulationSplatPipeline = undefined;
    this.accumulationCompositePipeline = undefined;
    this.densitySplatBindGroups = [];
    this.accumulationSplatBindGroups = [];
    this.densityCompositeBindGroup = undefined;
    this.densityCompositeFogTexture = undefined;
    this.volumeDensityClearPipeline = undefined;
    this.volumeDensityDepositPipeline = undefined;
    this.volumeDensityResolvePipeline = undefined;
    this.volumeDensityBlurXPipeline = undefined;
    this.volumeDensityBlurYPipeline = undefined;
    this.volumeDensityBlurZPipeline = undefined;
    this.volumeDensityDownsamplePipeline = undefined;
    this.volumeDensityBlurXHalfPipeline = undefined;
    this.volumeDensityBlurYHalfPipeline = undefined;
    this.volumeDensityBlurZHalfPipeline = undefined;
    this.volumeDensityRenderPipeline = undefined;
    this.postPipeline = undefined;
    this.particleRenderPipelines = {};
    this.particleFastRenderPipelines = {};
    this.splatPreparePipeline = undefined;
    this.splatDrawPipelines = {};
    this.splatPrepareBindGroups = [];
    this.splatDrawBindGroups = {};
    this.splatAccumPipeline = undefined;
    this.splatResolvePipeline = undefined;
    this.splatAccumBindGroups = [];
    this.splatResolveBindGroup = undefined;
    this.updateTexturePipeline = undefined;
    this.updateTextureBindGroups = [];
    this.visualFieldSmoothPipeline = undefined;
    this.sortHistogramPipeline = undefined;
    this.sortScanPipeline = undefined;
    this.sortScatterPipeline = undefined;
    this.sortApplyPipeline = undefined;
    this.sortHistBindGroups = [];
    this.sortScanBindGroup = undefined;
    this.sortScatterBindGroups = [];
    this.sortApplyBindGroups = [];
    this.particleRenderBindGroupLayout = undefined;
    this.overlayPipeline = undefined;
    this.clearBindGroup = undefined;
    this.particleInitializationPending = false;
    this.depositBindGroups = [];
    this.visualDepositBindGroups = [];
    this.updateBindGroups = [];
    this.particleSupportClearBindGroup = undefined;
    this.particleSupportBuildBindGroups = [];
    this.particleSupportResolveBindGroups = [];
    this.fieldBindGroups = [];
    this.visualFieldBindGroups = [];
    this.fieldTextureBindGroups = [];
    this.fieldRenderBindGroups = [];
    this.particleRenderBindGroups = {};
    this.densitySplatBindGroups = [];
    this.overlayBindGroup = undefined;
    this.destroyDensityTextures();
    this.destroyAccumulationTextures();
    this.destroyVolumeDensityResources();
    this.destroyFogTextures();
  }

  setAudioLevels(low: number, mid: number, high: number): void {
    this.audioLevels.low = low;
    this.audioLevels.mid = mid;
    this.audioLevels.high = high;
  }

  async render(
    canvas: HTMLCanvasElement,
    controls: RenderControls,
    overlay: boolean,
    stepSimulation: boolean,
    config: LiveGpu3dConfig = defaultLiveGpu3dConfig,
    profileGpu = false
  ): Promise<LiveGpu3dDiagnostics> {
    const start = performance.now();
    const resizeStart = performance.now();
    const renderResolution = resizeCanvasToDisplayResolution(canvas, controls.rayResolution, this.aspectOverride);
    const particleResolutionScale = renderResolution[1] / maxRayResolution;
    const timings: LiveRenderTimings = {
      resizeMs: performance.now() - resizeStart,
      ensureMs: 0,
      encodeStepMs: 0,
      clearBrushEncodeMs: 0,
      depositEncodeMs: 0,
      particleUpdateEncodeMs: 0,
      fieldUpdateEncodeMs: 0,
      visualSmoothEncodeMs: 0,
      renderPassEncodeMs: 0,
      submitMs: 0,
      gpuQueueWaitMs: 0,
      timestampReadMs: 0,
      gpuPassMs: this.lastGpuPassTimings,
      statsReadMs: 0
    };
    if (this.failed) {
      throw new Error("live WebGPU renderer previously failed");
    }
    const ensureStart = performance.now();
    await this.ensureInitialized(canvas, config);
    if (stepSimulation) {
      await this.ensureEcologyResources();
    }
    timings.ensureMs = performance.now() - ensureStart;
    const requestedRenderLayer = controls.renderLayer;
    const device = this.device!;
    const context = this.context!;
    if (stepSimulation) {
      this.renderFrame += 1;
    }
    this.timestampFrame += 1;
    const useFogPipeline = controls.renderLayer === "both" || controls.renderLayer === "trails";
    const useVolumeDensityPipeline = controls.renderLayer === "volume-density";
    const useAccumulationPipeline = controls.renderLayer === "accumulation";
    const useScreenDensityPipeline = controls.renderLayer === "both" || controls.renderLayer === "density";
    const useScreenDensityTimings = controls.densityPassStrength > 0 && useScreenDensityPipeline;
    const useTemporalHistory = useFogPipeline && controls.fogTemporal && controls.fogTemporalBlend > 0;
    const useVolumeDensityTimings = useVolumeDensityPipeline && controls.densityPassStrength > 0;
    const useAccumulationTimings = useAccumulationPipeline;
    const rebuildVolumeDensity = useVolumeDensityPipeline && controls.densityPassStrength > 0 && this.shouldRebuildVolumeDensity(controls, stepSimulation);
    const useParticleSupportTimings = this.usesParticleSupportCompaction(controls);
    const usesFieldAlignedParticlePhase = controls.renderLayer !== "particles";
    const useTimestamps = profileGpu && this.timestampQueriesSupported && !!this.timestampQuerySet && this.shouldSampleTimestamps(stepSimulation);
    const screenDensityQueryCount = useScreenDensityTimings ? 6 : 0;
    const supportQueryCount = useParticleSupportTimings ? 6 : 0;
    const renderQueryCount = useFogPipeline
      ? (useTemporalHistory ? 6 : 4) + screenDensityQueryCount
      : useVolumeDensityTimings
        ? (stepSimulation ? 14 : rebuildVolumeDensity ? 14 : 2)
        : useAccumulationTimings
          ? 4
          : useScreenDensityTimings
            ? 8
            : 2;
    const postQueryCount = 2;
    const supportTimestampStart = stepSimulation ? 8 : 0;
    const renderTimestampStart = supportTimestampStart + supportQueryCount;
    const postTimestampStart = renderTimestampStart + renderQueryCount;
    const baseTimestampQueryCount = (stepSimulation ? 8 : 0) + supportQueryCount + renderQueryCount + postQueryCount;
    const timestampQueryCountUsed = baseTimestampQueryCount;
    const checkRenderValidation = !this.renderValidationChecked;
    if (checkRenderValidation) {
      device.pushErrorScope("validation");
    }

    let simStepsThisFrame = 0;
    let simulationTimeAdvance = 0;
    let simulationStepScales: number[] = [];
    if (stepSimulation) {
      const stepResult = this.stepSimulation(useTimestamps);
      simStepsThisFrame = stepResult.steps;
      simulationTimeAdvance = stepResult.timeAdvance;
      simulationStepScales = stepResult.stepScales;
      const { steps: _steps, stepScales: _stepScales, timeAdvance: _timeAdvance, ...stepTimings } = stepResult;
      Object.assign(timings, stepTimings);
    }
    const encoder = device.createCommandEncoder();
    // Ribbon trails: lazily create the history buffer, refresh the ribbon uniform, and (only while
    // stepping) append the new positions before the render pass reads them. Fully gated: off => skip.
    if (controls.ribbonFraction > 0.0001 && this.ribbonRenderPipeline) {
      this.ensureTrailBuffer();
      if (stepSimulation && simStepsThisFrame > 0) {
        this.ribbonFrame += 1;
      }
      this.writeRibbonUniform(controls);
      if (stepSimulation && simStepsThisFrame > 0) {
        this.encodeRibbonAppend(encoder);
      }
    }
    const visualFieldSmoothing = stepSimulation && this.shouldUseVisualFieldSmoothing(controls);
    const visualFieldTargetLerpT = visualFieldSmoothing ? this.renderLerpT : 1;
    const visualFieldPhaseReset = visualFieldSmoothing && simStepsThisFrame > 0;
    const visualFieldSourceBufferIndex = visualFieldSourceBufferIndexForFrame(this.readIndex, this.writeIndex, visualFieldPhaseReset);
    const visualFieldCanContinue =
      visualFieldSmoothing &&
      this.visualFieldActiveLastFrame &&
      this.visualFieldPrepared &&
      !visualFieldPhaseReset &&
      visualFieldTargetLerpT >= this.visualFieldLerpT;
    const syncVisualField = visualFieldSmoothing && !visualFieldCanContinue;
    const visualFieldCorrection = 0;
    const visualFieldStepScale = visualFieldSmoothing
      ? Math.max(0, syncVisualField ? visualFieldTargetLerpT : visualFieldTargetLerpT - this.visualFieldLerpT)
      : 0;
    if (visualFieldSmoothing) {
      timings.visualSmoothEncodeMs = this.encodeVisualFieldSmoothing(
        encoder,
        syncVisualField,
        visualFieldSourceBufferIndex,
        visualFieldStepScale,
        visualFieldTargetLerpT
      );
      this.visualFieldLerpT = visualFieldTargetLerpT;
      this.visualFieldActiveLastFrame = true;
    } else {
      this.visualFieldLerpT = 0;
      this.visualFieldActiveLastFrame = false;
    }
    const fieldSourceBuffer = visualFieldSmoothing
      ? this.visualFieldBuffers[this.visualFieldReadIndex]
      : this.fieldBuffers[this.readIndex];
    const particleRenderLerpT = stepSimulation
      ? (usesFieldAlignedParticlePhase ? (visualFieldSmoothing ? this.renderLerpT : 0) : this.renderLerpT)
      : 1;
    const particleSupportSourceBufferIndex = particleSourceBufferIndexForLerp(this.readIndex, particleRenderLerpT);
    const useFieldTexture = controls.fieldTextureSampling && useFogPipeline && !visualFieldSmoothing;
    if (useFieldTexture) {
      if (!stepSimulation) {
        device.queue.writeBuffer(this.simUniformBuffer!, 0, encodeSimUniforms(this.config, this.timestep, 0));
      }
      this.encodeFieldTextureUpdate(encoder);
    }

    const fogScale = useFogPipeline ? clampNumber(controls.fogRenderScale, 0.25, 1) : 1;
    const fogWidth = Math.max(1, Math.round(canvas.width * fogScale));
    const fogHeight = Math.max(1, Math.round(canvas.height * fogScale));
    const effectiveFogSteps = useFogPipeline ? clampRayStepCount(controls.raySteps * controls.fogStepScale) : controls.raySteps;
    const fogHistoryChanged = useTemporalHistory
      ? this.shouldResetFogHistory(controls, fogWidth, fogHeight, effectiveFogSteps, useFieldTexture)
      : this.clearFogHistoryView();
    if (fogHistoryChanged) {
      this.fogHistoryResetFrames = Math.max(this.fogHistoryResetFrames, 12);
    }
    const fogHistoryReset = useTemporalHistory && this.fogHistoryResetFrames > 0;
    const effectiveFogTemporalBlend = useTemporalHistory && !fogHistoryReset ? controls.fogTemporalBlend : 0;
    const fogNoiseFrame = controls.fogBlueNoise && effectiveFogTemporalBlend > 0 ? this.renderFrame : 0;
    const particleDensityReference = this.advanceParticleDensityReference();
    const particleGradientSensitivity = config.colorByCohort ? 0 : config.hueSensitivity;
    const colorModeTarget = particleColorModeToUniform(controls.particleColorMode);
    const colorBlendState = this.resolveColorBlend(colorModeTarget);
    const renderUniformValues: RenderUniforms = {
      resolution: [canvas.width, canvas.height],
      dimensions: [this.config.width, this.config.height, this.config.depth],
      particleCount: this.config.particleCount,
      voxelCount: this.voxelCount(),
      timestep: this.timestep,
      density: controls.density,
      exposure: controls.exposure,
      focusDistance: controls.focusDistance,
      aperture: controls.aperture,
      focalK: 1 / Math.tan((clampNumber(controls.fov, 10, 170) * Math.PI) / 360),
      dofBlur: controls.dofBlur,
      dofDebug: controls.dofDebug ? 1 : 0,
      dofEnabled: controls.dofEnabled ? 1 : 0,
      // Display-only interpolation factor between the previous and current whole-step particle
      // states. Field/trail-backed layers stay locked to their authored whole-step field phase.
      // Particle-only views keep interpolation; when they use a field density mask, the mask field
      // is rebuilt from the same visual particle phase below.
      renderLerpT: particleRenderLerpT,
      sceneBrightness: controls.sceneBrightness,
      overlay: overlay ? 1 : 0,
      palette: controls.palette === "aurora" ? 0 : controls.palette === "ember" ? 1 : 2,
      filament: controls.filament,
      yaw: controls.cameraYaw,
      pitch: controls.cameraPitch,
      yawCos: Math.cos(controls.cameraYaw),
      yawSin: Math.sin(controls.cameraYaw),
      pitchCos: Math.cos(controls.cameraPitch),
      pitchSin: Math.sin(controls.cameraPitch),
      distance: controls.cameraDistance,
      raySteps: controls.raySteps,
      fogStepScale: controls.fogStepScale,
      fogTemporalBlend: effectiveFogTemporalBlend,
      fogBlueNoise: controls.fogBlueNoise ? 1 : 0,
      fogFrame: fogNoiseFrame,
      fieldTextureMode: useFieldTexture ? 1 : 0,
      emptySpaceSkip: controls.emptySpaceSkipping ? 1 : 0,
      emptySpaceThreshold: controls.emptySpaceThreshold,
      emptySpaceStride: controls.emptySpaceStride,
      particleSizePx: controls.particleSizePx * particleResolutionScale,
      particleMinPx: controls.particleMinPx * particleResolutionScale,
      particleMaxPx: controls.particleMaxPx * particleResolutionScale,
      particleOpacity: controls.particleOpacity,
      particleBrightness: controls.particleBrightness,
      particleColorMode: colorModeTarget,
      particleColorModeFrom: colorBlendState.from,
      particleColorBlend: colorBlendState.blend,
      particleCutoffPrepass: controls.particleCutoffPrepass ? 1 : 0,
      particleVelocityStretch: controls.particleVelocityStretch ? 1 : 0,
      particleStretch: controls.particleStretch,
      particleStretchMin: controls.particleStretchMin,
      particleStretchSpeed: controls.particleStretchSpeed,
      particleSpeedCutoff: controls.particleSpeedCutoff,
      particleSlowCutoff: controls.particleSlowCutoff,
      particleGlowCore: controls.particleGlowCore,
      particleHotCore: controls.particleHotCore,
      particleExponent: controls.particleExponent,
      particleBrightnessBoost: controls.particleBrightnessBoost,
      particleSupportSmoothing: controls.particleSupportSmoothing,
      particleHazeCull: controls.particleHazeCull,
      particleGradientSensitivity,
      cohorts: config.cohorts,
      domainShape: config.domainShape,
      particleBlendMode: particleBlendModeToUniform(controls.particleBlendMode),
      particleDensityCutoff: controls.particleDensityCutoff,
      particleDensityRadius: controls.particleDensityRadius,
      particleDensityNormalize: controls.particleDensityNormalize,
      particleDensitySoftness: controls.particleDensitySoftness,
      particleDensityReference,
      particleSupportMask: controls.particleSupportMask,
      particleSupportRadius: controls.particleSupportRadius,
      particleSupportNeighbors: controls.particleSupportNeighbors,
      particleSupportFlow: controls.particleSupportFlow,
      particleSupportGridSize: this.particleSupportGridSize,
      trailOpacity: controls.trailOpacity,
      fogBrightness: controls.fogBrightness,
      trailThreshold: controls.trailThreshold,
      renderLayer: renderLayerToUniform(controls.renderLayer),
      trailColorMode: trailColorModeToUniform(controls.trailColorMode),
      cameraPanX: controls.cameraPanX,
      cameraPanY: controls.cameraPanY,
      fogTint: hexColorToRgb01(controls.fogTint),
      particleTint: hexColorToRgb01(controls.particleTint),
      densityPassStrength: controls.densityPassStrength,
      densitySmallScale: controls.densitySmallScale,
      densityLargeScale: controls.densityLargeScale,
      densityLargeThreshold: controls.densityLargeThreshold,
      densityContrastGain: controls.densityContrastGain,
      densityContrastBalance: controls.densityContrastBalance,
      densityEmissionPower: controls.densityEmissionPower,
      densityOcclusion: controls.densityOcclusion,
      accumulationStrength: controls.accumulationStrength,
      accumulationRadius: controls.accumulationRadius,
      accumulationCurve: controls.accumulationCurve,
      accumulationMemory: controls.accumulationMemory,
      accumulationNoiseReject: controls.accumulationNoiseReject,
      variationMaster: controls.variationMaster,
      // Smooth, frame-rate-stable clock for variation time drift. Uses the render frame counter
      // (not the integer sim timestep, which is steppy and pauses when the sim is paused).
      variationTime: this.renderFrame / 60,
      variationDrift: controls.variationDrift,
      variationNoiseMix: controls.variationNoiseMix,
      variationFreq: controls.variationFreq,
      variationOctaves: controls.variationOctaves,
      variationGain: controls.variationGain,
      variationLacunarity: controls.variationLacunarity,
      variationSizeAmount: controls.variationSizeAmount,
      variationSizeCurve: controls.variationSizeCurve,
      variationSizeMin: controls.variationSizeMin,
      variationSizeMax: controls.variationSizeMax,
      variationBrightAmount: controls.variationBrightAmount,
      variationBrightCurve: controls.variationBrightCurve,
      variationBrightMin: controls.variationBrightMin,
      variationBrightMax: controls.variationBrightMax,
      variationOpacityAmount: controls.variationOpacityAmount,
      variationOpacityCurve: controls.variationOpacityCurve,
      variationOpacityMin: controls.variationOpacityMin,
      variationOpacityMax: controls.variationOpacityMax,
      variationColorAmount: controls.variationColorAmount,
      variationColorCurve: controls.variationColorCurve,
      variationColorMin: controls.variationColorMin,
      variationColorMax: controls.variationColorMax,
      audioLow: this.audioLevels.low,
      audioMid: this.audioLevels.mid,
      audioHigh: this.audioLevels.high
    };
    const fogUniformValues: RenderUniforms = {
      ...renderUniformValues,
      resolution: [fogWidth, fogHeight],
      raySteps: effectiveFogSteps
    };
    device.queue.writeBuffer(this.renderUniformBuffer!, 0, encodeRenderUniforms(renderUniformValues));
    device.queue.writeBuffer(this.fogRenderUniformBuffer!, 0, encodeRenderUniforms(fogUniformValues));

    const renderPassStart = performance.now();
    this.encodeParticleSupportMask(
      encoder,
      controls,
      particleSupportSourceBufferIndex,
      useTimestamps && useParticleSupportTimings ? supportTimestampStart : undefined
    );
    this.encodeCutoffPrepass(encoder, controls, fieldSourceBuffer);
    this.encodeSplatPrepare(encoder, controls);
    const swapchainTexture = context.getCurrentTexture();
    const textureView = swapchainTexture.createView();
    this.ensureSceneTexture(canvas.width, canvas.height);
    this.encodeSplatAccumulate(encoder, controls);
    const sceneView = this.sceneTexture!.createView();
    if (useFogPipeline) {
      this.ensureFogTextures(fogWidth, fogHeight);
      const historyWriteIndex = 1 - this.fogHistoryReadIndex;
      const fogPass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: this.fogCurrentTexture!.createView(),
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
            loadOp: "clear",
            storeOp: "store"
          }
        ],
        ...(useTimestamps ? { timestampWrites: this.timestampWrites(renderTimestampStart, renderTimestampStart + 1) } : {})
      });
      fogPass.setPipeline(this.fieldFogRenderPipeline!);
      fogPass.setBindGroup(0, this.createFieldRenderBindGroup(this.fogRenderUniformBuffer!, this.fieldFogRenderPipeline!, fieldSourceBuffer));
      fogPass.draw(3);
      fogPass.end();

      const presentFogTexture = useTemporalHistory ? this.fogHistoryTextures[historyWriteIndex] : this.fogCurrentTexture!;
      if (useTemporalHistory) {
        const compositePass = encoder.beginRenderPass({
          colorAttachments: [
            {
              view: this.fogHistoryTextures[historyWriteIndex].createView(),
              clearValue: { r: 0, g: 0, b: 0, a: 0 },
              loadOp: "clear",
              storeOp: "store"
            }
          ],
          ...(useTimestamps ? { timestampWrites: this.timestampWrites(renderTimestampStart + 2, renderTimestampStart + 3) } : {})
        });
        compositePass.setPipeline(this.fogCompositePipeline!);
        compositePass.setBindGroup(0, this.createFogCompositeBindGroup(this.fogCurrentTexture!, this.fogHistoryTextures[this.fogHistoryReadIndex], this.fogRenderUniformBuffer!));
        compositePass.draw(3);
        compositePass.end();
      }

      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: sceneView,
            clearValue: { r: 0.004, g: 0.006, b: 0.008, a: 1 },
            loadOp: "clear",
            storeOp: "store"
          }
        ],
        ...(useTimestamps
          ? { timestampWrites: this.timestampWrites(useTemporalHistory ? renderTimestampStart + 4 : renderTimestampStart + 2, useTemporalHistory ? renderTimestampStart + 5 : renderTimestampStart + 3) }
          : {})
      });
      pass.setPipeline(this.fogCompositePipeline!);
      pass.setBindGroup(0, this.createFogCompositeBindGroup(presentFogTexture, presentFogTexture, this.renderUniformBuffer!));
      pass.draw(3);
      this.drawParticlesAndOverlay(pass, controls, overlay, fieldSourceBuffer);
      pass.end();
      this.renderDensityLayer(
        encoder,
        controls,
        presentFogTexture,
        useTimestamps && useScreenDensityTimings ? renderTimestampStart + (useTemporalHistory ? 6 : 4) : undefined,
        false,
        fieldSourceBuffer
      );
      if (useTemporalHistory) {
        this.fogHistoryReadIndex = historyWriteIndex;
      }
      if (this.fogHistoryResetFrames > 0) {
        this.fogHistoryResetFrames -= 1;
      }
    } else if (useAccumulationPipeline) {
      this.renderAccumulationLayer(
        encoder,
        controls,
        overlay,
        useTimestamps && useAccumulationTimings ? renderTimestampStart : undefined
      );
    } else if (useVolumeDensityPipeline) {
      this.renderVolumeDensityLayer(
        encoder,
        controls,
        overlay,
        rebuildVolumeDensity,
        useTimestamps ? renderTimestampStart : undefined
      );
    } else {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: sceneView,
            clearValue: { r: 0.004, g: 0.006, b: 0.008, a: 1 },
            loadOp: "clear",
            storeOp: "store"
          }
        ],
        ...(useTimestamps ? { timestampWrites: this.timestampWrites(renderTimestampStart, renderTimestampStart + 1) } : {})
      });
      if (controls.renderLayer !== "particles" && controls.renderLayer !== "density" && controls.renderLayer !== "volume-density" && controls.renderLayer !== "accumulation") {
        pass.setPipeline(this.fieldRenderPipeline!);
        pass.setBindGroup(0, this.createFieldRenderBindGroup(this.renderUniformBuffer!, this.fieldRenderPipeline!, fieldSourceBuffer));
        pass.draw(controls.renderLayer === "debug-voxels" ? this.voxelCount() * 6 : 3);
      }
      this.drawParticlesAndOverlay(pass, controls, overlay, fieldSourceBuffer);
      pass.end();
      this.renderDensityLayer(
        encoder,
        controls,
        undefined,
        useTimestamps && useScreenDensityTimings ? renderTimestampStart + 2 : undefined,
        false,
        fieldSourceBuffer
      );
    }
    // Post-process: Fog Glow bloom (Spencer 1995 PSF, reconstructed as a
    // Spencer-weighted sum of octave-Gaussian mips) + AgX grade into the swapchain.
    const bloomEnabled = controls.bloomStrength > 0.0001;
    const useNoBloomPost = !bloomEnabled && controls.fastNoBloomPost;
    const mips = bloomEnabled ? this.bloomTextures : [];
    const fogWeights = spencerFogGlowWeights(mips.length, controls.bloomRadius);
    device.queue.writeBuffer(
      this.postUniformBuffer!,
      0,
      encodePostUniforms({
        bloomStrength: controls.bloomStrength,
        bloomThreshold: controls.bloomThreshold,
        bloomRadius: controls.bloomRadius,
        colorSaturation: controls.colorSaturation,
        colorContrast: controls.colorContrast,
        tint: [1, 1, 1],
        levelCount: mips.length,
        weights: fogWeights,
        chromaticAberration: controls.chromaticAberration,
        vignetteStrength: controls.vignetteStrength,
        vignetteSoftness: controls.vignetteSoftness,
        streakStrength: controls.streakStrength,
        streakLength: controls.streakLength,
        streakVertical: controls.streakVertical,
        flareHeight: controls.flareHeight,
        flareCutoff: controls.flareCutoff,
        despeckle: controls.particleDespeckle,
        hdrGain: this.captureFinalHDR ? Math.max(0.0001, this.hdrCaptureHeadroom) : 0
      })
    );
    if (bloomEnabled && mips.length > 0) {
      // Build the octave-Gaussian pyramid: prefilter the bright-pass into mip0, then halve
      // down. The composite samples each mip directly with Spencer weights, so the Bloom slider
      // changes the actual final pixels without relying on hidden accumulated state in mip0.
      const prefilter = encoder.beginRenderPass({ colorAttachments: [{ view: mips[0].createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" }] });
      prefilter.setPipeline(this.bloomPrefilterPipeline!);
      prefilter.setBindGroup(0, this.createBloomBindGroup(this.bloomPrefilterPipeline!, this.sceneTexture!));
      prefilter.draw(3);
      prefilter.end();
      for (let i = 0; i < mips.length - 1; i += 1) {
        const down = encoder.beginRenderPass({ colorAttachments: [{ view: mips[i + 1].createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" }] });
        down.setPipeline(this.bloomDownsamplePipeline!);
        down.setBindGroup(0, this.createBloomBindGroup(this.bloomDownsamplePipeline!, mips[i]));
        down.draw(3);
        down.end();
      }
    }
    const postPass = encoder.beginRenderPass({
      colorAttachments: [{ view: textureView, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
      ...(useTimestamps ? { timestampWrites: this.timestampWrites(postTimestampStart, postTimestampStart + 1) } : {})
    });
    postPass.setPipeline(useNoBloomPost ? this.postNoBloomPipeline! : this.postPipeline!);
    postPass.setBindGroup(0, useNoBloomPost ? this.createPostNoBloomBindGroup(this.sceneTexture!) : this.createPostBindGroup(this.sceneTexture!, mips));
    postPass.draw(3);
    postPass.end();
    // Offline export: stash the post-processed frame before it presents. The canvas is configured
    // COPY_SRC (see webgpuCanvas.ts), so we can copy the swapchain into a persistent texture and
    // read the FINAL HDR (bloom + grade) back later — readCanvasHDR can't, the frame is gone by then.
    if (this.captureFinalHDR) {
      this.ensureFinalTexture(swapchainTexture.width, swapchainTexture.height);
      encoder.copyTextureToTexture(
        { texture: swapchainTexture },
        { texture: this.finalTexture! },
        { width: swapchainTexture.width, height: swapchainTexture.height }
      );
    }
    timings.renderPassEncodeMs = performance.now() - renderPassStart;
    if (useTimestamps) {
      encoder.resolveQuerySet(this.timestampQuerySet!, 0, timestampQueryCountUsed, this.timestampResolveBuffer!, 0);
      encoder.copyBufferToBuffer(this.timestampResolveBuffer!, 0, this.timestampReadBuffer!, 0, timestampQueryCountUsed * BigUint64Array.BYTES_PER_ELEMENT);
    }

    const submitStart = performance.now();
    device.queue.submit([encoder.finish()]);
    timings.submitMs = performance.now() - submitStart;
    const cpuFrameTimeMs = performance.now() - start;
    if (profileGpu) {
      const gpuWaitStart = performance.now();
      await device.queue.onSubmittedWorkDone();
      timings.gpuQueueWaitMs = performance.now() - gpuWaitStart;
    }
    const frameTimeMs = performance.now() - start;
    if (useTimestamps) {
      const timestampReadStart = performance.now();
      this.lastGpuPassTimings = await this.readGpuPassTimings(
        stepSimulation,
        useFogPipeline,
        useTemporalHistory,
        useVolumeDensityTimings,
        rebuildVolumeDensity,
        useScreenDensityTimings,
        useAccumulationTimings,
        useParticleSupportTimings
      );
      timings.gpuPassMs = this.lastGpuPassTimings;
      timings.timestampReadMs = performance.now() - timestampReadStart;
    }
    if (checkRenderValidation) {
      this.renderValidationChecked = true;
      const error = await device.popErrorScope();
      if (error) {
        throw new Error(`live WebGPU render validation failed: ${error.message}`);
      }
    }
    const normalizedDensityMaskActive = controls.particleDensityNormalize > 0.0001 && this.usesParticleSplatLayer(controls);
    const statsCadence = normalizedDensityMaskActive
      ? this.config.particleCount >= 1048576
        ? 96
        : this.config.particleCount >= 262144
          ? 48
          : 16
      : this.config.particleCount >= 1048576
        ? 512
        : this.config.particleCount >= 262144
          ? 128
          : this.config.particleCount >= 16384
            ? 64
            : 16;
    const needsDensityReference = normalizedDensityMaskActive && particleDensityReference <= 0;
    const fieldStatsBootstrapNeeded = (normalizedDensityMaskActive || controls.renderLayer !== "particles") &&
      this.lastFieldStats.nonzeroVoxels <= 0;
    const fieldStatsRefreshNeeded = normalizedDensityMaskActive || controls.renderLayer !== "particles";
    if (!profileGpu && stepSimulation && (needsDensityReference || (fieldStatsBootstrapNeeded && this.timestep <= 2) || (fieldStatsRefreshNeeded && this.timestep % statsCadence === 0))) {
      const statsStart = performance.now();
      this.scheduleFieldStatsRead();
      timings.statsReadMs = performance.now() - statsStart;
    }
    return {
      renderer: "webgpu-live-fluoddity-3d",
      webgpu: true,
      rulePort: "fluoddity-core-fourier",
      fieldKind: "vector-volume",
      adaptation: "two-plane-3d",
      renderMode: controls.renderLayer === "debug-voxels" ? "debug-voxel-splats" : controls.renderLayer === "density" ? "screen-space-density" : controls.renderLayer === "volume-density" ? "volume-density-raymarch" : controls.renderLayer === "accumulation" ? "accumulation" : controls.renderLayer === "particles" ? "particle-splats" : "volume-raymarch+particle-splats",
      depositMode: "particle-atomic-fixed-point",
      particleColorModel: controls.particleColorMode,
      particleSizeModel: "perspective-3d",
      trailColorMode: controls.trailColorMode,
      sceneBrightness: controls.sceneBrightness,
      particleBrightness: controls.particleBrightness,
      fogBrightness: controls.fogBrightness,
      particleVelocityStretch: controls.particleVelocityStretch,
      particleStretch: controls.particleStretch,
      particleStretchMin: controls.particleStretchMin,
      particleStretchSpeed: controls.particleStretchSpeed,
      particleSpeedCutoff: controls.particleSpeedCutoff,
      particleSlowCutoff: controls.particleSlowCutoff,
      particleGradientSensitivity,
      simulationSpeed: config.simulationSpeed,
      simulationTime: this.simulationTime,
      simulationTimeAdvance,
      simulationStepScales,
      depositTaps: Math.pow(this.config.depositTapRadius * 2 + 1, 3),
      depositScale: depositScale(this.config),
      sensorModel: "fluoddity-core-two-sensor-two-plane-3d",
      pheromoneModel: "vector-current-density-volume",
      particleCount: this.config.particleCount,
      voxelCount: this.voxelCount(),
      particleBufferBytes: this.config.particleCount * particleFloatCount * Float32Array.BYTES_PER_ELEMENT,
      fieldBufferBytes: this.voxelCount() * 4 * Float32Array.BYTES_PER_ELEMENT,
      requestedRenderLayer,
      effectiveRenderLayer: controls.renderLayer,
      deviceLimits: {
        maxBufferSize: device.limits.maxBufferSize,
        maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize
      },
      particleBlendMode: controls.particleBlendMode,
      particleDensityCutoff: controls.particleDensityCutoff,
      particleDensityRadius: controls.particleDensityRadius,
      particleDensityNormalize: controls.particleDensityNormalize,
      particleDensitySoftness: controls.particleDensitySoftness,
      particleDensityReference,
      particleDensityReferenceTarget: this.targetParticleDensityReference,
      statsReadPending: this.fieldStatsReadPending,
      particleSupportMask: controls.particleSupportMask,
      particleSupportRadius: controls.particleSupportRadius,
      particleSupportNeighbors: controls.particleSupportNeighbors,
      particleSupportFlow: controls.particleSupportFlow,
      particleSupportGridSize: this.particleSupportGridSize,
      particleSupportSourceBufferIndex,
      fastParticleRender: this.usesFastParticlePipeline(controls),
      cutoffPrepassActive: this.usesCutoffPrepass(controls),
      densityLargeHalfResActive: controls.densityLargeHalfRes && controls.renderLayer === "volume-density",
      splatPath: this.usesComputeSplat(controls) ? "compute" : this.usesPreparedSplatPath(controls) ? "prepared" : "classic",
      lastSortTimestep: this.lastSortTimestep,
      fastNoBloomPost: useNoBloomPost,
      timestep: this.timestep,
      renderFrame: this.renderFrame,
      renderLerpT: particleRenderLerpT,
      visualFieldSmoothing,
      visualFieldStepScale,
      visualFieldCorrection,
      visualFieldPhaseReset,
      visualFieldSourceBufferIndex,
      fogNoiseFrame,
      frameTimeMs,
      cpuFrameTimeMs,
      profileGpu,
      timestampQueries: this.timestampQueriesSupported,
      timings,
      renderResolution,
      particleResolutionScale,
      canvasColor: this.canvasColor!,
      raySteps: controls.raySteps,
      effectiveFogSteps,
      fogTemporal: useTemporalHistory,
      fogRenderScale: fogScale,
      fogTemporalBlend: effectiveFogTemporalBlend,
      fogHistoryReset,
      fieldTextureSampling: useFieldTexture,
      emptySpaceSkipping: controls.emptySpaceSkipping,
      emptySpaceThreshold: controls.emptySpaceThreshold,
      emptySpaceStride: controls.emptySpaceStride,
      dimensions: [this.config.width, this.config.height, this.config.depth],
      fieldStats: this.lastFieldStats,
      ruleEvidence: this.ruleEvidence,
      camera: {
        yaw: controls.cameraYaw,
        pitch: controls.cameraPitch,
        distance: controls.cameraDistance,
        panX: controls.cameraPanX,
        panY: controls.cameraPanY
      }
    };
  }

  private morphStepsRemaining = 0;

  private async ensureInitialized(canvas: HTMLCanvasElement, config: LiveGpu3dConfig): Promise<void> {
    if (!navigator.gpu) {
      this.failed = true;
      throw new Error("navigator.gpu unavailable");
    }
    const oldConfig = this.config;
    const configChanged = oldConfig !== config && JSON.stringify(oldConfig) !== JSON.stringify(config);
    if (configChanged) {
      this.config = config;
      this.ruleEvidence = computeRuleEvidence(config);
      if (requiresBufferReset(oldConfig, config)) {
        await this.reset();
      } else {
        // initialConditions changed without a buffer reset -> morph live toward the new layout
        // (guarded on existing buffers so the first init still seeds normally rather than morphing).
        if (this.particleChunks.length > 0 && oldConfig.initialConditions !== config.initialConditions) {
          this.morphStepsRemaining = initialMorphSteps;
        }
        if (this.ruleBuffer && !sameRule(oldConfig.rule, config.rule)) {
          this.device!.queue.writeBuffer(this.ruleBuffer, 0, createRuleData(config.rule));
        }
      }
    }
    if (!this.device) {
      const adapter = await requestHighPerformanceWebGpuAdapter(navigator.gpu);
      if (!adapter) {
        this.failed = true;
        throw new Error("No WebGPU adapter");
      }
      const requiredFeatures: GPUFeatureName[] = adapter.features.has("timestamp-query") ? ["timestamp-query"] : [];
      this.timestampQueriesSupported = requiredFeatures.length > 0;
      // float32-filterable lets the trail field live in an rgba32float 3D texture (bit-exact
      // f32 sensing through the texture cache) while fog keeps sampling it with a linear sampler.
      this.float32FilterableSupported = adapter.features.has("float32-filterable");
      if (this.float32FilterableSupported) {
        requiredFeatures.push("float32-filterable");
      }
      const descriptor: GPUDeviceDescriptor = {};
      if (requiredFeatures.length > 0) {
        descriptor.requiredFeatures = requiredFeatures;
      }
      this.device = await adapter.requestDevice(descriptor);
    }
    if (this.timestampQueriesSupported && !this.timestampQuerySet) {
      this.createTimestampResources();
    }
    if (!this.context) {
      const context = canvas.getContext("webgpu");
      if (!context) {
        this.failed = true;
        throw new Error("No WebGPU canvas context");
      }
      this.context = context;
    }
    const canvasColor = configureHdrWebGpuCanvas(this.context, this.device);
    if (this.format && this.format !== canvasColor.format) {
      this.resetPipelines();
    }
    this.format = canvasColor.format;
    this.canvasColor = canvasColor;
    if (this.particleChunks.length === 0) {
      this.createBuffers();
    }
    if (!this.updatePipeline) {
      await this.createPipelines();
    }
    if (this.particleInitializationPending) {
      this.initializeParticleBuffers();
    }
  }

  private createBuffers(): void {
    const device = this.device!;
    const field = new Float32Array(this.voxelCount() * 4);
    const particleBytes = particleBufferByteLength(this.config);
    const fieldBytes = field.byteLength;
    const initializeOnGpu = particleBytes > defaultMaxBufferSize;
    const particlesPerChunk = maxParticlesPerStorageBuffer(device.limits);
    this.particleSupportGridSize = particleSupportGridSizeForLimits(device.limits);
    this.particleChunks = [];
    for (let offset = 0; offset < this.config.particleCount; offset += particlesPerChunk) {
      const count = Math.min(particlesPerChunk, this.config.particleCount - offset);
      const size = particleBufferByteLength({ particleCount: count });
      this.particleChunks.push({
        offset,
        count,
        buffers: [
          device.createBuffer({
            size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
          }),
          device.createBuffer({
            size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
          })
        ],
        supportBuffer: device.createBuffer({
          size: count * Float32Array.BYTES_PER_ELEMENT,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        }),
        cutoffSignalBuffer: device.createBuffer({
          size: count * Float32Array.BYTES_PER_ELEMENT,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        }),
        activeIndexBuffer: device.createBuffer({
          size: count * Uint32Array.BYTES_PER_ELEMENT,
          usage: GPUBufferUsage.STORAGE
        }),
        activeDrawBuffer: device.createBuffer({
          size: 4 * Uint32Array.BYTES_PER_ELEMENT,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST
        }),
        splatRecordBuffer: device.createBuffer({
          size: count * 32,
          usage: GPUBufferUsage.STORAGE
        }),
        sortIdBuffer: device.createBuffer({
          size: count * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        }),
        sortScratchBuffer: device.createBuffer({
          size: particleBufferByteLength({ particleCount: count }),
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        }),
        sortScratchIdBuffer: device.createBuffer({
          size: count * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        }),
        sortMapBuffer: device.createBuffer({
          size: count * 4,
          usage: GPUBufferUsage.STORAGE
        })
      });
      // Seed persistent ids with the global particle index; the sort permutes them in lockstep.
      const ids = new Uint32Array(count);
      for (let k = 0; k < count; k += 1) ids[k] = offset + k;
      device.queue.writeBuffer(this.particleChunks[this.particleChunks.length - 1].sortIdBuffer, 0, ids);
    }
    this.fieldBuffers = [
      device.createBuffer({
        size: fieldBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
      }),
      device.createBuffer({
        size: fieldBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
      })
    ];
    this.visualFieldBuffers = [
      device.createBuffer({
        size: fieldBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
      }),
      device.createBuffer({
        size: fieldBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
      })
    ];
    this.particleSupportGridBuffer = device.createBuffer({
      size: particleSupportGridByteLength(this.particleSupportGridSize),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    this.fieldTexture = device.createTexture({
      size: { width: this.config.width, height: this.config.height, depthOrArrayLayers: this.config.depth },
      dimension: "3d",
      // rgba32float (when filterable) keeps texture-path sensing bit-exact with the f32
      // storage buffer; rgba16float fallback only serves the fog raymarch as before.
      format: this.fieldTextureFormat(),
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
    });
    this.fieldTextureSampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear"
    });
    // Sensing sampler: repeat addressing reproduces the manual periodic-wrap trilinear.
    this.fieldSimSampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "repeat",
      addressModeV: "repeat",
      addressModeW: "repeat"
    });
    this.brushBuffer = device.createBuffer({
      size: fieldBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
    this.ruleBuffer = device.createBuffer({
      size: 80 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    if (initializeOnGpu) {
      this.particleInitializationPending = true;
    } else {
      for (const chunk of this.particleChunks) {
        const particles = createInitialParticles(this.config, chunk.offset, chunk.count);
        device.queue.writeBuffer(chunk.buffers[0], 0, particles);
        device.queue.writeBuffer(chunk.buffers[1], 0, particles);
      }
    }
    device.queue.writeBuffer(this.fieldBuffers[0], 0, field);
    device.queue.writeBuffer(this.fieldBuffers[1], 0, field);
    this.fieldTextureDirty = true;
    device.queue.writeBuffer(this.visualFieldBuffers[0], 0, field);
    device.queue.writeBuffer(this.visualFieldBuffers[1], 0, field);
    device.queue.writeBuffer(this.ruleBuffer, 0, createRuleData(this.config.rule));
    this.simUniformBuffer = device.createBuffer({
      size: SIM_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.chunkSimUniformBuffers = this.particleChunks.map(() => device.createBuffer({
      size: SIM_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    }));
    this.renderUniformBuffer = device.createBuffer({
      size: renderUniformByteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.fogRenderUniformBuffer = device.createBuffer({
      size: renderUniformByteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
  }

  private destroyBuffers(): void {
    for (const buffer of [...this.particleChunks.flatMap((chunk) => chunk.buffers), ...this.fieldBuffers, ...this.visualFieldBuffers]) {
      buffer.destroy();
    }
    for (const chunk of this.particleChunks) {
      chunk.supportBuffer.destroy();
      chunk.cutoffSignalBuffer.destroy();
      chunk.activeIndexBuffer.destroy();
      chunk.activeDrawBuffer.destroy();
      chunk.splatRecordBuffer.destroy();
      chunk.sortIdBuffer.destroy();
      chunk.sortScratchBuffer.destroy();
      chunk.sortScratchIdBuffer.destroy();
      chunk.sortMapBuffer.destroy();
    }
    this.particleSupportGridBuffer?.destroy();
    this.trailBuffer?.destroy();
    this.trailBuffer = undefined;
    this.ribbonUniformBuffer?.destroy();
    this.ribbonUniformBuffer = undefined;
    this.brushBuffer?.destroy();
    this.ruleBuffer?.destroy();
    this.ecologyFieldBuffer?.destroy();
    this.ecologyBrushBuffer?.destroy();
    this.ecologyFieldBuffer = undefined;
    this.ecologyBrushBuffer = undefined;
    this.ecologyClearBindGroup = undefined;
    this.ecologyUpdateBindGroup = undefined;
    this.ecologyDepositBindGroups = [];
    this.ecologyParticleBindGroups = [];
    this.simUniformBuffer?.destroy();
    for (const buffer of this.chunkSimUniformBuffers) {
      buffer.destroy();
    }
    this.renderUniformBuffer?.destroy();
    this.fogRenderUniformBuffer?.destroy();
    this.destroyFogTextures();
    this.destroyAccumulationTextures();
    this.destroyVolumeDensityResources();
    this.particleChunks = [];
    this.fieldBuffers = [];
    this.visualFieldBuffers = [];
    this.brushBuffer = undefined;
    this.ruleBuffer = undefined;
    this.simUniformBuffer = undefined;
    this.chunkSimUniformBuffers = [];
    this.renderUniformBuffer = undefined;
    this.fogRenderUniformBuffer = undefined;
    this.particleSupportGridBuffer = undefined;
    this.particleSupportGridSize = defaultParticleSupportGridSize;
    this.clearBindGroup = undefined;
    this.particleInitializationPending = false;
    this.depositBindGroups = [];
    this.visualDepositBindGroups = [];
    this.updateBindGroups = [];
    this.particleSupportClearBindGroup = undefined;
    this.particleSupportBuildBindGroups = [];
    this.particleSupportResolveBindGroups = [];
    this.fieldBindGroups = [];
    this.visualFieldBindGroups = [];
    this.fieldTextureBindGroups = [];
    this.fieldRenderBindGroups = [];
    this.particleRenderBindGroups = {};
    this.splatPrepareBindGroups = [];
    this.splatDrawBindGroups = {};
    this.splatAccumBuffer?.destroy();
    this.splatAccumBuffer = undefined;
    this.splatAccumBindGroups = [];
    this.splatResolveBindGroup = undefined;
    this.sortHistBuffer?.destroy();
    this.sortHistBuffer = undefined;
    this.sortHistBindGroups = [];
    this.sortScanBindGroup = undefined;
    this.sortScatterBindGroups = [];
    this.sortApplyBindGroups = [];
    this.particleRenderBindGroupLayout = undefined;
    this.densitySplatBindGroups = [];
    this.accumulationSplatBindGroups = [];
    this.densityCompositeBindGroup = undefined;
    this.densityCompositeFogTexture = undefined;
    this.overlayBindGroup = undefined;
    this.fieldTexture?.destroy();
    this.fieldTexture = undefined;
    this.fieldTextureSampler = undefined;
    this.fieldSimSampler = undefined;
    this.updateTextureBindGroups = [];
    this.fieldStatsReadPending = false;
    this.fieldStatsReadGeneration += 1;
    this.visualFieldReadIndex = 0;
    this.visualFieldLerpT = 0;
    this.visualFieldPrepared = false;
    this.visualFieldActiveLastFrame = false;
  }

  private destroyDeviceResources(): void {
    this.destroyBuffers();
    this.destroyPostResources();
    this.timestampQuerySet?.destroy();
    this.timestampResolveBuffer?.destroy();
    this.timestampReadBuffer?.destroy();
    this.timestampQuerySet = undefined;
    this.timestampResolveBuffer = undefined;
    this.timestampReadBuffer = undefined;
    this.resetPipelines();
    this.device?.destroy();
    this.device = undefined;
    this.timestampQueriesSupported = false;
  }

  private destroyPostResources(): void {
    this.sceneTexture?.destroy();
    for (const texture of this.bloomTextures) {
      texture.destroy();
    }
    this.destroyDensityTextures();
    this.destroyAccumulationTextures();
    this.postUniformBuffer?.destroy();
    this.sceneTexture = undefined;
    this.sceneTextureSize = [0, 0];
    this.bloomTextures = [];
    this.postUniformBuffer = undefined;
    this.postSampler = undefined;
  }

  private destroyDensityTextures(): void {
    this.densitySmallTexture?.destroy();
    this.densityLargeTexture?.destroy();
    this.densityEmptyFogTexture?.destroy();
    this.densitySmallTexture = undefined;
    this.densityLargeTexture = undefined;
    this.densityEmptyFogTexture = undefined;
    this.densityTextureSize = [0, 0];
    this.densitySampler = undefined;
    this.densitySplatBindGroups = [];
    this.densityCompositeBindGroup = undefined;
    this.densityCompositeFogTexture = undefined;
  }

  private destroyAccumulationTextures(): void {
    this.accumulationCurrentTexture?.destroy();
    for (const texture of this.accumulationHistoryTextures) {
      texture.destroy();
    }
    this.accumulationCurrentTexture = undefined;
    this.accumulationHistoryTextures = [];
    this.accumulationTextureSize = [0, 0];
    this.accumulationSampler = undefined;
    this.accumulationSplatBindGroups = [];
    this.accumulationHistoryReadIndex = 0;
    this.accumulationHistoryKey = "";
    this.accumulationHistoryPrepared = false;
  }

  private destroyVolumeDensityResources(): void {
    this.volumeDensityAccumBuffer?.destroy();
    this.volumeDensitySmallBuffer?.destroy();
    this.volumeDensityBlurTempBuffer?.destroy();
    this.volumeDensityLargeBuffer?.destroy();
    this.volumeDensitySmallTexture?.destroy();
    this.volumeDensityLargeTexture?.destroy();
    this.volumeDensitySmallHalfBuffer?.destroy();
    this.volumeDensityBlurTempHalfBuffer?.destroy();
    this.volumeDensityLargeHalfBuffer?.destroy();
    this.volumeDensityLargeTextureHalf?.destroy();
    this.volumeDensityAccumBuffer = undefined;
    this.volumeDensitySmallBuffer = undefined;
    this.volumeDensityBlurTempBuffer = undefined;
    this.volumeDensityLargeBuffer = undefined;
    this.volumeDensitySmallTexture = undefined;
    this.volumeDensityLargeTexture = undefined;
    this.volumeDensitySmallHalfBuffer = undefined;
    this.volumeDensityBlurTempHalfBuffer = undefined;
    this.volumeDensityLargeHalfBuffer = undefined;
    this.volumeDensityLargeTextureHalf = undefined;
    this.volumeDensitySampler = undefined;
    this.volumeDensityResourceSize = [0, 0, 0];
    this.volumeDensityClearBindGroup = undefined;
    this.volumeDensityDepositBindGroups = [];
    this.volumeDensityResolveBindGroup = undefined;
    this.volumeDensityBlurXBindGroup = undefined;
    this.volumeDensityBlurYBindGroup = undefined;
    this.volumeDensityBlurZBindGroup = undefined;
    this.volumeDensityDownsampleBindGroup = undefined;
    this.volumeDensityBlurXHalfBindGroup = undefined;
    this.volumeDensityBlurYHalfBindGroup = undefined;
    this.volumeDensityBlurZHalfBindGroup = undefined;
    this.volumeDensityRenderBindGroup = undefined;
    this.volumeDensityRenderHalfBindGroup = undefined;
    this.volumeDensityPreparedKey = "";
    this.volumeDensityPrepared = false;
  }

  private createTimestampResources(): void {
    const device = this.device!;
    this.timestampQuerySet = device.createQuerySet({
      type: "timestamp",
      count: timestampQueryCount
    });
    this.timestampResolveBuffer = device.createBuffer({
      size: timestampQueryBufferSize,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
    });
    this.timestampReadBuffer = device.createBuffer({
      size: timestampQueryBufferSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
  }

  private async createPipelines(): Promise<void> {
    const device = this.device!;
    const tasks: Array<() => Promise<unknown>> = [];
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: liveComputeClearBrushShader }),
        entryPoint: "clear_brush"
      }
    }).then((p) => { this.clearBrushPipeline = p; }))
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: liveComputeInitParticleShader }),
        entryPoint: "init_particles"
      }
    }).then((p) => { this.initParticlePipeline = p; }))
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: liveComputeDepositShader }),
        entryPoint: "deposit_particles"
      }
    }).then((p) => { this.depositPipeline = p; }))
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: liveComputeVisualDepositShader }),
        entryPoint: "deposit_visual_particles"
      }
    }).then((p) => { this.visualDepositPipeline = p; }))
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: makeLiveComputeParticleShader(false) }),
        entryPoint: "update_particles"
      }
    }).then((p) => { this.updatePipeline = p; }))
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: makeLiveComputeParticleShader(true) }),
        entryPoint: "update_particles"
      }
    }).then((p) => { this.updateTexturePipeline = p; }))
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: makeLiveComputeFieldShader(true, this.fieldTextureFormat()) }),
        entryPoint: "update_field"
      }
    }).then((p) => { this.fieldPipeline = p; }))
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: makeLiveComputeFieldShader(false) }),
        entryPoint: "update_field"
      }
    }).then((p) => { this.visualFieldSmoothPipeline = p; }))
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: makeLiveComputeFieldTextureShader(this.fieldTextureFormat()) }),
        entryPoint: "update_field_texture"
      }
    }).then((p) => { this.fieldTexturePipeline = p; }))
    const particleSupportModule = device.createShaderModule({ code: liveParticleSupportComputeShader });
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module: particleSupportModule,
        entryPoint: "clear_particle_support_grid"
      }
    }).then((p) => { this.particleSupportClearPipeline = p; }))
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module: particleSupportModule,
        entryPoint: "build_particle_support_grid"
      }
    }).then((p) => { this.particleSupportBuildPipeline = p; }))
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module: particleSupportModule,
        entryPoint: "resolve_particle_support"
      }
    }).then((p) => { this.particleSupportResolvePipeline = p; }))
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: liveCutoffPrepassShader }),
        entryPoint: "cutoff_prepass"
      }
    }).then((p) => { this.cutoffPrepassPipeline = p; }))
    const fieldRenderModule = device.createShaderModule({ code: liveVolumeRenderShader });
    const particleRenderModule = device.createShaderModule({ code: liveParticleRenderShader });
    const densitySplatModule = device.createShaderModule({ code: liveDensitySplatShader });
    const densityCompositeModule = device.createShaderModule({ code: liveDensityCompositeShader });
    const accumulationSplatModule = device.createShaderModule({ code: liveAccumulationSplatShader });
    const accumulationCompositeModule = device.createShaderModule({ code: liveAccumulationCompositeShader });
    const volumeDensityComputeModule = device.createShaderModule({ code: liveVolumeDensityComputeShader });
    const volumeDensityRenderModule = device.createShaderModule({ code: liveVolumeDensityRenderShader });
    const fogCompositeModule = device.createShaderModule({ code: liveFogCompositeShader });
    const postModule = device.createShaderModule({ code: livePostShader });
    const overlayModule = device.createShaderModule({ code: liveOverlayShader });
    this.particleRenderBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 5, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 6, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 7, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } }
      ]
    });
    const particleRenderPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.particleRenderBindGroupLayout]
    });
    const additiveBlend = {
      // Clean additive of the already-premultiplied source (color*intensity). srcFactor "one"
      // (not "src-alpha") so intensity is applied once, consistent with the alpha blend.
      color: {
        srcFactor: "one" as GPUBlendFactor,
        dstFactor: "one" as GPUBlendFactor,
        operation: "add" as GPUBlendOperation
      },
      alpha: {
        srcFactor: "one" as GPUBlendFactor,
        dstFactor: "one-minus-src-alpha" as GPUBlendFactor,
        operation: "add" as GPUBlendOperation
      }
    };
    const alphaBlend = {
      color: {
        srcFactor: "one" as GPUBlendFactor,
        dstFactor: "one-minus-src-alpha" as GPUBlendFactor,
        operation: "add" as GPUBlendOperation
      },
      alpha: {
        srcFactor: "one" as GPUBlendFactor,
        dstFactor: "one-minus-src-alpha" as GPUBlendFactor,
        operation: "add" as GPUBlendOperation
      }
    };
    tasks.push(() => device.createRenderPipelineAsync({
      layout: "auto",
      vertex: {
        module: fieldRenderModule,
        entryPoint: "field_vs"
      },
      fragment: {
        module: fieldRenderModule,
        entryPoint: "volume_fs",
        targets: [{ format: this.format!, blend: additiveBlend }]
      },
      primitive: { topology: "triangle-list" }
    }).then((p) => { this.fieldRenderPipeline = p; }))
    tasks.push(() => device.createRenderPipelineAsync({
      layout: "auto",
      vertex: {
        module: fieldRenderModule,
        entryPoint: "field_vs"
      },
      fragment: {
        module: fieldRenderModule,
        entryPoint: "volume_fs",
        targets: [{ format: this.format! }]
      },
      primitive: { topology: "triangle-list" }
    }).then((p) => { this.fieldFogRenderPipeline = p; }))
    tasks.push(() => device.createRenderPipelineAsync({
      layout: "auto",
      vertex: {
        module: fogCompositeModule,
        entryPoint: "fullscreen_vs"
      },
      fragment: {
        module: fogCompositeModule,
        entryPoint: "fog_composite_fs",
        targets: [{ format: this.format! }]
      },
      primitive: { topology: "triangle-list" }
    }).then((p) => { this.fogCompositePipeline = p; }))
    tasks.push(() => device.createRenderPipelineAsync({
      layout: "auto",
      vertex: {
        module: densitySplatModule,
        entryPoint: "density_small_vs"
      },
      fragment: {
        module: densitySplatModule,
        entryPoint: "density_splat_fs",
        targets: [{ format: densityTextureFormat, blend: additiveBlend }]
      },
      primitive: { topology: "triangle-list" }
    }).then((p) => { this.densitySmallPipeline = p; }))
    tasks.push(() => device.createRenderPipelineAsync({
      layout: "auto",
      vertex: {
        module: densitySplatModule,
        entryPoint: "density_large_vs"
      },
      fragment: {
        module: densitySplatModule,
        entryPoint: "density_splat_fs",
        targets: [{ format: densityTextureFormat, blend: additiveBlend }]
      },
      primitive: { topology: "triangle-list" }
    }).then((p) => { this.densityLargePipeline = p; }))
    tasks.push(() => device.createRenderPipelineAsync({
      layout: "auto",
      vertex: {
        module: densityCompositeModule,
        entryPoint: "fullscreen_vs"
      },
      fragment: {
        module: densityCompositeModule,
        entryPoint: "density_composite_fs",
        targets: [{ format: this.format!, blend: additiveBlend }]
      },
      primitive: { topology: "triangle-list" }
    }).then((p) => { this.densityCompositePipeline = p; }))
    tasks.push(() => device.createRenderPipelineAsync({
      layout: "auto",
      vertex: {
        module: accumulationSplatModule,
        entryPoint: "accumulation_vs"
      },
      fragment: {
        module: accumulationSplatModule,
        entryPoint: "accumulation_splat_fs",
        targets: [{ format: accumulationTextureFormat, blend: additiveBlend }]
      },
      primitive: { topology: "triangle-list" }
    }).then((p) => { this.accumulationSplatPipeline = p; }))
    tasks.push(() => device.createRenderPipelineAsync({
      layout: "auto",
      vertex: {
        module: accumulationCompositeModule,
        entryPoint: "fullscreen_vs"
      },
      fragment: {
        module: accumulationCompositeModule,
        entryPoint: "accumulation_composite_fs",
        targets: [
          { format: this.format!, blend: additiveBlend },
          { format: accumulationTextureFormat }
        ]
      },
      primitive: { topology: "triangle-list" }
    }).then((p) => { this.accumulationCompositePipeline = p; }))
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module: volumeDensityComputeModule,
        entryPoint: "clear_volume_density"
      }
    }).then((p) => { this.volumeDensityClearPipeline = p; }))
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module: volumeDensityComputeModule,
        entryPoint: "deposit_volume_density"
      }
    }).then((p) => { this.volumeDensityDepositPipeline = p; }))
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module: volumeDensityComputeModule,
        entryPoint: "resolve_volume_density"
      }
    }).then((p) => { this.volumeDensityResolvePipeline = p; }))
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module: volumeDensityComputeModule,
        entryPoint: "blur_volume_density_x"
      }
    }).then((p) => { this.volumeDensityBlurXPipeline = p; }))
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module: volumeDensityComputeModule,
        entryPoint: "blur_volume_density_y"
      }
    }).then((p) => { this.volumeDensityBlurYPipeline = p; }))
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module: volumeDensityComputeModule,
        entryPoint: "blur_volume_density_z"
      }
    }).then((p) => { this.volumeDensityBlurZPipeline = p; }))
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: volumeDensityComputeModule, entryPoint: "downsample_volume_small" }
    }).then((p) => { this.volumeDensityDownsamplePipeline = p; }))
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: volumeDensityComputeModule, entryPoint: "blur_volume_density_x_half" }
    }).then((p) => { this.volumeDensityBlurXHalfPipeline = p; }))
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: volumeDensityComputeModule, entryPoint: "blur_volume_density_y_half" }
    }).then((p) => { this.volumeDensityBlurYHalfPipeline = p; }))
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: volumeDensityComputeModule, entryPoint: "blur_volume_density_z_half" }
    }).then((p) => { this.volumeDensityBlurZHalfPipeline = p; }))
    tasks.push(() => device.createRenderPipelineAsync({
      layout: "auto",
      vertex: {
        module: volumeDensityRenderModule,
        entryPoint: "volume_density_vs"
      },
      fragment: {
        module: volumeDensityRenderModule,
        entryPoint: "volume_density_fs",
        targets: [{ format: this.format!, blend: additiveBlend }]
      },
      primitive: { topology: "triangle-list" }
    }).then((p) => { this.volumeDensityRenderPipeline = p; }))
    const postTarget: GPUColorTargetState = { format: this.format! };
    // Upsample accumulation: standard dual-filter (additive coarse->fine 2x tent), which gives
    // a smooth wide veil. Over 8 octaves this fuses the particle cores into one soft fog-glow
    // halo (no blocky direct upscale, no per-core clumping).
    const bloomAddTarget: GPUColorTargetState = {
      format: this.format!,
      blend: {
        color: { srcFactor: "one", dstFactor: "one", operation: "add" },
        alpha: { srcFactor: "one", dstFactor: "one", operation: "add" }
      }
    };
    tasks.push(() => device.createRenderPipelineAsync({
      layout: "auto",
      vertex: { module: postModule, entryPoint: "fullscreen_vs" },
      fragment: { module: postModule, entryPoint: "prefilter_fs", targets: [postTarget] },
      primitive: { topology: "triangle-list" }
    }).then((p) => { this.bloomPrefilterPipeline = p; }))
    tasks.push(() => device.createRenderPipelineAsync({
      layout: "auto",
      vertex: { module: postModule, entryPoint: "fullscreen_vs" },
      fragment: { module: postModule, entryPoint: "downsample_fs", targets: [postTarget] },
      primitive: { topology: "triangle-list" }
    }).then((p) => { this.bloomDownsamplePipeline = p; }))
    tasks.push(() => device.createRenderPipelineAsync({
      layout: "auto",
      vertex: { module: postModule, entryPoint: "fullscreen_vs" },
      fragment: { module: postModule, entryPoint: "upsample_fs", targets: [bloomAddTarget] },
      primitive: { topology: "triangle-list" }
    }).then((p) => { this.bloomUpsamplePipeline = p; }))
    tasks.push(() => device.createRenderPipelineAsync({
      layout: "auto",
      vertex: { module: postModule, entryPoint: "fullscreen_vs" },
      fragment: { module: postModule, entryPoint: "composite_fs", targets: [postTarget] },
      primitive: { topology: "triangle-list" }
    }).then((p) => { this.postPipeline = p; }))
    tasks.push(() => device.createRenderPipelineAsync({
      layout: "auto",
      vertex: { module: postModule, entryPoint: "fullscreen_vs" },
      fragment: { module: postModule, entryPoint: "composite_no_bloom_fs", targets: [postTarget] },
      primitive: { topology: "triangle-list" }
    }).then((p) => { this.postNoBloomPipeline = p; }))
    const particleTargets: Record<ParticleBlendMode, GPUColorTargetState> = {
      additive: { format: this.format!, blend: additiveBlend },
      alpha: { format: this.format!, blend: alphaBlend },
      opaque: { format: this.format! }
    };
    for (const mode of particleBlendModes) {
      tasks.push(() => device.createRenderPipelineAsync({
        layout: particleRenderPipelineLayout,
        vertex: {
          module: particleRenderModule,
          entryPoint: "particle_vs"
        },
        fragment: {
          module: particleRenderModule,
          entryPoint: "splat_fs",
          targets: [particleTargets[mode]]
        },
        primitive: { topology: "triangle-strip" }
      }).then((p) => { this.particleRenderPipelines[mode] = p; }))
      tasks.push(() => device.createRenderPipelineAsync({
        layout: particleRenderPipelineLayout,
        vertex: {
          module: particleRenderModule,
          entryPoint: "particle_fast_vs"
        },
        fragment: {
          module: particleRenderModule,
          entryPoint: "splat_fs",
          targets: [particleTargets[mode]]
        },
        primitive: { topology: "triangle-strip" }
      }).then((p) => { this.particleFastRenderPipelines[mode] = p; }))
    }
    const splatDrawModule = device.createShaderModule({ code: liveSplatDrawShader });
    for (const mode of particleBlendModes) {
      tasks.push(() => device.createRenderPipelineAsync({
        layout: "auto",
        vertex: { module: splatDrawModule, entryPoint: "particle_record_vs" },
        fragment: { module: splatDrawModule, entryPoint: "splat_fs", targets: [particleTargets[mode]] },
        primitive: { topology: "triangle-strip" }
      }).then((p) => { this.splatDrawPipelines[mode] = p; }))
    }
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: liveSplatPrepareShader }), entryPoint: "prepare_splats" }
    }).then((p) => { this.splatPreparePipeline = p; }))
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: liveSplatAccumShader }), entryPoint: "accumulate_splats" }
    }).then((p) => { this.splatAccumPipeline = p; }))
    const sortModule = device.createShaderModule({ code: liveSortShader });
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: sortModule, entryPoint: "sort_histogram" }
    }).then((p) => { this.sortHistogramPipeline = p; }))
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: sortModule, entryPoint: "sort_scan" }
    }).then((p) => { this.sortScanPipeline = p; }))
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: sortModule, entryPoint: "sort_scatter" }
    }).then((p) => { this.sortScatterPipeline = p; }))
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: liveSortApplyShader }), entryPoint: "sort_apply_map" }
    }).then((p) => { this.sortApplyPipeline = p; }))
    const splatResolveModule = device.createShaderModule({ code: liveSplatResolveShader });
    tasks.push(() => device.createRenderPipelineAsync({
      layout: "auto",
      vertex: { module: splatResolveModule, entryPoint: "splat_resolve_vs" },
      fragment: { module: splatResolveModule, entryPoint: "splat_resolve_fs", targets: [particleTargets.additive] },
      primitive: { topology: "triangle-list" }
    }).then((p) => { this.splatResolvePipeline = p; }))
    tasks.push(() => device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: liveRibbonAppendShader }), entryPoint: "append_trail" }
    }).then((p) => { this.trailAppendPipeline = p; }))
    tasks.push(() => device.createRenderPipelineAsync({
      layout: "auto",
      vertex: { module: device.createShaderModule({ code: liveRibbonRenderShader }), entryPoint: "ribbon_vs" },
      fragment: {
        module: device.createShaderModule({ code: liveRibbonRenderShader }),
        entryPoint: "ribbon_fs",
        targets: [{ format: this.format!, blend: additiveBlend }]
      },
      primitive: { topology: "triangle-strip" }
    }).then((p) => { this.ribbonRenderPipeline = p; }))
    tasks.push(() => device.createRenderPipelineAsync({
      layout: "auto",
      vertex: {
        module: overlayModule,
        entryPoint: "fullscreen_vs"
      },
      fragment: {
        module: overlayModule,
        entryPoint: "overlay_fs",
        targets: [{ format: this.format!, blend: additiveBlend }]
      },
      primitive: { topology: "triangle-list" }
    }).then((p) => { this.overlayPipeline = p; }))
    if (this.parallelPipelineCompile) {
      await Promise.all(tasks.map((task) => task()));
    } else {
      for (const task of tasks) {
        await task();
      }
    }
  }

  private initializeParticleBuffers(): void {
    const device = this.device!;
    if (!this.initParticlePipeline || !this.simUniformBuffer || this.particleChunks.length === 0) {
      return;
    }
    for (const chunk of this.particleChunks) {
      device.queue.writeBuffer(this.simUniformBuffer, 0, encodeSimUniforms(this.config, 0, 0, chunk.offset, chunk.count));
      const encoder = device.createCommandEncoder();
      const initialBuffer = chunk.buffers[0];
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.initParticlePipeline);
      pass.setBindGroup(0, device.createBindGroup({
        layout: this.initParticlePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: initialBuffer } },
          { binding: 1, resource: { buffer: this.simUniformBuffer } }
        ]
      }));
      pass.dispatchWorkgroups(Math.ceil(chunk.count / 256));
      pass.end();
      encoder.copyBufferToBuffer(initialBuffer, 0, chunk.buffers[1], 0, particleBufferByteLength({ particleCount: chunk.count }));
      device.queue.submit([encoder.finish()]);
    }
    this.particleInitializationPending = false;
  }

  private stepSimulation(useTimestamps: boolean): Pick<LiveRenderTimings, "encodeStepMs" | "clearBrushEncodeMs" | "depositEncodeMs" | "particleUpdateEncodeMs" | "fieldUpdateEncodeMs"> & { steps: number; stepScales: number[]; timeAdvance: number } {
    const device = this.device!;
    const timings = {
      encodeStepMs: 0,
      clearBrushEncodeMs: 0,
      depositEncodeMs: 0,
      particleUpdateEncodeMs: 0,
      fieldUpdateEncodeMs: 0
    };
    const plan = planSimulationSteps(this.simStepAccumulator, this.config.simulationSpeed);
    this.simStepAccumulator = plan.accumulator;
    this.renderLerpT = plan.renderLerpT;
    for (let s = 0; s < plan.stepScales.length; s += 1) {
      const stepScale = plan.stepScales[s];
      const encoder = device.createCommandEncoder();
      if (this.shouldSortAtCurrentStep()) {
        this.encodeSortPass(encoder);
      }
      const stepTimings = this.encodeStepPasses(encoder, useTimestamps && s === 0, stepScale);
      device.queue.submit([encoder.finish()]);
      [this.readIndex, this.writeIndex] = [this.writeIndex, this.readIndex];
      this.timestep += 1;
      this.simulationTime += stepScale;
      timings.encodeStepMs += stepTimings.encodeStepMs;
      timings.clearBrushEncodeMs += stepTimings.clearBrushEncodeMs;
      timings.depositEncodeMs += stepTimings.depositEncodeMs;
      timings.particleUpdateEncodeMs += stepTimings.particleUpdateEncodeMs;
      timings.fieldUpdateEncodeMs += stepTimings.fieldUpdateEncodeMs;
    }
    return { ...timings, steps: plan.steps, stepScales: plan.stepScales, timeAdvance: plan.timeAdvance };
  }

  private encodeStepPasses(encoder: GPUCommandEncoder, useTimestamps: boolean, stepTimeScale: number): Pick<LiveRenderTimings, "encodeStepMs" | "clearBrushEncodeMs" | "depositEncodeMs" | "particleUpdateEncodeMs" | "fieldUpdateEncodeMs"> {
    const stepStart = performance.now();
    const timings = {
      encodeStepMs: 0,
      clearBrushEncodeMs: 0,
      depositEncodeMs: 0,
      particleUpdateEncodeMs: 0,
      fieldUpdateEncodeMs: 0
    };
    const device = this.device!;
    const morphAlpha = this.morphStepsRemaining > 0 ? initialMorphAlpha : 0;
    if (this.morphStepsRemaining > 0) {
      this.morphStepsRemaining -= 1;
    }
    device.queue.writeBuffer(this.simUniformBuffer!, 0, encodeSimUniforms(this.config, this.timestep, stepTimeScale));
    if (this.fieldTextureDirty && this.usesFieldTextureSensing() && this.fieldTexturePipeline) {
      // Field buffers were uploaded directly (seed/cache/zero); re-mirror them into the
      // sensing texture before the update pass reads it.
      const syncPass = encoder.beginComputePass();
      syncPass.setPipeline(this.fieldTexturePipeline);
      syncPass.setBindGroup(0, this.createFieldTextureBindGroup());
      syncPass.dispatchWorkgroups(
        Math.ceil(this.config.width / 4),
        Math.ceil(this.config.height / 4),
        Math.ceil(this.config.depth / 4)
      );
      syncPass.end();
      this.fieldTextureDirty = false;
    }
    let passStart = performance.now();
    const clearPass = encoder.beginComputePass(useTimestamps ? { timestampWrites: this.timestampWrites(0, 1) } : undefined);
    clearPass.setPipeline(this.clearBrushPipeline!);
    clearPass.setBindGroup(0, this.getClearBindGroup());
    clearPass.dispatchWorkgroups(Math.ceil((this.voxelCount() * 4) / 256));
    clearPass.end();
    timings.clearBrushEncodeMs = performance.now() - passStart;
    for (let chunkIndex = 0; chunkIndex < this.particleChunks.length; chunkIndex += 1) {
      const chunk = this.particleChunks[chunkIndex];
      device.queue.writeBuffer(this.chunkSimUniformBuffers[chunkIndex], 0, encodeSimUniforms(this.config, this.timestep, stepTimeScale, chunk.offset, chunk.count, stepTimeScale, morphAlpha));
    }
    passStart = performance.now();
    const depositPass = encoder.beginComputePass(useTimestamps ? { timestampWrites: this.timestampWrites(2, 3) } : undefined);
    depositPass.setPipeline(this.depositPipeline!);
    for (let chunkIndex = 0; chunkIndex < this.particleChunks.length; chunkIndex += 1) {
      const chunk = this.particleChunks[chunkIndex];
      const bindGroups = this.getParticleStepBindGroups(chunkIndex);
      depositPass.setBindGroup(0, bindGroups.deposit);
      depositPass.dispatchWorkgroups(Math.ceil(chunk.count / 256));
    }
    depositPass.end();
    timings.depositEncodeMs = performance.now() - passStart;
    // Ecology field passes (species/predator/alarm). Only encoded when active; otherwise the step is
    // exactly the classic clear -> deposit -> update -> field sequence and stays byte-identical.
    const ecologyReady = this.ecologyReady();
    if (ecologyReady) {
      const ecoClear = encoder.beginComputePass();
      ecoClear.setPipeline(this.clearBrushPipeline!);
      ecoClear.setBindGroup(0, this.getEcologyClearBindGroup());
      ecoClear.dispatchWorkgroups(Math.ceil((this.voxelCount() * 4) / 256));
      ecoClear.end();
      const ecoDeposit = encoder.beginComputePass();
      ecoDeposit.setPipeline(this.ecologyDepositPipeline!);
      for (let chunkIndex = 0; chunkIndex < this.particleChunks.length; chunkIndex += 1) {
        const chunk = this.particleChunks[chunkIndex];
        ecoDeposit.setBindGroup(0, this.getEcologyDepositBindGroup(chunkIndex));
        ecoDeposit.dispatchWorkgroups(Math.ceil(chunk.count / 256));
      }
      ecoDeposit.end();
      const ecoUpdate = encoder.beginComputePass();
      ecoUpdate.setPipeline(this.ecologyUpdatePipeline!);
      ecoUpdate.setBindGroup(0, this.getEcologyUpdateBindGroup());
      ecoUpdate.dispatchWorkgroups(Math.ceil(this.voxelCount() / 256));
      ecoUpdate.end();
    }
    passStart = performance.now();
    const updatePass = encoder.beginComputePass(useTimestamps ? { timestampWrites: this.timestampWrites(4, 5) } : undefined);
    updatePass.setPipeline(ecologyReady ? this.ecologyParticlePipeline! : (this.usesFieldTextureSensing() ? this.updateTexturePipeline! : this.updatePipeline!));
    for (let chunkIndex = 0; chunkIndex < this.particleChunks.length; chunkIndex += 1) {
      const chunk = this.particleChunks[chunkIndex];
      const update = ecologyReady
        ? this.getEcologyParticleBindGroup(chunkIndex)
        : this.getParticleStepBindGroups(chunkIndex).update;
      updatePass.setBindGroup(0, update);
      updatePass.dispatchWorkgroups(Math.ceil(chunk.count / 256));
    }
    updatePass.end();
    timings.particleUpdateEncodeMs = performance.now() - passStart;
    device.queue.writeBuffer(this.simUniformBuffer!, 0, encodeSimUniforms(this.config, this.timestep, stepTimeScale));
    passStart = performance.now();
    const fieldPass = encoder.beginComputePass(useTimestamps ? { timestampWrites: this.timestampWrites(6, 7) } : undefined);
    fieldPass.setPipeline(this.fieldPipeline!);
    fieldPass.setBindGroup(0, this.getFieldStepBindGroup());
    fieldPass.dispatchWorkgroups(Math.ceil(this.voxelCount() / 256));
    fieldPass.end();
    timings.fieldUpdateEncodeMs = performance.now() - passStart;
    timings.encodeStepMs = performance.now() - stepStart;
    return timings;
  }

  private shouldUseVisualFieldSmoothing(controls: RenderControls): boolean {
    // Only particle-only density masks need a render-time visual field. The simulation still reads
    // and writes whole-step field buffers, so changing sim speed cannot feed fractional fields back
    // into the forces.
    if (controls.renderLayer !== "particles") {
      return false;
    }
    return controls.particleDensityCutoff > 0 || controls.particleDensityNormalize > 0.0001;
  }

  private encodeVisualFieldSmoothing(
    encoder: GPUCommandEncoder,
    syncFromSimulation: boolean,
    sourceFieldBufferIndex: number,
    stepTimeScale: number,
    visualLerpT: number
  ): number {
    const start = performance.now();
    const device = this.device!;
    const fieldBytes = this.voxelCount() * 4 * Float32Array.BYTES_PER_ELEMENT;
    if (!this.visualFieldPrepared || syncFromSimulation) {
      const sourceIndex = sourceFieldBufferIndex === 0 ? 0 : 1;
      encoder.copyBufferToBuffer(this.fieldBuffers[sourceIndex], 0, this.visualFieldBuffers[0], 0, fieldBytes);
      encoder.copyBufferToBuffer(this.fieldBuffers[sourceIndex], 0, this.visualFieldBuffers[1], 0, fieldBytes);
      this.visualFieldReadIndex = 0;
      this.visualFieldPrepared = true;
    }
    const scale = clampNumber(stepTimeScale, 0, 1);
    if (scale <= 0) {
      return performance.now() - start;
    }

    const clearPass = encoder.beginComputePass();
    clearPass.setPipeline(this.clearBrushPipeline!);
    clearPass.setBindGroup(0, this.getClearBindGroup());
    clearPass.dispatchWorkgroups(Math.ceil((this.voxelCount() * 4) / 256));
    clearPass.end();

    for (let chunkIndex = 0; chunkIndex < this.particleChunks.length; chunkIndex += 1) {
      const chunk = this.particleChunks[chunkIndex];
      device.queue.writeBuffer(
        this.chunkSimUniformBuffers[chunkIndex],
        0,
        encodeSimUniforms(this.config, this.timestep, scale, chunk.offset, chunk.count, visualLerpT)
      );
      const depositPass = encoder.beginComputePass();
      depositPass.setPipeline(this.visualDepositPipeline!);
      depositPass.setBindGroup(0, this.getVisualDepositBindGroup(chunkIndex));
      depositPass.dispatchWorkgroups(Math.ceil(chunk.count / 256));
      depositPass.end();
    }

    device.queue.writeBuffer(this.simUniformBuffer!, 0, encodeSimUniforms(this.config, this.timestep, scale, 0, this.config.particleCount, visualLerpT));
    const fieldPass = encoder.beginComputePass();
    fieldPass.setPipeline(this.visualFieldSmoothPipeline!);
    fieldPass.setBindGroup(0, this.getVisualFieldBindGroup());
    fieldPass.dispatchWorkgroups(Math.ceil(this.voxelCount() / 256));
    fieldPass.end();
    this.visualFieldReadIndex = this.visualFieldReadIndex === 0 ? 1 : 0;
    return performance.now() - start;
  }

  private timestampWrites(beginningOfPassWriteIndex: number, endOfPassWriteIndex: number): GPUComputePassTimestampWrites {
    return {
      querySet: this.timestampQuerySet!,
      beginningOfPassWriteIndex,
      endOfPassWriteIndex
    };
  }

  private encodeFieldTextureUpdate(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.fieldTexturePipeline!);
    pass.setBindGroup(0, this.createFieldTextureBindGroup());
    pass.dispatchWorkgroups(
      Math.ceil(this.config.width / 4),
      Math.ceil(this.config.height / 4),
      Math.ceil(this.config.depth / 4)
    );
    pass.end();
  }

  private shouldSampleTimestamps(playing: boolean): boolean {
    if (!this.lastGpuPassTimings) return true;
    return playing ? this.timestep % 24 === 0 : this.timestampFrame % 24 === 0;
  }

  private async readGpuPassTimings(
    playing: boolean,
    temporalFog: boolean,
    temporalHistory: boolean,
    volumeDensity: boolean,
    volumeDensityRebuilt: boolean,
    screenDensity: boolean,
    accumulation: boolean,
    particleSupport: boolean
  ): Promise<LiveGpuPassTimings | null> {
    const buffer = this.timestampReadBuffer;
    if (!buffer) return null;
    await buffer.mapAsync(GPUMapMode.READ);
    const values = new BigUint64Array(buffer.getMappedRange().slice(0));
    buffer.unmap();
    const deltaMs = (start: number, end: number) => {
      if (values.length <= end || values[end] < values[start]) return 0;
      return Number(values[end] - values[start]) / 1_000_000;
    };
    const densityTimings = (base: number) => {
      if (!screenDensity) {
        return { densitySmallMs: 0, densityLargeMs: 0, densityCompositeMs: 0 };
      }
      return {
        densitySmallMs: deltaMs(base, base + 1),
        densityLargeMs: deltaMs(base + 2, base + 3),
        densityCompositeMs: deltaMs(base + 4, base + 5)
      };
    };
    const emptyAccumulation = { accumulationSplatMs: 0, accumulationCompositeMs: 0 };
    const emptySupport: Pick<LiveGpuPassTimings, "particleSupportClearMs" | "particleSupportBuildMs" | "particleSupportResolveMs"> = {
      particleSupportClearMs: 0,
      particleSupportBuildMs: 0,
      particleSupportResolveMs: 0
    };
    const supportBase = playing ? 8 : 0;
    const support: Pick<LiveGpuPassTimings, "particleSupportClearMs" | "particleSupportBuildMs" | "particleSupportResolveMs"> = particleSupport
      ? {
          particleSupportClearMs: deltaMs(supportBase, supportBase + 1),
          particleSupportBuildMs: deltaMs(supportBase + 2, supportBase + 3),
          particleSupportResolveMs: deltaMs(supportBase + 4, supportBase + 5)
        }
      : emptySupport;
    const supportMs = support.particleSupportClearMs + support.particleSupportBuildMs + support.particleSupportResolveMs;
    const renderBase = supportBase + (particleSupport ? 6 : 0);
    const screenDensityQueryCount = screenDensity ? 6 : 0;
    const renderQueryCount = temporalFog
      ? (temporalHistory ? 6 : 4) + screenDensityQueryCount
      : volumeDensity
        ? (playing ? 14 : volumeDensityRebuilt ? 14 : 2)
        : accumulation
          ? 4
          : screenDensity
            ? 8
            : 2;
    const postBase = renderBase + renderQueryCount;
    const postMs = deltaMs(postBase, postBase + 1);
    if (!playing) {
      if (accumulation) {
        const accumulationSplatMs = deltaMs(renderBase, renderBase + 1);
        const accumulationCompositeMs = deltaMs(renderBase + 2, renderBase + 3);
        const renderMs = accumulationSplatMs + accumulationCompositeMs;
        return {
          clearBrushMs: 0,
          depositMs: 0,
          particleUpdateMs: 0,
          fieldUpdateMs: 0,
          ...support,
          densitySmallMs: 0,
          densityLargeMs: 0,
          densityCompositeMs: 0,
          accumulationSplatMs,
          accumulationCompositeMs,
          volumeDensityClearMs: 0,
          volumeDensityDepositMs: 0,
          volumeDensityResolveMs: 0,
          volumeDensityBlurMs: 0,
          volumeDensityRaymarchMs: 0,
          fogCompositeMs: 0,
          fogPresentMs: 0,
          renderMs,
          postMs,
          totalMeasuredMs: supportMs + renderMs + postMs
        };
      }
      if (volumeDensity) {
        if (!volumeDensityRebuilt) {
          const volumeDensityRaymarchMs = deltaMs(renderBase, renderBase + 1);
          return {
            clearBrushMs: 0,
            depositMs: 0,
            particleUpdateMs: 0,
            fieldUpdateMs: 0,
            ...support,
            densitySmallMs: 0,
            densityLargeMs: 0,
            densityCompositeMs: 0,
            ...emptyAccumulation,
            volumeDensityClearMs: 0,
            volumeDensityDepositMs: 0,
            volumeDensityResolveMs: 0,
            volumeDensityBlurMs: 0,
            volumeDensityRaymarchMs,
            fogCompositeMs: 0,
            fogPresentMs: 0,
            renderMs: volumeDensityRaymarchMs,
            postMs,
            totalMeasuredMs: supportMs + volumeDensityRaymarchMs + postMs
          };
        }
        const volumeDensityClearMs = deltaMs(renderBase, renderBase + 1);
        const volumeDensityDepositMs = deltaMs(renderBase + 2, renderBase + 3);
        const volumeDensityResolveMs = deltaMs(renderBase + 4, renderBase + 5);
        const volumeDensityBlurMs = deltaMs(renderBase + 6, renderBase + 7) + deltaMs(renderBase + 8, renderBase + 9) + deltaMs(renderBase + 10, renderBase + 11);
        const volumeDensityRaymarchMs = deltaMs(renderBase + 12, renderBase + 13);
        const renderMs = volumeDensityClearMs + volumeDensityDepositMs + volumeDensityResolveMs + volumeDensityBlurMs + volumeDensityRaymarchMs;
        return {
          clearBrushMs: 0,
          depositMs: 0,
          particleUpdateMs: 0,
          fieldUpdateMs: 0,
          ...support,
          densitySmallMs: 0,
          densityLargeMs: 0,
          densityCompositeMs: 0,
          ...emptyAccumulation,
          volumeDensityClearMs,
          volumeDensityDepositMs,
          volumeDensityResolveMs,
          volumeDensityBlurMs,
          volumeDensityRaymarchMs,
          fogCompositeMs: 0,
          fogPresentMs: 0,
          renderMs,
          postMs,
          totalMeasuredMs: supportMs + renderMs + postMs
        };
      }
      const fogMs = deltaMs(renderBase, renderBase + 1);
      const fogCompositeMs = temporalFog && temporalHistory ? deltaMs(renderBase + 2, renderBase + 3) : 0;
      const fogPresentMs = temporalFog ? deltaMs(renderBase + (temporalHistory ? 4 : 2), renderBase + (temporalHistory ? 5 : 3)) : 0;
      const density = densityTimings(renderBase + (temporalFog ? (temporalHistory ? 6 : 4) : 2));
      const densityMs = density.densitySmallMs + density.densityLargeMs + density.densityCompositeMs;
      const renderMs = fogMs + fogCompositeMs + fogPresentMs + densityMs;
      return {
        clearBrushMs: 0,
        depositMs: 0,
        particleUpdateMs: 0,
        fieldUpdateMs: 0,
        ...support,
        ...density,
        ...emptyAccumulation,
        volumeDensityClearMs: 0,
        volumeDensityDepositMs: 0,
        volumeDensityResolveMs: 0,
        volumeDensityBlurMs: 0,
        volumeDensityRaymarchMs: 0,
        fogCompositeMs,
        fogPresentMs,
        renderMs,
        postMs,
        totalMeasuredMs: supportMs + renderMs + postMs
      };
    }
    const clearBrushMs = deltaMs(0, 1);
    const depositMs = deltaMs(2, 3);
    const particleUpdateMs = deltaMs(4, 5);
    const fieldUpdateMs = deltaMs(6, 7);
    if (accumulation) {
      const accumulationSplatMs = deltaMs(renderBase, renderBase + 1);
      const accumulationCompositeMs = deltaMs(renderBase + 2, renderBase + 3);
      const renderMs = accumulationSplatMs + accumulationCompositeMs;
      return {
        clearBrushMs,
        depositMs,
        particleUpdateMs,
        fieldUpdateMs,
        ...support,
        densitySmallMs: 0,
        densityLargeMs: 0,
        densityCompositeMs: 0,
        accumulationSplatMs,
        accumulationCompositeMs,
        volumeDensityClearMs: 0,
        volumeDensityDepositMs: 0,
        volumeDensityResolveMs: 0,
        volumeDensityBlurMs: 0,
        volumeDensityRaymarchMs: 0,
        fogCompositeMs: 0,
        fogPresentMs: 0,
        renderMs,
        postMs,
        totalMeasuredMs: clearBrushMs + depositMs + particleUpdateMs + fieldUpdateMs + supportMs + renderMs + postMs
      };
    }
    if (volumeDensity) {
      const volumeDensityClearMs = deltaMs(renderBase, renderBase + 1);
      const volumeDensityDepositMs = deltaMs(renderBase + 2, renderBase + 3);
      const volumeDensityResolveMs = deltaMs(renderBase + 4, renderBase + 5);
      const volumeDensityBlurMs = deltaMs(renderBase + 6, renderBase + 7) + deltaMs(renderBase + 8, renderBase + 9) + deltaMs(renderBase + 10, renderBase + 11);
      const volumeDensityRaymarchMs = deltaMs(renderBase + 12, renderBase + 13);
      const renderMs = volumeDensityClearMs + volumeDensityDepositMs + volumeDensityResolveMs + volumeDensityBlurMs + volumeDensityRaymarchMs;
      return {
        clearBrushMs,
        depositMs,
        particleUpdateMs,
        fieldUpdateMs,
        ...support,
        densitySmallMs: 0,
        densityLargeMs: 0,
        densityCompositeMs: 0,
        ...emptyAccumulation,
        volumeDensityClearMs,
        volumeDensityDepositMs,
        volumeDensityResolveMs,
        volumeDensityBlurMs,
        volumeDensityRaymarchMs,
        fogCompositeMs: 0,
        fogPresentMs: 0,
        renderMs,
        postMs,
        totalMeasuredMs: clearBrushMs + depositMs + particleUpdateMs + fieldUpdateMs + supportMs + renderMs + postMs
      };
    }
    const fogMs = deltaMs(renderBase, renderBase + 1);
    const fogCompositeMs = temporalFog && temporalHistory ? deltaMs(renderBase + 2, renderBase + 3) : 0;
    const fogPresentMs = temporalFog ? deltaMs(renderBase + (temporalHistory ? 4 : 2), renderBase + (temporalHistory ? 5 : 3)) : 0;
    const density = densityTimings(renderBase + (temporalFog ? (temporalHistory ? 6 : 4) : 2));
    const densityMs = density.densitySmallMs + density.densityLargeMs + density.densityCompositeMs;
    const renderMs = fogMs + fogCompositeMs + fogPresentMs + densityMs;
    return {
      clearBrushMs,
      depositMs,
      particleUpdateMs,
      fieldUpdateMs,
      ...support,
      ...density,
      ...emptyAccumulation,
      volumeDensityClearMs: 0,
      volumeDensityDepositMs: 0,
      volumeDensityResolveMs: 0,
      volumeDensityBlurMs: 0,
      volumeDensityRaymarchMs: 0,
      fogCompositeMs,
      fogPresentMs,
      renderMs,
      postMs,
      totalMeasuredMs: clearBrushMs + depositMs + particleUpdateMs + fieldUpdateMs + supportMs + renderMs + postMs
    };
  }

  // Read the current canvas swap-chain texture back to CPU as Float32 HDR.
  // The canvas is rgba16float (4 half-floats per pixel); we decode to a plain
  // Float32Array of length width*height*4. Used by parity tests for true
  // linear-space color comparison (bypasses Chrome's tonemap+sRGB PNG path).
  async readCanvasHDR(): Promise<{ width: number; height: number; pixels: Float32Array } | null> {
    const device = this.device;
    const context = this.context;
    if (!device || !context) return null;
    let tex: GPUTexture;
    try {
      tex = context.getCurrentTexture();
    } catch {
      return null;
    }
    return this.readRgba16FloatTexture(tex, tex.width, tex.height);
  }

  async readSceneHDR(): Promise<{ width: number; height: number; pixels: Float32Array } | null> {
    if (!this.sceneTexture || this.sceneTextureSize[0] <= 0 || this.sceneTextureSize[1] <= 0) return null;
    return this.readRgba16FloatTexture(this.sceneTexture, this.sceneTextureSize[0], this.sceneTextureSize[1]);
  }

  // Final post-processed HDR frame (bloom/vignette/grade applied) captured during render while
  // captureFinalHDR was set. This is what the offline HDR export reads — the full performance look.
  async readFinalHDR(): Promise<{ width: number; height: number; pixels: Float32Array } | null> {
    if (!this.finalTexture || this.finalTextureSize[0] <= 0 || this.finalTextureSize[1] <= 0) return null;
    return this.readRgba16FloatTexture(this.finalTexture, this.finalTextureSize[0], this.finalTextureSize[1]);
  }

  // Raw post-processed HDR frame: the mapped rgba16float bytes (with row padding), NOT decoded to
  // float in JS. The offline export uses this — the per-pixel half->float decode loop and base64 of
  // a 16-byte/px float array were ~1.3s/frame; shipping the raw 8-byte/px halves and decoding in node
  // roughly halves both the work and the transfer.
  async readFinalHDRRaw(): Promise<{ width: number; height: number; bytesPerRow: number; data: Uint8Array } | null> {
    if (!this.finalTexture || this.finalTextureSize[0] <= 0 || this.finalTextureSize[1] <= 0) return null;
    const device = this.device!;
    const W = this.finalTextureSize[0];
    const H = this.finalTextureSize[1];
    const bytesPerRow = Math.ceil((W * 8) / 256) * 256;
    const buf = device.createBuffer({ size: bytesPerRow * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: this.finalTexture }, { buffer: buf, bytesPerRow }, { width: W, height: H });
    device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const data = new Uint8Array(buf.getMappedRange()).slice(); // copy out before unmap
    buf.unmap();
    buf.destroy();
    return { width: W, height: H, bytesPerRow, data };
  }

  private ensureFinalTexture(width: number, height: number): void {
    if (this.finalTexture && this.finalTextureSize[0] === width && this.finalTextureSize[1] === height) return;
    this.finalTexture?.destroy();
    this.finalTexture = this.device!.createTexture({
      size: { width, height },
      format: this.format!,
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING
    });
    this.finalTextureSize = [width, height];
  }

  private async readRgba16FloatTexture(texture: GPUTexture, W: number, H: number): Promise<{ width: number; height: number; pixels: Float32Array }> {
    const device = this.device!;
    const bytesPerRow = Math.ceil((W * 8) / 256) * 256; // rgba16float = 8 bytes/pixel, row aligned to 256
    const buf = device.createBuffer({
      size: bytesPerRow * H,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture }, { buffer: buf, bytesPerRow }, { width: W, height: H });
    device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const range = buf.getMappedRange();
    const view = new DataView(range);
    const pixels = new Float32Array(W * H * 4);
    // Decode rgba16float row-by-row, skipping the row-padding bytes.
    for (let y = 0; y < H; y++) {
      const rowOffset = y * bytesPerRow;
      for (let x = 0; x < W; x++) {
        const i = rowOffset + x * 8;
        const d = (y * W + x) * 4;
        pixels[d + 0] = halfToFloat(view.getUint16(i + 0, true));
        pixels[d + 1] = halfToFloat(view.getUint16(i + 2, true));
        pixels[d + 2] = halfToFloat(view.getUint16(i + 4, true));
        pixels[d + 3] = halfToFloat(view.getUint16(i + 6, true));
      }
    }
    buf.unmap();
    buf.destroy();
    return { width: W, height: H, pixels };
  }

  // Inject deterministic particle data for parity tests. Writes to both
  // ping-pong buffers so the next render reads our data regardless of readIndex.
  // Caller must match this.config.particleCount * 12 floats.
  setParticleBufferData(data: Float32Array): void {
    const device = this.device;
    if (!device || this.particleChunks.length === 0) return;
    const expected = this.config.particleCount * particleFloatCount * Float32Array.BYTES_PER_ELEMENT;
    if (data.byteLength !== expected) {
      throw new Error(`setParticleBufferData: expected ${expected} bytes, got ${data.byteLength}`);
    }
    for (const chunk of this.particleChunks) {
      const start = chunk.offset * particleFloatCount;
      const end = start + chunk.count * particleFloatCount;
      const chunkData = data.subarray(start, end);
      device.queue.writeBuffer(chunk.buffers[0], 0, chunkData);
      device.queue.writeBuffer(chunk.buffers[1], 0, chunkData);
    }
    this.volumeDensityPrepared = false;
    this.volumeDensityPreparedKey = "";
    this.accumulationHistoryPrepared = false;
    this.accumulationHistoryKey = "";
    this.fieldStatsReadGeneration += 1;
    this.fieldStatsReadPending = false;
    this.lastFieldStats = emptyStats;
    this.targetParticleDensityReference = 0;
    this.smoothedParticleDensityReference = 0;
  }

  // Clear the field buffer (used during parity tests to remove any deposited fog).
  clearField(): void {
    const device = this.device;
    if (!device || this.fieldBuffers.length === 0) return;
    const zeros = new Float32Array(this.voxelCount() * 4);
    device.queue.writeBuffer(this.fieldBuffers[0], 0, zeros);
    device.queue.writeBuffer(this.fieldBuffers[1], 0, zeros);
    this.fieldTextureDirty = true;
    this.volumeDensityPrepared = false;
    this.volumeDensityPreparedKey = "";
    this.accumulationHistoryPrepared = false;
    this.accumulationHistoryKey = "";
  }

  // Seed deterministic field values for render parity/color tests that need the
  // density gates to see local support without advancing the simulation.
  setFieldBufferData(data: Float32Array): void {
    const device = this.device;
    if (!device || this.fieldBuffers.length === 0) return;
    const expected = this.voxelCount() * 4 * Float32Array.BYTES_PER_ELEMENT;
    if (data.byteLength !== expected) {
      throw new Error(`setFieldBufferData: expected ${expected} bytes, got ${data.byteLength}`);
    }
    device.queue.writeBuffer(this.fieldBuffers[0], 0, data);
    device.queue.writeBuffer(this.fieldBuffers[1], 0, data);
    this.fieldTextureDirty = true;
    this.fieldStatsReadGeneration += 1;
    this.fieldStatsReadPending = false;
    this.applyFieldStats(fieldStatsFromValues(data));
    this.volumeDensityPrepared = false;
    this.volumeDensityPreparedKey = "";
    this.accumulationHistoryPrepared = false;
    this.accumulationHistoryKey = "";
  }

  async captureState(): Promise<CapturedSimState | null> {
    const device = this.device;
    if (!device || this.fieldBuffers.length === 0 || this.particleChunks.length === 0) {
      return null;
    }
    const field = await this.readBackF32(this.fieldBuffers[this.readIndex], this.voxelCount() * 4);
    const particles = await this.captureParticles();
    if (!particles) return null;
    return {
      field,
      particles,
      voxelCount: this.voxelCount(),
      particleCount: this.config.particleCount,
      width: this.config.width,
      height: this.config.height,
      depth: this.config.depth
    };
  }

  // Particle-only readback for camera tracking (skips the heavy field buffer).
  async captureParticles(maxParticles = this.config.particleCount): Promise<Float32Array | null> {
    const device = this.device;
    if (!device || this.particleChunks.length === 0) return null;
    const particleCount = Math.max(0, Math.min(this.config.particleCount, Math.floor(maxParticles)));
    const out = new Float32Array(particleCount * particleFloatCount);
    let copied = 0;
    let allocated = 0;
    let cumulative = 0;
    for (let chunkIndex = 0; chunkIndex < this.particleChunks.length; chunkIndex += 1) {
      const chunk = this.particleChunks[chunkIndex];
      if (copied >= particleCount) break;
      cumulative += chunk.count;
      const targetThroughChunk = chunkIndex === this.particleChunks.length - 1
        ? particleCount
        : Math.round((cumulative / Math.max(1, this.config.particleCount)) * particleCount);
      const count = Math.min(chunk.count, Math.max(0, targetThroughChunk - allocated), particleCount - copied);
      allocated += count;
      if (count === 0) continue;
      const values = await this.readBackF32(chunk.buffers[this.readIndex], count * particleFloatCount);
      out.set(values, copied * particleFloatCount);
      copied += count;
    }
    return out;
  }

  get liveParticleCount(): number {
    return this.config.particleCount;
  }

  private async readBackF32(src: GPUBuffer, floatCount: number): Promise<Float32Array> {
    const device = this.device!;
    const bytes = floatCount * Float32Array.BYTES_PER_ELEMENT;
    const staging = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(src, 0, staging, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  }

  private ribbonStride(controls: RenderControls): number {
    return Math.max(1, Math.round(1 / Math.max(0.0001, controls.ribbonFraction)));
  }

  private ribbonActiveCount(controls: RenderControls): number {
    return Math.min(RIBBON_MAX, Math.ceil(this.config.particleCount / this.ribbonStride(controls)));
  }

  private ensureTrailBuffer(): void {
    if (!this.trailBuffer) {
      this.trailBuffer = this.device!.createBuffer({
        size: RIBBON_MAX * RIBBON_TRAIL_MAX * 4 * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE
      });
    }
    if (!this.ribbonUniformBuffer) {
      this.ribbonUniformBuffer = this.device!.createBuffer({
        size: ribbonUniformByteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      });
    }
  }

  private ribbonJointCount(controls: RenderControls): number {
    return Math.max(2, Math.min(RIBBON_TRAIL_MAX, Math.round(controls.ribbonJoints)));
  }

  private writeRibbonUniform(controls: RenderControls): void {
    const buf = new ArrayBuffer(ribbonUniformByteLength);
    const u32 = new Uint32Array(buf);
    const f32 = new Float32Array(buf);
    u32[0] = this.ribbonStride(controls);
    u32[1] = RIBBON_MAX;
    u32[2] = this.ribbonFrame >>> 0;
    u32[3] = Math.max(2, Math.min(RIBBON_TRAIL_MAX, Math.round(controls.ribbonLength)));
    u32[4] = this.ribbonJointCount(controls);
    f32[5] = controls.ribbonWidth;
    f32[6] = controls.ribbonTaper;       // length-fade amount
    f32[7] = controls.ribbonFadeStart;
    f32[8] = controls.ribbonEdgeFade;    // cross-ribbon edge falloff
    this.device!.queue.writeBuffer(this.ribbonUniformBuffer!, 0, buf);
  }

  // Append each ribboned particle's current position into its trail ring slot. Runs only while
  // stepping (paused frames keep the static trail). Reads the post-step current particle buffer.
  private encodeRibbonAppend(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.trailAppendPipeline!);
    for (let chunkIndex = 0; chunkIndex < this.particleChunks.length; chunkIndex += 1) {
      const chunk = this.particleChunks[chunkIndex];
      pass.setBindGroup(0, this.device!.createBindGroup({
        layout: this.trailAppendPipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: chunk.buffers[this.readIndex] } },
          { binding: 1, resource: { buffer: this.trailBuffer! } },
          { binding: 2, resource: { buffer: this.chunkSimUniformBuffers[chunkIndex] } },
          { binding: 3, resource: { buffer: this.ribbonUniformBuffer! } }
        ]
      }));
      pass.dispatchWorkgroups(Math.ceil(chunk.count / 256));
    }
    pass.end();
  }

  private drawRibbons(pass: GPURenderPassEncoder, controls: RenderControls): void {
    if (!this.trailBuffer || !this.ribbonRenderPipeline) return;
    pass.setPipeline(this.ribbonRenderPipeline);
    pass.setBindGroup(0, this.device!.createBindGroup({
      layout: this.ribbonRenderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.trailBuffer } },
        { binding: 1, resource: { buffer: this.renderUniformBuffer! } },
        { binding: 2, resource: { buffer: this.ribbonUniformBuffer! } }
      ]
    }));
    pass.draw(this.ribbonJointCount(controls) * 2, this.ribbonActiveCount(controls));
  }

  private drawParticlesAndOverlay(pass: GPURenderPassEncoder, controls: RenderControls, overlay: boolean, fieldBuffer = this.fieldBuffers[this.readIndex]): void {
    this.ribbonsActive = controls.ribbonFraction > 0.0001;
    if (this.ribbonsActive) {
      this.drawRibbons(pass, controls);
    }
    if (this.usesParticleSplatLayer(controls)) {
      const particleBlendMode = controls.particleBlendMode;
      if (this.usesComputeSplat(controls)) {
        // Splats were accumulated in compute (fixed-point, no rasterizer); one fullscreen
        // additive resolve replaces tens of millions of binned quads.
        pass.setPipeline(this.splatResolvePipeline!);
        pass.setBindGroup(0, this.createSplatResolveBindGroup());
        pass.draw(3);
      } else if (this.usesPreparedSplatPath(controls) && this.splatDrawPipelines[particleBlendMode]) {
        // Prepared-splat path: per-particle work happened once in prepare_splats; the vertex
        // shader only offsets corners. Culled particles are degenerate off-screen records, so
        // a plain draw keeps particle-index order (exact blend order for alpha/opaque modes).
        pass.setPipeline(this.splatDrawPipelines[particleBlendMode]!);
        for (let chunkIndex = 0; chunkIndex < this.particleChunks.length; chunkIndex += 1) {
          const chunk = this.particleChunks[chunkIndex];
          pass.setBindGroup(0, this.createSplatDrawBindGroup(particleBlendMode, chunkIndex));
          pass.draw(4, chunk.count);
        }
      } else {
        const pipelines = this.usesFastParticlePipeline(controls) ? this.particleFastRenderPipelines : this.particleRenderPipelines;
        pass.setPipeline(pipelines[particleBlendMode] ?? pipelines.additive!);
        const useIndirectSupportDraw = this.usesParticleSupportCompaction(controls);
        for (let chunkIndex = 0; chunkIndex < this.particleChunks.length; chunkIndex += 1) {
          const chunk = this.particleChunks[chunkIndex];
          pass.setBindGroup(0, this.createParticleRenderBindGroup(particleBlendMode, chunkIndex, fieldBuffer));
          if (useIndirectSupportDraw) {
            pass.drawIndirect(chunk.activeDrawBuffer, 0);
          } else {
            pass.draw(4, chunk.count);
          }
        }
      }
    }
    if (overlay) {
      pass.setPipeline(this.overlayPipeline!);
      pass.setBindGroup(0, this.createOverlayBindGroup());
      pass.draw(3);
    }
  }

  private usesParticleSplatLayer(controls: RenderControls): boolean {
    return controls.renderLayer === "both" || controls.renderLayer === "particles" || controls.renderLayer === "debug-voxels";
  }

  private usesParticleSupportCompaction(controls: RenderControls): boolean {
    return controls.particleSupportMask > 0.0001 && this.usesParticleSplatLayer(controls) && !!this.particleSupportGridBuffer;
  }

  private usesFastParticlePipeline(controls: RenderControls): boolean {
    return controls.fastParticleRender &&
      controls.particleDensityCutoff <= 0 &&
      controls.particleDensityNormalize <= 0.0001 &&
      !controls.dofEnabled &&
      !controls.dofDebug;
  }

  // Optimization A: the cutoff prepass only matters when the density-cutoff signal is actually
  // consumed (cutoff or normalize active) and particles are drawn. The cutoff/normalize gate
  // also forces the classic (non-fast) particle_vs path, which is the one paying the 4x cost.
  private usesCutoffPrepass(controls: RenderControls): boolean {
    return controls.particleCutoffPrepass &&
      (controls.particleDensityCutoff > 0 || controls.particleDensityNormalize > 0.0001) &&
      this.usesParticleSplatLayer(controls) &&
      !!this.cutoffPrepassPipeline;
  }

  // Bindings mirror createParticleRenderBindGroup (0=current, 2=prev, 3=field) so the prepass
  // computes interp_pos from the exact same buffers particle_vs reads; 8 is the f32 output.
  private createCutoffPrepassBindGroup(chunkIndex: number, fieldBuffer: GPUBuffer): GPUBindGroup {
    const chunk = this.particleChunks[chunkIndex];
    return this.device!.createBindGroup({
      layout: this.cutoffPrepassPipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: chunk.buffers[this.readIndex] } },
        { binding: 1, resource: { buffer: this.renderUniformBuffer! } },
        { binding: 2, resource: { buffer: chunk.buffers[this.readIndex === 0 ? 1 : 0] } },
        { binding: 3, resource: { buffer: fieldBuffer } },
        { binding: 8, resource: { buffer: chunk.cutoffSignalBuffer } }
      ]
    });
  }

  private encodeCutoffPrepass(encoder: GPUCommandEncoder, controls: RenderControls, fieldBuffer: GPUBuffer): void {
    if (!this.usesCutoffPrepass(controls)) {
      return;
    }
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.cutoffPrepassPipeline!);
    for (let chunkIndex = 0; chunkIndex < this.particleChunks.length; chunkIndex += 1) {
      const chunk = this.particleChunks[chunkIndex];
      pass.setBindGroup(0, this.createCutoffPrepassBindGroup(chunkIndex, fieldBuffer));
      pass.dispatchWorkgroups(Math.ceil(chunk.count / 256));
    }
    pass.end();
  }

  private encodeParticleSupportMask(
    encoder: GPUCommandEncoder,
    controls: RenderControls,
    particleSourceBufferIndex: number,
    timestampBase?: number
  ): void {
    if (controls.particleSupportMask <= 0.0001 || !this.usesParticleSplatLayer(controls) || !this.particleSupportGridBuffer) {
      return;
    }
    // clearBuffer zeroes the grid in the copy engine - far faster than a 256-wide
    // atomicStore dispatch over gridSize^3*4 lanes and produces identical zeros.
    encoder.clearBuffer(this.particleSupportGridBuffer);
    if (timestampBase !== undefined) {
      // Keep the clear-slot timestamp pair so readGpuPassTimings' query layout is unchanged.
      const clearStamp = encoder.beginComputePass({ timestampWrites: this.timestampWrites(timestampBase, timestampBase + 1) });
      clearStamp.end();
    }

    const buildPass = encoder.beginComputePass(timestampBase !== undefined ? { timestampWrites: this.timestampWrites(timestampBase + 2, timestampBase + 3) } : undefined);
    buildPass.setPipeline(this.particleSupportBuildPipeline!);
    for (let chunkIndex = 0; chunkIndex < this.particleChunks.length; chunkIndex += 1) {
      const chunk = this.particleChunks[chunkIndex];
      buildPass.setBindGroup(0, this.createParticleSupportBuildBindGroup(chunkIndex, particleSourceBufferIndex));
      buildPass.dispatchWorkgroups(Math.ceil(chunk.count / 256));
    }
    buildPass.end();

    for (const chunk of this.particleChunks) {
      this.device!.queue.writeBuffer(chunk.activeDrawBuffer, 0, particleSupportIndirectDrawReset);
    }

    const resolvePass = encoder.beginComputePass(timestampBase !== undefined ? { timestampWrites: this.timestampWrites(timestampBase + 4, timestampBase + 5) } : undefined);
    resolvePass.setPipeline(this.particleSupportResolvePipeline!);
    for (let chunkIndex = 0; chunkIndex < this.particleChunks.length; chunkIndex += 1) {
      const chunk = this.particleChunks[chunkIndex];
      resolvePass.setBindGroup(0, this.createParticleSupportResolveBindGroup(chunkIndex, particleSourceBufferIndex));
      resolvePass.dispatchWorkgroups(Math.ceil(chunk.count / 256));
    }
    resolvePass.end();
  }

  private renderDensityLayer(
    encoder: GPUCommandEncoder,
    controls: RenderControls,
    fogTexture?: GPUTexture,
    timestampBase?: number,
    force = false,
    fieldBuffer = this.fieldBuffers[this.readIndex]
  ): void {
    if (controls.densityPassStrength <= 0 || (!force && controls.renderLayer !== "both" && controls.renderLayer !== "density")) {
      return;
    }
    this.ensureDensityTextures(this.sceneTextureSize[0], this.sceneTextureSize[1]);
    this.drawDensitySplatPass(encoder, this.densitySmallPipeline!, this.densitySmallTexture!, timestampBase, fieldBuffer);
    this.drawDensitySplatPass(encoder, this.densityLargePipeline!, this.densityLargeTexture!, timestampBase !== undefined ? timestampBase + 2 : undefined, fieldBuffer);
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.sceneTexture!.createView(),
          loadOp: "load",
          storeOp: "store"
        }
      ],
      ...(timestampBase !== undefined ? { timestampWrites: this.timestampWrites(timestampBase + 4, timestampBase + 5) } : {})
    });
    pass.setPipeline(this.densityCompositePipeline!);
    pass.setBindGroup(0, this.createDensityCompositeBindGroup(fogTexture ?? this.ensureDensityEmptyFogTexture()));
    pass.draw(3);
    pass.end();
  }

  private renderAccumulationLayer(
    encoder: GPUCommandEncoder,
    controls: RenderControls,
    overlay: boolean,
    timestampBase?: number
  ): void {
    this.ensureAccumulationTextures(this.sceneTextureSize[0], this.sceneTextureSize[1]);
    if (this.shouldResetAccumulationHistory(controls, this.sceneTextureSize[0], this.sceneTextureSize[1])) {
      this.clearAccumulationHistory(encoder);
    }

    this.drawAccumulationSplatPass(encoder, timestampBase);
    const historyWriteIndex = 1 - this.accumulationHistoryReadIndex;
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.sceneTexture!.createView(),
          clearValue: { r: 0.004, g: 0.006, b: 0.008, a: 1 },
          loadOp: "clear",
          storeOp: "store"
        },
        {
          view: this.accumulationHistoryTextures[historyWriteIndex].createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store"
        }
      ],
      ...(timestampBase !== undefined ? { timestampWrites: this.timestampWrites(timestampBase + 2, timestampBase + 3) } : {})
    });
    pass.setPipeline(this.accumulationCompositePipeline!);
    pass.setBindGroup(0, this.createAccumulationCompositeBindGroup(this.accumulationHistoryTextures[this.accumulationHistoryReadIndex]));
    pass.draw(3);
    pass.end();

    this.accumulationHistoryReadIndex = historyWriteIndex;
    this.accumulationHistoryPrepared = true;

    if (overlay) {
      const overlayPass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: this.sceneTexture!.createView(),
            loadOp: "load",
            storeOp: "store"
          }
        ]
      });
      overlayPass.setPipeline(this.overlayPipeline!);
      overlayPass.setBindGroup(0, this.createOverlayBindGroup());
      overlayPass.draw(3);
      overlayPass.end();
    }
  }

  private renderVolumeDensityLayer(
    encoder: GPUCommandEncoder,
    controls: RenderControls,
    overlay: boolean,
    rebuildVolume: boolean,
    timestampBase?: number
  ): void {
    this.ensureVolumeDensityResources();
    if (controls.densityPassStrength > 0 && rebuildVolume) {
      const clearPass = encoder.beginComputePass(timestampBase !== undefined ? { timestampWrites: this.timestampWrites(timestampBase + 0, timestampBase + 1) } : undefined);
      clearPass.setPipeline(this.volumeDensityClearPipeline!);
      clearPass.setBindGroup(0, this.createVolumeDensityClearBindGroup());
      clearPass.dispatchWorkgroups(Math.ceil((this.voxelCount() * 4) / 256));
      clearPass.end();

      for (let chunkIndex = 0; chunkIndex < this.particleChunks.length; chunkIndex += 1) {
        const chunk = this.particleChunks[chunkIndex];
        this.device!.queue.writeBuffer(this.chunkSimUniformBuffers[chunkIndex], 0, encodeSimUniforms(this.config, this.timestep, 1, chunk.offset, chunk.count));
      }
      const depositPass = encoder.beginComputePass(timestampBase !== undefined ? { timestampWrites: this.timestampWrites(timestampBase + 2, timestampBase + 3) } : undefined);
      depositPass.setPipeline(this.volumeDensityDepositPipeline!);
      for (let chunkIndex = 0; chunkIndex < this.particleChunks.length; chunkIndex += 1) {
        const chunk = this.particleChunks[chunkIndex];
        depositPass.setBindGroup(0, this.createVolumeDensityDepositBindGroup(chunkIndex));
        depositPass.dispatchWorkgroups(Math.ceil(chunk.count / 256));
      }
      depositPass.end();

      const resolvePass = encoder.beginComputePass(timestampBase !== undefined ? { timestampWrites: this.timestampWrites(timestampBase + 4, timestampBase + 5) } : undefined);
      resolvePass.setPipeline(this.volumeDensityResolvePipeline!);
      resolvePass.setBindGroup(0, this.createVolumeDensityResolveBindGroup());
      resolvePass.dispatchWorkgroups(
        Math.ceil(this.config.width / 4),
        Math.ceil(this.config.height / 4),
        Math.ceil(this.config.depth / 4)
      );
      resolvePass.end();

      if (controls.densityLargeHalfRes) {
        // Optimization C: downsample the resolved small channel to half res, then blur it there
        // (1/8 the voxels). The full-res small channel and the deposit/resolve above are untouched.
        const halfW = Math.max(1, Math.ceil(this.config.width / 2));
        const halfH = Math.max(1, Math.ceil(this.config.height / 2));
        const halfD = Math.max(1, Math.ceil(this.config.depth / 2));
        const halfVoxels = halfW * halfH * halfD;

        const downsamplePass = encoder.beginComputePass();
        downsamplePass.setPipeline(this.volumeDensityDownsamplePipeline!);
        downsamplePass.setBindGroup(0, this.createVolumeDownsampleBindGroup());
        downsamplePass.dispatchWorkgroups(Math.ceil(halfW / 4), Math.ceil(halfH / 4), Math.ceil(halfD / 4));
        downsamplePass.end();

        const blurXPass = encoder.beginComputePass(timestampBase !== undefined ? { timestampWrites: this.timestampWrites(timestampBase + 6, timestampBase + 7) } : undefined);
        blurXPass.setPipeline(this.volumeDensityBlurXHalfPipeline!);
        blurXPass.setBindGroup(0, this.createVolumeBlurHalfBindGroup(this.volumeDensityBlurXHalfPipeline!, this.volumeDensitySmallHalfBuffer!, this.volumeDensityBlurTempHalfBuffer!));
        blurXPass.dispatchWorkgroups(Math.ceil(halfVoxels / 256));
        blurXPass.end();

        const blurYPass = encoder.beginComputePass(timestampBase !== undefined ? { timestampWrites: this.timestampWrites(timestampBase + 8, timestampBase + 9) } : undefined);
        blurYPass.setPipeline(this.volumeDensityBlurYHalfPipeline!);
        blurYPass.setBindGroup(0, this.createVolumeBlurHalfBindGroup(this.volumeDensityBlurYHalfPipeline!, this.volumeDensityBlurTempHalfBuffer!, this.volumeDensityLargeHalfBuffer!));
        blurYPass.dispatchWorkgroups(Math.ceil(halfVoxels / 256));
        blurYPass.end();

        const blurZPass = encoder.beginComputePass(timestampBase !== undefined ? { timestampWrites: this.timestampWrites(timestampBase + 10, timestampBase + 11) } : undefined);
        blurZPass.setPipeline(this.volumeDensityBlurZHalfPipeline!);
        blurZPass.setBindGroup(0, this.createVolumeBlurZHalfBindGroup());
        blurZPass.dispatchWorkgroups(Math.ceil(halfW / 4), Math.ceil(halfH / 4), Math.ceil(halfD / 4));
        blurZPass.end();
      } else {
        const blurXPass = encoder.beginComputePass(timestampBase !== undefined ? { timestampWrites: this.timestampWrites(timestampBase + 6, timestampBase + 7) } : undefined);
        blurXPass.setPipeline(this.volumeDensityBlurXPipeline!);
        blurXPass.setBindGroup(0, this.createVolumeDensityBlurBindGroup(this.volumeDensityBlurXPipeline!, this.volumeDensitySmallBuffer!, this.volumeDensityBlurTempBuffer!));
        blurXPass.dispatchWorkgroups(Math.ceil(this.voxelCount() / 256));
        blurXPass.end();

        const blurYPass = encoder.beginComputePass(timestampBase !== undefined ? { timestampWrites: this.timestampWrites(timestampBase + 8, timestampBase + 9) } : undefined);
        blurYPass.setPipeline(this.volumeDensityBlurYPipeline!);
        blurYPass.setBindGroup(0, this.createVolumeDensityBlurBindGroup(this.volumeDensityBlurYPipeline!, this.volumeDensityBlurTempBuffer!, this.volumeDensityLargeBuffer!));
        blurYPass.dispatchWorkgroups(Math.ceil(this.voxelCount() / 256));
        blurYPass.end();

        const blurZPass = encoder.beginComputePass(timestampBase !== undefined ? { timestampWrites: this.timestampWrites(timestampBase + 10, timestampBase + 11) } : undefined);
        blurZPass.setPipeline(this.volumeDensityBlurZPipeline!);
        blurZPass.setBindGroup(0, this.createVolumeDensityBlurZBindGroup());
        blurZPass.dispatchWorkgroups(
          Math.ceil(this.config.width / 4),
          Math.ceil(this.config.height / 4),
          Math.ceil(this.config.depth / 4)
        );
        blurZPass.end();
      }
      this.volumeDensityPreparedKey = this.volumeDensityCacheKey(controls);
      this.volumeDensityPrepared = true;
    }

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.sceneTexture!.createView(),
          clearValue: { r: 0.004, g: 0.006, b: 0.008, a: 1 },
          loadOp: "clear",
          storeOp: "store"
        }
      ],
      ...(timestampBase !== undefined
        ? {
            timestampWrites: controls.densityPassStrength > 0 && rebuildVolume
              ? this.timestampWrites(timestampBase + 12, timestampBase + 13)
              : this.timestampWrites(timestampBase, timestampBase + 1)
          }
        : {})
    });
    if (controls.densityPassStrength > 0) {
      pass.setPipeline(this.volumeDensityRenderPipeline!);
      pass.setBindGroup(0, this.createVolumeDensityRenderBindGroup(controls.densityLargeHalfRes));
      pass.draw(3);
    }
    pass.end();

    this.renderDensityLayer(encoder, controls, undefined, undefined, true);

    if (overlay) {
      const overlayPass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: this.sceneTexture!.createView(),
            loadOp: "load",
            storeOp: "store"
          }
        ]
      });
      overlayPass.setPipeline(this.overlayPipeline!);
      overlayPass.setBindGroup(0, this.createOverlayBindGroup());
      overlayPass.draw(3);
      overlayPass.end();
    }
  }

  private shouldRebuildVolumeDensity(controls: RenderControls, stepSimulation: boolean): boolean {
    if (stepSimulation || !this.volumeDensityPrepared) {
      return true;
    }
    return this.volumeDensityPreparedKey !== this.volumeDensityCacheKey(controls);
  }

  private volumeDensityCacheKey(controls: RenderControls): string {
    const tint = hexColorToRgb01(controls.particleTint);
    return [
      controls.densityLargeHalfRes ? "half" : "full",
      this.config.width,
      this.config.height,
      this.config.depth,
      this.config.particleCount,
      this.timestep,
      this.readIndex,
      controls.particleOpacity.toFixed(5),
      controls.particleSpeedCutoff.toFixed(5),
      controls.particleSlowCutoff.toFixed(5),
      controls.particleDensityCutoff.toFixed(6),
      controls.particleDensityRadius.toFixed(5),
      controls.densitySmallScale.toFixed(4),
      controls.densityLargeScale.toFixed(4),
      controls.particleColorMode,
      tint.map((value) => value.toFixed(4)).join(","),
      this.config.hueSensitivity.toFixed(5)
    ].join("|");
  }

  private drawDensitySplatPass(encoder: GPUCommandEncoder, pipeline: GPURenderPipeline, target: GPUTexture, timestampStart?: number, fieldBuffer = this.fieldBuffers[this.readIndex]): void {
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: target.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store"
        }
      ],
      ...(timestampStart !== undefined ? { timestampWrites: this.timestampWrites(timestampStart, timestampStart + 1) } : {})
    });
    pass.setPipeline(pipeline);
    for (let chunkIndex = 0; chunkIndex < this.particleChunks.length; chunkIndex += 1) {
      const chunk = this.particleChunks[chunkIndex];
      pass.setBindGroup(0, this.createDensitySplatBindGroup(pipeline, chunkIndex, fieldBuffer));
      pass.draw(chunk.count * 6);
    }
    pass.end();
  }

  private drawAccumulationSplatPass(encoder: GPUCommandEncoder, timestampStart?: number): void {
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.accumulationCurrentTexture!.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store"
        }
      ],
      ...(timestampStart !== undefined ? { timestampWrites: this.timestampWrites(timestampStart, timestampStart + 1) } : {})
    });
    pass.setPipeline(this.accumulationSplatPipeline!);
    for (let chunkIndex = 0; chunkIndex < this.particleChunks.length; chunkIndex += 1) {
      const chunk = this.particleChunks[chunkIndex];
      pass.setBindGroup(0, this.createAccumulationSplatBindGroup(chunkIndex));
      pass.draw(chunk.count * 6);
    }
    pass.end();
  }

  private createFieldRenderBindGroup(uniformBuffer: GPUBuffer, pipeline: GPURenderPipeline, fieldBuffer = this.fieldBuffers[this.readIndex]): GPUBindGroup {
    const defaultFieldBuffer = this.fieldBuffers[this.readIndex];
    if (uniformBuffer !== this.renderUniformBuffer || fieldBuffer !== defaultFieldBuffer) {
      return this.device!.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: fieldBuffer } },
          { binding: 1, resource: { buffer: uniformBuffer } },
          { binding: 2, resource: this.fieldTexture!.createView() },
          { binding: 3, resource: this.fieldTextureSampler! }
        ]
      });
    }
    this.fieldRenderBindGroups[this.readIndex] ??= this.device!.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: defaultFieldBuffer } },
        { binding: 1, resource: { buffer: uniformBuffer } },
        { binding: 2, resource: this.fieldTexture!.createView() },
        { binding: 3, resource: this.fieldTextureSampler! }
      ]
    });
    return this.fieldRenderBindGroups[this.readIndex];
  }

  private ensureSceneTexture(width: number, height: number): void {
    this.postSampler ??= this.device!.createSampler({ magFilter: "linear", minFilter: "linear" });
    this.postUniformBuffer ??= this.device!.createBuffer({
      size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    if (this.sceneTexture && this.sceneTextureSize[0] === width && this.sceneTextureSize[1] === height) {
      return;
    }
    this.sceneTexture?.destroy();
    for (const t of this.bloomTextures) t.destroy();
    this.sceneTexture = this.device!.createTexture({
      size: { width, height },
      format: this.format!,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC
    });
    // Fixed-point RGB accumulation target for the compute splatter (3 x u32 per pixel).
    this.splatAccumBuffer?.destroy();
    this.splatAccumBuffer = this.device!.createBuffer({
      size: width * height * 3 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    this.splatAccumBindGroups = [];
    this.splatResolveBindGroup = undefined;
    // Bloom mip chain: half res, halving to a small base. 8 octaves so the widest level
    // spreads ~256px (matching Cycles Fog Glow's wide kernel) and the veil fuses cores.
    this.bloomTextures = [];
    let w = Math.max(1, Math.floor(width / 2));
    let h = Math.max(1, Math.floor(height / 2));
    for (let level = 0; level < 8 && w > 2 && h > 2; level += 1) {
      this.bloomTextures.push(
        this.device!.createTexture({
          size: { width: w, height: h },
          format: this.format!,
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        })
      );
      w = Math.max(1, Math.floor(w / 2));
      h = Math.max(1, Math.floor(h / 2));
    }
    this.sceneTextureSize = [width, height];
  }

  private ensureDensityTextures(width: number, height: number): void {
    this.densitySampler ??= this.device!.createSampler({
      magFilter: "linear",
      minFilter: "linear"
    });
    if (this.densitySmallTexture && this.densityLargeTexture && this.densityTextureSize[0] === width && this.densityTextureSize[1] === height) {
      return;
    }
    this.densitySmallTexture?.destroy();
    this.densityLargeTexture?.destroy();
    const descriptor: GPUTextureDescriptor = {
      size: { width, height },
      format: densityTextureFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
    };
    this.densitySmallTexture = this.device!.createTexture(descriptor);
    this.densityLargeTexture = this.device!.createTexture(descriptor);
    this.densityTextureSize = [width, height];
    this.densityCompositeBindGroup = undefined;
    this.densityCompositeFogTexture = undefined;
  }

  private ensureAccumulationTextures(width: number, height: number): void {
    if (
      this.accumulationCurrentTexture &&
      this.accumulationHistoryTextures.length === 2 &&
      this.accumulationTextureSize[0] === width &&
      this.accumulationTextureSize[1] === height
    ) {
      return;
    }
    this.destroyAccumulationTextures();
    const descriptor: GPUTextureDescriptor = {
      size: { width, height },
      format: accumulationTextureFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
    };
    this.accumulationCurrentTexture = this.device!.createTexture(descriptor);
    this.accumulationHistoryTextures = [
      this.device!.createTexture(descriptor),
      this.device!.createTexture(descriptor)
    ];
    this.accumulationSampler = this.device!.createSampler({
      magFilter: "linear",
      minFilter: "linear"
    });
    this.accumulationTextureSize = [width, height];
    this.accumulationHistoryReadIndex = 0;
    this.accumulationHistoryPrepared = false;
  }

  private clearAccumulationHistory(encoder: GPUCommandEncoder): void {
    for (const texture of this.accumulationHistoryTextures) {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: texture.createView(),
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
            loadOp: "clear",
            storeOp: "store"
          }
        ]
      });
      pass.end();
    }
    this.accumulationHistoryReadIndex = 0;
    this.accumulationHistoryPrepared = false;
  }

  private shouldResetAccumulationHistory(controls: RenderControls, width: number, height: number): boolean {
    const key = this.accumulationHistoryCacheKey(controls, width, height);
    const reset = !this.accumulationHistoryPrepared || this.accumulationHistoryKey !== key;
    this.accumulationHistoryKey = key;
    return reset;
  }

  private accumulationHistoryCacheKey(controls: RenderControls, width: number, height: number): string {
    const tint = hexColorToRgb01(controls.particleTint);
    return [
      width,
      height,
      this.config.width,
      this.config.height,
      this.config.depth,
      this.config.particleCount,
      controls.cameraYaw.toFixed(6),
      controls.cameraPitch.toFixed(6),
      controls.cameraDistance.toFixed(5),
      controls.cameraPanX.toFixed(5),
      controls.cameraPanY.toFixed(5),
      controls.particleColorMode,
      tint.map((value) => value.toFixed(4)).join(","),
      this.config.hueSensitivity.toFixed(5),
      controls.particleOpacity.toFixed(5),
      controls.particleBrightness.toFixed(5),
      controls.particleSpeedCutoff.toFixed(5),
      controls.particleSlowCutoff.toFixed(5),
      controls.particleDensityCutoff.toFixed(6),
      controls.particleDensityRadius.toFixed(5),
      controls.accumulationRadius.toFixed(4),
      controls.accumulationNoiseReject.toFixed(4)
    ].join("|");
  }

  private ensureVolumeDensityResources(): void {
    const width = this.config.width;
    const height = this.config.height;
    const depth = this.config.depth;
    this.volumeDensitySampler ??= this.device!.createSampler({
      magFilter: "linear",
      minFilter: "linear"
    });
    if (
      this.volumeDensityAccumBuffer &&
      this.volumeDensitySmallBuffer &&
      this.volumeDensityBlurTempBuffer &&
      this.volumeDensityLargeBuffer &&
      this.volumeDensitySmallTexture &&
      this.volumeDensityLargeTexture &&
      this.volumeDensityLargeTextureHalf &&
      this.volumeDensityResourceSize[0] === width &&
      this.volumeDensityResourceSize[1] === height &&
      this.volumeDensityResourceSize[2] === depth
    ) {
      return;
    }
    this.destroyVolumeDensityResources();
    const voxelCount = this.voxelCount();
    const atomicBytes = voxelCount * 4 * Int32Array.BYTES_PER_ELEMENT;
    const volumeBytes = voxelCount * 4 * Float32Array.BYTES_PER_ELEMENT;
    this.volumeDensityAccumBuffer = this.device!.createBuffer({
      size: atomicBytes,
      usage: GPUBufferUsage.STORAGE
    });
    this.volumeDensitySmallBuffer = this.device!.createBuffer({
      size: volumeBytes,
      usage: GPUBufferUsage.STORAGE
    });
    this.volumeDensityBlurTempBuffer = this.device!.createBuffer({
      size: volumeBytes,
      usage: GPUBufferUsage.STORAGE
    });
    this.volumeDensityLargeBuffer = this.device!.createBuffer({
      size: volumeBytes,
      usage: GPUBufferUsage.STORAGE
    });
    const textureDescriptor: GPUTextureDescriptor = {
      size: { width, height, depthOrArrayLayers: depth },
      dimension: "3d",
      format: "rgba16float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
    };
    this.volumeDensitySmallTexture = this.device!.createTexture(textureDescriptor);
    this.volumeDensityLargeTexture = this.device!.createTexture(textureDescriptor);
    // Optimization C: half-resolution large-channel grid (ceil(dim/2)). Cheap (1/8 voxels), so it
    // is always allocated; it is only written/sampled when controls.densityLargeHalfRes is on.
    const halfWidth = Math.max(1, Math.ceil(width / 2));
    const halfHeight = Math.max(1, Math.ceil(height / 2));
    const halfDepth = Math.max(1, Math.ceil(depth / 2));
    const halfVolumeBytes = halfWidth * halfHeight * halfDepth * 4 * Float32Array.BYTES_PER_ELEMENT;
    this.volumeDensitySmallHalfBuffer = this.device!.createBuffer({ size: halfVolumeBytes, usage: GPUBufferUsage.STORAGE });
    this.volumeDensityBlurTempHalfBuffer = this.device!.createBuffer({ size: halfVolumeBytes, usage: GPUBufferUsage.STORAGE });
    this.volumeDensityLargeHalfBuffer = this.device!.createBuffer({ size: halfVolumeBytes, usage: GPUBufferUsage.STORAGE });
    this.volumeDensityLargeTextureHalf = this.device!.createTexture({
      size: { width: halfWidth, height: halfHeight, depthOrArrayLayers: halfDepth },
      dimension: "3d",
      format: "rgba16float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
    });
    this.volumeDensitySampler = this.device!.createSampler({
      magFilter: "linear",
      minFilter: "linear"
    });
    this.volumeDensityResourceSize = [width, height, depth];
  }

  private ensureDensityEmptyFogTexture(): GPUTexture {
    if (this.densityEmptyFogTexture) {
      return this.densityEmptyFogTexture;
    }
    this.densityEmptyFogTexture = this.device!.createTexture({
      size: { width: 1, height: 1 },
      format: fogTextureFormat,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
    });
    this.device!.queue.writeTexture(
      { texture: this.densityEmptyFogTexture },
      new Uint8Array(256),
      { bytesPerRow: 256, rowsPerImage: 1 },
      { width: 1, height: 1 }
    );
    return this.densityEmptyFogTexture;
  }

  private createBloomBindGroup(pipeline: GPURenderPipeline, input: GPUTexture): GPUBindGroup {
    return this.device!.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: input.createView() },
        { binding: 1, resource: this.postSampler! },
        { binding: 2, resource: { buffer: this.postUniformBuffer! } }
      ]
    });
  }

  private createPostBindGroup(scene: GPUTexture, mips: GPUTexture[]): GPUBindGroup {
    const bloom = mips[0] ?? scene;
    const mip = (index: number): GPUTexture => mips[index] ?? bloom;
    return this.device!.createBindGroup({
      layout: this.postPipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: scene.createView() },
        { binding: 1, resource: this.postSampler! },
        { binding: 2, resource: { buffer: this.postUniformBuffer! } },
        { binding: 3, resource: bloom.createView() },
        { binding: 4, resource: mip(1).createView() },
        { binding: 5, resource: mip(2).createView() },
        { binding: 6, resource: mip(3).createView() },
        { binding: 7, resource: mip(4).createView() },
        { binding: 8, resource: mip(5).createView() },
        { binding: 9, resource: mip(6).createView() },
        { binding: 10, resource: mip(7).createView() }
      ]
    });
  }

  private createPostNoBloomBindGroup(scene: GPUTexture): GPUBindGroup {
    return this.device!.createBindGroup({
      layout: this.postNoBloomPipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: scene.createView() },
        { binding: 1, resource: this.postSampler! },
        { binding: 2, resource: { buffer: this.postUniformBuffer! } }
      ]
    });
  }

  private ensureFogTextures(width: number, height: number): void {
    if (this.fogCurrentTexture && this.fogTextureSize[0] === width && this.fogTextureSize[1] === height) {
      this.fogSampler = this.device!.createSampler({
        magFilter: "linear",
        minFilter: "linear"
      });
      return;
    }
    this.destroyFogTextures();
    const descriptor: GPUTextureDescriptor = {
      size: { width, height },
      format: this.format!,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
    };
    this.fogCurrentTexture = this.device!.createTexture(descriptor);
    this.fogHistoryTextures = [
      this.device!.createTexture(descriptor),
      this.device!.createTexture(descriptor)
    ];
    this.fogSampler = this.device!.createSampler({
      magFilter: "linear",
      minFilter: "linear"
    });
    this.fogTextureSize = [width, height];
    this.fogHistoryReadIndex = 0;
    this.lastFogHistoryView = undefined;
    this.fogHistoryResetFrames = 0;
  }

  private destroyFogTextures(): void {
    this.fogCurrentTexture?.destroy();
    for (const texture of this.fogHistoryTextures) {
      texture.destroy();
    }
    this.fogCurrentTexture = undefined;
    this.fogHistoryTextures = [];
    this.fogTextureSize = [0, 0];
    this.fogHistoryReadIndex = 0;
    this.lastFogHistoryView = undefined;
    this.fogHistoryResetFrames = 0;
    this.fogSampler = undefined;
  }

  private clearFogHistoryView(): false {
    this.lastFogHistoryView = undefined;
    this.fogHistoryResetFrames = 0;
    return false;
  }

  private shouldResetFogHistory(
    controls: RenderControls,
    width: number,
    height: number,
    steps: number,
    useFieldTexture: boolean
  ): boolean {
    const next: FogHistoryView = {
      width,
      height,
      steps,
      layer: renderLayerToUniform(controls.renderLayer),
      fieldTextureMode: useFieldTexture ? 1 : 0,
      emptySpaceSkip: controls.emptySpaceSkipping ? 1 : 0,
      emptySpaceThreshold: controls.emptySpaceThreshold,
      emptySpaceStride: controls.emptySpaceStride,
      density: controls.density,
      exposure: controls.exposure,
      sceneBrightness: controls.sceneBrightness,
      trailOpacity: controls.trailOpacity,
      fogBrightness: controls.fogBrightness,
      trailThreshold: controls.trailThreshold,
      trailColorMode: trailColorModeToUniform(controls.trailColorMode),
      palette: controls.palette === "aurora" ? 0 : controls.palette === "ember" ? 1 : 2,
      filament: controls.filament,
      fogTint: hexColorToRgb01(controls.fogTint),
      yaw: controls.cameraYaw,
      pitch: controls.cameraPitch,
      distance: controls.cameraDistance,
      panX: controls.cameraPanX,
      panY: controls.cameraPanY
    };
    const previous = this.lastFogHistoryView;
    this.lastFogHistoryView = next;
    if (!previous) return true;
    return (
      previous.width !== next.width ||
      previous.height !== next.height ||
      previous.steps !== next.steps ||
      previous.layer !== next.layer ||
      previous.fieldTextureMode !== next.fieldTextureMode ||
      previous.emptySpaceSkip !== next.emptySpaceSkip ||
      numberChanged(previous.emptySpaceThreshold, next.emptySpaceThreshold) ||
      numberChanged(previous.emptySpaceStride, next.emptySpaceStride) ||
      numberChanged(previous.density, next.density) ||
      numberChanged(previous.exposure, next.exposure) ||
      numberChanged(previous.sceneBrightness, next.sceneBrightness) ||
      numberChanged(previous.trailOpacity, next.trailOpacity) ||
      numberChanged(previous.fogBrightness, next.fogBrightness) ||
      numberChanged(previous.trailThreshold, next.trailThreshold) ||
      previous.trailColorMode !== next.trailColorMode ||
      previous.palette !== next.palette ||
      numberChanged(previous.filament, next.filament) ||
      numberChanged(previous.fogTint[0], next.fogTint[0]) ||
      numberChanged(previous.fogTint[1], next.fogTint[1]) ||
      numberChanged(previous.fogTint[2], next.fogTint[2]) ||
      numberChanged(previous.yaw, next.yaw) ||
      numberChanged(previous.pitch, next.pitch) ||
      numberChanged(previous.distance, next.distance) ||
      numberChanged(previous.panX, next.panX) ||
      numberChanged(previous.panY, next.panY)
    );
  }

  private createFogCompositeBindGroup(current: GPUTexture, history: GPUTexture, uniformBuffer: GPUBuffer): GPUBindGroup {
    return this.device!.createBindGroup({
      layout: this.fogCompositePipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: current.createView() },
        { binding: 1, resource: history.createView() },
        { binding: 2, resource: this.fogSampler! },
        { binding: 3, resource: { buffer: uniformBuffer } }
      ]
    });
  }

  private createParticleRenderBindGroup(mode: ParticleBlendMode, chunkIndex: number, fieldBuffer = this.fieldBuffers[this.readIndex]): GPUBindGroup {
    const pipeline = this.particleRenderPipelines[mode] ??
      this.particleFastRenderPipelines[mode] ??
      this.particleRenderPipelines.additive ??
      this.particleFastRenderPipelines.additive;
    const layout = this.particleRenderBindGroupLayout ?? pipeline?.getBindGroupLayout(0);
    if (!layout) {
      throw new Error("particle render pipeline unavailable");
    }
    const chunk = this.particleChunks[chunkIndex];
    if (!chunk) {
      throw new Error("particle chunk unavailable");
    }
    const defaultFieldBuffer = this.fieldBuffers[this.readIndex];
    if (fieldBuffer !== defaultFieldBuffer) {
      return this.device!.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: chunk.buffers[this.readIndex] } },
          { binding: 1, resource: { buffer: this.renderUniformBuffer! } },
          { binding: 2, resource: { buffer: chunk.buffers[this.readIndex === 0 ? 1 : 0] } },
          { binding: 3, resource: { buffer: fieldBuffer } },
          { binding: 4, resource: { buffer: chunk.supportBuffer } },
          { binding: 5, resource: { buffer: chunk.activeIndexBuffer } },
          { binding: 6, resource: { buffer: chunk.cutoffSignalBuffer } },
          { binding: 7, resource: { buffer: chunk.sortIdBuffer } }
        ]
      });
    }
    const bindGroups = this.particleRenderBindGroups[mode] ?? (this.particleRenderBindGroups[mode] = []);
    const chunkBindGroups = bindGroups[chunkIndex] ?? (bindGroups[chunkIndex] = []);
    chunkBindGroups[this.readIndex] ??= this.device!.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: chunk.buffers[this.readIndex] } },
        { binding: 1, resource: { buffer: this.renderUniformBuffer! } },
        // Previous whole-step particle state for display interpolation.
        { binding: 2, resource: { buffer: chunk.buffers[this.readIndex === 0 ? 1 : 0] } },
        { binding: 3, resource: { buffer: defaultFieldBuffer } },
        { binding: 4, resource: { buffer: chunk.supportBuffer } },
        { binding: 5, resource: { buffer: chunk.activeIndexBuffer } },
        { binding: 6, resource: { buffer: chunk.cutoffSignalBuffer } },
        { binding: 7, resource: { buffer: chunk.sortIdBuffer } }
      ]
    });
    return chunkBindGroups[this.readIndex];
  }

  private createSplatPrepareBindGroup(chunkIndex: number): GPUBindGroup {
    const chunk = this.particleChunks[chunkIndex];
    if (!chunk) {
      throw new Error("particle chunk unavailable");
    }
    const byChunk = this.splatPrepareBindGroups[chunkIndex] ?? (this.splatPrepareBindGroups[chunkIndex] = []);
    byChunk[this.readIndex] ??= this.device!.createBindGroup({
      layout: this.splatPreparePipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: chunk.buffers[this.readIndex] } },
        { binding: 1, resource: { buffer: this.renderUniformBuffer! } },
        { binding: 2, resource: { buffer: chunk.buffers[this.readIndex === 0 ? 1 : 0] } },
        { binding: 3, resource: { buffer: this.fieldBuffers[this.readIndex] } },
        { binding: 4, resource: { buffer: chunk.supportBuffer } },
        { binding: 6, resource: { buffer: chunk.splatRecordBuffer } },
        { binding: 7, resource: { buffer: chunk.sortIdBuffer } }
      ]
    });
    return byChunk[this.readIndex];
  }

  private createSplatDrawBindGroup(mode: ParticleBlendMode, chunkIndex: number): GPUBindGroup {
    const chunk = this.particleChunks[chunkIndex];
    if (!chunk) {
      throw new Error("particle chunk unavailable");
    }
    const byMode = this.splatDrawBindGroups[mode] ?? (this.splatDrawBindGroups[mode] = []);
    byMode[chunkIndex] ??= this.device!.createBindGroup({
      layout: this.splatDrawPipelines[mode]!.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: { buffer: this.renderUniformBuffer! } },
        { binding: 6, resource: { buffer: chunk.splatRecordBuffer } }
      ]
    });
    return byMode[chunkIndex];
  }

  private encodeSplatPrepare(encoder: GPUCommandEncoder, controls: RenderControls): void {
    if (!this.usesPreparedSplatPath(controls) || !this.splatPreparePipeline) {
      return;
    }
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.splatPreparePipeline);
    for (let chunkIndex = 0; chunkIndex < this.particleChunks.length; chunkIndex += 1) {
      const chunk = this.particleChunks[chunkIndex];
      pass.setBindGroup(0, this.createSplatPrepareBindGroup(chunkIndex));
      pass.dispatchWorkgroups(Math.ceil(chunk.count / 256));
    }
    pass.end();
  }

  // Runtime escape hatch for A/B-testing the prepared-splat path against the classic
  // per-corner vertex shader (window.__fluodditySetSplatPrepass in the app shell).
  splatPrepassEnabled = true;
  // A/B escape for the compute splatter (additive only); falls back to the prepared raster draw.
  computeSplatEnabled = true;
  // One-time init switch: when true, createPipelines() fires all ~45 pipeline compiles
  // concurrently (Promise.all) instead of awaiting each serially, cutting cold-start /
  // time-to-first-frame. Read once inside createPipelines(); flipping it after init has no
  // effect until pipelines are torn down and rebuilt. Default false = current serial path
  // (byte-identical output either way — only compile scheduling differs). Set via
  // ?parallelPipelines=1 launch param or window.__fluodditySetParallelPipelineCompile.
  parallelPipelineCompile = false;

  private usesPreparedSplatPath(controls: RenderControls): boolean {
    return this.splatPrepassEnabled && this.usesFastParticlePipeline(controls) && this.usesParticleSplatLayer(controls) && controls.renderLayer !== "debug-voxels";
  }

  private usesComputeSplat(controls: RenderControls): boolean {
    return (
      this.computeSplatEnabled &&
      controls.particleBlendMode === "additive" &&
      this.usesPreparedSplatPath(controls) &&
      !!this.splatAccumPipeline &&
      !!this.splatResolvePipeline &&
      !!this.splatAccumBuffer
    );
  }

  private createSplatAccumBindGroup(chunkIndex: number): GPUBindGroup {
    const chunk = this.particleChunks[chunkIndex];
    if (!chunk) {
      throw new Error("particle chunk unavailable");
    }
    this.splatAccumBindGroups[chunkIndex] ??= this.device!.createBindGroup({
      layout: this.splatAccumPipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: { buffer: this.renderUniformBuffer! } },
        { binding: 6, resource: { buffer: chunk.splatRecordBuffer } },
        { binding: 7, resource: { buffer: this.splatAccumBuffer! } }
      ]
    });
    return this.splatAccumBindGroups[chunkIndex];
  }

  private createSplatResolveBindGroup(): GPUBindGroup {
    this.splatResolveBindGroup ??= this.device!.createBindGroup({
      layout: this.splatResolvePipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: { buffer: this.renderUniformBuffer! } },
        { binding: 6, resource: { buffer: this.splatAccumBuffer! } }
      ]
    });
    return this.splatResolveBindGroup;
  }

  private encodeSplatAccumulate(encoder: GPUCommandEncoder, controls: RenderControls): void {
    if (!this.usesComputeSplat(controls)) {
      return;
    }
    encoder.clearBuffer(this.splatAccumBuffer!);
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.splatAccumPipeline!);
    for (let chunkIndex = 0; chunkIndex < this.particleChunks.length; chunkIndex += 1) {
      const chunk = this.particleChunks[chunkIndex];
      pass.setBindGroup(0, this.createSplatAccumBindGroup(chunkIndex));
      pass.dispatchWorkgroups(Math.ceil(chunk.count / 256));
    }
    pass.end();
  }

  private createParticleSupportClearBindGroup(): GPUBindGroup {
    this.particleSupportClearBindGroup ??= this.device!.createBindGroup({
      layout: this.particleSupportClearPipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: { buffer: this.renderUniformBuffer! } },
        { binding: 2, resource: { buffer: this.particleSupportGridBuffer! } }
      ]
    });
    return this.particleSupportClearBindGroup;
  }

  private createParticleSupportBuildBindGroup(chunkIndex: number, sourceBufferIndex: number): GPUBindGroup {
    const chunk = this.particleChunks[chunkIndex];
    const bufferIndex = sourceBufferIndex === 0 ? 0 : 1;
    const bySourceIndex = this.particleSupportBuildBindGroups[chunkIndex] ?? (this.particleSupportBuildBindGroups[chunkIndex] = []);
    bySourceIndex[bufferIndex] ??= this.device!.createBindGroup({
      layout: this.particleSupportBuildPipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: chunk.buffers[bufferIndex] } },
        { binding: 1, resource: { buffer: this.renderUniformBuffer! } },
        { binding: 2, resource: { buffer: this.particleSupportGridBuffer! } }
      ]
    });
    return bySourceIndex[bufferIndex];
  }

  private createParticleSupportResolveBindGroup(chunkIndex: number, sourceBufferIndex: number): GPUBindGroup {
    const chunk = this.particleChunks[chunkIndex];
    const bufferIndex = sourceBufferIndex === 0 ? 0 : 1;
    const bySourceIndex = this.particleSupportResolveBindGroups[chunkIndex] ?? (this.particleSupportResolveBindGroups[chunkIndex] = []);
    bySourceIndex[bufferIndex] ??= this.device!.createBindGroup({
      layout: this.particleSupportResolvePipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: chunk.buffers[bufferIndex] } },
        { binding: 1, resource: { buffer: this.renderUniformBuffer! } },
        { binding: 2, resource: { buffer: this.particleSupportGridBuffer! } },
        { binding: 3, resource: { buffer: chunk.supportBuffer } },
        { binding: 4, resource: { buffer: chunk.activeIndexBuffer } },
        { binding: 5, resource: { buffer: chunk.activeDrawBuffer } }
      ]
    });
    return bySourceIndex[bufferIndex];
  }

  private createDensitySplatBindGroup(pipeline: GPURenderPipeline, chunkIndex: number, fieldBuffer = this.fieldBuffers[this.readIndex]): GPUBindGroup {
    const chunk = this.particleChunks[chunkIndex];
    if (!chunk) {
      throw new Error("particle chunk unavailable");
    }
    const defaultFieldBuffer = this.fieldBuffers[this.readIndex];
    if (fieldBuffer !== defaultFieldBuffer) {
      return this.device!.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: chunk.buffers[this.readIndex] } },
          { binding: 1, resource: { buffer: this.renderUniformBuffer! } },
          { binding: 2, resource: { buffer: chunk.buffers[this.readIndex === 0 ? 1 : 0] } },
          { binding: 3, resource: { buffer: fieldBuffer } }
        ]
      });
    }
    const pipelineIndex = pipeline === this.densitySmallPipeline ? 0 : 1;
    const pipelineBindGroups = this.densitySplatBindGroups[pipelineIndex] ?? (this.densitySplatBindGroups[pipelineIndex] = []);
    const chunkBindGroups = pipelineBindGroups[chunkIndex] ?? (pipelineBindGroups[chunkIndex] = []);
    chunkBindGroups[this.readIndex] ??= this.device!.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: chunk.buffers[this.readIndex] } },
        { binding: 1, resource: { buffer: this.renderUniformBuffer! } },
        { binding: 2, resource: { buffer: chunk.buffers[this.readIndex === 0 ? 1 : 0] } },
        { binding: 3, resource: { buffer: defaultFieldBuffer } }
      ]
    });
    return chunkBindGroups[this.readIndex];
  }

  private createDensityCompositeBindGroup(fogTexture: GPUTexture): GPUBindGroup {
    if (this.densityCompositeBindGroup && this.densityCompositeFogTexture === fogTexture) {
      return this.densityCompositeBindGroup;
    }
    this.densityCompositeBindGroup = this.device!.createBindGroup({
      layout: this.densityCompositePipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.densitySmallTexture!.createView() },
        { binding: 1, resource: { buffer: this.renderUniformBuffer! } },
        { binding: 2, resource: this.densityLargeTexture!.createView() },
        { binding: 3, resource: fogTexture.createView() },
        { binding: 4, resource: this.densitySampler! }
      ]
    });
    this.densityCompositeFogTexture = fogTexture;
    return this.densityCompositeBindGroup;
  }

  private createAccumulationSplatBindGroup(chunkIndex: number): GPUBindGroup {
    const chunk = this.particleChunks[chunkIndex];
    if (!chunk) {
      throw new Error("particle chunk unavailable");
    }
    const chunkBindGroups = this.accumulationSplatBindGroups[chunkIndex] ?? (this.accumulationSplatBindGroups[chunkIndex] = []);
    chunkBindGroups[this.readIndex] ??= this.device!.createBindGroup({
      layout: this.accumulationSplatPipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: chunk.buffers[this.readIndex] } },
        { binding: 1, resource: { buffer: this.renderUniformBuffer! } },
        { binding: 2, resource: { buffer: chunk.buffers[this.readIndex === 0 ? 1 : 0] } },
        { binding: 3, resource: { buffer: this.fieldBuffers[this.readIndex] } }
      ]
    });
    return chunkBindGroups[this.readIndex];
  }

  private createAccumulationCompositeBindGroup(historyTexture: GPUTexture): GPUBindGroup {
    return this.device!.createBindGroup({
      layout: this.accumulationCompositePipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.accumulationCurrentTexture!.createView() },
        { binding: 1, resource: { buffer: this.renderUniformBuffer! } },
        { binding: 2, resource: historyTexture.createView() },
        { binding: 3, resource: this.accumulationSampler! }
      ]
    });
  }

  private createVolumeDensityClearBindGroup(): GPUBindGroup {
    this.volumeDensityClearBindGroup ??= this.device!.createBindGroup({
      layout: this.volumeDensityClearPipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.volumeDensityAccumBuffer! } },
        { binding: 1, resource: { buffer: this.renderUniformBuffer! } }
      ]
    });
    return this.volumeDensityClearBindGroup;
  }

  private createVolumeDensityDepositBindGroup(chunkIndex: number): GPUBindGroup {
    const chunk = this.particleChunks[chunkIndex];
    if (!chunk) {
      throw new Error("particle chunk unavailable");
    }
    const byReadIndex = this.volumeDensityDepositBindGroups[chunkIndex] ?? (this.volumeDensityDepositBindGroups[chunkIndex] = []);
    byReadIndex[this.readIndex] ??= this.device!.createBindGroup({
      layout: this.volumeDensityDepositPipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.volumeDensityAccumBuffer! } },
        { binding: 1, resource: { buffer: this.renderUniformBuffer! } },
        { binding: 2, resource: { buffer: chunk.buffers[this.readIndex] } },
        { binding: 3, resource: { buffer: chunk.buffers[this.readIndex === 0 ? 1 : 0] } },
        { binding: 9, resource: { buffer: this.chunkSimUniformBuffers[chunkIndex] } },
        { binding: 10, resource: { buffer: this.fieldBuffers[this.readIndex] } }
      ]
    });
    return byReadIndex[this.readIndex];
  }

  private createVolumeDensityResolveBindGroup(): GPUBindGroup {
    this.volumeDensityResolveBindGroup ??= this.device!.createBindGroup({
      layout: this.volumeDensityResolvePipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.volumeDensityAccumBuffer! } },
        { binding: 1, resource: { buffer: this.renderUniformBuffer! } },
        { binding: 4, resource: { buffer: this.volumeDensitySmallBuffer! } },
        { binding: 5, resource: this.volumeDensitySmallTexture!.createView() }
      ]
    });
    return this.volumeDensityResolveBindGroup;
  }

  private createVolumeDensityBlurBindGroup(pipeline: GPUComputePipeline, source: GPUBuffer, destination: GPUBuffer): GPUBindGroup {
    const cached = pipeline === this.volumeDensityBlurXPipeline ? this.volumeDensityBlurXBindGroup : this.volumeDensityBlurYBindGroup;
    if (cached) {
      return cached;
    }
    const bindGroup = this.device!.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: { buffer: this.renderUniformBuffer! } },
        { binding: 6, resource: { buffer: source } },
        { binding: 7, resource: { buffer: destination } }
      ]
    });
    if (pipeline === this.volumeDensityBlurXPipeline) {
      this.volumeDensityBlurXBindGroup = bindGroup;
    } else {
      this.volumeDensityBlurYBindGroup = bindGroup;
    }
    return bindGroup;
  }

  private createVolumeDensityBlurZBindGroup(): GPUBindGroup {
    this.volumeDensityBlurZBindGroup ??= this.device!.createBindGroup({
      layout: this.volumeDensityBlurZPipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: { buffer: this.renderUniformBuffer! } },
        { binding: 6, resource: { buffer: this.volumeDensityLargeBuffer! } },
        { binding: 8, resource: this.volumeDensityLargeTexture!.createView() }
      ]
    });
    return this.volumeDensityBlurZBindGroup;
  }

  // Optimization C half-res passes. Downsample + blur X/Y use uniforms(1)/src(6)/dst(7); blur Z
  // writes the half-res large texture (binding 8). All read/write the half-res buffers.
  private createVolumeDownsampleBindGroup(): GPUBindGroup {
    this.volumeDensityDownsampleBindGroup ??= this.device!.createBindGroup({
      layout: this.volumeDensityDownsamplePipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: { buffer: this.renderUniformBuffer! } },
        { binding: 6, resource: { buffer: this.volumeDensitySmallBuffer! } },
        { binding: 7, resource: { buffer: this.volumeDensitySmallHalfBuffer! } }
      ]
    });
    return this.volumeDensityDownsampleBindGroup;
  }

  private createVolumeBlurHalfBindGroup(pipeline: GPUComputePipeline, source: GPUBuffer, destination: GPUBuffer): GPUBindGroup {
    const cached = pipeline === this.volumeDensityBlurXHalfPipeline ? this.volumeDensityBlurXHalfBindGroup : this.volumeDensityBlurYHalfBindGroup;
    if (cached) {
      return cached;
    }
    const bindGroup = this.device!.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: { buffer: this.renderUniformBuffer! } },
        { binding: 6, resource: { buffer: source } },
        { binding: 7, resource: { buffer: destination } }
      ]
    });
    if (pipeline === this.volumeDensityBlurXHalfPipeline) {
      this.volumeDensityBlurXHalfBindGroup = bindGroup;
    } else {
      this.volumeDensityBlurYHalfBindGroup = bindGroup;
    }
    return bindGroup;
  }

  private createVolumeBlurZHalfBindGroup(): GPUBindGroup {
    this.volumeDensityBlurZHalfBindGroup ??= this.device!.createBindGroup({
      layout: this.volumeDensityBlurZHalfPipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: { buffer: this.renderUniformBuffer! } },
        { binding: 6, resource: { buffer: this.volumeDensityLargeHalfBuffer! } },
        { binding: 8, resource: this.volumeDensityLargeTextureHalf!.createView() }
      ]
    });
    return this.volumeDensityBlurZHalfBindGroup;
  }

  private createVolumeDensityRenderBindGroup(useHalfLarge: boolean): GPUBindGroup {
    if (useHalfLarge) {
      this.volumeDensityRenderHalfBindGroup ??= this.device!.createBindGroup({
        layout: this.volumeDensityRenderPipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.volumeDensitySmallTexture!.createView() },
          { binding: 1, resource: { buffer: this.renderUniformBuffer! } },
          { binding: 2, resource: this.volumeDensityLargeTextureHalf!.createView() },
          { binding: 3, resource: this.volumeDensitySampler! }
        ]
      });
      return this.volumeDensityRenderHalfBindGroup;
    }
    this.volumeDensityRenderBindGroup ??= this.device!.createBindGroup({
      layout: this.volumeDensityRenderPipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.volumeDensitySmallTexture!.createView() },
        { binding: 1, resource: { buffer: this.renderUniformBuffer! } },
        { binding: 2, resource: this.volumeDensityLargeTexture!.createView() },
        { binding: 3, resource: this.volumeDensitySampler! }
      ]
    });
    return this.volumeDensityRenderBindGroup;
  }

  private createOverlayBindGroup(): GPUBindGroup {
    this.overlayBindGroup ??= this.device!.createBindGroup({
      layout: this.overlayPipeline!.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.renderUniformBuffer! } }]
    });
    return this.overlayBindGroup;
  }

  private createFieldTextureBindGroup(): GPUBindGroup {
    this.fieldTextureBindGroups[this.readIndex] ??= this.device!.createBindGroup({
      layout: this.fieldTexturePipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.fieldBuffers[this.readIndex] } },
        { binding: 1, resource: this.fieldTexture!.createView() },
        { binding: 2, resource: { buffer: this.simUniformBuffer! } }
      ]
    });
    return this.fieldTextureBindGroups[this.readIndex];
  }

  private getClearBindGroup(): GPUBindGroup {
    const device = this.device!;
    this.clearBindGroup ??= device.createBindGroup({
      layout: this.clearBrushPipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.brushBuffer! } },
        { binding: 1, resource: { buffer: this.simUniformBuffer! } }
      ]
    });
    return this.clearBindGroup;
  }

  // In-memory spatial sort. Set false to keep buffer order stable (e.g. for ribbon trails,
  // which select particles by slot, or state captures that assume seed order).
  particleSortEnabled = true;
  private static readonly SORT_INTERVAL_STEPS = 16;

  private shouldSortAtCurrentStep(): boolean {
    return (
      this.particleSortEnabled &&
      !this.ribbonsActive &&
      this.timestep > 0 &&
      this.timestep % RealtimeGpuSim3d.SORT_INTERVAL_STEPS === 0 &&
      !!this.sortHistogramPipeline &&
      !!this.sortScanPipeline &&
      !!this.sortScatterPipeline &&
      !!this.sortApplyPipeline
    );
  }

  lastSortTimestep = -1;

  private encodeSortPass(encoder: GPUCommandEncoder): void {
    this.lastSortTimestep = this.timestep;
    const device = this.device!;
    this.sortHistBuffer ??= device.createBuffer({
      size: SORT_GRID_SIZE ** 3 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    const readIndex = this.readIndex;
    const prevIndex = readIndex === 0 ? 1 : 0;
    for (let chunkIndex = 0; chunkIndex < this.particleChunks.length; chunkIndex += 1) {
      const chunk = this.particleChunks[chunkIndex];
      const histGroups = this.sortHistBindGroups[chunkIndex] ?? (this.sortHistBindGroups[chunkIndex] = []);
      histGroups[readIndex] ??= device.createBindGroup({
        layout: this.sortHistogramPipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: chunk.buffers[readIndex] } },
          { binding: 1, resource: { buffer: this.sortHistBuffer! } }
        ]
      });
      this.sortScanBindGroup ??= device.createBindGroup({
        layout: this.sortScanPipeline!.getBindGroupLayout(0),
        entries: [{ binding: 1, resource: { buffer: this.sortHistBuffer! } }]
      });
      const scatterGroups = this.sortScatterBindGroups[chunkIndex] ?? (this.sortScatterBindGroups[chunkIndex] = []);
      scatterGroups[readIndex] ??= device.createBindGroup({
        layout: this.sortScatterPipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: chunk.buffers[readIndex] } },
          { binding: 1, resource: { buffer: this.sortHistBuffer! } },
          { binding: 2, resource: { buffer: chunk.sortIdBuffer } },
          { binding: 3, resource: { buffer: chunk.sortScratchBuffer } },
          { binding: 4, resource: { buffer: chunk.sortScratchIdBuffer } },
          { binding: 5, resource: { buffer: chunk.sortMapBuffer } }
        ]
      });
      const applyGroups = this.sortApplyBindGroups[chunkIndex] ?? (this.sortApplyBindGroups[chunkIndex] = []);
      applyGroups[readIndex] ??= device.createBindGroup({
        layout: this.sortApplyPipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: chunk.buffers[prevIndex] } },
          { binding: 1, resource: { buffer: chunk.sortMapBuffer } },
          { binding: 2, resource: { buffer: chunk.sortScratchBuffer } }
        ]
      });

      const workgroups = Math.ceil(chunk.count / 256);
      encoder.clearBuffer(this.sortHistBuffer!);
      const histPass = encoder.beginComputePass();
      histPass.setPipeline(this.sortHistogramPipeline!);
      histPass.setBindGroup(0, histGroups[readIndex]);
      histPass.dispatchWorkgroups(workgroups);
      histPass.end();
      const scanPass = encoder.beginComputePass();
      scanPass.setPipeline(this.sortScanPipeline!);
      scanPass.setBindGroup(0, this.sortScanBindGroup!);
      scanPass.dispatchWorkgroups(1);
      scanPass.end();
      const scatterPass = encoder.beginComputePass();
      scatterPass.setPipeline(this.sortScatterPipeline!);
      scatterPass.setBindGroup(0, scatterGroups[readIndex]);
      scatterPass.dispatchWorkgroups(workgroups);
      scatterPass.end();
      const particleBytes = chunk.count * 32;
      encoder.copyBufferToBuffer(chunk.sortScratchBuffer, 0, chunk.buffers[readIndex], 0, particleBytes);
      encoder.copyBufferToBuffer(chunk.sortScratchIdBuffer, 0, chunk.sortIdBuffer, 0, chunk.count * 4);
      const applyPass = encoder.beginComputePass();
      applyPass.setPipeline(this.sortApplyPipeline!);
      applyPass.setBindGroup(0, applyGroups[readIndex]);
      applyPass.dispatchWorkgroups(workgroups);
      applyPass.end();
      encoder.copyBufferToBuffer(chunk.sortScratchBuffer, 0, chunk.buffers[prevIndex], 0, particleBytes);
    }
  }

  private getParticleStepBindGroups(chunkIndex: number): { deposit: GPUBindGroup; update: GPUBindGroup } {
    const device = this.device!;
    const chunk = this.particleChunks[chunkIndex];
    if (!chunk) {
      throw new Error("particle chunk unavailable");
    }
    const readIndex = this.readIndex;
    const writeIndex = this.writeIndex;
    const depositByReadIndex = this.depositBindGroups[chunkIndex] ?? (this.depositBindGroups[chunkIndex] = []);
    const updateByReadIndex = this.updateBindGroups[chunkIndex] ?? (this.updateBindGroups[chunkIndex] = []);
    depositByReadIndex[readIndex] ??= device.createBindGroup({
      layout: this.depositPipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: chunk.buffers[readIndex] } },
        { binding: 1, resource: { buffer: this.brushBuffer! } },
        { binding: 2, resource: { buffer: this.chunkSimUniformBuffers[chunkIndex] } },
        { binding: 4, resource: { buffer: chunk.sortIdBuffer } }
      ]
    });
    updateByReadIndex[readIndex] ??= device.createBindGroup({
      layout: this.updatePipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: chunk.buffers[readIndex] } },
        { binding: 1, resource: { buffer: this.fieldBuffers[readIndex] } },
        { binding: 2, resource: { buffer: chunk.buffers[writeIndex] } },
        { binding: 3, resource: { buffer: this.chunkSimUniformBuffers[chunkIndex] } },
        { binding: 4, resource: { buffer: this.ruleBuffer! } },
        { binding: 5, resource: { buffer: chunk.sortIdBuffer } }
      ]
    });
    if (this.usesFieldTextureSensing()) {
      const texByReadIndex = this.updateTextureBindGroups[chunkIndex] ?? (this.updateTextureBindGroups[chunkIndex] = []);
      texByReadIndex[readIndex] ??= device.createBindGroup({
        layout: this.updateTexturePipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: chunk.buffers[readIndex] } },
          { binding: 1, resource: this.fieldTexture!.createView() },
          { binding: 2, resource: { buffer: chunk.buffers[writeIndex] } },
          { binding: 3, resource: { buffer: this.chunkSimUniformBuffers[chunkIndex] } },
          { binding: 4, resource: { buffer: this.ruleBuffer! } },
          { binding: 5, resource: { buffer: chunk.sortIdBuffer } }
        ]
      });
      return {
        deposit: depositByReadIndex[readIndex],
        update: texByReadIndex[readIndex]
      };
    }
    return {
      deposit: depositByReadIndex[readIndex],
      update: updateByReadIndex[readIndex]
    };
  }

  // f16 texture sensing changes update-pass numerics slightly; keep a runtime escape so the
  // exact f32 buffer path can be A/B'd (window.__fluodditySetFieldTextureSensing).
  fieldTextureSensingEnabled = true;

  private usesFieldTextureSensing(): boolean {
    // Sensing through the texture is only allowed when it can be bit-exact (rgba32float).
    return (
      this.fieldTextureSensingEnabled &&
      this.float32FilterableSupported &&
      !!this.updateTexturePipeline &&
      !!this.fieldTexture
    );
  }

  private fieldTextureFormat(): "rgba32float" | "rgba16float" {
    return this.float32FilterableSupported ? "rgba32float" : "rgba16float";
  }

  private getVisualDepositBindGroup(chunkIndex: number): GPUBindGroup {
    const device = this.device!;
    const chunk = this.particleChunks[chunkIndex];
    if (!chunk) {
      throw new Error("particle chunk unavailable");
    }
    const byReadIndex = this.visualDepositBindGroups[chunkIndex] ?? (this.visualDepositBindGroups[chunkIndex] = []);
    byReadIndex[this.readIndex] ??= device.createBindGroup({
      layout: this.visualDepositPipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: chunk.buffers[this.readIndex] } },
        { binding: 1, resource: { buffer: this.brushBuffer! } },
        { binding: 2, resource: { buffer: this.chunkSimUniformBuffers[chunkIndex] } },
        { binding: 3, resource: { buffer: chunk.buffers[this.readIndex === 0 ? 1 : 0] } },
        { binding: 4, resource: { buffer: chunk.sortIdBuffer } }
      ]
    });
    return byReadIndex[this.readIndex];
  }

  private getFieldStepBindGroup(): GPUBindGroup {
    const device = this.device!;
    const readIndex = this.readIndex;
    const writeIndex = this.writeIndex;
    this.fieldBindGroups[readIndex] ??= device.createBindGroup({
      layout: this.fieldPipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.fieldBuffers[readIndex] } },
        { binding: 1, resource: { buffer: this.brushBuffer! } },
        { binding: 2, resource: { buffer: this.fieldBuffers[writeIndex] } },
        { binding: 3, resource: { buffer: this.simUniformBuffer! } },
        { binding: 4, resource: this.fieldTexture!.createView() }
      ]
    });
    return this.fieldBindGroups[readIndex];
  }

  private ecologyActive(): boolean {
    return this.config.speciesForce > 0 || this.config.predator > 0 || this.config.alarm > 0;
  }

  // Lazily create the ecology field buffers + pipelines on first activation. Async (pipeline compile)
  // so it is awaited from render()/advanceSteps() before the sync step encoding runs. No-op when the
  // species/predator/alarm behaviors are all off, so the default path never pays for it.
  private async ensureEcologyResources(): Promise<void> {
    const device = this.device;
    if (!device || !this.ecologyActive()) return;
    const bytes = this.voxelCount() * 4 * Float32Array.BYTES_PER_ELEMENT;
    if (!this.ecologyFieldBuffer) {
      this.ecologyFieldBuffer = device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
      });
    }
    if (!this.ecologyBrushBuffer) {
      this.ecologyBrushBuffer = device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
      });
    }
    if (!this.ecologyDepositPipeline) {
      this.ecologyDepositPipeline = await device.createComputePipelineAsync({
        layout: "auto",
        compute: { module: device.createShaderModule({ code: liveComputeEcologyDepositShader }), entryPoint: "ecology_deposit" }
      });
    }
    if (!this.ecologyUpdatePipeline) {
      this.ecologyUpdatePipeline = await device.createComputePipelineAsync({
        layout: "auto",
        compute: { module: device.createShaderModule({ code: liveComputeEcologyUpdateShader }), entryPoint: "ecology_update" }
      });
    }
    if (!this.ecologyParticlePipeline) {
      this.ecologyParticlePipeline = await device.createComputePipelineAsync({
        layout: "auto",
        compute: { module: device.createShaderModule({ code: makeLiveComputeParticleEcologyShader(false) }), entryPoint: "update_particles" }
      });
    }
  }

  private ecologyReady(): boolean {
    return (
      this.ecologyActive() &&
      !!this.ecologyFieldBuffer &&
      !!this.ecologyBrushBuffer &&
      !!this.ecologyDepositPipeline &&
      !!this.ecologyUpdatePipeline &&
      !!this.ecologyParticlePipeline
    );
  }

  private getEcologyClearBindGroup(): GPUBindGroup {
    this.ecologyClearBindGroup ??= this.device!.createBindGroup({
      layout: this.clearBrushPipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.ecologyBrushBuffer! } },
        { binding: 1, resource: { buffer: this.simUniformBuffer! } }
      ]
    });
    return this.ecologyClearBindGroup;
  }

  private getEcologyDepositBindGroup(chunkIndex: number): GPUBindGroup {
    const chunk = this.particleChunks[chunkIndex];
    const arr = this.ecologyDepositBindGroups[chunkIndex] ?? (this.ecologyDepositBindGroups[chunkIndex] = []);
    arr[this.readIndex] ??= this.device!.createBindGroup({
      layout: this.ecologyDepositPipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: chunk.buffers[this.readIndex] } },
        { binding: 1, resource: { buffer: this.ecologyBrushBuffer! } },
        { binding: 2, resource: { buffer: this.chunkSimUniformBuffers[chunkIndex] } }
      ]
    });
    return arr[this.readIndex];
  }

  private getEcologyUpdateBindGroup(): GPUBindGroup {
    this.ecologyUpdateBindGroup ??= this.device!.createBindGroup({
      layout: this.ecologyUpdatePipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.ecologyBrushBuffer! } },
        { binding: 1, resource: { buffer: this.ecologyFieldBuffer! } },
        { binding: 2, resource: { buffer: this.simUniformBuffer! } }
      ]
    });
    return this.ecologyUpdateBindGroup;
  }

  private getEcologyParticleBindGroup(chunkIndex: number): GPUBindGroup {
    const chunk = this.particleChunks[chunkIndex];
    const arr = this.ecologyParticleBindGroups[chunkIndex] ?? (this.ecologyParticleBindGroups[chunkIndex] = []);
    arr[this.readIndex] ??= this.device!.createBindGroup({
      layout: this.ecologyParticlePipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: chunk.buffers[this.readIndex] } },
        { binding: 1, resource: { buffer: this.fieldBuffers[this.readIndex] } },
        { binding: 2, resource: { buffer: chunk.buffers[this.writeIndex] } },
        { binding: 3, resource: { buffer: this.chunkSimUniformBuffers[chunkIndex] } },
        { binding: 4, resource: { buffer: this.ruleBuffer! } },
        { binding: 5, resource: { buffer: chunk.sortIdBuffer } },
        { binding: 6, resource: { buffer: this.ecologyFieldBuffer! } }
      ]
    });
    return arr[this.readIndex];
  }

  private getVisualFieldBindGroup(): GPUBindGroup {
    const device = this.device!;
    const readIndex = this.visualFieldReadIndex;
    const writeIndex = readIndex === 0 ? 1 : 0;
    this.visualFieldBindGroups[readIndex] ??= device.createBindGroup({
      layout: this.visualFieldSmoothPipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.visualFieldBuffers[readIndex] } },
        { binding: 1, resource: { buffer: this.brushBuffer! } },
        { binding: 2, resource: { buffer: this.visualFieldBuffers[writeIndex] } },
        { binding: 3, resource: { buffer: this.simUniformBuffer! } }
      ]
    });
    return this.visualFieldBindGroups[readIndex];
  }

  private advanceParticleDensityReference(): number {
    if (this.targetParticleDensityReference <= 0) {
      this.targetParticleDensityReference = particleDensityReferenceForStats(this.lastFieldStats);
    }
    this.smoothedParticleDensityReference = smoothParticleDensityReference(
      this.smoothedParticleDensityReference,
      this.targetParticleDensityReference
    );
    return this.smoothedParticleDensityReference;
  }

  // Advance the color-mode cross-fade toward `targetMode` and return the from-mode + blend factor to
  // hand the render shader. On a fresh change from a settled state we start a new ease (from = old
  // target, blend = 0); if a change arrives mid-ease we keep the original `from` and just retarget,
  // so swiping through modes glides into the final pick rather than snapping. dt is wall-clock so it
  // stays correct regardless of how often render() runs per frame.
  private resolveColorBlend(targetMode: number): { from: number; blend: number } {
    const now = performance.now();
    const dt = this.colorBlendLastMs === 0 ? 0 : Math.max(0, now - this.colorBlendLastMs);
    this.colorBlendLastMs = now;
    if (targetMode !== this.colorBlendToMode) {
      if (this.colorBlend >= 1) {
        this.colorBlendFromMode = this.colorBlendToMode;
        this.colorBlend = 0;
      }
      this.colorBlendToMode = targetMode;
    }
    if (this.colorBlend < 1) {
      this.colorBlend = Math.min(1, this.colorBlend + (COLOR_BLEND_MS > 0 ? dt / COLOR_BLEND_MS : 1));
    }
    return { from: this.colorBlendFromMode, blend: this.colorBlend };
  }

  private applyFieldStats(stats: LiveFieldStats): void {
    this.lastFieldStats = stats;
    const nextReference = particleDensityReferenceForStats(stats);
    if (nextReference > 0) {
      this.targetParticleDensityReference = nextReference;
      if (this.smoothedParticleDensityReference <= 0) {
        this.smoothedParticleDensityReference = nextReference;
      }
    }
  }

  private scheduleFieldStatsRead(): void {
    const device = this.device;
    const sourceBuffer = this.fieldBuffers[this.readIndex];
    if (!device || !sourceBuffer || this.fieldStatsReadPending) {
      return;
    }
    const generation = this.fieldStatsReadGeneration;
    this.fieldStatsReadPending = true;
    this.readFieldStatsFromBuffer(sourceBuffer, generation)
      .then((stats) => {
        if (generation === this.fieldStatsReadGeneration) {
          this.applyFieldStats(stats);
        }
      })
      .catch(() => {
        // Device loss or teardown can invalidate the staging map; the next live frame will retry.
      })
      .finally(() => {
        if (generation === this.fieldStatsReadGeneration) {
          this.fieldStatsReadPending = false;
        }
      });
  }

  private async readFieldStats(): Promise<LiveFieldStats> {
    const sourceBuffer = this.fieldBuffers[this.readIndex];
    if (!sourceBuffer) {
      return this.lastFieldStats;
    }
    return this.readFieldStatsFromBuffer(sourceBuffer, this.fieldStatsReadGeneration);
  }

  private async readFieldStatsFromBuffer(sourceBuffer: GPUBuffer, generation: number): Promise<LiveFieldStats> {
    const device = this.device!;
    const size = this.voxelCount() * 4 * Float32Array.BYTES_PER_ELEMENT;
    const readback = device.createBuffer({
      size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(sourceBuffer, 0, readback, 0, size);
    device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      if (generation !== this.fieldStatsReadGeneration) {
        return this.lastFieldStats;
      }
      const values = new Float32Array(readback.getMappedRange().slice(0));
      return fieldStatsFromValues(values);
    } finally {
      if (readback.mapState === "mapped") {
        readback.unmap();
      }
      readback.destroy();
    }
  }

  private voxelCount(): number {
    return this.config.width * this.config.height * this.config.depth;
  }
}

function fieldStatsFromValues(values: Float32Array): LiveFieldStats {
  let densitySum = 0;
  let flowSum = 0;
  let nonzeroVoxels = 0;
  let maxDensity = 0;
  let maxDensitySignal = 0;
  let densitySignalSum = 0;
  for (let i = 0; i < values.length; i += 4) {
    const flow = Math.hypot(values[i], values[i + 1], values[i + 2]);
    const density = Math.max(0, values[i + 3]);
    const densitySignal = particleFieldDensitySignal(density, flow);
    densitySum += density;
    flowSum += flow;
    maxDensity = Math.max(maxDensity, density);
    maxDensitySignal = Math.max(maxDensitySignal, densitySignal);
    if (densitySignal > 0.0001) {
      nonzeroVoxels += 1;
      densitySignalSum += densitySignal;
    }
  }
  const densitySignalMean = nonzeroVoxels > 0 ? densitySignalSum / nonzeroVoxels : 0;
  return {
    densitySum,
    flowSum,
    nonzeroVoxels,
    maxDensity,
    maxDensitySignal,
    densitySignalMean,
    densitySignalP90: densitySignalPercentile(values, maxDensitySignal, nonzeroVoxels, 0.9)
  };
}

function particleFieldDensitySignal(density: number, flow: number): number {
  return Math.max(0, density + flow * 0.25);
}

function densitySignalPercentile(values: Float32Array, maxDensitySignal: number, nonzeroVoxels: number, percentile: number): number {
  if (maxDensitySignal <= 0 || nonzeroVoxels <= 0) {
    return 0;
  }
  const binCount = 128;
  const histogram = new Uint32Array(binCount);
  const binScale = (binCount - 1) / maxDensitySignal;
  for (let i = 0; i < values.length; i += 4) {
    const flow = Math.hypot(values[i], values[i + 1], values[i + 2]);
    const densitySignal = particleFieldDensitySignal(Math.max(0, values[i + 3]), flow);
    if (densitySignal > 0.0001) {
      const bin = Math.min(binCount - 1, Math.floor(densitySignal * binScale));
      histogram[bin] += 1;
    }
  }
  const target = Math.max(1, Math.ceil(nonzeroVoxels * clampNumber(percentile, 0, 1)));
  let cumulative = 0;
  for (let bin = 0; bin < binCount; bin += 1) {
    cumulative += histogram[bin];
    if (cumulative >= target) {
      return (bin + 0.5) / binScale;
    }
  }
  return maxDensitySignal;
}

function particleDensityReferenceForStats(stats: LiveFieldStats): number {
  if (stats.densitySignalP90 > 0) return stats.densitySignalP90;
  if (stats.densitySignalMean > 0) return stats.densitySignalMean;
  return stats.maxDensitySignal;
}

export function smoothParticleDensityReference(current: number, target: number, blend = 0.04): number {
  if (target <= 0) {
    return current > 0 ? current : 0;
  }
  if (current <= 0) {
    return target;
  }
  const t = clampNumber(blend, 0, 1);
  return current + (target - current) * t;
}

function encodeSimUniforms(
  config: LiveGpu3dConfig,
  timestep: number,
  stepTimeScale: number,
  particleOffset = 0,
  activeParticleCount = config.particleCount,
  visualLerpT = stepTimeScale,
  morphAlpha = 0
): ArrayBuffer {
  const buffer = new ArrayBuffer(SIM_UNIFORM_BYTES);
  const u32 = new Uint32Array(buffer);
  const f32 = new Float32Array(buffer);
  const scale = clampNumber(stepTimeScale, 0, 1);
  u32[0] = config.width;
  u32[1] = config.height;
  u32[2] = config.depth;
  u32[3] = config.particleCount;
  u32[4] = timestep;
  u32[5] = config.seed >>> 0;
  u32[6] = config.cohorts;
  u32[7] = config.boundaryMode;
  f32[8] = config.dt * scale;
  f32[9] = config.sensorGain;
  f32[10] = config.sensorAngle;
  f32[11] = config.sensorDistance;
  f32[12] = config.mutationScale;
  f32[13] = config.globalForceMult;
  f32[14] = config.drag;
  f32[15] = config.strafePower;
  f32[16] = config.axialForce;
  f32[17] = config.lateralForce;
  f32[18] = config.hazardRate;
  f32[19] = config.trailPersistence;
  f32[20] = config.trailDiffusion;
  f32[21] = config.depositRadius;
  f32[22] = config.depositMass;
  f32[23] = config.sigma;
  f32[24] = config.ruleSeed;
  f32[25] = config.orientationMix;
  u32[26] = (config.symmetryAxes >>> 0) & 7;
  u32[27] = config.absoluteOrientation;
  u32[28] = config.initialConditions;
  u32[29] = Math.max(0, Math.min(6, Math.round(config.depositTapRadius)));
  f32[30] = config.hueSensitivity;
  u32[31] = config.colorByCohort ? 1 : 0;
  f32[32] = scale;
  f32[33] = clampNumber(visualLerpT, 0, 1);
  f32[34] = config.recycleCutoff;
  u32[35] = config.recycleEnabled ? 1 : 0;
  u32[36] = Math.max(0, Math.floor(particleOffset));
  u32[37] = Math.max(0, Math.floor(activeParticleCount));
  u32[38] = config.domainShape;
  f32[38] = clampNumber(morphAlpha, 0, 1);
  // idx 39 = morph_alpha (left as the existing packing does). Pulse fields follow at 40/41.
  f32[40] = config.pulseDepth;
  f32[41] = config.pulseRate;
  f32[42] = config.restlessness;
  // --- Emergent behaviors (slots 43..60). All default to 0 (off) so the sim is byte-identical to
  // the classic engine; the matching WGSL `struct SimConfig` members and gated `if (x > 0.0)`
  // branches are appended in the same order. ---
  f32[43] = config.mips;
  f32[44] = config.anisoFollow;
  f32[45] = config.flockAlign;
  f32[46] = config.flockSeparate;
  f32[47] = config.chemotaxis;
  f32[48] = config.quorumStrength;
  f32[49] = config.quorumThreshold;
  f32[50] = config.leniaStrength;
  f32[51] = config.leniaCenter;
  f32[52] = config.leniaWidth;
  f32[53] = config.speciesForce;
  f32[54] = config.predator;
  f32[55] = config.alarm;
  f32[56] = config.grayScott;
  f32[57] = config.gsFeed;
  f32[58] = config.gsKill;
  f32[59] = config.energy;
  f32[60] = config.energyDrain;
  const strafeMomentum = Number.isFinite(config.strafeMomentum) ? config.strafeMomentum : 0;
  f32[61] = clampNumber(strafeMomentum, 0, 1);
  return buffer;
}

export const MAX_SIMULATION_SPEED = 4;

export type SimulationStepPlan = {
  steps: number;
  accumulator: number;
  stepScales: number[];
  timeAdvance: number;
  renderLerpT: number;
};

// Simulation speed is transport, not physics. The sim itself advances in whole
// authored ticks so changing speed preserves the trajectory; sub-1x smoothness
// comes from particle-only render interpolation between the last two whole states.
export function planSimulationSteps(
  accumulator: number,
  simulationSpeed: number
): SimulationStepPlan {
  const speed = clampNumber(simulationSpeed, 0, MAX_SIMULATION_SPEED);
  if (speed <= 0) {
    return { steps: 0, accumulator: 0, stepScales: [], timeAdvance: 0, renderLerpT: 1 };
  }
  let nextAccumulator = accumulator + speed;
  const wholeSteps = Math.floor(nextAccumulator);
  nextAccumulator -= wholeSteps;
  const stepScales: number[] = [];
  for (let step = 0; step < wholeSteps; step += 1) {
    stepScales.push(1);
  }
  return {
    steps: stepScales.length,
    accumulator: nextAccumulator,
    stepScales,
    timeAdvance: stepScales.length,
    renderLerpT: speed < 1 ? nextAccumulator : 1
  };
}

export function renderLerpTForSimulationSpeed(simulationSpeed: number, accumulator: number): number {
  const speed = clampNumber(simulationSpeed, 0, MAX_SIMULATION_SPEED);
  return speed > 0 && speed < 1 ? clampNumber(accumulator, 0, 1) : 1;
}

export function particleSourceBufferIndexForLerp(readIndex: number, renderLerpT: number): number {
  const currentIndex = readIndex === 0 ? 0 : 1;
  return renderLerpT <= 0.000001 ? 1 - currentIndex : currentIndex;
}

export function visualFieldSourceBufferIndexForFrame(
  readIndex: number,
  writeIndex: number,
  phaseReset: boolean
): number {
  return phaseReset ? (writeIndex === 0 ? 0 : 1) : (readIndex === 0 ? 0 : 1);
}

// Per-octave weights that reconstruct a Fog Glow PSF (Spencer et al. 1995) as a sum
// of octave Gaussians. The Spencer glare PSF (achromatic) is a tight Gaussian core plus two
// power-law veiling tails: f1 = 1/(theta)^3 and f2 = 1/(theta)^2. The core is the original
// pixel (already in the scene), so the bloom carries only the f1+f2 tails. Each mip L is a
// Gaussian at radius ~2^(L+1) px; weighting it by the PSF's annular energy P(r)*r^2 makes the
// summed octaves approximate the continuous veiling kernel -> one smooth wide halo, no clumps.
export function spencerFogGlowWeights(levelCount: number, radius: number): number[] {
  const weights: number[] = [0, 0, 0, 0, 0, 0, 0, 0];
  const eps = 0.02;
  const scale = 0.006 * Math.max(0.05, radius); // pixels -> Spencer theta; widens with Bloom Radius
  let sum = 0;
  for (let level = 0; level < levelCount && level < 8; level += 1) {
    const rpx = Math.pow(2, level + 1);
    const theta = rpx * scale;
    const f1 = 20.91 / Math.pow(theta + eps, 3);
    const f2 = 72.37 / Math.pow(theta + eps, 2);
    const psf = 0.478 * f1 + 0.138 * f2;
    // Annular energy P(r)*r^2, with an extra *r^2 bias toward the broad veil octaves so the
    // wide tail dominates and fuses cores into a smooth wash (Cycles Fog Glow look) rather
    // than leaving the f1-concentrated near-core glare that reads as clumpy.
    const w = psf * rpx * rpx * rpx;
    weights[level] = w;
    sum += w;
  }
  if (sum > 0) {
    for (let i = 0; i < 8; i += 1) weights[i] /= sum;
  }
  return weights;
}

export function encodePostUniforms(values: {
  bloomStrength: number;
  bloomThreshold: number;
  bloomRadius: number;
  colorSaturation: number;
  colorContrast: number;
  tint: [number, number, number];
  levelCount: number;
  weights: number[];
  chromaticAberration?: number;
  vignetteStrength?: number;
  vignetteSoftness?: number;
  streakStrength?: number;
  streakLength?: number;
  streakVertical?: number;
  flareHeight?: number;
  flareCutoff?: number;
  despeckle?: number;
  hdrGain?: number;
}): ArrayBuffer {
  const buffer = new ArrayBuffer(128);
  const f32 = new Float32Array(buffer);
  f32[0] = values.bloomStrength;
  f32[1] = values.bloomThreshold;
  f32[2] = values.bloomRadius;
  f32[3] = values.levelCount;
  f32[4] = values.tint[0];
  f32[5] = values.tint[1];
  f32[6] = values.tint[2];
  f32[8] = values.colorSaturation;
  f32[9] = values.colorContrast;
  // grade.w (f32[11]) = HDR-capture gain. 0 => live SDR/AGX path (unchanged). >0 => bypass AGX and
  // emit linear HDR scaled by this gain (see apply_color_grade_hdr). Only set during HDR export.
  f32[11] = values.hdrGain ?? 0;
  // weights_a = w0..w3, weights_b = w4..w7
  for (let i = 0; i < 4; i += 1) f32[12 + i] = values.weights[i] ?? 0;
  for (let i = 0; i < 4; i += 1) f32[16 + i] = values.weights[4 + i] ?? 0;
  // lens = (chromatic aberration, vignette strength, vignette softness, reserved)
  f32[20] = values.chromaticAberration ?? 0;
  f32[21] = values.vignetteStrength ?? 0;
  f32[22] = values.vignetteSoftness ?? 0.5;
  // streak = (strength, length, vertical/star mix, reserved)
  f32[24] = values.streakStrength ?? 0;
  f32[25] = values.streakLength ?? 0.6;
  f32[26] = values.streakVertical ?? 0;
  // flare = (vertical height, luminance cutoff, screen-space despeckle, reserved)
  f32[28] = values.flareHeight ?? 0.3;
  f32[29] = values.flareCutoff ?? 0;
  f32[30] = Number.isFinite(values.despeckle as number) ? (values.despeckle as number) : 0;
  f32[31] = 0;
  return buffer;
}

function encodeRenderUniforms(uniforms: RenderUniforms): ArrayBuffer {
  const buffer = new ArrayBuffer(renderUniformByteLength);
  const f32 = new Float32Array(buffer);
  const u32 = new Uint32Array(buffer);
  f32[0] = uniforms.resolution[0];
  f32[1] = uniforms.resolution[1];
  u32[2] = uniforms.dimensions[0];
  u32[3] = uniforms.dimensions[1];
  u32[4] = uniforms.dimensions[2];
  u32[5] = uniforms.particleCount;
  u32[6] = uniforms.voxelCount;
  u32[7] = uniforms.timestep;
  f32[8] = uniforms.density;
  f32[9] = uniforms.exposure;
  f32[10] = uniforms.focusDistance;
  f32[11] = uniforms.aperture;
  u32[12] = uniforms.overlay;
  u32[13] = uniforms.palette;
  f32[14] = uniforms.filament;
  f32[15] = uniforms.yawCos;
  f32[16] = uniforms.yawSin;
  f32[17] = uniforms.pitchCos;
  f32[18] = uniforms.pitchSin;
  f32[19] = uniforms.distance;
  u32[20] = uniforms.raySteps;
  f32[21] = uniforms.fogStepScale;
  f32[22] = uniforms.fogTemporalBlend;
  u32[23] = uniforms.fogBlueNoise;
  u32[24] = uniforms.fogFrame;
  u32[25] = uniforms.fieldTextureMode;
  u32[26] = uniforms.emptySpaceSkip;
  f32[27] = uniforms.emptySpaceThreshold;
  f32[28] = uniforms.emptySpaceStride;
  f32[29] = uniforms.particleSizePx;
  f32[30] = uniforms.particleMinPx;
  f32[31] = uniforms.particleMaxPx;
  f32[32] = uniforms.particleOpacity;
  u32[33] = uniforms.particleBlendMode;
  f32[34] = uniforms.particleDensityCutoff;
  f32[35] = uniforms.particleDensityRadius;
  f32[36] = uniforms.trailOpacity;
  f32[37] = uniforms.trailThreshold;
  u32[38] = uniforms.renderLayer;
  f32[39] = uniforms.cameraPanX;
  f32[40] = uniforms.cameraPanY;
  u32[41] = uniforms.trailColorMode;
  f32[42] = uniforms.fogTint[0];
  f32[43] = uniforms.fogTint[1];
  f32[44] = uniforms.fogTint[2];
  f32[45] = uniforms.particleTint[0];
  f32[46] = uniforms.particleTint[1];
  f32[47] = uniforms.particleTint[2];
  f32[48] = uniforms.sceneBrightness;
  f32[49] = uniforms.particleBrightness;
  f32[50] = uniforms.fogBrightness;
  u32[51] = uniforms.particleColorMode;
  u32[52] = uniforms.particleVelocityStretch;
  f32[53] = uniforms.particleStretch;
  f32[54] = uniforms.particleGradientSensitivity;
  f32[55] = uniforms.focalK;
  f32[56] = uniforms.dofBlur;
  u32[57] = uniforms.dofDebug;
  f32[58] = uniforms.renderLerpT;
  u32[59] = uniforms.dofEnabled;
  f32[60] = uniforms.densityPassStrength;
  f32[61] = uniforms.densitySmallScale;
  f32[62] = uniforms.densityLargeScale;
  f32[63] = uniforms.densityLargeThreshold;
  f32[64] = uniforms.densityContrastGain;
  f32[65] = uniforms.densityContrastBalance;
  f32[66] = uniforms.densityEmissionPower;
  f32[67] = uniforms.densityOcclusion;
  u32[68] = Math.max(1, Math.round(uniforms.cohorts));
  f32[69] = uniforms.particleSlowCutoff;
  f32[70] = uniforms.particleGlowCore;
  f32[71] = uniforms.particleHotCore;
  f32[72] = uniforms.accumulationStrength;
  f32[73] = uniforms.accumulationRadius;
  f32[74] = uniforms.accumulationCurve;
  f32[75] = uniforms.accumulationMemory;
  f32[76] = uniforms.accumulationNoiseReject;
  f32[77] = uniforms.particleDensityReference;
  f32[78] = uniforms.particleDensityNormalize;
  f32[79] = uniforms.particleDensitySoftness;
  f32[80] = uniforms.particleSupportMask;
  f32[81] = uniforms.particleSupportRadius;
  f32[82] = uniforms.particleSupportNeighbors;
  f32[83] = uniforms.particleSupportFlow;
  u32[84] = Math.max(1, Math.round(uniforms.particleSupportGridSize));
  f32[85] = uniforms.particleStretchMin;
  f32[86] = uniforms.particleStretchSpeed;
  f32[87] = uniforms.particleSpeedCutoff;
  f32[88] = uniforms.variationMaster;
  f32[89] = uniforms.variationTime;
  f32[90] = uniforms.variationDrift;
  f32[91] = uniforms.variationNoiseMix;
  f32[92] = uniforms.variationFreq;
  u32[93] = Math.max(1, Math.round(uniforms.variationOctaves));
  f32[94] = uniforms.variationGain;
  f32[95] = uniforms.variationLacunarity;
  f32[96] = uniforms.variationSizeAmount;
  f32[97] = uniforms.variationSizeCurve;
  f32[98] = uniforms.variationSizeMin;
  f32[99] = uniforms.variationSizeMax;
  f32[100] = uniforms.variationBrightAmount;
  f32[101] = uniforms.variationBrightCurve;
  f32[102] = uniforms.variationBrightMin;
  f32[103] = uniforms.variationBrightMax;
  f32[104] = uniforms.variationOpacityAmount;
  f32[105] = uniforms.variationOpacityCurve;
  f32[106] = uniforms.variationOpacityMin;
  f32[107] = uniforms.variationOpacityMax;
  f32[108] = uniforms.variationColorAmount;
  f32[109] = uniforms.variationColorCurve;
  f32[110] = uniforms.variationColorMin;
  f32[111] = uniforms.variationColorMax;
  u32[112] = uniforms.domainShape;
  // Audio-reactive levels (slots 113-115 use the existing 464B padding; no buffer growth).
  f32[113] = uniforms.audioLow;
  f32[114] = uniforms.audioMid;
  f32[115] = uniforms.audioHigh;
  // Time-based color-mode cross-fade (slots 116-117; buffer grown 464B -> 480B for these).
  u32[116] = uniforms.particleColorModeFrom;
  f32[117] = uniforms.particleColorBlend;
  // Render-side optimization toggles (slots 118-123; buffer grown 480B -> 496B).
  u32[118] = uniforms.particleCutoffPrepass;
  f32[124] = Number.isFinite(uniforms.particleExponent) ? uniforms.particleExponent : 1;
  f32[125] = Number.isFinite(uniforms.particleBrightnessBoost) ? uniforms.particleBrightnessBoost : 1;
  f32[126] = Number.isFinite(uniforms.particleSupportSmoothing) ? uniforms.particleSupportSmoothing : 0;
  f32[127] = Number.isFinite(uniforms.particleHazeCull) ? uniforms.particleHazeCull : 0;
  return buffer;
}

function renderLayerToUniform(layer: RenderControls["renderLayer"]): number {
  if (layer === "particles") return 1;
  if (layer === "trails") return 2;
  if (layer === "debug-voxels") return 3;
  if (layer === "density") return 4;
  if (layer === "volume-density") return 5;
  if (layer === "accumulation") return 6;
  return 0;
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, Number.isFinite(value) ? value : min));
}

function numberChanged(a: number, b: number, epsilon = 0.000001): boolean {
  return Math.abs(a - b) > epsilon;
}

function clampRayStepCount(value: number): number {
  return Math.round(clampNumber(value, 8, 512) / 8) * 8;
}

function trailColorModeToUniform(mode: TrailColorMode): number {
  if (mode === "flow") return 1;
  if (mode === "thermal") return 2;
  if (mode === "tint") return 3;
  return 0;
}

function particleColorModeToUniform(mode: ParticleColorMode): number {
  if (mode === "gradient-inferno") return 1;
  if (mode === "gradient-magma") return 2;
  if (mode === "gradient-viridis") return 3;
  if (mode === "gradient-turbo") return 4;
  if (mode === "gradient-rainbow") return 5;
  if (mode === "gradient-spectral") return 6;
  if (mode === "gradient-plasma") return 7;
  if (mode === "gradient-cosmic") return 8;
  if (mode === "gradient-ice") return 9;
  if (mode === "gradient-ember") return 10;
  if (mode === "velocity") return 11;
  if (mode === "cohort") return 12;
  if (mode === "audio") return 13;
  // Velocity-driven variants (speed -> a specific ramp).
  if (mode === "velocity-inferno") return 14;
  if (mode === "velocity-viridis") return 15;
  if (mode === "velocity-spectral") return 16;
  if (mode === "velocity-cosmic") return 17;
  if (mode === "velocity-ice") return 18;
  // Audio-driven variants (band-shifted + bass brightness pulse -> a specific ramp).
  if (mode === "audio-magma") return 19;
  if (mode === "audio-viridis") return 20;
  if (mode === "audio-turbo") return 21;
  if (mode === "audio-cosmic") return 22;
  if (mode === "audio-ice") return 23;
  if (mode === "audio-ember") return 24;
  if (mode === "audio-plasma") return 25;
  return 0;
}

function hexColorToRgb01(value: string): [number, number, number] {
  const match = /^#([0-9a-f]{6})$/i.exec(value);
  if (!match) return [1, 1, 1];
  const hex = match[1];
  return [
    parseInt(hex.slice(0, 2), 16) / 255,
    parseInt(hex.slice(2, 4), 16) / 255,
    parseInt(hex.slice(4, 6), 16) / 255
  ];
}

const particleBlendModes = ["additive", "alpha", "opaque"] as const satisfies readonly ParticleBlendMode[];

function particleBlendModeToUniform(mode: ParticleBlendMode): number {
  if (mode === "alpha") return 1;
  if (mode === "opaque") return 2;
  return 0;
}

function createRuleData(rule: number[] = []): Float32Array {
  const packed = new Float32Array(80);
  for (let i = 0; i < Math.min(80, rule.length); i += 1) {
    packed[i] = Math.fround(rule[i]);
  }
  return packed;
}

function requiresBufferReset(previous: LiveGpu3dConfig, next: LiveGpu3dConfig): boolean {
  return (
    previous.width !== next.width ||
    previous.height !== next.height ||
    previous.depth !== next.depth ||
    previous.particleCount !== next.particleCount ||
    previous.seed !== next.seed ||
    previous.cohorts !== next.cohorts ||
    // boundaryMode is a per-step uniform (every mode keeps particles in-bounds), so it switches live.
    // initialConditions changes are absorbed by the GPU morph in update_particles, not a reseed.
    !sameRule(previous.rule, next.rule)
  );
}

// When initialConditions changes we morph particles toward the new layout over a fixed number of
// sim steps instead of reseeding. Per-step blend toward the target eases (fast then slow); the step
// budget is sized so the blend has visually converged before it stops.
const initialMorphAlpha = 0.13;
const initialMorphSteps = 64;

const defaultMaxStorageBufferBindingSize = 134217728;
const defaultMaxBufferSize = 268435456;
const particleByteStride = particleFloatCount * Float32Array.BYTES_PER_ELEMENT;

function particleBufferByteLength(config: Pick<LiveGpu3dConfig, "particleCount">): number {
  return config.particleCount * particleByteStride;
}

function particleSupportGridByteLength(size: number): number {
  return size * size * size * 4 * Int32Array.BYTES_PER_ELEMENT;
}

function particleSupportGridSizeForLimits(limits: GPUSupportedLimits): number {
  const maxBytes = Math.min(
    limits.maxBufferSize || defaultMaxBufferSize,
    limits.maxStorageBufferBindingSize || defaultMaxStorageBufferBindingSize
  );
  for (const size of [128, 96, 64]) {
    if (particleSupportGridByteLength(size) <= maxBytes) {
      return size;
    }
  }
  return 32;
}

function maxParticlesPerStorageBuffer(limits: GPUSupportedLimits): number {
  const maxBytes = Math.max(
    particleByteStride,
    Math.min(
      limits.maxBufferSize || defaultMaxBufferSize,
      limits.maxStorageBufferBindingSize || defaultMaxStorageBufferBindingSize
    )
  );
  return Math.max(1, Math.floor(maxBytes / particleByteStride));
}

function sameRule(a: number[] = [], b: number[] = []): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    if (Math.fround(a[i]) !== Math.fround(b[i])) return false;
  }
  return true;
}

function createInitialParticles(config: LiveGpu3dConfig, particleOffset = 0, particleCount = config.particleCount): Float32Array {
  const particles = new Float32Array(particleCount * particleFloatCount);
  for (let i = 0; i < particleCount; i += 1) {
    const id = particleOffset + i;
    const seeded = initialParticle(id, config);
    seeded.position = clampIntoShape(seeded.position, config.domainShape);
    const cohort = Math.floor((config.cohorts * id) / Math.max(1, config.particleCount));
    const color = initialParticleColor(cohort, config);
    const base = i * particleFloatCount;
    particles[base] = Math.fround(seeded.position[0]);
    particles[base + 1] = Math.fround(seeded.position[1]);
    particles[base + 2] = Math.fround(seeded.position[2]);
    particles[base + 3] = cohort;
    particles[base + 4] = Math.fround(seeded.velocity[0]);
    particles[base + 5] = Math.fround(seeded.velocity[1]);
    particles[base + 6] = Math.fround(seeded.velocity[2]);
    // vel_id.w now carries hue (was particle id; id is derivable from the buffer index).
    particles[base + 7] = Math.fround(color[0]);
  }
  return particles;
}

function initialParticleColor(cohort: number, config: LiveGpu3dConfig): [number, number, number, number] {
  const hue = config.colorByCohort ? hashVec2(Math.floor(cohort), Math.floor(cohort)) : 0;
  return [Math.fround(hue), 0.8, 1, 0.045];
}

function domainSdf(p: readonly [number, number, number], shape: number): number {
  if (shape === 1) return Math.hypot(p[0], p[1], p[2]) - 1;
  if (shape === 2) {
    const dRadial = Math.hypot(p[0], p[2]) - 1;
    const dCap = Math.abs(p[1]) - 1;
    return Math.hypot(Math.max(dRadial, 0), Math.max(dCap, 0)) + Math.min(Math.max(dRadial, dCap), 0);
  }
  if (shape === 3) {
    const halfW = (1 - p[1]) * 0.5;
    return Math.max(Math.max(-1 - p[1], Math.abs(p[0]) - halfW), Math.abs(p[2]) - halfW);
  }
  const qx = Math.abs(p[0]) - 1;
  const qy = Math.abs(p[1]) - 1;
  const qz = Math.abs(p[2]) - 1;
  return Math.hypot(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0)) + Math.min(Math.max(qx, qy, qz), 0);
}

function clampIntoShape(p: [number, number, number], shape: number): [number, number, number] {
  if (shape === 0) return p;
  const d = domainSdf(p, shape);
  if (d <= 0) return p;
  const e = 0.0015;
  const nx = domainSdf([p[0] + e, p[1], p[2]], shape) - domainSdf([p[0] - e, p[1], p[2]], shape);
  const ny = domainSdf([p[0], p[1] + e, p[2]], shape) - domainSdf([p[0], p[1] - e, p[2]], shape);
  const nz = domainSdf([p[0], p[1], p[2] + e], shape) - domainSdf([p[0], p[1], p[2] - e], shape);
  const k = (d + 0.0005) / (Math.hypot(nx, ny, nz) || 1);
  return [p[0] - nx * k, p[1] - ny * k, p[2] - nz * k];
}

function initialParticle(id: number, config: LiveGpu3dConfig): { position: [number, number, number]; velocity: [number, number, number] } {
  const theta = randomRange(config.seed, id, 0, 0, 0, Math.PI * 2);
  const cohortValue = config.cohorts * id / Math.max(1, config.particleCount);
  const cohortJitter = randomRange(config.seed, id, 0, 5, -0.045, 0.045);
  const vx = randomRange(config.seed, id, 0, 6, -0.00005, 0.00005);
  const vy = randomRange(config.seed, id, 0, 7, -0.00005, 0.00005);
  const vz = randomRange(config.seed, id, 0, 8, -0.00005, 0.00005);

  if (config.initialConditions === 0) {
    const side = Math.ceil(Math.cbrt(Math.max(1, config.cohorts)));
    const cohort = Math.floor(cohortValue);
    const gx = cohort % side;
    const gy = Math.floor(cohort / side) % side;
    const gz = Math.floor(cohort / (side * side)) % side;
    const cell = side > 1 ? 1.55 / (side - 1) : 0;
    const center: [number, number, number] = [
      -0.775 + gx * cell,
      -0.775 + gy * cell,
      -0.775 + gz * cell
    ];
    return {
      position: [
        wrapDomain(center[0] + randomRange(config.seed, id, 0, 9, -0.055, 0.055)),
        wrapDomain(center[1] + randomRange(config.seed, id, 0, 10, -0.055, 0.055)),
        wrapDomain(center[2] + randomRange(config.seed, id, 0, 11, -0.055, 0.055))
      ],
      velocity: [vx, vy, vz]
    };
  }

  if (config.initialConditions === 1) {
    return {
      position: [
        randomRange(config.seed, id, 0, 1, -0.86, 0.86),
        randomRange(config.seed, id, 0, 2, -0.86, 0.86),
        randomRange(config.seed, id, 0, 3, -0.86, 0.86)
      ],
      velocity: [vx, vy, vz]
    };
  }

  if (config.initialConditions === 3) {
    const probe = trailProbeParticle(id);
    return {
      position: probe.position,
      velocity: probe.velocity
    };
  }

  const u = id / Math.max(1, config.particleCount);

  if (config.initialConditions === 4) {
    // Double helix along Z (two strands offset by PI), flowing along the axis.
    const angle = u * 4 * Math.PI * 2 + (id % 2) * Math.PI;
    return {
      position: [
        0.55 * Math.cos(angle) + randomRange(config.seed, id, 0, 12, -0.04, 0.04),
        0.55 * Math.sin(angle) + randomRange(config.seed, id, 0, 13, -0.04, 0.04),
        -0.82 + 1.64 * u
      ],
      velocity: [-Math.sin(angle) * 0.02 + vx, Math.cos(angle) * 0.02 + vy, 0.004 + vz]
    };
  }

  if (config.initialConditions === 5) {
    // Flat disc in the XY plane (area-even radial distribution).
    const ang = randomRange(config.seed, id, 0, 12, 0, Math.PI * 2);
    const rad = Math.sqrt(randomRange(config.seed, id, 0, 13, 0, 1)) * 0.82;
    return {
      position: [Math.cos(ang) * rad, Math.sin(ang) * rad, randomRange(config.seed, id, 0, 14, -0.03, 0.03)],
      velocity: [vx, vy, vz]
    };
  }

  if (config.initialConditions === 6) {
    // Two blobs at +/-x drifting toward each other.
    const side = id % 2;
    const cx = side === 0 ? -0.45 : 0.45;
    const off = (s: number) => (randomRange(config.seed, id, 0, s, 0, 1) + randomRange(config.seed, id, 0, s + 3, 0, 1) - 1) * 0.22;
    return {
      position: [cx + off(12), off(13), off(14)],
      velocity: [(side === 0 ? 0.01 : -0.01) + vx, vy, vz]
    };
  }

  if (config.initialConditions === 7) {
    // Particles spread across the six cube faces, drifting inward.
    const f = Math.min(5, Math.floor(randomRange(config.seed, id, 0, 12, 0, 6)));
    const axis = f >> 1;
    const sign = (f & 1) ? 0.86 : -0.86;
    const a = randomRange(config.seed, id, 0, 13, -0.86, 0.86);
    const b = randomRange(config.seed, id, 0, 14, -0.86, 0.86);
    const position: [number, number, number] = axis === 0 ? [sign, a, b] : axis === 1 ? [a, sign, b] : [a, b, sign];
    const inv = 0.006 / Math.max(0.0001, Math.hypot(position[0], position[1], position[2]));
    return {
      position,
      velocity: [-position[0] * inv + vx, -position[1] * inv + vy, -position[2] * inv + vz]
    };
  }

  if (config.initialConditions === 8) {
    // Vortex ring (torus) swirling around the main ring.
    const phi = u * Math.PI * 2;
    const psi = randomRange(config.seed, id, 0, 12, 0, Math.PI * 2);
    const ring = 0.55 + 0.18 * Math.cos(psi);
    return {
      position: [ring * Math.cos(phi), ring * Math.sin(phi), 0.18 * Math.sin(psi)],
      velocity: [-Math.sin(phi) * 0.018 + vx, Math.cos(phi) * 0.018 + vy, vz]
    };
  }

  const z = randomRange(config.seed, id, 0, 1, -0.82, 0.82);
  const radial = Math.sqrt(Math.max(0.01, 1 - z * z));
  const radius = Math.pow(randomRange(config.seed, id, 0, 2, 0.08, 1), 0.55);
  const wobble = randomRange(config.seed, id, 0, 3, -0.18, 0.18);
  return {
    position: [
      Math.cos(theta) * radial * radius * 0.72,
      Math.sin(theta) * radial * radius * 0.72,
      wrapDomain((z + wobble) * radius * 0.72)
    ],
    velocity: [
      -Math.sin(theta) * radial * 0.032 + vx,
      Math.cos(theta) * radial * 0.032 + vy,
      randomRange(config.seed, id, 0, 4, -0.018, 0.018)
    ]
  };
}

function trailProbeParticle(id: number): { position: [number, number, number]; velocity: [number, number, number] } {
  const probes: Array<{ position: [number, number, number]; velocity: [number, number, number] }> = [
    { position: [-0.72, -0.45, 0], velocity: [0.009, 0, 0] },
    { position: [-0.72, 0, 0.0], velocity: [0.009, 0, 0] },
    { position: [-0.72, 0.45, 0], velocity: [0.009, 0, 0] },
    { position: [0.72, -0.25, 0], velocity: [-0.009, 0, 0] },
    { position: [0, -0.72, 0], velocity: [0, 0.009, 0] },
    { position: [0, 0.72, 0], velocity: [0, -0.009, 0] },
    { position: [-0.42, -0.42, 0], velocity: [0.0065, 0.0065, 0] },
    { position: [0.42, 0.42, 0], velocity: [-0.0065, -0.0065, 0] }
  ];
  return probes[id % probes.length];
}

function randomRange(seed: number, entity: number, timestep: number, stream: number, min: number, max: number): number {
  return min + (max - min) * randomUnit(seed, entity, timestep, stream);
}

function randomUnit(seed: number, entity: number, timestep: number, stream: number): number {
  let x = (seed ^ Math.imul(entity + 1, 0x9e3779b1) ^ Math.imul(timestep + 1, 0x85ebca6b) ^ Math.imul(stream + 1, 0xc2b2ae35)) >>> 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x7feb352d) >>> 0;
  x ^= x >>> 15;
  x = Math.imul(x, 0x846ca68b) >>> 0;
  x ^= x >>> 16;
  return (x >>> 0) / 0xffffffff;
}

function wrapDomain(value: number): number {
  return Math.fround(2 * (fract(value / 2 - 0.5) - 0.5));
}

function depositScale(config: LiveGpu3dConfig): number {
  const voxelSize = 2 / Math.max(1, config.width);
  const radiusCells = Math.max(1, config.depositRadius / voxelSize);
  const footprint = Math.max(1, Math.pow(radiusCells, 3) * 8);
  const voxels = config.width * config.height * config.depth;
  return Math.min(1, voxels / Math.max(1, config.particleCount * footprint));
}

function fract(value: number): number {
  return value - Math.floor(value);
}

export function computeRuleEvidence(config: LiveGpu3dConfig = defaultLiveGpu3dConfig): LiveRuleEvidence {
  const first = ruleSignature(config.ruleSeed, 0, config.mutationScale, config.rule);
  const repeat = ruleSignature(config.ruleSeed, 0, config.mutationScale, config.rule);
  const second = ruleSignature(config.ruleSeed, Math.max(1, Math.min(config.cohorts - 1, 2)), config.mutationScale, config.rule);
  return {
    nonzero: Math.abs(first) > 0.001,
    cohortDependent: Math.abs(first - second) > 0.001,
    stable: first === repeat,
    firstSignature: first,
    secondSignature: second
  };
}

function ruleSignature(seed: number, cohort: number, mutationScale: number, sourceRule: number[]): number {
  const rule = sourceRule.some((value) => Math.abs(value) > 0.000001)
    ? createRuleData(sourceRule)
    : generateRule(seed + Math.floor(cohort));
  mutateRule(rule, mutationScale, seed + Math.floor(cohort));
  let total = 0;
  for (let i = 0; i < rule.length; i += 1) {
    total = Math.fround(total + Math.fround(rule[i] * Math.sin(i + 1)));
  }
  return total;
}

function generateRule(seed: number): Float32Array {
  const rule = new Float32Array(80);
  for (let i = 0; i < 10; i += 1) {
    const freqScale = Math.fround(1 + 2 * Math.pow(hash2(seed, i * 8), 2));
    for (let j = 0; j < 4; j += 1) {
      rule[i * 8 + j] = Math.fround((hash2(seed, i * 8 + j) * 2 - 1) * freqScale);
    }
    for (let j = 0; j < 4; j += 1) {
      rule[i * 8 + 4 + j] = Math.fround(hash2(seed, i * 8 + 4 + j) * 2 - 1);
    }
  }
  return rule;
}

function mutateRule(rule: Float32Array, amount: number, cohort: number): void {
  const seed = hashVec2(
    rule[4 * 8] + rule[7 * 8 + 5] + rule[1 * 8 + 2],
    rule[4 * 8 + 1] + rule[7 * 8 + 4] + rule[1 * 8 + 3]
  ) + cohort;
  for (let i = 0; i < 10; i += 1) {
    const h0 = hashVec2(-0.5 - i + seed, -0.5 + i);
    const h1 = hashVec2(0.5 + i - seed, 5.5 - i);
    const h2 = hashVec2(i - 100, seed + 100);
    const h3 = hashVec2(25 - i, 25 - seed);
    rule[i * 8 + 4] = Math.fround(rule[i * 8 + 4] + amount * (-1 + 2 * h0));
    rule[i * 8 + 5] = Math.fround(rule[i * 8 + 5] + amount * (-1 + 2 * h1));
    rule[i * 8 + 6] = Math.fround(rule[i * 8 + 6] + amount * (-1 + 2 * h2));
    rule[i * 8 + 7] = Math.fround(rule[i * 8 + 7] + amount * (-1 + 2 * h3));
    const scale = Math.fround(1 + amount * 0.5 * (hashVec2(seed, i) - 0.5));
    for (let j = 0; j < 4; j += 1) {
      rule[i * 8 + j] = Math.fround(rule[i * 8 + j] * scale);
    }
  }
}

function hash2(seed: number, stream: number): number {
  return hashVec2(seed, stream);
}

const bitcastBuffer = new ArrayBuffer(4);
const bitcastF32 = new Float32Array(bitcastBuffer);
const bitcastU32 = new Uint32Array(bitcastBuffer);

function hashVec2(x: number, y: number): number {
  const ux = floatBitsToUint(x);
  const uy = floatBitsToUint(y);
  const h = pcgHash((ux ^ pcgHash(uy)) >>> 0);
  return h / 0xffffffff;
}

function floatBitsToUint(value: number): number {
  bitcastF32[0] = Math.fround(value);
  return bitcastU32[0] >>> 0;
}

function pcgHash(seed: number): number {
  let state = (Math.imul(seed >>> 0, 747796405) + 2891336453) >>> 0;
  const shift = ((state >>> 28) + 4) >>> 0;
  let word = Math.imul(((state >>> shift) ^ state) >>> 0, 277803737) >>> 0;
  word = ((word >>> 22) ^ word) >>> 0;
  return word;
}

const commonComputeShader = /* wgsl */ `
const PI = 3.141592653589793;
const DOMAIN_SIZE = 2.0;
// Half-saturation density for the emergent behaviors that read the (small ~1e-4) w/density channel.
// Used as a scale-robust Hill normalization d/(d+DENSITY_HALF) so MIPS/quorum/Lenia respond to
// relative crowding regardless of the absolute deposit scale, instead of a fragile linear threshold.
const DENSITY_HALF = 0.0001;
// Per-step density a fully-on Lenia growth term adds (a few x the natural deposit so it can build
// visible self-organising structure without instantly saturating the field clamp).
const LENIA_RATE = 0.0006;
// Hard speed ceiling applied only while an emergent steering behavior is active. The field-following
// terms add velocity proportional to the current speed along the deposited current, which can compound
// at low drag; this bounds it. Far above normal sim speeds, so it only ever catches a runaway.
const EMERGENT_VMAX = 0.5;
// Per-step strafe-slip budget in grid cells (CFL-style). Beyond this a slip outruns the field structure
// it senses in one step; for a position-derived orientation that is the runaway driving the 2-cycle
// jitter, so the update_particles anti-oscillation override drops such slips. Only ever trips on the
// degenerate runaway (~10-24 cells at the bug preset's density); normal motion stays well under it.
const STRAFE_MAX_CELLS = 2.0;
// FIX-S strafe-into-velocity gain. The sensing strafe is folded into velocity (not applied raw to position)
// when strafe_momentum is enabled; this gain keeps the velocity path in the same visual scale as the classic
// per-step position slip while drag supplies damping.
const STRAFE_VELOCITY_GAIN = 0.25;

struct Particle {
  pos_cohort: vec4f,  // xyz = position, w = cohort
  vel_id: vec4f,      // xyz = velocity, w = hue (was particle id; id is derivable from index)
};

struct SimConfig {
  width: u32,
  height: u32,
  depth: u32,
  particle_count: u32,
  timestep: u32,
  seed: u32,
  cohorts: u32,
  boundary_mode: u32,
  dt: f32,
  sensor_gain: f32,
  sensor_angle: f32,
  sensor_distance: f32,
  mutation_scale: f32,
  global_force_mult: f32,
  drag: f32,
  strafe_power: f32,
  axial_force: f32,
  lateral_force: f32,
  hazard_rate: f32,
  trail_persistence: f32,
  trail_diffusion: f32,
  deposit_radius: f32,
  deposit_mass: f32,
  sigma: f32,
  rule_seed: f32,
  orientation_mix: f32,
  symmetry_axes: u32,
  absolute_orientation: u32,
  initial_conditions: u32,
  deposit_tap_radius: u32,
  hue_sensitivity: f32,
  color_by_cohort: u32,
  step_time_scale: f32,
  visual_lerp_t: f32,
  recycle_cutoff: f32,
  recycle_enabled: u32,
  particle_offset: u32,
  active_particle_count: u32,
  domain_shape: u32,
  morph_alpha: f32,
  pulse_depth: f32,
  pulse_rate: f32,
  restlessness: f32,
  // Emergent behaviors (slots 43..60, same order as encodeSimUniforms). 0 = off = byte-identical.
  mips: f32,
  aniso_follow: f32,
  flock_align: f32,
  flock_separate: f32,
  chemotaxis: f32,
  quorum_strength: f32,
  quorum_threshold: f32,
  lenia_strength: f32,
  lenia_center: f32,
  lenia_width: f32,
  species_force: f32,
  predator: f32,
  alarm: f32,
  gray_scott: f32,
  gs_feed: f32,
  gs_kill: f32,
  energy: f32,
  energy_drain: f32,
  // Strafe momentum blend (slot 61). 0 = classic strafe-to-position; 1 = strafe folded into velocity.
  strafe_momentum: f32,
};

struct FourierCenter {
  frequency: vec4f,
  amplitude: vec4f,
};

struct Rule {
  centers: array<FourierCenter, 10>,
};

struct PlaneBehavior {
  force: vec3f,
  strafe: vec3f,
  color: vec2f,
};

fn pcg_hash(seed: u32) -> u32 {
  let state = seed * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn hash(co: vec2f) -> f32 {
  let u = vec2u(bitcast<u32>(co.x), bitcast<u32>(co.y));
  let h = pcg_hash(u.x ^ pcg_hash(u.y));
  return f32(h) / 4294967295.0;
}

fn hash4(co: vec2f) -> vec4f {
  return vec4f(
    hash(co),
    hash(co * -1.0 + vec2f(5.0)),
    hash(co.yx - vec2f(100.0)),
    hash(co.yx * -1.0 + vec2f(25.0))
  );
}

fn fourier_noise(rule: Rule, signals: vec4f) -> vec4f {
  var result = vec4f(0.0);
  for (var i = 0u; i < 10u; i = i + 1u) {
    let phase = dot(signals, rule.centers[i].frequency);
    let phase_offset = 2.0 * f32(i) * 0.6283 + rule.centers[i].amplitude.w * 3.14159;
    let basis = vec4f(
      sin(phase + phase_offset),
      cos(phase + phase_offset * 0.7),
      sin(phase * 2.0 + phase_offset * 1.3),
      cos(phase * 2.0 + phase_offset * 0.5)
    );
    result = result + rule.centers[i].amplitude * basis;
  }
  return result;
}

fn generate_random_centers(seed: f32) -> Rule {
  var rule: Rule;
  for (var i = 0u; i < 10u; i = i + 1u) {
    let base = f32(i * 8u);
    let freq_scale = 1.0 + 2.0 * pow(hash(vec2f(seed, base)), 2.0);
    rule.centers[i].frequency = vec4f(
      (hash(vec2f(seed, base + 0.0)) * 2.0 - 1.0) * freq_scale,
      (hash(vec2f(seed, base + 1.0)) * 2.0 - 1.0) * freq_scale,
      (hash(vec2f(seed, base + 2.0)) * 2.0 - 1.0) * freq_scale,
      (hash(vec2f(seed, base + 3.0)) * 2.0 - 1.0) * freq_scale
    );
    rule.centers[i].amplitude = vec4f(
      hash(vec2f(seed, base + 4.0)) * 2.0 - 1.0,
      hash(vec2f(seed, base + 5.0)) * 2.0 - 1.0,
      hash(vec2f(seed, base + 6.0)) * 2.0 - 1.0,
      hash(vec2f(seed, base + 7.0)) * 2.0 - 1.0
    );
  }
  return rule;
}

fn mutate_rule(rule_in: Rule, amount: f32, cohort: f32) -> Rule {
  var rule = rule_in;
  let seed = hash(rule.centers[4].frequency.xy + rule.centers[7].amplitude.yx + rule.centers[1].frequency.zw) + cohort;
  for (var i = 0u; i < 10u; i = i + 1u) {
    let amp_mutation = amount * (-1.0 + 2.0 * hash4(-0.5 + vec2f(-f32(i) + seed, f32(i))));
    rule.centers[i].amplitude = rule.centers[i].amplitude + amp_mutation;
    rule.centers[i].frequency = rule.centers[i].frequency * (1.0 + amount * 0.5 * (hash(vec2f(seed, f32(i))) - 0.5));
  }
  return rule;
}

fn y_reflect(p: vec2f) -> vec2f {
  return p * vec2f(1.0, -1.0);
}

// Per-axis in-plane reflection for the XYZ symmetry modes. flip_fwd flips the forward (axial)
// component, flip_lat flips the lateral component. Note y_reflect(p) == reflect2(p, false, true).
fn reflect2(p: vec2f, flip_fwd: bool, flip_lat: bool) -> vec2f {
  return p * vec2f(select(1.0, -1.0, flip_fwd), select(1.0, -1.0, flip_lat));
}

fn black_box(left_signal: vec2f, right_signal: vec2f, rule: Rule) -> vec4f {
  return fourier_noise(rule, vec4f(left_signal, right_signal));
}

fn safe_normalize(v: vec3f) -> vec3f {
  let len = length(v);
  if (len <= 0.000001) {
    return vec3f(1.0, 0.0, 0.0);
  }
  return v / len;
}

fn guard_velocity_reversal(previous_velocity: vec3f, candidate_velocity: vec3f) -> vec3f {
  let previous_speed = length(previous_velocity);
  if (previous_speed <= 0.00001) {
    return candidate_velocity;
  }
  let previous_dir = previous_velocity / previous_speed;
  let forward_speed = dot(candidate_velocity, previous_dir);
  if (forward_speed >= 0.0) {
    return candidate_velocity;
  }
  let sideways_velocity = candidate_velocity - previous_dir * forward_speed;
  return sideways_velocity + previous_dir * max(0.00001, previous_speed * 0.02);
}

fn sim_time_scale(config: SimConfig) -> f32 {
  return clamp(config.step_time_scale, 0.0, 1.0);
}

fn velocity_drag_factor(drag: f32, time_scale: f32) -> f32 {
  if (drag >= 0.0) {
    return pow(clamp(drag, 0.0, 1.0), time_scale);
  }
  // Negative drag is a deliberately discrete sign-flip effect, so there is no
  // real-valued fractional step that composes exactly. Keep it finite and match
  // the authored full-step value when time_scale reaches 1.
  return 1.0 + (drag - 1.0) * time_scale;
}

fn velocity_force_factor(drag: f32, drag_factor: f32, time_scale: f32) -> f32 {
  if (drag >= 0.0 && abs(1.0 - drag) > 0.000001) {
    return (1.0 - drag_factor) / max(0.000001, 1.0 - drag);
  }
  return time_scale;
}

fn wrap_domain_scalar(value: f32) -> f32 {
  return 2.0 * (fract(value / 2.0 - 0.5) - 0.5);
}

fn wrap_domain_vec(value: vec3f) -> vec3f {
  return vec3f(
    wrap_domain_scalar(value.x),
    wrap_domain_scalar(value.y),
    wrap_domain_scalar(value.z)
  );
}

// Procedural domain shapes inscribed in the [-1,1] cube. domain_sdf < 0 inside, > 0 outside.
// shape: 0 Cube, 1 Sphere, 2 Cylinder (vertical Y axis), 3 Pyramid (base at y=-1, apex at +y).
fn domain_sdf(p: vec3f, shape: u32) -> f32 {
  if (shape == 1u) {
    return length(p) - 1.0;
  }
  if (shape == 2u) {
    let d_radial = length(p.xz) - 1.0;
    let d_cap = abs(p.y) - 1.0;
    let outside = length(max(vec2f(d_radial, d_cap), vec2f(0.0)));
    let inside = min(max(d_radial, d_cap), 0.0);
    return outside + inside;
  }
  if (shape == 3u) {
    let half_w = (1.0 - p.y) * 0.5;
    let d_base = -1.0 - p.y;
    return max(max(d_base, abs(p.x) - half_w), abs(p.z) - half_w);
  }
  let q = abs(p) - vec3f(1.0);
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn domain_sdf_normal(p: vec3f, shape: u32) -> vec3f {
  let e = 0.0015;
  let dx = domain_sdf(p + vec3f(e, 0.0, 0.0), shape) - domain_sdf(p - vec3f(e, 0.0, 0.0), shape);
  let dy = domain_sdf(p + vec3f(0.0, e, 0.0), shape) - domain_sdf(p - vec3f(0.0, e, 0.0), shape);
  let dz = domain_sdf(p + vec3f(0.0, 0.0, e), shape) - domain_sdf(p - vec3f(0.0, 0.0, e), shape);
  return safe_normalize(vec3f(dx, dy, dz));
}

// Pull a point onto/just inside the shape surface if it has left the shape; no-op for the cube.
fn clamp_into_shape(p: vec3f, shape: u32) -> vec3f {
  if (shape == 0u) {
    return p;
  }
  let d = domain_sdf(p, shape);
  if (d <= 0.0) {
    return p;
  }
  return p - domain_sdf_normal(p, shape) * (d + 0.0005);
}

fn wrap_i(value: i32, size: u32) -> u32 {
  let s = i32(size);
  return u32(((value % s) + s) % s);
}

fn index_xyz(x: i32, y: i32, z: i32, config: SimConfig) -> u32 {
  let ix = wrap_i(x, config.width);
  let iy = wrap_i(y, config.height);
  let iz = wrap_i(z, config.depth);
  return (iz * config.height + iy) * config.width + ix;
}

fn grid_to_world(x: u32, y: u32, z: u32, config: SimConfig) -> vec3f {
  return vec3f(
    ((f32(x) + 0.5) / f32(config.width)) * 2.0 - 1.0,
    ((f32(y) + 0.5) / f32(config.height)) * 2.0 - 1.0,
    ((f32(z) + 0.5) / f32(config.depth)) * 2.0 - 1.0
  );
}

fn periodic_delta(a: f32, b: f32) -> f32 {
  var delta = a - b;
  delta = delta - round(delta / DOMAIN_SIZE) * DOMAIN_SIZE;
  return delta;
}

fn rotate_sensor(forward: vec3f, lateral: vec3f, angle: f32, dist: f32) -> vec3f {
  return (forward * cos(angle) + lateral * sin(angle)) * dist;
}

fn build_frame(velocity: vec3f) -> mat3x3f {
  let forward = safe_normalize(velocity);
  // Branchless orthonormal basis: keeps the sensor frame continuous as velocity
  // passes near world-up/world-down instead of swapping reference axes.
  let sign = select(-1.0, 1.0, forward.z >= 0.0);
  let a = -1.0 / (sign + forward.z);
  let b = forward.x * forward.y * a;
  let left = safe_normalize(vec3f(1.0 + sign * forward.x * forward.x * a, sign * b, -sign * forward.x));
  let up = safe_normalize(vec3f(b, sign + forward.y * forward.y * a, -forward.y));
  return mat3x3f(forward, left, up);
}

fn calculate_plane_behavior(left_sample_3d: vec3f, right_sample_3d: vec3f, forward: vec3f, lateral: vec3f, rule: Rule, config: SimConfig) -> PlaneBehavior {
  let left_sample = vec2f(dot(left_sample_3d, forward), dot(left_sample_3d, lateral));
  let right_sample = vec2f(dot(right_sample_3d, forward), dot(right_sample_3d, lateral));
  let baseterm = black_box(left_sample, right_sample, rule);
  var local_force = baseterm.xy;
  var local_strafe = baseterm.zw;
  var color = baseterm.xy;
  // XYZ symmetry: each enabled axis adds an independent reflected rule term, generalizing the old
  // single y-reflection. Bit 0 = X (forward flip), bit 1 = Y (lateral flip -- reproduces the legacy
  // symmetry exactly), bit 2 = Z (both / point reflection). symmetry_axes == 0u => no mirror.
  let sym = config.symmetry_axes;
  if ((sym & 1u) != 0u) {
    let m = black_box(reflect2(right_sample, true, false), reflect2(left_sample, true, false), rule);
    local_force = local_force + reflect2(m.xy, true, false);
    local_strafe = local_strafe + reflect2(m.zw, true, false);
    color = color + m.xy;
  }
  if ((sym & 2u) != 0u) {
    let m = black_box(reflect2(right_sample, false, true), reflect2(left_sample, false, true), rule);
    local_force = local_force + reflect2(m.xy, false, true);
    local_strafe = local_strafe + reflect2(m.zw, false, true);
    color = color + m.xy;
  }
  if ((sym & 4u) != 0u) {
    let m = black_box(reflect2(right_sample, true, true), reflect2(left_sample, true, true), rule);
    local_force = local_force + reflect2(m.xy, true, true);
    local_strafe = local_strafe + reflect2(m.zw, true, true);
    color = color + m.xy;
  }
  return PlaneBehavior(
    forward * local_force.x * config.axial_force + lateral * local_force.y * config.lateral_force,
    forward * local_strafe.x * config.axial_force + lateral * local_strafe.y * config.lateral_force,
    color
  );
}

fn initial_particle_color(cohort: f32, config: SimConfig) -> vec4f {
  var hue = 0.0;
  if (config.color_by_cohort == 1u) {
    hue = hash(vec2f(floor(cohort), floor(cohort)));
  }
  return vec4f(hue, 0.8, 1.0, 0.045);
}

fn orientation_axis(position: vec3f, velocity: vec3f, config: SimConfig) -> vec3f {
  var axis = safe_normalize(velocity);
  let mix_amount = min(1.0, f32(config.absolute_orientation)) * clamp(config.orientation_mix, 0.0, 1.0);
  if (config.absolute_orientation == 1u) {
    axis = safe_normalize(mix(axis, vec3f(0.0, 1.0, 0.0), mix_amount));
  } else if (config.absolute_orientation == 2u) {
    axis = safe_normalize(mix(axis, -safe_normalize(position), mix_amount));
  } else if (config.absolute_orientation == 3u) {
    // Outward: orient away from the origin (opposite of Radial) -> spreading / exploding flows.
    axis = safe_normalize(mix(axis, safe_normalize(position), mix_amount));
  } else if (config.absolute_orientation == 4u) {
    // Swirl: orient tangent to the world-Y axis -> a global vortex.
    let radial = position - vec3f(0.0, position.y, 0.0);
    let tangent = safe_normalize(cross(vec3f(0.0, 1.0, 0.0), radial));
    axis = safe_normalize(mix(axis, tangent, mix_amount));
  } else if (config.absolute_orientation == 5u) {
    // Noise-aligned: orient along a hash-based vector field -> turbulent marbling.
    let n = vec3f(
      hash(vec2f(position.x * 3.1 + 11.0, position.y * 3.1 - 7.0)) - 0.5,
      hash(vec2f(position.y * 3.1 + 5.0, position.z * 3.1 + 2.0)) - 0.5,
      hash(vec2f(position.z * 3.1 - 9.0, position.x * 3.1 + 4.0)) - 0.5
    );
    axis = safe_normalize(mix(axis, safe_normalize(n), mix_amount));
  }
  return axis;
}

fn reset_particle_raw(id: u32, config: SimConfig) -> Particle {
  let cohort = floor(f32(config.cohorts) * f32(id) / max(1.0, f32(config.particle_count)));
  let jitter = vec3f(
    hash(vec2f(f32(config.seed), f32(id) + 9.0)) * 0.11 - 0.055,
    hash(vec2f(f32(config.seed), f32(id) + 10.0)) * 0.11 - 0.055,
    hash(vec2f(f32(config.seed), f32(id) + 11.0)) * 0.11 - 0.055
  );
  let random_velocity = vec3f(
    hash(vec2f(f32(config.seed), f32(id) + 6.0)) * 0.0001 - 0.00005,
    hash(vec2f(f32(config.seed), f32(id) + 7.0)) * 0.0001 - 0.00005,
    hash(vec2f(f32(config.seed), f32(id) + 8.0)) * 0.0001 - 0.00005
  );
  if (config.initial_conditions == 0u) {
    let side = u32(ceil(pow(max(1.0, f32(config.cohorts)), 1.0 / 3.0)));
    let c = u32(cohort);
    let gx = c % side;
    let gy = (c / side) % side;
    let gz = c / max(1u, side * side);
    var cell = 0.0;
    if (side > 1u) {
      cell = 1.55 / f32(side - 1u);
    }
    let center = vec3f(-0.775 + f32(gx) * cell, -0.775 + f32(gy) * cell, -0.775 + f32(gz) * cell);
    return Particle(vec4f(wrap_domain_vec(center + jitter), cohort), vec4f(random_velocity, initial_particle_color(cohort, config).x));
  }
  if (config.initial_conditions == 1u) {
    let pos = vec3f(
      hash(vec2f(f32(config.seed), f32(id) + 1.0)) * 1.72 - 0.86,
      hash(vec2f(f32(config.seed), f32(id) + 2.0)) * 1.72 - 0.86,
      hash(vec2f(f32(config.seed), f32(id) + 3.0)) * 1.72 - 0.86
    );
    return Particle(vec4f(pos, cohort), vec4f(random_velocity, initial_particle_color(cohort, config).x));
  }
  if (config.initial_conditions == 3u) {
    let slot = id % 8u;
    var pos = vec3f(-0.72, -0.45, 0.0);
    var vel = vec3f(0.009, 0.0, 0.0);
    if (slot == 1u) {
      pos = vec3f(-0.72, 0.0, 0.0);
      vel = vec3f(0.009, 0.0, 0.0);
    } else if (slot == 2u) {
      pos = vec3f(-0.72, 0.45, 0.0);
      vel = vec3f(0.009, 0.0, 0.0);
    } else if (slot == 3u) {
      pos = vec3f(0.72, -0.25, 0.0);
      vel = vec3f(-0.009, 0.0, 0.0);
    } else if (slot == 4u) {
      pos = vec3f(0.0, -0.72, 0.0);
      vel = vec3f(0.0, 0.009, 0.0);
    } else if (slot == 5u) {
      pos = vec3f(0.0, 0.72, 0.0);
      vel = vec3f(0.0, -0.009, 0.0);
    } else if (slot == 6u) {
      pos = vec3f(-0.42, -0.42, 0.0);
      vel = vec3f(0.0065, 0.0065, 0.0);
    } else if (slot == 7u) {
      pos = vec3f(0.42, 0.42, 0.0);
      vel = vec3f(-0.0065, -0.0065, 0.0);
    }
    return Particle(vec4f(pos, cohort), vec4f(vel, initial_particle_color(cohort, config).x));
  }
  let u_frac = f32(id) / max(1.0, f32(config.particle_count));
  if (config.initial_conditions == 4u) {
    let angle = u_frac * 4.0 * PI * 2.0 + f32(id % 2u) * PI;
    let pos = vec3f(
      0.55 * cos(angle) + (hash(vec2f(f32(config.seed), f32(id) + 12.0)) * 0.08 - 0.04),
      0.55 * sin(angle) + (hash(vec2f(f32(config.seed), f32(id) + 13.0)) * 0.08 - 0.04),
      -0.82 + 1.64 * u_frac
    );
    let vel = vec3f(-sin(angle) * 0.02, cos(angle) * 0.02, 0.004) + random_velocity;
    return Particle(vec4f(pos, cohort), vec4f(vel, initial_particle_color(cohort, config).x));
  }
  if (config.initial_conditions == 5u) {
    let ang = hash(vec2f(f32(config.seed), f32(id) + 12.0)) * PI * 2.0;
    let rad = sqrt(hash(vec2f(f32(config.seed), f32(id) + 13.0))) * 0.82;
    let pos = vec3f(cos(ang) * rad, sin(ang) * rad, hash(vec2f(f32(config.seed), f32(id) + 14.0)) * 0.06 - 0.03);
    return Particle(vec4f(pos, cohort), vec4f(random_velocity, initial_particle_color(cohort, config).x));
  }
  if (config.initial_conditions == 6u) {
    let side = id % 2u;
    let cx = select(0.45, -0.45, side == 0u);
    let ox = (hash(vec2f(f32(config.seed), f32(id) + 12.0)) + hash(vec2f(f32(config.seed), f32(id) + 15.0)) - 1.0) * 0.22;
    let oy = (hash(vec2f(f32(config.seed), f32(id) + 13.0)) + hash(vec2f(f32(config.seed), f32(id) + 16.0)) - 1.0) * 0.22;
    let oz = (hash(vec2f(f32(config.seed), f32(id) + 14.0)) + hash(vec2f(f32(config.seed), f32(id) + 17.0)) - 1.0) * 0.22;
    let pos = vec3f(cx + ox, oy, oz);
    let vel = vec3f(select(-0.01, 0.01, side == 0u), 0.0, 0.0) + random_velocity;
    return Particle(vec4f(pos, cohort), vec4f(vel, initial_particle_color(cohort, config).x));
  }
  if (config.initial_conditions == 7u) {
    let f = min(5u, u32(floor(hash(vec2f(f32(config.seed), f32(id) + 12.0)) * 6.0)));
    let axis = f >> 1u;
    let sgn = select(-0.86, 0.86, (f & 1u) == 1u);
    let a = hash(vec2f(f32(config.seed), f32(id) + 13.0)) * 1.72 - 0.86;
    let b = hash(vec2f(f32(config.seed), f32(id) + 14.0)) * 1.72 - 0.86;
    var pos = vec3f(sgn, a, b);
    if (axis == 1u) { pos = vec3f(a, sgn, b); }
    else if (axis == 2u) { pos = vec3f(a, b, sgn); }
    let vel = -safe_normalize(pos) * 0.006 + random_velocity;
    return Particle(vec4f(pos, cohort), vec4f(vel, initial_particle_color(cohort, config).x));
  }
  if (config.initial_conditions == 8u) {
    let phi = u_frac * PI * 2.0;
    let psi = hash(vec2f(f32(config.seed), f32(id) + 12.0)) * PI * 2.0;
    let ring = 0.55 + 0.18 * cos(psi);
    let pos = vec3f(ring * cos(phi), ring * sin(phi), 0.18 * sin(psi));
    let vel = vec3f(-sin(phi) * 0.018, cos(phi) * 0.018, 0.0) + random_velocity;
    return Particle(vec4f(pos, cohort), vec4f(vel, initial_particle_color(cohort, config).x));
  }
  let theta = hash(vec2f(f32(config.seed), f32(id) + 0.0)) * PI * 2.0;
  let z = hash(vec2f(f32(config.seed), f32(id) + 1.0)) * 1.64 - 0.82;
  let radial = sqrt(max(0.01, 1.0 - z * z));
  let radius = pow(hash(vec2f(f32(config.seed), f32(id) + 2.0)) * 0.92 + 0.08, 0.55);
  let wobble = hash(vec2f(f32(config.seed), f32(id) + 3.0)) * 0.36 - 0.18;
  let pos = vec3f(cos(theta) * radial * radius * 0.72, sin(theta) * radial * radius * 0.72, (z + wobble) * radius * 0.72);
  let vel = vec3f(-sin(theta) * radial * 0.032, cos(theta) * radial * 0.032, hash(vec2f(f32(config.seed), f32(id) + 4.0)) * 0.036 - 0.018) + random_velocity;
  return Particle(vec4f(wrap_domain_vec(pos), cohort), vec4f(vel, initial_particle_color(cohort, config).x));
}

// Wrapper: confine every spawn/respawn to the active domain shape (no-op for the cube).
fn reset_particle(id: u32, config: SimConfig) -> Particle {
  var particle = reset_particle_raw(id, config);
  particle.pos_cohort = vec4f(clamp_into_shape(particle.pos_cohort.xyz, config.domain_shape), particle.pos_cohort.w);
  return particle;
}
`;

const liveComputeClearBrushShader = /* wgsl */ `
${commonComputeShader}

@group(0) @binding(0) var<storage, read_write> brush_out: array<atomic<i32>>;
@group(0) @binding(1) var<uniform> config: SimConfig;

@compute @workgroup_size(256)
fn clear_brush(@builtin(global_invocation_id) gid: vec3u) {
  let flat = gid.x;
  let lane_count = config.width * config.height * config.depth * 4u;
  if (flat >= lane_count) {
    return;
  }
  atomicStore(&brush_out[flat], 0);
}
`;

const liveComputeInitParticleShader = /* wgsl */ `
${commonComputeShader}

@group(0) @binding(0) var<storage, read_write> particles_out: array<Particle>;
@group(0) @binding(1) var<uniform> config: SimConfig;

@compute @workgroup_size(256)
fn init_particles(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= config.active_particle_count) {
    return;
  }
  particles_out[i] = reset_particle(i + config.particle_offset, config);
}
`;

const liveComputeDepositShader = /* wgsl */ `
${commonComputeShader}

const BRUSH_SCALE = 65536.0;
const FLOW_DEPOSIT_GAIN = 48.0;

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> brush_out: array<atomic<i32>>;
@group(0) @binding(2) var<uniform> config: SimConfig;
@group(0) @binding(4) var<storage, read> particle_ids: array<u32>;

fn add_brush(voxel: u32, component: u32, value: f32) {
  let scaled = i32(round(clamp(value, -4096.0, 4096.0) * BRUSH_SCALE));
  if (scaled != 0) {
    atomicAdd(&brush_out[voxel * 4u + component], scaled);
  }
}

fn safe_direction(value: vec3f, fallback_seed: f32) -> vec3f {
  let len_value = length(value);
  if (len_value > 0.000001) {
    return value / len_value;
  }
  let theta = hash(vec2f(fallback_seed, 17.0)) * PI * 2.0;
  let z = hash(vec2f(fallback_seed, 19.0)) * 2.0 - 1.0;
  let r = sqrt(max(0.0, 1.0 - z * z));
  return vec3f(cos(theta) * r, sin(theta) * r, z);
}

fn deposit_scale(config: SimConfig) -> f32 {
  let voxel_size = 2.0 / max(1.0, f32(config.width));
  let radius_cells = max(1.0, config.deposit_radius / voxel_size);
  let footprint = max(1.0, pow(radius_cells, 3.0) * 8.0);
  let voxels = config.width * config.height * config.depth;
  return min(1.0, f32(voxels) / max(1.0, f32(config.particle_count) * footprint));
}

fn smoothstep_range(edge0: f32, edge1: f32, value: f32) -> f32 {
  let t = clamp((value - edge0) / max(0.000001, edge1 - edge0), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

fn cell_weight(offset: i32, fraction: f32) -> f32 {
  if (offset == 0i) {
    return 1.0 - fraction;
  }
  if (offset == 1i) {
    return fraction;
  }
  return 0.0;
}

// Workgroup deposit tile: with spatially sorted particles a 256-thread workgroup touches a
// handful of neighboring field cells, so per-tap contributions are first accumulated in an
// 8^3-cell shared-memory tile (one i32 lane per cell component) and flushed to the global
// brush once per lane. Integer adds are associative, so the final sums are bit-identical to
// per-tap global atomics. Taps outside the tile (unsorted startup, huge radius) fall back to
// direct global atomics, keeping the kernel correct regardless of ordering.
const DEPOSIT_TILE_DIM = 8i;
const DEPOSIT_TILE_LANES = 2048u; // 8^3 cells x 4 components
var<workgroup> deposit_tile: array<atomic<i32>, DEPOSIT_TILE_LANES>;
var<workgroup> deposit_corner_x: atomic<i32>;
var<workgroup> deposit_corner_y: atomic<i32>;
var<workgroup> deposit_corner_z: atomic<i32>;

fn deposit_tap(unwrapped: vec3i, corner: vec3i, component: u32, value: f32) {
  let scaled = i32(round(clamp(value, -4096.0, 4096.0) * BRUSH_SCALE));
  if (scaled == 0) {
    return;
  }
  let rel = unwrapped - corner;
  if (all(rel >= vec3i(0)) && all(rel < vec3i(DEPOSIT_TILE_DIM))) {
    let cell = u32((rel.z * DEPOSIT_TILE_DIM + rel.y) * DEPOSIT_TILE_DIM + rel.x);
    atomicAdd(&deposit_tile[cell * 4u + component], scaled);
    return;
  }
  let x = wrap_i(unwrapped.x, config.width);
  let y = wrap_i(unwrapped.y, config.height);
  let z = wrap_i(unwrapped.z, config.depth);
  let flat = (z * config.height + y) * config.width + x;
  atomicAdd(&brush_out[flat * 4u + component], scaled);
}

@compute @workgroup_size(256)
fn deposit_particles(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let i = gid.x;
  let is_active = i < config.active_particle_count;
  if (lid == 0u) {
    atomicStore(&deposit_corner_x, 2147483647);
    atomicStore(&deposit_corner_y, 2147483647);
    atomicStore(&deposit_corner_z, 2147483647);
  }
  for (var lane = lid; lane < DEPOSIT_TILE_LANES; lane = lane + 256u) {
    atomicStore(&deposit_tile[lane], 0);
  }
  workgroupBarrier();
  // Out-of-range tail threads read robustness-clamped garbage; is_active guards every use.
  let particle = particles[i];
  let particle_id = particle_ids[i];
  let sigma2 = max(0.000001, 2.0 * config.sigma * config.sigma);
  let dims = vec3f(f32(config.width), f32(config.height), f32(config.depth));
  let grid = (particle.pos_cohort.xyz + vec3f(1.0)) * 0.5 * dims - vec3f(0.5);
  let base = vec3i(floor(grid));
  let fraction = fract(grid);
  let center = vec3i(round(grid));
  let radius_cells = i32(clamp(config.deposit_tap_radius, 0u, 6u));
  if (is_active) {
    atomicMin(&deposit_corner_x, center.x - radius_cells);
    atomicMin(&deposit_corner_y, center.y - radius_cells);
    atomicMin(&deposit_corner_z, center.z - radius_cells);
  }
  workgroupBarrier();
  let tile_corner = vec3i(
    atomicLoad(&deposit_corner_x),
    atomicLoad(&deposit_corner_y),
    atomicLoad(&deposit_corner_z)
  );
  // Pulse: oscillatory cytoplasmic streaming. Radial traveling waves of deposit intensity emanate
  // through the volume -- the network breathes/throbs instead of just flowing. pulse_depth (0 = off)
  // is amplitude; pulse_rate is temporal frequency. The phase is stateless (agent position + global
  // frame counter), so this costs only ALU and is hidden behind the deposit pass's per-particle
  // memory latency -- effectively free on this I/O-bound pass.
  let pulse_phase = length(particle.pos_cohort.xyz) * 9.0 - f32(config.timestep) * config.pulse_rate * 0.08;
  let pulse = max(0.0, 1.0 + config.pulse_depth * sin(pulse_phase));
  let emission = config.deposit_mass * deposit_scale(config) * pulse;
  let voxel_size = 2.0 / max(1.0, f32(config.width));
  let anti_alias = 1.0 - smoothstep_range(0.45, 1.2, config.deposit_radius / voxel_size);
  var flow_velocity = particle.vel_id.xyz;
  if (length(flow_velocity) < 0.000001) {
    flow_velocity = safe_direction(flow_velocity, f32(config.seed ^ particle_id)) * 0.00001;
  }
  if (is_active) {
    for (var dz = -radius_cells; dz <= radius_cells; dz = dz + 1i) {
      for (var dy = -radius_cells; dy <= radius_cells; dy = dy + 1i) {
        for (var dx = -radius_cells; dx <= radius_cells; dx = dx + 1i) {
          let unwrapped = center + vec3i(dx, dy, dz);
          let x = wrap_i(unwrapped.x, config.width);
          let y = wrap_i(unwrapped.y, config.height);
          let z = wrap_i(unwrapped.z, config.depth);
          let voxel = grid_to_world(x, y, z, config);
          let delta = vec3f(
            periodic_delta(voxel.x, particle.pos_cohort.x),
            periodic_delta(voxel.y, particle.pos_cohort.y),
            periodic_delta(voxel.z, particle.pos_cohort.z)
          );
          let d2 = dot(delta, delta);
          let gaussian_kernel = exp(-d2 / sigma2);
          let cell_offset = unwrapped - base;
          let linear_kernel =
            cell_weight(cell_offset.x, fraction.x) *
            cell_weight(cell_offset.y, fraction.y) *
            cell_weight(cell_offset.z, fraction.z);
          let shape = mix(gaussian_kernel, linear_kernel, anti_alias);
          let kernel = emission * shape;
          if (kernel > 0.0000005) {
            let flow = flow_velocity * kernel * FLOW_DEPOSIT_GAIN;
            deposit_tap(unwrapped, tile_corner, 0u, flow.x);
            deposit_tap(unwrapped, tile_corner, 1u, flow.y);
            deposit_tap(unwrapped, tile_corner, 2u, flow.z);
            deposit_tap(unwrapped, tile_corner, 3u, kernel * 0.01);
          }
        }
      }
    }
  }
  workgroupBarrier();
  // Flush the shared tile: one global atomic per touched lane for the whole workgroup.
  for (var lane = lid; lane < DEPOSIT_TILE_LANES; lane = lane + 256u) {
    let value = atomicLoad(&deposit_tile[lane]);
    if (value != 0) {
      let cell = lane / 4u;
      let component = lane % 4u;
      let rel = vec3i(
        i32(cell % 8u),
        i32((cell / 8u) % 8u),
        i32(cell / 64u)
      );
      let unwrapped = tile_corner + rel;
      let x = wrap_i(unwrapped.x, config.width);
      let y = wrap_i(unwrapped.y, config.height);
      let z = wrap_i(unwrapped.z, config.depth);
      let flat = (z * config.height + y) * config.width + x;
      atomicAdd(&brush_out[flat * 4u + component], value);
    }
  }
}
`;

const liveComputeVisualDepositShader = liveComputeDepositShader
  .replace(
    "@group(0) @binding(2) var<uniform> config: SimConfig;",
    `@group(0) @binding(2) var<uniform> config: SimConfig;
@group(0) @binding(3) var<storage, read> previous_particles: array<Particle>;

const VISUAL_INTERP_SNAP_DISTANCE = 0.35;

fn visual_shortest_delta(previous: vec3f, current: vec3f) -> vec3f {
  var delta = current - previous;
  if (delta.x > 1.0) {
    delta.x = delta.x - 2.0;
  } else if (delta.x < -1.0) {
    delta.x = delta.x + 2.0;
  }
  if (delta.y > 1.0) {
    delta.y = delta.y - 2.0;
  } else if (delta.y < -1.0) {
    delta.y = delta.y + 2.0;
  }
  if (delta.z > 1.0) {
    delta.z = delta.z - 2.0;
  } else if (delta.z < -1.0) {
    delta.z = delta.z + 2.0;
  }
  return delta;
}

fn visual_wrap_crossed(previous: vec3f, current: vec3f) -> bool {
  let raw_delta = current - previous;
  return abs(raw_delta.x) > 1.0 || abs(raw_delta.y) > 1.0 || abs(raw_delta.z) > 1.0;
}

fn visual_lerp_amount(previous: vec3f, current: vec3f, requested_t: f32) -> f32 {
  let t = clamp(requested_t, 0.0, 1.0);
  if (visual_wrap_crossed(previous, current)) {
    return 1.0;
  }
  if (length(visual_shortest_delta(previous, current)) > VISUAL_INTERP_SNAP_DISTANCE) {
    return 1.0;
  }
  return t;
}`
  )
  .replace(
    "fn deposit_particles(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32)",
    "fn deposit_visual_particles(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32)"
  )
  .replace(
    "let particle = particles[i];",
    `var particle = particles[i];
  let previous_particle = previous_particles[i];
  let visual_t = visual_lerp_amount(previous_particle.pos_cohort.xyz, particle.pos_cohort.xyz, config.visual_lerp_t);
  particle.pos_cohort = vec4f(wrap_domain_vec(previous_particle.pos_cohort.xyz + visual_shortest_delta(previous_particle.pos_cohort.xyz, particle.pos_cohort.xyz) * visual_t), particle.pos_cohort.w);
  particle.vel_id = vec4f(mix(previous_particle.vel_id.xyz, particle.vel_id.xyz, visual_t), particle.vel_id.w);`
  );

// Two sample_field implementations with IDENTICAL math: the classic 8-load manual trilinear
// over the f32 storage buffer, and the same 8 taps via textureLoad on the rgba32float mirror
// texture (kept in sync by the field-update pass). Same f32 data, same f32 weights - bit-exact
// results; the texture path just wins on 3D-swizzled cache locality for the clustered taps.
const makeLiveComputeParticleShader = (useFieldTexture: boolean): string => /* wgsl */ `
${commonComputeShader}

@group(0) @binding(0) var<storage, read> particles_in: array<Particle>;
${useFieldTexture
    ? `@group(0) @binding(1) var field_texture_in: texture_3d<f32>;`
    : `@group(0) @binding(1) var<storage, read> field_in: array<vec4f>;`}
@group(0) @binding(2) var<storage, read_write> particles_out: array<Particle>;
@group(0) @binding(3) var<uniform> config: SimConfig;
@group(0) @binding(4) var<storage, read> config_rule: array<vec4f>;
@group(0) @binding(5) var<storage, read> particle_ids: array<u32>;

${useFieldTexture
    ? `fn sample_field(position_in: vec3f) -> vec4f {
  let position = wrap_domain_vec(position_in);
  let dims = vec3f(f32(config.width), f32(config.height), f32(config.depth));
  let grid = (position + vec3f(1.0)) * 0.5 * dims - vec3f(0.5);
  let base = vec3i(floor(grid));
  let t = fract(grid);
  var accum = vec4f(0.0);
  for (var dz = 0i; dz <= 1i; dz = dz + 1i) {
    for (var dy = 0i; dy <= 1i; dy = dy + 1i) {
      for (var dx = 0i; dx <= 1i; dx = dx + 1i) {
        let weight = mix(1.0 - t.x, t.x, f32(dx)) * mix(1.0 - t.y, t.y, f32(dy)) * mix(1.0 - t.z, t.z, f32(dz));
        let coords = vec3i(
          i32(wrap_i(base.x + dx, config.width)),
          i32(wrap_i(base.y + dy, config.height)),
          i32(wrap_i(base.z + dz, config.depth))
        );
        accum = accum + textureLoad(field_texture_in, coords, 0) * weight;
      }
    }
  }
  return accum;
}`
    : `fn sample_field(position_in: vec3f) -> vec4f {
  let position = wrap_domain_vec(position_in);
  let dims = vec3f(f32(config.width), f32(config.height), f32(config.depth));
  let grid = (position + vec3f(1.0)) * 0.5 * dims - vec3f(0.5);
  let base = vec3i(floor(grid));
  let t = fract(grid);
  var accum = vec4f(0.0);
  for (var dz = 0i; dz <= 1i; dz = dz + 1i) {
    for (var dy = 0i; dy <= 1i; dy = dy + 1i) {
      for (var dx = 0i; dx <= 1i; dx = dx + 1i) {
        let weight = mix(1.0 - t.x, t.x, f32(dx)) * mix(1.0 - t.y, t.y, f32(dy)) * mix(1.0 - t.z, t.z, f32(dz));
        accum = accum + field_in[index_xyz(base.x + dx, base.y + dy, base.z + dz, config)] * weight;
      }
    }
  }
  return accum;
}`}

fn load_config_rule() -> Rule {
  var rule: Rule;
  for (var i = 0u; i < 10u; i = i + 1u) {
    rule.centers[i].frequency = config_rule[i * 2u];
    rule.centers[i].amplitude = config_rule[i * 2u + 1u];
  }
  return rule;
}

@compute @workgroup_size(256)
fn update_particles(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= config.active_particle_count) {
    return;
  }
  let particle_id = particle_ids[i];
  var particle = particles_in[i];
  // Initial-condition morph: instead of a hard reseed, ease every particle toward its spawn target
  // for the new pattern (reset_particle is a pure function of id + config), skipping physics so the
  // layout transition reads as a smooth migration rather than a flash.
  if (config.morph_alpha > 0.0) {
    let morph_target = reset_particle(particle_id, config);
    particles_out[i].pos_cohort = vec4f(mix(particle.pos_cohort.xyz, morph_target.pos_cohort.xyz, config.morph_alpha), particle.pos_cohort.w);
    particles_out[i].vel_id = vec4f(mix(particle.vel_id.xyz, morph_target.vel_id.xyz, config.morph_alpha), particle.vel_id.w);
    return;
  }
  let time_scale = sim_time_scale(config);
  let hazard = config.hazard_rate * time_scale > hash(vec2f(f32(particle_id) / max(1.0, f32(config.particle_count)), f32(config.timestep)));
  if (hazard) {
    particles_out[i] = reset_particle(particle_id, config);
    return;
  }
  // Recycle: a particle that has drifted into a sub-cutoff (empty) region is reset back into the
  // spawn distribution instead of wandering the fringe -- it rejoins the active core and
  // concentrates density there. Checked before the expensive sensing so it also saves that work.
  if (config.recycle_enabled == 1u && config.recycle_cutoff > 0.0) {
    let here = sample_field(particle.pos_cohort.xyz);
    if (max(0.0, here.w + length(here.xyz) * 0.25) < config.recycle_cutoff) {
      particles_out[i] = reset_particle(particle_id, config);
      return;
    }
  }
  // Shared center-voxel sample for every field-reading emergent behavior (energy, the steering block,
  // and the gradient base). Computed at most once and only when something needs it, so the off-path
  // adds no field sample and stays byte-identical. want_field/want_grad are reused below.
  let want_field = config.mips > 0.0 || config.aniso_follow > 0.0 || config.flock_align > 0.0 || config.quorum_strength > 0.0;
  let want_grad = config.flock_separate > 0.0 || abs(config.chemotaxis) > 0.0;
  let need_center = config.energy > 0.0 || want_field || want_grad;
  var center_field = vec4f(0.0);
  if (need_center) {
    center_field = sample_field(particle.pos_cohort.xyz);
  }
  // Energy / metabolism: stateless density-gated lifecycle. A particle "feeds" on the local trail
  // density; in starved (sparse) regions it dies and respawns into the seed distribution, so the
  // swarm concentrates on its food, depletes the fringe, and pulses through boom/bust cycles. The
  // death roll is hashed from (id, timestep) so replays stay bit-exact, and the whole branch is
  // skipped when energy == 0 (byte-identical to the classic engine).
  if (config.energy > 0.0) {
    let fed = center_field.w / (center_field.w + DENSITY_HALF);
    let starve = (1.0 - fed) * config.energy * config.energy_drain;
    if (hash(vec2f(f32(particle_id), f32(config.timestep) * 1.3 + 0.5)) < starve * 0.05 * time_scale) {
      particles_out[i] = reset_particle(particle_id, config);
      return;
    }
  }
  let cohort = floor(particle.pos_cohort.w);
  var rule = load_config_rule();
  if (all(rule.centers[0].frequency == vec4f(0.0)) && all(rule.centers[5].amplitude == vec4f(0.0))) {
    rule = generate_random_centers(config.rule_seed + cohort);
  }
  rule = mutate_rule(rule, config.mutation_scale * time_scale, config.rule_seed + cohort);
  let frame = build_frame(orientation_axis(particle.pos_cohort.xyz, particle.vel_id.xyz, config));
  let forward = frame[0];
  let left = frame[1];
  let up = frame[2];
  let angle = config.sensor_angle * PI;
  let sample_dist = config.sensor_distance * 0.01;
  let horizontal_left_offset = rotate_sensor(forward, left, angle, sample_dist);
  let horizontal_right_offset = rotate_sensor(forward, left, -angle, sample_dist);
  let vertical_up_offset = rotate_sensor(forward, up, angle, sample_dist);
  let vertical_down_offset = rotate_sensor(forward, up, -angle, sample_dist);
  let sensor_scaling = 19.4275 * config.sensor_gain;
  let ltap = sample_field(particle.pos_cohort.xyz + horizontal_left_offset).xyz * sensor_scaling;
  let rtap = sample_field(particle.pos_cohort.xyz + horizontal_right_offset).xyz * sensor_scaling;
  let utap = sample_field(particle.pos_cohort.xyz + vertical_up_offset).xyz * sensor_scaling;
  let dtap = sample_field(particle.pos_cohort.xyz + vertical_down_offset).xyz * sensor_scaling;
  let horizontal = calculate_plane_behavior(ltap, rtap, forward, left, rule, config);
  let vertical = calculate_plane_behavior(utap, dtap, forward, up, rule, config);
  let force = (horizontal.force + vertical.force) * 0.5 * config.global_force_mult / 200.0;
  let strafe = (horizontal.strafe + vertical.strafe) * 0.5 * config.global_force_mult / 10.0;
  let color_params = (horizontal.color + vertical.color) * 0.5;
  var hue = config.hue_sensitivity * color_params.x;
  if (config.color_by_cohort == 1u) {
    hue = hash(vec2f(floor(cohort), floor(cohort)));
  }
  let hue_blend = clamp(time_scale * 0.08, 0.0, 1.0);
  var new_hue = mix(particle.vel_id.w, hue, hue_blend);
  let drag_factor = velocity_drag_factor(config.drag, time_scale);
  let force_factor = velocity_force_factor(config.drag, drag_factor, time_scale);
  var velocity = particle.vel_id.xyz * drag_factor + force * force_factor;
  // Strafe momentum blend (config.strafe_momentum, 0..1). The sensing strafe steers sideways. Applied raw
  // to position (the classic path, momentum=0) it has no inertia; folded into velocity (momentum=1) it gains
  // inertia + drag damping and avoids the low-speed per-step position 2-cycle.
  velocity = velocity + strafe * config.strafe_power * STRAFE_VELOCITY_GAIN * config.strafe_momentum;
  velocity = guard_velocity_reversal(particle.vel_id.xyz, velocity);
  if (length(velocity) < 0.00001) {
    velocity = forward * 0.00001;
  }
  // Stall rescue: structures that lock into a self-reinforcing field pattern freeze with
  // near-zero velocity (stuck starfish arms, static bipolar oscillators). When restlessness
  // is on, particles below the stall speed get a small deterministic pseudorandom kick
  // (hashed from persistent id + timestep, so timeline replays stay bit-exact) scaled by how
  // stalled they are; the random-walk jitter persists until field sensing re-engages them.
  // Moving particles are untouched and 0 disables the branch entirely.
  if (config.restlessness > 0.0) {
    let stall_speed = 0.008;
    let speed_now = length(velocity);
    if (speed_now < stall_speed) {
      let deficit = 1.0 - speed_now / stall_speed;
      let kick_seed = f32(particle_id % 1048576u) * 0.6180339887 + f32(config.timestep % 16384u) * 7.5625 + f32(config.seed % 65536u) * 0.001;
      let kick_theta = hash(vec2f(kick_seed, 23.0)) * PI * 2.0;
      let kick_z = hash(vec2f(kick_seed, 29.0)) * 2.0 - 1.0;
      let kick_r = sqrt(max(0.0, 1.0 - kick_z * kick_z));
      let kick_dir = vec3f(cos(kick_theta) * kick_r, sin(kick_theta) * kick_r, kick_z);
      velocity = velocity + kick_dir * (config.restlessness * 0.004 * deficit * time_scale);
    }
  }
  // --- Emergent behaviors: field-driven velocity steering + quorum ignition. Reuses center_field
  // (sampled once above). want_field/want_grad were computed above; when both are false nothing here
  // runs, so velocity/hue/displacement are untouched and the step is byte-identical. ---
  if (want_field || want_grad) {
    let epos = particle.pos_cohort.xyz;
    let local_density = center_field.w;
    let speed0 = length(velocity);
    if (config.mips > 0.0) {
      // Motility-induced phase separation: self-propelled particles slow in crowds. With no
      // attraction at all, this alone makes a dense droplet phase separate from a dilute gas.
      let crowd = local_density / (local_density + DENSITY_HALF);
      velocity = velocity * (1.0 - config.mips * crowd * 0.85);
    }
    if (config.aniso_follow > 0.0) {
      // Follow the deposited current's *direction* (the field already stores velocity). Polarizes
      // trails into one-way highways and self-sustaining vortex rings.
      let current = center_field.xyz;
      let cmag = length(current);
      if (cmag > 0.0001) {
        velocity = velocity + (current / cmag) * speed0 * config.aniso_follow * 0.25;
      }
    }
    if (config.flock_align > 0.0) {
      // Couzin/Vicsek alignment: rotate velocity toward the local mean flow -> schooling/murmuration.
      let flow = center_field.xyz;
      let fmag = length(flow);
      if (fmag > 0.0001 && speed0 > 0.0001) {
        velocity = mix(velocity, (flow / fmag) * speed0, clamp(config.flock_align * 0.5, 0.0, 1.0));
      }
    }
    if (config.quorum_strength > 0.0) {
      // Quorum sensing / bioluminescent ignition: above a local-density threshold the particle
      // "ignites" (hue flash) and surges along the local stream, concentrating it further so the
      // ignition spreads as a contagious wave.
      let dens_norm = local_density / (local_density + DENSITY_HALF);
      let ignite = smoothstep(config.quorum_threshold, config.quorum_threshold + 0.1, dens_norm);
      new_hue = mix(new_hue, 0.08, clamp(ignite * config.quorum_strength, 0.0, 1.0));
      let cur = center_field.xyz;
      let cmag2 = length(cur);
      if (cmag2 > 0.0001 && ignite > 0.0) {
        velocity = velocity + (cur / cmag2) * speed0 * ignite * config.quorum_strength * 0.4;
      }
    }
    if (want_grad) {
      // Forward-difference gradient: reuse center_field as the base, so 3 extra samples instead of 6.
      let h = 2.0 / max(1.0, f32(config.width));
      let grad = vec3f(
        sample_field(epos + vec3f(h, 0.0, 0.0)).w - local_density,
        sample_field(epos + vec3f(0.0, h, 0.0)).w - local_density,
        sample_field(epos + vec3f(0.0, 0.0, h)).w - local_density
      );
      let gmag = length(grad);
      if (gmag > 0.000001) {
        let gdir = grad / gmag;
        if (config.flock_separate > 0.0) {
          // Steer down the density gradient -> spacing / collision avoidance.
          velocity = velocity - gdir * speed0 * config.flock_separate * 0.3;
        }
        if (abs(config.chemotaxis) > 0.0) {
          // Climb (+) or descend (-) the density gradient.
          velocity = velocity + gdir * speed0 * config.chemotaxis * 0.3;
        }
      }
    }
  }
  // __ECOLOGY_FORCES__ (replaced with species/predator/alarm steering in the ecology shader variant)
  // Runaway guard: the field-following terms add velocity along the deposited current and can compound
  // when stacked at low drag. Bound the final speed whenever any emergent steering is active (gated so
  // the off-path is untouched / byte-identical). Covers both this base shader and the ecology variant
  // (ecology forces are injected just above at the marker).
  if (want_field || want_grad || config.species_force > 0.0 || config.predator > 0.0 || config.alarm > 0.0) {
    let vmag = length(velocity);
    if (vmag > EMERGENT_VMAX) {
      velocity = velocity * (EMERGENT_VMAX / vmag);
    }
  }
  let displacement = guard_velocity_reversal(
    particle.vel_id.xyz * config.dt,
    velocity * config.dt + strafe * config.strafe_power * config.dt * (1.0 - config.strafe_momentum)
  );
  var position = particle.pos_cohort.xyz + displacement;
  // Anti-oscillation override. Strafe is a per-step sideways slip applied straight to position along the
  // orientation axis; for a position-derived orientation (absoluteOrientation = Radial/Outward/Swirl/
  // Noise = >= 2) a large negative axialForce can make the slip leap the particle clear across the
  // origin in one step, flipping that axis so the slip flips with it -- a degenerate per-step position
  // 2-cycle (the "switching between two configurations" jitter; the renderer derives speed/color/stretch
  // from per-frame displacement, so these read as big/fast/red). Velocity-aligned orientations (0/1)
  // cannot flip that way, so this is gated off for them and they stay byte-identical. On the gated path,
  // a slip larger than the cell budget is super-CFL -- it outruns the field structure it senses and is
  // the runaway that drives the cycle -- so redo the step without it; the particle coasts on its
  // velocity and the cloud gracefully relaxes to a drift instead of flickering. Dropping it every step
  // (not every other step) is what keeps it calm rather than trading one oscillation for another.
  if (config.absolute_orientation >= 2u) {
    // Derive the slip from the already-computed displacement so this guard sees the same raw position slip
    // that will be applied this step.
    let slip = displacement - velocity * config.dt;
    let strafe_cell = DOMAIN_SIZE / f32(max(1u, min(min(config.width, config.height), config.depth)));
    if (length(slip) > STRAFE_MAX_CELLS * strafe_cell) {
      position = particle.pos_cohort.xyz + guard_velocity_reversal(
        particle.vel_id.xyz * config.dt,
        velocity * config.dt
      );
    }
  }
  if (config.domain_shape == 0u) {
  if (config.boundary_mode == 0u) {
    if (position.x < -1.0 || position.x > 1.0) {
      velocity.x = -velocity.x;
      position.x = clamp(position.x, -1.0, 1.0);
    }
    if (position.y < -1.0 || position.y > 1.0) {
      velocity.y = -velocity.y;
      position.y = clamp(position.y, -1.0, 1.0);
    }
    if (position.z < -1.0 || position.z > 1.0) {
      velocity.z = -velocity.z;
      position.z = clamp(position.z, -1.0, 1.0);
    }
  } else if (config.boundary_mode == 1u) {
    if (any(position < vec3f(-1.0)) || any(position > vec3f(1.0))) {
      particles_out[i] = reset_particle(particle_id, config);
      return;
    }
  } else if (config.boundary_mode == 3u) {
    // Per-axis: wrap horizontally (X/Y), bounce off floor/ceiling (Z) -> endless tube.
    position.x = wrap_domain_scalar(position.x);
    position.y = wrap_domain_scalar(position.y);
    if (position.z < -1.0 || position.z > 1.0) {
      velocity.z = -velocity.z;
      position.z = clamp(position.z, -1.0, 1.0);
    }
  } else if (config.boundary_mode == 4u) {
    // Sticky/absorb: particles that reach the wall freeze onto it (a crust forms).
    if (any(position < vec3f(-1.0)) || any(position > vec3f(1.0))) {
      position = clamp(position, vec3f(-1.0), vec3f(1.0));
      velocity = vec3f(0.0);
    }
  } else if (config.boundary_mode == 5u) {
    // Soft wall: a spring force cushions particles back inside past 0.95 (rounded dome).
    let over = max(vec3f(0.0), abs(position) - vec3f(0.95));
    velocity = velocity - sign(position) * over * 0.5;
    position = clamp(position, vec3f(-1.0), vec3f(1.0));
  } else if (config.boundary_mode == 6u) {
    // Portal: leaving one side re-injects on the opposite side with a lateral shear (wormhole).
    if (any(position < vec3f(-1.0)) || any(position > vec3f(1.0))) {
      position = wrap_domain_vec(position + vec3f(0.37, -0.19, 0.0));
    }
  } else {
    position = wrap_domain_vec(position);
  }
  } else {
    // Non-cube domain: confine to the shape via its SDF, honoring the boundary mode.
    let d = domain_sdf(position, config.domain_shape);
    if (d > 0.0) {
      if (config.boundary_mode == 1u) {
        particles_out[i] = reset_particle(particle_id, config);
        return;
      }
      let n = domain_sdf_normal(position, config.domain_shape);
      if (config.boundary_mode == 0u) {
        velocity = reflect(velocity, n);
        position = position - n * (d + 0.0005);
      } else if (config.boundary_mode == 4u) {
        position = position - n * (d + 0.0005);
        velocity = vec3f(0.0);
      } else if (config.boundary_mode == 5u) {
        velocity = velocity - n * d * 0.5;
        position = position - n * (d + 0.0005);
      } else {
        // Wrap/portal/tube have no clean analogue on a curved domain — fall back to a bounce so
        // the cloud is actually contained by (and takes the form of) the selected shape.
        velocity = reflect(velocity, n);
        position = position - n * (d + 0.0005);
      }
    }
  }
  particles_out[i].pos_cohort = vec4f(position, cohort);
  particles_out[i].vel_id = vec4f(velocity, new_hue);
}
`;

// --- Ecology field (multi-species pheromone volume) -------------------------------------------------
// A separate rgba volume: channels x/y/z = density of species group 0/1/2 (cohort %% 3); channel w =
// the predator "scent" prey flee. It powers three behaviors (speciesForce, predator, alarm). The whole
// subsystem (buffers + passes + the sensing shader variant below) only runs when one of those is > 0,
// so the default path is untouched and byte-identical. ECO_SCALE matches the main brush fixed point.
const ECO_BRUSH_SCALE = 65536.0;

const liveComputeEcologyDepositShader = /* wgsl */ `
${commonComputeShader}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> eco_brush: array<atomic<i32>>;
@group(0) @binding(2) var<uniform> config: SimConfig;

@compute @workgroup_size(256)
fn ecology_deposit(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= config.active_particle_count) {
    return;
  }
  let particle = particles[i];
  let dims = vec3f(f32(config.width), f32(config.height), f32(config.depth));
  let grid = (particle.pos_cohort.xyz + vec3f(1.0)) * 0.5 * dims - vec3f(0.5);
  let center = vec3i(round(grid));
  let idx = index_xyz(center.x, center.y, center.z, config);
  let species = u32(floor(particle.pos_cohort.w)) % 3u;
  let amount = i32(config.deposit_mass * 0.02 * ${ECO_BRUSH_SCALE});
  atomicAdd(&eco_brush[idx * 4u + species], amount);
  // Species 0 is the predator group: it also lays a scent trail into channel w that prey flee.
  if (species == 0u) {
    atomicAdd(&eco_brush[idx * 4u + 3u], amount);
  }
}
`;

const liveComputeEcologyUpdateShader = /* wgsl */ `
${commonComputeShader}

@group(0) @binding(0) var<storage, read> eco_brush: array<i32>;
@group(0) @binding(1) var<storage, read_write> eco_field: array<vec4f>;
@group(0) @binding(2) var<uniform> config: SimConfig;

@compute @workgroup_size(256)
fn ecology_update(@builtin(global_invocation_id) gid: vec3u) {
  let flat = gid.x;
  let voxel_count = config.width * config.height * config.depth;
  if (flat >= voxel_count) {
    return;
  }
  // Clamp the fixed-point brush to a sane non-negative range before folding in: a pathological
  // pile-up of tens of thousands of particles into one voxel could otherwise overflow the i32 atomic
  // and wrap negative. clamp(>=0) absorbs that without affecting normal densities.
  let brush = clamp(vec4f(
    f32(eco_brush[flat * 4u + 0u]) / ${ECO_BRUSH_SCALE},
    f32(eco_brush[flat * 4u + 1u]) / ${ECO_BRUSH_SCALE},
    f32(eco_brush[flat * 4u + 2u]) / ${ECO_BRUSH_SCALE},
    f32(eco_brush[flat * 4u + 3u]) / ${ECO_BRUSH_SCALE}
  ), vec4f(0.0), vec4f(64.0));
  let decay = 0.9;
  eco_field[flat] = min(eco_field[flat] * decay + brush, vec4f(16.0));
}
`;

// Ecology-sensing variant of update_particles: same physics as the buffer-sensing base shader, plus a
// read-only ecology field at binding 6 and the species/predator/alarm steering injected at the marker.
const makeLiveComputeParticleEcologyShader = (useFieldTexture: boolean): string =>
  makeLiveComputeParticleShader(useFieldTexture)
    .replace(
      "@group(0) @binding(5) var<storage, read> particle_ids: array<u32>;",
      `@group(0) @binding(5) var<storage, read> particle_ids: array<u32>;
@group(0) @binding(6) var<storage, read> eco_field: array<vec4f>;

fn sample_ecology(position_in: vec3f) -> vec4f {
  let position = wrap_domain_vec(position_in);
  let dims = vec3f(f32(config.width), f32(config.height), f32(config.depth));
  let grid = (position + vec3f(1.0)) * 0.5 * dims - vec3f(0.5);
  let c = vec3i(round(grid));
  return eco_field[index_xyz(c.x, c.y, c.z, config)];
}

// Asymmetric inter-species affinity matrix derived from the rule seed: positive = attract species j,
// negative = repel. Asymmetry (M[i][j] != M[j][i]) is what produces chasing/orbiting particle-life cells.
fn species_affinity(i: u32, j: u32, seed: f32) -> f32 {
  return hash(vec2f(f32(i * 3u + j) + 0.5, seed * 97.13 + 3.0)) * 2.0 - 1.0;
}`
    )
    .replace(
      "  // __ECOLOGY_FORCES__ (replaced with species/predator/alarm steering in the ecology shader variant)",
      `  if (config.species_force > 0.0 || config.predator > 0.0 || config.alarm > 0.0) {
    let epos2 = particle.pos_cohort.xyz;
    let myspec = u32(floor(particle.pos_cohort.w)) % 3u;
    let h2 = 2.0 / max(1.0, f32(config.width));
    let exp_p = sample_ecology(epos2 + vec3f(h2, 0.0, 0.0));
    let exp_m = sample_ecology(epos2 - vec3f(h2, 0.0, 0.0));
    let eyp = sample_ecology(epos2 + vec3f(0.0, h2, 0.0));
    let eym = sample_ecology(epos2 - vec3f(0.0, h2, 0.0));
    let ezp = sample_ecology(epos2 + vec3f(0.0, 0.0, h2));
    let ezm = sample_ecology(epos2 - vec3f(0.0, 0.0, h2));
    let grad0 = vec3f(exp_p.x - exp_m.x, eyp.x - eym.x, ezp.x - ezm.x);
    let grad1 = vec3f(exp_p.y - exp_m.y, eyp.y - eym.y, ezp.y - ezm.y);
    let grad2 = vec3f(exp_p.z - exp_m.z, eyp.z - eym.z, ezp.z - ezm.z);
    let grad_scent = vec3f(exp_p.w - exp_m.w, eyp.w - eym.w, ezp.w - ezm.w);
    let eco_speed = length(velocity);
    if (config.species_force > 0.0) {
      let f = grad0 * species_affinity(myspec, 0u, config.rule_seed)
        + grad1 * species_affinity(myspec, 1u, config.rule_seed)
        + grad2 * species_affinity(myspec, 2u, config.rule_seed);
      let fl = length(f);
      if (fl > 0.00000001) { velocity = velocity + (f / fl) * eco_speed * config.species_force * 0.3; }
    }
    if (config.predator > 0.0 && myspec == 0u) {
      let prey = grad1 + grad2;
      let pl = length(prey);
      if (pl > 0.00000001) { velocity = velocity + (prey / pl) * eco_speed * config.predator * 0.4; }
    }
    if (config.alarm > 0.0 && myspec != 0u) {
      let al = length(grad_scent);
      if (al > 0.00000001) { velocity = velocity - (grad_scent / al) * eco_speed * config.alarm * 0.4; }
    }
  }`
    );

// writeTexture=true mirrors field_out into the rgba16float texture each step so the next
// step's particle update can sense via hardware trilinear filtering. The visual-field
// smoothing pass uses the writeTexture=false variant - it operates on the render-side visual
// buffers and must never clobber the sensing texture.
const makeLiveComputeFieldShader = (writeTexture: boolean, fieldTextureStorageFormat = "rgba32float"): string => /* wgsl */ `
${commonComputeShader}

const BRUSH_SCALE = 65536.0;

@group(0) @binding(0) var<storage, read> field_in: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> brush: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> field_out: array<vec4f>;
@group(0) @binding(3) var<uniform> config: SimConfig;
${writeTexture ? `@group(0) @binding(4) var field_texture_out: texture_storage_3d<${fieldTextureStorageFormat}, write>;` : ""}

@compute @workgroup_size(256)
fn update_field(@builtin(global_invocation_id) gid: vec3u) {
  let flat = gid.x;
  let voxel_count = config.width * config.height * config.depth;
  if (flat >= voxel_count) {
    return;
  }
  let x = flat % config.width;
  let y = (flat / config.width) % config.height;
  let z = flat / (config.width * config.height);
  let center = field_in[flat];
  var canvas_color = center;
  let time_scale = sim_time_scale(config);
  let diffusion = clamp(config.trail_diffusion, 0.0, 1.0);
  if (diffusion > 0.0) {
    let k_slider = max(0.001, diffusion * diffusion);
    let k = 6.0 / (pow(5.0, k_slider) - 1.0);
    let neighbors =
      field_in[index_xyz(i32(x) - 1, i32(y), i32(z), config)] +
      field_in[index_xyz(i32(x) + 1, i32(y), i32(z), config)] +
      field_in[index_xyz(i32(x), i32(y) - 1, i32(z), config)] +
      field_in[index_xyz(i32(x), i32(y) + 1, i32(z), config)] +
      field_in[index_xyz(i32(x), i32(y), i32(z) - 1, config)] +
      field_in[index_xyz(i32(x), i32(y), i32(z) + 1, config)];
    let diffused_color = (center * k + neighbors) / (6.0 + k);
    canvas_color = mix(center, diffused_color, time_scale);
  }
  let persistence = pow(clamp(config.trail_persistence, 0.0, 0.999), time_scale);
  let brush_value = vec4f(
    f32(atomicLoad(&brush[flat * 4u + 0u])) / BRUSH_SCALE,
    f32(atomicLoad(&brush[flat * 4u + 1u])) / BRUSH_SCALE,
    f32(atomicLoad(&brush[flat * 4u + 2u])) / BRUSH_SCALE,
    f32(atomicLoad(&brush[flat * 4u + 3u])) / BRUSH_SCALE
  );
  let mixed = canvas_color * persistence + brush_value * (1.0 - persistence);
  var w_final = mixed.w;
  // Lenia-style continuous-CA growth on the density channel. A smooth growth function of the local
  // neighbourhood mean grows the field where density sits near the kernel centre and erodes it
  // elsewhere -> self-organising blobs/gliders the particles ride. 0 strength = no extra reads,
  // w_final == mixed.w, byte-identical to the classic field update.
  // Lenia growth + single-channel reaction-diffusion both read the 6-neighbour mean of the density
  // channel; compute it once when either is active. Both are 0-gated -> byte-identical when off.
  if (config.lenia_strength > 0.0 || config.gray_scott > 0.0) {
    let n6 =
      field_in[index_xyz(i32(x) - 1, i32(y), i32(z), config)] +
      field_in[index_xyz(i32(x) + 1, i32(y), i32(z), config)] +
      field_in[index_xyz(i32(x), i32(y) - 1, i32(z), config)] +
      field_in[index_xyz(i32(x), i32(y) + 1, i32(z), config)] +
      field_in[index_xyz(i32(x), i32(y), i32(z) - 1, config)] +
      field_in[index_xyz(i32(x), i32(y), i32(z) + 1, config)];
    let local_mean = (center.w + n6.w) / 7.0;
    // Normalize the small (~1e-4) density to 0..1 so the slider params are meaningful ranges.
    let mean_norm = local_mean / (local_mean + DENSITY_HALF);
    let u_norm = w_final / (w_final + DENSITY_HALF);
    if (config.lenia_strength > 0.0) {
      // Lenia: smooth growth function of the neighbourhood mean -> self-organising blobs/gliders.
      let sigma = max(0.001, config.lenia_width);
      let delta_u = mean_norm - config.lenia_center;
      let growth = 2.0 * exp(-(delta_u * delta_u) / (2.0 * sigma * sigma)) - 1.0;
      w_final = max(0.0, w_final + config.lenia_strength * growth * time_scale * LENIA_RATE);
    }
    if (config.gray_scott > 0.0) {
      // Single-channel reaction-diffusion (Nagumo/Allen-Cahn bistable): diffusion smooths the
      // normalised density while a cubic reaction pushes it toward 0 or 1 about a threshold,
      // forming Turing-like spots and labyrinths. gs_feed scales diffusion, gs_kill sets the
      // threshold. (One-channel approximation of Gray-Scott; no extra field buffer.)
      let lap = mean_norm - u_norm;
      let diff = clamp(config.gs_feed * 10.0, 0.0, 1.0);
      let a = clamp(config.gs_kill * 10.0, 0.05, 0.95);
      let reaction = u_norm * (1.0 - u_norm) * (u_norm - a);
      let du = diff * lap + reaction;
      w_final = max(0.0, w_final + config.gray_scott * du * time_scale * LENIA_RATE * 4.0);
    }
  }
  let result = vec4f(clamp(mixed.xyz, vec3f(-8.0), vec3f(8.0)), min(w_final, 8.0));
  field_out[flat] = result;
  ${writeTexture ? "textureStore(field_texture_out, vec3i(i32(x), i32(y), i32(z)), result);" : ""}
}
`;

const makeLiveComputeFieldTextureShader = (fieldTextureStorageFormat: string): string => /* wgsl */ `
${commonComputeShader}

@group(0) @binding(0) var<storage, read> field_values: array<vec4f>;
@group(0) @binding(1) var field_texture: texture_storage_3d<${fieldTextureStorageFormat}, write>;
@group(0) @binding(2) var<uniform> config: SimConfig;

@compute @workgroup_size(4, 4, 4)
fn update_field_texture(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= config.width || gid.y >= config.height || gid.z >= config.depth) {
    return;
  }
  let flat = (gid.z * config.height + gid.y) * config.width + gid.x;
  textureStore(field_texture, vec3i(gid), field_values[flat]);
}
`;

const liveSplatRenderCommon = /* wgsl */ `
const PI = 3.141592653589793;

// View depth (along the camera axis) at which a splat sits exactly on the near plane.
// Anything at or in front of the camera by less than this is behind the viewer and culled.
const PARTICLE_NEAR = 0.05;
// Soft density cutoff: instead of hard-culling particles below particle_density_cutoff, the
// splat scales down to zero size as the neighborhood density falls from this multiple of the
// cutoff down to the cutoff itself, so the cloud edge dissolves instead of popping.
const PARTICLE_CUTOFF_FADE = 2.25;
// If a particle jumps farther than this between two whole-step states, treat it as a reset or
// extreme aliasing event and snap to the current state instead of drawing an interpolation streak.
const PARTICLE_INTERP_SNAP_DISTANCE = 0.35;
// Legacy focal (1.85*0.92). The FOV slider feeds uniforms.focal_k; at this value the
// projection is identical to the verified default.
const DEFAULT_FOCAL_K = 1.702;
// Reference view depth at which particle_size_px renders at its literal pixel value. This is
// the default camera distance, so at default framing sizes match the slider; nearer particles
// grow and farther ones shrink with true perspective.
const PARTICLE_SIZE_REF_DEPTH = 3.15;

struct Particle {
  pos_cohort: vec4f,  // xyz = position, w = cohort
  vel_id: vec4f,      // xyz = velocity, w = hue (was particle id; id is derivable from index)
};

struct RenderUniforms {
  resolution: vec2f,
  width: u32,
  height: u32,
  depth: u32,
  particle_count: u32,
  voxel_count: u32,
  timestep: u32,
  density: f32,
  exposure: f32,
  focus_distance: f32,
  aperture: f32,
  overlay: u32,
  palette: u32,
  filament: f32,
  yaw_cos: f32,
  yaw_sin: f32,
  pitch_cos: f32,
  pitch_sin: f32,
  distance: f32,
  ray_steps: u32,
  fog_step_scale: f32,
  fog_temporal_blend: f32,
  fog_blue_noise: u32,
  fog_frame: u32,
  field_texture_mode: u32,
  empty_space_skip: u32,
  empty_space_threshold: f32,
  empty_space_stride: f32,
  particle_size_px: f32,
  particle_min_px: f32,
  particle_max_px: f32,
  particle_opacity: f32,
  particle_blend_mode: u32,
  particle_density_cutoff: f32,
  particle_density_radius: f32,
  trail_opacity: f32,
  trail_threshold: f32,
  render_layer: u32,
  camera_pan_x: f32,
  camera_pan_y: f32,
  trail_color_mode: u32,
  fog_tint_r: f32,
  fog_tint_g: f32,
  fog_tint_b: f32,
  particle_tint_r: f32,
  particle_tint_g: f32,
  particle_tint_b: f32,
  scene_brightness: f32,
  particle_brightness: f32,
  fog_brightness: f32,
  particle_color_mode: u32,
  particle_velocity_stretch: u32,
  particle_stretch: f32,
  particle_gradient_sensitivity: f32,
  focal_k: f32,
  dof_blur: f32,
  dof_debug: u32,
  render_lerp_t: f32,
  dof_enabled: u32,
  density_pass_strength: f32,
  density_small_scale: f32,
  density_large_scale: f32,
  density_large_threshold: f32,
  density_contrast_gain: f32,
  density_contrast_balance: f32,
  density_emission_power: f32,
  density_occlusion: f32,
  cohorts: u32,
  particle_slow_cutoff: f32,
  particle_glow_core: f32,
  particle_hot_core: f32,
  accumulation_strength: f32,
  accumulation_radius: f32,
  accumulation_curve: f32,
  accumulation_memory: f32,
  accumulation_noise_reject: f32,
  particle_density_reference: f32,
  particle_density_normalize: f32,
  particle_density_softness: f32,
  particle_support_mask: f32,
  particle_support_radius: f32,
  particle_support_neighbors: f32,
  particle_support_flow: f32,
  particle_support_grid_size: u32,
  particle_stretch_min: f32,
  particle_stretch_speed: f32,
  particle_speed_cutoff: f32,
  variation_master: f32,
  variation_time: f32,
  variation_drift: f32,
  variation_noise_mix: f32,
  variation_freq: f32,
  variation_octaves: u32,
  variation_gain: f32,
  variation_lacunarity: f32,
  variation_size_amount: f32,
  variation_size_curve: f32,
  variation_size_min: f32,
  variation_size_max: f32,
  variation_bright_amount: f32,
  variation_bright_curve: f32,
  variation_bright_min: f32,
  variation_bright_max: f32,
  variation_opacity_amount: f32,
  variation_opacity_curve: f32,
  variation_opacity_min: f32,
  variation_opacity_max: f32,
  variation_color_amount: f32,
  variation_color_curve: f32,
  variation_color_min: f32,
  variation_color_max: f32,
  domain_shape: u32,
  audio_low: f32,
  audio_mid: f32,
  audio_high: f32,
  particle_color_mode_from: u32,
  particle_color_blend: f32,
  particle_cutoff_prepass: u32,
  _reserved_119: u32,
  _reserved_120: u32,
  _reserved_121: u32,
  _reserved_122: u32,
  _reserved_123: u32,
  // Particle haze/contrast controls (slots 124-127). Defaults are visual no-ops.
  particle_exponent: f32,
  particle_brightness_boost: f32,
  particle_support_smoothing: f32,
  particle_haze_cull: f32,
};

@group(0) @binding(1) var<uniform> uniforms: RenderUniforms;

// ---- Per-particle variation + fractal noise -------------------------------------------------
// Self-contained (this shared chunk has no PRNG of its own). Each particle draws stable,
// per-index randomness ("frozen identity") optionally blended with slow time drift, and/or a
// coherent fBm field sampled at the particle's world position so neighbours vary together.
const VARIATION_MAX_OCTAVES: u32 = 4u;
const VARIATION_DRIFT_RATE: f32 = 0.05;
const VARIATION_FIELD_DRIFT: f32 = 0.08;

fn var_pcg_hash(seed: u32) -> u32 {
  let state = seed * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn var_hash_u(seed: u32) -> f32 {
  return f32(var_pcg_hash(seed)) / 4294967295.0;
}

fn var_hash3i(p: vec3i) -> f32 {
  let ux = bitcast<u32>(p.x);
  let uy = bitcast<u32>(p.y);
  let uz = bitcast<u32>(p.z);
  let h = var_pcg_hash((ux * 73856093u) ^ var_pcg_hash((uy * 19349663u) ^ var_pcg_hash(uz * 83492791u)));
  return f32(h) / 4294967295.0;
}

fn var_value_noise3(p: vec3f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let w = f * f * (3.0 - 2.0 * f);
  let bi = vec3i(i);
  let c000 = var_hash3i(bi + vec3i(0, 0, 0));
  let c100 = var_hash3i(bi + vec3i(1, 0, 0));
  let c010 = var_hash3i(bi + vec3i(0, 1, 0));
  let c110 = var_hash3i(bi + vec3i(1, 1, 0));
  let c001 = var_hash3i(bi + vec3i(0, 0, 1));
  let c101 = var_hash3i(bi + vec3i(1, 0, 1));
  let c011 = var_hash3i(bi + vec3i(0, 1, 1));
  let c111 = var_hash3i(bi + vec3i(1, 1, 1));
  let x00 = mix(c000, c100, w.x);
  let x10 = mix(c010, c110, w.x);
  let x01 = mix(c001, c101, w.x);
  let x11 = mix(c011, c111, w.x);
  let y0 = mix(x00, x10, w.y);
  let y1 = mix(x01, x11, w.y);
  return mix(y0, y1, w.z);
}

fn var_fbm3(p_in: vec3f, octaves: u32, gain: f32, lacunarity: f32) -> f32 {
  var amp = 0.5;
  var freq = 1.0;
  var sum = 0.0;
  var norm = 0.0;
  let n = min(octaves, VARIATION_MAX_OCTAVES);
  for (var o = 0u; o < VARIATION_MAX_OCTAVES; o = o + 1u) {
    if (o >= n) { break; }
    sum = sum + amp * var_value_noise3(p_in * freq);
    norm = norm + amp;
    amp = amp * gain;
    freq = freq * lacunarity;
  }
  return sum / max(0.0001, norm);
}

struct ParticleVariation {
  size_mul: f32,
  bright_mul: f32,
  opacity_mul: f32,
  hue_off: f32,
  sat_mul: f32,
  val_mul: f32,
};

fn var_stream_hash(index: u32, stream: u32) -> f32 {
  return var_hash_u(index * 747796405u + stream * 2654435761u + 374761393u);
}

// Frozen identity blended toward a slow temporal drift (smooth, no per-frame flicker).
fn var_white(index: u32, stream: u32, drift: f32, t: f32) -> f32 {
  let frozen = var_stream_hash(index, stream);
  let phase = t * VARIATION_DRIFT_RATE;
  let bucket = floor(phase);
  let frac = fract(phase);
  let a = var_stream_hash(index ^ (bitcast<u32>(i32(bucket)) * 2246822519u), stream);
  let b = var_stream_hash(index ^ (bitcast<u32>(i32(bucket) + 1) * 2246822519u), stream);
  let drifted = mix(a, b, smoothstep(0.0, 1.0, frac));
  return mix(frozen, drifted, clamp(drift, 0.0, 1.0));
}

// Coherent fBm signal sampled at the particle's world position (per-stream decorrelated).
fn var_coherent(pos: vec3f, stream: f32, t: f32) -> f32 {
  let p = pos * uniforms.variation_freq
    + vec3f(stream * 17.3, stream * 9.1, stream * 23.7)
    + vec3f(0.0, 0.0, t * VARIATION_FIELD_DRIFT);
  return var_fbm3(p, uniforms.variation_octaves, uniforms.variation_gain, uniforms.variation_lacunarity);
}

// Shape a unit signal with a power curve and map into [lo,hi] as a multiplier (neutral 1.0).
fn var_mul(u_white: f32, u_coherent: f32, curve: f32, lo: f32, hi: f32, amount: f32) -> f32 {
  let u = mix(u_white, u_coherent, clamp(uniforms.variation_noise_mix, 0.0, 1.0));
  let shaped = pow(clamp(u, 0.0, 1.0), max(0.0001, curve));
  return mix(1.0, mix(lo, hi, shaped), clamp(amount, 0.0, 1.0));
}

// Same shaping mapped as a signed offset (neutral 0.0).
fn var_off(u_white: f32, u_coherent: f32, curve: f32, lo: f32, hi: f32, amount: f32) -> f32 {
  let u = mix(u_white, u_coherent, clamp(uniforms.variation_noise_mix, 0.0, 1.0));
  let shaped = pow(clamp(u, 0.0, 1.0), max(0.0001, curve));
  return mix(0.0, mix(lo, hi, shaped), clamp(amount, 0.0, 1.0));
}

fn compute_particle_variation(index: u32, pos: vec3f) -> ParticleVariation {
  var v: ParticleVariation;
  v.size_mul = 1.0;
  v.bright_mul = 1.0;
  v.opacity_mul = 1.0;
  v.hue_off = 0.0;
  v.sat_mul = 1.0;
  v.val_mul = 1.0;
  let master = uniforms.variation_master;
  let any_amount = uniforms.variation_size_amount + uniforms.variation_bright_amount + uniforms.variation_opacity_amount + uniforms.variation_color_amount;
  if (master <= 0.0 || any_amount <= 0.0) {
    return v;
  }
  let drift = uniforms.variation_drift;
  let t = uniforms.variation_time;

  let size_v = var_mul(var_white(index, 0u, drift, t), var_coherent(pos, 0.0, t), uniforms.variation_size_curve, uniforms.variation_size_min, uniforms.variation_size_max, uniforms.variation_size_amount);
  v.size_mul = mix(1.0, size_v, master);

  let bright_v = var_mul(var_white(index, 1u, drift, t), var_coherent(pos, 1.0, t), uniforms.variation_bright_curve, uniforms.variation_bright_min, uniforms.variation_bright_max, uniforms.variation_bright_amount);
  v.bright_mul = mix(1.0, bright_v, master);

  let opacity_v = var_mul(var_white(index, 2u, drift, t), var_coherent(pos, 2.0, t), uniforms.variation_opacity_curve, uniforms.variation_opacity_min, uniforms.variation_opacity_max, uniforms.variation_opacity_amount);
  v.opacity_mul = mix(1.0, opacity_v, master);

  let hue = var_off(var_white(index, 3u, drift, t), var_coherent(pos, 3.0, t), uniforms.variation_color_curve, uniforms.variation_color_min, uniforms.variation_color_max, uniforms.variation_color_amount);
  v.hue_off = hue * master;
  let col_amt = clamp(uniforms.variation_color_amount, 0.0, 1.0) * master;
  v.sat_mul = mix(1.0, mix(0.6, 1.0, var_white(index, 4u, drift, t)), col_amt);
  v.val_mul = mix(1.0, mix(0.6, 1.2, var_white(index, 5u, drift, t)), col_amt);
  return v;
}

// Apply per-particle size variation. The reachable size range is defined entirely by the
// Size Min/Max multiplier envelope (variation_size_min/max), so no absolute pixel clamp is
// imposed here — that keeps the old particle_min_px/max_px contract intact and avoids the
// live-mode default max from capping the very particles the distribution is meant to enlarge.
fn particle_variation_sized_px(base_size_px: f32, size_mul: f32) -> f32 {
  return base_size_px * size_mul;
}
// -------------------------------------------------------------------------------------------

struct SplatOut {
  @builtin(position) position: vec4f,
  @location(0) local: vec2f,
  @location(1) alpha: f32,
  @location(2) density: f32,
  @location(3) depth: f32,
  @location(4) species: f32,
  @location(5) uv: vec2f,
  @location(6) mode: f32,
  @location(7) color_hsv: vec4f,
  @location(8) defocus: f32,
  @location(9) sprite_blur: f32,
  @location(10) bright_mul: f32,
  @location(11) speed: f32,
};

struct ParticleDensityStats {
  center: f32,
  peak: f32,
  mean: f32,
  support: f32,
  coherence: f32,
};

fn density_stats_from_axis_samples(center: f32, px: f32, nx: f32, py: f32, ny: f32, pz: f32, nz: f32) -> ParticleDensityStats {
  let peak = max(center, max(max(px, nx), max(max(py, ny), max(pz, nz))));
  let mean = max(0.0, (center * 2.0 + px + nx + py + ny + pz + nz) * 0.125);
  let support = mean / max(0.000001, peak);
  let axis_x = max(0.0, (px + nx) * 0.5);
  let axis_y = max(0.0, (py + ny) * 0.5);
  let axis_z = max(0.0, (pz + nz) * 0.5);
  let axis_mean = max(0.000001, (axis_x + axis_y + axis_z) / 3.0);
  let axis_peak = max(axis_x, max(axis_y, axis_z));
  let axis_floor = min(axis_x, min(axis_y, axis_z));
  let axis_contrast = (axis_peak - axis_floor) / max(0.000001, axis_peak + axis_mean * 0.5);
  let center_bias = clamp((center - mean) / max(0.000001, peak), 0.0, 1.0);
  let coherence = clamp(axis_contrast * 0.78 + center_bias * 0.22, 0.0, 1.0);
  return ParticleDensityStats(center, peak, mean, support, coherence);
}

fn corner(vertex_index: u32) -> vec2f {
  let c = vertex_index % 6u;
  if (c == 0u) { return vec2f(-1.0, -1.0); }
  if (c == 1u) { return vec2f(1.0, -1.0); }
  if (c == 2u) { return vec2f(-1.0, 1.0); }
  if (c == 3u) { return vec2f(-1.0, 1.0); }
  if (c == 4u) { return vec2f(1.0, -1.0); }
  return vec2f(1.0, 1.0);
}

fn billboard_corner(vertex_index: u32) -> vec2f {
  let c = vertex_index & 3u;
  if (c == 0u) { return vec2f(-1.0, -1.0); }
  if (c == 1u) { return vec2f(1.0, -1.0); }
  if (c == 2u) { return vec2f(-1.0, 1.0); }
  return vec2f(1.0, 1.0);
}

fn particle_splat_base_size_px(perspective: f32) -> f32 {
  // True 3D perspective: on-screen size scales with 1/view_depth, exactly like the projected
  // positions, so a particle GROWS as the camera dollies closer and SHRINKS as it recedes -- in
  // lockstep with the rest of the scene, the way real 3D works. The 'perspective' arg is
  // projected.z (= 1.85 / view_depth); the PARTICLE_SIZE_REF_DEPTH / 1.85 factor normalizes it so
  // particle_size_px renders at its literal pixel value at the reference framing, and the focal
  // ratio scales with the FOV slider. Do not apply a screen-space min/max floor here; old saved
  // P Min Px values must not turn live particles into clamped sprites. (A previous build dropped the
  // 'perspective' term to make size dolly-independent; with a constant pixel size the dots stop
  // shrinking with the receding cloud and APPEAR to scale up on zoom-out, which is not real 3D.)
  return max(0.0, uniforms.particle_size_px) * perspective * (PARTICLE_SIZE_REF_DEPTH / 1.85) * (uniforms.focal_k / DEFAULT_FOCAL_K);
}

fn particle_splat_radius_for_size_px(size_px: f32) -> vec2f {
  return vec2f(size_px * 2.0 / uniforms.resolution.x, size_px * 2.0 / uniforms.resolution.y);
}

fn particle_splat_radius_ndc(perspective: f32) -> vec2f {
  return particle_splat_radius_for_size_px(particle_splat_base_size_px(perspective));
}

fn particle_dof_splat_radius_ndc(perspective: f32, focus_blur_px: f32) -> vec2f {
  return particle_splat_radius_for_size_px(particle_splat_base_size_px(perspective) + max(0.0, focus_blur_px));
}

fn particle_dof_blur_px(depth: f32) -> f32 {
  return particle_dof_blur_px_from_defocus(particle_focus_defocus(depth));
}

fn particle_focus_defocus(depth: f32) -> f32 {
  let focus_error = abs(depth - uniforms.focus_distance);
  return clamp(focus_error * max(0.0, uniforms.aperture) * 8.0, 0.0, 1.0);
}

fn particle_dof_blur_px_from_defocus(defocus: f32) -> f32 {
  if (uniforms.dof_enabled == 0u) {
    return 0.0;
  }
  return clamp(defocus, 0.0, 1.0) * max(0.0, uniforms.dof_blur) * 8.0;
}

fn particle_sprite_blur_amount(focus_blur_px: f32, base_size_px: f32) -> f32 {
  let relative_blur = focus_blur_px / max(0.75, base_size_px * 0.75);
  return smoothstep(0.08, 1.0, relative_blur);
}

fn voxel_to_world(x: u32, y: u32, z: u32) -> vec3f {
  return vec3f(
    ((f32(x) + 0.5) / f32(uniforms.width)) * 2.0 - 1.0,
    ((f32(y) + 0.5) / f32(uniforms.height)) * 2.0 - 1.0,
    ((f32(z) + 0.5) / f32(uniforms.depth)) * 2.0 - 1.0
  );
}

fn project(position: vec3f) -> vec3f {
  let camera = world_to_camera(position);
  let inv_depth = 1.0 / max(PARTICLE_NEAR, uniforms.distance + camera.z);
  let aspect = uniforms.resolution.x / uniforms.resolution.y;
  // x,y use the current focal length (FOV slider). .z stays the FOV-independent depth metric
  // (1.85 / view_depth) that DOF focus distance and splat size are calibrated against.
  return vec3f(camera.x * uniforms.focal_k * inv_depth / aspect, camera.y * uniforms.focal_k * inv_depth, 1.85 * inv_depth);
}

fn world_to_camera(position: vec3f) -> vec3f {
  let cy = uniforms.yaw_cos;
  let sy = uniforms.yaw_sin;
  let cp = uniforms.pitch_cos;
  let sp = uniforms.pitch_sin;
  let rx = position.x * cy - position.z * sy;
  let rz = position.x * sy + position.z * cy;
  let ry = position.y * cp - rz * sp;
  let rz2 = position.y * sp + rz * cp;
  return vec3f(rx + uniforms.camera_pan_x, ry + uniforms.camera_pan_y, rz2);
}

fn world_vector_to_camera(vector: vec3f) -> vec3f {
  let cy = uniforms.yaw_cos;
  let sy = uniforms.yaw_sin;
  let cp = uniforms.pitch_cos;
  let sp = uniforms.pitch_sin;
  let rx = vector.x * cy - vector.z * sy;
  let rz = vector.x * sy + vector.z * cy;
  let ry = vector.y * cp - rz * sp;
  let rz2 = vector.y * sp + rz * cp;
  return vec3f(rx, ry, rz2);
}

fn camera_to_world(position: vec3f) -> vec3f {
  let cy = uniforms.yaw_cos;
  let sy = uniforms.yaw_sin;
  let cp = uniforms.pitch_cos;
  let sp = uniforms.pitch_sin;
  let y = position.y * cp + position.z * sp;
  let rz = -position.y * sp + position.z * cp;
  let x = position.x * cy + rz * sy;
  let z = -position.x * sy + rz * cy;
  return vec3f(x, y, z);
}

fn palette(kind: u32, density: f32, depth: f32, species: f32) -> vec3f {
  let species_mix = vec3f(0.07 * species, 0.045 * (1.0 - abs(species - 0.5)), 0.09 * (1.0 - species));
  if (kind == 1u) {
    return vec3f(1.0, 0.34 + density * 0.42, 0.10 + depth * 0.14) + species_mix;
  }
  if (kind == 2u) {
    return vec3f(0.48 + density * 0.44, 0.30 + depth * 0.45, 1.0) + species_mix;
  }
  return vec3f(0.20 + depth * 0.34, 0.74 + density * 0.24, 0.68 + density * 0.28) + species_mix;
}

fn hsv2rgb(c: vec3f) -> vec3f {
  let k = vec4f(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  let p = abs(fract(c.xxx + k.xyz) * 6.0 - k.www);
  return c.z * mix(k.xxx, clamp(p - k.xxx, vec3f(0.0), vec3f(1.0)), c.y);
}

fn saturate_color(color: vec3f, amount: f32) -> vec3f {
  let luma = dot(color, vec3f(0.2126, 0.7152, 0.0722));
  return max(vec3f(0.0), mix(vec3f(luma), color, amount));
}

fn fog_tint_color() -> vec3f {
  return vec3f(uniforms.fog_tint_r, uniforms.fog_tint_g, uniforms.fog_tint_b);
}

fn particle_tint_color() -> vec3f {
  return vec3f(uniforms.particle_tint_r, uniforms.particle_tint_g, uniforms.particle_tint_b);
}

fn apply_tint(color: vec3f, tint: vec3f, amount: f32) -> vec3f {
  return mix(color, color * tint, amount);
}

fn scene_gain() -> f32 {
  return 0.35 + max(0.0, uniforms.scene_brightness) * 0.65;
}

fn particle_gain() -> f32 {
  return uniforms.exposure * scene_gain() * max(0.0, uniforms.particle_brightness);
}

fn fog_gain() -> f32 {
  return uniforms.exposure * scene_gain() * max(0.0, uniforms.fog_brightness) * 3.0;
}

fn ramp4(t: f32, a: vec3f, b: vec3f, c: vec3f, d: vec3f) -> vec3f {
  let x = clamp(t, 0.0, 1.0);
  if (x < 0.333333) {
    return mix(a, b, smoothstep(0.0, 0.333333, x));
  }
  if (x < 0.666667) {
    return mix(b, c, smoothstep(0.333333, 0.666667, x));
  }
  return mix(c, d, smoothstep(0.666667, 1.0, x));
}

fn particle_gradient_color(t: f32, mode_in: u32) -> vec3f {
  // Velocity/audio VARIANT modes reuse the base gradient ramps; remap the variant id to its ramp id
  // here so every caller (particle splat, density, accumulation, volume) gets the right palette.
  var mode = mode_in;
  if (mode_in == 14u) { mode = 1u; }        // velocity -> inferno
  else if (mode_in == 15u) { mode = 3u; }   // velocity -> viridis
  else if (mode_in == 16u) { mode = 6u; }   // velocity -> spectral
  else if (mode_in == 17u) { mode = 8u; }   // velocity -> cosmic
  else if (mode_in == 18u) { mode = 9u; }   // velocity -> ice
  else if (mode_in == 19u) { mode = 2u; }   // audio -> magma
  else if (mode_in == 20u) { mode = 3u; }   // audio -> viridis
  else if (mode_in == 21u) { mode = 4u; }   // audio -> turbo
  else if (mode_in == 22u) { mode = 8u; }   // audio -> cosmic
  else if (mode_in == 23u) { mode = 9u; }   // audio -> ice
  else if (mode_in == 24u) { mode = 10u; }  // audio -> ember
  else if (mode_in == 25u) { mode = 7u; }   // audio -> plasma
  if (mode == 1u) {
    return ramp4(t, vec3f(0.02, 0.01, 0.08), vec3f(0.45, 0.05, 0.35), vec3f(0.91, 0.24, 0.12), vec3f(0.99, 0.96, 0.64));
  }
  if (mode == 2u) {
    return ramp4(t, vec3f(0.01, 0.00, 0.03), vec3f(0.32, 0.07, 0.48), vec3f(0.82, 0.25, 0.32), vec3f(0.99, 0.87, 0.58));
  }
  if (mode == 3u) {
    return ramp4(t, vec3f(0.17, 0.00, 0.33), vec3f(0.10, 0.38, 0.55), vec3f(0.20, 0.70, 0.48), vec3f(0.99, 0.91, 0.14));
  }
  if (mode == 4u) {
    return ramp4(t, vec3f(0.19, 0.07, 0.23), vec3f(0.10, 0.48, 0.97), vec3f(0.51, 0.99, 0.37), vec3f(0.98, 0.17, 0.09));
  }
  if (mode == 5u) {
    return hsv2rgb(vec3f(t, 0.92, 1.0));
  }
  if (mode == 7u) { // plasma: violet -> magenta -> orange -> hot white
    return ramp4(t, vec3f(0.05, 0.00, 0.20), vec3f(0.50, 0.00, 0.60), vec3f(1.00, 0.30, 0.40), vec3f(1.00, 0.92, 0.62));
  }
  if (mode == 8u) { // cosmic: deep blue -> indigo -> magenta -> cyan white
    return ramp4(t, vec3f(0.02, 0.00, 0.10), vec3f(0.20, 0.05, 0.50), vec3f(0.80, 0.10, 0.62), vec3f(0.60, 0.95, 1.00));
  }
  if (mode == 9u) { // ice: midnight teal -> blue -> cyan -> white
    return ramp4(t, vec3f(0.00, 0.05, 0.12), vec3f(0.05, 0.32, 0.62), vec3f(0.42, 0.82, 0.96), vec3f(0.92, 1.00, 1.00));
  }
  if (mode == 10u) { // ember: near-black red -> red -> orange -> warm white
    return ramp4(t, vec3f(0.05, 0.00, 0.00), vec3f(0.62, 0.05, 0.00), vec3f(1.00, 0.42, 0.06), vec3f(1.00, 0.95, 0.72));
  }
  if (mode == 11u) { // velocity/speed: blue (slow) -> cyan -> yellow -> red (fast)
    return ramp4(t, vec3f(0.05, 0.10, 0.55), vec3f(0.10, 0.75, 0.90), vec3f(0.95, 0.85, 0.20), vec3f(1.00, 0.18, 0.08));
  }
  if (mode == 12u) { // cohort: evenly-spaced distinct hue per cohort
    return hsv2rgb(vec3f(t, 0.85, 1.0));
  }
  if (mode == 13u) { // audio-reactive: vivid spectral (t is audio-shifted in particle_color)
    return ramp4(t, vec3f(0.15, 0.00, 0.45), vec3f(0.95, 0.10, 0.55), vec3f(1.00, 0.65, 0.10), vec3f(0.40, 0.95, 1.00));
  }
  return ramp4(t, vec3f(0.37, 0.31, 0.64), vec3f(0.15, 0.68, 0.50), vec3f(0.99, 0.87, 0.35), vec3f(0.83, 0.18, 0.14));
}

fn particle_velocity_screen_axis(velocity: vec3f) -> vec2f {
  let speed = length(velocity);
  if (speed <= 0.000001) {
    return vec2f(0.0);
  }
  let aspect = uniforms.resolution.x / uniforms.resolution.y;
  let camera_velocity = world_vector_to_camera(velocity);
  let axis = camera_velocity.xy * vec2f(aspect, 1.0);
  let axis_length = length(axis);
  if (axis_length <= 0.000001) {
    return vec2f(0.0);
  }
  return axis / axis_length;
}

fn particle_velocity_stretch_amount(velocity: vec3f) -> f32 {
  let min_stretch = clamp(uniforms.particle_stretch_min, 0.0, 6.0);
  let max_stretch = max(min_stretch, clamp(uniforms.particle_stretch, 0.0, 6.0));
  let speed_t = smoothstep(0.0, max(0.000001, uniforms.particle_stretch_speed), length(velocity));
  return mix(min_stretch, max_stretch, speed_t);
}

fn particle_density_gate_radius() -> f32 {
  let configured = max(0.0, uniforms.particle_density_radius);
  if (uniforms.particle_density_cutoff <= 0.0 && uniforms.particle_density_normalize <= 0.0001) {
    return configured;
  }
  let max_dim = max(f32(uniforms.width), max(f32(uniforms.height), f32(uniforms.depth)));
  let voxel_radius = 2.0 / max(1.0, max_dim);
  return max(configured, voxel_radius * 1.5);
}

fn particle_speed_cutoff_visibility(velocity: vec3f) -> f32 {
  let speed = length(velocity);
  let upper_cutoff = max(0.0, uniforms.particle_speed_cutoff);
  let lower_cutoff = max(0.0, uniforms.particle_slow_cutoff);
  if (upper_cutoff <= 0.0 && lower_cutoff <= 0.0) {
    return 1.0;
  }
  var visibility = 1.0;
  if (upper_cutoff > 0.0) {
    let upper_activation = smoothstep(0.0, 0.02, upper_cutoff);
    let upper_edge_width = max(0.0015, upper_cutoff * 0.08);
    let upper_visibility = 1.0 - smoothstep(upper_cutoff, upper_cutoff + upper_edge_width, speed);
    visibility = visibility * mix(1.0, upper_visibility, upper_activation);
  }
  if (lower_cutoff > 0.0) {
    let lower_activation = smoothstep(0.0, 0.02, lower_cutoff);
    let lower_edge_width = max(0.0015, lower_cutoff * 0.08);
    let lower_visibility = smoothstep(max(0.0, lower_cutoff - lower_edge_width), lower_cutoff, speed);
    visibility = visibility * mix(1.0, lower_visibility, lower_activation);
  }
  return visibility;
}

fn particle_gradient_coordinate(velocity: vec3f, base: f32) -> f32 {
  let speed = length(velocity);
  if (speed <= 0.000001) {
    return fract(base);
  }
  let dir = velocity / speed;
  let azimuth = atan2(dir.z, dir.x) / (2.0 * PI) + 0.5;
  let elevation = asin(clamp(dir.y, -1.0, 1.0)) / PI + 0.5;
  let speed_band = smoothstep(0.00001, 0.035, speed);
  let velocity_t = fract(azimuth + elevation * 0.18 + speed_band * 0.23);
  let sensitivity = clamp(uniforms.particle_gradient_sensitivity, -1.0, 1.0);
  let directed_velocity_t = select(velocity_t, 1.0 - velocity_t, sensitivity < 0.0);
  let velocity_mix = smoothstep(0.0, 1.0, abs(sensitivity));
  return fract(mix(fract(base), directed_velocity_t, velocity_mix));
}

fn wrap_particle_position(position: vec3f) -> vec3f {
  return fract((position + vec3f(1.0)) * 0.5) * 2.0 - vec3f(1.0);
}

fn shortest_particle_delta(previous: vec3f, current: vec3f) -> vec3f {
  var delta = current - previous;
  if (delta.x > 1.0) {
    delta.x = delta.x - 2.0;
  } else if (delta.x < -1.0) {
    delta.x = delta.x + 2.0;
  }
  if (delta.y > 1.0) {
    delta.y = delta.y - 2.0;
  } else if (delta.y < -1.0) {
    delta.y = delta.y + 2.0;
  }
  if (delta.z > 1.0) {
    delta.z = delta.z - 2.0;
  } else if (delta.z < -1.0) {
    delta.z = delta.z + 2.0;
  }
  return delta;
}

fn particle_wrap_crossed(previous: vec3f, current: vec3f) -> bool {
  let raw_delta = current - previous;
  return abs(raw_delta.x) > 1.0 || abs(raw_delta.y) > 1.0 || abs(raw_delta.z) > 1.0;
}

fn particle_lerp_amount(previous: vec3f, current: vec3f, requested_t: f32) -> f32 {
  let t = clamp(requested_t, 0.0, 1.0);
  if (particle_wrap_crossed(previous, current)) {
    return 1.0;
  }
  let motion = shortest_particle_delta(previous, current);
  if (length(motion) > PARTICLE_INTERP_SNAP_DISTANCE) {
    return 1.0;
  }
  return t;
}

fn particle_large_motion_fade(previous: vec3f, current: vec3f, requested_t: f32) -> f32 {
  let motion = shortest_particle_delta(previous, current);
  let motion_length = length(motion);
  if (motion_length <= PARTICLE_INTERP_SNAP_DISTANCE && !particle_wrap_crossed(previous, current)) {
    return 1.0;
  }
  let t = clamp(requested_t, 0.0, 1.0);
  return smoothstep(0.05, 0.85, t);
}

fn interpolate_particle_position(previous: vec3f, current: vec3f, t: f32) -> vec3f {
  return wrap_particle_position(previous + shortest_particle_delta(previous, current) * t);
}

fn interpolate_hue(previous: f32, current: f32, t: f32) -> f32 {
  let delta = fract(current - previous + 0.5) - 0.5;
  return fract(previous + delta * t);
}

// Reconstruct the particle color vec4 from the packed struct: only hue varies (stored in
// vel_id.w); saturation/value/alpha are constants (0.8, 1.0, 0.045) as before the struct was
// shrunk from 48 to 32 bytes.
fn stored_particle_color(p: Particle) -> vec4f {
  return vec4f(p.vel_id.w, 0.8, 1.0, 0.045);
}

fn interpolate_particle_color(previous: vec4f, current: vec4f, t: f32) -> vec4f {
  return vec4f(interpolate_hue(previous.x, current.x, t), mix(previous.yzw, current.yzw, t));
}

fn particle_velocity_blend(previous: Particle, current: Particle, t: f32) -> vec3f {
  return mix(previous.vel_id.xyz, current.vel_id.xyz, t);
}

fn particle_visual_velocity(previous: Particle, current: Particle, t: f32) -> vec3f {
  let velocity = particle_velocity_blend(previous, current, t);
  let motion = shortest_particle_delta(previous.pos_cohort.xyz, current.pos_cohort.xyz);
  let motion_length = length(motion);
  if (motion_length > PARTICLE_INTERP_SNAP_DISTANCE || particle_wrap_crossed(previous.pos_cohort.xyz, current.pos_cohort.xyz)) {
    return velocity;
  }
  if (motion_length > 0.000001) {
    return motion;
  }
  return velocity;
}

fn trail_vector_color(flow_vector: vec3f, signal: f32) -> vec3f {
  let magnitude = length(flow_vector);
  let confidence = smoothstep(0.004, 0.055, magnitude);
  let direction = flow_vector / max(0.000001, magnitude);
  let hue = atan2(direction.y, direction.x) / (2.0 * PI) + direction.z * 0.125;
  let density_signal = smoothstep(0.0, 0.85, signal);
  let palette_anchor = palette(uniforms.palette, density_signal, confidence, 0.42 + confidence * 0.28);
  let saturation = mix(0.42, 0.94, confidence) * clamp(0.72 + abs(direction.z) * 0.2, 0.0, 1.0);
  let value = smoothstep(0.0, 0.75, signal);
  let hue_color = hsv2rgb(vec3f(hue, saturation, 1.0));
  let chroma = confidence * smoothstep(0.08, 0.72, signal);
  return mix(palette_anchor, hue_color, chroma) * (0.16 + value * 2.25 + confidence * 0.28);
}

fn trail_stable_color(raw: vec4f, signal: f32) -> vec3f {
  let density_signal = smoothstep(0.0, 0.85, signal);
  let flow_energy = smoothstep(0.0, 0.32, length(raw.xyz) * 42.0);
  let palette_color = palette(uniforms.palette, density_signal, flow_energy, 0.38 + flow_energy * 0.42);
  let low = vec3f(0.08, 0.46, 0.52);
  let high = vec3f(1.0, 0.76, 0.34);
  let scalar_color = mix(low, high, density_signal);
  let color = saturate_color(mix(scalar_color, palette_color, 0.48), 1.22 + flow_energy * 0.18);
  return color * (0.18 + density_signal * 1.72 + flow_energy * 0.28);
}

fn trail_thermal_color(raw: vec4f, signal: f32) -> vec3f {
  let density_signal = smoothstep(0.0, 0.85, signal);
  let flow_energy = smoothstep(0.0, 0.32, length(raw.xyz) * 42.0);
  let cold = vec3f(0.05, 0.18, 0.72);
  let warm = vec3f(0.96, 0.32, 0.10);
  let hot = vec3f(1.0, 0.92, 0.48);
  let base = mix(cold, warm, density_signal);
  return saturate_color(mix(base, hot, smoothstep(0.68, 1.0, density_signal)), 1.18 + flow_energy * 0.22) * (0.14 + density_signal * 2.2 + flow_energy * 0.34);
}

fn trail_tint_color(signal: f32, flow_energy: f32) -> vec3f {
  return fog_tint_color() * (0.18 + smoothstep(0.0, 0.85, signal) * 2.25 + flow_energy * 0.3);
}
`;

const liveParticleSupportComputeShader = /* wgsl */ `
${liveSplatRenderCommon}

const PARTICLE_SUPPORT_VELOCITY_SCALE = 256.0;

@group(0) @binding(0) var<storage, read> support_particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> particle_support_grid: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> particle_support_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> particle_active_indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> particle_active_draw: array<atomic<u32>>;

fn support_wrap_i(value: i32, size: u32) -> u32 {
  let s = i32(size);
  return u32(((value % s) + s) % s);
}

fn particle_support_grid_size() -> u32 {
  return max(1u, uniforms.particle_support_grid_size);
}

fn particle_support_grid_cell(position: vec3f) -> vec3i {
  let size = f32(particle_support_grid_size());
  let wrapped = fract((position + vec3f(1.0)) * 0.5);
  return vec3i(floor(wrapped * size));
}

fn particle_support_grid_index(cell: vec3i) -> u32 {
  let size = particle_support_grid_size();
  let x = support_wrap_i(cell.x, size);
  let y = support_wrap_i(cell.y, size);
  let z = support_wrap_i(cell.z, size);
  return ((z * size + y) * size + x) * 4u;
}

fn particle_support_direction(velocity: vec3f) -> vec3f {
  let speed = length(velocity);
  if (speed <= 0.000001) {
    return vec3f(0.0);
  }
  return velocity / speed;
}

@compute @workgroup_size(256)
fn clear_particle_support_grid(@builtin(global_invocation_id) gid: vec3u) {
  let flat = gid.x;
  let size = particle_support_grid_size();
  let lane_count = size * size * size * 4u;
  if (flat >= lane_count) {
    return;
  }
  atomicStore(&particle_support_grid[flat], 0);
}

@compute @workgroup_size(256)
fn build_particle_support_grid(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&support_particles)) {
    return;
  }
  let particle = support_particles[i];
  let base = particle_support_grid_index(particle_support_grid_cell(particle.pos_cohort.xyz));
  let dir = particle_support_direction(particle.vel_id.xyz);
  atomicAdd(&particle_support_grid[base + 0u], 1);
  atomicAdd(&particle_support_grid[base + 1u], i32(round(dir.x * PARTICLE_SUPPORT_VELOCITY_SCALE)));
  atomicAdd(&particle_support_grid[base + 2u], i32(round(dir.y * PARTICLE_SUPPORT_VELOCITY_SCALE)));
  atomicAdd(&particle_support_grid[base + 3u], i32(round(dir.z * PARTICLE_SUPPORT_VELOCITY_SCALE)));
}

fn particle_support_alignment(particle_dir: vec3f, cell_dir_sum: vec3f) -> f32 {
  let particle_speed = length(particle_dir);
  let cell_speed = length(cell_dir_sum);
  if (particle_speed <= 0.000001 || cell_speed <= 0.000001) {
    return 1.0;
  }
  return smoothstep(-0.2, 0.8, dot(particle_dir / particle_speed, cell_dir_sum / cell_speed));
}

fn particle_support_visibility(support: f32) -> f32 {
  let amount = clamp(uniforms.particle_support_mask, 0.0, 1.0);
  if (amount <= 0.0001) {
    return 1.0;
  }
  let cutoff = mix(0.05, 0.74, amount);
  let softness = mix(0.34, 0.18, amount);
  return smoothstep(cutoff, min(1.0, cutoff + softness), clamp(support, 0.0, 1.0));
}

@compute @workgroup_size(256)
fn resolve_particle_support(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&support_particles)) {
    return;
  }
  let particle = support_particles[i];
  let center = particle_support_grid_cell(particle.pos_cohort.xyz);
  let particle_dir = particle_support_direction(particle.vel_id.xyz);
  let radius = clamp(uniforms.particle_support_radius, 0.35, 1.75);
  let radius_inner = max(0.0, radius - 0.75);
  var weighted_neighbors = 0.0;
  var weighted_alignment = 0.0;
  var alignment_weight = 0.0;
  var strongest_cell = 0.0;
  for (var dz = -1i; dz <= 1i; dz = dz + 1i) {
    for (var dy = -1i; dy <= 1i; dy = dy + 1i) {
      for (var dx = -1i; dx <= 1i; dx = dx + 1i) {
        let offset = vec3i(dx, dy, dz);
        let offset_distance = length(vec3f(f32(dx), f32(dy), f32(dz)));
        if (offset_distance <= radius + 0.001) {
          let base = particle_support_grid_index(center + offset);
          let self_count = select(0, 1, dx == 0i && dy == 0i && dz == 0i);
          let neighbor_count = max(0, atomicLoad(&particle_support_grid[base + 0u]) - self_count);
          if (neighbor_count > 0) {
            let radial_weight = 1.0 - smoothstep(radius_inner, radius + 0.001, offset_distance);
            let count_weight = f32(neighbor_count) * max(0.0, radial_weight);
            let cell_dir = vec3f(
              f32(atomicLoad(&particle_support_grid[base + 1u])),
              f32(atomicLoad(&particle_support_grid[base + 2u])),
              f32(atomicLoad(&particle_support_grid[base + 3u]))
            ) / PARTICLE_SUPPORT_VELOCITY_SCALE;
            weighted_neighbors = weighted_neighbors + count_weight;
            strongest_cell = max(strongest_cell, count_weight);
            weighted_alignment = weighted_alignment + particle_support_alignment(particle_dir, cell_dir) * count_weight;
            alignment_weight = alignment_weight + count_weight;
          }
        }
      }
    }
  }
  let min_neighbors = max(0.25, uniforms.particle_support_neighbors);
  let neighbor_signal = max(weighted_neighbors, strongest_cell * 1.45);
  let neighbor_score = smoothstep(min_neighbors * 0.35, min_neighbors, neighbor_signal);
  let alignment_score = select(1.0, weighted_alignment / max(0.000001, alignment_weight), alignment_weight > 0.000001);
  let flow_gate = mix(1.0, alignment_score, clamp(uniforms.particle_support_flow, 0.0, 1.0));
  let support = clamp(neighbor_score * flow_gate, 0.0, 1.0);
  particle_support_out[i] = support;
  if (particle_support_visibility(support) > 0.000001) {
    let slot = atomicAdd(&particle_active_draw[1], 1u);
    particle_active_indices[slot] = i;
  }
}
`;

const liveSplatFragmentShader = /* wgsl */ `
// Focus-debug tint: green is in focus, yellow/orange is outside the focal band, red is fully defocused.
// This is intentionally near/far agnostic so the debug view answers one question quickly:
// "which particles are outside the focus plane?" DOF Blur controls how soft those red areas render.
fn dof_debug_tint(defocus: f32) -> vec3f {
  let blur_t = smoothstep(0.0, 1.0, clamp(defocus, 0.0, 1.0));
  let focus_color = vec3f(0.0, 1.0, 0.12);
  let blur_color = vec3f(1.0, 0.02, 0.0);
  return mix(focus_color, blur_color, blur_t);
}

// Parameterized so the compute splatter shares the exact same color math as the fragment path.
fn splat_color_for_mode(color_hsv: vec4f, speed: f32, species: f32, mode: u32) -> vec3f {
  if (mode == 0u) {
    return particle_tint_color();
  }
  // Coordinate-source modes override the gradient coordinate t with a different per-particle signal.
  // Velocity family (11 signature, 14-18 inferno/viridis/spectral/cosmic/ice) maps t to speed.
  // Audio family (13 signature, 19-25 magma/viridis/turbo/cosmic/ice/ember/plasma) shifts t by the
  // live bands and pulses brightness with the bass. 12 = cohort. The ramp per mode is chosen inside
  // particle_gradient_color().
  let is_velocity = (mode == 11u) || (mode >= 14u && mode <= 18u);
  let is_audio = (mode == 13u) || (mode >= 19u && mode <= 25u);
  var t = fract(color_hsv.x);
  if (is_velocity) {
    t = clamp(speed, 0.0, 1.0);
  } else if (mode == 12u) {
    t = fract(species);
  } else if (is_audio) {
    t = fract(color_hsv.x + uniforms.audio_low * 0.5 + uniforms.audio_mid * 0.25);
  }
  var color = particle_gradient_color(t, mode);
  if (is_audio) {
    color = color * (1.0 + uniforms.audio_low * 1.5 + uniforms.audio_high * 0.6);
  }
  // Per-particle saturation jitter rides in color_hsv.y (base 0.8 from stored_particle_color);
  // ratio 1.0 leaves the gradient untouched when color variation is off.
  color = saturate_color(color, clamp(color_hsv.y / 0.8, 0.0, 2.0));
  return apply_tint(color, particle_tint_color(), 0.28);
}

fn particle_color_for_mode(in: SplatOut, mode: u32) -> vec3f {
  return splat_color_for_mode(in.color_hsv, in.speed, in.species, mode);
}

fn splat_color(color_hsv: vec4f, speed: f32, species: f32) -> vec3f {
  // Time-based cross-fade between color modes: when the user changes mode the CPU eases
  // particle_color_blend 0->1 over a short window (and retargets to wherever they land), so
  // switching rolls smoothly into the new palette instead of hard-cutting. When settled (blend>=1)
  // or from==to we compute the color once -- no extra cost outside the brief transition.
  let c_to = splat_color_for_mode(color_hsv, speed, species, uniforms.particle_color_mode);
  if (uniforms.particle_color_blend >= 1.0 || uniforms.particle_color_mode_from == uniforms.particle_color_mode) {
    return c_to;
  }
  let c_from = splat_color_for_mode(color_hsv, speed, species, uniforms.particle_color_mode_from);
  return mix(c_from, c_to, clamp(uniforms.particle_color_blend, 0.0, 1.0));
}

fn particle_color(in: SplatOut) -> vec3f {
  if (uniforms.dof_debug == 1u) {
    return dof_debug_tint(in.defocus);
  }
  return splat_color(in.color_hsv, in.speed, in.species);
}

fn particle_intensity(glow: f32, alpha: f32, bright_mul: f32) -> f32 {
  return glow * alpha * particle_gain() * bright_mul;
}

fn particle_sprite_profile(local: vec2f, defocus: f32) -> f32 {
  let r = length(local);
  let amount = clamp(defocus, 0.0, 1.0);
  let aa = max(fwidth(r) * 1.35, 0.008);
  let sharp = 1.0 - smoothstep(1.0 - aa * 2.0, 1.0 + aa, r);
  var mask = clamp(sharp, 0.0, 1.0);
  if (amount > 0.001) {
    let soft_inner = mix(0.86, 0.05, amount);
    let soft_disc = 1.0 - smoothstep(soft_inner, 1.0, r);
    let center = 1.0 - smoothstep(0.0, 1.0, r);
    let soft = soft_disc * (0.8 + center * 0.2);
    mask = clamp(mix(sharp, soft, smoothstep(0.0, 1.0, amount)), 0.0, 1.0);
  }
  // Optional radial glow core (particle_glow_core). 0 = flat anti-aliased disc (the original look,
  // with an early-out so there is no extra cost when disabled). >0 fades the disc from a bright
  // center to a soft edge for an ember/star profile. Same rasterized fragment count as the flat
  // disc, so no fill-rate cost beyond a few math ops; the fragment stage is not the bottleneck.
  let glow_amt = clamp(uniforms.particle_glow_core, 0.0, 1.0);
  if (glow_amt <= 0.001) {
    return mask;
  }
  let core = pow(clamp(1.0 - r, 0.0, 1.0), mix(0.7, 3.0, glow_amt));
  return mask * mix(1.0, core, glow_amt);
}

@fragment
fn splat_fs(in: SplatOut) -> @location(0) vec4f {
  let glow = particle_sprite_profile(in.local, in.sprite_blur);
  var color = particle_color(in);
  // Optional hot core (particle_hot_core): pushes the very center of the sprite toward white for a
  // glowing plasma/ember look while leaving the colored halo intact. 0 = off (no cost beyond one
  // compare); higher values whiten a tighter, brighter core.
  let hot = clamp(uniforms.particle_hot_core, 0.0, 1.0);
  if (hot > 0.001) {
    let centerness = pow(clamp(1.0 - length(in.local), 0.0, 1.0), 2.0);
    color = mix(color, vec3f(1.0), centerness * hot);
  }
  // One convention for every mode: premultiplied color = color * intensity, alpha = intensity,
  // where intensity = glow * opacity * gain. Additive adds it, alpha composites it over, opaque
  // writes it solid (no blend). intensity is computed once so all modes share it.
  let intensity = particle_intensity(glow, in.alpha, in.bright_mul);
  if (uniforms.particle_blend_mode != 1u && (glow < 0.01 || in.alpha <= 0.001)) {
    discard;
  }
  if (uniforms.particle_blend_mode == 2u) {
    return vec4f(color * intensity, 1.0);
  }
  return vec4f(color * intensity, clamp(intensity, 0.0, 1.0));
}
`;

const liveVolumeRenderShader = /* wgsl */ `
${liveSplatRenderCommon}

@group(0) @binding(0) var<storage, read> values: array<vec4f>;
@group(0) @binding(2) var field_texture: texture_3d<f32>;
@group(0) @binding(3) var field_texture_sampler: sampler;

fn index_clamped(x: i32, y: i32, z: i32) -> u32 {
  let ix = u32(clamp(x, 0i, i32(uniforms.width) - 1i));
  let iy = u32(clamp(y, 0i, i32(uniforms.height) - 1i));
  let iz = u32(clamp(z, 0i, i32(uniforms.depth) - 1i));
  return (iz * uniforms.height + iy) * uniforms.width + ix;
}

fn domain_sdf(p: vec3f, shape: u32) -> f32 {
  if (shape == 1u) {
    return length(p) - 1.0;
  }
  if (shape == 2u) {
    let d_radial = length(p.xz) - 1.0;
    let d_cap = abs(p.y) - 1.0;
    return length(max(vec2f(d_radial, d_cap), vec2f(0.0))) + min(max(d_radial, d_cap), 0.0);
  }
  if (shape == 3u) {
    let half_w = (1.0 - p.y) * 0.5;
    return max(max(-1.0 - p.y, abs(p.x) - half_w), abs(p.z) - half_w);
  }
  let q = abs(p) - vec3f(1.0);
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sample_volume(position: vec3f) -> vec4f {
  if (any(position < vec3f(-1.0)) || any(position > vec3f(1.0))) {
    return vec4f(0.0);
  }
  if (uniforms.field_texture_mode == 1u) {
    let uvw = clamp((position + vec3f(1.0)) * 0.5, vec3f(0.0), vec3f(1.0));
    return textureSampleLevel(field_texture, field_texture_sampler, uvw, 0.0);
  }
  let dims = vec3f(f32(uniforms.width), f32(uniforms.height), f32(uniforms.depth));
  let grid = (position + vec3f(1.0)) * 0.5 * dims - vec3f(0.5);
  let base = vec3i(floor(grid));
  let t = fract(grid);
  var accum = vec4f(0.0);
  for (var dz = 0i; dz <= 1i; dz = dz + 1i) {
    for (var dy = 0i; dy <= 1i; dy = dy + 1i) {
      for (var dx = 0i; dx <= 1i; dx = dx + 1i) {
        let weight = mix(1.0 - t.x, t.x, f32(dx)) * mix(1.0 - t.y, t.y, f32(dy)) * mix(1.0 - t.z, t.z, f32(dz));
        accum = accum + values[index_clamped(base.x + dx, base.y + dy, base.z + dz)] * weight;
      }
    }
  }
  return accum;
}

fn intersect_box(origin: vec3f, direction: vec3f) -> vec2f {
  let inv_dir = 1.0 / direction;
  let t0 = (vec3f(-1.0) - origin) * inv_dir;
  let t1 = (vec3f(1.0) - origin) * inv_dir;
  let tmin3 = min(t0, t1);
  let tmax3 = max(t0, t1);
  let tmin = max(max(tmin3.x, tmin3.y), tmin3.z);
  let tmax = min(min(tmax3.x, tmax3.y), tmax3.z);
  return vec2f(tmin, tmax);
}

fn hash21(p: vec2f) -> f32 {
  let h = dot(p, vec2f(127.1, 311.7));
  return fract(sin(h) * 43758.5453123);
}

fn interleaved_gradient_noise(pixel: vec2f, frame: u32) -> f32 {
  let frame_offset = f32(frame % 64u) * 0.071;
  return fract(52.9829189 * fract(0.06711056 * (pixel.x + frame_offset) + 0.00583715 * (pixel.y - frame_offset)));
}

@vertex
fn field_vs(@builtin(vertex_index) vertex_index: u32) -> SplatOut {
  var out: SplatOut;
  if (uniforms.render_layer == 3u) {
    let voxel = vertex_index / 6u;
    let x = voxel % uniforms.width;
    let y = (voxel / uniforms.width) % uniforms.height;
    let z = voxel / (uniforms.width * uniforms.height);
    let raw = values[voxel];
    let density = clamp((length(raw.xyz) * 2.2 + raw.w * 0.22) * uniforms.density, 0.0, 1.0);
    let projected = project(voxel_to_world(x, y, z));
    let aspect = uniforms.resolution.x / uniforms.resolution.y;
    let local = corner(vertex_index);
    let focus_defocus = particle_focus_defocus(projected.z);
    let focus_blur_px = particle_dof_blur_px_from_defocus(focus_defocus);
    let size = (max(0.7, uniforms.particle_size_px * 0.55) + focus_blur_px) * 2.0 / uniforms.resolution.y;
    out.position = vec4f(projected.x + local.x * size / aspect, projected.y + local.y * size, 0.0, 1.0);
    out.local = local;
    out.alpha = smoothstep(uniforms.trail_threshold, uniforms.trail_threshold + 0.2, density) * uniforms.trail_opacity * 0.72;
    out.density = density;
    out.depth = projected.z;
    out.species = fract(f32(x + y * 2u + z * 3u) * 0.37);
    out.uv = vec2f(0.0);
    out.mode = 3.0;
    out.color_hsv = vec4f(0.0);
    out.defocus = focus_defocus;
    out.sprite_blur = particle_sprite_blur_amount(focus_blur_px, max(0.7, uniforms.particle_size_px * 0.55));
    return out;
  }
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -3.0),
    vec2f(-1.0, 1.0),
    vec2f(3.0, 1.0)
  );
  let position = positions[vertex_index];
  out.position = vec4f(position, 0.0, 1.0);
  out.local = vec2f(0.0);
  out.alpha = 1.0;
  out.density = 0.0;
  out.depth = 0.0;
  out.species = 0.0;
  out.uv = position * 0.5 + vec2f(0.5);
  out.mode = 0.0;
  out.color_hsv = vec4f(0.0);
  out.defocus = 0.0;
  out.sprite_blur = 0.0;
  return out;
}

@fragment
fn volume_fs(in: SplatOut) -> @location(0) vec4f {
  if (in.mode > 2.5) {
    let d = dot(in.local, in.local);
    let glow = exp(-d * 2.8);
    let color = palette(uniforms.palette, in.density, in.depth, in.species);
    let intensity = glow * in.alpha * fog_gain();
    return vec4f(color * intensity, clamp(intensity, 0.0, 1.0));
  }
  if (uniforms.render_layer == 1u) {
    return vec4f(0.0);
  }
  let aspect = uniforms.resolution.x / uniforms.resolution.y;
  var uv = in.uv * 2.0 - vec2f(1.0);
  uv.x = uv.x * aspect;
  let focal = uniforms.focal_k;
  let origin = camera_to_world(vec3f(-uniforms.camera_pan_x, -uniforms.camera_pan_y, -uniforms.distance));
  let direction = normalize(camera_to_world(vec3f(uv.x / focal, uv.y / focal, 1.0)));
  let hit = intersect_box(origin, direction);
  var t = max(0.0, hit.x);
  let t_end = hit.y;
  if (t_end <= t) {
    return vec4f(0.0);
  }
  let steps = clamp(uniforms.ray_steps, 8u, 512u);
  let dt = (t_end - t) / f32(steps);
  let pixel = in.uv * uniforms.resolution;
  let jitter = select(hash21(pixel), interleaved_gradient_noise(pixel, uniforms.fog_frame), uniforms.fog_blue_noise == 1u);
  let load = f32(uniforms.particle_count) / max(1.0, f32(uniforms.voxel_count));
  let maturity = smoothstep(32.0, 256.0, f32(uniforms.timestep));
  let density_weight = mix(130.0, 64.0, maturity);
  let flow_weight = mix(68.0, 42.0, maturity);
  let adaptive_threshold = uniforms.trail_threshold + maturity * max(0.0, log2(max(1.0, load)) * 0.018);
  let shape_power = 1.0 + uniforms.filament * 0.75;
  let inv_log5 = 1.0 / log2(5.0);
  let alpha_scale = uniforms.trail_opacity * (dt * 8.4);
  var color = vec3f(0.0);
  var alpha = 0.0;
  var marched = 0u;
  var ray_t = t + dt * jitter;
  for (var step = 0u; step < 512u; step = step + 1u) {
    if (marched >= steps || alpha > 0.965) {
      break;
    }
    let position = origin + direction * ray_t;
    if (uniforms.domain_shape != 0u && domain_sdf(position, uniforms.domain_shape) > 0.0) {
      ray_t = ray_t + dt;
      marched = marched + 1u;
      continue;
    }
    let raw = sample_volume(position);
    let flow = length(raw.xyz);
    let flow_signal = flow * flow_weight;
    let raw_signal = max(0.0, raw.w * density_weight + flow_signal);
    let compressed = log2(1.0 + raw_signal * uniforms.density) * inv_log5;
    if (uniforms.empty_space_skip == 1u && compressed <= adaptive_threshold + uniforms.empty_space_threshold) {
      let stride = min(u32(max(1.0, uniforms.empty_space_stride)), steps - marched);
      ray_t = ray_t + dt * f32(stride);
      marched = marched + stride;
      continue;
    }
    let signal = max(0.0, compressed - adaptive_threshold);
    let visible_signal = smoothstep(0.018, 1.0, signal);
    let shaped_signal = pow(visible_signal, shape_power);
    let normalized_depth = clamp((ray_t - t) / max(0.0001, t_end - t), 0.0, 1.0);
    let dof_focus_gain = 1.0 / (1.0 + abs(normalized_depth - uniforms.focus_distance) * uniforms.aperture * 4.0);
    let focus_gain = select(1.0, dof_focus_gain, uniforms.dof_enabled == 1u);
    let sample_alpha = clamp(shaped_signal * alpha_scale * focus_gain, 0.0, 1.0);
    if (sample_alpha > 0.000001) {
      let color_signal = clamp(shaped_signal * 1.25, 0.0, 1.0);
      var sample_color = trail_stable_color(raw, color_signal);
      if (uniforms.trail_color_mode == 1u) {
        sample_color = trail_vector_color(raw.xyz, color_signal);
      } else if (uniforms.trail_color_mode == 2u) {
        sample_color = trail_thermal_color(raw, color_signal);
      } else if (uniforms.trail_color_mode == 3u) {
        sample_color = trail_tint_color(color_signal, flow);
      }
      if (uniforms.trail_color_mode != 3u) {
        sample_color = apply_tint(sample_color, fog_tint_color(), 0.42);
      }
      color = color + (1.0 - alpha) * sample_color * sample_alpha;
      alpha = alpha + (1.0 - alpha) * sample_alpha;
    }
    ray_t = ray_t + dt;
    marched = marched + 1u;
  }
  color = max(color * fog_gain(), vec3f(0.0));
  return vec4f(color, select(0.0, 1.0, alpha > 0.0001));
}
`;

const liveFogCompositeShader = /* wgsl */ `
struct RenderUniforms {
  resolution: vec2f,
  width: u32,
  height: u32,
  depth: u32,
  particle_count: u32,
  voxel_count: u32,
  timestep: u32,
  density: f32,
  exposure: f32,
  focus_distance: f32,
  aperture: f32,
  overlay: u32,
  palette: u32,
  filament: f32,
  yaw_cos: f32,
  yaw_sin: f32,
  pitch_cos: f32,
  pitch_sin: f32,
  distance: f32,
  ray_steps: u32,
  fog_step_scale: f32,
  fog_temporal_blend: f32,
  fog_blue_noise: u32,
  fog_frame: u32,
  field_texture_mode: u32,
  empty_space_skip: u32,
  empty_space_threshold: f32,
  empty_space_stride: f32,
  particle_size_px: f32,
  particle_min_px: f32,
  particle_max_px: f32,
  particle_opacity: f32,
  particle_blend_mode: u32,
  particle_density_cutoff: f32,
  particle_density_radius: f32,
  trail_opacity: f32,
  trail_threshold: f32,
  render_layer: u32,
  camera_pan_x: f32,
  camera_pan_y: f32,
  trail_color_mode: u32,
  fog_tint_r: f32,
  fog_tint_g: f32,
  fog_tint_b: f32,
  particle_tint_r: f32,
  particle_tint_g: f32,
  particle_tint_b: f32,
  scene_brightness: f32,
  particle_brightness: f32,
  fog_brightness: f32,
  particle_color_mode: u32,
  particle_velocity_stretch: u32,
  particle_stretch: f32,
  particle_gradient_sensitivity: f32,
  focal_k: f32,
  dof_blur: f32,
  dof_debug: u32,
  render_lerp_t: f32,
  dof_enabled: u32,
  density_pass_strength: f32,
  density_small_scale: f32,
  density_large_scale: f32,
  density_large_threshold: f32,
  density_contrast_gain: f32,
  density_contrast_balance: f32,
  density_emission_power: f32,
  density_occlusion: f32,
  cohorts: u32,
  particle_slow_cutoff: f32,
  particle_glow_core: f32,
  particle_hot_core: f32,
  accumulation_strength: f32,
  accumulation_radius: f32,
  accumulation_curve: f32,
  accumulation_memory: f32,
  accumulation_noise_reject: f32,
  particle_density_reference: f32,
  particle_density_normalize: f32,
  particle_density_softness: f32,
  particle_support_mask: f32,
  particle_support_radius: f32,
  particle_support_neighbors: f32,
  particle_support_flow: f32,
  particle_support_grid_size: u32,
  particle_stretch_min: f32,
  particle_stretch_speed: f32,
  particle_speed_cutoff: f32,
  variation_master: f32,
  variation_time: f32,
  variation_drift: f32,
  variation_noise_mix: f32,
  variation_freq: f32,
  variation_octaves: u32,
  variation_gain: f32,
  variation_lacunarity: f32,
  variation_size_amount: f32,
  variation_size_curve: f32,
  variation_size_min: f32,
  variation_size_max: f32,
  variation_bright_amount: f32,
  variation_bright_curve: f32,
  variation_bright_min: f32,
  variation_bright_max: f32,
  variation_opacity_amount: f32,
  variation_opacity_curve: f32,
  variation_opacity_min: f32,
  variation_opacity_max: f32,
  variation_color_amount: f32,
  variation_color_curve: f32,
  variation_color_min: f32,
  variation_color_max: f32,
  domain_shape: u32,
  audio_low: f32,
  audio_mid: f32,
  audio_high: f32,
};

@group(0) @binding(0) var fog_current: texture_2d<f32>;
@group(0) @binding(1) var fog_history: texture_2d<f32>;
@group(0) @binding(2) var fog_sampler: sampler;
@group(0) @binding(3) var<uniform> uniforms: RenderUniforms;

struct FullscreenOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn fullscreen_vs(@builtin(vertex_index) vertex_index: u32) -> FullscreenOut {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -3.0),
    vec2f(-1.0, 1.0),
    vec2f(3.0, 1.0)
  );
  let position = positions[vertex_index];
  var out: FullscreenOut;
  out.position = vec4f(position, 0.0, 1.0);
  out.uv = position * 0.5 + vec2f(0.5);
  return out;
}

@fragment
fn fog_composite_fs(in: FullscreenOut) -> @location(0) vec4f {
  let sample_uv = vec2f(in.uv.x, 1.0 - in.uv.y);
  let current = textureSample(fog_current, fog_sampler, sample_uv);
  let history = textureSample(fog_history, fog_sampler, sample_uv);
  let blend = select(0.0, clamp(uniforms.fog_temporal_blend, 0.0, 0.96), uniforms.fog_frame > 2u);
  let color = current.rgb * (1.0 - blend) + history.rgb * blend;
  let alpha = max(current.a, history.a * blend);
  return vec4f(color, alpha);
}
`;

// Post-process: AgX view transform (Benjamin Wrensch's minimal AgX, MIT) ported to WGSL so
// the live HDR output preserves a filmic AgX-style grade instead of clipping linear.
const livePostShader = /* wgsl */ `
struct FullscreenOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn fullscreen_vs(@builtin(vertex_index) vertex_index: u32) -> FullscreenOut {
  var positions = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(-1.0, 1.0), vec2f(3.0, 1.0));
  let position = positions[vertex_index];
  var out: FullscreenOut;
  out.position = vec4f(position, 0.0, 1.0);
  // Flip V: the scene is rendered into an offscreen texture top-row-first, and this
  // fullscreen sampling round-trip would otherwise invert it vertically. Applied uniformly
  // to every post pass (prefilter/downsample/upsample/composite) so the bloom chain stays
  // internally consistent and only the net scene orientation is corrected.
  out.uv = vec2f(position.x * 0.5 + 0.5, 0.5 - position.y * 0.5);
  return out;
}

struct PostUniforms {
  bloom_strength: f32,
  bloom_threshold: f32,
  bloom_radius: f32,
  level_count: f32,
  tint: vec4f,
  grade: vec4f,
  // Per-octave weights reconstructing the Spencer 1995 fog-glow PSF (w0..w3, then w4..w7).
  weights_a: vec4f,
  weights_b: vec4f,
  // Lens FX: x = chromatic aberration strength, y = vignette strength,
  // z = vignette softness (0 = from centre .. ~1 = corners only), w = reserved.
  lens: vec4f,
  // Anamorphic flare (reuses the bloom bright mips): x = strength, y = length,
  // z = vertical/star mix (0 = horizontal only .. 1 = full star), w = reserved.
  streak: vec4f,
  // Flare shaping: x = vertical height (independent of length), y = luminance cutoff
  // (only cores above it streak), z/w = reserved.
  flare: vec4f,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;
@group(0) @binding(2) var<uniform> post: PostUniforms;
@group(0) @binding(3) var bloom_tex: texture_2d<f32>;
// Octave mips for the fog-glow composite (Spencer-weighted sum). bloom_tex is octave 0.
@group(0) @binding(4) var bloom_m1: texture_2d<f32>;
@group(0) @binding(5) var bloom_m2: texture_2d<f32>;
@group(0) @binding(6) var bloom_m3: texture_2d<f32>;
@group(0) @binding(7) var bloom_m4: texture_2d<f32>;
@group(0) @binding(8) var bloom_m5: texture_2d<f32>;
@group(0) @binding(9) var bloom_m6: texture_2d<f32>;
@group(0) @binding(10) var bloom_m7: texture_2d<f32>;

fn src_texel() -> vec2f {
  return 1.0 / vec2f(textureDimensions(src_tex));
}

fn tap(uv: vec2f) -> vec3f {
  return textureSampleLevel(src_tex, scene_sampler, uv, 0.0).rgb;
}

// Soft-knee bright-pass (matches the realtime-compositor / Eevee bloom prefilter).
fn knee_prefilter(c: vec3f) -> vec3f {
  let br = max(c.r, max(c.g, c.b));
  let knee = post.bloom_threshold * 0.5 + 1e-5;
  var soft = clamp(br - post.bloom_threshold + knee, 0.0, 2.0 * knee);
  soft = soft * soft / (4.0 * knee + 1e-5);
  let contrib = max(soft, br - post.bloom_threshold) / max(br, 1e-5);
  return c * contrib;
}

fn agx_contrast_approx(x: vec3f) -> vec3f {
  let x2 = x * x;
  let x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

fn agx(val_in: vec3f) -> vec3f {
  let agx_mat = mat3x3<f32>(
    0.842479062253094, 0.0784335999999992, 0.0792237451477643,
    0.0423282422610123, 0.878468636469772, 0.0791661274605434,
    0.0423756549057051, 0.0784336, 0.879142973793104);
  let min_ev = -12.47393;
  let max_ev = 4.026069;
  var val = agx_mat * val_in;
  val = clamp(log2(max(val, vec3f(1e-10))), vec3f(min_ev), vec3f(max_ev));
  val = (val - min_ev) / (max_ev - min_ev);
  val = agx_contrast_approx(val);
  return val;
}

fn agx_eotf(val: vec3f) -> vec3f {
  let agx_mat_inv = mat3x3<f32>(
    1.19687900512017, -0.0980208811401368, -0.0990297440797205,
    -0.0528968517574562, 1.15190312990417, -0.0989611768448433,
    -0.0529716355144438, -0.0980434501171241, 1.15107367264116);
  var v = agx_mat_inv * val;
  // Back to display-linear for the linear float HDR canvas.
  v = pow(max(v, vec3f(0.0)), vec3f(2.2));
  return v;
}

fn apply_color_grade(val: vec3f) -> vec3f {
  let luma = dot(val, vec3f(0.2126, 0.7152, 0.0722));
  var graded = mix(vec3f(luma), val, clamp(post.grade.x, 0.0, 2.0));
  graded = (graded - vec3f(0.5)) * clamp(post.grade.y, 0.5, 1.8) + vec3f(0.5);
  return clamp(graded, vec3f(0.0), vec3f(1.0));
}

// HDR-capture tone-map — CONFORMANCE BY CONSTRUCTION. An earlier attempt replaced AGX with a
// different curve (luminance Reinhard + path-to-white); that changed exposure/contrast/highlight
// rolloff, so the export no longer matched the live look ("more bloom / different exposure"). The fix
// is to NOT invent a new operator: the BASE here is the *exact* live SDR transform — same agx() +
// apply_color_grade() (incl. contrast + the AgX inset-matrix path-to-white) + agx_eotf() — so mids,
// shadows, exposure, contrast, saturation and bloom balance are pixel-identical to what the user sees
// live. We then ADD BACK ONLY the highlight energy AGX threw away: AgX clamps scene luminance above
// 2^max_ev (~16.3) to display white, discarding all highlight headroom. We measure how far the scene
// exceeded that clip point and lift the (already near-white, AGX-desaturated) bright pixels above 1.0
// into real HDR. A smoothstep "bright" gate keeps mids/shadows at extension 0, so they stay
// bit-conformant with live; only genuine highlights gain nits. headroom (post.grade.w) sets how many
// ×paper-white the hottest cores may reach (e.g. headroom 4 + export paper-white 203nit => ~1000nit).
fn tonemap_hdr(val: vec3f, headroom: f32) -> vec3f {
  let combined = max(val, vec3f(0.0));
  let sdr = agx_eotf(apply_color_grade(agx(combined)));   // EXACT live look, display-linear [0,1]
  // Extend the highlights AGX pins to white. Threshold = where the AGX base actually saturates
  // (scene luma ~5), NOT AGX's internal log-clamp (16.3) — gating at 16.3 left a flat WHITE PLATEAU
  // from scene ~6..16 (verified by sampling: out luma pinned at 1.000) that read as clipping, then a
  // harsh jump. Ramp in over tau stops with 1-exp(-(stops/tau)^2): slope 0 at the threshold so there
  // is no seam/discontinuity, then a smooth log-space S-curve that keeps gradation in the cores (no
  // flat-topping) and soft-caps at headroom. Mids (scene < threshold) get ext=0 => bit-identical to live.
  let scene_l = max(dot(combined, vec3f(0.2126, 0.7152, 0.0722)), 1e-6);
  let thresh = 5.0;
  let tau = 3.5;
  let r = max(log2(scene_l) - log2(thresh), 0.0) / tau;
  let ext = max(headroom, 0.0) * (1.0 - exp(-r * r));
  let bright = smoothstep(0.6, 1.0, dot(sdr, vec3f(0.2126, 0.7152, 0.0722)));
  return sdr * (1.0 + ext * bright);
}

@fragment
fn prefilter_fs(in: FullscreenOut) -> @location(0) vec4f {
  // 13-tap Karis-averaged downsample of the scene, then the bright-pass (firefly-safe).
  let t = src_texel();
  let a = tap(in.uv + t * vec2f(-2.0, 2.0));
  let b = tap(in.uv + t * vec2f(0.0, 2.0));
  let c = tap(in.uv + t * vec2f(2.0, 2.0));
  let d = tap(in.uv + t * vec2f(-2.0, 0.0));
  let e = tap(in.uv);
  let f = tap(in.uv + t * vec2f(2.0, 0.0));
  let g = tap(in.uv + t * vec2f(-2.0, -2.0));
  let h = tap(in.uv + t * vec2f(0.0, -2.0));
  let i = tap(in.uv + t * vec2f(2.0, -2.0));
  let j = tap(in.uv + t * vec2f(-1.0, 1.0));
  let k = tap(in.uv + t * vec2f(1.0, 1.0));
  let l = tap(in.uv + t * vec2f(-1.0, -1.0));
  let m = tap(in.uv + t * vec2f(1.0, -1.0));
  let down = (j + k + l + m) * 0.5 * 0.25 + (a + b + d + e) * 0.125 * 0.25 + (b + c + e + f) * 0.125 * 0.25 + (d + e + g + h) * 0.125 * 0.25 + (e + f + h + i) * 0.125 * 0.25;
  return vec4f(knee_prefilter(max(down, vec3f(0.0))), 1.0);
}

@fragment
fn downsample_fs(in: FullscreenOut) -> @location(0) vec4f {
  _ = post.bloom_radius; // keep the {tex, sampler, uniform} bind-group layout shared across bloom passes
  let t = src_texel();
  let a = tap(in.uv + t * vec2f(-2.0, 2.0));
  let b = tap(in.uv + t * vec2f(0.0, 2.0));
  let c = tap(in.uv + t * vec2f(2.0, 2.0));
  let d = tap(in.uv + t * vec2f(-2.0, 0.0));
  let e = tap(in.uv);
  let f = tap(in.uv + t * vec2f(2.0, 0.0));
  let g = tap(in.uv + t * vec2f(-2.0, -2.0));
  let h = tap(in.uv + t * vec2f(0.0, -2.0));
  let i = tap(in.uv + t * vec2f(2.0, -2.0));
  let j = tap(in.uv + t * vec2f(-1.0, 1.0));
  let k = tap(in.uv + t * vec2f(1.0, 1.0));
  let l = tap(in.uv + t * vec2f(-1.0, -1.0));
  let m = tap(in.uv + t * vec2f(1.0, -1.0));
  let down = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
  return vec4f(down, 1.0);
}

@fragment
fn upsample_fs(in: FullscreenOut) -> @location(0) vec4f {
  // 3x3 tent upsample; additive blend on the pipeline accumulates up the pyramid.
  let t = src_texel() * post.bloom_radius;
  var sum = tap(in.uv) * 4.0;
  sum = sum + (tap(in.uv + t * vec2f(-1.0, 0.0)) + tap(in.uv + t * vec2f(1.0, 0.0)) + tap(in.uv + t * vec2f(0.0, -1.0)) + tap(in.uv + t * vec2f(0.0, 1.0))) * 2.0;
  sum = sum + (tap(in.uv + t * vec2f(-1.0, -1.0)) + tap(in.uv + t * vec2f(1.0, -1.0)) + tap(in.uv + t * vec2f(-1.0, 1.0)) + tap(in.uv + t * vec2f(1.0, 1.0)));
  return vec4f(sum / 16.0, 1.0);
}

// Chromatic aberration: split the R/B channels along the radial (center->edge) direction so
// the effect is strongest at the frame edges and zero at the center. lens.x = 0 samples one
// texel for all channels => bit-identical to the original single .rgb tap.
fn sample_scene_ca(uv: vec2f) -> vec3f {
  let ca = post.lens.x;
  if (ca <= 0.0001) {
    return textureSampleLevel(src_tex, scene_sampler, uv, 0.0).rgb;
  }
  // Real-lens chromatic aberration: the R/B split is negligible across the centre and
  // ramps up sharply toward the frame edges (concentrated on the outside, like vignette),
  // rather than a uniform full-screen split. The edge ramp (radius^2.5) keeps the middle clean.
  let d = uv - vec2f(0.5);
  let rad = clamp(length(d) * 2.0, 0.0, 1.5);
  let edge = pow(rad, 2.5);
  let off = d * ca * 0.06 * edge;
  let cr = textureSampleLevel(src_tex, scene_sampler, uv + off, 0.0).r;
  let cg = textureSampleLevel(src_tex, scene_sampler, uv, 0.0).g;
  let cb = textureSampleLevel(src_tex, scene_sampler, uv - off, 0.0).b;
  return vec3f(cr, cg, cb);
}

// Radial vignette applied to the final graded color. lens.y = 0 returns the color unchanged.
fn apply_vignette(color: vec3f, uv: vec2f) -> vec3f {
  let strength = post.lens.y;
  if (strength <= 0.0001) {
    return color;
  }
  let softness = clamp(post.lens.z, 0.0, 0.999);
  let dims = vec2f(textureDimensions(src_tex));
  let aspect = dims.x / max(1.0, dims.y);
  let p = (uv - vec2f(0.5)) * vec2f(aspect, 1.0);
  let d = length(p) / max(0.0001, length(vec2f(aspect, 1.0) * 0.5));
  let v = 1.0 - strength * smoothstep(softness, 1.0, d);
  return color * clamp(v, 0.0, 1.0);
}

// Anamorphic lens flare: smear the already-built bloom bright mip horizontally (and, scaled by
// the star mix, vertically) to grow cinematic streaks off bright cores. Uniform early-return at
// strength 0 means the GPU never runs the tap loop when the effect is off (zero cost). Requires
// the bloom path (samples bloom_m1), so it contributes nothing when bloom is disabled.
fn anamorphic_streak(uv: vec2f) -> vec3f {
  let strength = post.streak.x;
  if (strength <= 0.0001 || post.level_count < 2.5) {
    return vec3f(0.0);
  }
  let span = 0.16 * clamp(post.streak.y, 0.0, 1.0);   // horizontal length
  let vmix = clamp(post.streak.z, 0.0, 1.0);          // star / vertical amount
  let vspan = 0.16 * clamp(post.flare.x, 0.0, 1.0);   // vertical height, independent of length
  let cutoff = max(0.0, post.flare.y);                // luminance floor: only bright cores streak
  var acc = vec3f(0.0);
  var wsum = 0.0;
  // 25 taps/axis off the eighth-res bloom_m2 read as a smooth streak. Vertical taps run only when
  // the star mix is up and use their own height span, so a streak can be long yet short ("not too
  // tall"). The cutoff subtracts a floor so dim haze doesn't smear, leaving clean flares.
  for (var i = -12; i <= 12; i = i + 1) {
    let f = f32(i) / 12.0;
    let w = exp(-abs(f) * 2.5);
    let hs = max(textureSampleLevel(bloom_m2, scene_sampler, uv + vec2f(f * span, 0.0), 0.0).rgb - vec3f(cutoff), vec3f(0.0));
    acc = acc + hs * w;
    wsum = wsum + w;
    if (vmix > 0.001) {
      let vs = max(textureSampleLevel(bloom_m2, scene_sampler, uv + vec2f(0.0, f * vspan), 0.0).rgb - vec3f(cutoff), vec3f(0.0));
      acc = acc + vs * w * vmix;
      wsum = wsum + w * vmix;
    }
  }
  return acc / max(0.0001, wsum) * strength;
}

fn despeckle_scene(uv: vec2f, center: vec3f) -> vec3f {
  let strength = post.flare.z;
  if (strength <= 0.0001) {
    return center;
  }
  let texel = 1.0 / vec2f(textureDimensions(src_tex));
  var nmax = vec3f(0.0);
  for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      if (dx == 0 && dy == 0) { continue; }
      let s = textureSampleLevel(src_tex, scene_sampler, uv + vec2f(f32(dx), f32(dy)) * texel, 0.0).rgb;
      nmax = max(nmax, s);
    }
  }
  let cl = max(center.r, max(center.g, center.b));
  let nl = max(nmax.r, max(nmax.g, nmax.b));
  let isolation = smoothstep(0.12, 0.45, (cl - nl) / max(cl, 0.0001));
  return mix(center, min(center, nmax), strength * isolation);
}

@fragment
fn composite_fs(in: FullscreenOut) -> @location(0) vec4f {
  let scene = despeckle_scene(in.uv, sample_scene_ca(in.uv));
  var bloom = textureSampleLevel(bloom_tex, scene_sampler, in.uv, 0.0).rgb * post.weights_a.x;
  if (post.level_count > 1.5) { bloom = bloom + textureSampleLevel(bloom_m1, scene_sampler, in.uv, 0.0).rgb * post.weights_a.y; }
  if (post.level_count > 2.5) { bloom = bloom + textureSampleLevel(bloom_m2, scene_sampler, in.uv, 0.0).rgb * post.weights_a.z; }
  if (post.level_count > 3.5) { bloom = bloom + textureSampleLevel(bloom_m3, scene_sampler, in.uv, 0.0).rgb * post.weights_a.w; }
  if (post.level_count > 4.5) { bloom = bloom + textureSampleLevel(bloom_m4, scene_sampler, in.uv, 0.0).rgb * post.weights_b.x; }
  if (post.level_count > 5.5) { bloom = bloom + textureSampleLevel(bloom_m5, scene_sampler, in.uv, 0.0).rgb * post.weights_b.y; }
  if (post.level_count > 6.5) { bloom = bloom + textureSampleLevel(bloom_m6, scene_sampler, in.uv, 0.0).rgb * post.weights_b.z; }
  if (post.level_count > 7.5) { bloom = bloom + textureSampleLevel(bloom_m7, scene_sampler, in.uv, 0.0).rgb * post.weights_b.w; }
  let combined = scene + bloom * post.bloom_strength * post.tint.rgb + anamorphic_streak(in.uv) * post.tint.rgb;
  if (post.grade.w > 0.0) {
    return vec4f(apply_vignette(tonemap_hdr(combined, post.grade.w), in.uv), 1.0);
  }
  let graded = agx_eotf(apply_color_grade(agx(max(combined, vec3f(0.0)))));
  return vec4f(apply_vignette(graded, in.uv), 1.0);
}

@fragment
fn composite_no_bloom_fs(in: FullscreenOut) -> @location(0) vec4f {
  let scene = despeckle_scene(in.uv, sample_scene_ca(in.uv));
  if (post.grade.w > 0.0) {
    return vec4f(apply_vignette(tonemap_hdr(scene, post.grade.w), in.uv), 1.0);
  }
  let graded = agx_eotf(apply_color_grade(agx(max(scene, vec3f(0.0)))));
  return vec4f(apply_vignette(graded, in.uv), 1.0);
}
`;

// Bindings + per-particle fade/density helpers shared by the classic splat vertex shaders and
// the splat-prepare compute pass (which does the same per-particle work once instead of per corner).
const liveParticleVsCommon = /* wgsl */ `
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
// Previous whole-step particle state, for display interpolation (binding 2). Equals the
// current visual state when paused/frozen because render_lerp_t = 1 selects the current buffer.
@group(0) @binding(2) var<storage, read> particles_prev: array<Particle>;
@group(0) @binding(3) var<storage, read> particle_density_field_values: array<vec4f>;
@group(0) @binding(4) var<storage, read> particle_support_values: array<f32>;
@group(0) @binding(5) var<storage, read> particle_active_indices: array<u32>;
// Density-cutoff signal precomputed once per particle by the cutoff prepass (binding 6);
// read by particle_vs only when uniforms.particle_cutoff_prepass == 1.
@group(0) @binding(6) var<storage, read> particle_cutoff_signal: array<f32>;
// Persistent per-particle ids (permuted alongside the spatial sort) so identity-derived
// visual randomness (variation) sticks to the particle, not to its buffer slot.
@group(0) @binding(7) var<storage, read> particle_persistent_ids: array<u32>;

fn culled_splat(local_density: f32) -> SplatOut {
  var out: SplatOut;
  out.position = vec4f(2.0, 2.0, 0.0, 1.0);
  out.local = vec2f(0.0);
  out.alpha = 0.0;
  out.density = local_density;
  out.depth = 0.0;
  out.species = 0.0;
  out.uv = vec2f(0.0);
  out.mode = 1.0;
  out.color_hsv = vec4f(0.0);
  out.defocus = 0.0;
  out.sprite_blur = 0.0;
  out.bright_mul = 1.0;
  return out;
}

fn particle_density_field_signal(raw: vec4f) -> f32 {
  return max(0.0, raw.w + length(raw.xyz) * 0.25);
}

fn particle_density_field_cell(x: i32, y: i32, z: i32) -> u32 {
  let w = i32(uniforms.width);
  let h = i32(uniforms.height);
  let d = i32(uniforms.depth);
  let cx = clamp(x, 0, w - 1);
  let cy = clamp(y, 0, h - 1);
  let cz = clamp(z, 0, d - 1);
  return u32((cz * h + cy) * w + cx);
}

fn particle_density_field_at(position: vec3f) -> vec4f {
  if (any(position < vec3f(-1.0)) || any(position > vec3f(1.0))) {
    return vec4f(0.0);
  }
  let dims = vec3f(f32(uniforms.width), f32(uniforms.height), f32(uniforms.depth));
  let grid = (position + vec3f(1.0)) * 0.5 * dims - vec3f(0.5);
  let base = vec3i(floor(grid));
  let t = fract(grid);
  var accum = vec4f(0.0);
  for (var dz = 0i; dz <= 1i; dz = dz + 1i) {
    for (var dy = 0i; dy <= 1i; dy = dy + 1i) {
      for (var dx = 0i; dx <= 1i; dx = dx + 1i) {
        let weight = mix(1.0 - t.x, t.x, f32(dx)) * mix(1.0 - t.y, t.y, f32(dy)) * mix(1.0 - t.z, t.z, f32(dz));
        accum = accum + particle_density_field_values[particle_density_field_cell(base.x + dx, base.y + dy, base.z + dz)] * weight;
      }
    }
  }
  return accum;
}

fn particle_density_signal_at(position: vec3f) -> f32 {
  return particle_density_field_signal(particle_density_field_at(position));
}

fn particle_cutoff_support_signal(position: vec3f, radius: f32) -> f32 {
  let center = particle_density_signal_at(position);
  if (radius <= 0.00001) {
    return center;
  }
  let axis = (
    particle_density_signal_at(position + vec3f(radius, 0.0, 0.0)) +
    particle_density_signal_at(position - vec3f(radius, 0.0, 0.0)) +
    particle_density_signal_at(position + vec3f(0.0, radius, 0.0)) +
    particle_density_signal_at(position - vec3f(0.0, radius, 0.0)) +
    particle_density_signal_at(position + vec3f(0.0, 0.0, radius)) +
    particle_density_signal_at(position - vec3f(0.0, 0.0, radius))
  ) / 6.0;
  let diagonal_step = radius * 0.57735027;
  let diagonal = (
    particle_density_signal_at(position + vec3f(diagonal_step, diagonal_step, diagonal_step)) +
    particle_density_signal_at(position + vec3f(diagonal_step, diagonal_step, -diagonal_step)) +
    particle_density_signal_at(position + vec3f(diagonal_step, -diagonal_step, diagonal_step)) +
    particle_density_signal_at(position + vec3f(diagonal_step, -diagonal_step, -diagonal_step)) +
    particle_density_signal_at(position + vec3f(-diagonal_step, diagonal_step, diagonal_step)) +
    particle_density_signal_at(position + vec3f(-diagonal_step, diagonal_step, -diagonal_step)) +
    particle_density_signal_at(position + vec3f(-diagonal_step, -diagonal_step, diagonal_step)) +
    particle_density_signal_at(position + vec3f(-diagonal_step, -diagonal_step, -diagonal_step))
  ) * 0.125;
  return center * 0.38 + axis * 0.28 + diagonal * 0.34;
}

fn particle_normalized_density_fade(local_density: f32) -> f32 {
  let amount = clamp(uniforms.particle_density_normalize, 0.0, 1.0);
  if (amount <= 0.0001 || uniforms.particle_density_reference <= 0.000001) {
    return 1.0;
  }
  let normalized_density = local_density / max(0.000001, uniforms.particle_density_reference);
  let threshold = mix(0.06, 1.08, amount);
  let softness = max(0.01, uniforms.particle_density_softness);
  return smoothstep(threshold, threshold + softness, normalized_density);
}

// Half-saturation for the field-based support membership (matches the sim's canonical DENSITY_HALF).
const SUPPORT_FIELD_HALF = 0.0001;

fn particle_support_fade(index: u32, position: vec3f) -> f32 {
  let amount = clamp(uniforms.particle_support_mask, 0.0, 1.0);
  if (amount <= 0.0001) {
    return 1.0;
  }
  let support = clamp(particle_support_values[index], 0.0, 1.0);
  let cutoff = mix(0.05, 0.74, amount);
  let softness = mix(0.34, 0.18, amount);
  let grid_fade = smoothstep(cutoff, min(1.0, cutoff + softness), support);
  let smoothing = clamp(uniforms.particle_support_smoothing, 0.0, 1.0);
  if (smoothing <= 0.0001) {
    return grid_fade;
  }
  let radius = clamp(uniforms.particle_support_radius, 0.35, 1.75);
  let field_sig = particle_cutoff_support_signal(position, radius);
  let k = SUPPORT_FIELD_HALF * mix(0.15, 1.5, amount);
  let field_fade = field_sig / (field_sig + k);
  return mix(grid_fade, field_fade, smoothing);
}

fn particle_haze_fade(position: vec3f) -> f32 {
  let amount = clamp(uniforms.particle_haze_cull, 0.0, 1.0);
  if (amount <= 0.0001) {
    return 1.0;
  }
  let density = particle_density_field_at(position).w;
  let density_ref = select(0.00018, uniforms.particle_density_reference, uniforms.particle_density_reference > 0.000001);
  let structure = density / density_ref;
  let threshold = amount * 0.15;
  return smoothstep(threshold * 0.5, threshold, structure);
}

fn particle_instance_index(instance_index: u32) -> u32 {
  if (uniforms.particle_support_mask > 0.0001) {
    return particle_active_indices[instance_index];
  }
  return instance_index;
}
`;

// Compact per-particle splat record produced by the prepare pass: 32 bytes. Center stays f32
// for subpixel-exact placement; axes/color/fades ride in f16 pairs (sub-0.1% error on
// pixel-scale quantities). Culled particles store an off-screen center with zero axes, so all
// four corner vertices land on one clipped point and rasterization drops them.
const liveSplatRecordCommon = /* wgsl */ `
struct SplatRecord {
  center: vec2f,
  axis: vec2u,       // pack2x16float: .x = axis_a.xy, .y = axis_b.xy (radius baked in)
  packed_a: vec2u,   // .x = (alpha, species), .y = (speed, bright_mul)
  packed_hsv: vec2u, // .x = (hue, sat), .y = (val, w)
};

fn splat_record_off_screen() -> SplatRecord {
  var rec: SplatRecord;
  rec.center = vec2f(2.0, 2.0);
  rec.axis = vec2u(0u);
  rec.packed_a = vec2u(0u);
  rec.packed_hsv = vec2u(0u);
  return rec;
}
`;

// Optimization A: evaluate the per-particle density-cutoff signal ONCE in a compute prepass
// (instead of recomputing it for all 4 billboard corners in particle_vs). Reuses the exact
// liveSplatRenderCommon + liveParticleVsCommon helpers and uniforms, so the value written here
// is bit-identical to the inline path; particle_vs reads it from binding 6 when the toggle is on.
const liveCutoffPrepassShader = /* wgsl */ `
${liveSplatRenderCommon}
${liveParticleVsCommon}
@group(0) @binding(8) var<storage, read_write> cutoff_signal_out: array<f32>;
@compute @workgroup_size(256)
fn cutoff_prepass(@builtin(global_invocation_id) gid: vec3u) {
  let index = gid.x;
  if (index >= arrayLength(&particles)) {
    return;
  }
  let particle = particles[index];
  let previous_particle = particles_prev[index];
  // Mirror particle_vs exactly: same render_lerp_t, same torus-shortest interpolation.
  let particle_lerp_t = particle_lerp_amount(previous_particle.pos_cohort.xyz, particle.pos_cohort.xyz, uniforms.render_lerp_t);
  let interp_pos = interpolate_particle_position(previous_particle.pos_cohort.xyz, particle.pos_cohort.xyz, particle_lerp_t);
  var local_density = 1.0;
  if (uniforms.particle_density_cutoff > 0.0 || uniforms.particle_density_normalize > 0.0001) {
    local_density = particle_cutoff_support_signal(interp_pos, particle_density_gate_radius());
  }
  cutoff_signal_out[index] = local_density;
}
`;

const liveParticleRenderShader = /* wgsl */ `
${liveSplatRenderCommon}
${liveParticleVsCommon}
@vertex
fn particle_vs(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> SplatOut {
  let index = particle_instance_index(instance_index);
  let particle = particles[index];
  let previous_particle = particles_prev[index];
  // Interpolate position between the previous and current whole-step state for smooth motion
  // at any sim speed. Use torus-shortest interpolation so wrapped particles do not streak
  // across the volume on fractional-speed frames.
  let particle_lerp_t = particle_lerp_amount(previous_particle.pos_cohort.xyz, particle.pos_cohort.xyz, uniforms.render_lerp_t);
  let large_motion_fade = particle_large_motion_fade(previous_particle.pos_cohort.xyz, particle.pos_cohort.xyz, uniforms.render_lerp_t);
  let interp_pos = interpolate_particle_position(previous_particle.pos_cohort.xyz, particle.pos_cohort.xyz, particle_lerp_t);
  let visual_velocity = particle_visual_velocity(previous_particle, particle, particle_lerp_t);
  let speed_cutoff_visibility = particle_speed_cutoff_visibility(visual_velocity);
  if (speed_cutoff_visibility <= 0.000001) {
    return culled_splat(0.0);
  }
  let visual_color = interpolate_particle_color(stored_particle_color(previous_particle), stored_particle_color(particle), particle_lerp_t);
  let p = vec4f(interp_pos, particle.pos_cohort.w);
  var local_density = 1.0;
  if (uniforms.particle_density_cutoff > 0.0 || uniforms.particle_density_normalize > 0.0001) {
    if (uniforms.particle_cutoff_prepass == 1u) {
      // Precomputed once per particle by the cutoff prepass (same interp_pos, same math).
      local_density = particle_cutoff_signal[index];
    } else {
      local_density = particle_cutoff_support_signal(interp_pos, particle_density_gate_radius());
    }
  }
  var density_fade = 1.0;
  if (uniforms.particle_density_cutoff > 0.0) {
    let cutoff = uniforms.particle_density_cutoff;
    density_fade = smoothstep(cutoff, cutoff * PARTICLE_CUTOFF_FADE, local_density);
    if (density_fade <= 0.000001) {
      return culled_splat(local_density);
    }
  }
  let normalized_density_fade = particle_normalized_density_fade(local_density);
  if (normalized_density_fade <= 0.000001) {
    return culled_splat(local_density);
  }
  let support_fade = particle_support_fade(index, interp_pos);
  if (support_fade <= 0.000001) {
    return culled_splat(local_density);
  }
  let particle_fade = density_fade * normalized_density_fade * support_fade * speed_cutoff_visibility * particle_haze_fade(interp_pos);
  if (particle_fade <= 0.000001) {
    return culled_splat(local_density);
  }
  let view_depth = uniforms.distance + world_to_camera(p.xyz).z;
  if (view_depth <= PARTICLE_NEAR) {
    // Particle is at or behind the camera (e.g. zoomed past it) — clip it, don't pin it to screen.
    return culled_splat(local_density);
  }
  let projected = project(p.xyz);
  let local = billboard_corner(vertex_index);
  let focus_defocus = particle_focus_defocus(projected.z);
  let focus_blur_px = particle_dof_blur_px_from_defocus(focus_defocus);
  let variation = compute_particle_variation(particle_persistent_ids[index], interp_pos);
  let base_size_px = particle_splat_base_size_px(projected.z);
  let sized_px = particle_variation_sized_px(base_size_px, variation.size_mul);
  let splat_radius = particle_splat_radius_for_size_px(sized_px + max(0.0, focus_blur_px));
  let cohort_count = max(1.0, f32(uniforms.cohorts));
  var render_local = local;
  if (uniforms.particle_velocity_stretch == 1u && max(uniforms.particle_stretch, uniforms.particle_stretch_min) > 0.001) {
    let direction = particle_velocity_screen_axis(visual_velocity);
    if (length(direction) > 0.000001) {
      let perpendicular = vec2f(-direction.y, direction.x);
      let stretch = particle_velocity_stretch_amount(visual_velocity);
      render_local = direction * local.x * (1.0 + stretch) + perpendicular * local.y;
    }
  }
  var out: SplatOut;
  out.position = vec4f(projected.x + render_local.x * splat_radius.x, projected.y + render_local.y * splat_radius.y, 0.0, 1.0);
  out.local = local;
  out.alpha = uniforms.particle_opacity * clamp(visual_color.w / 0.045, 0.0, 2.0) * particle_fade * large_motion_fade * variation.opacity_mul;
  out.density = local_density;
  out.depth = projected.z; // DOF depth metric (1.85 / view_depth), kept for palette/depth effects.
  out.species = fract(p.w / cohort_count);
  out.uv = vec2f(0.0);
  out.mode = 1.0;
  let base_gradient = fract(p.w / cohort_count + visual_color.x * 0.17);
  let varied_hue = fract(particle_gradient_coordinate(visual_velocity, base_gradient) + variation.hue_off);
  out.color_hsv = vec4f(varied_hue, visual_color.y * variation.sat_mul, visual_color.z, visual_color.w);
  out.speed = clamp(length(visual_velocity) / 0.04, 0.0, 1.0);
  out.defocus = focus_defocus;
  out.sprite_blur = particle_sprite_blur_amount(focus_blur_px, sized_px);
  out.bright_mul = variation.bright_mul * variation.val_mul;
  return out;
}

@vertex
fn particle_fast_vs(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> SplatOut {
  let index = particle_instance_index(instance_index);
  let particle = particles[index];
  let previous_particle = particles_prev[index];
  let particle_lerp_t = particle_lerp_amount(previous_particle.pos_cohort.xyz, particle.pos_cohort.xyz, uniforms.render_lerp_t);
  let large_motion_fade = particle_large_motion_fade(previous_particle.pos_cohort.xyz, particle.pos_cohort.xyz, uniforms.render_lerp_t);
  let interp_pos = interpolate_particle_position(previous_particle.pos_cohort.xyz, particle.pos_cohort.xyz, particle_lerp_t);
  let visual_velocity = particle_visual_velocity(previous_particle, particle, particle_lerp_t);
  let speed_cutoff_visibility = particle_speed_cutoff_visibility(visual_velocity);
  if (speed_cutoff_visibility <= 0.000001) {
    return culled_splat(1.0);
  }
  let support_fade = particle_support_fade(index, interp_pos);
  if (support_fade <= 0.000001) {
    return culled_splat(1.0);
  }
  let particle_fade = support_fade * speed_cutoff_visibility * particle_haze_fade(interp_pos);
  if (particle_fade <= 0.000001) {
    return culled_splat(1.0);
  }
  let view_depth = uniforms.distance + world_to_camera(interp_pos).z;
  if (view_depth <= PARTICLE_NEAR) {
    return culled_splat(1.0);
  }
  let projected = project(interp_pos);
  let local = billboard_corner(vertex_index);
  let variation = compute_particle_variation(particle_persistent_ids[index], interp_pos);
  let base_size_px = particle_splat_base_size_px(projected.z);
  let sized_px = particle_variation_sized_px(base_size_px, variation.size_mul);
  let splat_radius = particle_splat_radius_for_size_px(sized_px);
  let cohort_count = max(1.0, f32(uniforms.cohorts));
  var render_local = local;
  if (uniforms.particle_velocity_stretch == 1u && max(uniforms.particle_stretch, uniforms.particle_stretch_min) > 0.001) {
    let direction = particle_velocity_screen_axis(visual_velocity);
    if (length(direction) > 0.000001) {
      let perpendicular = vec2f(-direction.y, direction.x);
      let stretch = particle_velocity_stretch_amount(visual_velocity);
      render_local = direction * local.x * (1.0 + stretch) + perpendicular * local.y;
    }
  }
  let visual_color = interpolate_particle_color(stored_particle_color(previous_particle), stored_particle_color(particle), particle_lerp_t);
  var out: SplatOut;
  out.position = vec4f(projected.x + render_local.x * splat_radius.x, projected.y + render_local.y * splat_radius.y, 0.0, 1.0);
  out.local = local;
  out.alpha = uniforms.particle_opacity * clamp(visual_color.w / 0.045, 0.0, 2.0) * particle_fade * large_motion_fade * variation.opacity_mul;
  out.density = 1.0;
  out.depth = projected.z;
  out.species = fract(particle.pos_cohort.w / cohort_count);
  out.uv = vec2f(0.0);
  out.mode = 1.0;
  let base_gradient = fract(particle.pos_cohort.w / cohort_count + visual_color.x * 0.17);
  let varied_hue = fract(particle_gradient_coordinate(visual_velocity, base_gradient) + variation.hue_off);
  out.color_hsv = vec4f(varied_hue, visual_color.y * variation.sat_mul, visual_color.z, visual_color.w);
  out.speed = clamp(length(visual_velocity) / 0.04, 0.0, 1.0);
  out.defocus = 0.0;
  out.sprite_blur = 0.0;
  out.bright_mul = variation.bright_mul * variation.val_mul;
  return out;
}
${liveSplatFragmentShader}
`;

// Splat prepare: does the full per-particle work of particle_fast_vs exactly once per particle
// (instead of once per quad corner) and writes a compact SplatRecord. Records keep particle
// index order so alpha/opaque blend order matches the classic path bit-for-bit.
const liveSplatPrepareShader = /* wgsl */ `
${liveSplatRenderCommon}
${liveParticleVsCommon}
${liveSplatRecordCommon}
@group(0) @binding(6) var<storage, read_write> splat_records: array<SplatRecord>;

@compute @workgroup_size(256)
fn prepare_splats(@builtin(global_invocation_id) gid: vec3u) {
  let index = gid.x;
  if (index >= arrayLength(&particles)) {
    return;
  }
  let particle = particles[index];
  let previous_particle = particles_prev[index];
  let particle_lerp_t = particle_lerp_amount(previous_particle.pos_cohort.xyz, particle.pos_cohort.xyz, uniforms.render_lerp_t);
  let large_motion_fade = particle_large_motion_fade(previous_particle.pos_cohort.xyz, particle.pos_cohort.xyz, uniforms.render_lerp_t);
  let interp_pos = interpolate_particle_position(previous_particle.pos_cohort.xyz, particle.pos_cohort.xyz, particle_lerp_t);
  let visual_velocity = particle_visual_velocity(previous_particle, particle, particle_lerp_t);
  let speed_cutoff_visibility = particle_speed_cutoff_visibility(visual_velocity);
  if (speed_cutoff_visibility <= 0.000001) {
    splat_records[index] = splat_record_off_screen();
    return;
  }
  let support_fade = particle_support_fade(index, interp_pos);
  if (support_fade <= 0.000001) {
    splat_records[index] = splat_record_off_screen();
    return;
  }
  let particle_fade = support_fade * speed_cutoff_visibility * particle_haze_fade(interp_pos);
  if (particle_fade <= 0.000001) {
    splat_records[index] = splat_record_off_screen();
    return;
  }
  let view_depth = uniforms.distance + world_to_camera(interp_pos).z;
  if (view_depth <= PARTICLE_NEAR) {
    splat_records[index] = splat_record_off_screen();
    return;
  }
  let projected = project(interp_pos);
  let variation = compute_particle_variation(particle_persistent_ids[index], interp_pos);
  let base_size_px = particle_splat_base_size_px(projected.z);
  let sized_px = particle_variation_sized_px(base_size_px, variation.size_mul);
  let splat_radius = particle_splat_radius_for_size_px(sized_px);
  let cohort_count = max(1.0, f32(uniforms.cohorts));
  var axis_a = vec2f(splat_radius.x, 0.0);
  var axis_b = vec2f(0.0, splat_radius.y);
  if (uniforms.particle_velocity_stretch == 1u && max(uniforms.particle_stretch, uniforms.particle_stretch_min) > 0.001) {
    let direction = particle_velocity_screen_axis(visual_velocity);
    if (length(direction) > 0.000001) {
      let perpendicular = vec2f(-direction.y, direction.x);
      let stretch = particle_velocity_stretch_amount(visual_velocity);
      axis_a = direction * (1.0 + stretch) * splat_radius;
      axis_b = perpendicular * splat_radius;
    }
  }
  let visual_color = interpolate_particle_color(stored_particle_color(previous_particle), stored_particle_color(particle), particle_lerp_t);
  let alpha = uniforms.particle_opacity * clamp(visual_color.w / 0.045, 0.0, 2.0) * particle_fade * large_motion_fade * variation.opacity_mul;
  let species = fract(particle.pos_cohort.w / cohort_count);
  let base_gradient = fract(particle.pos_cohort.w / cohort_count + visual_color.x * 0.17);
  let varied_hue = fract(particle_gradient_coordinate(visual_velocity, base_gradient) + variation.hue_off);
  let speed = clamp(length(visual_velocity) / 0.04, 0.0, 1.0);
  var rec: SplatRecord;
  rec.center = projected.xy;
  rec.axis = vec2u(pack2x16float(axis_a), pack2x16float(axis_b));
  rec.packed_a = vec2u(
    pack2x16float(vec2f(alpha, species)),
    pack2x16float(vec2f(speed, variation.bright_mul * variation.val_mul))
  );
  rec.packed_hsv = vec2u(
    pack2x16float(vec2f(varied_hue, visual_color.y * variation.sat_mul)),
    pack2x16float(vec2f(visual_color.z, visual_color.w))
  );
  splat_records[index] = rec;
}
`;

// Slim draw path for prepared splats: the vertex shader only fetches the 32-byte record and
// offsets the billboard corner - all per-particle math already happened in prepare_splats.
const liveSplatDrawShader = /* wgsl */ `
${liveSplatRenderCommon}
${liveSplatRecordCommon}
@group(0) @binding(6) var<storage, read> splat_records: array<SplatRecord>;

@vertex
fn particle_record_vs(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> SplatOut {
  let rec = splat_records[instance_index];
  let local = billboard_corner(vertex_index);
  let axis_a = unpack2x16float(rec.axis.x);
  let axis_b = unpack2x16float(rec.axis.y);
  let alpha_species = unpack2x16float(rec.packed_a.x);
  let speed_bright = unpack2x16float(rec.packed_a.y);
  let hue_sat = unpack2x16float(rec.packed_hsv.x);
  let val_w = unpack2x16float(rec.packed_hsv.y);
  var out: SplatOut;
  out.position = vec4f(rec.center + local.x * axis_a + local.y * axis_b, 0.0, 1.0);
  out.local = local;
  out.alpha = alpha_species.x;
  out.density = 1.0;
  out.depth = 0.0;
  out.species = alpha_species.y;
  out.uv = vec2f(0.0);
  out.mode = 1.0;
  out.color_hsv = vec4f(hue_sat.x, hue_sat.y, val_w.x, val_w.y);
  out.defocus = 0.0;
  out.sprite_blur = 0.0;
  out.bright_mul = speed_bright.y;
  out.speed = speed_bright.x;
  return out;
}
${liveSplatFragmentShader}
`;

// Compute splatter: replaces rasterization of tens of millions of tiny quads (binning-bound on
// TBDR GPUs) with one thread per prepared splat that walks the quad's pixel bounding box and
// atomically accumulates fixed-point RGB. Color math is shared with the fragment path via
// splat_color(); the radial profile matches particle_sprite_profile with fwidth computed
// analytically from the inverse splat basis. Additive blend mode only.
const SPLAT_ACCUM_WGSL_SCALE = 4096.0;
const liveSplatAccumShader = /* wgsl */ `
${liveSplatRenderCommon}
${liveSplatRecordCommon}
${liveSplatFragmentShader}
@group(0) @binding(6) var<storage, read> splat_records: array<SplatRecord>;
@group(0) @binding(7) var<storage, read_write> splat_accum: array<atomic<u32>>;

const SPLAT_FIXED_SCALE = ${SPLAT_ACCUM_WGSL_SCALE}.0;

// particle_sprite_profile with defocus = 0 and fwidth supplied analytically.
fn splat_profile_analytic(r: f32, fw: f32) -> f32 {
  let aa = max(fw * 1.35, 0.008);
  let sharp = 1.0 - smoothstep(1.0 - aa * 2.0, 1.0 + aa, r);
  var mask = clamp(sharp, 0.0, 1.0);
  let glow_amt = clamp(uniforms.particle_glow_core, 0.0, 1.0);
  if (glow_amt <= 0.001) {
    return mask;
  }
  let core = pow(clamp(1.0 - r, 0.0, 1.0), mix(0.7, 3.0, glow_amt));
  return mask * mix(1.0, core, glow_amt);
}

@compute @workgroup_size(256)
fn accumulate_splats(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&splat_records)) {
    return;
  }
  let rec = splat_records[i];
  if (rec.center.x > 1.5) {
    return; // off-screen sentinel written for culled particles
  }
  let alpha_species = unpack2x16float(rec.packed_a.x);
  let alpha = alpha_species.x;
  if (alpha <= 0.000001) {
    return;
  }
  let axis_a = unpack2x16float(rec.axis.x);
  let axis_b = unpack2x16float(rec.axis.y);
  let speed_bright = unpack2x16float(rec.packed_a.y);
  let hue_sat = unpack2x16float(rec.packed_hsv.x);
  let val_w = unpack2x16float(rec.packed_hsv.y);
  let res = uniforms.resolution;
  // Clip -> pixel space (y flips: NDC +y is up, pixel +y is down).
  let center_px = vec2f((rec.center.x * 0.5 + 0.5) * res.x, (0.5 - rec.center.y * 0.5) * res.y);
  let a_px = vec2f(axis_a.x * 0.5 * res.x, -axis_a.y * 0.5 * res.y);
  let b_px = vec2f(axis_b.x * 0.5 * res.x, -axis_b.y * 0.5 * res.y);
  let det = a_px.x * b_px.y - a_px.y * b_px.x;
  if (abs(det) < 0.000001) {
    return;
  }
  // Inverse basis: local = inv * (pixel - center); quad coverage is |local| <= 1 on both axes,
  // identical to the rasterized quad's interpolated local coordinate.
  let inv_r0 = vec2f(b_px.y, -b_px.x) / det;
  let inv_r1 = vec2f(-a_px.y, a_px.x) / det;
  let ext = vec2f(abs(a_px.x) + abs(b_px.x), abs(a_px.y) + abs(b_px.y));
  let x0 = max(0, i32(floor(center_px.x - ext.x)));
  let x1 = min(i32(res.x) - 1, i32(ceil(center_px.x + ext.x)));
  let y0 = max(0, i32(floor(center_px.y - ext.y)));
  let y1 = min(i32(res.y) - 1, i32(ceil(center_px.y + ext.y)));
  if (x1 < x0 || y1 < y0) {
    return;
  }
  let color_hsv = vec4f(hue_sat.x, hue_sat.y, val_w.x, val_w.y);
  let base_color = splat_color(color_hsv, speed_bright.x, alpha_species.y);
  let base_intensity = alpha * particle_gain() * speed_bright.y;
  let hot = clamp(uniforms.particle_hot_core, 0.0, 1.0);
  let width_px = u32(res.x);
  for (var py = y0; py <= y1; py = py + 1) {
    for (var px = x0; px <= x1; px = px + 1) {
      let d = vec2f(f32(px) + 0.5, f32(py) + 0.5) - center_px;
      let local = vec2f(dot(inv_r0, d), dot(inv_r1, d));
      if (abs(local.x) > 1.0 || abs(local.y) > 1.0) {
        continue;
      }
      let r = length(local);
      // Analytic fwidth(r): |dr/dpx| + |dr/dpy| from the inverse basis columns.
      var fw = 0.0;
      if (r > 0.000001) {
        let dr_dx = dot(local, vec2f(inv_r0.x, inv_r1.x)) / r;
        let dr_dy = dot(local, vec2f(inv_r0.y, inv_r1.y)) / r;
        fw = abs(dr_dx) + abs(dr_dy);
      } else {
        fw = length(vec2f(inv_r0.x, inv_r1.x)) + length(vec2f(inv_r0.y, inv_r1.y));
      }
      let glow = splat_profile_analytic(r, fw);
      let intensity = glow * base_intensity;
      if (intensity <= 0.0000001) {
        continue;
      }
      var color = base_color;
      if (hot > 0.001) {
        let centerness = pow(clamp(1.0 - r, 0.0, 1.0), 2.0);
        color = mix(color, vec3f(1.0), centerness * hot);
      }
      let contribution = color * intensity * SPLAT_FIXED_SCALE;
      let base_index = (u32(py) * width_px + u32(px)) * 3u;
      let cr = u32(round(clamp(contribution.r, 0.0, 4294900000.0 / SPLAT_FIXED_SCALE)));
      let cg = u32(round(clamp(contribution.g, 0.0, 4294900000.0 / SPLAT_FIXED_SCALE)));
      let cb = u32(round(clamp(contribution.b, 0.0, 4294900000.0 / SPLAT_FIXED_SCALE)));
      if (cr > 0u) { atomicAdd(&splat_accum[base_index], cr); }
      if (cg > 0u) { atomicAdd(&splat_accum[base_index + 1u], cg); }
      if (cb > 0u) { atomicAdd(&splat_accum[base_index + 2u], cb); }
    }
  }
}
`;

// Fullscreen resolve: converts the fixed-point accumulation buffer back to float and adds it
// into the scene with the same additive blend the rasterized splats used.
const liveSplatResolveShader = /* wgsl */ `
${liveSplatRenderCommon}
@group(0) @binding(6) var<storage, read> splat_accum_values: array<u32>;

struct ResolveOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn splat_resolve_vs(@builtin(vertex_index) vertex_index: u32) -> ResolveOut {
  var out: ResolveOut;
  let uv = vec2f(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
  out.position = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
  out.uv = uv;
  return out;
}

@fragment
fn splat_resolve_fs(in: ResolveOut) -> @location(0) vec4f {
  let px = vec2u(in.position.xy);
  let base_index = (px.y * u32(uniforms.resolution.x) + px.x) * 3u;
  let scale = 1.0 / ${SPLAT_ACCUM_WGSL_SCALE}.0;
  let rgb = vec3f(
    f32(splat_accum_values[base_index]),
    f32(splat_accum_values[base_index + 1u]),
    f32(splat_accum_values[base_index + 2u])
  ) * scale;
  let lum = max(rgb.r, max(rgb.g, rgb.b));
  let density_curve = uniforms.particle_brightness_boost * pow(max(lum, 1e-6), uniforms.particle_exponent - 1.0);
  return vec4f(rgb * density_curve, 0.0);
}
`;

// Spatial reorder: counting sort of each particle chunk by a coarse 32^3 cell every N steps.
// Makes neighbors in space neighbors in memory, so the deposit atomics, field taps, and splat
// passes hit warm cache lines. Behavior is preserved exactly: per-particle math is a pure
// function of (state, persistent id, field); the persistent id buffer is permuted alongside so
// every id-derived hash (recycle, hazard, deposit jitter, variation) is order-invariant, and
// both ping-pong buffers get the same permutation so render interpolation stays coherent.
const SORT_GRID_SIZE = 32;
const liveSortShader = /* wgsl */ `
struct Particle {
  pos_cohort: vec4f,
  vel_id: vec4f,
};

const SORT_GRID = ${SORT_GRID_SIZE}u;

fn sort_cell_of(position: vec3f) -> u32 {
  let g = clamp(
    vec3u((clamp(position, vec3f(-1.0), vec3f(1.0)) + vec3f(1.0)) * 0.5 * f32(SORT_GRID)),
    vec3u(0u),
    vec3u(SORT_GRID - 1u)
  );
  return (g.z * SORT_GRID + g.y) * SORT_GRID + g.x;
}

@group(0) @binding(0) var<storage, read> sort_particles_in: array<Particle>;
@group(0) @binding(1) var<storage, read_write> sort_hist: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> sort_ids_in: array<u32>;
@group(0) @binding(3) var<storage, read_write> sort_particles_out: array<Particle>;
@group(0) @binding(4) var<storage, read_write> sort_ids_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> sort_map: array<u32>;

@compute @workgroup_size(256)
fn sort_histogram(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&sort_particles_in)) {
    return;
  }
  atomicAdd(&sort_hist[sort_cell_of(sort_particles_in[i].pos_cohort.xyz)], 1u);
}

// Serial exclusive scan over 32^3 cells. A single thread walking 32768 entries finishes in
// tens of microseconds and avoids a multi-pass hierarchical scan.
@compute @workgroup_size(1)
fn sort_scan() {
  var running = 0u;
  let cells = SORT_GRID * SORT_GRID * SORT_GRID;
  for (var c = 0u; c < cells; c = c + 1u) {
    let count = atomicLoad(&sort_hist[c]);
    atomicStore(&sort_hist[c], running);
    running = running + count;
  }
}

@compute @workgroup_size(256)
fn sort_scatter(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&sort_particles_in)) {
    return;
  }
  let particle = sort_particles_in[i];
  let dst = atomicAdd(&sort_hist[sort_cell_of(particle.pos_cohort.xyz)], 1u);
  sort_particles_out[dst] = particle;
  sort_ids_out[dst] = sort_ids_in[i];
  sort_map[i] = dst;
}
`;

// Applies the scatter's permutation to the second (previous-state) ping-pong buffer.
const liveSortApplyShader = /* wgsl */ `
struct Particle {
  pos_cohort: vec4f,
  vel_id: vec4f,
};

@group(0) @binding(0) var<storage, read> apply_particles_in: array<Particle>;
@group(0) @binding(1) var<storage, read> apply_map: array<u32>;
@group(0) @binding(2) var<storage, read_write> apply_particles_out: array<Particle>;

@compute @workgroup_size(256)
fn sort_apply_map(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&apply_particles_in)) {
    return;
  }
  apply_particles_out[apply_map[i]] = apply_particles_in[i];
}
`;

// Ribbon trail append: one invocation per particle; the selected subset (id % stride == 0)
// writes its current position into the ring slot for its ribbon index. A break flag (w=0) is
// set when the particle jumped (wrap/reset) so the ribbon segment is cut there instead of
// streaking across the domain.
const liveRibbonAppendShader = /* wgsl */ `
${commonComputeShader}
struct RibbonUniforms {
  stride: u32,
  max_ribbons: u32,
  frame: u32,
  length: u32,    // history frames spanned (<= ring max)
  joints: u32,    // spine points rendered across the span
  width: f32,
  fade: f32,       // 0 = solid, 1 = full opacity fade toward the tail
  fade_start: f32, // 0..1 along the trail (0 = tail) where the fade region begins
  edge_fade: f32,  // cross-ribbon exponential edge falloff (0 = hard-ish, 1 = soft glow edges)
};
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> trail: array<vec4f>;
@group(0) @binding(2) var<uniform> config: SimConfig;
@group(0) @binding(3) var<uniform> rib: RibbonUniforms;

@compute @workgroup_size(256)
fn append_trail(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= config.active_particle_count) {
    return;
  }
  let global_i = i + config.particle_offset;
  if (rib.stride == 0u || (global_i % rib.stride) != 0u) {
    return;
  }
  let r = global_i / rib.stride;
  if (r >= rib.max_ribbons) {
    return;
  }
  let pos = particles[i].pos_cohort.xyz;
  let L = 32u; // RIBBON_TRAIL_MAX ring size (fixed; ribbonLength selects how much to render)
  let slot = rib.frame % L;
  let prev = trail[r * L + ((rib.frame + L - 1u) % L)];
  // Store a CONTINUOUS (unwrapped) position so domain wraps don't streak the ribbon across the
  // screen: accumulate the shortest per-axis delta from the previous sample (the [-1,1] domain has
  // width 2). w=1 marks a filled slot; zero-initialised slots stay w=0 (invisible until filled).
  var stored = pos;
  if (prev.w > 0.5) {
    var d = pos - prev.xyz;
    if (d.x > 1.0) { d.x = d.x - 2.0; } else if (d.x < -1.0) { d.x = d.x + 2.0; }
    if (d.y > 1.0) { d.y = d.y - 2.0; } else if (d.y < -1.0) { d.y = d.y + 2.0; }
    if (d.z > 1.0) { d.z = d.z - 2.0; } else if (d.z < -1.0) { d.z = d.z + 2.0; }
    stored = prev.xyz + d;
  }
  trail[r * L + slot] = vec4f(stored, 1.0);
}
`;

// Ribbon render: one camera-facing triangle strip per ribbon (instance), L*2 vertices. Each spine
// point is projected with the shared project(), offset perpendicular in NDC by the (tapered) width,
// faded head->tail. Additive, order-independent. Colour is a stable per-ribbon hue.
const liveRibbonRenderShader = /* wgsl */ `
${liveSplatRenderCommon}
struct RibbonUniforms {
  stride: u32,
  max_ribbons: u32,
  frame: u32,
  length: u32,    // history frames spanned (<= ring max)
  joints: u32,    // spine points rendered across the span
  width: f32,
  fade: f32,       // 0 = solid, 1 = full opacity fade toward the tail
  fade_start: f32, // 0..1 along the trail (0 = tail) where the fade region begins
  edge_fade: f32,  // cross-ribbon exponential edge falloff (0 = hard-ish, 1 = soft glow edges)
};
@group(0) @binding(0) var<storage, read> trail: array<vec4f>;
@group(0) @binding(2) var<uniform> rib: RibbonUniforms;

struct RibbonOut {
  @builtin(position) position: vec4f,
  @location(0) alpha: f32,
  @location(1) color: vec3f,
  @location(2) edge: f32, // -1..+1 across the ribbon width (0 = centre), for per-pixel edge fade
};

// Read the ring slot 'age' frames behind the newest write (0 = head/newest). Ring size is fixed 32.
fn ribbon_sample(r: u32, age: u32) -> vec4f {
  let RING = 32u;
  let slot = (rib.frame + RING - (age % RING)) % RING;
  return trail[r * RING + slot];
}

@vertex
fn ribbon_vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) inst: u32) -> RibbonOut {
  var out: RibbonOut;
  let r = inst;
  let J = max(2u, rib.joints);
  let S = max(2u, rib.length);
  let j = vi / 2u;                          // spine point index 0..J-1
  let side = f32(vi % 2u) * 2.0 - 1.0;      // -1 / +1 across the ribbon
  let t = f32(j) / f32(J - 1u);             // 0 = tail .. 1 = head
  let head = ribbon_sample(r, 0u);
  // Wrap the head into [-1,1]; the trail is drawn RELATIVE to it. Stored slots are continuous
  // (unwrapped) and share one frame of reference, so (slot - head) is the small, correct offset.
  // Anchoring at the wrapped head keeps ribbon coordinates bounded near the particle instead of
  // flying off after repeated domain wraps (the cause of the "bouncing all over" artifact).
  let head_anchor = head.xyz - 2.0 * round(head.xyz * 0.5);
  let age = u32(round((1.0 - t) * f32(S - 1u)));
  let cur = ribbon_sample(r, age);
  let tn = f32(min(j + 1u, J - 1u)) / f32(J - 1u);
  let nxt = ribbon_sample(r, u32(round((1.0 - tn) * f32(S - 1u))));
  var spine = head_anchor + (cur.xyz - head.xyz);
  var spine_nxt = head_anchor + (nxt.xyz - head.xyz);
  var visible = 1.0;
  if (cur.w < 0.5) { spine = head_anchor; spine_nxt = head_anchor; visible = 0.0; } // unfilled ring slot -> collapse, invisible
  let p = project(spine);
  let pn = project(spine_nxt);
  let d = pn.xy - p.xy;
  let dlen = length(d);
  let dir = select(vec2f(1.0, 0.0), d / dlen, dlen > 1e-6);
  let perp = vec2f(-dir.y, dir.x);
  // Particle SIZE modulates ribbon width (rib.width is a multiplier on top of the particle-size
  // slider); ribbons never feed back into particle size. Per-particle size variation is intentionally
  // NOT applied here so width stays steady (no per-frame pulsing). Converted px -> NDC via res.y.
  let width_px = max(0.0, uniforms.particle_size_px) * rib.width;
  let width_ndc = (width_px / max(1.0, uniforms.resolution.y)) * 2.0;
  out.position = vec4f(p.xy + perp * (width_ndc * side), 0.0, 1.0);
  // Opacity fade toward the tail; fade_start sets how far along (0 = tail) the fade region begins.
  let fade_ramp = smoothstep(0.0, max(0.01, rib.fade_start), t);
  out.alpha = visible * mix(1.0, fade_ramp, clamp(rib.fade, 0.0, 1.0)) * 0.3;
  out.color = hsv2rgb(vec3f(fract(f32(r) * 0.6180339887), 0.72, 1.0));
  out.edge = side;
  return out;
}

@fragment
fn ribbon_fs(in: RibbonOut) -> @location(0) vec4f {
  // Per-pixel cross-ribbon edge fade: pow(1-|edge|, exponent) -> 0 at the edges, so the ribbon
  // always feathers out (anti-aliased) rather than showing a hard, crawling edge. edge_fade raises
  // the exponent for a tighter bright core with a soft exponential falloff, like the particle glow.
  let e = clamp(1.0 - abs(in.edge), 0.0, 1.0);
  let edge_alpha = pow(e, mix(0.8, 5.0, clamp(rib.edge_fade, 0.0, 1.0)));
  let a = in.alpha * edge_alpha;
  return vec4f(in.color * a, a); // premultiplied additive
}
`;

const liveDensitySplatShader = /* wgsl */ `
${liveSplatRenderCommon}
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read> particles_prev: array<Particle>;
@group(0) @binding(3) var<storage, read> particle_field_values: array<vec4f>;

fn density_particle_field_signal(raw: vec4f) -> f32 {
  return max(0.0, raw.w + length(raw.xyz) * 0.25);
}

fn density_particle_field_cell(x: i32, y: i32, z: i32) -> u32 {
  let w = i32(uniforms.width);
  let h = i32(uniforms.height);
  let d = i32(uniforms.depth);
  let cx = clamp(x, 0, w - 1);
  let cy = clamp(y, 0, h - 1);
  let cz = clamp(z, 0, d - 1);
  return u32((cz * h + cy) * w + cx);
}

fn density_particle_field_at(position: vec3f) -> f32 {
  if (any(position < vec3f(-1.0)) || any(position > vec3f(1.0))) {
    return 0.0;
  }
  let dims = vec3f(f32(uniforms.width), f32(uniforms.height), f32(uniforms.depth));
  let grid = (position + vec3f(1.0)) * 0.5 * dims - vec3f(0.5);
  let base = vec3i(floor(grid));
  let t = fract(grid);
  var accum = vec4f(0.0);
  for (var dz = 0i; dz <= 1i; dz = dz + 1i) {
    for (var dy = 0i; dy <= 1i; dy = dy + 1i) {
      for (var dx = 0i; dx <= 1i; dx = dx + 1i) {
        let weight = mix(1.0 - t.x, t.x, f32(dx)) * mix(1.0 - t.y, t.y, f32(dy)) * mix(1.0 - t.z, t.z, f32(dz));
        accum = accum + particle_field_values[density_particle_field_cell(base.x + dx, base.y + dy, base.z + dz)] * weight;
      }
    }
  }
  return density_particle_field_signal(accum);
}

fn density_particle_stats(position: vec3f, radius: f32) -> ParticleDensityStats {
  let center = density_particle_field_at(position);
  if (radius <= 0.00001) {
    return ParticleDensityStats(center, center, center, 1.0, 0.0);
  }
  return density_stats_from_axis_samples(
    center,
    density_particle_field_at(position + vec3f(radius, 0.0, 0.0)),
    density_particle_field_at(position - vec3f(radius, 0.0, 0.0)),
    density_particle_field_at(position + vec3f(0.0, radius, 0.0)),
    density_particle_field_at(position - vec3f(0.0, radius, 0.0)),
    density_particle_field_at(position + vec3f(0.0, 0.0, radius)),
    density_particle_field_at(position - vec3f(0.0, 0.0, radius))
  );
}

fn density_particle_neighborhood(position: vec3f) -> f32 {
  return density_particle_stats(position, max(0.0, uniforms.particle_density_radius)).peak;
}

struct DensityOut {
  @builtin(position) position: vec4f,
  @location(0) local: vec2f,
  @location(1) weight: f32,
  @location(2) color: vec3f,
};

fn culled_density_splat() -> DensityOut {
  var out: DensityOut;
  out.position = vec4f(2.0, 2.0, 0.0, 1.0);
  out.local = vec2f(0.0);
  out.weight = 0.0;
  out.color = vec3f(0.0);
  return out;
}

fn density_particle_color(velocity: vec3f, cohort: f32, visual_color: vec4f) -> vec3f {
  if (uniforms.particle_color_mode == 0u) {
    return particle_tint_color();
  }
  let base_gradient = fract(cohort / max(1.0, f32(uniforms.cohorts)) + visual_color.x * 0.17);
  let t = particle_gradient_coordinate(velocity, base_gradient);
  let color = particle_gradient_color(t, uniforms.particle_color_mode);
  return apply_tint(color, particle_tint_color(), 0.28);
}

fn density_vs(vertex_index: u32, scale: f32) -> DensityOut {
  let index = vertex_index / 6u;
  let particle = particles[index];
  let previous_particle = particles_prev[index];
  let particle_lerp_t = particle_lerp_amount(previous_particle.pos_cohort.xyz, particle.pos_cohort.xyz, uniforms.render_lerp_t);
  let large_motion_fade = particle_large_motion_fade(previous_particle.pos_cohort.xyz, particle.pos_cohort.xyz, uniforms.render_lerp_t);
  let interp_pos = interpolate_particle_position(previous_particle.pos_cohort.xyz, particle.pos_cohort.xyz, particle_lerp_t);
  let visual_velocity = particle_visual_velocity(previous_particle, particle, particle_lerp_t);
  let speed_cutoff_visibility = particle_speed_cutoff_visibility(visual_velocity);
  if (speed_cutoff_visibility <= 0.000001) {
    return culled_density_splat();
  }
  let visual_color = interpolate_particle_color(stored_particle_color(previous_particle), stored_particle_color(particle), particle_lerp_t);
  let density_stats = density_particle_stats(interp_pos, max(0.0, uniforms.particle_density_radius));
  let local_density = density_stats.peak;
  let density_gate_min = max(0.003, uniforms.particle_density_cutoff * 0.35);
  let density_gate_max = max(0.022, uniforms.particle_density_cutoff * 2.0 + 0.018);
  let density_gate = smoothstep(density_gate_min, density_gate_max, local_density);
  if (density_gate <= 0.0) {
    return culled_density_splat();
  }
  let view_depth = uniforms.distance + world_to_camera(interp_pos).z;
  if (view_depth <= PARTICLE_NEAR) {
    return culled_density_splat();
  }
  let projected = project(interp_pos);
  let local = corner(vertex_index);
  let safe_scale = max(0.05, scale);
  let splat_radius = particle_splat_radius_ndc(projected.z) * safe_scale;
  let area_norm = 1.0 / max(1.0, safe_scale * safe_scale);
  var render_local = local;
  let direction = particle_velocity_screen_axis(visual_velocity);
  if (scale <= 2.5 && length(direction) > 0.000001) {
    let perpendicular = vec2f(-direction.y, direction.x);
    let stretch = 1.0 + clamp(uniforms.filament, 0.0, 1.0) * 1.25;
    render_local = direction * local.x * stretch + perpendicular * local.y;
  }
  var out: DensityOut;
  out.position = vec4f(projected.x + render_local.x * splat_radius.x, projected.y + render_local.y * splat_radius.y, 0.0, 1.0);
  out.local = local;
  out.weight = density_gate * speed_cutoff_visibility * area_norm * large_motion_fade;
  out.color = density_particle_color(visual_velocity, particle.pos_cohort.w, visual_color);
  return out;
}

@vertex
fn density_small_vs(@builtin(vertex_index) vertex_index: u32) -> DensityOut {
  return density_vs(vertex_index, uniforms.density_small_scale);
}

@vertex
fn density_large_vs(@builtin(vertex_index) vertex_index: u32) -> DensityOut {
  return density_vs(vertex_index, uniforms.density_large_scale);
}

@fragment
fn density_splat_fs(in: DensityOut) -> @location(0) vec4f {
  let d = dot(in.local, in.local);
  if (d > 1.45 || in.weight <= 0.0) {
    discard;
  }
  let glow = exp(-d * 2.35) * in.weight;
  return vec4f(in.color * glow, glow);
}
`;

const liveDensityCompositeShader = /* wgsl */ `
${liveSplatRenderCommon}
@group(0) @binding(0) var density_small: texture_2d<f32>;
@group(0) @binding(2) var density_large: texture_2d<f32>;
@group(0) @binding(3) var fog_tex: texture_2d<f32>;
@group(0) @binding(4) var density_sampler: sampler;

struct FullscreenOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

fn saturate(value: f32) -> f32 {
  return clamp(value, 0.0, 1.0);
}

fn preserve_density_chroma(color: vec3f, amount: f32) -> vec3f {
  let peak = max(0.0001, max(color.r, max(color.g, color.b)));
  let hue_color = color / peak;
  return saturate_color(mix(color, hue_color, clamp(amount, 0.0, 1.0)), 1.18 + amount * 0.82);
}

@vertex
fn fullscreen_vs(@builtin(vertex_index) vertex_index: u32) -> FullscreenOut {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -3.0),
    vec2f(-1.0, 1.0),
    vec2f(3.0, 1.0)
  );
  let position = positions[vertex_index];
  var out: FullscreenOut;
  out.position = vec4f(position, 0.0, 1.0);
  out.uv = position * 0.5 + vec2f(0.5);
  return out;
}

@fragment
fn density_composite_fs(in: FullscreenOut) -> @location(0) vec4f {
  let sample_uv = vec2f(in.uv.x, 1.0 - in.uv.y);
  let texel = 1.0 / max(uniforms.resolution, vec2f(1.0));
  let small_sample = textureSampleLevel(density_small, density_sampler, sample_uv, 0.0);
  let large_sample = textureSampleLevel(density_large, density_sampler, sample_uv, 0.0);
  let small_raw = small_sample.a;
  let large_raw = large_sample.a;
  let small_l = textureSampleLevel(density_small, density_sampler, sample_uv - vec2f(texel.x, 0.0), 0.0).a;
  let small_r = textureSampleLevel(density_small, density_sampler, sample_uv + vec2f(texel.x, 0.0), 0.0).a;
  let small_u = textureSampleLevel(density_small, density_sampler, sample_uv + vec2f(0.0, texel.y), 0.0).a;
  let small_d = textureSampleLevel(density_small, density_sampler, sample_uv - vec2f(0.0, texel.y), 0.0).a;
  let neighbor_mean = (small_l + small_r + small_u + small_d) * 0.25;
  let ridge_mix = clamp(uniforms.density_contrast_balance, 0.0, 1.0);
  let local_peak = max(0.0, small_raw - neighbor_mean * mix(0.72, 1.18, ridge_mix));
  let edge_energy = length(vec2f(small_r - small_l, small_u - small_d));
  let ridge_field = log2(1.0 + local_peak * 14.0 + edge_energy * 5.5);
  let body_field = log2(1.0 + small_raw * 0.56) * (1.0 - ridge_mix) * 0.22;
  let large_field = log2(1.0 + large_raw * 2.35);
  let support = smoothstep(uniforms.density_large_threshold, uniforms.density_large_threshold + 0.24, large_field);
  let density_response = max(0.0, ridge_field * mix(0.42, 1.18, ridge_mix) + body_field) * support;
  let filament = pow(1.0 - exp(-density_response), max(0.25, uniforms.density_emission_power));
  let fog = textureSampleLevel(fog_tex, density_sampler, sample_uv, 0.0);
  let occlusion = mix(1.0, 1.0 - fog.a * 0.72, clamp(uniforms.density_occlusion, 0.0, 1.0));
  let emission_gain = max(0.0, uniforms.density_contrast_gain);
  let emission = exp2(filament * emission_gain * 0.42) - 1.0;
  let volume_detail_scale = select(1.0, 0.72, uniforms.render_layer == 5u);
  let signal = emission * max(0.0, uniforms.density_pass_strength) * occlusion * volume_detail_scale;
  if (signal <= 0.00001) {
    discard;
  }
  let color_weight = max(0.0001, small_sample.a + large_sample.a * 0.18);
  let density_color = max(vec3f(0.0), (small_sample.rgb + large_sample.rgb * 0.18) / color_weight);
  let color = preserve_density_chroma(density_color, 0.32 + filament * 0.38);
  let intensity = signal * particle_gain() * 0.38;
  return vec4f(color * intensity, clamp(intensity, 0.0, 1.0));
}
`;

const liveAccumulationSplatShader = /* wgsl */ `
${liveSplatRenderCommon}
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read> particles_prev: array<Particle>;
@group(0) @binding(3) var<storage, read> particle_field_values: array<vec4f>;

fn accumulation_particle_field_signal(raw: vec4f) -> f32 {
  return max(0.0, raw.w + length(raw.xyz) * 0.25);
}

fn accumulation_particle_field_cell(x: i32, y: i32, z: i32) -> u32 {
  let w = i32(uniforms.width);
  let h = i32(uniforms.height);
  let d = i32(uniforms.depth);
  let cx = clamp(x, 0, w - 1);
  let cy = clamp(y, 0, h - 1);
  let cz = clamp(z, 0, d - 1);
  return u32((cz * h + cy) * w + cx);
}

fn accumulation_particle_field_at(position: vec3f) -> f32 {
  if (any(position < vec3f(-1.0)) || any(position > vec3f(1.0))) {
    return 0.0;
  }
  let dims = vec3f(f32(uniforms.width), f32(uniforms.height), f32(uniforms.depth));
  let grid = (position + vec3f(1.0)) * 0.5 * dims - vec3f(0.5);
  let base = vec3i(floor(grid));
  let t = fract(grid);
  var accum = vec4f(0.0);
  for (var dz = 0i; dz <= 1i; dz = dz + 1i) {
    for (var dy = 0i; dy <= 1i; dy = dy + 1i) {
      for (var dx = 0i; dx <= 1i; dx = dx + 1i) {
        let weight = mix(1.0 - t.x, t.x, f32(dx)) * mix(1.0 - t.y, t.y, f32(dy)) * mix(1.0 - t.z, t.z, f32(dz));
        accum = accum + particle_field_values[accumulation_particle_field_cell(base.x + dx, base.y + dy, base.z + dz)] * weight;
      }
    }
  }
  return accumulation_particle_field_signal(accum);
}

fn accumulation_particle_stats(position: vec3f, radius: f32) -> ParticleDensityStats {
  let center = accumulation_particle_field_at(position);
  if (radius <= 0.00001) {
    return ParticleDensityStats(center, center, center, 1.0, 0.0);
  }
  return density_stats_from_axis_samples(
    center,
    accumulation_particle_field_at(position + vec3f(radius, 0.0, 0.0)),
    accumulation_particle_field_at(position - vec3f(radius, 0.0, 0.0)),
    accumulation_particle_field_at(position + vec3f(0.0, radius, 0.0)),
    accumulation_particle_field_at(position - vec3f(0.0, radius, 0.0)),
    accumulation_particle_field_at(position + vec3f(0.0, 0.0, radius)),
    accumulation_particle_field_at(position - vec3f(0.0, 0.0, radius))
  );
}

struct AccumulationOut {
  @builtin(position) position: vec4f,
  @location(0) local: vec2f,
  @location(1) weight: f32,
  @location(2) color: vec3f,
};

fn culled_accumulation_splat() -> AccumulationOut {
  var out: AccumulationOut;
  out.position = vec4f(2.0, 2.0, 0.0, 1.0);
  out.local = vec2f(0.0);
  out.weight = 0.0;
  out.color = vec3f(0.0);
  return out;
}

fn accumulation_particle_color(velocity: vec3f, cohort: f32, visual_color: vec4f) -> vec3f {
  if (uniforms.particle_color_mode == 0u) {
    return particle_tint_color();
  }
  let base_gradient = fract(cohort / max(1.0, f32(uniforms.cohorts)) + visual_color.x * 0.17);
  let t = particle_gradient_coordinate(velocity, base_gradient);
  let color = particle_gradient_color(t, uniforms.particle_color_mode);
  return apply_tint(color, particle_tint_color(), 0.26);
}

@vertex
fn accumulation_vs(@builtin(vertex_index) vertex_index: u32) -> AccumulationOut {
  let index = vertex_index / 6u;
  let particle = particles[index];
  let previous_particle = particles_prev[index];
  let particle_lerp_t = particle_lerp_amount(previous_particle.pos_cohort.xyz, particle.pos_cohort.xyz, uniforms.render_lerp_t);
  let large_motion_fade = particle_large_motion_fade(previous_particle.pos_cohort.xyz, particle.pos_cohort.xyz, uniforms.render_lerp_t);
  let interp_pos = interpolate_particle_position(previous_particle.pos_cohort.xyz, particle.pos_cohort.xyz, particle_lerp_t);
  let visual_velocity = particle_visual_velocity(previous_particle, particle, particle_lerp_t);
  let speed_cutoff_visibility = particle_speed_cutoff_visibility(visual_velocity);
  if (speed_cutoff_visibility <= 0.000001) {
    return culled_accumulation_splat();
  }
  let visual_color = interpolate_particle_color(stored_particle_color(previous_particle), stored_particle_color(particle), particle_lerp_t);

  let reject = clamp(uniforms.accumulation_noise_reject, 0.0, 1.0);
  let density_radius = max(max(0.018, uniforms.particle_density_radius), uniforms.accumulation_radius * 0.018);
  let density_stats = accumulation_particle_stats(interp_pos, density_radius);
  let structure_support = density_stats.support;
  let field_confidence = smoothstep(0.0002, 0.006, max(density_stats.peak, density_stats.mean));
  let cutoff = max(uniforms.particle_density_cutoff, mix(0.0012, 0.0095, reject));
  let density_width = max(0.003, cutoff * 1.8 + mix(0.006, 0.016, reject));
  let density_gate_raw = smoothstep(cutoff, cutoff + density_width, mix(density_stats.mean, density_stats.peak, 0.38));
  let density_gate = mix(mix(0.20, 0.10, reject), density_gate_raw, field_confidence);
  let support_min = mix(0.08, 0.30, reject);
  let support_gate = smoothstep(support_min, min(0.94, support_min + 0.28), structure_support);
  let ridge_gate = 1.0;
  let support_term = mix(mix(0.48, 0.30, reject), support_gate, field_confidence);
  let ridge_term = mix(mix(0.86, 0.60, reject), ridge_gate, field_confidence);
  let accumulation_gate = density_gate * mix(0.45, support_term, 0.65) * ridge_term;
  if (accumulation_gate <= 0.0) {
    return culled_accumulation_splat();
  }

  let view_depth = uniforms.distance + world_to_camera(interp_pos).z;
  if (view_depth <= PARTICLE_NEAR) {
    return culled_accumulation_splat();
  }
  let projected = project(interp_pos);
  let local = corner(vertex_index);
  let safe_scale = max(0.05, uniforms.accumulation_radius);
  let splat_radius = particle_splat_radius_ndc(projected.z) * safe_scale;
  let area_norm = 1.0 / max(1.0, safe_scale * safe_scale);
  var render_local = local;
  let direction = particle_velocity_screen_axis(visual_velocity);
  if (length(direction) > 0.000001) {
    let perpendicular = vec2f(-direction.y, direction.x);
    let stretch = 1.0 + clamp(uniforms.filament, 0.0, 1.0) * mix(0.35, 1.45, 1.0 - reject);
    render_local = direction * local.x * stretch + perpendicular * local.y;
  }

  var out: AccumulationOut;
  out.position = vec4f(projected.x + render_local.x * splat_radius.x, projected.y + render_local.y * splat_radius.y, 0.0, 1.0);
  out.local = local;
  out.weight = max(0.0, uniforms.particle_opacity) * clamp(visual_color.w / 0.045, 0.0, 2.0) * accumulation_gate * speed_cutoff_visibility * area_norm * large_motion_fade;
  out.color = accumulation_particle_color(visual_velocity, particle.pos_cohort.w, visual_color);
  return out;
}

@fragment
fn accumulation_splat_fs(in: AccumulationOut) -> @location(0) vec4f {
  let d = dot(in.local, in.local);
  if (d > 1.18 || in.weight <= 0.000001) {
    discard;
  }
  let core = exp(-d * 4.8);
  let skirt = exp(-d * 1.4) * 0.035;
  let glow = (core + skirt) * in.weight;
  return vec4f(in.color * glow, glow);
}
`;

const liveAccumulationCompositeShader = /* wgsl */ `
${liveSplatRenderCommon}
@group(0) @binding(0) var accumulation_current: texture_2d<f32>;
@group(0) @binding(2) var accumulation_history: texture_2d<f32>;
@group(0) @binding(3) var accumulation_sampler: sampler;

struct FullscreenOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

struct AccumulationCompositeOut {
  @location(0) scene: vec4f,
  @location(1) history: vec4f,
};

@vertex
fn fullscreen_vs(@builtin(vertex_index) vertex_index: u32) -> FullscreenOut {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -3.0),
    vec2f(-1.0, 1.0),
    vec2f(3.0, 1.0)
  );
  let position = positions[vertex_index];
  var out: FullscreenOut;
  out.position = vec4f(position, 0.0, 1.0);
  out.uv = position * 0.5 + vec2f(0.5);
  return out;
}

fn preserve_accumulation_chroma(color: vec3f, energy: f32, shaped: f32) -> vec3f {
  let peak = max(0.0001, max(color.r, max(color.g, color.b)));
  let hue_color = color / peak;
  let chroma_mix = clamp(0.22 + shaped * 0.55 + smoothstep(0.02, 0.75, energy) * 0.22, 0.0, 0.92);
  return saturate_color(mix(color, hue_color, chroma_mix), 1.1 + shaped * 0.85);
}

fn accumulation_sample(uv: vec2f, memory: f32) -> vec4f {
  let current = max(textureSampleLevel(accumulation_current, accumulation_sampler, uv, 0.0), vec4f(0.0));
  let history = max(textureSampleLevel(accumulation_history, accumulation_sampler, uv, 0.0), vec4f(0.0));
  return current + history * memory;
}

@fragment
fn accumulation_composite_fs(in: FullscreenOut) -> AccumulationCompositeOut {
  let sample_uv = vec2f(in.uv.x, 1.0 - in.uv.y);
  let memory = clamp(uniforms.accumulation_memory, 0.0, 0.96);
  let texel = 1.0 / max(uniforms.resolution, vec2f(1.0));
  let center = accumulation_sample(sample_uv, memory);
  let left = accumulation_sample(sample_uv + vec2f(-texel.x, 0.0), memory);
  let right = accumulation_sample(sample_uv + vec2f(texel.x, 0.0), memory);
  let up = accumulation_sample(sample_uv + vec2f(0.0, texel.y), memory);
  let down = accumulation_sample(sample_uv + vec2f(0.0, -texel.y), memory);
  let diagonal_a = accumulation_sample(sample_uv + texel * vec2f(-1.5, 1.5), memory);
  let diagonal_b = accumulation_sample(sample_uv + texel * vec2f(1.5, 1.5), memory);
  let diagonal_c = accumulation_sample(sample_uv + texel * vec2f(-1.5, -1.5), memory);
  let diagonal_d = accumulation_sample(sample_uv + texel * vec2f(1.5, -1.5), memory);
  let wide_left = accumulation_sample(sample_uv + vec2f(-texel.x * 3.0, 0.0), memory);
  let wide_right = accumulation_sample(sample_uv + vec2f(texel.x * 3.0, 0.0), memory);
  let wide_up = accumulation_sample(sample_uv + vec2f(0.0, texel.y * 3.0), memory);
  let wide_down = accumulation_sample(sample_uv + vec2f(0.0, -texel.y * 3.0), memory);
  let wide_diagonal_a = accumulation_sample(sample_uv + texel * vec2f(-3.5, 3.5), memory);
  let wide_diagonal_b = accumulation_sample(sample_uv + texel * vec2f(3.5, 3.5), memory);
  let wide_diagonal_c = accumulation_sample(sample_uv + texel * vec2f(-3.5, -3.5), memory);
  let wide_diagonal_d = accumulation_sample(sample_uv + texel * vec2f(3.5, -3.5), memory);
  let cardinal = (left + right + up + down) * 0.25;
  let diagonal = (diagonal_a + diagonal_b + diagonal_c + diagonal_d) * 0.25;
  let wide_cardinal = (wide_left + wide_right + wide_up + wide_down) * 0.25;
  let wide_diagonal = (wide_diagonal_a + wide_diagonal_b + wide_diagonal_c + wide_diagonal_d) * 0.25;
  let wide = wide_cardinal * 0.65 + wide_diagonal * 0.35;
  let neighborhood = center * 0.38 + cardinal * 0.30 + diagonal * 0.16 + wide * 0.16;
  let neighbor_energy = cardinal.a * 0.58 + diagonal.a * 0.24 + wide.a * 0.18;
  let reject = clamp(uniforms.accumulation_noise_reject, 0.0, 1.0);
  let support_ratio = neighbor_energy / max(center.a, 0.0001);
  let support_floor = mix(0.002, 0.022, reject);
  let support_gate = smoothstep(mix(0.10, 0.28, reject), mix(0.34, 0.70, reject), support_ratio) *
    smoothstep(support_floor, support_floor + mix(0.012, 0.055, reject), neighbor_energy);
  let ridge_energy = max(0.0, center.a - neighbor_energy * mix(0.32, 0.64, reject));
  let ridge_gate = smoothstep(mix(0.0006, 0.0045, reject), mix(0.010, 0.038, reject), ridge_energy + neighbor_energy * 0.35);
  let filament_excess = max(0.0, neighborhood.a - wide.a * mix(0.30, 0.56, reject));
  let coherent_gate = max(support_gate * ridge_gate, support_gate * smoothstep(0.001, 0.032, filament_excess) * 0.42);
  let coherent_energy = (filament_excess + neighborhood.a * coherent_gate * 0.24) * coherent_gate;
  let average_color = neighborhood.rgb / max(0.0001, neighborhood.a);
  let normalized = 1.0 - exp(-coherent_energy * max(0.0, uniforms.accumulation_strength) * 0.22);
  let shaped = pow(clamp(normalized, 0.0, 1.0), max(0.5, uniforms.accumulation_curve));
  let hot_core = exp2(shaped * max(0.0, uniforms.accumulation_strength) * 0.46) - 1.0;
  let soft_body = log2(1.0 + coherent_energy * 0.45) * (1.0 - smoothstep(0.0, 0.85, shaped)) * 0.12;
  let emission = (hot_core * 0.36 + soft_body) * particle_gain();
  var out: AccumulationCompositeOut;
  if (emission <= 0.000001 || coherent_energy <= 0.000001) {
    out.scene = vec4f(0.0);
  } else {
    let color = preserve_accumulation_chroma(max(average_color, vec3f(0.0)), coherent_energy, shaped);
    out.scene = vec4f(color * emission, clamp(emission, 0.0, 1.0));
  }
  let history_gate = clamp(max(coherent_gate, smoothstep(0.02, 0.24, coherent_energy) * 0.65), 0.0, 1.0);
  out.history = vec4f(clamp(neighborhood.rgb * history_gate, vec3f(0.0), vec3f(512.0)), clamp(neighborhood.a * history_gate, 0.0, 512.0));
  return out;
}
`;

const liveVolumeDensityComputeShader = /* wgsl */ `
${liveSplatRenderCommon}

const VOLUME_ACCUM_SCALE = 1024.0;
const VOLUME_MAX_ACCUM = 512.0;
const VOLUME_RESOLVE_MAX = 4096.0;
const VOLUME_ACCUM_FIXED_LIMIT = 1073741824i;
const VOLUME_DENSITY_PARTICLE_BUDGET = 262144u;

struct VolumeChunkConfig {
  width: u32,
  height: u32,
  depth: u32,
  particle_count: u32,
  timestep: u32,
  seed: u32,
  cohorts: u32,
  boundary_mode: u32,
  dt: f32,
  sensor_gain: f32,
  sensor_angle: f32,
  sensor_distance: f32,
  mutation_scale: f32,
  global_force_mult: f32,
  drag: f32,
  strafe_power: f32,
  axial_force: f32,
  lateral_force: f32,
  hazard_rate: f32,
  trail_persistence: f32,
  trail_diffusion: f32,
  deposit_radius: f32,
  deposit_mass: f32,
  sigma: f32,
  rule_seed: f32,
  orientation_mix: f32,
  symmetry_axes: u32,
  absolute_orientation: u32,
  initial_conditions: u32,
  deposit_tap_radius: u32,
  hue_sensitivity: f32,
  color_by_cohort: u32,
  step_time_scale: f32,
  visual_lerp_t: f32,
  recycle_cutoff: f32,
  recycle_enabled: u32,
  particle_offset: u32,
  active_particle_count: u32,
  domain_shape: u32,
};

@group(0) @binding(0) var<storage, read_write> volume_accum: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read> volume_particles: array<Particle>;
@group(0) @binding(3) var<storage, read> volume_particles_prev: array<Particle>;
@group(0) @binding(4) var<storage, read_write> volume_small_out: array<vec4f>;
@group(0) @binding(5) var volume_small_texture: texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var<storage, read> volume_src: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> volume_dst: array<vec4f>;
@group(0) @binding(8) var volume_large_texture: texture_storage_3d<rgba16float, write>;
@group(0) @binding(9) var<uniform> chunk_config: VolumeChunkConfig;
@group(0) @binding(10) var<storage, read> volume_particle_field_values: array<vec4f>;

fn volume_flat(cell: vec3u) -> u32 {
  return (cell.z * uniforms.height + cell.y) * uniforms.width + cell.x;
}

fn volume_cell_from_flat(flat: u32) -> vec3u {
  let x = flat % uniforms.width;
  let y = (flat / uniforms.width) % uniforms.height;
  let z = flat / (uniforms.width * uniforms.height);
  return vec3u(x, y, z);
}

fn volume_flat_clamped(x: i32, y: i32, z: i32) -> u32 {
  let cx = u32(clamp(x, 0i, i32(uniforms.width) - 1i));
  let cy = u32(clamp(y, 0i, i32(uniforms.height) - 1i));
  let cz = u32(clamp(z, 0i, i32(uniforms.depth) - 1i));
  return (cz * uniforms.height + cy) * uniforms.width + cx;
}

fn volume_particle_color(velocity: vec3f, cohort: f32, visual_color: vec4f) -> vec3f {
  if (uniforms.particle_color_mode == 0u) {
    return particle_tint_color();
  }
  let base_gradient = fract(cohort / max(1.0, f32(uniforms.cohorts)) + visual_color.x * 0.17);
  let t = particle_gradient_coordinate(velocity, base_gradient);
  let color = particle_gradient_color(t, uniforms.particle_color_mode);
  let tinted = apply_tint(color, particle_tint_color(), 0.28);
  let peak = max(0.0001, max(tinted.r, max(tinted.g, tinted.b)));
  let hue_color = tinted / peak;
  return saturate_color(mix(tinted, hue_color, 0.58), 1.78);
}

fn volume_particle_density_signal(raw: vec4f) -> f32 {
  return max(0.0, raw.w + length(raw.xyz) * 0.25);
}

fn volume_particle_field_cell(x: i32, y: i32, z: i32) -> u32 {
  let w = i32(uniforms.width);
  let h = i32(uniforms.height);
  let d = i32(uniforms.depth);
  let cx = clamp(x, 0, w - 1);
  let cy = clamp(y, 0, h - 1);
  let cz = clamp(z, 0, d - 1);
  return u32((cz * h + cy) * w + cx);
}

fn volume_particle_density_at(position: vec3f) -> f32 {
  if (any(position < vec3f(-1.0)) || any(position > vec3f(1.0))) {
    return 0.0;
  }
  let dims = vec3f(f32(uniforms.width), f32(uniforms.height), f32(uniforms.depth));
  let grid = (position + vec3f(1.0)) * 0.5 * dims - vec3f(0.5);
  let base = vec3i(floor(grid));
  let t = fract(grid);
  var accum = vec4f(0.0);
  for (var dz = 0i; dz <= 1i; dz = dz + 1i) {
    for (var dy = 0i; dy <= 1i; dy = dy + 1i) {
      for (var dx = 0i; dx <= 1i; dx = dx + 1i) {
        let weight = mix(1.0 - t.x, t.x, f32(dx)) * mix(1.0 - t.y, t.y, f32(dy)) * mix(1.0 - t.z, t.z, f32(dz));
        accum = accum + volume_particle_field_values[volume_particle_field_cell(base.x + dx, base.y + dy, base.z + dz)] * weight;
      }
    }
  }
  return volume_particle_density_signal(accum);
}

fn volume_particle_density_stats(position: vec3f, radius: f32) -> ParticleDensityStats {
  let center = volume_particle_density_at(position);
  if (radius <= 0.00001) {
    return ParticleDensityStats(center, center, center, 1.0, 0.0);
  }
  return density_stats_from_axis_samples(
    center,
    volume_particle_density_at(position + vec3f(radius, 0.0, 0.0)),
    volume_particle_density_at(position - vec3f(radius, 0.0, 0.0)),
    volume_particle_density_at(position + vec3f(0.0, radius, 0.0)),
    volume_particle_density_at(position - vec3f(0.0, radius, 0.0)),
    volume_particle_density_at(position + vec3f(0.0, 0.0, radius)),
    volume_particle_density_at(position - vec3f(0.0, 0.0, radius))
  );
}

fn volume_particle_density_neighborhood(position: vec3f) -> f32 {
  return volume_particle_density_stats(position, max(0.0, uniforms.particle_density_radius)).peak;
}

fn add_volume_density_flat(flat: u32, color: vec3f, weight: f32) {
  let density = clamp(weight, 0.0, VOLUME_MAX_ACCUM);
  if (density <= 0.0) {
    return;
  }
  let base = flat * 4u;
  let color_sum = clamp(color * density, vec3f(0.0), vec3f(VOLUME_MAX_ACCUM));
  let packed = vec4i(round(vec4f(color_sum, density) * VOLUME_ACCUM_SCALE));
  volume_atomic_add_saturating(&volume_accum[base + 0u], packed.x);
  volume_atomic_add_saturating(&volume_accum[base + 1u], packed.y);
  volume_atomic_add_saturating(&volume_accum[base + 2u], packed.z);
  volume_atomic_add_saturating(&volume_accum[base + 3u], packed.w);
}

fn volume_atomic_add_saturating(slot: ptr<storage, atomic<i32>, read_write>, delta: i32) {
  if (delta <= 0i) {
    return;
  }
  var old_value = atomicLoad(slot);
  for (var attempt = 0u; attempt < 8u; attempt = attempt + 1u) {
    let clamped_old = clamp(old_value, 0i, VOLUME_ACCUM_FIXED_LIMIT);
    if (clamped_old >= VOLUME_ACCUM_FIXED_LIMIT) {
      return;
    }
    let next_value = clamped_old + min(delta, VOLUME_ACCUM_FIXED_LIMIT - clamped_old);
    let result = atomicCompareExchangeWeak(slot, old_value, next_value);
    if (result.exchanged) {
      return;
    }
    old_value = result.old_value;
  }
}

fn add_volume_density_trilinear(position: vec3f, color: vec3f, weight: f32) {
  if (weight <= 0.0 || any(position < vec3f(-1.0)) || any(position > vec3f(1.0))) {
    return;
  }
  let dims = vec3f(f32(uniforms.width), f32(uniforms.height), f32(uniforms.depth));
  let grid = (position + vec3f(1.0)) * 0.5 * dims - vec3f(0.5);
  let base = vec3i(floor(grid));
  let t = fract(grid);
  for (var dz = 0i; dz <= 1i; dz = dz + 1i) {
    for (var dy = 0i; dy <= 1i; dy = dy + 1i) {
      for (var dx = 0i; dx <= 1i; dx = dx + 1i) {
        let tri_weight = mix(1.0 - t.x, t.x, f32(dx)) * mix(1.0 - t.y, t.y, f32(dy)) * mix(1.0 - t.z, t.z, f32(dz));
        add_volume_density_flat(volume_flat_clamped(base.x + dx, base.y + dy, base.z + dz), color, weight * tri_weight);
      }
    }
  }
}

fn load_volume_accum(flat: u32) -> vec4f {
  let base = flat * 4u;
  return vec4f(
    f32(atomicLoad(&volume_accum[base + 0u])),
    f32(atomicLoad(&volume_accum[base + 1u])),
    f32(atomicLoad(&volume_accum[base + 2u])),
    f32(atomicLoad(&volume_accum[base + 3u]))
  ) / VOLUME_ACCUM_SCALE;
}

fn clamp_volume_density_value(value: vec4f) -> vec4f {
  return clamp(value, vec4f(0.0), vec4f(VOLUME_RESOLVE_MAX));
}

fn volume_density_particle_stride() -> u32 {
  return max(1u, (uniforms.particle_count + VOLUME_DENSITY_PARTICLE_BUDGET - 1u) / VOLUME_DENSITY_PARTICLE_BUDGET);
}

fn volume_blur_radius() -> i32 {
  return i32(clamp(round(max(0.0, uniforms.density_large_scale) * 0.14), 0.0, 8.0));
}

fn volume_blur_weight(offset: i32, radius: i32) -> f32 {
  if (radius <= 0i) {
    return select(0.0, 1.0, offset == 0i);
  }
  let sigma = max(0.65, f32(radius) * 0.46);
  let x = f32(offset);
  return exp(-(x * x) / max(0.0001, 2.0 * sigma * sigma));
}

fn blur_volume_axis(flat: u32, axis: u32) -> vec4f {
  let cell = volume_cell_from_flat(flat);
  let radius = volume_blur_radius();
  var sum = vec4f(0.0);
  var weight_sum = 0.0;
  for (var offset = -8i; offset <= 8i; offset = offset + 1i) {
    if (abs(offset) > radius) {
      continue;
    }
    let weight = volume_blur_weight(offset, radius);
    var sample_flat = flat;
    if (axis == 0u) {
      sample_flat = volume_flat_clamped(i32(cell.x) + offset, i32(cell.y), i32(cell.z));
    } else if (axis == 1u) {
      sample_flat = volume_flat_clamped(i32(cell.x), i32(cell.y) + offset, i32(cell.z));
    } else {
      sample_flat = volume_flat_clamped(i32(cell.x), i32(cell.y), i32(cell.z) + offset);
    }
    sum = sum + max(volume_src[sample_flat], vec4f(0.0)) * weight;
    weight_sum = weight_sum + weight;
  }
  return clamp_volume_density_value(sum / max(0.0001, weight_sum));
}

@compute @workgroup_size(256)
fn clear_volume_density(@builtin(global_invocation_id) gid: vec3u) {
  let lane = gid.x;
  if (lane >= uniforms.voxel_count * 4u) {
    return;
  }
  atomicStore(&volume_accum[lane], 0);
}

@compute @workgroup_size(256)
fn deposit_volume_density(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let stride = volume_density_particle_stride();
  let source_i = i * stride + (chunk_config.particle_offset % stride);
  if (source_i >= chunk_config.active_particle_count) {
    return;
  }
  let particle = volume_particles[source_i];
  let previous_particle = volume_particles_prev[source_i];
  let particle_lerp_t = particle_lerp_amount(previous_particle.pos_cohort.xyz, particle.pos_cohort.xyz, uniforms.render_lerp_t);
  let large_motion_fade = particle_large_motion_fade(previous_particle.pos_cohort.xyz, particle.pos_cohort.xyz, uniforms.render_lerp_t);
  let interp_pos = interpolate_particle_position(previous_particle.pos_cohort.xyz, particle.pos_cohort.xyz, particle_lerp_t);
  let visual_velocity = particle_visual_velocity(previous_particle, particle, particle_lerp_t);
  let speed_cutoff_visibility = particle_speed_cutoff_visibility(visual_velocity);
  if (speed_cutoff_visibility <= 0.000001) {
    return;
  }
  let visual_color = interpolate_particle_color(stored_particle_color(previous_particle), stored_particle_color(particle), particle_lerp_t);
  var density_fade = 1.0;
  if (uniforms.particle_density_cutoff > 0.0) {
    let cutoff_stats = volume_particle_density_stats(interp_pos, max(0.0, uniforms.particle_density_radius));
    let local_density = cutoff_stats.peak;
    let cutoff = uniforms.particle_density_cutoff;
    density_fade = smoothstep(cutoff, cutoff * PARTICLE_CUTOFF_FADE, local_density);
    if (density_fade <= 0.0) {
      return;
    }
  }
  let alpha_weight = clamp(visual_color.w / 0.045, 0.0, 2.0);
  let count_normalizer = clamp(f32(uniforms.voxel_count) / max(1.0, f32(uniforms.particle_count)), 0.02, 24.0);
  let density_weight = max(0.0, uniforms.particle_opacity) * alpha_weight * density_fade * speed_cutoff_visibility * large_motion_fade * max(0.04, uniforms.density_small_scale) * count_normalizer * f32(stride) * 0.22;
  if (density_weight <= 0.000001) {
    return;
  }
  let color = volume_particle_color(visual_velocity, particle.pos_cohort.w, visual_color);
  add_volume_density_trilinear(interp_pos, color, density_weight);
}

@compute @workgroup_size(4, 4, 4)
fn resolve_volume_density(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= uniforms.width || gid.y >= uniforms.height || gid.z >= uniforms.depth) {
    return;
  }
  let flat = (gid.z * uniforms.height + gid.y) * uniforms.width + gid.x;
  let raw = clamp_volume_density_value(load_volume_accum(flat));
  volume_small_out[flat] = raw;
  textureStore(volume_small_texture, vec3i(gid), raw);
}

@compute @workgroup_size(256)
fn blur_volume_density_x(@builtin(global_invocation_id) gid: vec3u) {
  let flat = gid.x;
  if (flat >= uniforms.voxel_count) {
    return;
  }
  volume_dst[flat] = blur_volume_axis(flat, 0u);
}

@compute @workgroup_size(256)
fn blur_volume_density_y(@builtin(global_invocation_id) gid: vec3u) {
  let flat = gid.x;
  if (flat >= uniforms.voxel_count) {
    return;
  }
  volume_dst[flat] = blur_volume_axis(flat, 1u);
}

@compute @workgroup_size(4, 4, 4)
fn blur_volume_density_z(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= uniforms.width || gid.y >= uniforms.height || gid.z >= uniforms.depth) {
    return;
  }
  let flat = (gid.z * uniforms.height + gid.y) * uniforms.width + gid.x;
  let blurred = clamp_volume_density_value(blur_volume_axis(flat, 2u));
  textureStore(volume_large_texture, vec3i(gid), blurred);
}

// --- Optimization C: half-resolution large/blurred channel ---
// The large channel is a heavily-blurred, low-weight veil, so it can be built at half grid
// resolution. We downsample the full-res resolved small channel into a half-res grid, blur it
// there (1/8 the voxels), and the raymarch samples the half-res large texture transparently
// (normalized trilinear). Half dims are derived in-shader; no new uniforms, no re-deposit, and
// the full-res path above is left untouched (toggle OFF == baseline).
fn volume_half_dims() -> vec3u {
  return vec3u((uniforms.width + 1u) / 2u, (uniforms.height + 1u) / 2u, (uniforms.depth + 1u) / 2u);
}
fn volume_half_voxel_count() -> u32 {
  let d = volume_half_dims();
  return d.x * d.y * d.z;
}
fn volume_cell_from_flat_half(flat: u32) -> vec3u {
  let d = volume_half_dims();
  return vec3u(flat % d.x, (flat / d.x) % d.y, flat / (d.x * d.y));
}
fn volume_flat_clamped_half(x: i32, y: i32, z: i32) -> u32 {
  let d = volume_half_dims();
  let cx = u32(clamp(x, 0i, i32(d.x) - 1i));
  let cy = u32(clamp(y, 0i, i32(d.y) - 1i));
  let cz = u32(clamp(z, 0i, i32(d.z) - 1i));
  return (cz * d.y + cy) * d.x + cx;
}
fn blur_volume_axis_half(flat: u32, axis: u32) -> vec4f {
  let cell = volume_cell_from_flat_half(flat);
  let radius = volume_blur_radius();
  var sum = vec4f(0.0);
  var weight_sum = 0.0;
  for (var offset = -8i; offset <= 8i; offset = offset + 1i) {
    if (abs(offset) > radius) {
      continue;
    }
    let weight = volume_blur_weight(offset, radius);
    var sample_flat = flat;
    if (axis == 0u) {
      sample_flat = volume_flat_clamped_half(i32(cell.x) + offset, i32(cell.y), i32(cell.z));
    } else if (axis == 1u) {
      sample_flat = volume_flat_clamped_half(i32(cell.x), i32(cell.y) + offset, i32(cell.z));
    } else {
      sample_flat = volume_flat_clamped_half(i32(cell.x), i32(cell.y), i32(cell.z) + offset);
    }
    sum = sum + max(volume_src[sample_flat], vec4f(0.0)) * weight;
    weight_sum = weight_sum + weight;
  }
  return clamp_volume_density_value(sum / max(0.0001, weight_sum));
}

// Average each 2x2x2 block of the full-res resolved small channel (volume_src) into one half-res
// voxel (volume_dst). Averaging (not summing) preserves per-voxel density, so brightness is kept.
@compute @workgroup_size(4, 4, 4)
fn downsample_volume_small(@builtin(global_invocation_id) gid: vec3u) {
  let half = volume_half_dims();
  if (gid.x >= half.x || gid.y >= half.y || gid.z >= half.z) {
    return;
  }
  var sum = vec4f(0.0);
  var n = 0.0;
  for (var dz = 0u; dz < 2u; dz = dz + 1u) {
    for (var dy = 0u; dy < 2u; dy = dy + 1u) {
      for (var dx = 0u; dx < 2u; dx = dx + 1u) {
        let sx = gid.x * 2u + dx;
        let sy = gid.y * 2u + dy;
        let sz = gid.z * 2u + dz;
        if (sx >= uniforms.width || sy >= uniforms.height || sz >= uniforms.depth) {
          continue;
        }
        let sflat = (sz * uniforms.height + sy) * uniforms.width + sx;
        sum = sum + max(volume_src[sflat], vec4f(0.0));
        n = n + 1.0;
      }
    }
  }
  let dflat = (gid.z * half.y + gid.y) * half.x + gid.x;
  volume_dst[dflat] = clamp_volume_density_value(sum / max(1.0, n));
}

@compute @workgroup_size(256)
fn blur_volume_density_x_half(@builtin(global_invocation_id) gid: vec3u) {
  let flat = gid.x;
  if (flat >= volume_half_voxel_count()) {
    return;
  }
  volume_dst[flat] = blur_volume_axis_half(flat, 0u);
}

@compute @workgroup_size(256)
fn blur_volume_density_y_half(@builtin(global_invocation_id) gid: vec3u) {
  let flat = gid.x;
  if (flat >= volume_half_voxel_count()) {
    return;
  }
  volume_dst[flat] = blur_volume_axis_half(flat, 1u);
}

@compute @workgroup_size(4, 4, 4)
fn blur_volume_density_z_half(@builtin(global_invocation_id) gid: vec3u) {
  let half = volume_half_dims();
  if (gid.x >= half.x || gid.y >= half.y || gid.z >= half.z) {
    return;
  }
  let flat = (gid.z * half.y + gid.y) * half.x + gid.x;
  textureStore(volume_large_texture, vec3i(gid), clamp_volume_density_value(blur_volume_axis_half(flat, 2u)));
}
`;

const liveVolumeDensityRenderShader = /* wgsl */ `
${liveSplatRenderCommon}

@group(0) @binding(0) var volume_small_texture: texture_3d<f32>;
@group(0) @binding(2) var volume_large_texture: texture_3d<f32>;
@group(0) @binding(3) var volume_density_sampler: sampler;

struct FullscreenOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

struct VolumeDensitySample {
  small: vec4f,
  large: vec4f,
};

@vertex
fn volume_density_vs(@builtin(vertex_index) vertex_index: u32) -> FullscreenOut {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -3.0),
    vec2f(-1.0, 1.0),
    vec2f(3.0, 1.0)
  );
  let position = positions[vertex_index];
  var out: FullscreenOut;
  out.position = vec4f(position, 0.0, 1.0);
  out.uv = position * 0.5 + vec2f(0.5);
  return out;
}

fn intersect_volume_box(origin: vec3f, direction: vec3f) -> vec2f {
  let inv_dir = 1.0 / direction;
  let t0 = (vec3f(-1.0) - origin) * inv_dir;
  let t1 = (vec3f(1.0) - origin) * inv_dir;
  let tmin3 = min(t0, t1);
  let tmax3 = max(t0, t1);
  let tmin = max(max(tmin3.x, tmin3.y), tmin3.z);
  let tmax = min(min(tmax3.x, tmax3.y), tmax3.z);
  return vec2f(tmin, tmax);
}

fn volume_hash21(p: vec2f) -> f32 {
  let h = dot(p, vec2f(127.1, 311.7));
  return fract(sin(h) * 43758.5453123);
}

fn volume_interleaved_gradient_noise(pixel: vec2f, frame: u32) -> f32 {
  let frame_offset = f32(frame % 64u) * 0.071;
  return fract(52.9829189 * fract(0.06711056 * (pixel.x + frame_offset) + 0.00583715 * (pixel.y - frame_offset)));
}

fn volume_density_steps() -> u32 {
  let requested = f32(uniforms.ray_steps) * clamp(uniforms.fog_step_scale, 0.125, 1.0);
  return u32(clamp(round(requested / 4.0) * 4.0, 16.0, 384.0));
}

fn sample_volume_density(position: vec3f) -> VolumeDensitySample {
  if (any(position < vec3f(-1.0)) || any(position > vec3f(1.0))) {
    return VolumeDensitySample(vec4f(0.0), vec4f(0.0));
  }
  let uvw = clamp((position + vec3f(1.0)) * 0.5, vec3f(0.0), vec3f(1.0));
  let small = textureSampleLevel(volume_small_texture, volume_density_sampler, uvw, 0.0);
  let large = textureSampleLevel(volume_large_texture, volume_density_sampler, uvw, 0.0);
  return VolumeDensitySample(max(small, vec4f(0.0)), max(large, vec4f(0.0)));
}

fn volume_density_color(sample: VolumeDensitySample, filament: f32) -> vec3f {
  let large_color_mix = mix(0.04, 0.014, clamp(filament, 0.0, 1.0));
  let color_weight = max(0.0001, sample.small.a + sample.large.a * large_color_mix);
  let color_sum = sample.small.rgb + sample.large.rgb * large_color_mix;
  let color = max(vec3f(0.0), color_sum / color_weight);
  let peak = max(0.0001, max(color.r, max(color.g, color.b)));
  let hue_color = color / peak;
  let chroma = mix(color, hue_color, 0.12 + filament * 0.18);
  return saturate_color(chroma, 1.10 + filament * 0.46);
}

@fragment
fn volume_density_fs(in: FullscreenOut) -> @location(0) vec4f {
  let aspect = uniforms.resolution.x / uniforms.resolution.y;
  var uv = in.uv * 2.0 - vec2f(1.0);
  uv.x = uv.x * aspect;
  let origin = camera_to_world(vec3f(-uniforms.camera_pan_x, -uniforms.camera_pan_y, -uniforms.distance));
  let direction = normalize(camera_to_world(vec3f(uv.x / uniforms.focal_k, uv.y / uniforms.focal_k, 1.0)));
  let hit = intersect_volume_box(origin, direction);
  var t = max(0.0, hit.x);
  let t_end = hit.y;
  if (t_end <= t) {
    return vec4f(0.0);
  }
  let steps = volume_density_steps();
  let dt = (t_end - t) / f32(steps);
  let pixel = in.uv * uniforms.resolution;
  let jitter = select(volume_hash21(pixel), volume_interleaved_gradient_noise(pixel, uniforms.fog_frame), uniforms.fog_blue_noise == 1u);
  var ray_t = t + dt * jitter;
  var marched = 0u;
  var radiance = vec3f(0.0);
  var dominant_color = vec3f(0.0);
  var dominant_signal = 0.0;
  var transmittance = 1.0;
  let ridge_mix = clamp(uniforms.density_contrast_balance, 0.0, 1.5);
  let ridge_amount = clamp(ridge_mix, 0.0, 1.0);
  let support_width = mix(0.06, 0.22, clamp(1.0 - ridge_mix * 0.50, 0.0, 1.0));
  for (var step = 0u; step < 384u; step = step + 1u) {
    if (marched >= steps || transmittance <= 0.035) {
      break;
    }
    let position = origin + direction * ray_t;
    let sample = sample_volume_density(position);
    let boundary_distance = 1.0 - max(abs(position.x), max(abs(position.y), abs(position.z)));
    let boundary_fade = smoothstep(0.02, 0.34, boundary_distance);
    let small_density = sample.small.a;
    let large_density = sample.large.a;
    let small_field = log2(1.0 + small_density * (0.85 + max(0.0, uniforms.density_small_scale) * 0.35));
    let large_field = log2(1.0 + large_density * 1.55);
    let support = smoothstep(uniforms.density_large_threshold, uniforms.density_large_threshold + support_width, large_field);
    if (uniforms.empty_space_skip == 1u && support <= max(0.001, uniforms.empty_space_threshold)) {
      let stride = min(u32(max(1.0, uniforms.empty_space_stride)), steps - marched);
      ray_t = ray_t + dt * f32(stride);
      marched = marched + stride;
      continue;
    }
    let core_peak = max(0.0, small_field - large_field * mix(0.34, 0.86, ridge_amount));
    let core_mask = smoothstep(0.09, 0.40, core_peak * (0.85 + ridge_mix * 0.45));
    let contrast_gate = smoothstep(0.045, 0.28 + ridge_mix * 0.18, core_peak);
    let structural_gate = contrast_gate * mix(0.72, 1.0, core_mask);
    let ridge_field = max(0.0, small_field * (1.02 + ridge_mix * 0.78) - large_field * (0.34 + ridge_mix * 0.62));
    let body_field = small_field * max(0.0, 1.0 - ridge_amount * 1.15);
    let density_response = max(0.0, ridge_field * (0.98 + ridge_mix * 0.58) * (0.68 + core_mask * 0.78) + body_field * 0.002) * support * boundary_fade * structural_gate;
    let filament = pow(1.0 - exp(-density_response * max(0.05, uniforms.density)), max(0.25, uniforms.density_emission_power));
    let glow_gate = smoothstep(mix(0.08, 0.20, ridge_amount), mix(0.28, 0.56, ridge_amount), filament);
    let emission_drive = clamp(filament * max(0.0, uniforms.density_contrast_gain) * (0.30 + core_mask * 0.32), 0.0, 10.0);
    let curved_emission = exp2(emission_drive * mix(0.32, 0.48, core_mask)) - 1.0;
    let exponential_emission = curved_emission * glow_gate * (0.72 + core_mask * 0.28);
    let filament_signal = filament * (0.08 + core_mask * 0.30) * glow_gate;
    let signal = (filament_signal + exponential_emission) * max(0.0, uniforms.density_pass_strength);
    if (signal > 0.000001) {
      let sample_color = volume_density_color(sample, filament);
      let step_gain = dt * particle_gain() * 0.12;
      let occlusion = mix(1.0, transmittance, clamp(uniforms.density_occlusion, 0.0, 1.0));
      let sample_signal = signal * step_gain * occlusion;
      if (sample_signal > dominant_signal) {
        dominant_signal = sample_signal;
        dominant_color = sample_color;
      }
      radiance = radiance + sample_color * sample_signal;
      let absorption = clamp(filament * uniforms.density_occlusion * dt * 5.5, 0.0, 0.34);
      transmittance = transmittance * (1.0 - absorption);
    }
    ray_t = ray_t + dt;
    marched = marched + 1u;
  }
  if (max(radiance.r, max(radiance.g, radiance.b)) <= 0.000001) {
    discard;
  }
  let radiance_peak = max(0.0001, max(radiance.r, max(radiance.g, radiance.b)));
  let dominant_peak = max(0.0001, max(dominant_color.r, max(dominant_color.g, dominant_color.b)));
  let dominant_hue = dominant_color / dominant_peak;
  let chroma_mix = clamp(0.04 + ridge_amount * 0.12, 0.0, 0.18);
  let final_radiance = mix(radiance, dominant_hue * radiance_peak, chroma_mix) * 0.08;
  return vec4f(max(final_radiance, vec3f(0.0)), clamp(1.0 - transmittance, 0.0, 1.0));
}
`;

export const liveShaderSources = {
  deposit: liveComputeDepositShader,
  visualDeposit: liveComputeVisualDepositShader,
  field: makeLiveComputeFieldShader(false),
  fieldWithTextureMirror: makeLiveComputeFieldShader(true),
  particleUpdate: makeLiveComputeParticleShader(false),
  particleUpdateTextureSensing: makeLiveComputeParticleShader(true),
  particleUpdateEcology: makeLiveComputeParticleEcologyShader(false),
  ecologyDeposit: liveComputeEcologyDepositShader,
  ecologyUpdate: liveComputeEcologyUpdateShader,
  splatCommon: liveSplatRenderCommon,
  particleRender: liveParticleRenderShader,
  densitySplat: liveDensitySplatShader,
  densityComposite: liveDensityCompositeShader,
  accumulationSplat: liveAccumulationSplatShader,
  accumulationComposite: liveAccumulationCompositeShader,
  volumeDensityCompute: liveVolumeDensityComputeShader,
  volumeDensityRender: liveVolumeDensityRenderShader,
  post: livePostShader
};

const liveOverlayShader = /* wgsl */ `
struct RenderUniforms {
  resolution: vec2f,
  width: u32,
  height: u32,
  depth: u32,
  particle_count: u32,
  voxel_count: u32,
  timestep: u32,
  density: f32,
  exposure: f32,
  focus_distance: f32,
  aperture: f32,
  overlay: u32,
  palette: u32,
  filament: f32,
  yaw_cos: f32,
  yaw_sin: f32,
  pitch_cos: f32,
  pitch_sin: f32,
  distance: f32,
  ray_steps: u32,
  fog_step_scale: f32,
  fog_temporal_blend: f32,
  fog_blue_noise: u32,
  fog_frame: u32,
  field_texture_mode: u32,
  empty_space_skip: u32,
  empty_space_threshold: f32,
  empty_space_stride: f32,
  particle_size_px: f32,
  particle_min_px: f32,
  particle_max_px: f32,
  particle_opacity: f32,
  particle_blend_mode: u32,
  particle_density_cutoff: f32,
  particle_density_radius: f32,
  trail_opacity: f32,
  trail_threshold: f32,
  render_layer: u32,
  camera_pan_x: f32,
  camera_pan_y: f32,
  trail_color_mode: u32,
  fog_tint_r: f32,
  fog_tint_g: f32,
  fog_tint_b: f32,
  particle_tint_r: f32,
  particle_tint_g: f32,
  particle_tint_b: f32,
  scene_brightness: f32,
  particle_brightness: f32,
  fog_brightness: f32,
  particle_color_mode: u32,
  particle_velocity_stretch: u32,
  particle_stretch: f32,
  particle_gradient_sensitivity: f32,
  focal_k: f32,
  dof_blur: f32,
  dof_debug: u32,
  render_lerp_t: f32,
  dof_enabled: u32,
  density_pass_strength: f32,
  density_small_scale: f32,
  density_large_scale: f32,
  density_large_threshold: f32,
  density_contrast_gain: f32,
  density_contrast_balance: f32,
  density_emission_power: f32,
  density_occlusion: f32,
  cohorts: u32,
  particle_slow_cutoff: f32,
  particle_glow_core: f32,
  particle_hot_core: f32,
  accumulation_strength: f32,
  accumulation_radius: f32,
  accumulation_curve: f32,
  accumulation_memory: f32,
  accumulation_noise_reject: f32,
  particle_density_reference: f32,
  particle_density_normalize: f32,
  particle_density_softness: f32,
  particle_support_mask: f32,
  particle_support_radius: f32,
  particle_support_neighbors: f32,
  particle_support_flow: f32,
  particle_support_grid_size: u32,
  particle_stretch_min: f32,
  particle_stretch_speed: f32,
  particle_speed_cutoff: f32,
  variation_master: f32,
  variation_time: f32,
  variation_drift: f32,
  variation_noise_mix: f32,
  variation_freq: f32,
  variation_octaves: u32,
  variation_gain: f32,
  variation_lacunarity: f32,
  variation_size_amount: f32,
  variation_size_curve: f32,
  variation_size_min: f32,
  variation_size_max: f32,
  variation_bright_amount: f32,
  variation_bright_curve: f32,
  variation_bright_min: f32,
  variation_bright_max: f32,
  variation_opacity_amount: f32,
  variation_opacity_curve: f32,
  variation_opacity_min: f32,
  variation_opacity_max: f32,
  variation_color_amount: f32,
  variation_color_curve: f32,
  variation_color_min: f32,
  variation_color_max: f32,
  domain_shape: u32,
  audio_low: f32,
  audio_mid: f32,
  audio_high: f32,
};

@group(0) @binding(0) var<uniform> uniforms: RenderUniforms;

struct FullscreenOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn fullscreen_vs(@builtin(vertex_index) vertex_index: u32) -> FullscreenOut {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -3.0),
    vec2f(-1.0, 1.0),
    vec2f(3.0, 1.0)
  );
  let position = positions[vertex_index];
  var out: FullscreenOut;
  out.position = vec4f(position, 0.0, 1.0);
  out.uv = position * 0.5 + vec2f(0.5);
  return out;
}

@fragment
fn overlay_fs(in: FullscreenOut) -> @location(0) vec4f {
  let uv = in.uv;
  let frame = step(0.045, uv.x) * step(0.045, uv.y) * step(uv.x, 0.955) * step(uv.y, 0.955);
  let outer = step(0.034, uv.x) * step(0.034, uv.y) * step(uv.x, 0.966) * step(uv.y, 0.966);
  let border = (outer - frame) * f32(uniforms.overlay);
  return vec4f(vec3f(1.0, 0.82, 0.42) * border * 0.42, border * 0.55);
}
`;
