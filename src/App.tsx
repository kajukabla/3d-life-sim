import { createContext, memo, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent, PointerEvent, ReactNode } from "react";
import { sliderPosToValue, sliderValueToPos } from "./sliderCurve";
import type { RenderControls } from "./renderControls";
import {
  defaultGpuSim3dConfig,
  GpuSim3dSummary,
  runWebGpuComputeConformance,
  summarizeCompute3d
} from "./gpuSim3d";
import {
  defaultLiveGpu3dConfig,
  LiveGpu3dConfig,
  LiveGpu3dDiagnostics,
  MAX_SIMULATION_SPEED,
  RealtimeGpuSim3d
} from "./realtimeGpuSim3d";
import { analyzeParticleOscillation, type ParticleOscillationReport } from "./particleOscillationProbe";
import { createAutomationRecorder, type AutomationRecording } from "./automationRecorder";
import { createPerformanceAudioRecorder, audioExtensionFor, type PerformanceAudioRecording } from "./performanceAudioRecorder";
import { getLivePreset, liveParticleStep, livePresets, maxLiveParticles, minLiveParticles } from "./livePresets";
import {
  clampRayResolution,
  maxRayResolution,
  minRayResolution,
  rayResolutionStep
} from "./renderTarget";
import { probeWebGpu, WebGpuDiagnostics } from "./webgpu";
import {
  SwimmerTracker,
  pickParticle,
  springStep,
  epsForParticleCount,
  DEFAULT_TRACKER_PARAMS,
  type Ray,
  type Spring
} from "./cameraTracking";
import { rayForNdc, worldToCamera, focalToFocus, focusToFocal, type CameraMath3d, type Vec3 } from "./cameraMath3d";
import { TIMELINE_FPS, TIMELINE_TOTAL_FRAMES, advanceFrame, normalizeLoop, planSeek, type TimelineState } from "./timeline";
import { useTimelineTransport } from "./useTimelineTransport";
import { TimelineBar } from "./TimelineBar";
import { createCameraRecorder, type CameraPose } from "./cameraRecorder";
import { audioMicFromLaunch, audioReactiveUrlFromLaunch, chooseBootSettingsId, demoLaunchFromSearch, embedFromLaunch, parallelPipelinesFromLaunch, shouldStartPlaying } from "./launchOptions";
import { startMicAudio, type MicAudioStatus, type MicAudioController } from "./micAudio";
import {
  connectAudioReactiveSocket,
  type AudioAnalysisFrame,
  type AudioInputDeviceInfo,
  type AudioReactiveMapping,
  type AudioReactiveSocketController,
  type AudioReactiveSocketStatus
} from "./audioReactive";
import {
  applyAudioModulation,
  audioBuckets,
  createAudioModulationRuntime,
  defaultAudioPanelState,
  defaultSliderModulation,
  effectiveSliderRange,
  getSliderModulation,
  hasEnabledSliderModulation,
  sanitizeAudioPanelState,
  sanitizeSliderModulationSettings,
  updateAudioMeters,
  type AudioBucket,
  type AudioModulationRuntime,
  type AudioPanelState,
  type SliderModulationConfig,
  type SliderModulationSettings
} from "./audioModulation";
import {
  hasEnabledMidiMapping,
  mapMidiValue,
  midiAllInputsId,
  midiControlIndexKey,
  midiControlLabel,
  midiMappingFromControl,
  midiMappingMatches,
  midiMessageIndexKeys,
  parseMidiControlMessage,
  type MidiControlMessage,
  type SliderMidiMapping
} from "./midiMapping";
import { selectDemoDriftCandidates, DEMO_DRIFT_DENYLIST } from "./demoDrift";
import { controlHint } from "./controlHints";

declare global {
  interface Window {
    __fluoddityTrackState?: () => { active: boolean; members: number; centroid: readonly [number, number, number]; panX: number; panY: number; distance: number };
    __fluoddityDiagnostics?: () => Record<string, unknown>;
    __fluoddityCaptureState?: () => Promise<{
      manifest: Record<string, unknown>;
      fieldB64: string;
      particlesB64: string;
    } | null>;
    __fluoddityParticleOscillationProbe?: (options?: {
      steps?: number;
      particleCount?: number;
      maxParticles?: number;
      topK?: number;
      snapDistance?: number;
      reset?: boolean;
      config?: Partial<LiveGpu3dConfig>;
    }) => Promise<(ParticleOscillationReport & {
      capStride: number;
      config: Pick<LiveGpu3dConfig, "particleCount" | "simulationSpeed" | "dt" | "sensorAngle" | "sensorDistance" | "drag" | "strafePower" | "strafeMomentum" | "axialForce" | "lateralForce" | "globalForceMult" | "hazardRate" | "cohorts" | "colorByCohort" | "boundaryMode">;
    }) | null>;
    // Deterministic byte-identity probe: reset -> render -> advanceSteps -> captureParticles -> FNV-1a
    // hash of the raw f32 bytes. Same inputs on one device => identical hash, so the emergent-behavior
    // E2E can assert all-zero == baseline and behaviorX>0 != off.
    __fluoddityHashParticles?: (options?: {
      steps?: number;
      particleCount?: number;
      maxParticles?: number;
      reset?: boolean;
      config?: Partial<LiveGpu3dConfig>;
    }) => Promise<{ hash: string; fieldHash: string; particleCount: number; steps: number; maxDensity: number } | null>;
    // Wall-clock benchmark of N whole sim steps (advanceSteps awaits GPU completion). Used to prove
    // "off costs the same as baseline" and to measure each behavior's GPU cost when on.
    __fluoddityBenchSteps?: (options?: {
      steps?: number;
      reps?: number;
      warmup?: number;
      particleCount?: number;
      config?: Partial<LiveGpu3dConfig>;
    }) => Promise<{ medianMs: number; samples: number[]; steps: number; particleCount: number } | null>;
    __fluoddityLoadParityScene?: (spec: {
      name?: string;
      resolution?: { width: number; height: number };
      camera: { yaw: number; pitch: number; distance: number; panX: number; panY: number; fov: number; focusDistance: number; aperture: number; dofBlur: number; dofEnabled?: boolean };
      particles: Array<{ x: number; y: number; z: number; rgb?: [number, number, number]; cohort?: number; velocity?: [number, number, number]; hue?: number; sat?: number; val?: number; alpha?: number }>;
      renderControls?: Partial<RenderControls>;
      seedField?: boolean;
    }) => Promise<{ canvasWidth: number; canvasHeight: number; particleCount: number; hdr?: { width: number; height: number; pixelsB64: string } | null; hdrError?: string }>;
    __parityReady?: boolean;
    __fluoddityReadCanvasHDR?: () => Promise<{ width: number; height: number; pixelsB64: string } | null>;
    // Offline HDR replay-export hooks (driven by tools/hdr-export). ReplayBegin suspends the live
    // loop, seeds the sim config, and resets to frame 0. ReplayFrame steps the sim by the recorded
    // timestep delta, renders with the recorded controls (camera baked in) + audio, and returns the
    // linear HDR float pixels. ReplayEnd resumes the live loop.
    __fluoddityReplayBegin?: (firstConfig: LiveGpu3dConfig) => Promise<void>;
    __fluoddityReplayFrame?: (spec: {
      timestep: number;
      controls: RenderControls;
      config: LiveGpu3dConfig;
      audio: { low: number; mid: number; high: number };
      // Optional forced render aspect (width / height). The exporter passes 9/16 for portrait so the
      // output is exactly 9:16 regardless of the replay browser window; omitted = follow the canvas.
      aspect?: number;
    }) => Promise<{ width: number; height: number; bytesPerRow: number; halfB64: string } | null>;
    __fluoddityReplayEnd?: () => void;
    // Timeline evidence hooks: deterministically seek to a frame (resolves after that
    // frame renders), and set the orbit camera instantly (no smoothing) for reproducible
    // camera moves during sequence capture.
    __fluoddityTimelineSeek?: (frame: number) => Promise<number>;
    __fluodditySetCameraOrbit?: (pose: { yaw: number; pitch: number; distance: number; panX?: number; panY?: number }) => void;
    __fluodditySetPlaying?: (playing: boolean) => void;
    __fluodditySetRenderControl?: (patch: Partial<RenderControls>) => void;
    __fluodditySetSplatPrepass?: (enabled: boolean) => void;
    __fluodditySetComputeSplat?: (enabled: boolean) => void;
    __fluodditySetFieldTextureSensing?: (enabled: boolean) => void;
    __fluodditySetParticleSort?: (enabled: boolean) => void;
    // Init-time flag (see RealtimeGpuSim3d.parallelPipelineCompile): only takes effect on the
    // next pipeline build, so callers must set it before first init (or use ?parallelPipelines=1).
    __fluodditySetParallelPipelineCompile?: (enabled: boolean) => void;
    __fluoddityApplyPreset?: (preset: SavedSettingsPreset) => void;
    __fluoddityInjectAudioFrame?: (frame: AudioAnalysisFrame, dtSec?: number) => void;
    __fluoddityInjectMidiMessage?: (data: ArrayLike<number>, inputId?: string, inputName?: string) => void;
    __fluoddityAudioReactive?: () => AudioReactiveSocketStatus & { enabled: boolean };
    __fluoddityMicAudio?: () => { status: MicAudioStatus; frameCount: number; lastFrame: AudioAnalysisFrame | null };
  }
}

const deterministicSeed = 3405691582;
type DisplayMode = "live";
type SavedSettingsPreset = {
  version: 1 | 2;
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  presetId: string;
  displayMode: DisplayMode;
  controls: RenderControls;
  liveConfig: LiveGpu3dConfig;
  ui: SavedUiState;
  audio: SavedAudioState;
  fileBacked?: boolean;
};

type SavedAudioState = {
  panel: AudioPanelState;
  sliders: SliderModulationSettings;
};

type SavedUiState = {
  playing: boolean;
  overlay: boolean;
  viewLocked: boolean;
  demoMode: boolean;
  frameCap: number;
  cameraOrbit: {
    enabled: boolean;
    speed: number;
  };
  timeline: {
    enabled: boolean;
    state: TimelineState;
  };
  trackingControls: TrackingControls;
};

type FilePickerFileType = {
  description: string;
  accept: Record<string, string[]>;
};

type WritableFileHandle = {
  write: (data: Blob) => Promise<void>;
  close: () => Promise<void>;
};

type SaveFileHandle = {
  name: string;
  createWritable: () => Promise<WritableFileHandle>;
};

type OpenFileHandle = {
  name: string;
  getFile: () => Promise<File>;
};

type FilePickerWindow = Window & {
  showOpenFilePicker?: (options: {
    multiple?: boolean;
    types?: FilePickerFileType[];
    excludeAcceptAllOption?: boolean;
  }) => Promise<OpenFileHandle[]>;
  showSaveFilePicker?: (options: {
    suggestedName?: string;
    types?: FilePickerFileType[];
    excludeAcceptAllOption?: boolean;
  }) => Promise<SaveFileHandle>;
};

type MidiInputState = "connected" | "disconnected";
type MidiInputConnection = "open" | "closed" | "pending";

type MidiMessageEventLike = Event & {
  data?: ArrayLike<number>;
};

type MidiInputLike = EventTarget & {
  id: string;
  name?: string | null;
  state?: MidiInputState;
  connection?: MidiInputConnection;
  onmidimessage?: ((event: MidiMessageEventLike) => void) | null;
};

type MidiAccessLike = EventTarget & {
  inputs: { values: () => IterableIterator<MidiInputLike> };
  onstatechange?: (() => void) | null;
};

type MidiInputInfo = {
  id: string;
  name: string;
  state: MidiInputState;
  connection: MidiInputConnection;
};

type MidiPanelStatus = {
  supported: boolean;
  enabled: boolean;
  error: string | null;
  messageCount: number;
  lastMessage: string | null;
};

type OrbitCamera = {
  yaw: number;
  pitch: number;
  distance: number;
  panX: number;
  panY: number;
  fov: number;
  targetYaw: number;
  targetPitch: number;
  targetDistance: number;
  targetPanX: number;
  targetPanY: number;
  targetFov: number;
};

type ViewportDragMode = "orbit" | "pan";

const cameraDefaults = {
  yaw: 0.42,
  pitch: -0.32,
  distance: 3.15
};

type TrackingControls = {
  follow: boolean;
  look: boolean;
  followSpeed: number;
  followSmoothing: number;
  lookSpeed: number;
  lookSmoothing: number;
  followDistance: number;
  followHeight: number;
  cohesion: number;
};

const trackingDefaults: TrackingControls = {
  follow: false,
  look: false,
  followSpeed: 3,
  followSmoothing: 1,
  lookSpeed: 4,
  lookSmoothing: 1,
  followDistance: 3.15,
  followHeight: 0,
  cohesion: 0.5
};

type SliderRegistryEntry = {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
};

type SliderModulationContextValue = {
  expandedKey: string | null;
  settings: SliderModulationSettings;
  midiStatus: MidiPanelStatus;
  midiLearningKey: string | null;
  register: (key: string, entry: SliderRegistryEntry | null) => void;
  setExpandedKey: (key: string | null) => void;
  updateConfig: (key: string, config: SliderModulationConfig) => void;
  startMidiLearn: (key: string) => void;
  cancelMidiLearn: () => void;
};

const SliderModulationContext = createContext<SliderModulationContextValue | null>(null);

const manualSliderAudioMapping: AudioReactiveMapping = {
  id: "manual-slider-audio-modulation",
  rules: []
};

const renderControlSliderTargets = {
  "p-bright": "particleBrightness",
  "particle-size-slider": "particleSizePx",
  "p-opacity": "particleOpacity",
  "particle-density-cutoff-slider": "particleDensityCutoff",
  "particle-density-radius-slider": "particleDensityRadius",
  "particle-density-normalize-slider": "particleDensityNormalize",
  "particle-density-softness-slider": "particleDensitySoftness",
  "particle-support-mask-slider": "particleSupportMask",
  "particle-support-radius-slider": "particleSupportRadius",
  "particle-support-neighbors-slider": "particleSupportNeighbors",
  "particle-support-flow-slider": "particleSupportFlow",
  "particle-stretch-min-slider": "particleStretchMin",
  "particle-stretch-max-slider": "particleStretch",
  "particle-stretch-speed-slider": "particleStretchSpeed",
  "particle-speed-cutoff-slider": "particleSpeedCutoff",
  "particle-slow-cutoff-slider": "particleSlowCutoff",
  "particle-exponent-slider": "particleExponent",
  "particle-brightness-boost-slider": "particleBrightnessBoost",
  "particle-support-smoothing-slider": "particleSupportSmoothing",
  "particle-haze-cull-slider": "particleHazeCull",
  "particle-despeckle-slider": "particleDespeckle",
  exposure: "exposure",
  "scene-gain": "sceneBrightness",
  "bloom-strength-slider": "bloomStrength",
  "bloom-threshold-slider": "bloomThreshold",
  "bloom-radius-slider": "bloomRadius",
  "color-saturation-slider": "colorSaturation",
  "color-contrast-slider": "colorContrast",
  "chromatic-aberration-slider": "chromaticAberration",
  "vignette-strength-slider": "vignetteStrength",
  "vignette-softness-slider": "vignetteSoftness",
  "streak-strength-slider": "streakStrength",
  "streak-length-slider": "streakLength",
  "streak-vertical-slider": "streakVertical",
  "flare-height-slider": "flareHeight",
  "flare-cutoff-slider": "flareCutoff",
  "fov-slider": "fov",
  "aperture-slider": "aperture",
  "dof-blur-slider": "dofBlur",
  "variation-master-slider": "variationMaster",
  "variation-drift-slider": "variationDrift",
  "variation-noise-mix-slider": "variationNoiseMix",
  "variation-freq-slider": "variationFreq",
  "variation-octaves-slider": "variationOctaves",
  "variation-gain-slider": "variationGain",
  "variation-lacunarity-slider": "variationLacunarity",
  "variation-size-amount-slider": "variationSizeAmount",
  "variation-size-curve-slider": "variationSizeCurve",
  "variation-size-min-slider": "variationSizeMin",
  "variation-size-max-slider": "variationSizeMax",
  "variation-bright-amount-slider": "variationBrightAmount",
  "variation-bright-curve-slider": "variationBrightCurve",
  "variation-bright-min-slider": "variationBrightMin",
  "variation-bright-max-slider": "variationBrightMax",
  "variation-opacity-amount-slider": "variationOpacityAmount",
  "variation-opacity-curve-slider": "variationOpacityCurve",
  "variation-opacity-min-slider": "variationOpacityMin",
  "variation-opacity-max-slider": "variationOpacityMax",
  "variation-color-amount-slider": "variationColorAmount",
  "variation-color-curve-slider": "variationColorCurve",
  "variation-color-min-slider": "variationColorMin",
  "variation-color-max-slider": "variationColorMax"
} as const satisfies Partial<Record<string, keyof RenderControls>>;

const PICK_CONE_RADIUS = 0.04;
const RECLUSTER_INTERVAL_MS = 66; // ~15 Hz
const TRACKING_PARTICLE_READBACK_LIMIT = 262144;
const liveParticleFloatCount = 8;

const minCameraDistance = 0;
const maxCameraDistance = 100000;
const minCameraPitch = -1.32;
const maxCameraPitch = 1.32;
const minCameraPan = -4;
const maxCameraPan = 4;
const cameraPanFocal = 1.85 * 0.92;
const minVolumeSize = 32;
const maxVolumeSize = 96;
const volumeStep = 8;
const minTrailRadius = 0.0015;
const maxTrailRadiusCap = 0.28;
const savedSettingsStorageKey = "fluoddity-3d.saved-settings.v1";
const filePresetModules = import.meta.glob<string>("../Presets/*", {
  eager: true,
  import: "default",
  query: "?raw"
});
// Curated sweep order (first = velocity signature, last = audio signature) then the standalone
// gradient ramps, which stay valid for preset compatibility but aren't in the slider sweep.
const particleColorModes = [
  "velocity",
  "velocity-inferno",
  "velocity-viridis",
  "velocity-spectral",
  "velocity-cosmic",
  "velocity-ice",
  "solid",
  "audio-magma",
  "audio-viridis",
  "audio-turbo",
  "audio-cosmic",
  "audio-ice",
  "audio-ember",
  "audio-plasma",
  "cohort",
  "audio",
  "gradient-inferno",
  "gradient-magma",
  "gradient-viridis",
  "gradient-turbo",
  "gradient-rainbow",
  "gradient-spectral",
  "gradient-plasma",
  "gradient-cosmic",
  "gradient-ice",
  "gradient-ember"
] as const satisfies readonly RenderControls["particleColorMode"][];

// The curated 16-mode sweep the P Color slider/select expose, in order. Mapped left-to-right
// for live control: velocity family -> solid -> audio family -> cohort -> audio signature.
const particleColorSweep: [RenderControls["particleColorMode"], string][] = [
  ["velocity", "Velocity / Speed"],
  ["velocity-inferno", "Velocity · Inferno"],
  ["velocity-viridis", "Velocity · Viridis"],
  ["velocity-spectral", "Velocity · Spectral"],
  ["velocity-cosmic", "Velocity · Cosmic"],
  ["velocity-ice", "Velocity · Ice"],
  ["solid", "Solid Color"],
  ["audio-magma", "Audio · Magma"],
  ["audio-viridis", "Audio · Viridis"],
  ["audio-turbo", "Audio · Turbo"],
  ["audio-cosmic", "Audio · Cosmic"],
  ["audio-ice", "Audio · Ice"],
  ["audio-ember", "Audio · Ember"],
  ["audio-plasma", "Audio · Plasma"],
  ["cohort", "Cohort"],
  ["audio", "Audio Reactive"]
];

type SliderRange = {
  min: number;
  max: number;
  step: number;
};

const cameraDistanceSliderRange: SliderRange = { min: 0, max: 25, step: 0.001 };

const maxFocusDistanceUniform = 18.5;
// Focal distance in world camera-space units. Maps to the stored focusDistance via 1.85/d
// (focalToFocus). The near end is intentionally lens-close; exact zero would be singular.
// 0.1 -> 18.5 (near), 25 -> ~0.074 (far). Default focusDistance 0.54 -> ~3.43.
const focalDistanceSliderRange: SliderRange = { min: 0.1, max: 25, step: 0.01 };

// Auto-orbit yaw speed in radians/second; negative reverses direction.
const orbitSpeedSliderRange: SliderRange = { min: -1.5, max: 1.5, step: 0.01 };
const defaultCameraOrbit = { enabled: false, speed: 0.3 };

const trackingSliderRanges = {
  followSpeed: { min: 0.5, max: 10, step: 0.1 },
  followSmoothing: { min: 0.4, max: 2, step: 0.05 },
  lookSpeed: { min: 0.5, max: 10, step: 0.1 },
  lookSmoothing: { min: 0.4, max: 2, step: 0.05 },
  followDistance: { min: 0.3, max: 12, step: 0.05 },
  followHeight: { min: -3, max: 3, step: 0.05 },
  cohesion: { min: 0, max: 1, step: 0.02 }
} satisfies Record<keyof Omit<TrackingControls, "follow" | "look">, SliderRange>;

// Vertical FOV (degrees) whose K = 1/tan(fov/2) equals the legacy 1.85*0.92 focal, so the
// default leaves the verified projection unchanged.
const defaultFovDegrees = (2 * Math.atan(1 / (1.85 * 0.92)) * 180) / Math.PI;

const renderSliderRanges = {
  bloomStrength: { min: 0, max: 1, step: 0.01 },
  bloomThreshold: { min: 0, max: 2, step: 0.05 },
  bloomRadius: { min: 0.5, max: 3, step: 0.05 },
  colorSaturation: { min: 0, max: 2, step: 0.01 },
  colorContrast: { min: 0.5, max: 1.8, step: 0.01 },
  chromaticAberration: { min: 0, max: 1, step: 0.01 },
  vignetteStrength: { min: 0, max: 1, step: 0.01 },
  vignetteSoftness: { min: 0, max: 1, step: 0.01 },
  streakStrength: { min: 0, max: 1, step: 0.01 },
  streakLength: { min: 0, max: 1, step: 0.01 },
  streakVertical: { min: 0, max: 1, step: 0.01 },
  flareHeight: { min: 0, max: 1, step: 0.01 },
  flareCutoff: { min: 0, max: 1, step: 0.01 },
  ribbonFraction: { min: 0, max: 1, step: 0.01 },
  ribbonWidth: { min: 0.05, max: 3, step: 0.05 },
  ribbonTaper: { min: 0, max: 1, step: 0.01 },
  ribbonLength: { min: 2, max: 32, step: 1 },
  ribbonJoints: { min: 2, max: 32, step: 1 },
  ribbonFadeStart: { min: 0, max: 1, step: 0.01 },
  ribbonEdgeFade: { min: 0, max: 1, step: 0.01 },
  density: { min: 0.25, max: 2.8, step: 0.05 },
  exposure: { min: 0.4, max: 2.6, step: 0.05 },
  sceneBrightness: { min: 0, max: 2.6, step: 0.05 },
  fov: { min: 20, max: 110, step: 0.5 },
  aperture: { min: 0.05, max: 0.9, step: 0.02 },
  focusDistance: { min: 0.05, max: maxFocusDistanceUniform, step: 0.02 },
  dofBlur: { min: 0, max: 4, step: 0.05 },
  raySteps: { min: 8, max: 512, step: 8 },
  fogRenderScale: { min: 0.25, max: 1, step: 0.05 },
  fogStepScale: { min: 0.125, max: 1, step: 0.025 },
  fogTemporalBlend: { min: 0, max: 0.96, step: 0.02 },
  emptySpaceThreshold: { min: 0, max: 0.2, step: 0.005 },
  emptySpaceStride: { min: 1, max: 8, step: 1 },
  particleSizePx: { min: 0.2, max: 24, step: 0.1 },
  particleMinPx: { min: 0, max: 8, step: 0.1 },
  particleMaxPx: { min: 0.2, max: 32, step: 0.1 },
  particleOpacity: { min: 0, max: 1, step: 0.02 },
  particleBrightness: { min: 0, max: 8, step: 0.05 },
  particleStretch: { min: 0, max: 6, step: 0.05 },
  particleStretchMin: { min: 0, max: 6, step: 0.05 },
  particleStretchSpeed: { min: 0.004, max: 0.12, step: 0.001 },
  particleSpeedCutoff: { min: 0, max: 0.12, step: 0.001 },
  particleSlowCutoff: { min: 0, max: 0.12, step: 0.001 },
  particleGlowCore: { min: 0, max: 1, step: 0.01 },
  particleHotCore: { min: 0, max: 1, step: 0.01 },
  particleExponent: { min: 0, max: 10, step: 0.1 },
  particleBrightnessBoost: { min: 0, max: 5, step: 0.05 },
  particleSupportSmoothing: { min: 0, max: 1, step: 0.01 },
  particleHazeCull: { min: 0, max: 1, step: 0.01 },
  particleDespeckle: { min: 0, max: 1, step: 0.01 },
  particleDensityCutoff: { min: 0, max: 0.03, step: 0.0001 },
  particleDensityRadius: { min: 0, max: 0.18, step: 0.005 },
  particleDensityNormalize: { min: 0, max: 1, step: 0.01 },
  particleDensitySoftness: { min: 0.02, max: 1, step: 0.01 },
  particleSupportMask: { min: 0, max: 1, step: 0.01 },
  particleSupportRadius: { min: 0.35, max: 1.75, step: 0.05 },
  particleSupportNeighbors: { min: 1, max: 24, step: 1 },
  particleSupportFlow: { min: 0, max: 1, step: 0.01 },
  densityPassStrength: { min: 0, max: 8, step: 0.05 },
  densitySmallScale: { min: 0.25, max: 5, step: 0.05 },
  densityLargeScale: { min: 1, max: 36, step: 0.1 },
  densityLargeThreshold: { min: 0, max: 4, step: 0.01 },
  densityContrastGain: { min: 0, max: 20, step: 0.05 },
  densityContrastBalance: { min: 0, max: 1.5, step: 0.02 },
  densityEmissionPower: { min: 0.25, max: 5, step: 0.05 },
  densityOcclusion: { min: 0, max: 1, step: 0.02 },
  accumulationStrength: { min: 0, max: 12, step: 0.05 },
  accumulationRadius: { min: 0.2, max: 4, step: 0.05 },
  accumulationCurve: { min: 0.5, max: 6, step: 0.05 },
  accumulationMemory: { min: 0, max: 0.96, step: 0.01 },
  accumulationNoiseReject: { min: 0, max: 1, step: 0.01 },
  trailOpacity: { min: 0, max: 2.5, step: 0.05 },
  fogBrightness: { min: 0, max: 8, step: 0.05 },
  trailThreshold: { min: 0, max: 0.28, step: 0.005 },
  filament: { min: 0, max: 1, step: 0.02 },
  cameraYaw: { min: -3.14, max: 3.14, step: 0.02 },
  cameraPitch: { min: minCameraPitch, max: maxCameraPitch, step: 0.02 },
  cameraDistance: cameraDistanceSliderRange,
  variationMaster: { min: 0, max: 1, step: 0.01 },
  variationDrift: { min: 0, max: 1, step: 0.01 },
  variationNoiseMix: { min: 0, max: 1, step: 0.01 },
  variationFreq: { min: 0.2, max: 12, step: 0.05 },
  variationOctaves: { min: 1, max: 4, step: 1 },
  variationGain: { min: 0.2, max: 0.8, step: 0.01 },
  variationLacunarity: { min: 1.5, max: 3, step: 0.05 },
  variationSizeAmount: { min: 0, max: 1, step: 0.01 },
  variationSizeCurve: { min: 0.3, max: 6, step: 0.05 },
  variationSizeMin: { min: 0, max: 3, step: 0.01 },
  variationSizeMax: { min: 0, max: 3, step: 0.01 },
  variationBrightAmount: { min: 0, max: 1, step: 0.01 },
  variationBrightCurve: { min: 0.3, max: 6, step: 0.05 },
  variationBrightMin: { min: 0, max: 3, step: 0.01 },
  variationBrightMax: { min: 0, max: 3, step: 0.01 },
  variationOpacityAmount: { min: 0, max: 1, step: 0.01 },
  variationOpacityCurve: { min: 0.3, max: 6, step: 0.05 },
  variationOpacityMin: { min: 0, max: 2, step: 0.01 },
  variationOpacityMax: { min: 0, max: 2, step: 0.01 },
  variationColorAmount: { min: 0, max: 1, step: 0.01 },
  variationColorCurve: { min: 0.3, max: 6, step: 0.05 },
  variationColorMin: { min: -0.5, max: 0.5, step: 0.01 },
  variationColorMax: { min: -0.5, max: 0.5, step: 0.01 }
} satisfies Record<
  | "bloomStrength"
  | "bloomThreshold"
  | "bloomRadius"
  | "colorSaturation"
  | "colorContrast"
  | "chromaticAberration"
  | "vignetteStrength"
  | "vignetteSoftness"
  | "streakStrength"
  | "streakLength"
  | "streakVertical"
  | "flareHeight"
  | "flareCutoff"
  | "ribbonFraction"
  | "ribbonWidth"
  | "ribbonTaper"
  | "ribbonLength"
  | "ribbonJoints"
  | "ribbonFadeStart"
  | "ribbonEdgeFade"
  | "density"
  | "exposure"
  | "sceneBrightness"
  | "fov"
  | "aperture"
  | "focusDistance"
  | "dofBlur"
  | "raySteps"
  | "fogRenderScale"
  | "fogStepScale"
  | "fogTemporalBlend"
  | "emptySpaceThreshold"
  | "emptySpaceStride"
  | "particleSizePx"
  | "particleMinPx"
  | "particleMaxPx"
  | "particleOpacity"
  | "particleBrightness"
  | "particleStretch"
  | "particleStretchMin"
  | "particleStretchSpeed"
  | "particleSpeedCutoff"
  | "particleSlowCutoff"
  | "particleGlowCore"
  | "particleHotCore"
  | "particleExponent"
  | "particleBrightnessBoost"
  | "particleSupportSmoothing"
  | "particleHazeCull"
  | "particleDespeckle"
  | "particleDensityCutoff"
  | "particleDensityRadius"
  | "particleDensityNormalize"
  | "particleDensitySoftness"
  | "particleSupportMask"
  | "particleSupportRadius"
  | "particleSupportNeighbors"
  | "particleSupportFlow"
  | "densityPassStrength"
  | "densitySmallScale"
  | "densityLargeScale"
  | "densityLargeThreshold"
  | "densityContrastGain"
  | "densityContrastBalance"
  | "densityEmissionPower"
  | "densityOcclusion"
  | "accumulationStrength"
  | "accumulationRadius"
  | "accumulationCurve"
  | "accumulationMemory"
  | "accumulationNoiseReject"
  | "trailOpacity"
  | "fogBrightness"
  | "trailThreshold"
  | "filament"
  | "cameraYaw"
  | "cameraPitch"
  | "cameraDistance"
  | "variationMaster"
  | "variationDrift"
  | "variationNoiseMix"
  | "variationFreq"
  | "variationOctaves"
  | "variationGain"
  | "variationLacunarity"
  | "variationSizeAmount"
  | "variationSizeCurve"
  | "variationSizeMin"
  | "variationSizeMax"
  | "variationBrightAmount"
  | "variationBrightCurve"
  | "variationBrightMin"
  | "variationBrightMax"
  | "variationOpacityAmount"
  | "variationOpacityCurve"
  | "variationOpacityMin"
  | "variationOpacityMax"
  | "variationColorAmount"
  | "variationColorCurve"
  | "variationColorMin"
  | "variationColorMax",
  SliderRange
>;

const liveSliderRanges = {
  simulationSpeed: { min: 0, max: 2, step: 0.01 },
  hueSensitivity: { min: -1, max: 1, step: 0.01 },
  depositMass: { min: 0.05, max: 2, step: 0.05 },
  trailPersistence: { min: 0.85, max: 0.999, step: 0.001 },
  trailDiffusion: { min: 0, max: 1, step: 0.01 },
  sensorGain: { min: 0, max: 10, step: 0.05 },
  sensorAngle: { min: -1, max: 1, step: 0.01 },
  sensorDistance: { min: 0, max: 3, step: 0.01 },
  mutationScale: { min: 0, max: 0.5, step: 0.005 },
  drag: { min: -0.5, max: 1, step: 0.01 },
  axialForce: { min: -1, max: 1, step: 0.01 },
  lateralForce: { min: -1, max: 1, step: 0.01 },
  globalForceMult: { min: 0, max: 2, step: 0.01 },
  strafePower: { min: 0, max: 0.5, step: 0.005 },
  strafeMomentum: { min: 0, max: 1, step: 0.01 },
  hazardRate: { min: 0, max: 0.05, step: 0.0005 },
  orientationMix: { min: 0, max: 1, step: 0.01 },
  pulseDepth: { min: 0, max: 1, step: 0.01 },
  restlessness: { min: 0, max: 1, step: 0.01 },
  pulseRate: { min: 0, max: 4, step: 0.01 }
} satisfies Record<
  | "simulationSpeed"
  | "hueSensitivity"
  | "depositMass"
  | "trailPersistence"
  | "trailDiffusion"
  | "sensorGain"
  | "sensorAngle"
  | "sensorDistance"
  | "mutationScale"
  | "drag"
  | "axialForce"
  | "lateralForce"
  | "globalForceMult"
  | "strafePower"
  | "strafeMomentum"
  | "hazardRate"
  | "orientationMix"
  | "pulseDepth"
  | "restlessness"
  | "pulseRate",
  SliderRange
>;

const emergentSliderRanges = {
  mips: { min: 0, max: 1, step: 0.01 },
  anisoFollow: { min: 0, max: 1, step: 0.01 },
  flockAlign: { min: 0, max: 1, step: 0.01 },
  flockSeparate: { min: 0, max: 1, step: 0.01 },
  chemotaxis: { min: -1, max: 1, step: 0.01 },
  quorumStrength: { min: 0, max: 1, step: 0.01 },
  quorumThreshold: { min: 0, max: 1, step: 0.01 },
  leniaStrength: { min: 0, max: 1, step: 0.01 },
  leniaCenter: { min: 0, max: 1, step: 0.01 },
  leniaWidth: { min: 0.001, max: 0.2, step: 0.001 },
  speciesForce: { min: 0, max: 1, step: 0.01 },
  predator: { min: 0, max: 1, step: 0.01 },
  alarm: { min: 0, max: 1, step: 0.01 },
  grayScott: { min: 0, max: 1, step: 0.01 },
  gsFeed: { min: 0, max: 0.1, step: 0.001 },
  gsKill: { min: 0, max: 0.1, step: 0.001 },
  energy: { min: 0, max: 1, step: 0.01 },
  energyDrain: { min: 0, max: 1, step: 0.01 }
} satisfies Record<string, SliderRange>;

const trailRadiusSliderRange = { min: minTrailRadius, max: maxTrailRadiusCap, step: 0.0005 } satisfies SliderRange;
const rayResolutionSliderRange = { min: minRayResolution, max: maxRayResolution, step: rayResolutionStep } satisfies SliderRange;
const settingsFileTypes = [{
  description: "3D Life settings",
  accept: { "application/json": [".json"] }
}] satisfies FilePickerFileType[];

const defaultControls: RenderControls = {
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
  density: 1.15,
  exposure: 1,
  fov: defaultFovDegrees,
  aperture: 0.38,
  focusDistance: 0.54,
  dofBlur: 1,
  dofEnabled: false,
  dofDebug: false,
  sceneBrightness: 1,
  raySteps: 64,
  rayResolution: maxRayResolution,
  fogTemporal: true,
  fogRenderScale: 0.5,
  fogStepScale: 0.4,
  fogTemporalBlend: 0.84,
  fogBlueNoise: false,
  fieldTextureSampling: false,
  emptySpaceSkipping: false,
  emptySpaceThreshold: 0.035,
  emptySpaceStride: 4,
  particleSizePx: 4.5,
  particleMinPx: 0,
  particleMaxPx: 4.8,
  particleOpacity: 0.42,
  particleBrightness: 1,
  particleColorMode: "solid",
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
  particleDensityRadius: 0.035,
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
  trailOpacity: 1.18,
  fogBrightness: 1,
  trailThreshold: 0,
  trailColorMode: "stable",
  fogTint: "#ffffff",
  particleTint: "#ffffff",
  renderLayer: "particles",
  palette: "aurora",
  filament: 0.72,
  cameraYaw: cameraDefaults.yaw,
  cameraPitch: cameraDefaults.pitch,
  cameraDistance: cameraDefaults.distance,
  cameraPanX: 0,
  cameraPanY: 0,
  // Master defaults to full so the per-feature Amount sliders (Size/Bright/Opacity/Color)
  // work on their own; with every Amount at 0 the variation pass is still a no-op (and the
  // shader early-outs), so the default look is unchanged.
  variationMaster: 1,
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

export function App() {
  const initialLiveState = useMemo(createInitialLiveState, []);
  const profileGpu = useMemo(() => shouldEnableGpuProfiling(new URLSearchParams(window.location.search)), []);
  const initialPlaying = useMemo(() => shouldStartPlaying(window.location.search), []);
  const parallelPipelines = useMemo(() => parallelPipelinesFromLaunch(window.location.search), []);
  const audioReactiveUrl = useMemo(() => audioReactiveUrlFromLaunch(window.location.search, {
    VITE_AUDIO_REACTIVE_DEFAULT: import.meta.env.VITE_AUDIO_REACTIVE_DEFAULT
  }), []);
  // ?audio=mic: in-browser microphone analyzer (website embed). Mutually exclusive with the
  // WebSocket backend above (audioReactiveUrlFromLaunch returns null in mic mode).
  const audioMicMode = useMemo(() => audioMicFromLaunch(window.location.search), []);
  const appShellRef = useRef<HTMLElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const liveRendererRef = useRef(new RealtimeGpuSim3d());
  // Apply the init-time parallel-compile switch before the renderer's first createPipelines().
  // Idempotent; runs synchronously during render, ahead of any ensureInitialized()/rAF.
  liveRendererRef.current.parallelPipelineCompile = parallelPipelines;
  const renderBusyRef = useRef(false);
  const pendingLiveResetRef = useRef(false);
  const controlsRef = useRef<RenderControls>(initialLiveState.controls);
  const liveConfigRef = useRef<LiveGpu3dConfig>(initialLiveState.config);
  const playingRef = useRef(initialPlaying);
  const overlayRef = useRef(false);
  const frameRef = useRef(0);
  const lastUiPublishRef = useRef(0);
  const fpsRef = useRef({ lastTime: 0, frames: 0, value: 0 });
  const jsonInputRef = useRef<HTMLInputElement | null>(null);
  const liveDiagnosticsRef = useRef<LiveGpu3dDiagnostics | null>(null);
  const audioReactiveStatusRef = useRef<AudioReactiveSocketStatus>({
    connected: false,
    frameCount: 0,
    lastSequence: null,
    url: audioReactiveUrl ?? ""
  });
  const audioSocketControllerRef = useRef<AudioReactiveSocketController | null>(null);
  const audioConfirmedInputRef = useRef<string | null>(null);
  const audioPendingInputRef = useRef<string | null>(null);
  const audioModulationRuntimeRef = useRef<AudioModulationRuntime>(createAudioModulationRuntime());
  const audioPanelRef = useRef<AudioPanelState>(defaultAudioPanelState);
  const pendingAudioUiRef = useRef<{ meters: Record<AudioBucket, number>; frame: AudioAnalysisFrame } | null>(null);
  const audioUiRafRef = useRef<number | null>(null);
  const lastAudioMeterMsRef = useRef(0);
  const lastModUiMsRef = useRef(0);
  const sliderModulationsRef = useRef<SliderModulationSettings>({});
  const sliderRegistryRef = useRef<Record<string, SliderRegistryEntry>>({});
  const midiAccessRef = useRef<MidiAccessLike | null>(null);
  const midiActiveInputIdRef = useRef<string>(midiAllInputsId);
  const midiLearningKeyRef = useRef<string | null>(null);
  const midiMappingIndexRef = useRef<Record<string, string[]>>({});
  const midiMessageCountRef = useRef(0);
  const pendingMidiUiRef = useRef<string | null>(null);
  const midiUiRafRef = useRef<number | null>(null);
  const lastAudioStatusPublishRef = useRef(0);
  const cameraRef = useRef<OrbitCamera>({
    yaw: initialLiveState.controls.cameraYaw,
    pitch: initialLiveState.controls.cameraPitch,
    distance: initialLiveState.controls.cameraDistance,
    panX: initialLiveState.controls.cameraPanX,
    panY: initialLiveState.controls.cameraPanY,
    fov: initialLiveState.controls.fov,
    targetYaw: initialLiveState.controls.cameraYaw,
    targetPitch: initialLiveState.controls.cameraPitch,
    targetDistance: initialLiveState.controls.cameraDistance,
    targetPanX: initialLiveState.controls.cameraPanX,
    targetPanY: initialLiveState.controls.cameraPanY,
    targetFov: initialLiveState.controls.fov
  });
  const dragRef = useRef<{ active: boolean; mode: ViewportDragMode | null; pointerId: number | null; x: number; y: number; startX: number; startY: number }>({
    active: false,
    mode: null,
    pointerId: null,
    x: 0,
    y: 0,
    startX: 0,
    startY: 0
  });
  const trackerRef = useRef<SwimmerTracker>(new SwimmerTracker({ ...DEFAULT_TRACKER_PARAMS }));
  const trackingRef = useRef<TrackingControls>(trackingDefaults);
  // Orbit mode: auto-advance yaw each frame so the camera circles the current tumble point.
  const cameraOrbitRef = useRef(defaultCameraOrbit.enabled);
  const cameraOrbitSpeedRef = useRef(defaultCameraOrbit.speed);
  const followSpringRef = useRef<{ panX: Spring; panY: Spring; distance: Spring }>({
    panX: { value: 0, velocity: 0 },
    panY: { value: 0, velocity: 0 },
    distance: { value: cameraDefaults.distance, velocity: 0 }
  });
  const lastFollowMsRef = useRef(0);
  // Timeline transport: deterministic playhead (1 timeline frame === 1 sim timestep) +
  // per-frame camera recording for exact replay. See timeline.ts / cameraRecorder.ts.
  const { state: timeline, dispatch: timelineDispatch } = useTimelineTransport();
  const timelineRef = useRef(timeline);
  const playheadRef = useRef(0);
  const playbackAccumRef = useRef(0);
  const wasPlayingRef = useRef(false);
  const cameraRecorderRef = useRef(createCameraRecorder());
  // Performance recorder for offline HDR export (see automationRecorder.ts + tools/hdr-export).
  const automationRecorderRef = useRef(createAutomationRecorder());
  const [recordingPerformance, setRecordingPerformance] = useState(false);
  // Real-time audio capture alongside the performance data: the visuals are replayed offline for HDR,
  // but the audio is recorded live and muxed onto the render afterward (see performanceAudioRecorder.ts).
  const performanceAudioRecorderRef = useRef(createPerformanceAudioRecorder());
  // The live mic/loopback controller, kept so the audio recorder can grab the exact reactive stream.
  const micControllerRef = useRef<MicAudioController | null>(null);
  // A fallback capture stream acquired on demand when not in mic mode (held so a later MIDI/hotkey
  // start needs no fresh permission gesture).
  const ownCaptureStreamRef = useRef<MediaStream | null>(null);
  const [audioCaptured, setAudioCaptured] = useState(false);
  // MIDI mapping for the single record toggle: map one pad/button so one press starts and the next
  // stops. Edge-triggered (fires on the rising edge of a press), kept in refs so the MIDI handler —
  // defined before togglePerformanceRecording — reads current values without re-subscribing.
  const [recordMidiMapping, setRecordMidiMapping] = useState<SliderMidiMapping | null>(null);
  const [recordMidiLearning, setRecordMidiLearning] = useState(false);
  const recordMidiMappingRef = useRef<SliderMidiMapping | null>(null);
  const recordMidiLearningRef = useRef(false);
  const recordMidiPrevHighRef = useRef(false);
  const togglePerformanceRecordingRef = useRef<() => void>(() => {});
  // Constrain the live canvas to a 9:16 box so portrait takes are composed (camera framing) and
  // recorded in the aspect they'll be posted in. Render width follows the canvas CSS box, so this
  // alone makes the live render, camera, and recording header all portrait — see tools/hdr-export.
  const [portraitMode, setPortraitMode] = useState(false);
  // While a replay-export driver is driving the renderer via __fluoddityReplayFrame, the live RAF
  // loop must stand down so it doesn't fight the deterministic replay. replayPrevTimestepRef tracks
  // the sim timestep between replay frames so we step by the exact recorded delta.
  const replayActiveRef = useRef(false);
  const replayPrevTimestepRef = useRef(0);
  const timelineSeekWaitersRef = useRef<Array<{ frame: number; resolve: () => void }>>([]);
  // Timeline transport is opt-in. Off by default = simple free-run play/pause driven by the
  // Sim Speed slider (no deterministic seek/replay on unpause). Deterministic seek still runs
  // whenever an external capture seek is pending, so export stays frame-exact regardless.
  const timelineEnabledRef = useRef(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [trackingControls, setTrackingControls] = useState<TrackingControls>(trackingDefaults);
  const [trackState, setTrackState] = useState<{ active: boolean; members: number; miss?: string }>({ active: false, members: 0 });
  const [cameraOrbit, setCameraOrbit] = useState(defaultCameraOrbit.enabled);
  const [cameraOrbitSpeed, setCameraOrbitSpeed] = useState(defaultCameraOrbit.speed);
  // When locked, mouse drag-orbit / pan / wheel-zoom are ignored so the viewport can't be nudged
  // mid-set. Off by default (mouse moves the camera as usual). Read via a ref in the input handlers.
  const [viewLocked, setViewLocked] = useState(false);
  const viewLockedRef = useRef(false);
  // Frame-rate cap (fps). The render loop free-runs at the display refresh; when the render time
  // doesn't divide evenly into the refresh, frames land at uneven intervals (e.g. alternating
  // 13/17ms) and motion -- especially camera orbit -- judders. Capping to an even target paces
  // frames uniformly. 0 = uncapped. Default 60 (matches a 60Hz projector and smooths high-refresh
  // laptop panels). Read via a ref in the loop so changing it doesn't re-create the loop.
  const [frameCap, setFrameCap] = useState(60);
  const frameCapRef = useRef(60);
  const lastRenderMsRef = useRef(0);
  // Idle "demo mode": arming toggle (off by default). When armed and the user has been idle for
  // DEMO_IDLE_MS, the rAF loop gently orbits and drifts MIDI-mapped sliders. Audio reactivity is
  // NOT treated as interaction; any real pointer/key/wheel/UI input or new MIDI CC exits demo.
  const [demoMode, setDemoMode] = useState(() => demoLaunchFromSearch(window.location.search).idle);
  const demoModeRef = useRef(false);
  // Instant demo: a separate toggle (off by default) that runs the SAME demo motion as the idle
  // mode above, but engages immediately (no DEMO_IDLE_MS wait) and never exits on interaction --
  // so an embedded sim auto-runs and a viewer can still grab sliders. Sliders the viewer touches
  // are recorded in touchedSlidersRef and dropped from the drift so manual control always wins.
  // Boots on when launched with ?demo=instant (used by the website embed), which also arms the
  // idle demo so the sim re-enters demo if a viewer toggles Instant Demo off and walks away.
  const [instantDemo, setInstantDemo] = useState(() => demoLaunchFromSearch(window.location.search).instant);
  const instantDemoRef = useRef(false);
  const touchedSlidersRef = useRef<Set<string>>(new Set());
  // True once the viewer manually toggles the Orbit checkbox (or nudges Orbit Speed) during a demo
  // session. Like touchedSlidersRef, it makes manual control win: the demo's auto-orbit yields, so
  // unchecking Orbit actually stops the camera even while the demo keeps drifting other sliders.
  const orbitTouchedRef = useRef(false);
  const lastInteractionRef = useRef(0);
  const demoStateRef = useRef<{
    active: boolean;
    priorOrbit: { enabled: boolean; speed: number } | null;
    picks: Array<{ key: string; from: number; to: number; lo: number; hi: number; startMs: number; durMs: number }>;
    lastPickMs: number;
    lastDriftMs: number;
  }>({ active: false, priorOrbit: null, picks: [], lastPickMs: 0, lastDriftMs: 0 });
  const displayMode: DisplayMode = "live";
  const [controlsState, setControls] = useState<RenderControls>(initialLiveState.controls);
  const [selectedPresetId, setSelectedPresetId] = useState(initialLiveState.presetId);
  const [liveConfig, setLiveConfig] = useState<LiveGpu3dConfig>(initialLiveState.config);
  const [playing, setPlaying] = useState(initialPlaying);
  const [timelineEnabled, setTimelineEnabled] = useState(false);
  const [overlay, setOverlay] = useState(false);
  const [dragMode, setDragMode] = useState<ViewportDragMode | null>(null);
  // Embedded mode boots clean (menu + FPS hidden) so site visitors see only the sim;
  // the M key toggles the cockpit back.
  const [panelOpen, setPanelOpen] = useState(() => !embedFromLaunch(window.location.search));
  // Auto-hide the mouse cursor after a short idle while fullscreen (revealed on any move), so the
  // projected image is clean. Gated to :fullscreen in CSS, so it never hides in windowed mode.
  const [cursorIdle, setCursorIdle] = useState(false);
  // Mirror panelOpen into a ref so the rAF loop can skip publishing diagnostics state (which
  // re-renders the whole ~126-control panel ~5x/sec and causes frame-time stutter) when the
  // panel is hidden. Hiding the menu (M) then yields zero per-frame React re-renders.
  const panelOpenRef = useRef(true);
  useEffect(() => { panelOpenRef.current = panelOpen; }, [panelOpen]);
  // Latest audio band levels (low/mid/high), updated every audio frame regardless of panel
  // visibility, so the audio-reactive color mode keeps driving the visuals during a performance.
  const audioLevelsRef = useRef({ low: 0, mid: 0, high: 0 });
  const [frame, setFrame] = useState(0);
  const [fps, setFps] = useState(0);
  const [savedSettings, setSavedSettings] = useState<SavedSettingsPreset[]>(loadSavedSettings);
  const [selectedSettingsId, setSelectedSettingsId] = useState("");
  const [liveDiagnostics, setLiveDiagnostics] = useState<LiveGpu3dDiagnostics | null>(null);
  const [audioStatus, setAudioStatus] = useState<AudioReactiveSocketStatus>(audioReactiveStatusRef.current);
  const [audioPanel, setAudioPanel] = useState<AudioPanelState>(defaultAudioPanelState);
  const [audioMeters, setAudioMeters] = useState<Record<AudioBucket, number>>({ low: 0, mid: 0, high: 0 });
  const [audioLastFrame, setAudioLastFrame] = useState<AudioAnalysisFrame | null>(null);
  const [audioInputDevices, setAudioInputDevices] = useState<AudioInputDeviceInfo[]>([]);
  const [audioActiveInput, setAudioActiveInput] = useState<string | null>(null);
  const [audioPendingInput, setAudioPendingInput] = useState<string | null>(null);
  const [audioBackendError, setAudioBackendError] = useState<string | null>(null);
  const [audioReconnectNonce, setAudioReconnectNonce] = useState(0);
  const [sliderModulations, setSliderModulations] = useState<SliderModulationSettings>({});
  const [expandedSliderModulationKey, setExpandedSliderModulationKey] = useState<string | null>(null);
  const [midiStatus, setMidiStatus] = useState<MidiPanelStatus>(() => ({
    supported: typeof navigator.requestMIDIAccess === "function",
    enabled: false,
    error: typeof navigator.requestMIDIAccess === "function" ? null : "Web MIDI unavailable",
    messageCount: 0,
    lastMessage: null
  }));
  const [midiInputs, setMidiInputs] = useState<MidiInputInfo[]>([]);
  const [midiActiveInputId, setMidiActiveInputId] = useState<string>(midiAllInputsId);
  const [midiLearningKey, setMidiLearningKey] = useState<string | null>(null);
  const [compute3d, setCompute3d] = useState<GpuSim3dSummary | null>(null);
  const [webgpu, setWebgpu] = useState<WebGpuDiagnostics>({
    checked: false,
    available: false,
    adapterOk: false,
    deviceOk: false,
    bufferRoundtripOk: false,
    adapterInfo: "pending"
  });

  useEffect(() => {
    probeWebGpu().then(setWebgpu);
    const skipAppCompute = new URLSearchParams(window.location.search).get("skipAppCompute") === "1";
    if (!skipAppCompute) {
      runWebGpuComputeConformance()
        .then((result) => setCompute3d(summarizeCompute3d(result)))
        .catch((error: unknown) => {
          setCompute3d({
            available: false,
            passed: false,
            mode: "failed",
            width: defaultGpuSim3dConfig.width,
            height: defaultGpuSim3dConfig.height,
            depth: defaultGpuSim3dConfig.depth,
            particleCount: defaultGpuSim3dConfig.particleCount,
            steps: defaultGpuSim3dConfig.steps,
            cpuFieldSum: 0,
            gpuFieldSum: 0,
            gpuNonzeroVoxels: 0,
            fieldMeanAbsError: Number.POSITIVE_INFINITY,
            fieldMaxAbsError: Number.POSITIVE_INFINITY,
            particlePositionMeanError: Number.POSITIVE_INFINITY,
            particlePositionMaxError: Number.POSITIVE_INFINITY,
            tolerances: {
              fieldMeanAbsError: 0,
              fieldMaxAbsError: 0,
              particlePositionMeanError: 0,
              particlePositionMaxError: 0
            },
            error: error instanceof Error ? error.message : String(error)
          });
        });
    }
  }, []);

  const currentSavedUi = useMemo(() => buildSavedUiState({
    playing,
    overlay,
    viewLocked,
    demoMode,
    frameCap,
    cameraOrbitEnabled: cameraOrbit,
    cameraOrbitSpeed,
    timelineEnabled,
    timeline,
    trackingControls
  }), [cameraOrbit, cameraOrbitSpeed, demoMode, frameCap, overlay, playing, timeline, timelineEnabled, trackingControls, viewLocked]);
  const currentSavedAudio = useMemo(() => buildSavedAudioState({
    panel: audioPanel,
    sliders: sliderModulations
  }), [audioPanel, sliderModulations]);
  const controls = useMemo(
    () => normalizeRenderControlsForDisplayMode(
      displayMode,
      renderControlsWithModulationRangeOverrides(sanitizeRenderControls(controlsState), controlsState, sliderModulations)
    ),
    [controlsState, displayMode, sliderModulations]
  );

  useEffect(() => {
    controlsRef.current = controls;
    // Orbit owns yaw while active, so don't snap it back to the slider value on other edits.
    if (!cameraOrbitRef.current) cameraRef.current.targetYaw = controls.cameraYaw;
    cameraRef.current.targetPitch = controls.cameraPitch;
    cameraRef.current.targetDistance = controls.cameraDistance;
    cameraRef.current.targetPanX = controls.cameraPanX;
    cameraRef.current.targetPanY = controls.cameraPanY;
    cameraRef.current.targetFov = controls.fov;
  }, [controls]);

  useEffect(() => {
    demoModeRef.current = demoMode;
  }, [demoMode]);

  useEffect(() => {
    instantDemoRef.current = instantDemo;
  }, [instantDemo]);

  // Track real user interaction (NOT audio reactivity, which mutates sliders programmatically and
  // never dispatches DOM input events). Resets the idle timer so demo mode yields to live control.
  useEffect(() => {
    const mark = (event?: Event) => {
      lastInteractionRef.current = performance.now();
      // Instant demo stays active through interaction; record which slider the viewer grabbed so
      // its auto-drift yields to them. A slider <input>'s registry key is `data-testid ?? name`
      // (see SliderImpl), reconstructed here from the event target. Demo's own writes are direct
      // onChange() calls (no DOM event), so this only fires for real user input.
      if (!event || (event.type !== "input" && event.type !== "change")) return;
      const target = event.target;
      if (!(target instanceof HTMLInputElement) || target.type !== "range") return;
      const key = target.dataset.testid ?? target.name;
      if (!key || !sliderRegistryRef.current[key]) return;
      touchedSlidersRef.current.add(key);
      const demo = demoStateRef.current;
      if (demo.picks.length) demo.picks = demo.picks.filter((pick) => pick.key !== key);
    };
    mark();
    const opts: AddEventListenerOptions = { capture: true, passive: true };
    const events: Array<keyof WindowEventMap> = ["pointerdown", "keydown", "wheel", "input", "change"];
    for (const ev of events) window.addEventListener(ev, mark, opts);
    return () => {
      for (const ev of events) window.removeEventListener(ev, mark, opts);
    };
  }, []);

  useEffect(() => {
    trackingRef.current = trackingControls;
    trackerRef.current.setCohesion(trackingControls.cohesion);
  }, [trackingControls]);

  useEffect(() => {
    cameraOrbitRef.current = cameraOrbit;
    cameraOrbitSpeedRef.current = cameraOrbitSpeed;
  }, [cameraOrbit, cameraOrbitSpeed]);

  useEffect(() => {
    viewLockedRef.current = viewLocked;
  }, [viewLocked]);

  useEffect(() => {
    frameCapRef.current = frameCap;
  }, [frameCap]);

  useEffect(() => {
    liveConfigRef.current = liveConfig;
  }, [liveConfig]);

  useEffect(() => {
    audioPanelRef.current = audioPanel;
  }, [audioPanel]);

  useEffect(() => {
    sliderModulationsRef.current = sliderModulations;
    const index: Record<string, string[]> = {};
    for (const [key, config] of Object.entries(sliderModulations)) {
      if (!hasEnabledMidiMapping(config.midi) || !config.midi?.control) continue;
      const indexKey = midiControlIndexKey(config.midi.control);
      index[indexKey] = [...(index[indexKey] ?? []), key];
    }
    midiMappingIndexRef.current = index;
  }, [sliderModulations]);

  useEffect(() => {
    midiActiveInputIdRef.current = midiActiveInputId;
  }, [midiActiveInputId]);

  useEffect(() => {
    midiLearningKeyRef.current = midiLearningKey;
  }, [midiLearningKey]);

  useEffect(() => {
    if (!audioPendingInput) return;
    const timer = window.setTimeout(() => {
      if (audioPendingInputRef.current === audioPendingInput) {
        audioPendingInputRef.current = null;
        setAudioPendingInput(null);
      }
    }, 1200);
    return () => window.clearTimeout(timer);
  }, [audioPendingInput]);

  const registerSliderForAudio = useCallback((key: string, entry: SliderRegistryEntry | null) => {
    if (entry) {
      sliderRegistryRef.current[key] = entry;
    } else {
      delete sliderRegistryRef.current[key];
    }
  }, []);

  const updateSliderModulationConfig = useCallback((key: string, config: SliderModulationConfig) => {
    setSliderModulations((current) => {
      const next = { ...current, [key]: config };
      sliderModulationsRef.current = next;
      return next;
    });
    delete audioModulationRuntimeRef.current.bucketValues[key];
  }, []);

  const applyMidiMessageToSlider = useCallback((key: string, config: SliderModulationConfig, message: MidiControlMessage) => {
    const entry = sliderRegistryRef.current[key];
    const midi = config.midi;
    if (!entry || !midi) return;
    const nextValue = clamp(
      mapMidiValue(midi.min, midi.max, message.value, entry.value),
      Math.min(entry.min, midi.min, midi.max),
      Math.max(entry.max, midi.min, midi.max)
    );
    if (nextValue !== entry.value) {
      entry.onChange(nextValue);
    }
  }, []);

  const handleMidiControl = useCallback((message: MidiControlMessage) => {
    const activeInputId = midiActiveInputIdRef.current;
    if (activeInputId !== midiAllInputsId && message.inputId !== activeInputId) return;
    // A real MIDI control message counts as live interaction and exits demo mode (audio does not).
    lastInteractionRef.current = performance.now();
    midiMessageCountRef.current += 1;
    pendingMidiUiRef.current = midiControlLabel(message);
    if (midiUiRafRef.current === null) {
      midiUiRafRef.current = requestAnimationFrame(() => {
        midiUiRafRef.current = null;
        const lastMessage = pendingMidiUiRef.current;
        pendingMidiUiRef.current = null;
        setMidiStatus((current) => ({
          ...current,
          enabled: true,
          error: null,
          messageCount: midiMessageCountRef.current,
          lastMessage: lastMessage ?? current.lastMessage
        }));
      });
    }

    const learningKey = midiLearningKeyRef.current;
    if (learningKey) {
      const entry = sliderRegistryRef.current[learningKey];
      if (entry) {
        const currentConfig = getSliderModulation(sliderModulationsRef.current, learningKey, entry.min, entry.max);
        const nextConfig: SliderModulationConfig = {
          ...currentConfig,
          midi: midiMappingFromControl(message, currentConfig.midi?.min ?? entry.min, currentConfig.midi?.max ?? entry.max)
        };
        midiLearningKeyRef.current = null;
        setMidiLearningKey(null);
        updateSliderModulationConfig(learningKey, nextConfig);
        applyMidiMessageToSlider(learningKey, nextConfig, message);
      }
    }

    const targets = new Set<string>();
    for (const indexKey of midiMessageIndexKeys(message)) {
      for (const target of midiMappingIndexRef.current[indexKey] ?? []) {
        targets.add(target);
      }
    }
    for (const key of targets) {
      const config = sliderModulationsRef.current[key];
      if (!config || !midiMappingMatches(config.midi, message)) continue;
      applyMidiMessageToSlider(key, config, message);
    }

    // Record-toggle pad. In learn mode the next message becomes the binding; otherwise a rising edge
    // (press past the half-way point, having been below it) toggles recording once — so press = start,
    // next press = stop, and the matching note-off/release doesn't double-fire.
    if (recordMidiLearningRef.current) {
      const mapping = midiMappingFromControl(message, 0, 1);
      recordMidiMappingRef.current = mapping;
      recordMidiLearningRef.current = false;
      recordMidiPrevHighRef.current = message.value > 0.5;
      setRecordMidiMapping(mapping);
      setRecordMidiLearning(false);
    } else if (recordMidiMappingRef.current && midiMappingMatches(recordMidiMappingRef.current, message)) {
      const high = message.value > 0.5;
      if (high && !recordMidiPrevHighRef.current) togglePerformanceRecordingRef.current();
      recordMidiPrevHighRef.current = high;
    }
  }, [applyMidiMessageToSlider, updateSliderModulationConfig]);

  const handleMidiData = useCallback((inputId: string, inputName: string, data: ArrayLike<number> | null | undefined) => {
    const message = parseMidiControlMessage(inputId, inputName, data);
    if (message) handleMidiControl(message);
  }, [handleMidiControl]);

  const refreshMidiInputs = useCallback((access: MidiAccessLike) => {
    const inputs = midiInputList(access);
    setMidiInputs(inputs);
    setMidiActiveInputId((current) => {
      if (current === midiAllInputsId || inputs.some((input) => input.id === current)) return current;
      midiActiveInputIdRef.current = midiAllInputsId;
      return midiAllInputsId;
    });
    for (const input of access.inputs.values()) {
      input.onmidimessage = (event) => {
        handleMidiData(input.id, input.name ?? input.id, event.data);
      };
    }
  }, [handleMidiData]);

  const ensureMidiAccess = useCallback(async () => {
    if (midiAccessRef.current) {
      refreshMidiInputs(midiAccessRef.current);
      setMidiStatus((current) => ({ ...current, supported: true, enabled: true, error: null }));
      return midiAccessRef.current;
    }
    if (typeof navigator.requestMIDIAccess !== "function") {
      setMidiStatus((current) => ({
        ...current,
        supported: false,
        enabled: false,
        error: "Web MIDI unavailable"
      }));
      return null;
    }
    try {
      const access = await navigator.requestMIDIAccess({ sysex: false }) as unknown as MidiAccessLike;
      midiAccessRef.current = access;
      access.onstatechange = () => refreshMidiInputs(access);
      refreshMidiInputs(access);
      setMidiStatus((current) => ({ ...current, supported: true, enabled: true, error: null }));
      return access;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setMidiStatus((current) => ({
        ...current,
        supported: true,
        enabled: false,
        error: message || "MIDI access denied"
      }));
      return null;
    }
  }, [refreshMidiInputs]);

  useEffect(() => {
    if (!Object.values(sliderModulations).some((config) => hasEnabledMidiMapping(config.midi))) return;
    void ensureMidiAccess();
  }, [ensureMidiAccess, sliderModulations]);

  const startMidiLearn = useCallback((key: string) => {
    setExpandedSliderModulationKey(key);
    midiLearningKeyRef.current = key;
    setMidiLearningKey(key);
    void ensureMidiAccess();
  }, [ensureMidiAccess]);

  const cancelMidiLearn = useCallback(() => {
    midiLearningKeyRef.current = null;
    setMidiLearningKey(null);
  }, []);

  useEffect(() => {
    window.__fluoddityInjectMidiMessage = (data: ArrayLike<number>, inputId = "test-midi", inputName = "Test MIDI") => {
      handleMidiData(inputId, inputName, data);
    };
    return () => {
      delete window.__fluoddityInjectMidiMessage;
      if (midiUiRafRef.current !== null) {
        cancelAnimationFrame(midiUiRafRef.current);
        midiUiRafRef.current = null;
      }
      const access = midiAccessRef.current;
      if (access) {
        access.onstatechange = null;
        for (const input of access.inputs.values()) {
          input.onmidimessage = null;
        }
      }
    };
  }, [handleMidiData]);

  const sliderModulationContext = useMemo<SliderModulationContextValue>(() => ({
    expandedKey: expandedSliderModulationKey,
    settings: sliderModulations,
    midiStatus,
    midiLearningKey,
    register: registerSliderForAudio,
    setExpandedKey: setExpandedSliderModulationKey,
    updateConfig: updateSliderModulationConfig,
    startMidiLearn,
    cancelMidiLearn
  }), [
    cancelMidiLearn,
    expandedSliderModulationKey,
    midiLearningKey,
    midiStatus,
    registerSliderForAudio,
    sliderModulations,
    startMidiLearn,
    updateSliderModulationConfig
  ]);

  const handleAudioFrame = useCallback((frame: AudioAnalysisFrame, dtSec: number) => {
    const runtime = audioModulationRuntimeRef.current;
    const meters = updateAudioMeters(runtime, frame, audioPanelRef.current, dtSec);
    audioLevelsRef.current = meters;
    pendingAudioUiRef.current = { meters, frame };
    if (audioUiRafRef.current === null) {
      audioUiRafRef.current = requestAnimationFrame(() => {
        audioUiRafRef.current = null;
        const pending = pendingAudioUiRef.current;
        pendingAudioUiRef.current = null;
        if (!pending) return;
        // The audio meters live in the control panel; updating them re-renders the whole panel.
        // Skip entirely while the panel is hidden (the modulation below still drives the visuals
        // every frame via audioLevelsRef). While the panel IS open, throttle the meter readout to
        // ~20 Hz so live audio doesn't re-render App every audio frame (~200/s) and stutter the panel.
        if (!panelOpenRef.current) return;
        const now = performance.now();
        if (now - lastAudioMeterMsRef.current < 50) return;
        lastAudioMeterMsRef.current = now;
        setAudioMeters(pending.meters);
        setAudioLastFrame(pending.frame);
      });
    }

    const settings = sliderModulationsRef.current;
    if (Object.keys(settings).length === 0) return;
    const registry = sliderRegistryRef.current;
    // The modulation envelope must advance every audio frame (it integrates attack/decay over
    // dtSec), so always compute it. Base each slider on its live value.
    const baseValues: Record<string, number> = {};
    for (const key of Object.keys(settings)) {
      const entry = registry[key];
      if (entry) baseValues[key] = entry.value;
    }
    const patches = applyAudioModulation(runtime, frame, audioPanelRef.current, settings, baseValues, dtSec);
    // Apply through each slider's onChange so the value reaches the correct control field (the
    // registry key is the slider's testId, NOT a RenderControls property name, and some sliders
    // are backed by liveConfig rather than controls). onChange -> setControls re-renders App, so
    // throttle the apply: firing per audio frame (up to ~90/s) was the intermittent hitching.
    const now = performance.now();
    if (now - lastModUiMsRef.current < 33) return;
    lastModUiMsRef.current = now;
    for (const [key, value] of Object.entries(patches)) {
      const entry = registry[key];
      if (!entry) continue;
      const nextValue = clamp(value, entry.min, entry.max);
      if (nextValue !== entry.value) entry.onChange(nextValue);
    }
  }, []);

  useEffect(() => () => {
    if (audioUiRafRef.current !== null) {
      cancelAnimationFrame(audioUiRafRef.current);
      audioUiRafRef.current = null;
    }
  }, []);

  useEffect(() => {
    window.__fluoddityInjectAudioFrame = (frame: AudioAnalysisFrame, dtSec = 1 / 60) => {
      handleAudioFrame(frame, dtSec);
    };
    return () => { delete window.__fluoddityInjectAudioFrame; };
  }, [handleAudioFrame]);

  // Mic mode: feed handleAudioFrame from the in-browser analyzer. The handler goes through a
  // ref so a re-render never tears down the mic stream (and re-prompts the AudioContext).
  const handleAudioFrameRef = useRef(handleAudioFrame);
  useEffect(() => { handleAudioFrameRef.current = handleAudioFrame; }, [handleAudioFrame]);
  useEffect(() => {
    if (!audioMicMode) return;
    let frameCount = 0;
    let status: MicAudioStatus = "starting";
    let lastFrame: AudioAnalysisFrame | null = null;
    window.__fluoddityMicAudio = () => ({ status, frameCount, lastFrame });
    const controller = startMicAudio({
      onFrame: (frame, dtSec) => {
        frameCount += 1;
        lastFrame = frame;
        handleAudioFrameRef.current(frame, dtSec);
      },
      onStatus: (next) => { status = next; }
    });
    micControllerRef.current = controller;
    return () => {
      controller.stop();
      if (micControllerRef.current === controller) micControllerRef.current = null;
      delete window.__fluoddityMicAudio;
    };
  }, [audioMicMode]);

  useEffect(() => {
    if (!audioReactiveUrl || audioStatus.connected) return;
    const timer = window.setInterval(() => setAudioReconnectNonce((value) => value + 1), 3000);
    return () => window.clearInterval(timer);
  }, [audioReactiveUrl, audioStatus.connected]);

  useEffect(() => {
    if (!audioReactiveUrl) {
      audioSocketControllerRef.current = null;
      window.__fluoddityAudioReactive = () => ({ ...audioReactiveStatusRef.current, enabled: false });
      return () => {
        delete window.__fluoddityAudioReactive;
      };
    }
    const controller = connectAudioReactiveSocket({
      url: audioReactiveUrl,
      mapping: manualSliderAudioMapping,
      getState: () => ({
        render: controlsRef.current as unknown as Record<string, unknown>,
        live: liveConfigRef.current as unknown as Record<string, unknown>
      }),
      apply: () => {},
      onStatus: (status) => {
        const previous = audioReactiveStatusRef.current;
        audioReactiveStatusRef.current = status;
        const now = performance.now();
        if (previous.connected !== status.connected || now - lastAudioStatusPublishRef.current >= 250) {
          lastAudioStatusPublishRef.current = now;
          setAudioStatus(status);
        }
      },
      onFrame: handleAudioFrame,
      onDevices: (activeInput, devices) => {
        setAudioActiveInput(activeInput);
        setAudioInputDevices(devices);
        setAudioBackendError(null);
        audioConfirmedInputRef.current = activeInput;
        const pending = audioPendingInputRef.current;
        if (pending && activeInput !== pending) {
          setAudioActiveInput(pending);
          return;
        }
        audioPendingInputRef.current = null;
        setAudioPendingInput(null);
      },
      onBackendError: (message) => {
        audioPendingInputRef.current = null;
        setAudioPendingInput(null);
        setAudioActiveInput(audioConfirmedInputRef.current);
        setAudioBackendError(message);
      }
    });
    audioSocketControllerRef.current = controller;
    window.__fluoddityAudioReactive = () => ({ ...controller.status(), enabled: true });
    return () => {
      audioSocketControllerRef.current = null;
      controller.close();
      delete window.__fluoddityAudioReactive;
    };
  }, [audioReactiveUrl, audioReconnectNonce, handleAudioFrame]);

  useEffect(() => {
    playingRef.current = playing;
  }, [playing]);

  useEffect(() => {
    window.__fluodditySetPlaying = (nextPlaying: boolean) => {
      playingRef.current = nextPlaying;
      setPlaying(nextPlaying);
    };
    return () => { delete window.__fluodditySetPlaying; };
  }, []);

  useEffect(() => {
    overlayRef.current = overlay;
  }, [overlay]);

  useEffect(() => {
    frameRef.current = frame;
  }, [frame]);

  useEffect(() => {
    timelineRef.current = timeline;
  }, [timeline]);

  // Single play source of truth is `playing`; mirror it into the timeline for the bar UI.
  useEffect(() => {
    timelineDispatch({ type: "play", playing });
  }, [playing]);

  useEffect(() => {
    timelineEnabledRef.current = timelineEnabled;
  }, [timelineEnabled]);

  useEffect(() => {
    let cancelled = false;
    let raf = 0;
    const loop = async () => {
      // Frame-rate cap: if the target interval hasn't elapsed since the last rendered frame, skip
      // this refresh (re-arm and return) so frames are paced to an even cadence instead of beating
      // against the display refresh. The -1ms tolerance absorbs rAF jitter so a 60-cap on a 60Hz
      // panel still renders every refresh. 0 = uncapped.
      const cap = frameCapRef.current;
      if (cap > 0) {
        const nowCap = performance.now();
        if (nowCap - lastRenderMsRef.current < 1000 / cap - 1) {
          raf = requestAnimationFrame(loop);
          return;
        }
        lastRenderMsRef.current = nowCap;
      }
      // An offline replay-export driver is owning the renderer right now; the live loop stands
      // down so its free-running step+render can't race the deterministic replay. Keep the RAF
      // alive so the loop resumes the instant __fluoddityReplayEnd clears the flag.
      if (replayActiveRef.current) {
        raf = requestAnimationFrame(loop);
        return;
      }
      if (canvasRef.current && !renderBusyRef.current) {
        renderBusyRef.current = true;
        try {
          const nowMs = performance.now();
          const followDt = lastFollowMsRef.current === 0 ? 1 / 60 : Math.min(0.05, (nowMs - lastFollowMsRef.current) / 1000);
          lastFollowMsRef.current = nowMs;
          const aspect = Math.max(0.0001, (canvasRef.current.clientWidth || 1) / (canvasRef.current.clientHeight || 1));
          if (lastInteractionRef.current === 0) {
            lastInteractionRef.current = nowMs;
          }
          {
            const demo = demoStateRef.current;
            // Instant demo engages immediately and ignores the idle timer entirely, so interaction
            // never ends it; idle demo still waits for DEMO_IDLE_MS of no real input.
            const shouldDemo = instantDemoRef.current || (demoModeRef.current && nowMs - lastInteractionRef.current > 60000);
            if (shouldDemo && !demo.active) {
              demo.active = true;
              demo.priorOrbit = { enabled: cameraOrbitRef.current, speed: cameraOrbitSpeedRef.current };
              demo.picks = [];
              demo.lastPickMs = 0;
              touchedSlidersRef.current.clear();
              orbitTouchedRef.current = false;
            } else if (!shouldDemo && demo.active) {
              demo.active = false;
              // Only undo the demo's auto-orbit if the viewer never took it over. If they did
              // (orbitTouchedRef), their Orbit setting already stands and must survive demo exit.
              if (demo.priorOrbit && !orbitTouchedRef.current) {
                cameraOrbitRef.current = demo.priorOrbit.enabled;
                cameraOrbitSpeedRef.current = demo.priorOrbit.speed;
                setCameraOrbit(demo.priorOrbit.enabled);
                setCameraOrbitSpeed(demo.priorOrbit.speed);
              }
              demo.priorOrbit = null;
              demo.picks = [];
              touchedSlidersRef.current.clear();
              orbitTouchedRef.current = false;
            }
            if (demo.active) {
              // Gentle auto-orbit -- UNLESS the viewer has grabbed the Orbit control this session,
              // in which case the demo yields and the Orbit checkbox / Orbit Speed slider win (so
              // unchecking Orbit stops the camera). Keep the speed inside the orbit-speed slider's
              // MIDI mapping range (the user's configured bounds) so demo never spins faster than
              // the mapping allows; fall back to the slider's own min/max when there's no MIDI
              // mapping. Set every frame so it holds against the auto-orbit advance below.
              if (!orbitTouchedRef.current) {
                const orbitCfg = sliderModulationsRef.current["orbit-speed-slider"];
                const orbitEntry = sliderRegistryRef.current["orbit-speed-slider"];
                const orbitMidi = orbitCfg && hasEnabledMidiMapping(orbitCfg.midi) ? orbitCfg.midi : null;
                const orbitLo = orbitMidi ? orbitMidi.min : (orbitEntry ? orbitEntry.min : orbitSpeedSliderRange.min);
                const orbitHi = orbitMidi ? orbitMidi.max : (orbitEntry ? orbitEntry.max : orbitSpeedSliderRange.max);
                const nextOrbitSpeed = clamp(0.75, Math.min(orbitLo, orbitHi), Math.max(orbitLo, orbitHi));
                if (!cameraOrbitRef.current) {
                  cameraOrbitRef.current = true;
                  setCameraOrbit(true);
                }
                if (Math.abs(cameraOrbitSpeedRef.current - nextOrbitSpeed) > 0.0001) {
                  cameraOrbitSpeedRef.current = nextOrbitSpeed;
                  setCameraOrbitSpeed(nextOrbitSpeed);
                }
              }
              // Every ~8s, retarget EVERY MIDI-mapped slider to a new value within its mapping range.
              // The MIDI mapping is the selection mechanism: if the user mapped it, demo drives it --
              // including discrete/mode sliders like P Color (the eased value is quantized to the
              // slider's step below). orbit-speed is driven by the dedicated block above; the denylist
              // backstops structural controls (seed/volume/count) that reallocate buffers or reseed.
              if (demo.picks.length === 0 || nowMs - demo.lastPickMs > 24000) {
                demo.lastPickMs = nowMs;
                // Sliders the viewer has grabbed (touchedSlidersRef) are excluded so manual control
                // wins; see selectDemoDriftCandidates for the full selection rules.
                const candidates = selectDemoDriftCandidates(
                  sliderModulationsRef.current,
                  sliderRegistryRef.current,
                  touchedSlidersRef.current,
                  DEMO_DRIFT_DENYLIST
                );
                demo.picks = candidates.map((key) => {
                  const entry = sliderRegistryRef.current[key];
                  // Stay inside the MIDI mapping's configured range (which may be a sub-range of the
                  // slider's full min/max), so demo never drives past what the mapping allows.
                  const midi = sliderModulationsRef.current[key].midi;
                  const rangeLo = midi ? midi.min : entry.min;
                  const rangeHi = midi ? midi.max : entry.max;
                  const lo = Math.min(rangeLo, rangeHi);
                  const hi = Math.max(rangeLo, rangeHi);
                  const from = Math.min(hi, Math.max(lo, entry.value));
                  // Slow, cinematic drift (was 6-11s). Color-ish controls ease ~3x slower so palette
                  // and mode changes are gentle -- with discrete sliders the longer ease lingers on
                  // each step. The ~24s re-pick cadence is long enough that most eases finish first.
                  const isColor = /color|tint|hue|palette|gradient/.test(key.toLowerCase());
                  const durMs = (14000 + Math.random() * 9000) * (isColor ? 3 : 1);
                  return { key, from, to: lo + (hi - lo) * Math.random(), lo, hi, startMs: nowMs, durMs };
                });
              }
              // Apply at ~30 Hz: each onChange -> setControls re-renders App (React batches all of a
              // frame's changes into one render), so per-frame apply with many mapped sliders would
              // be the same churn we throttled for audio. The eases are 6-11s, so 30 Hz looks smooth.
              if (nowMs - demo.lastDriftMs >= 33) {
                demo.lastDriftMs = nowMs;
                for (const pick of demo.picks) {
                  const entry = sliderRegistryRef.current[pick.key];
                  if (!entry) continue;
                  const t = Math.min(1, (nowMs - pick.startMs) / pick.durMs);
                  const eased = t * t * (3 - 2 * t);
                  const raw = pick.from + (pick.to - pick.from) * eased;
                  // Quantize to the slider's step so discrete/mode sliders (step >= 1, e.g. P Color)
                  // land on valid options and we don't spam onChange with sub-step jitter.
                  const step = entry.step > 0 ? entry.step : 0;
                  // Quantize, then clamp back into the mapping range (rounding can nudge a value with
                  // an off-grid bound, e.g. max 0.595 at step 0.01, just past hi).
                  const value = Math.min(pick.hi, Math.max(pick.lo, step > 0 ? Math.round(raw / step) * step : raw));
                  const eps = step > 0 ? step / 2 : 1e-6;
                  if (Math.abs(value - entry.value) > eps) entry.onChange(value);
                }
              }
            }
          }
          if (cameraOrbitRef.current && !dragRef.current.active) {
            // Auto-orbit: advance yaw at the chosen speed (rad/s) around the current tumble point.
            const orbitedYaw = normalizeAngle(cameraRef.current.yaw + cameraOrbitSpeedRef.current * followDt);
            cameraRef.current.yaw = orbitedYaw;
            cameraRef.current.targetYaw = orbitedYaw;
          }
          applyTrackingToCamera(cameraRef.current, trackerRef.current, trackingRef.current, followSpringRef.current, aspect, followDt);
          const currentControls = smoothedRenderControls(controlsRef.current, cameraRef.current);
          const isPlaying = playingRef.current;
          const isOverlay = overlayRef.current;
          {
            if (pendingLiveResetRef.current) {
              await liveRendererRef.current.reset();
              pendingLiveResetRef.current = false;
            }
            const currentConfig = liveConfigRef.current;
            const tl = timelineRef.current;
            const sim = liveRendererRef.current;
            sim.setAudioLevels(audioLevelsRef.current.low, audioLevelsRef.current.mid, audioLevelsRef.current.high);
            // Live playback uses the proven coupled step+render path (stepSimulation=true) so
            // particles and temporal fog render correctly. Paused/scrubbing uses a
            // deterministic seek (forward = step, backward = reset + replay from zero) then a
            // state render. External capture drives the paused/seek path (see
            // __fluoddityTimelineSeek), so frame-exact determinism is unaffected by playback.
            let targetFrame: number;
            let diagnostics: LiveGpu3dDiagnostics;
            // Free-run unless the timeline transport is enabled, or a deterministic export seek
            // is pending (so capture stays frame-exact even with the timeline UI hidden).
            const useTimeline = timelineEnabledRef.current || timelineSeekWaitersRef.current.length > 0;
            if (!useTimeline) {
              // Plain play/pause: play steps at the Sim Speed slider (currentConfig.simulationSpeed);
              // pause just re-renders the current state. No seek/replay -> no unpause jank.
              diagnostics = await sim.render(canvasRef.current, currentControls, isOverlay, isPlaying, currentConfig, profileGpu);
              targetFrame = diagnostics.timestep;
            } else if (isPlaying) {
              if (tl.loopOut > tl.loopIn && sim.currentTimestep >= tl.loopOut) {
                // Loop region wrap: jump back to loopIn deterministically, render that state.
                await sim.reset();
                if (tl.loopIn > 0) {
                  await sim.advanceSteps(canvasRef.current, currentConfig, tl.loopIn);
                }
                diagnostics = await sim.render(canvasRef.current, currentControls, isOverlay, false, currentConfig, profileGpu);
              } else {
                // Step forward at the chosen playback speed (1 frame === 1 sim step at 1x).
                const playConfig = { ...currentConfig, simulationSpeed: tl.playbackSpeed };
                diagnostics = await sim.render(canvasRef.current, currentControls, isOverlay, true, playConfig, profileGpu);
              }
              targetFrame = diagnostics.timestep;
            } else {
              targetFrame = tl.currentFrame;
              const seek = planSeek(sim.currentTimestep, targetFrame);
              if (seek.needsReset) {
                await sim.reset();
                await sim.advanceSteps(canvasRef.current, currentConfig, seek.steps);
              } else if (seek.steps > 0) {
                await sim.advanceSteps(canvasRef.current, currentConfig, seek.steps);
              }
              diagnostics = await sim.render(canvasRef.current, currentControls, isOverlay, false, currentConfig, profileGpu);
            }
            liveDiagnosticsRef.current = diagnostics;
            frameRef.current = targetFrame;
            // Record the exact camera pose played at this frame for deterministic replay.
            const playedPose: CameraPose = {
              yaw: currentControls.cameraYaw,
              pitch: currentControls.cameraPitch,
              distance: currentControls.cameraDistance,
              panX: currentControls.cameraPanX,
              panY: currentControls.cameraPanY,
              fov: currentControls.fov,
              focusDistance: currentControls.focusDistance,
              aperture: currentControls.aperture,
              dofBlur: currentControls.dofBlur,
              dofEnabled: currentControls.dofEnabled
            };
            cameraRecorderRef.current.record(targetFrame, playedPose);
            // Capture the performance for offline HDR re-render: the smoothed controls (camera
            // baked in), the live config, the sim timestep, and the audio levels feeding the
            // render uniforms. Cheap (numbers only) so it never perturbs the live framerate.
            if (automationRecorderRef.current.isRecording()) {
              automationRecorderRef.current.record(
                {
                  timestep: targetFrame,
                  controls: currentControls,
                  config: currentConfig,
                  audio: { ...audioLevelsRef.current }
                },
                nowMs
              );
            }
            // Resolve evidence-orchestrator seek waiters now parked on this frame.
            if (timelineSeekWaitersRef.current.length > 0) {
              const stillWaiting: Array<{ frame: number; resolve: () => void }> = [];
              for (const waiter of timelineSeekWaitersRef.current) {
                if (waiter.frame === targetFrame) waiter.resolve();
                else stillWaiting.push(waiter);
              }
              timelineSeekWaitersRef.current = stillWaiting;
            }
            if (!cancelled && panelOpenRef.current) {
              publishUiDiagnostics(() => {
                setLiveDiagnostics(diagnostics);
                setFrame(targetFrame);
                if (isPlaying) timelineDispatch({ type: "scrub", frame: targetFrame });
              }, lastUiPublishRef);
            }
          }
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          console.error(error);
          if (!cancelled) setLoadError(message);
          liveRendererRef.current.recover();
        } finally {
          renderBusyRef.current = false;
        }
      }
      publishFps(fpsRef.current, (value) => {
        if (!cancelled && panelOpenRef.current) setFps(value);
      });
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
    };
  }, []);

  useEffect(() => {
    let busy = false;
    let cancelled = false;
    let timer = 0;
    const tick = async () => {
      if (cancelled) return;
      if (!busy && trackerRef.current.state.active) {
        busy = true;
        try {
          const particles = await liveRendererRef.current.captureParticles(TRACKING_PARTICLE_READBACK_LIMIT);
          if (particles) {
            const count = particles.length / 12;
            trackerRef.current.setEps(epsForParticleCount(count));
            trackerRef.current.recluster(particles, count, performance.now());
            const s = trackerRef.current.state;
            setTrackState({ active: s.active, members: s.members });
          }
        } catch (error) {
          console.error(error);
        } finally {
          busy = false;
        }
      }
      if (cancelled) return;
      // captureParticles() is a GPU->CPU readback (mapAsync), which forces a pipeline sync.
      // Keep tracking on a bounded sample and back the cadence off with frame cost so Follow
      // stays smooth at high particle counts; the tracker dead-reckons between samples.
      const frameMs = liveDiagnosticsRef.current?.frameTimeMs ?? 16;
      const delay = Math.max(RECLUSTER_INTERVAL_MS, Math.min(250, frameMs * 6));
      timer = window.setTimeout(tick, delay);
    };
    timer = window.setTimeout(tick, RECLUSTER_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    window.__fluoddityTrackState = () => {
      const s = trackerRef.current.state;
      const cam = cameraRef.current;
      return { active: s.active, members: s.members, centroid: s.centroid, panX: cam.panX, panY: cam.panY, distance: cam.distance };
    };
    return () => {
      delete window.__fluoddityTrackState;
    };
  }, []);

  useEffect(() => {
    window.__fluoddityDiagnostics = () => ({
      webgpu,
      mode: "live",
      renderer: liveDiagnosticsRef.current,
      live: liveDiagnosticsRef.current,
      controls,
      audio: {
        status: audioStatus,
        activeInput: audioActiveInput,
        pendingInput: audioPendingInput,
        inputs: audioInputDevices,
        error: audioBackendError,
        panel: audioPanel,
        meters: audioMeters,
        lastSequence: audioLastFrame?.sequence ?? null,
        sliders: sliderModulations
      },
      midi: {
        status: midiStatus,
        inputs: midiInputs,
        activeInputId: midiActiveInputId,
        learningKey: midiLearningKey
      },
      fps,
      vertexCount: liveVertexCount(liveDiagnosticsRef.current, liveConfig),
      savedSettings: savedSettings.map((settings) => ({ id: settings.id, name: settings.name, updatedAt: settings.updatedAt })),
      selectedSettingsId,
      preset: selectedPresetId,
      liveConfig,
      frames: frameRef.current,
      playing,
      overlay,
      profileGpu,
      conformance: {
        deterministicPreset: true,
        visualNonBlank: (liveDiagnosticsRef.current?.timestep ?? 0) > 0,
        webgpuProbeComplete: webgpu.checked,
        webgpuRenderer: liveDiagnosticsRef.current?.renderer === "webgpu-live-fluoddity-3d",
        webgpuCompute3d: compute3d?.passed === true,
        webgpuLive3d: liveWebGpuConformance(liveDiagnosticsRef.current)
      }
    });
    return () => {
      delete window.__fluoddityDiagnostics;
    };
  }, [audioActiveInput, audioBackendError, audioInputDevices, audioLastFrame, audioMeters, audioPanel, audioPendingInput, audioStatus, compute3d, controls, fps, frame, liveConfig, liveDiagnostics, midiActiveInputId, midiInputs, midiLearningKey, midiStatus, overlay, profileGpu, savedSettings, selectedPresetId, selectedSettingsId, sliderModulations, webgpu]);

  useEffect(() => {
    const toBase64 = (array: Float32Array): string => {
      const bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
      let binary = "";
      const chunk = 0x8000;
      for (let i = 0; i < bytes.length; i += chunk) {
        binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
      }
      return btoa(binary);
    };
    window.__fluoddityCaptureState = async () => {
      for (let attempt = 0; renderBusyRef.current && attempt < 120; attempt += 1) {
        await new Promise<void>((resolve) => window.setTimeout(resolve, 16));
      }
      if (renderBusyRef.current) {
        throw new Error("State capture could not acquire the renderer");
      }
      renderBusyRef.current = true;
      try {
        const state = await liveRendererRef.current.captureState();
        if (!state) return null;
        const rendered = smoothedRenderControls(controlsRef.current, cameraRef.current);
        const canvas = canvasRef.current;
        return {
          manifest: {
            version: 1,
            frame: frameRef.current,
            camera: {
              yaw: rendered.cameraYaw,
              pitch: rendered.cameraPitch,
              distance: rendered.cameraDistance,
              panX: rendered.cameraPanX,
              panY: rendered.cameraPanY,
              fov: rendered.fov,
              focusDistance: rendered.focusDistance,
              aperture: rendered.aperture,
              dofBlur: rendered.dofBlur,
              dofEnabled: rendered.dofEnabled
            },
            resolution: {
              width: canvas?.width ?? 1024,
              height: canvas?.height ?? 1024
            },
            grid: { width: state.width, height: state.height, depth: state.depth },
            voxelCount: state.voxelCount,
            particleCount: state.particleCount,
            timeline: {
              fps: TIMELINE_FPS,
              totalFrames: TIMELINE_TOTAL_FRAMES,
              currentFrame: timelineRef.current.currentFrame,
              loopIn: timelineRef.current.loopIn,
              loopOut: timelineRef.current.loopOut
            },
            liveConfig: liveConfigRef.current,
            renderControls: sanitizeRenderControls(rendered)
          },
          fieldB64: toBase64(state.field),
          particlesB64: toBase64(state.particles)
        };
      } finally {
        renderBusyRef.current = false;
      }
    };
    return () => {
      delete window.__fluoddityCaptureState;
    };
  }, []);

  useEffect(() => {
    window.__fluoddityParticleOscillationProbe = async (options = {}) => {
      const canvas = canvasRef.current;
      if (!canvas) return null;

      for (let attempt = 0; renderBusyRef.current && attempt < 120; attempt += 1) {
        await new Promise<void>((resolve) => window.setTimeout(resolve, 16));
      }
      if (renderBusyRef.current) {
        throw new Error("Particle oscillation probe could not acquire the renderer");
      }

      const wasPlaying = playingRef.current;
      playingRef.current = false;
      setPlaying(false);
      renderBusyRef.current = true;

      const sim = liveRendererRef.current;
      const currentConfig = liveConfigRef.current;
      const requestedParticleCount = options.particleCount ?? options.config?.particleCount ?? Math.min(currentConfig.particleCount, 8192);
      const probeConfig = sanitizeLiveConfig({
        ...currentConfig,
        ...options.config,
        particleCount: clampInteger(requestedParticleCount, minLiveParticles, 65536),
        simulationSpeed: 1,
        hazardRate: options.config?.hazardRate ?? 0,
        recycleEnabled: options.config?.recycleEnabled ?? false
      });
      const steps = clampInteger(options.steps ?? 80, 4, 240);
      const maxParticles = clampInteger(options.maxParticles ?? probeConfig.particleCount, 1, probeConfig.particleCount);
      const topK = clampInteger(options.topK ?? 12, 1, 32);
      const frames: Float32Array[] = [];

      try {
        if (options.reset ?? true) {
          await sim.reset();
        }
        await sim.render(canvas, controlsRef.current, overlayRef.current, false, probeConfig, profileGpu);
        const initial = await sim.captureParticles(maxParticles);
        if (!initial) throw new Error("Particle oscillation probe could not capture the initial state");
        frames.push(initial);
        for (let step = 0; step < steps; step += 1) {
          await sim.advanceSteps(canvas, probeConfig, 1);
          const particles = await sim.captureParticles(maxParticles);
          if (!particles) throw new Error(`Particle oscillation probe capture failed at step ${step + 1}`);
          frames.push(particles);
        }
        const capStride = frames.length ? Math.max(4, Math.round(frames[0].length / maxParticles)) : liveParticleFloatCount;
        const report = analyzeParticleOscillation(frames, {
          particleCount: maxParticles,
          stride: capStride,
          topK,
          minStepDistance: 0.00002,
          sampleCount: 12,
          snapDistance: options.snapDistance
        });
        return {
          ...report,
          capStride,
          config: {
            particleCount: probeConfig.particleCount,
            simulationSpeed: probeConfig.simulationSpeed,
            dt: probeConfig.dt,
            sensorAngle: probeConfig.sensorAngle,
            sensorDistance: probeConfig.sensorDistance,
            drag: probeConfig.drag,
            strafePower: probeConfig.strafePower,
            strafeMomentum: probeConfig.strafeMomentum,
            axialForce: probeConfig.axialForce,
            lateralForce: probeConfig.lateralForce,
            globalForceMult: probeConfig.globalForceMult,
            hazardRate: probeConfig.hazardRate,
            cohorts: probeConfig.cohorts,
            colorByCohort: probeConfig.colorByCohort,
            boundaryMode: probeConfig.boundaryMode
          }
        };
      } finally {
        try {
          await sim.reset();
          const diagnostics = await sim.render(canvas, controlsRef.current, overlayRef.current, false, currentConfig, profileGpu);
          liveDiagnosticsRef.current = diagnostics;
          setLiveDiagnostics(diagnostics);
        } finally {
          renderBusyRef.current = false;
          playingRef.current = wasPlaying;
          setPlaying(wasPlaying);
        }
      }
    };
    return () => {
      delete window.__fluoddityParticleOscillationProbe;
    };
  }, [profileGpu]);

  useEffect(() => {
    window.__fluoddityHashParticles = async (options = {}) => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      for (let attempt = 0; renderBusyRef.current && attempt < 120; attempt += 1) {
        await new Promise<void>((resolve) => window.setTimeout(resolve, 16));
      }
      if (renderBusyRef.current) {
        throw new Error("Hash-particles hook could not acquire the renderer");
      }
      const wasPlaying = playingRef.current;
      playingRef.current = false;
      setPlaying(false);
      renderBusyRef.current = true;

      const sim = liveRendererRef.current;
      const currentConfig = liveConfigRef.current;
      const requestedParticleCount = options.particleCount ?? options.config?.particleCount ?? Math.min(currentConfig.particleCount, 8192);
      const probeConfig = sanitizeLiveConfig({
        ...currentConfig,
        ...options.config,
        particleCount: clampInteger(requestedParticleCount, minLiveParticles, 65536),
        simulationSpeed: 1,
        hazardRate: options.config?.hazardRate ?? 0,
        recycleEnabled: options.config?.recycleEnabled ?? false
      });
      const steps = clampInteger(options.steps ?? 64, 1, 480);

      try {
        if (options.reset ?? true) {
          await sim.reset();
        }
        await sim.render(canvas, controlsRef.current, overlayRef.current, false, probeConfig, profileGpu);
        await sim.advanceSteps(canvas, probeConfig, steps);
        const state = await sim.captureState();
        if (!state) throw new Error("Hash-particles hook could not capture sim state");
        const fnv = (bytes: Uint8Array, start: number, end: number): number => {
          let h = 0x811c9dc5;
          for (let i = start; i < end; i += 1) {
            h ^= bytes[i];
            h = Math.imul(h, 0x01000193);
          }
          return h >>> 0;
        };
        const hashField = (arr: Float32Array): string => {
          const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
          return fnv(bytes, 0, bytes.length).toString(16).padStart(8, "0");
        };
        // Order-independent particle hash: the render depth-sort permutes the particle buffer between
        // runs, so a plain byte hash differs even when the physics is bit-identical. Summing
        // per-particle hashes is commutative -> reordering is invisible, but any real change to a
        // particle's position/velocity/hue still moves the total.
        const hashParticles = (arr: Float32Array): string => {
          const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
          const stride = liveParticleFloatCount * Float32Array.BYTES_PER_ELEMENT;
          let acc = 0;
          for (let p = 0; p + stride <= bytes.length; p += stride) {
            acc = (acc + fnv(bytes, p, p + stride)) >>> 0;
          }
          return (acc >>> 0).toString(16).padStart(8, "0");
        };
        let maxDensity = 0;
        for (let i = 3; i < state.field.length; i += 4) {
          if (state.field[i] > maxDensity) maxDensity = state.field[i];
        }
        // Particle hash catches motion/hue behaviors; field hash catches density-field behaviors
        // (Lenia, Gray-Scott) that particles don't sense for force.
        return { hash: hashParticles(state.particles), fieldHash: hashField(state.field), particleCount: state.particleCount, steps, maxDensity };
      } finally {
        try {
          await sim.reset();
          const diagnostics = await sim.render(canvas, controlsRef.current, overlayRef.current, false, currentConfig, profileGpu);
          liveDiagnosticsRef.current = diagnostics;
          setLiveDiagnostics(diagnostics);
        } finally {
          renderBusyRef.current = false;
          playingRef.current = wasPlaying;
          setPlaying(wasPlaying);
        }
      }
    };
    return () => {
      delete window.__fluoddityHashParticles;
    };
  }, [profileGpu]);

  useEffect(() => {
    window.__fluoddityBenchSteps = async (options = {}) => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      for (let attempt = 0; renderBusyRef.current && attempt < 120; attempt += 1) {
        await new Promise<void>((resolve) => window.setTimeout(resolve, 16));
      }
      if (renderBusyRef.current) {
        throw new Error("Bench hook could not acquire the renderer");
      }
      const wasPlaying = playingRef.current;
      playingRef.current = false;
      setPlaying(false);
      renderBusyRef.current = true;

      const sim = liveRendererRef.current;
      const currentConfig = liveConfigRef.current;
      const requestedParticleCount = options.particleCount ?? options.config?.particleCount ?? Math.min(currentConfig.particleCount, 8192);
      const benchConfig = sanitizeLiveConfig({
        ...currentConfig,
        ...options.config,
        particleCount: clampInteger(requestedParticleCount, minLiveParticles, 65536),
        simulationSpeed: 1,
        hazardRate: options.config?.hazardRate ?? 0,
        recycleEnabled: options.config?.recycleEnabled ?? false
      });
      const steps = clampInteger(options.steps ?? 200, 1, 2000);
      const reps = clampInteger(options.reps ?? 5, 1, 30);
      const warmup = clampInteger(options.warmup ?? 30, 0, 500);

      try {
        await sim.reset();
        await sim.render(canvas, controlsRef.current, overlayRef.current, false, benchConfig, profileGpu);
        if (warmup > 0) await sim.advanceSteps(canvas, benchConfig, warmup);
        const samples: number[] = [];
        for (let rep = 0; rep < reps; rep += 1) {
          const t0 = performance.now();
          await sim.advanceSteps(canvas, benchConfig, steps); // awaits onSubmittedWorkDone -> wall time ~= GPU time
          samples.push(performance.now() - t0);
        }
        const sorted = [...samples].sort((a, b) => a - b);
        const medianMs = sorted[Math.floor(sorted.length / 2)];
        return { medianMs, samples, steps, particleCount: benchConfig.particleCount };
      } finally {
        try {
          await sim.reset();
          const diagnostics = await sim.render(canvas, controlsRef.current, overlayRef.current, false, currentConfig, profileGpu);
          liveDiagnosticsRef.current = diagnostics;
          setLiveDiagnostics(diagnostics);
        } finally {
          renderBusyRef.current = false;
          playingRef.current = wasPlaying;
          setPlaying(wasPlaying);
        }
      }
    };
    return () => {
      delete window.__fluoddityBenchSteps;
    };
  }, [profileGpu]);

  // Evidence-orchestrator hooks: deterministic per-frame seek + instant camera set.
  useEffect(() => {
    window.__fluoddityTimelineSeek = (frame: number) => {
      const target = Math.max(0, Math.min(TIMELINE_TOTAL_FRAMES - 1, Math.round(frame)));
      playingRef.current = false;
      timelineRef.current = { ...timelineRef.current, playing: false, currentFrame: target };
      setPlaying(false);
      timelineDispatch({ type: "play", playing: false });
      timelineDispatch({ type: "scrub", frame: target });
      return new Promise<number>((resolve) => {
        timelineSeekWaitersRef.current.push({ frame: target, resolve: () => resolve(target) });
      });
    };
    window.__fluodditySetCameraOrbit = (pose) => {
      const cam = cameraRef.current;
      cam.yaw = pose.yaw;
      cam.targetYaw = pose.yaw;
      cam.pitch = pose.pitch;
      cam.targetPitch = pose.pitch;
      cam.distance = pose.distance;
      cam.targetDistance = pose.distance;
      if (pose.panX !== undefined) {
        cam.panX = pose.panX;
        cam.targetPanX = pose.panX;
      }
      if (pose.panY !== undefined) {
        cam.panY = pose.panY;
        cam.targetPanY = pose.panY;
      }
    };
    window.__fluodditySetRenderControl = (patch) => {
      setControls((c) => ({ ...c, ...patch }));
    };
    window.__fluodditySetSplatPrepass = (enabled: boolean) => {
      liveRendererRef.current.splatPrepassEnabled = enabled;
    };
    window.__fluodditySetComputeSplat = (enabled: boolean) => {
      liveRendererRef.current.computeSplatEnabled = enabled;
    };
    window.__fluodditySetFieldTextureSensing = (enabled: boolean) => {
      liveRendererRef.current.fieldTextureSensingEnabled = enabled;
    };
    window.__fluodditySetParticleSort = (enabled: boolean) => {
      liveRendererRef.current.particleSortEnabled = enabled;
    };
    window.__fluodditySetParallelPipelineCompile = (enabled: boolean) => {
      liveRendererRef.current.parallelPipelineCompile = enabled;
    };
    return () => {
      delete window.__fluoddityTimelineSeek;
      delete window.__fluodditySetCameraOrbit;
      delete window.__fluodditySetRenderControl;
      delete window.__fluodditySetSplatPrepass;
      delete window.__fluodditySetComputeSplat;
      delete window.__fluodditySetFieldTextureSensing;
      delete window.__fluodditySetParticleSort;
      delete window.__fluodditySetParallelPipelineCompile;
    };
  }, []);

  // Parity test hook: inject deterministic particles + camera into the live
  // WebGPU sim, pause stepping, and let the real WGSL splat pipeline render one
  // frame. The next take_screenshot of the canvas is the "browser" side of
  // render parity (not a 2D analytic redraw).
  useEffect(() => {
    const f32ToBase64 = (arr: Float32Array): string => {
      const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
      return u8ToBase64(bytes);
    };
    const u8ToBase64 = (bytes: Uint8Array): string => {
      let s = "";
      const chunk = 0x8000;
      for (let i = 0; i < bytes.length; i += chunk) {
        s += String.fromCharCode(...bytes.subarray(i, i + chunk));
      }
      return btoa(s);
    };
    window.__fluoddityReadCanvasHDR = async () => {
      const hdr = await liveRendererRef.current.readCanvasHDR();
      if (!hdr) return null;
      return { width: hdr.width, height: hdr.height, pixelsB64: f32ToBase64(hdr.pixels) };
    };
    // Offline HDR replay export. The driver suspends the live loop (replayActiveRef), then walks
    // the recording frame-by-frame: each ReplayFrame steps the deterministic sim by the recorded
    // timestep delta and re-renders with the recorded controls + audio, with no real-time clock.
    window.__fluoddityReplayBegin = async (firstConfig) => {
      replayActiveRef.current = true;
      playingRef.current = false;
      const sim = liveRendererRef.current;
      // Capture the post-processed swapchain into a persistent texture each render so we can read the
      // FINAL HDR (bloom + grade), not the pre-post scene and not the already-presented swapchain.
      sim.captureFinalHDR = true;
      // Seed this.config to the recording's baseline so the first advanceSteps doesn't see a
      // structural config change (which would trigger an extra reset), then reset to frame 0.
      liveConfigRef.current = firstConfig;
      await sim.reset();
      replayPrevTimestepRef.current = 0;
    };
    window.__fluoddityReplayFrame = async (spec) => {
      if (!canvasRef.current) return null;
      const sim = liveRendererRef.current;
      const stepDelta = spec.timestep - replayPrevTimestepRef.current;
      if (stepDelta > 0) {
        await sim.advanceSteps(canvasRef.current, spec.config, stepDelta);
      }
      replayPrevTimestepRef.current = spec.timestep;
      // Force the render aspect when the exporter asks for one (e.g. 9:16 portrait), so the output is
      // exact and independent of the headless replay window's shape. 0 = follow the canvas.
      sim.aspectOverride = spec.aspect && spec.aspect > 0 ? spec.aspect : 0;
      // Audio band levels feed the render uniforms (visual modulation); restore them before the draw.
      sim.setAudioLevels(spec.audio.low, spec.audio.mid, spec.audio.high);
      // stepSimulation=false: we already advanced via advanceSteps; this is a pure state render.
      // The smoothed controls carry the exact played camera, so no separate camera set is needed.
      await sim.render(canvasRef.current, spec.controls, false, false, spec.config, false);
      // Raw rgba16float bytes (with row padding) — node decodes half->float. Avoids the ~1.3s/frame
      // in-browser decode loop + halves the base64/transfer vs returning decoded float32.
      const hdr = await sim.readFinalHDRRaw();
      if (!hdr) return null;
      return { width: hdr.width, height: hdr.height, bytesPerRow: hdr.bytesPerRow, halfB64: u8ToBase64(hdr.data) };
    };
    window.__fluoddityReplayEnd = () => {
      replayActiveRef.current = false;
      replayPrevTimestepRef.current = 0;
      liveRendererRef.current.captureFinalHDR = false;
      // Clear the forced export aspect so the resumed live loop follows the canvas again.
      liveRendererRef.current.aspectOverride = 0;
    };
    return () => {
      delete window.__fluoddityReadCanvasHDR;
      delete window.__fluoddityReplayBegin;
      delete window.__fluoddityReplayFrame;
      delete window.__fluoddityReplayEnd;
    };
  }, []);

  useEffect(() => {
    const seedParityField = (
      config: LiveGpu3dConfig,
      particles: Array<{ x: number; y: number; z: number; velocity?: [number, number, number]; alpha?: number }>,
      visible: number
    ): Float32Array => {
      const width = config.width;
      const height = config.height;
      const depth = config.depth;
      const field = new Float32Array(width * height * depth * 4);
      const writeCell = (x: number, y: number, z: number, particle: { velocity?: [number, number, number]; alpha?: number }, weight: number) => {
        const cx = Math.max(0, Math.min(width - 1, x));
        const cy = Math.max(0, Math.min(height - 1, y));
        const cz = Math.max(0, Math.min(depth - 1, z));
        const base = ((cz * height + cy) * width + cx) * 4;
        const alpha = Math.max(0, particle.alpha ?? 0.045) / 0.045;
        const density = Math.max(0, weight * alpha);
        const velocity = particle.velocity ?? [0, 0, 0];
        field[base + 0] += velocity[0] * density;
        field[base + 1] += velocity[1] * density;
        field[base + 2] += velocity[2] * density;
        field[base + 3] = Math.max(field[base + 3], density);
      };
      for (let i = 0; i < visible; i += 1) {
        const particle = particles[i];
        if (!particle) continue;
        const gx = (particle.x + 1) * 0.5 * width - 0.5;
        const gy = (particle.y + 1) * 0.5 * height - 0.5;
        const gz = (particle.z + 1) * 0.5 * depth - 0.5;
        if (!Number.isFinite(gx + gy + gz)) continue;
        const bx = Math.round(gx);
        const by = Math.round(gy);
        const bz = Math.round(gz);
        for (let dz = -1; dz <= 1; dz += 1) {
          for (let dy = -1; dy <= 1; dy += 1) {
            for (let dx = -1; dx <= 1; dx += 1) {
              const distance = Math.hypot(dx, dy, dz);
              writeCell(bx + dx, by + dy, bz + dz, particle, Math.max(0.08, 1 - distance * 0.32));
            }
          }
        }
      }
      return field;
    };

    window.__parityReady = false;
    window.__fluoddityLoadParityScene = async (spec) => {
      const sim = liveRendererRef.current;
      const count = liveConfigRef.current.particleCount;
      // Build the full particle buffer: synthetic in [0..N-1], invisible elsewhere.
      // Packed 8-float layout: pos.xyz, cohort, vel.xyz, hue (saturation/value/alpha are
      // shader constants now; invisibility relies on pushing spares off-domain).
      const stride = 8;
      const buf = new Float32Array(count * stride);
      const visible = Math.min(spec.particles.length, count);
      for (let i = 0; i < visible; i++) {
        const p = spec.particles[i];
        const base = i * stride;
        buf[base + 0] = p.x; buf[base + 1] = p.y; buf[base + 2] = p.z;
        buf[base + 3] = p.cohort ?? 0;       // pos_cohort.w
        const v = p.velocity ?? [0, 0, 0];
        buf[base + 4] = v[0]; buf[base + 5] = v[1]; buf[base + 6] = v[2];
        buf[base + 7] = p.hue ?? (p.rgb ? p.rgb[0] : 0);  // vel_id.w = hue
      }
      for (let i = visible; i < count; i++) {
        const base = i * stride;
        // Push way off-domain so the splat lands outside the view and is invisible.
        buf[base + 0] = 1e6; buf[base + 1] = 1e6; buf[base + 2] = 1e6;
      }
      sim.clearField();
      if (spec.seedField ?? true) {
        sim.setFieldBufferData(seedParityField(liveConfigRef.current, spec.particles, visible));
      }
      sim.setParticleBufferData(buf);

      // Override camera (current + target) so smoothing doesn't drift.
      const c = spec.camera;
      cameraRef.current.yaw = c.yaw; cameraRef.current.targetYaw = c.yaw;
      cameraRef.current.pitch = c.pitch; cameraRef.current.targetPitch = c.pitch;
      cameraRef.current.distance = c.distance; cameraRef.current.targetDistance = c.distance;
      cameraRef.current.panX = c.panX; cameraRef.current.targetPanX = c.panX;
      cameraRef.current.panY = c.panY; cameraRef.current.targetPanY = c.panY;
      cameraRef.current.fov = c.fov; cameraRef.current.targetFov = c.fov;

      // Force deterministic render config: particles only, solid white, additive
      // by default. Spec.renderControls overrides any of these (color mode,
      // tint, size, velocity stretch, etc.) so the parity test can exercise
      // every particle-rendering knob.
      const overrides = spec.renderControls ?? {};
      const nextControls: RenderControls = {
        ...controlsRef.current,
        ...overrides,
        cameraYaw: c.yaw, cameraPitch: c.pitch, cameraDistance: c.distance,
        cameraPanX: c.panX, cameraPanY: c.panY,
        focusDistance: c.focusDistance, aperture: c.aperture,
        dofBlur: c.dofBlur, dofEnabled: c.dofEnabled ?? overrides.dofEnabled ?? controlsRef.current.dofEnabled,
        renderLayer: overrides.renderLayer ?? "particles",
        particleColorMode: overrides.particleColorMode ?? "solid",
        particleBlendMode: overrides.particleBlendMode ?? "additive",
        particleBrightness: overrides.particleBrightness ?? 1,
        particleOpacity: overrides.particleOpacity ?? 0.42,
        particleTint: overrides.particleTint ?? "#ffffff",
        particleSizePx: overrides.particleSizePx ?? controlsRef.current.particleSizePx,
        particleMinPx: overrides.particleMinPx ?? controlsRef.current.particleMinPx,
        particleMaxPx: overrides.particleMaxPx ?? controlsRef.current.particleMaxPx,
        particleVelocityStretch: overrides.particleVelocityStretch ?? false,
        particleStretch: overrides.particleStretch ?? controlsRef.current.particleStretch,
        particleStretchMin: overrides.particleStretchMin ?? controlsRef.current.particleStretchMin,
        particleStretchSpeed: overrides.particleStretchSpeed ?? controlsRef.current.particleStretchSpeed,
        // Force the brightness/exposure stack to known defaults so the WGSL
        // particle_gain() = exposure * (0.35 + sceneBrightness*0.65) * particleBrightness
        // is deterministic for parity tests.
        exposure: overrides.exposure ?? 1,
        sceneBrightness: overrides.sceneBrightness ?? 1
      };
      controlsRef.current = nextControls;
      setControls(nextControls);

      // Freeze sim stepping; the render loop will still draw frames.
      if (playingRef.current) {
        const btn = document.querySelector('[data-testid="play-toggle"]') as HTMLButtonElement | null;
        btn?.click();
      }

      const canvas = canvasRef.current;
      if (canvas) {
        for (let attempt = 0; renderBusyRef.current && attempt < 30; attempt += 1) {
          await new Promise<void>(r => window.setTimeout(r, 16));
        }
        renderBusyRef.current = true;
        try {
          const diagnostics = await liveRendererRef.current.render(canvas, controlsRef.current, overlayRef.current, false, liveConfigRef.current, profileGpu);
          liveDiagnosticsRef.current = diagnostics;
          setLiveDiagnostics(diagnostics);
        } finally {
          renderBusyRef.current = false;
        }
      } else {
        await new Promise<void>(r => requestAnimationFrame(() => requestAnimationFrame(() => r())));
      }

      // Read the persistent HDR scene texture immediately after the deterministic
      // render. This avoids the swap-chain texture lifetime issues that can make
      // post-present canvas readback come back blank.
      let hdr: { width: number; height: number; pixelsB64: string } | null = null;
      let hdrError = "";
      try {
        const raw = await liveRendererRef.current.readSceneHDR();
        if (raw) {
          const bytes = new Uint8Array(raw.pixels.buffer, raw.pixels.byteOffset, raw.pixels.byteLength);
          let s = "";
          const chunk = 0x8000;
          for (let i = 0; i < bytes.length; i += chunk) {
            s += String.fromCharCode(...bytes.subarray(i, i + chunk));
          }
          hdr = { width: raw.width, height: raw.height, pixelsB64: btoa(s) };
        }
      } catch (err) {
        hdrError = err instanceof Error ? err.message : String(err);
        console.warn("readSceneHDR failed:", err);
      }

      window.__parityReady = true;
      return {
        canvasWidth: canvas?.width ?? 0,
        canvasHeight: canvas?.height ?? 0,
        particleCount: visible,
        hdr,
        hdrError
      } as any;
    };
    return () => {
      delete window.__fluoddityLoadParityScene;
      delete window.__parityReady;
    };
  }, []);

  const requestLiveReset = useCallback(() => {
    pendingLiveResetRef.current = true;
    liveDiagnosticsRef.current = null;
    setLiveDiagnostics(null);
    frameRef.current = 0;
    setFrame(0);
  }, []);

  const reset = useCallback(() => {
    requestLiveReset();
  }, [requestLiveReset]);

  // Record / stop a live performance for offline HDR export. Start snapshots nothing itself — the
  // render loop appends one frame per rendered frame while recording is active (see automationRecorder
  // record call in the loop). Stop serializes the buffer and downloads it for tools/hdr-export.
  // The audio stream to record: prefer the live reactive mic/loopback stream (perfect sync, no extra
  // permission), fall back to a held own-capture stream, else acquire one. getUserMedia needs a user
  // gesture only on first acquisition; once cached, MIDI/hotkey starts reuse it gesture-free.
  const ensureCaptureStream = useCallback(async (): Promise<MediaStream | null> => {
    const live = (s: MediaStream | null | undefined) =>
      s && s.getAudioTracks().some((t) => t.readyState === "live") ? s : null;
    const mic = live(micControllerRef.current?.getStream());
    if (mic) return mic;
    const own = live(ownCaptureStreamRef.current);
    if (own) return own;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false }
      });
      ownCaptureStreamRef.current = stream;
      return stream;
    } catch {
      return null;
    }
  }, []);

  // One toggle for the whole take: starts/stops the visual performance recorder AND the live audio
  // recorder together, then downloads the recording JSON + the audio so tools/hdr-export can mux them.
  // Driven by the button, the 'R' hotkey, and a MIDI-mapped pad (see recordMidiMappingRef below).
  const togglePerformanceRecording = useCallback(async () => {
    const recorder = automationRecorderRef.current;
    const audioRecorder = performanceAudioRecorderRef.current;
    if (recorder.isRecording()) {
      const now = performance.now();
      const recording = recorder.stop();
      setRecordingPerformance(false);
      const audio = await audioRecorder.stop(now);
      const frameCount = recording?.frames.length ?? 0;
      const savedJson = recording ? await savePerformanceRecording(recording) : null;
      const savedAudio = audio ? await savePerformanceAudio(audio, frameCount) : null;
      if (savedJson || savedAudio) {
        console.info("[3d-life-sim] performance downloaded:", savedJson, savedAudio);
      }
      setAudioCaptured(false);
    } else {
      const canvas = canvasRef.current;
      const stream = await ensureCaptureStream();
      const now = performance.now();
      recorder.start({ version: 1, width: canvas?.width ?? 0, height: canvas?.height ?? 0 }, now);
      const started = stream ? audioRecorder.start(stream, now) : false;
      setAudioCaptured(started);
      setRecordingPerformance(true);
    }
  }, [ensureCaptureStream]);

  // Mirror record-toggle state into refs the (earlier-defined) MIDI handler reads, and keep a live
  // pointer to the latest toggle callback so MIDI/hotkey starts always hit the current closure.
  useEffect(() => { recordMidiMappingRef.current = recordMidiMapping; }, [recordMidiMapping]);
  useEffect(() => { recordMidiLearningRef.current = recordMidiLearning; }, [recordMidiLearning]);
  useEffect(() => {
    togglePerformanceRecordingRef.current = () => { void togglePerformanceRecording(); };
  }, [togglePerformanceRecording]);

  const startRecordMidiLearn = useCallback(() => {
    cancelMidiLearn();                 // a slider learn and the record learn can't both listen at once
    setRecordMidiLearning(true);
    void ensureMidiAccess();
  }, [cancelMidiLearn, ensureMidiAccess]);

  const updateLiveConfig = useCallback((patch: Partial<LiveGpu3dConfig>, resetSimulation = false) => {
    setLiveConfig((value) => {
      const next = normalizeTrailKernel({ ...value, ...patch });
      liveConfigRef.current = next;
      return next;
    });
    if (resetSimulation) {
      requestLiveReset();
    }
  }, [requestLiveReset]);

  const updateParticleBudget = useCallback((particleCount: number) => {
    const count = clampParticleCount(particleCount);
    const volumeSize = recommendedVolumeSize(count, liveConfigRef.current.width);
    updateLiveConfig({
      particleCount: count,
      width: volumeSize,
      height: volumeSize,
      depth: volumeSize
    }, true);
  }, [updateLiveConfig]);

  const applyPreset = useCallback((presetId: string) => {
    const preset = getLivePreset(presetId);
    const current = liveConfigRef.current;
    const usesPresetRenderControls = Boolean(preset.renderControls);
    setSelectedPresetId(preset.id);
    setSelectedSettingsId("");
    if (preset.renderControls) {
      setControls((currentControls) => {
        const nextControls = normalizeRenderControlsForDisplayMode("live", {
          ...currentControls,
          ...preset.renderControls
        });
        controlsRef.current = nextControls;
        return nextControls;
      });
    }
    updateLiveConfig({
      ...preset.config,
      particleCount: usesPresetRenderControls ? preset.config.particleCount : current.particleCount,
      width: usesPresetRenderControls ? preset.config.width : current.width,
      height: usesPresetRenderControls ? preset.config.height : current.height,
      depth: usesPresetRenderControls ? preset.config.depth : current.depth
    }, true);
  }, [updateLiveConfig]);

  const applySettingsPreset = useCallback((settings: SavedSettingsPreset) => {
    const nextConfig = sanitizeLiveConfig(settings.liveConfig);
    const nextControls = normalizeRenderControlsForDisplayMode("live", sanitizeRenderControls(settings.controls));
    const uiSource = isRecord(settings.ui) ? settings.ui : settings;
    const nextUi = sanitizeSavedUiState(uiSource, fallbackPlayingForSavedUi(uiSource));
    const nextAudio = sanitizeSavedAudioState(settings.audio);
    setSelectedSettingsId(settings.id);
    setSelectedPresetId(settings.presetId);
    setPlaying(nextUi.playing);
    playingRef.current = nextUi.playing;
    setOverlay(nextUi.overlay);
    overlayRef.current = nextUi.overlay;
    setCameraOrbit(nextUi.cameraOrbit.enabled);
    setCameraOrbitSpeed(nextUi.cameraOrbit.speed);
    cameraOrbitRef.current = nextUi.cameraOrbit.enabled;
    cameraOrbitSpeedRef.current = nextUi.cameraOrbit.speed;
    setViewLocked(nextUi.viewLocked);
    viewLockedRef.current = nextUi.viewLocked;
    setDemoMode(nextUi.demoMode);
    demoModeRef.current = nextUi.demoMode;
    setFrameCap(nextUi.frameCap);
    frameCapRef.current = nextUi.frameCap;
    setTrackingControls(nextUi.trackingControls);
    trackingRef.current = nextUi.trackingControls;
    trackerRef.current.setCohesion(nextUi.trackingControls.cohesion);
    setTimelineEnabled(nextUi.timeline.enabled);
    timelineEnabledRef.current = nextUi.timeline.enabled;
    timelineRef.current = { ...nextUi.timeline.state, playing: nextUi.playing };
    timelineDispatch({ type: "set-loop", loopIn: nextUi.timeline.state.loopIn, loopOut: nextUi.timeline.state.loopOut });
    timelineDispatch({ type: "set-speed", speed: nextUi.timeline.state.playbackSpeed });
    timelineDispatch({ type: "scrub", frame: nextUi.timeline.state.currentFrame });
    timelineDispatch({ type: "play", playing: nextUi.playing });
    controlsRef.current = nextControls;
    liveConfigRef.current = nextConfig;
    audioPanelRef.current = nextAudio.panel;
    sliderModulationsRef.current = nextAudio.sliders;
    setControls(nextControls);
    setLiveConfig(nextConfig);
    setAudioPanel(nextAudio.panel);
    setSliderModulations(nextAudio.sliders);
    audioModulationRuntimeRef.current = createAudioModulationRuntime();
    cameraRef.current = {
      yaw: nextControls.cameraYaw,
      pitch: nextControls.cameraPitch,
      distance: nextControls.cameraDistance,
      panX: nextControls.cameraPanX,
      panY: nextControls.cameraPanY,
      fov: nextControls.fov,
      targetYaw: nextControls.cameraYaw,
      targetPitch: nextControls.cameraPitch,
      targetDistance: nextControls.cameraDistance,
      targetPanX: nextControls.cameraPanX,
      targetPanY: nextControls.cameraPanY,
      targetFov: nextControls.fov
    };
    requestLiveReset();
  }, [requestLiveReset, timelineDispatch]);

  // Expose for preset-sweep automation.
  useEffect(() => {
    window.__fluoddityApplyPreset = (preset: SavedSettingsPreset) => {
      applySettingsPreset(preset);
    };
    return () => { delete window.__fluoddityApplyPreset; };
  }, [applySettingsPreset]);

  const chooseSavedSettings = useCallback((id: string) => {
    if (!id) {
      setSelectedSettingsId("");
      return;
    }
    const settings = savedSettings.find((item) => item.id === id);
    if (settings) applySettingsPreset(settings);
  }, [applySettingsPreset, savedSettings]);

  // On first load, apply the boot saved-settings preset (full apply, so MIDI/audio mappings load
  // too): `?settings=<name|id>` selects explicitly (the website embed passes ?settings=AR11),
  // otherwise the repo NewDefault/AR10 default. Skipped for automation/profiling
  // (?profileGpu or ?skipAppCompute) or an unknown ?settings name, and only runs once.
  const appliedDefaultPresetRef = useRef(false);
  useEffect(() => {
    if (appliedDefaultPresetRef.current) return;
    const targetId = chooseBootSettingsId(window.location.search, savedSettings);
    if (targetId === null) {
      const params = new URLSearchParams(window.location.search);
      if (params.has("profileGpu") || params.has("skipAppCompute") || params.has("settings")) appliedDefaultPresetRef.current = true;
      return;
    }
    const def = savedSettings.find((item) => item.id === targetId);
    if (def) {
      appliedDefaultPresetRef.current = true;
      applySettingsPreset(def);
      // An explicit ?particles= wins over the preset's count — the website embed
      // boots AR11 with a lighter particle budget for frame rate.
      const particlesParam = Number(new URLSearchParams(window.location.search).get("particles"));
      if (Number.isFinite(particlesParam) && particlesParam > 0) {
        setLiveConfig((value) => ({ ...value, particleCount: clampParticleCount(particlesParam) }));
      }
    }
  }, [savedSettings, applySettingsPreset]);

  const saveCurrentSettings = useCallback(async () => {
    const existing = savedSettings.find((item) => item.id === selectedSettingsId);
    const presetName = existing?.name ?? defaultSettingsName(selectedPresetId);
    try {
      const saved = await saveSettingsPresetWithFilePicker(presetName, (name) => (
        buildSavedSettingsPreset(name, selectedPresetId, displayMode, controls, liveConfig, currentSavedUi, currentSavedAudio, existing)
      ));
      if (!saved) return;
      publishSavedSettingsForAutomation(saved);
      const next = upsertSavedSettings(savedSettings, saved);
      persistSavedSettings(next);
      setSavedSettings(next);
      setSelectedSettingsId(saved.id);
    } catch (error) {
      window.alert(error instanceof Error ? error.message : String(error));
    }
  }, [controls, currentSavedAudio, currentSavedUi, displayMode, liveConfig, savedSettings, selectedPresetId, selectedSettingsId]);

  const deleteCurrentSettings = useCallback(() => {
    if (!selectedSettingsId) return;
    const settings = savedSettings.find((item) => item.id === selectedSettingsId);
    if (!settings || settings.fileBacked) return;
    const next = savedSettings.filter((item) => item.id !== selectedSettingsId);
    persistSavedSettings(next);
    setSavedSettings(next);
    setSelectedSettingsId("");
  }, [savedSettings, selectedSettingsId]);

  const exportCurrentSettings = useCallback(async () => {
    const existing = savedSettings.find((item) => item.id === selectedSettingsId);
    try {
      await saveSettingsPresetWithFilePicker(existing?.name ?? getLivePreset(selectedPresetId).name, (name) => (
        buildSavedSettingsPreset(name, selectedPresetId, displayMode, controls, liveConfig, currentSavedUi, currentSavedAudio, existing)
      ));
    } catch (error) {
      window.alert(error instanceof Error ? error.message : String(error));
    }
  }, [controls, currentSavedAudio, currentSavedUi, displayMode, liveConfig, savedSettings, selectedPresetId, selectedSettingsId]);

  const importSettingsFile = useCallback(async (file: File) => {
    const imported = settingsPresetFromJson(await file.text(), settingsNameFromFileName(file.name));
    const next = upsertSavedSettings(savedSettings, imported);
    persistSavedSettings(next);
    setSavedSettings(next);
    applySettingsPreset(imported);
  }, [applySettingsPreset, savedSettings]);

  const openSettingsJson = useCallback(async () => {
    const pickerWindow = window as FilePickerWindow;
    if (pickerWindow.showOpenFilePicker) {
      try {
        const [handle] = await pickerWindow.showOpenFilePicker({
          multiple: false,
          types: settingsFileTypes
        });
        if (!handle) return;
        await importSettingsFile(await handle.getFile());
      } catch (error) {
        if (!isAbortError(error)) {
          window.alert(error instanceof Error ? error.message : String(error));
        }
      }
      return;
    }
    jsonInputRef.current?.click();
  }, [importSettingsFile]);

  const importSettingsJson = useCallback(async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    try {
      await importSettingsFile(file);
    } catch (error) {
      window.alert(error instanceof Error ? error.message : String(error));
    }
  }, [importSettingsFile]);

  const randomizeAllParameters = useCallback(() => {
    const currentConfig = liveConfigRef.current;
    const nextConfig = normalizeTrailKernel({
      ...currentConfig,
      ruleSeed: Math.random(),
      depositMass: randomSliderValue(liveSliderRanges.depositMass),
      depositRadius: randomSliderValue({
        ...trailRadiusSliderRange,
        max: maxSupportedTrailRadius(currentConfig)
      }),
      trailPersistence: randomSliderValue(liveSliderRanges.trailPersistence),
      trailDiffusion: randomSliderValue(liveSliderRanges.trailDiffusion),
      sensorGain: randomSliderValue(liveSliderRanges.sensorGain),
      sensorAngle: randomSliderValue(liveSliderRanges.sensorAngle),
      sensorDistance: randomSliderValue(liveSliderRanges.sensorDistance),
      mutationScale: randomSliderValue(liveSliderRanges.mutationScale),
      drag: randomSliderValue(liveSliderRanges.drag),
      axialForce: randomSliderValue(liveSliderRanges.axialForce),
      lateralForce: randomSliderValue(liveSliderRanges.lateralForce),
      globalForceMult: randomSliderValue(liveSliderRanges.globalForceMult),
      strafePower: randomSliderValue(liveSliderRanges.strafePower),
      strafeMomentum: randomSliderValue(liveSliderRanges.strafeMomentum),
      hazardRate: randomSliderValue(liveSliderRanges.hazardRate),
      initialConditions: randomChoice([0, 2, 4, 5, 6, 7, 8] as const),
      boundaryMode: randomChoice([0, 1, 2, 3, 4, 5, 6] as const),
      domainShape: randomChoice([0, 1, 2, 3] as const),
      absoluteOrientation: randomChoice([0, 1, 2, 3, 4, 5] as const),
      orientationMix: randomSliderValue(liveSliderRanges.orientationMix),
      symmetryAxes: randomChoice([0, 1, 2, 3, 4, 5, 6, 7] as const)
    });
    setSelectedSettingsId("");
    liveConfigRef.current = nextConfig;
    setLiveConfig(nextConfig);
    requestLiveReset();
  }, [requestLiveReset]);

  const setCameraTarget = useCallback((patch: Partial<Pick<OrbitCamera, "yaw" | "pitch" | "distance" | "panX" | "panY">>, immediate = false) => {
    const camera = cameraRef.current;
    const nextYaw = normalizeAngle(patch.yaw ?? camera.targetYaw);
    const nextPitch = clamp(patch.pitch ?? camera.targetPitch, minCameraPitch, maxCameraPitch);
    const nextDistance = clampCameraDistance(patch.distance ?? camera.targetDistance);
    const nextPanX = clamp(patch.panX ?? camera.targetPanX, minCameraPan, maxCameraPan);
    const nextPanY = clamp(patch.panY ?? camera.targetPanY, minCameraPan, maxCameraPan);
    camera.targetYaw = nextYaw;
    camera.targetPitch = nextPitch;
    camera.targetDistance = nextDistance;
    camera.targetPanX = nextPanX;
    camera.targetPanY = nextPanY;
    if (immediate) {
      camera.yaw = nextYaw;
      camera.pitch = nextPitch;
      camera.distance = nextDistance;
      camera.panX = nextPanX;
      camera.panY = nextPanY;
    }
    setControls((value) => ({
      ...value,
      cameraYaw: nextYaw,
      cameraPitch: nextPitch,
      cameraDistance: nextDistance,
      cameraPanX: nextPanX,
      cameraPanY: nextPanY
    }));
  }, []);

  const resetView = useCallback(() => {
    setCameraTarget({ yaw: cameraDefaults.yaw, pitch: cameraDefaults.pitch, distance: cameraDefaults.distance, panX: 0, panY: 0 }, true);
  }, [setCameraTarget]);

  const beginTumble = useCallback((event: PointerEvent<HTMLCanvasElement>) => {
    if (viewLockedRef.current) return;
    if (event.button !== 0 && event.button !== 1) return;
    event.preventDefault();
    const mode: ViewportDragMode = event.button === 1 ? "pan" : "orbit";
    const camera = cameraRef.current;
    setCameraTarget({
      yaw: camera.yaw,
      pitch: camera.pitch,
      distance: camera.distance,
      panX: camera.panX,
      panY: camera.panY
    });
    dragRef.current = { active: true, mode, pointerId: event.pointerId, x: event.clientX, y: event.clientY, startX: event.clientX, startY: event.clientY };
    event.currentTarget.setPointerCapture(event.pointerId);
    setDragMode(mode);
  }, [setCameraTarget]);

  const tumble = useCallback((event: PointerEvent<HTMLCanvasElement>) => {
    if (viewLockedRef.current) return;
    const drag = dragRef.current;
    if (!drag.active || drag.pointerId !== event.pointerId) return;
    event.preventDefault();
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    dragRef.current = { ...drag, x: event.clientX, y: event.clientY };
    const camera = cameraRef.current;
    if (drag.mode === "pan") {
      const canvas = event.currentTarget;
      const width = Math.max(1, canvas.clientWidth);
      const height = Math.max(1, canvas.clientHeight);
      const aspect = width / height;
      const scale = camera.targetDistance / cameraPanFocal;
      setCameraTarget({
        panX: camera.targetPanX + (dx * 2 / width) * aspect * scale,
        panY: camera.targetPanY - (dy * 2 / height) * scale
      });
    } else {
      setCameraTarget({
        yaw: camera.targetYaw + dx * 0.0048,
        pitch: camera.targetPitch - dy * 0.0042
      });
    }
  }, [setCameraTarget]);

  const pickAtClient = useCallback(async (clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, rect.width);
    const height = Math.max(1, rect.height);
    const ndc: [number, number] = [((clientX - rect.left) / width) * 2 - 1, -(((clientY - rect.top) / height) * 2 - 1)];
    const camera = cameraRef.current;
    const ray: Ray = rayForNdc(ndc, { yaw: camera.yaw, pitch: camera.pitch, distance: camera.distance, aspect: width / height });
    const particles = await liveRendererRef.current.captureParticles(TRACKING_PARTICLE_READBACK_LIMIT);
    if (!particles) return;
    const count = particles.length / 12;
    trackerRef.current.setEps(epsForParticleCount(count));
    const seed = pickParticle(ray, particles, count, PICK_CONE_RADIUS);
    if (seed < 0) {
      setTrackState({ active: false, members: 0, miss: "no particle under cursor" });
      return;
    }
    const locked = trackerRef.current.lockFromSeed(particles, count, seed);
    const s = trackerRef.current.state;
    setTrackState({ active: locked && s.active, members: s.members, miss: locked ? undefined : "too sparse here — try a denser blob" });
  }, []);

  const endTumble = useCallback((event: PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    if (drag.pointerId !== null && drag.pointerId !== event.pointerId) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    const moved = Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY);
    if (moved < 4 && drag.mode === "orbit" && (trackingRef.current.follow || trackingRef.current.look)) {
      void pickAtClient(event.clientX, event.clientY);
    }
    dragRef.current = { active: false, mode: null, pointerId: null, x: event.clientX, y: event.clientY, startX: drag.startX, startY: drag.startY };
    setDragMode(null);
  }, [pickAtClient]);

  const applyZoom = useCallback((deltaY: number, deltaMode: number) => {
    if (viewLockedRef.current) return;
    const unit = deltaMode === 1 ? 18 : deltaMode === 2 ? 240 : 1;
    const delta = clamp(deltaY * unit, -420, 420);
    const camera = cameraRef.current;
    const zoomBase = camera.targetDistance <= minCameraDistance && delta > 0
      ? cameraDistanceSliderRange.step
      : camera.targetDistance;
    setCameraTarget({ distance: zoomBase * Math.exp(delta * 0.00145) });
  }, [setCameraTarget]);

  const toggleViewportFullscreen = useCallback(() => {
    if (getFullscreenElement()) {
      const exitFullscreen = getExitFullscreen();
      if (!exitFullscreen) {
        console.warn("Fullscreen API is not available in this browser.");
        return;
      }
      Promise.resolve(exitFullscreen()).catch((error: unknown) => console.warn("Could not exit fullscreen", error));
      return;
    }
    const requestFullscreen = getRequestFullscreen(appShellRef.current ?? canvasRef.current);
    if (!requestFullscreen) {
      console.warn("Fullscreen API is not available in this browser.");
      return;
    }
    Promise.resolve(requestFullscreen()).catch((error: unknown) => console.warn("Could not enter fullscreen", error));
  }, []);

  useEffect(() => {
    const handleKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) return;
      if (isTextEntryTarget(event.target)) return;
      const key = event.key.toLowerCase();
      if (key === "f") {
        event.preventDefault();
        toggleViewportFullscreen();
      } else if (key === "m") {
        event.preventDefault();
        setPanelOpen((value) => !value);
      } else if (key === "r") {
        event.preventDefault();
        togglePerformanceRecordingRef.current();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [toggleViewportFullscreen]);

  // Hide the cursor after 2.5s of no movement while fullscreen; any mouse move (or leaving
  // fullscreen) reveals it again, so the control panel stays usable.
  useEffect(() => {
    let timer = 0;
    const armOrShow = () => {
      setCursorIdle(false);
      window.clearTimeout(timer);
      if (getFullscreenElement()) timer = window.setTimeout(() => setCursorIdle(true), 2500);
    };
    const onFsChange = () => {
      window.clearTimeout(timer);
      if (getFullscreenElement()) timer = window.setTimeout(() => setCursorIdle(true), 2500);
      else setCursorIdle(false);
    };
    window.addEventListener("mousemove", armOrShow);
    window.addEventListener("pointerdown", armOrShow);
    document.addEventListener("fullscreenchange", onFsChange);
    document.addEventListener("webkitfullscreenchange", onFsChange);
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener("mousemove", armOrShow);
      window.removeEventListener("pointerdown", armOrShow);
      document.removeEventListener("fullscreenchange", onFsChange);
      document.removeEventListener("webkitfullscreenchange", onFsChange);
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const stopViewportWheel = (event: globalThis.WheelEvent) => {
      event.preventDefault();
      event.stopPropagation();
      applyZoom(event.deltaY, event.deltaMode);
    };
    canvas.addEventListener("wheel", stopViewportWheel, { passive: false });
    return () => {
      canvas.removeEventListener("wheel", stopViewportWheel);
    };
  }, [applyZoom]);

  const preset = useMemo(() => JSON.stringify({
    version: 2,
    name: getLivePreset(selectedPresetId).name,
    presetId: selectedPresetId,
    displayMode,
    seed: deterministicSeed,
    controls: normalizeRenderControlsForDisplayMode(
      displayMode,
      renderControlsWithModulationRangeOverrides(sanitizeRenderControls(controls), controls, currentSavedAudio.sliders)
    ),
    liveConfig: sanitizeLiveConfig(liveConfig),
    ui: currentSavedUi,
    audio: currentSavedAudio
  }, null, 2), [controls, currentSavedAudio, currentSavedUi, displayMode, liveConfig, selectedPresetId]);
  const vertexCount = liveVertexCount(liveDiagnostics, liveConfig);
  return (
    <SliderModulationContext.Provider value={sliderModulationContext}>
      <main ref={appShellRef} className={`app-shell${panelOpen ? "" : " panel-collapsed"}${cursorIdle ? " cursor-idle" : ""}`}>
      <section className={`viewport-band${portraitMode ? " is-portrait" : ""}`} data-testid="viewport-band">
        <canvas
          ref={canvasRef}
          className={`simulation-canvas${dragMode === "orbit" ? " is-tumbling" : dragMode === "pan" ? " is-panning" : ""}`}
          data-testid="simulation-canvas"
          onPointerDown={beginTumble}
          onPointerMove={tumble}
          onPointerUp={endTumble}
          onPointerCancel={endTumble}
          onLostPointerCapture={endTumble}
          onAuxClick={(event) => event.preventDefault()}
        />
        {panelOpen ? (
          <div className="viewport-stats" data-testid="viewport-stats" aria-label="Viewport performance">
            <div><span>FPS</span><strong>{fps > 0 ? fps.toFixed(0) : "..."}</strong></div>
            <div><span>Verts</span><strong>{formatCount(vertexCount)}</strong></div>
          </div>
        ) : null}
        <script
          type="application/json"
          data-testid="profile-json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              webgpu,
              mode: "live",
              renderer: liveDiagnostics,
              live: liveDiagnostics,
              controls,
              audio: {
                status: audioStatus,
                activeInput: audioActiveInput,
                pendingInput: audioPendingInput,
                inputs: audioInputDevices,
                error: audioBackendError,
                panel: audioPanel,
                meters: audioMeters,
                lastSequence: audioLastFrame?.sequence ?? null,
                sliders: sliderModulations
              },
              midi: {
                status: midiStatus,
                inputs: midiInputs,
                activeInputId: midiActiveInputId,
                learningKey: midiLearningKey
              },
              fps,
              vertexCount,
              preset: selectedPresetId,
              liveConfig,
              frames: frame,
              playing,
              overlay,
              profileGpu
            }).replace(/</g, "\\u003c")
          }}
        />
        {loadError ? <div className="load-error">{loadError}</div> : null}
      </section>

      <aside id="control-panel" className="control-panel" data-testid="control-panel" aria-label="3D Life cockpit controls">
        <div className="panel-row button-row">
          <button data-testid="play-toggle" onClick={() => setPlaying((value) => !value)}>{playing ? "Pause" : "Play"}</button>
          <button data-testid="timeline-toggle" className={timelineEnabled ? "active" : ""} onClick={() => setTimelineEnabled((value) => !value)}>{timelineEnabled ? "Timeline: On" : "Timeline: Off"}</button>
          <button data-testid="reset" onClick={reset}>Reset</button>
          <button data-testid="reset-view" onClick={resetView}>Reset View</button>
          <button data-testid="render-preview" onClick={() => setOverlay((value) => !value)}>Render Preview</button>
          <button
            className={portraitMode ? "active" : ""}
            data-testid="portrait-mode"
            onClick={() => setPortraitMode((value) => !value)}
            title="Frame the canvas as 9:16 portrait so the live composition and recording match Instagram"
          >
            {portraitMode ? "Portrait 9:16: On" : "Portrait 9:16"}
          </button>
          <button
            className={recordingPerformance ? "active" : ""}
            data-testid="record-performance"
            onClick={() => void togglePerformanceRecording()}
            title="Record this live performance + audio (or press R) to re-render as HDR video offline"
          >
            {recordingPerformance
              ? `● Stop Recording${audioCaptured ? " (+audio)" : ""}`
              : "Record Performance"}
          </button>
          <button
            className={recordMidiLearning ? "active" : ""}
            data-testid="record-midi-learn"
            onClick={() => (recordMidiLearning ? setRecordMidiLearning(false) : startRecordMidiLearn())}
            title="MIDI-map the record toggle to a single pad/button (press to learn, then hit a pad)"
          >
            {recordMidiLearning
              ? "Listening… press a pad"
              : recordMidiMapping?.control
                ? `Rec MIDI: ${midiControlLabel(recordMidiMapping.control)}`
                : "MIDI-map Rec"}
          </button>
          <button className="wide-button" data-testid="randomize-all" onClick={randomizeAllParameters}>Randomize all parameters</button>
        </div>

        {timelineEnabled && (
          <TimelineBar
            state={timeline}
            onTogglePlay={() => setPlaying((value) => !value)}
            onScrub={(targetFrame) => timelineDispatch({ type: "scrub", frame: targetFrame })}
            onSetLoop={(loopIn, loopOut) => timelineDispatch({ type: "set-loop", loopIn, loopOut })}
            onSetSpeed={(speed) => timelineDispatch({ type: "set-speed", speed })}
          />
        )}

        <PresetSelect value={selectedPresetId} onChange={applyPreset} />
        <ControlGroup title="Settings">
          <SavedSettingsSelect value={selectedSettingsId} settings={savedSettings} onChange={chooseSavedSettings} />
          <div className="settings-actions">
            <button data-testid="save-settings" onClick={saveCurrentSettings}>Save</button>
            <button data-testid="export-settings" onClick={exportCurrentSettings}>Export JSON</button>
            <button data-testid="import-settings" onClick={openSettingsJson}>Load JSON</button>
            <button data-testid="delete-settings" onClick={deleteCurrentSettings} disabled={!selectedSettingsId || savedSettings.find((settings) => settings.id === selectedSettingsId)?.fileBacked === true}>Delete</button>
          </div>
          <input
            ref={jsonInputRef}
            className="hidden-file-input"
            data-testid="json-preset-input"
            type="file"
            accept="application/json,.json"
            onChange={importSettingsJson}
          />
        </ControlGroup>
        <NumberField label="Seed" value={deterministicSeed} readOnly testId="seed-input" />
        <ControlGroup title="Particles">
          <NumberField
            label="Particles"
            value={liveConfig.particleCount}
            min={minLiveParticles}
            step={liveParticleStep}
            testId="particle-input"
            onChange={updateParticleBudget}
          />
          <Slider label="P Glow" value={controls.particleGlowCore} {...renderSliderRanges.particleGlowCore} testId="particle-glow-core-slider" onChange={(particleGlowCore) => setControls((c) => ({ ...c, particleGlowCore }))} />
          <Slider label="P Hot" value={controls.particleHotCore} {...renderSliderRanges.particleHotCore} testId="particle-hot-core-slider" onChange={(particleHotCore) => setControls((c) => ({ ...c, particleHotCore }))} />
          <Slider label="Density Exponent" value={controls.particleExponent} {...renderSliderRanges.particleExponent} testId="particle-exponent-slider" onChange={(particleExponent) => setControls((c) => ({ ...c, particleExponent }))} />
          <Slider label="Brightness Boost" value={controls.particleBrightnessBoost} {...renderSliderRanges.particleBrightnessBoost} testId="particle-brightness-boost-slider" onChange={(particleBrightnessBoost) => setControls((c) => ({ ...c, particleBrightnessBoost }))} />
          <ParticleColorModeSelect value={controls.particleColorMode} onChange={(particleColorMode) => setControls((c) => ({ ...c, particleColorMode }))} />
          <ModeSlider
            label="P Color"
            value={controls.particleColorMode}
            testId="particle-color-mode-slider"
            options={particleColorSweep}
            onChange={(value) => setControls((c) => ({ ...c, particleColorMode: value as RenderControls["particleColorMode"] }))}
          />
          <ColorField label="P Tint" value={controls.particleTint} onChange={(particleTint) => setControls((c) => ({ ...c, particleTint }))} />
          <ParticleBlendSelect value={controls.particleBlendMode} onChange={(particleBlendMode) => setControls((c) => ({ ...c, particleBlendMode }))} />
          {/* P Bright removed: in additive blend it and P Opacity are the same multiplier on the
              final contribution. P Opacity is the single particle-intensity knob now; brightness
              stays at its preset/default value so saved looks (incl. presets) are unchanged. */}
          <Slider label="P Scale" value={controls.particleSizePx} {...renderSliderRanges.particleSizePx} testId="particle-size-slider" onChange={(particleSizePx) => setControls((c) => ({ ...c, particleSizePx }))} />
          <Slider label="P Opacity" value={controls.particleOpacity} {...renderSliderRanges.particleOpacity} onChange={(particleOpacity) => setControls((c) => ({ ...c, particleOpacity }))} />
          <Slider label="P Cutoff" value={controls.particleDensityCutoff} {...renderSliderRanges.particleDensityCutoff} curve={3} testId="particle-density-cutoff-slider" onChange={(particleDensityCutoff) => setControls((c) => ({ ...c, particleDensityCutoff }))} />
          <Slider label="P Radius" value={controls.particleDensityRadius} {...renderSliderRanges.particleDensityRadius} testId="particle-density-radius-slider" onChange={(particleDensityRadius) => setControls((c) => ({ ...c, particleDensityRadius }))} />
          <>
              <Slider label="P Keep" value={controls.particleDensityNormalize} {...renderSliderRanges.particleDensityNormalize} testId="particle-density-normalize-slider" onChange={(particleDensityNormalize) => setControls((c) => ({ ...c, particleDensityNormalize }))} />
              <Slider label="P Soft" value={controls.particleDensitySoftness} {...renderSliderRanges.particleDensitySoftness} testId="particle-density-softness-slider" onChange={(particleDensitySoftness) => setControls((c) => ({ ...c, particleDensitySoftness }))} />
              <Slider label="Support" value={controls.particleSupportMask} {...renderSliderRanges.particleSupportMask} testId="particle-support-mask-slider" onChange={(particleSupportMask) => setControls((c) => ({ ...c, particleSupportMask }))} />
              <Slider label="Supp Rad" value={controls.particleSupportRadius} {...renderSliderRanges.particleSupportRadius} testId="particle-support-radius-slider" onChange={(particleSupportRadius) => setControls((c) => ({ ...c, particleSupportRadius }))} />
              <Slider label="Neighbors" value={controls.particleSupportNeighbors} {...renderSliderRanges.particleSupportNeighbors} testId="particle-support-neighbors-slider" onChange={(particleSupportNeighbors) => setControls((c) => ({ ...c, particleSupportNeighbors }))} />
              <Slider label="Flow Agree" value={controls.particleSupportFlow} {...renderSliderRanges.particleSupportFlow} testId="particle-support-flow-slider" onChange={(particleSupportFlow) => setControls((c) => ({ ...c, particleSupportFlow }))} />
              <Slider label="Haze Cull" value={controls.particleHazeCull} {...renderSliderRanges.particleHazeCull} testId="particle-haze-cull-slider" onChange={(particleHazeCull) => setControls((c) => ({ ...c, particleHazeCull }))} />
              <Slider label="Despeckle" value={controls.particleDespeckle} {...renderSliderRanges.particleDespeckle} testId="particle-despeckle-slider" onChange={(particleDespeckle) => setControls((c) => ({ ...c, particleDespeckle }))} />
              <Slider label="Supp Smooth" value={controls.particleSupportSmoothing} {...renderSliderRanges.particleSupportSmoothing} testId="particle-support-smoothing-slider" onChange={(particleSupportSmoothing) => setControls((c) => ({ ...c, particleSupportSmoothing }))} />
              <Checkbox label="Cutoff Prepass" checked={controls.particleCutoffPrepass} testId="cutoff-prepass-checkbox" onChange={(particleCutoffPrepass) => setControls((c) => ({ ...c, particleCutoffPrepass }))} />
          </>
          <Checkbox label="Recycle" checked={liveConfig.recycleEnabled} onChange={(recycleEnabled) => updateLiveConfig({ recycleEnabled })} />
          <Slider label="Flow Color" value={liveConfig.hueSensitivity} {...liveSliderRanges.hueSensitivity} testId="hue-sensitivity-slider" onChange={(hueSensitivity) => updateLiveConfig({ hueSensitivity })} />
          <Checkbox label="Vel Stretch" checked={controls.particleVelocityStretch} testId="particle-velocity-stretch-checkbox" onChange={(particleVelocityStretch) => setControls((c) => ({ ...c, particleVelocityStretch }))} />
          <Slider label="Stretch Min" value={controls.particleStretchMin} {...renderSliderRanges.particleStretchMin} testId="particle-stretch-min-slider" onChange={(particleStretchMin) => setControls((c) => ({ ...c, particleStretchMin, particleStretch: Math.max(c.particleStretch, particleStretchMin) }))} />
          <Slider label="Stretch Max" value={controls.particleStretch} {...renderSliderRanges.particleStretch} testId="particle-stretch-max-slider" onChange={(particleStretch) => setControls((c) => ({ ...c, particleStretch, particleStretchMin: Math.min(c.particleStretchMin, particleStretch) }))} />
          <Slider label="Stretch Speed" value={controls.particleStretchSpeed} {...renderSliderRanges.particleStretchSpeed} testId="particle-stretch-speed-slider" onChange={(particleStretchSpeed) => setControls((c) => ({ ...c, particleStretchSpeed }))} />
          <Slider label="P Speed Cut" value={controls.particleSpeedCutoff} {...renderSliderRanges.particleSpeedCutoff} testId="particle-speed-cutoff-slider" onChange={(particleSpeedCutoff) => setControls((c) => ({ ...c, particleSpeedCutoff }))} />
          <Slider label="P Slow Cut" value={controls.particleSlowCutoff} {...renderSliderRanges.particleSlowCutoff} testId="particle-slow-cutoff-slider" onChange={(particleSlowCutoff) => setControls((c) => ({ ...c, particleSlowCutoff }))} />
        </ControlGroup>

        <ControlGroup title="Variation">
          <Slider label="Var Master" value={controls.variationMaster} {...renderSliderRanges.variationMaster} testId="variation-master-slider" onChange={(variationMaster) => setControls((c) => ({ ...c, variationMaster }))} />
          <Slider label="Var Drift" value={controls.variationDrift} {...renderSliderRanges.variationDrift} testId="variation-drift-slider" onChange={(variationDrift) => setControls((c) => ({ ...c, variationDrift }))} />
          <Slider label="Var Coherence" value={controls.variationNoiseMix} {...renderSliderRanges.variationNoiseMix} testId="variation-noise-mix-slider" onChange={(variationNoiseMix) => setControls((c) => ({ ...c, variationNoiseMix }))} />
          <Slider label="Var Scale" value={controls.variationFreq} {...renderSliderRanges.variationFreq} testId="variation-freq-slider" onChange={(variationFreq) => setControls((c) => ({ ...c, variationFreq }))} />
          <Slider label="Var Octaves" value={controls.variationOctaves} {...renderSliderRanges.variationOctaves} testId="variation-octaves-slider" onChange={(variationOctaves) => setControls((c) => ({ ...c, variationOctaves }))} />
          <Slider label="Var Gain" value={controls.variationGain} {...renderSliderRanges.variationGain} testId="variation-gain-slider" onChange={(variationGain) => setControls((c) => ({ ...c, variationGain }))} />
          <Slider label="Var Detail" value={controls.variationLacunarity} {...renderSliderRanges.variationLacunarity} testId="variation-lacunarity-slider" onChange={(variationLacunarity) => setControls((c) => ({ ...c, variationLacunarity }))} />
          <Slider label="Size Amt" value={controls.variationSizeAmount} {...renderSliderRanges.variationSizeAmount} testId="variation-size-amount-slider" onChange={(variationSizeAmount) => setControls((c) => ({ ...c, variationSizeAmount }))} />
          <Slider label="Size Curve" value={controls.variationSizeCurve} {...renderSliderRanges.variationSizeCurve} curve={3} testId="variation-size-curve-slider" onChange={(variationSizeCurve) => setControls((c) => ({ ...c, variationSizeCurve }))} />
          <Slider label="Size Min" value={controls.variationSizeMin} {...renderSliderRanges.variationSizeMin} testId="variation-size-min-slider" onChange={(variationSizeMin) => setControls((c) => ({ ...c, variationSizeMin, variationSizeMax: Math.max(c.variationSizeMax, variationSizeMin) }))} />
          <Slider label="Size Max" value={controls.variationSizeMax} {...renderSliderRanges.variationSizeMax} testId="variation-size-max-slider" onChange={(variationSizeMax) => setControls((c) => ({ ...c, variationSizeMax, variationSizeMin: Math.min(c.variationSizeMin, variationSizeMax) }))} />
          <Slider label="Bright Amt" value={controls.variationBrightAmount} {...renderSliderRanges.variationBrightAmount} testId="variation-bright-amount-slider" onChange={(variationBrightAmount) => setControls((c) => ({ ...c, variationBrightAmount }))} />
          <Slider label="Bright Curve" value={controls.variationBrightCurve} {...renderSliderRanges.variationBrightCurve} curve={3} testId="variation-bright-curve-slider" onChange={(variationBrightCurve) => setControls((c) => ({ ...c, variationBrightCurve }))} />
          <Slider label="Bright Min" value={controls.variationBrightMin} {...renderSliderRanges.variationBrightMin} testId="variation-bright-min-slider" onChange={(variationBrightMin) => setControls((c) => ({ ...c, variationBrightMin, variationBrightMax: Math.max(c.variationBrightMax, variationBrightMin) }))} />
          <Slider label="Bright Max" value={controls.variationBrightMax} {...renderSliderRanges.variationBrightMax} testId="variation-bright-max-slider" onChange={(variationBrightMax) => setControls((c) => ({ ...c, variationBrightMax, variationBrightMin: Math.min(c.variationBrightMin, variationBrightMax) }))} />
          <Slider label="Opacity Amt" value={controls.variationOpacityAmount} {...renderSliderRanges.variationOpacityAmount} testId="variation-opacity-amount-slider" onChange={(variationOpacityAmount) => setControls((c) => ({ ...c, variationOpacityAmount }))} />
          <Slider label="Opacity Curve" value={controls.variationOpacityCurve} {...renderSliderRanges.variationOpacityCurve} curve={3} testId="variation-opacity-curve-slider" onChange={(variationOpacityCurve) => setControls((c) => ({ ...c, variationOpacityCurve }))} />
          <Slider label="Opacity Min" value={controls.variationOpacityMin} {...renderSliderRanges.variationOpacityMin} testId="variation-opacity-min-slider" onChange={(variationOpacityMin) => setControls((c) => ({ ...c, variationOpacityMin, variationOpacityMax: Math.max(c.variationOpacityMax, variationOpacityMin) }))} />
          <Slider label="Opacity Max" value={controls.variationOpacityMax} {...renderSliderRanges.variationOpacityMax} testId="variation-opacity-max-slider" onChange={(variationOpacityMax) => setControls((c) => ({ ...c, variationOpacityMax, variationOpacityMin: Math.min(c.variationOpacityMin, variationOpacityMax) }))} />
          <Slider label="Color Amt" value={controls.variationColorAmount} {...renderSliderRanges.variationColorAmount} testId="variation-color-amount-slider" onChange={(variationColorAmount) => setControls((c) => ({ ...c, variationColorAmount }))} />
          <Slider label="Color Curve" value={controls.variationColorCurve} {...renderSliderRanges.variationColorCurve} curve={3} testId="variation-color-curve-slider" onChange={(variationColorCurve) => setControls((c) => ({ ...c, variationColorCurve }))} />
          <Slider label="Hue Lo" value={controls.variationColorMin} {...renderSliderRanges.variationColorMin} testId="variation-color-min-slider" onChange={(variationColorMin) => setControls((c) => ({ ...c, variationColorMin, variationColorMax: Math.max(c.variationColorMax, variationColorMin) }))} />
          <Slider label="Hue Hi" value={controls.variationColorMax} {...renderSliderRanges.variationColorMax} testId="variation-color-max-slider" onChange={(variationColorMax) => setControls((c) => ({ ...c, variationColorMax, variationColorMin: Math.min(c.variationColorMin, variationColorMax) }))} />
        </ControlGroup>

        <ControlGroup title="Render">
          <Slider label="Exposure" value={controls.exposure} {...renderSliderRanges.exposure} onChange={(exposure) => setControls((c) => ({ ...c, exposure }))} />
          <Slider label="Scene Gain" value={controls.sceneBrightness} {...renderSliderRanges.sceneBrightness} onChange={(sceneBrightness) => setControls((c) => ({ ...c, sceneBrightness }))} />
          <Slider label="Bloom" value={controls.bloomStrength} {...renderSliderRanges.bloomStrength} testId="bloom-strength-slider" onChange={(bloomStrength) => setControls((c) => ({ ...c, bloomStrength }))} />
          <Slider label="Bloom Thresh" value={controls.bloomThreshold} {...renderSliderRanges.bloomThreshold} testId="bloom-threshold-slider" onChange={(bloomThreshold) => setControls((c) => ({ ...c, bloomThreshold }))} />
          <Slider label="Bloom Radius" value={controls.bloomRadius} {...renderSliderRanges.bloomRadius} testId="bloom-radius-slider" onChange={(bloomRadius) => setControls((c) => ({ ...c, bloomRadius }))} />
          <Slider label="Color Sat" value={controls.colorSaturation} {...renderSliderRanges.colorSaturation} testId="color-saturation-slider" onChange={(colorSaturation) => setControls((c) => ({ ...c, colorSaturation }))} />
          <Slider label="Contrast" value={controls.colorContrast} {...renderSliderRanges.colorContrast} testId="color-contrast-slider" onChange={(colorContrast) => setControls((c) => ({ ...c, colorContrast }))} />
          <Slider label="Chroma Ab" value={controls.chromaticAberration} {...renderSliderRanges.chromaticAberration} testId="chromatic-aberration-slider" onChange={(chromaticAberration) => setControls((c) => ({ ...c, chromaticAberration }))} />
          <Slider label="Vignette" value={controls.vignetteStrength} {...renderSliderRanges.vignetteStrength} testId="vignette-strength-slider" onChange={(vignetteStrength) => setControls((c) => ({ ...c, vignetteStrength }))} />
          <Slider label="Vign Soft" value={controls.vignetteSoftness} {...renderSliderRanges.vignetteSoftness} testId="vignette-softness-slider" onChange={(vignetteSoftness) => setControls((c) => ({ ...c, vignetteSoftness }))} />
          <Slider label="Flare" value={controls.streakStrength} {...renderSliderRanges.streakStrength} testId="streak-strength-slider" onChange={(streakStrength) => setControls((c) => ({ ...c, streakStrength }))} />
          <Slider label="Flare Len" value={controls.streakLength} {...renderSliderRanges.streakLength} testId="streak-length-slider" onChange={(streakLength) => setControls((c) => ({ ...c, streakLength }))} />
          <Slider label="Flare Star" value={controls.streakVertical} {...renderSliderRanges.streakVertical} testId="streak-vertical-slider" onChange={(streakVertical) => setControls((c) => ({ ...c, streakVertical }))} />
          <Slider label="Flare Height" value={controls.flareHeight} {...renderSliderRanges.flareHeight} testId="flare-height-slider" onChange={(flareHeight) => setControls((c) => ({ ...c, flareHeight }))} />
          <Slider label="Flare Cutoff" value={controls.flareCutoff} {...renderSliderRanges.flareCutoff} testId="flare-cutoff-slider" onChange={(flareCutoff) => setControls((c) => ({ ...c, flareCutoff }))} />
          <Slider label="Render Res" value={controls.rayResolution} {...rayResolutionSliderRange} onChange={(rayResolution) => setControls((c) => ({ ...c, rayResolution: clampRayResolution(rayResolution) }))} />
          <NumberField
            label="Sim Res"
            value={liveConfig.width}
            min={minVolumeSize}
            max={maxVolumeSize}
            step={volumeStep}
            testId="volume-input"
            onChange={(size) => {
              const volumeSize = clampVolumeSize(size);
              updateLiveConfig({ width: volumeSize, height: volumeSize, depth: volumeSize }, true);
            }}
          />
        </ControlGroup>

        <ControlGroup title="Trail Field">
          <Slider label="Emit" value={liveConfig.depositMass} {...liveSliderRanges.depositMass} testId="trail-amount-slider" onChange={(depositMass) => updateLiveConfig({ depositMass })} />
          <Slider label="Radius" value={liveConfig.depositRadius} min={trailRadiusSliderRange.min} max={maxSupportedTrailRadius(liveConfig)} step={trailRadiusSliderRange.step} testId="trail-radius-slider" onChange={(depositRadius) => updateLiveConfig({ depositRadius })} />
          <Slider label="Persist" value={liveConfig.trailPersistence} {...liveSliderRanges.trailPersistence} testId="trail-persistence-slider" onChange={(trailPersistence) => updateLiveConfig({ trailPersistence })} />
          <Slider label="Diffuse" value={liveConfig.trailDiffusion} {...liveSliderRanges.trailDiffusion} testId="trail-diffusion-slider" onChange={(trailDiffusion) => updateLiveConfig({ trailDiffusion })} />
          <Slider label="Pulse" value={liveConfig.pulseDepth} {...liveSliderRanges.pulseDepth} testId="pulse-depth-slider" onChange={(pulseDepth) => updateLiveConfig({ pulseDepth })} />
          <Slider label="Pulse Rate" value={liveConfig.pulseRate} {...liveSliderRanges.pulseRate} testId="pulse-rate-slider" onChange={(pulseRate) => updateLiveConfig({ pulseRate })} />
          <Slider label="Restless" value={liveConfig.restlessness} {...liveSliderRanges.restlessness} testId="restlessness-slider" onChange={(restlessness) => updateLiveConfig({ restlessness })} />
        </ControlGroup>

        <ControlGroup title="Motion">
          <Slider label="Sim Speed" value={liveConfig.simulationSpeed} {...liveSliderRanges.simulationSpeed} testId="sim-speed-slider" onChange={(simulationSpeed) => updateLiveConfig({ simulationSpeed })} />
          <Slider label="Sensor" value={liveConfig.sensorGain} {...liveSliderRanges.sensorGain} onChange={(sensorGain) => updateLiveConfig({ sensorGain })} />
          <Slider label="Angle" value={liveConfig.sensorAngle} {...liveSliderRanges.sensorAngle} onChange={(sensorAngle) => updateLiveConfig({ sensorAngle })} />
          <Slider label="Distance" value={liveConfig.sensorDistance} {...liveSliderRanges.sensorDistance} onChange={(sensorDistance) => updateLiveConfig({ sensorDistance })} />
          <Slider label="Mutation" value={liveConfig.mutationScale} {...liveSliderRanges.mutationScale} onChange={(mutationScale) => updateLiveConfig({ mutationScale })} />
          <Slider label="Drag" value={liveConfig.drag} {...liveSliderRanges.drag} onChange={(drag) => updateLiveConfig({ drag })} />
          <Slider label="Axial" value={liveConfig.axialForce} {...liveSliderRanges.axialForce} onChange={(axialForce) => updateLiveConfig({ axialForce })} />
          <Slider label="Lateral" value={liveConfig.lateralForce} {...liveSliderRanges.lateralForce} onChange={(lateralForce) => updateLiveConfig({ lateralForce })} />
          <Slider label="Force" value={liveConfig.globalForceMult} {...liveSliderRanges.globalForceMult} onChange={(globalForceMult) => updateLiveConfig({ globalForceMult })} />
          <Slider label="Strafe" value={liveConfig.strafePower} {...liveSliderRanges.strafePower} onChange={(strafePower) => updateLiveConfig({ strafePower })} />
          <Slider label="Strafe Momentum" value={liveConfig.strafeMomentum} {...liveSliderRanges.strafeMomentum} onChange={(strafeMomentum) => updateLiveConfig({ strafeMomentum })} />
          <Slider label="Hazard" value={liveConfig.hazardRate} {...liveSliderRanges.hazardRate} onChange={(hazardRate) => updateLiveConfig({ hazardRate })} />
        </ControlGroup>

        <ControlGroup title="Emergent Behaviors">
          <Slider label="MIPS" value={liveConfig.mips} {...emergentSliderRanges.mips} testId="mips-slider" onChange={(mips) => updateLiveConfig({ mips })} />
          <Slider label="Aniso Follow" value={liveConfig.anisoFollow} {...emergentSliderRanges.anisoFollow} testId="aniso-follow-slider" onChange={(anisoFollow) => updateLiveConfig({ anisoFollow })} />
          <Slider label="Flock Align" value={liveConfig.flockAlign} {...emergentSliderRanges.flockAlign} testId="flock-align-slider" onChange={(flockAlign) => updateLiveConfig({ flockAlign })} />
          <Slider label="Flock Sep" value={liveConfig.flockSeparate} {...emergentSliderRanges.flockSeparate} testId="flock-separate-slider" onChange={(flockSeparate) => updateLiveConfig({ flockSeparate })} />
          <Slider label="Chemotaxis" value={liveConfig.chemotaxis} {...emergentSliderRanges.chemotaxis} testId="chemotaxis-slider" onChange={(chemotaxis) => updateLiveConfig({ chemotaxis })} />
          <Slider label="Quorum" value={liveConfig.quorumStrength} {...emergentSliderRanges.quorumStrength} testId="quorum-strength-slider" onChange={(quorumStrength) => updateLiveConfig({ quorumStrength })} />
          <Slider label="Quorum Thr" value={liveConfig.quorumThreshold} {...emergentSliderRanges.quorumThreshold} testId="quorum-threshold-slider" onChange={(quorumThreshold) => updateLiveConfig({ quorumThreshold })} />
          <Slider label="Lenia" value={liveConfig.leniaStrength} {...emergentSliderRanges.leniaStrength} testId="lenia-strength-slider" onChange={(leniaStrength) => updateLiveConfig({ leniaStrength })} />
          <Slider label="Lenia Mu" value={liveConfig.leniaCenter} {...emergentSliderRanges.leniaCenter} testId="lenia-center-slider" onChange={(leniaCenter) => updateLiveConfig({ leniaCenter })} />
          <Slider label="Lenia Sigma" value={liveConfig.leniaWidth} {...emergentSliderRanges.leniaWidth} testId="lenia-width-slider" onChange={(leniaWidth) => updateLiveConfig({ leniaWidth })} />
          <Slider label="Species" value={liveConfig.speciesForce} {...emergentSliderRanges.speciesForce} testId="species-force-slider" onChange={(speciesForce) => updateLiveConfig({ speciesForce })} />
          <Slider label="Predator" value={liveConfig.predator} {...emergentSliderRanges.predator} testId="predator-slider" onChange={(predator) => updateLiveConfig({ predator })} />
          <Slider label="Alarm" value={liveConfig.alarm} {...emergentSliderRanges.alarm} testId="alarm-slider" onChange={(alarm) => updateLiveConfig({ alarm })} />
          <Slider label="Gray-Scott" value={liveConfig.grayScott} {...emergentSliderRanges.grayScott} testId="gray-scott-slider" onChange={(grayScott) => updateLiveConfig({ grayScott })} />
          <Slider label="GS Feed" value={liveConfig.gsFeed} {...emergentSliderRanges.gsFeed} testId="gs-feed-slider" onChange={(gsFeed) => updateLiveConfig({ gsFeed })} />
          <Slider label="GS Kill" value={liveConfig.gsKill} {...emergentSliderRanges.gsKill} testId="gs-kill-slider" onChange={(gsKill) => updateLiveConfig({ gsKill })} />
          <Slider label="Energy" value={liveConfig.energy} {...emergentSliderRanges.energy} testId="energy-slider" onChange={(energy) => updateLiveConfig({ energy })} />
          <Slider label="Energy Drain" value={liveConfig.energyDrain} {...emergentSliderRanges.energyDrain} testId="energy-drain-slider" onChange={(energyDrain) => updateLiveConfig({ energyDrain })} />
        </ControlGroup>

        <ControlGroup title="State">
          <ModeSelect
            label="Shape"
            value={String(liveConfig.domainShape)}
            options={[
              ["0", "Cube"],
              ["1", "Sphere"],
              ["2", "Cylinder"],
              ["3", "Pyramid"]
            ]}
            onChange={(value) => updateLiveConfig({ domainShape: Number(value) as LiveGpu3dConfig["domainShape"] })}
          />
          <ModeSlider
            label="Shape"
            testId="domain-shape-slider"
            value={String(liveConfig.domainShape)}
            options={[
              ["0", "Cube"],
              ["1", "Sphere"],
              ["2", "Cylinder"],
              ["3", "Pyramid"]
            ]}
            onChange={(value) => updateLiveConfig({ domainShape: Number(value) as LiveGpu3dConfig["domainShape"] })}
          />
          <ModeSelect
            label="Initial"
            value={String(liveConfig.initialConditions)}
            options={[
              ["0", "Grid"],
              ["2", "Ring"],
              ["4", "Helix"],
              ["5", "Disc"],
              ["6", "Dual"],
              ["7", "Shell"],
              ["8", "Vortex"]
            ]}
            onChange={(value) => updateLiveConfig({ initialConditions: Number(value) as LiveGpu3dConfig["initialConditions"] })}
          />
          <ModeSlider
            label="Initial"
            testId="initial-conditions-slider"
            value={String(liveConfig.initialConditions)}
            options={[
              ["0", "Grid"],
              ["2", "Ring"],
              ["4", "Helix"],
              ["5", "Disc"],
              ["6", "Dual"],
              ["7", "Shell"],
              ["8", "Vortex"]
            ]}
            onChange={(value) => updateLiveConfig({ initialConditions: Number(value) as LiveGpu3dConfig["initialConditions"] })}
          />
          <ModeSelect
            label="Boundary"
            value={String(liveConfig.boundaryMode)}
            options={[
              ["0", "Bounce"],
              ["1", "Reset"],
              ["2", "Wrap"],
              ["3", "Tube"],
              ["4", "Sticky"],
              ["5", "Soft"],
              ["6", "Portal"]
            ]}
            onChange={(value) => updateLiveConfig({ boundaryMode: Number(value) as LiveGpu3dConfig["boundaryMode"] })}
          />
          <ModeSlider
            label="Boundary"
            testId="boundary-mode-slider"
            value={String(liveConfig.boundaryMode)}
            options={[
              ["0", "Bounce"],
              ["1", "Reset"],
              ["2", "Wrap"],
              ["3", "Tube"],
              ["4", "Sticky"],
              ["5", "Soft"],
              ["6", "Portal"]
            ]}
            onChange={(value) => updateLiveConfig({ boundaryMode: Number(value) as LiveGpu3dConfig["boundaryMode"] })}
          />
          <ModeSelect
            label="Orient"
            value={String(liveConfig.absoluteOrientation)}
            options={[
              ["0", "Velocity"],
              ["1", "World Y"],
              ["2", "Radial"],
              ["3", "Outward"],
              ["4", "Swirl"],
              ["5", "Noise"]
            ]}
            onChange={(value) => updateLiveConfig({ absoluteOrientation: Number(value) as LiveGpu3dConfig["absoluteOrientation"] })}
          />
          <ModeSlider
            label="Orient"
            testId="absolute-orientation-slider"
            value={String(liveConfig.absoluteOrientation)}
            options={[
              ["0", "Velocity"],
              ["1", "World Y"],
              ["2", "Radial"],
              ["3", "Outward"],
              ["4", "Swirl"],
              ["5", "Noise"]
            ]}
            onChange={(value) => updateLiveConfig({ absoluteOrientation: Number(value) as LiveGpu3dConfig["absoluteOrientation"] })}
          />
          <Slider label="Mix" value={liveConfig.orientationMix} {...liveSliderRanges.orientationMix} onChange={(orientationMix) => updateLiveConfig({ orientationMix })} />
          <Checkbox label="Sym X" checked={(liveConfig.symmetryAxes & 1) !== 0} onChange={(on) => updateLiveConfig({ symmetryAxes: on ? (liveConfig.symmetryAxes | 1) : (liveConfig.symmetryAxes & ~1) })} />
          <Checkbox label="Sym Y" checked={(liveConfig.symmetryAxes & 2) !== 0} onChange={(on) => updateLiveConfig({ symmetryAxes: on ? (liveConfig.symmetryAxes | 2) : (liveConfig.symmetryAxes & ~2) })} />
          <Checkbox label="Sym Z" checked={(liveConfig.symmetryAxes & 4) !== 0} onChange={(on) => updateLiveConfig({ symmetryAxes: on ? (liveConfig.symmetryAxes | 4) : (liveConfig.symmetryAxes & ~4) })} />
        </ControlGroup>

        <ControlGroup title="Camera">
          <Slider label="Yaw" value={controls.cameraYaw} {...renderSliderRanges.cameraYaw} testId="camera-yaw-slider" onChange={(yaw) => setCameraTarget({ yaw })} />
          <Slider label="Pitch" value={controls.cameraPitch} {...renderSliderRanges.cameraPitch} onChange={(pitch) => setCameraTarget({ pitch })} />
          <Slider label="Zoom" value={controls.cameraDistance} {...renderSliderRanges.cameraDistance} testId="camera-zoom-slider" onChange={(distance) => setCameraTarget({ distance })} />
          <Slider label="FOV" value={controls.fov} {...renderSliderRanges.fov} testId="fov-slider" onChange={(fov) => setControls((c) => ({ ...c, fov }))} />
          <Checkbox label="DOF" checked={controls.dofEnabled} testId="dof-enabled-checkbox" onChange={(dofEnabled) => setControls((c) => ({ ...c, dofEnabled }))} />
          <Slider label="Aperture" value={controls.aperture} {...renderSliderRanges.aperture} testId="aperture-slider" onChange={(aperture) => setControls((c) => ({ ...c, aperture }))} />
          <Slider label="Focal Dist" value={focusToFocal(controls.focusDistance)} {...focalDistanceSliderRange} testId="focus-slider" onChange={(focalDistance) => setControls((c) => ({ ...c, focusDistance: clamp(focalToFocus(focalDistance), renderSliderRanges.focusDistance.min, renderSliderRanges.focusDistance.max) }))} />
          <Slider label="DOF Blur" value={controls.dofBlur} {...renderSliderRanges.dofBlur} testId="dof-blur-slider" onChange={(dofBlur) => setControls((c) => ({ ...c, dofBlur }))} />
          <Checkbox label="DOF Debug" checked={controls.dofDebug} onChange={(dofDebug) => setControls((c) => ({ ...c, dofDebug }))} />
          <Checkbox label="Orbit" checked={cameraOrbit} onChange={(value) => { if (demoStateRef.current.active) orbitTouchedRef.current = true; setCameraOrbit(value); }} />
          <Slider label="Orbit Speed" value={cameraOrbitSpeed} {...orbitSpeedSliderRange} testId="orbit-speed-slider" onChange={(value) => { if (demoStateRef.current.active) orbitTouchedRef.current = true; setCameraOrbitSpeed(value); }} />
          <Checkbox label="Demo" checked={demoMode} testId="demo-mode-checkbox" onChange={setDemoMode} />
          <Checkbox label="Instant Demo" checked={instantDemo} testId="instant-demo-checkbox" onChange={setInstantDemo} />
          <Checkbox label="Lock View" checked={viewLocked} testId="view-lock-checkbox" onChange={setViewLocked} />
          <Slider label="Max FPS" value={frameCap} min={0} max={120} step={1} testId="frame-cap-slider" formatValue={(v) => (v <= 0 ? "Off" : String(Math.round(v)))} onChange={(v) => setFrameCap(Math.round(v))} />
        </ControlGroup>

        <ControlGroup title="Track">
          <Checkbox label="Follow" checked={trackingControls.follow} testId="track-follow-checkbox" onChange={(follow) => setTrackingControls((t) => ({ ...t, follow }))} />
          <Checkbox label="Look" checked={trackingControls.look} testId="track-look-checkbox" onChange={(look) => setTrackingControls((t) => ({ ...t, look }))} />
          <Slider label="Follow Speed" value={trackingControls.followSpeed} {...trackingSliderRanges.followSpeed} testId="follow-speed-slider" onChange={(followSpeed) => setTrackingControls((t) => ({ ...t, followSpeed }))} />
          <Slider label="Follow Smooth" value={trackingControls.followSmoothing} {...trackingSliderRanges.followSmoothing} onChange={(followSmoothing) => setTrackingControls((t) => ({ ...t, followSmoothing }))} />
          <Slider label="Look Speed" value={trackingControls.lookSpeed} {...trackingSliderRanges.lookSpeed} testId="look-speed-slider" onChange={(lookSpeed) => setTrackingControls((t) => ({ ...t, lookSpeed }))} />
          <Slider label="Look Smooth" value={trackingControls.lookSmoothing} {...trackingSliderRanges.lookSmoothing} onChange={(lookSmoothing) => setTrackingControls((t) => ({ ...t, lookSmoothing }))} />
          <Slider label="Follow Dist" value={trackingControls.followDistance} {...trackingSliderRanges.followDistance} testId="follow-distance-slider" onChange={(followDistance) => setTrackingControls((t) => ({ ...t, followDistance }))} />
          <Slider label="Follow Height" value={trackingControls.followHeight} {...trackingSliderRanges.followHeight} testId="follow-height-slider" onChange={(followHeight) => setTrackingControls((t) => ({ ...t, followHeight }))} />
          <Slider label="Cohesion" value={trackingControls.cohesion} {...trackingSliderRanges.cohesion} testId="cohesion-slider" onChange={(cohesion) => setTrackingControls((t) => ({ ...t, cohesion }))} />
          <div className="control-row" data-testid="track-status">
            <span>Locked</span>
            <output>{trackState.active ? `${trackState.members} particles` : trackState.miss ?? "none"}</output>
          </div>
          <button
            data-testid="track-release"
            onClick={() => { trackerRef.current.release(); setTrackState({ active: false, members: 0 }); }}
          >
            Release
          </button>
        </ControlGroup>

        <AudioPanel
          status={audioStatus}
          meters={audioMeters}
          panel={audioPanel}
          inputDevices={audioInputDevices}
          activeInput={audioActiveInput}
          error={audioBackendError}
          lastSequence={audioLastFrame?.sequence ?? null}
          onInputChange={(input) => {
            audioPendingInputRef.current = input;
            setAudioPendingInput(input);
            setAudioActiveInput(input);
            setAudioBackendError(null);
            audioSocketControllerRef.current?.setInput(input);
          }}
          onRefreshInputs={() => audioSocketControllerRef.current?.requestDevices()}
          onChange={(next) => {
            audioPanelRef.current = next;
            setAudioPanel(next);
          }}
        />
        <MidiPanel
          status={midiStatus}
          inputs={midiInputs}
          activeInputId={midiActiveInputId}
          learningKey={midiLearningKey}
          onEnable={() => { void ensureMidiAccess(); }}
          onRefresh={() => {
            if (midiAccessRef.current) {
              refreshMidiInputs(midiAccessRef.current);
            } else {
              void ensureMidiAccess();
            }
          }}
          onInputChange={(inputId) => {
            midiActiveInputIdRef.current = inputId;
            setMidiActiveInputId(inputId);
          }}
        />

        <div className="metrics-grid">
          <Metric label="Live Step" value={String(liveDiagnostics?.timestep ?? "...")} />
          <Metric label="Voxels" value={String(liveDiagnostics?.voxelCount ?? liveConfig.width * liveConfig.height * liveConfig.depth)} />
          <Metric label="Render Target" value={liveDiagnostics ? `${liveDiagnostics.renderResolution[0]}x${liveDiagnostics.renderResolution[1]}` : `${controls.rayResolution}p`} />
          <Metric label="Renderer" value={liveDiagnostics ? "Live 3D" : "..."} />
          <Metric label="GPU" value={webgpu.deviceOk ? "OK" : webgpu.checked ? "Fallback" : "..."} />
          <Metric label="Flow" value={liveDiagnostics ? liveDiagnostics.fieldStats.flowSum.toFixed(2) : "..."} />
          <Metric label="Frame ms" value={liveDiagnostics ? liveDiagnostics.frameTimeMs.toFixed(1) : "..."} />
          <Metric label="Tap" value={String(liveDiagnostics?.depositTaps ?? Math.pow(liveConfig.depositTapRadius * 2 + 1, 3))} />
          <Metric label="Voxel" value={(2 / liveConfig.width).toFixed(4)} />
          <Metric label="Sigma" value={liveConfig.sigma.toFixed(4)} />
          <Metric label="Trail Scale" value={liveDiagnostics ? liveDiagnostics.depositScale.toFixed(4) : "..."} />
          <Metric label="GPU MB" value={liveDiagnostics ? ((liveDiagnostics.particleBufferBytes * 2 + liveDiagnostics.fieldBufferBytes * 3) / 1048576).toFixed(0) : "..."} />
        </div>
        <details className="diagnostics-panel" open data-testid="status">
          <summary>Diagnostics</summary>
          <div className="status-list">
            <span>{webgpu.checked ? (webgpu.deviceOk ? "WebGPU verified" : "WebGPU fallback verified") : "Checking GPU"}</span>
            <span>{compute3d ? (compute3d.passed ? "3D compute verified" : "3D compute pending") : "3D compute checking"}</span>
            <span>{liveDiagnostics?.particleCount ?? liveConfig.particleCount} live particles</span>
            <span>{liveDiagnostics?.renderMode ?? "particle renderer pending"}</span>
            <span>{liveDiagnostics?.depositMode ?? "deposit pending"}</span>
            <span>frame {frame}</span>
          </div>
        </details>
        <details className="preset-json">
          <summary>Preset JSON</summary>
          <textarea name="current-preset-json" value={preset} readOnly aria-label="Current preset JSON" />
        </details>
      </aside>
      </main>
    </SliderModulationContext.Provider>
  );
}

function AudioPanel(props: {
  status: AudioReactiveSocketStatus;
  meters: Record<AudioBucket, number>;
  panel: AudioPanelState;
  inputDevices: AudioInputDeviceInfo[];
  activeInput: string | null;
  error: string | null;
  lastSequence: number | null;
  onInputChange: (input: string) => void;
  onRefreshInputs: () => void;
  onChange: (panel: AudioPanelState) => void;
}) {
  const [inputMenuOpen, setInputMenuOpen] = useState(false);
  const inputSelectDisabled = !props.status.connected || props.inputDevices.length === 0;
  const activeInputLabel = props.activeInput
    ? audioDeviceLabel(props.inputDevices.find((device) => device.name === props.activeInput) ?? {
      id: props.activeInput,
      name: props.activeInput,
      isDefault: false
    })
    : props.status.connected ? "No inputs" : "Disconnected";
  const updateBucket = (bucket: AudioBucket, patch: Partial<AudioPanelState["buckets"][AudioBucket]>) => {
    props.onChange({
      ...props.panel,
      buckets: {
        ...props.panel.buckets,
        [bucket]: { ...props.panel.buckets[bucket], ...patch }
      }
    });
  };
  return (
    <section className="audio-panel" data-testid="audio-panel">
      <div className="audio-panel-title">
        <span>Audio</span>
        <strong>{props.status.connected ? "on" : "off"}</strong>
      </div>
      <div className="audio-input-row">
        <span>Input</span>
        <div className="audio-input-picker">
          <button
            className="audio-input-button"
            data-testid="audio-input-select"
            type="button"
            aria-haspopup="listbox"
            aria-expanded={inputMenuOpen}
            disabled={inputSelectDisabled}
            onClick={() => {
              props.onRefreshInputs();
              setInputMenuOpen((open) => !open);
            }}
            onKeyDown={(event) => {
              if (event.key === "Escape") setInputMenuOpen(false);
            }}
          >
            <span>{activeInputLabel}</span>
          </button>
          {inputMenuOpen && !inputSelectDisabled ? (
            <div className="audio-input-menu" data-testid="audio-input-menu" role="listbox">
              {props.inputDevices.map((device) => (
                <button
                  key={device.id}
                  className={device.name === props.activeInput ? "active" : ""}
                  data-testid={`audio-input-option-${inputName(device.name)}`}
                  type="button"
                  role="option"
                  aria-selected={device.name === props.activeInput}
                  onClick={() => {
                    props.onInputChange(device.name);
                    setInputMenuOpen(false);
                  }}
                >
                  {audioDeviceLabel(device)}
                </button>
              ))}
            </div>
          ) : null}
        </div>
      </div>
      <input
        type="hidden"
        name="audio-input"
        value={props.activeInput ?? ""}
      />
      {props.error ? <div className="audio-input-error" data-testid="audio-input-error">{props.error}</div> : null}
      <label className="audio-enable-row">
        <span>Enable</span>
        <input
          data-testid="audio-enabled"
          type="checkbox"
          checked={props.panel.enabled}
          onChange={(event) => props.onChange({ ...props.panel, enabled: event.currentTarget.checked })}
        />
      </label>
      <div className="audio-meter-grid">
        {audioBuckets.map((bucket) => (
          <div key={bucket} className="audio-meter" data-testid={`audio-meter-${bucket}`}>
            <span>{bucket}</span>
            <div><i style={{ width: `${Math.round(props.meters[bucket] * 100)}%` }} /></div>
            <strong>{props.meters[bucket].toFixed(2)}</strong>
          </div>
        ))}
      </div>
      <div className="audio-bucket-controls">
        <div className="audio-bucket-row audio-bucket-header" aria-hidden="true">
          <span />
          <span>gain</span>
          <span>exp</span>
          <span>attack ms</span>
          <span>decay ms</span>
        </div>
        {audioBuckets.map((bucket) => (
          <div key={bucket} className="audio-bucket-row">
            <span>{bucket}</span>
            <DraftNumberInput
              data-testid={`audio-${bucket}-gain`}
              aria-label={`${bucket} gain`}
              name={inputName(`${bucket} gain`)}
              min={0}
              max={8}
              step={0.05}
              value={props.panel.buckets[bucket].gain}
              onChange={(value) => updateBucket(bucket, { gain: clamp(value, 0, 8) })}
            />
            <DraftNumberInput
              data-testid={`audio-${bucket}-exp`}
              aria-label={`${bucket} exponent`}
              name={inputName(`${bucket} exponent`)}
              min={0.1}
              max={8}
              step={0.05}
              value={props.panel.buckets[bucket].exponent}
              onChange={(value) => updateBucket(bucket, { exponent: clamp(value, 0.1, 8) })}
            />
            <DraftNumberInput
              data-testid={`audio-${bucket}-attack`}
              aria-label={`${bucket} attack`}
              name={inputName(`${bucket} attack`)}
              min={1}
              max={2000}
              step={1}
              value={props.panel.buckets[bucket].attackMs}
              onChange={(value) => updateBucket(bucket, { attackMs: clamp(value, 1, 2000) })}
            />
            <DraftNumberInput
              data-testid={`audio-${bucket}-decay`}
              aria-label={`${bucket} decay`}
              name={inputName(`${bucket} decay`)}
              min={1}
              max={2000}
              step={1}
              value={props.panel.buckets[bucket].decayMs}
              onChange={(value) => updateBucket(bucket, { decayMs: clamp(value, 1, 2000) })}
            />
          </div>
        ))}
      </div>
      <div className="audio-frame-readout">{props.lastSequence === null ? "..." : props.lastSequence}</div>
    </section>
  );
}

function MidiPanel(props: {
  status: MidiPanelStatus;
  inputs: MidiInputInfo[];
  activeInputId: string;
  learningKey: string | null;
  onEnable: () => void;
  onRefresh: () => void;
  onInputChange: (inputId: string) => void;
}) {
  const [inputMenuOpen, setInputMenuOpen] = useState(false);
  const inputSelectDisabled = !props.status.enabled || props.inputs.length === 0;
  const activeInput = props.inputs.find((input) => input.id === props.activeInputId);
  const activeInputLabel = props.activeInputId === midiAllInputsId
    ? "All inputs"
    : activeInput
      ? midiInputLabel(activeInput)
      : props.activeInputId;
  const statusLabel = props.learningKey ? "learn" : props.status.enabled ? "on" : "off";
  return (
    <section className="midi-panel" data-testid="midi-panel">
      <div className="midi-panel-title">
        <span>MIDI</span>
        <strong>{statusLabel}</strong>
      </div>
      <div className="midi-action-row">
        <button data-testid="midi-enable" type="button" disabled={!props.status.supported} onClick={props.onEnable}>Enable</button>
        <button data-testid="midi-refresh" type="button" disabled={!props.status.supported} onClick={props.onRefresh}>Refresh</button>
      </div>
      <div className="midi-input-row">
        <span>Input</span>
        <div className="midi-input-picker">
          <button
            className="midi-input-button"
            data-testid="midi-input-select"
            type="button"
            aria-haspopup="listbox"
            aria-expanded={inputMenuOpen}
            disabled={inputSelectDisabled}
            onClick={() => setInputMenuOpen((open) => !open)}
            onKeyDown={(event) => {
              if (event.key === "Escape") setInputMenuOpen(false);
            }}
          >
            <span>{props.status.enabled ? activeInputLabel : "Disabled"}</span>
          </button>
          {inputMenuOpen && !inputSelectDisabled ? (
            <div className="midi-input-menu" data-testid="midi-input-menu" role="listbox">
              <button
                className={props.activeInputId === midiAllInputsId ? "active" : ""}
                data-testid="midi-input-option-all"
                type="button"
                role="option"
                aria-selected={props.activeInputId === midiAllInputsId}
                onClick={() => {
                  props.onInputChange(midiAllInputsId);
                  setInputMenuOpen(false);
                }}
              >
                All inputs
              </button>
              {props.inputs.map((input) => (
                <button
                  key={input.id}
                  className={input.id === props.activeInputId ? "active" : ""}
                  data-testid={`midi-input-option-${inputName(input.id)}`}
                  type="button"
                  role="option"
                  aria-selected={input.id === props.activeInputId}
                  onClick={() => {
                    props.onInputChange(input.id);
                    setInputMenuOpen(false);
                  }}
                >
                  {midiInputLabel(input)}
                </button>
              ))}
            </div>
          ) : null}
        </div>
      </div>
      <input type="hidden" name="midi-input" value={props.activeInputId} />
      {props.status.error ? <div className="midi-input-error" data-testid="midi-input-error">{props.status.error}</div> : null}
      <div className="midi-frame-readout" data-testid="midi-last-message">
        {props.status.lastMessage ?? String(props.status.messageCount)}
      </div>
    </section>
  );
}

function audioDeviceLabel(device: AudioInputDeviceInfo): string {
  const detail = [
    device.channels ? `${device.channels}ch` : "",
    device.sampleRate ? `${Math.round(device.sampleRate / 1000)}k` : "",
    device.isDefault ? "default" : ""
  ].filter(Boolean).join(" ");
  return detail ? `${device.name} ${detail}` : device.name;
}

function midiInputLabel(input: MidiInputInfo): string {
  const detail = [
    input.state === "connected" ? "" : input.state,
    input.connection === "open" ? "" : input.connection
  ].filter(Boolean).join(" ");
  return detail ? `${input.name} ${detail}` : input.name;
}

function midiInputList(access: MidiAccessLike): MidiInputInfo[] {
  return Array.from(access.inputs.values()).map((input) => ({
    id: input.id,
    name: input.name || input.id,
    state: input.state ?? "connected",
    connection: input.connection ?? "open"
  })).sort((a, b) => a.name.localeCompare(b.name));
}

function SliderImpl(props: { label: string; value: number; min: number; max: number; step: number; curve?: number; testId?: string; formatValue?: (value: number) => string; onChange: (value: number) => void }) {
  const precision = stepPrecision(props.step);
  const curve = props.curve ?? 1;
  const modulation = useContext(SliderModulationContext);
  const modulationKey = props.testId ?? inputName(props.label);
  const modulationConfig = modulation ? getSliderModulation(modulation.settings, modulationKey, props.min, props.max) : null;
  const modulationExpanded = modulation?.expandedKey === modulationKey;
  // Audio activity and MIDI mappings may target a range that ignores the slider's pre-baked
  // min/max. Expand the effective range to cover those mappings so the slider track and the
  // clamping applied to audio/MIDI-driven values let the value reach the full mapped range.
  const { min: effectiveMin, max: effectiveMax } = effectiveSliderRange(
    props.min,
    props.max,
    props.value,
    modulationConfig
  );
  useEffect(() => {
    if (!modulation) return;
    modulation.register(modulationKey, {
      value: props.value,
      min: effectiveMin,
      max: effectiveMax,
      step: props.step,
      onChange: props.onChange
    });
    return () => modulation.register(modulationKey, null);
  }, [effectiveMax, effectiveMin, modulation, modulationKey, props.onChange, props.step, props.value]);

  const label = (
    <span
      data-testid={`${modulationKey}-modulation-toggle`}
      title={controlHint(props.label)}
      tabIndex={0}
      onClick={(event) => {
        event.preventDefault();
        event.stopPropagation();
        modulation?.setExpandedKey(modulationExpanded ? null : modulationKey);
      }}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          modulation?.setExpandedKey(modulationExpanded ? null : modulationKey);
        }
      }}
    >
      {props.label}
    </span>
  );

  const body = modulation && modulationConfig && modulationExpanded ? (
    <SliderModulationBody
      modulationKey={modulationKey}
      config={modulationConfig}
      step={props.step}
      midiStatus={modulation.midiStatus}
      midiLearning={modulation.midiLearningKey === modulationKey}
      onChange={(config) => modulation.updateConfig(modulationKey, config)}
      onMidiLearn={() => modulation.startMidiLearn(modulationKey)}
      onMidiCancel={modulation.cancelMidiLearn}
    />
  ) : null;

  const shellClassName = `slider-shell${modulationConfig && (hasEnabledSliderModulation(modulationConfig) || hasEnabledMidiMapping(modulationConfig.midi)) ? " is-modulated" : ""}`;
  // Non-linear sliders drive a normalized 0..1 position through a power curve, giving fine
  // resolution at the low end where the useful values live (cutoff, speed thresholds).
  if (curve !== 1) {
    const pos = sliderValueToPos(props.value, effectiveMin, effectiveMax, curve);
    const apply = (raw: number) => props.onChange(sliderPosToValue(raw, effectiveMin, effectiveMax, curve));
    return (
      <div className={shellClassName}>
        <label className="control-row">
          {label}
          <input
            data-testid={props.testId}
            name={inputName(props.label)}
            type="range"
            value={pos}
            min={0}
            max={1}
            step={0.002}
            onInput={(event) => apply(Number(event.currentTarget.value))}
            onChange={(event) => apply(Number(event.currentTarget.value))}
          />
          <output>{props.formatValue ? props.formatValue(props.value) : props.value.toFixed(precision)}</output>
        </label>
        {body}
      </div>
    );
  }
  return (
    <div className={shellClassName}>
      <label className="control-row">
        {label}
        <input
          data-testid={props.testId}
          name={inputName(props.label)}
          type="range"
          value={props.value}
          min={effectiveMin}
          max={effectiveMax}
          step={props.step}
          onInput={(event) => props.onChange(Number(event.currentTarget.value))}
          onChange={(event) => props.onChange(Number(event.currentTarget.value))}
        />
        <output>{props.formatValue ? props.formatValue(props.value) : props.value.toFixed(precision)}</output>
      </label>
      {body}
    </div>
  );
}

function SliderModulationBody(props: {
  modulationKey: string;
  config: SliderModulationConfig;
  step: number;
  midiStatus: MidiPanelStatus;
  midiLearning: boolean;
  onChange: (config: SliderModulationConfig) => void;
  onMidiLearn: () => void;
  onMidiCancel: () => void;
}) {
  const precision = Math.max(0, Math.min(6, stepPrecision(props.step)));
  const setBucket = (bucket: AudioBucket, patch: Partial<SliderModulationConfig["buckets"][AudioBucket]>) => {
    props.onChange({
      ...props.config,
      buckets: {
        ...props.config.buckets,
        [bucket]: { ...props.config.buckets[bucket], ...patch }
      }
    });
  };
  const updateRange = (patch: Partial<Pick<SliderModulationConfig, "min" | "max">>) => {
    props.onChange({
      ...props.config,
      ...patch
    });
  };
  const setMidiEnabled = (enabled: boolean) => {
    props.onChange({
      ...props.config,
      midi: {
        enabled,
        min: props.config.midi?.min ?? props.config.min,
        max: props.config.midi?.max ?? props.config.max,
        control: props.config.midi?.control ?? null
      }
    });
  };
  const updateMidiRange = (patch: Partial<Pick<NonNullable<SliderModulationConfig["midi"]>, "min" | "max">>) => {
    props.onChange({
      ...props.config,
      midi: {
        enabled: props.config.midi?.enabled ?? false,
        min: patch.min ?? props.config.midi?.min ?? props.config.min,
        max: patch.max ?? props.config.midi?.max ?? props.config.max,
        control: props.config.midi?.control ?? null
      }
    });
  };
  const clearMidi = () => {
    if (props.midiLearning) props.onMidiCancel();
    props.onChange({
      ...props.config,
      midi: {
        enabled: false,
        min: props.config.midi?.min ?? props.config.min,
        max: props.config.midi?.max ?? props.config.max,
        control: null
      }
    });
  };
  const midiMapped = hasEnabledMidiMapping(props.config.midi);
  const midiMin = props.config.midi?.min ?? props.config.min;
  const midiMax = props.config.midi?.max ?? props.config.max;
  return (
    <div className="slider-modulation-body" data-testid={`${props.modulationKey}-modulation-body`}>
      <div className="slider-modulation-row">
        {audioBuckets.map((bucket) => (
          <input
            key={bucket}
            data-testid={`${props.modulationKey}-${bucket}-mod`}
            aria-label={`${bucket} modulation`}
            type="checkbox"
            checked={props.config.buckets[bucket].enabled}
            onChange={(event) => setBucket(bucket, { enabled: event.currentTarget.checked })}
          />
        ))}
        <DraftNumberInput
          data-testid={`${props.modulationKey}-mod-min`}
          aria-label="modulation min"
          step={props.step}
          precision={precision}
          value={props.config.min}
          onChange={(value) => updateRange({ min: value })}
        />
        <DraftNumberInput
          data-testid={`${props.modulationKey}-mod-max`}
          aria-label="modulation max"
          step={props.step}
          precision={precision}
          value={props.config.max}
          onChange={(value) => updateRange({ max: value })}
        />
      </div>
      <div className="slider-modulation-row gain-row">
        {audioBuckets.map((bucket) => (
          <DraftNumberInput
            key={bucket}
            data-testid={`${props.modulationKey}-${bucket}-gain`}
            aria-label={`${bucket} gain multiplier`}
            min={0}
            max={16}
            step={0.05}
            value={props.config.buckets[bucket].gain}
            onChange={(value) => setBucket(bucket, { gain: clamp(value, 0, 16) })}
          />
        ))}
      </div>
      <div className="slider-midi-row">
        <span>MIDI</span>
        <button
          data-testid={`${props.modulationKey}-midi-learn`}
          type="button"
          disabled={!props.midiStatus.supported}
          onClick={props.midiLearning ? props.onMidiCancel : props.onMidiLearn}
        >
          {props.midiLearning ? "Cancel" : midiMapped ? "Relearn" : "Learn"}
        </button>
        <button
          data-testid={`${props.modulationKey}-midi-clear`}
          type="button"
          disabled={!props.config.midi?.control}
          onClick={clearMidi}
        >
          Clear
        </button>
        <label>
          <input
            data-testid={`${props.modulationKey}-midi-enabled`}
            type="checkbox"
            checked={props.config.midi?.enabled === true}
            disabled={!props.config.midi?.control}
            onChange={(event) => setMidiEnabled(event.currentTarget.checked)}
          />
          <span>On</span>
        </label>
        <DraftNumberInput
          className="midi-range-min"
          data-testid={`${props.modulationKey}-midi-min`}
          aria-label="MIDI min"
          step={props.step}
          precision={precision}
          value={midiMin}
          onChange={(value) => updateMidiRange({ min: value })}
        />
        <DraftNumberInput
          className="midi-range-max"
          data-testid={`${props.modulationKey}-midi-max`}
          aria-label="MIDI max"
          step={props.step}
          precision={precision}
          value={midiMax}
          onChange={(value) => updateMidiRange({ max: value })}
        />
        <output data-testid={`${props.modulationKey}-midi-mapping`}>
          {props.midiLearning ? "Listening" : midiControlLabel(props.config.midi?.control)}
        </output>
      </div>
    </div>
  );
}

function ControlGroup(props: { title: string; children: ReactNode }) {
  return (
    <section className="control-group">
      <div className="control-group-title">{props.title}</div>
      {props.children}
    </section>
  );
}

// Like ControlGroup but collapsible via native <details> for less frequently used controls.
function CollapsibleGroup(props: { title: string; testId?: string; defaultOpen?: boolean; children: ReactNode }) {
  return (
    <details className="control-group control-group-collapsible" data-testid={props.testId} open={props.defaultOpen}>
      <summary className="control-group-title">{props.title}</summary>
      {props.children}
    </details>
  );
}

function ParticleBlendSelect(props: { value: RenderControls["particleBlendMode"]; onChange: (value: RenderControls["particleBlendMode"]) => void }) {
  return (
    <label className="control-row">
      <span title={controlHint("Blend")}>Blend</span>
      <select data-testid="particle-blend-select" name="particle-blend" value={props.value} onChange={(event) => props.onChange(event.target.value as RenderControls["particleBlendMode"])}>
        <option value="additive">Additive</option>
        <option value="alpha">Alpha</option>
        <option value="opaque">Opaque</option>
      </select>
      <output />
    </label>
  );
}

function ParticleColorModeSelect(props: { value: RenderControls["particleColorMode"]; onChange: (value: RenderControls["particleColorMode"]) => void }) {
  return (
    <label className="control-row">
      <span title={controlHint("Particle Color Mode")}>Particle Color Mode</span>
      <select data-testid="particle-color-mode-select" name="particle-color-mode" value={props.value} onChange={(event) => props.onChange(event.target.value as RenderControls["particleColorMode"])}>
        {particleColorSweep.map(([value, label]) => (
          <option key={value} value={value}>{label}</option>
        ))}
        <optgroup label="Gradient (standalone)">
          <option value="gradient-inferno">Gradient Inferno</option>
          <option value="gradient-magma">Gradient Magma</option>
          <option value="gradient-viridis">Gradient Viridis</option>
          <option value="gradient-turbo">Gradient Turbo</option>
          <option value="gradient-rainbow">Gradient Rainbow</option>
          <option value="gradient-spectral">Gradient Spectral</option>
          <option value="gradient-plasma">Gradient Plasma</option>
          <option value="gradient-cosmic">Gradient Cosmic</option>
          <option value="gradient-ice">Gradient Ice</option>
          <option value="gradient-ember">Gradient Ember</option>
        </optgroup>
      </select>
      <output />
    </label>
  );
}

function ColorFieldImpl(props: { label: string; value: string; onChange: (value: string) => void }) {
  return (
    <label className="control-row color-row">
      <span title={controlHint(props.label)}>{props.label}</span>
      <input
        name={inputName(props.label)}
        type="color"
        value={props.value}
        onInput={(event) => props.onChange(event.currentTarget.value)}
        onChange={(event) => props.onChange(event.currentTarget.value)}
      />
      <output>{props.value}</output>
    </label>
  );
}

function ModeSelect(props: { label: string; value: string; options: [string, string][]; onChange: (value: string) => void }) {
  return (
    <label className="control-row">
      <span title={controlHint(props.label)}>{props.label}</span>
      <select name={inputName(props.label)} value={props.value} onChange={(event) => props.onChange(event.target.value)}>
        {props.options.map(([value, label]) => (
          <option key={value} value={value}>{label}</option>
        ))}
      </select>
      <output />
    </label>
  );
}

// Stepped slider that walks through the same options as a ModeSelect dropdown, one option per
// integer step. Lets modes be driven by audio/MIDI modulation (which plain <select>s can't do).
function ModeSliderImpl(props: { label: string; value: string; options: [string, string][]; testId?: string; onChange: (value: string) => void }) {
  const clampIndex = (raw: number) => Math.min(props.options.length - 1, Math.max(0, Math.round(raw)));
  const currentIndex = props.options.findIndex(([value]) => value === props.value);
  return (
    <Slider
      label={props.label}
      value={currentIndex < 0 ? 0 : currentIndex}
      min={0}
      max={Math.max(0, props.options.length - 1)}
      step={1}
      testId={props.testId}
      formatValue={(value) => props.options[clampIndex(value)]?.[1] ?? ""}
      onChange={(value) => props.onChange(props.options[clampIndex(value)][0])}
    />
  );
}

function Checkbox(props: { label: string; checked: boolean; testId?: string; onChange: (checked: boolean) => void }) {
  return (
    <label className="control-row checkbox-row">
      <span title={controlHint(props.label)}>{props.label}</span>
      <input data-testid={props.testId} name={inputName(props.label)} type="checkbox" checked={props.checked} onChange={(event) => props.onChange(event.target.checked)} />
      <output>{props.checked ? "On" : "Off"}</output>
    </label>
  );
}

function PresetSelect(props: { value: string; onChange: (value: string) => void }) {
  return (
    <label className="control-row">
      <span title={controlHint("Preset")}>Preset</span>
      <select data-testid="preset-select" name="preset" value={props.value} onChange={(event) => props.onChange(event.target.value)}>
        {livePresets.map((preset) => (
          <option key={preset.id} value={preset.id}>{preset.name}</option>
        ))}
      </select>
      <output />
    </label>
  );
}

function SavedSettingsSelect(props: {
  value: string;
  settings: SavedSettingsPreset[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="control-row">
      <span title={controlHint("Saved")}>Saved</span>
      <select data-testid="saved-settings-select" name="saved-settings" value={props.value} onChange={(event) => props.onChange(event.target.value)}>
        <option value="">Current</option>
        {props.settings.map((settings) => (
          <option key={settings.id} value={settings.id}>{settings.name}</option>
        ))}
      </select>
      <output>{props.settings.length}</output>
    </label>
  );
}

function NumberField(props: {
  label: string;
  value: number;
  readOnly?: boolean;
  testId: string;
  min?: number;
  max?: number;
  step?: number;
  onChange?: (value: number) => void;
}) {
  const [draft, setDraft] = useState(String(props.value));
  const [editing, setEditing] = useState(false);

  useEffect(() => {
    if (!editing) {
      setDraft(String(props.value));
    }
  }, [editing, props.value]);

  const commit = () => {
    setEditing(false);
    const next = Number(draft);
    if (Number.isFinite(next)) {
      props.onChange?.(next);
    } else {
      setDraft(String(props.value));
    }
  };
  const updateDraft = (value: string) => {
    setDraft(value);
    const next = Number(value);
    if (!props.readOnly && Number.isFinite(next)) {
      props.onChange?.(next);
    }
  };

  return (
    <label className="control-row">
      <span title={controlHint(props.label)}>{props.label}</span>
      <input
        data-testid={props.testId}
        name={inputName(props.label)}
        type="number"
        value={draft}
        readOnly={props.readOnly}
        min={props.min}
        max={props.max}
        step={props.step}
        onFocus={() => setEditing(true)}
        onChange={(event) => updateDraft(event.target.value)}
        onBlur={commit}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.currentTarget.blur();
          } else if (event.key === "Escape") {
            setEditing(false);
            setDraft(String(props.value));
            event.currentTarget.blur();
          }
        }}
      />
    </label>
  );
}

function DraftNumberInput(props: {
  value: number;
  min?: number;
  max?: number;
  step?: number;
  precision?: number;
  name?: string;
  className?: string;
  "data-testid"?: string;
  "aria-label"?: string;
  onChange: (value: number) => void;
}) {
  const [draft, setDraft] = useState(formatDraftNumber(props.value, props.precision));
  const [editing, setEditing] = useState(false);

  useEffect(() => {
    if (!editing) setDraft(formatDraftNumber(props.value, props.precision));
  }, [editing, props.precision, props.value]);

  const updateDraft = (value: string) => {
    setDraft(value);
    if (isPartialNumberDraft(value)) return;
    const next = Number(value);
    if (Number.isFinite(next)) props.onChange(next);
  };
  const commit = () => {
    setEditing(false);
    if (isPartialNumberDraft(draft)) {
      setDraft(formatDraftNumber(props.value, props.precision));
      return;
    }
    const next = Number(draft);
    if (Number.isFinite(next)) {
      props.onChange(next);
      setDraft(formatDraftNumber(next, props.precision));
    } else {
      setDraft(formatDraftNumber(props.value, props.precision));
    }
  };

  return (
    <input
      className={props.className}
      data-testid={props["data-testid"]}
      aria-label={props["aria-label"]}
      name={props.name}
      type="number"
      min={props.min}
      max={props.max}
      step={props.step}
      value={draft}
      onFocus={() => setEditing(true)}
      onChange={(event) => updateDraft(event.currentTarget.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.currentTarget.blur();
        } else if (event.key === "Escape") {
          setEditing(false);
          setDraft(formatDraftNumber(props.value, props.precision));
          event.currentTarget.blur();
        }
      }}
    />
  );
}

function MetricImpl(props: { label: string; value: string }) {
  return (
    <div>
      <span title={controlHint(props.label)}>{props.label}</span>
      <strong>{props.value}</strong>
    </div>
  );
}

// Memoized control widgets: a value-based comparator so a diagnostics/meter-driven App
// re-render skips every control whose value is unchanged (no panel reconcile while the menu
// is open). onChange identity is intentionally ignored — the closures use functional
// setControls/updateLiveConfig so a "stale" closure still applies the right update.
const Slider = memo(
  SliderImpl,
  (a, b) =>
    a.value === b.value &&
    a.min === b.min &&
    a.max === b.max &&
    a.step === b.step &&
    a.curve === b.curve &&
    a.label === b.label
);
const ColorField = memo(ColorFieldImpl, (a, b) => a.value === b.value && a.label === b.label);
const ModeSlider = memo(ModeSliderImpl, (a, b) => a.value === b.value && a.label === b.label);
const Metric = memo(MetricImpl, (a, b) => a.value === b.value && a.label === b.label);

function inputName(label: string): string {
  return label.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

function formatDraftNumber(value: number, precision?: number): string {
  if (!Number.isFinite(value)) return "";
  return precision === undefined ? String(value) : value.toFixed(precision);
}

function isPartialNumberDraft(value: string): boolean {
  const trimmed = value.trim();
  return trimmed === "" || trimmed === "-" || trimmed === "+" || trimmed === "." || trimmed === "-." || trimmed === "+.";
}

type FullscreenTarget = HTMLElement & {
  webkitRequestFullscreen?: () => Promise<void> | void;
};

type FullscreenDocument = Document & {
  webkitFullscreenElement?: Element | null;
  webkitExitFullscreen?: () => Promise<void> | void;
};

function getFullscreenElement(): Element | null {
  const fullscreenDocument = document as FullscreenDocument;
  return fullscreenDocument.fullscreenElement ?? fullscreenDocument.webkitFullscreenElement ?? null;
}

function getRequestFullscreen(target: HTMLElement | null): (() => Promise<void> | void) | undefined {
  const fullscreenTarget = target as FullscreenTarget | null;
  if (!fullscreenTarget) return undefined;
  const requestFullscreen = fullscreenTarget.requestFullscreen ?? fullscreenTarget.webkitRequestFullscreen;
  return requestFullscreen ? () => requestFullscreen.call(fullscreenTarget) : undefined;
}

function getExitFullscreen(): (() => Promise<void> | void) | undefined {
  const fullscreenDocument = document as FullscreenDocument;
  const exitFullscreen = fullscreenDocument.exitFullscreen ?? fullscreenDocument.webkitExitFullscreen;
  return exitFullscreen ? () => exitFullscreen.call(fullscreenDocument) : undefined;
}

function isTextEntryTarget(target: EventTarget | null): boolean {
  return target instanceof HTMLElement &&
    (target.isContentEditable || target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.tagName === "SELECT");
}

// Drives cameraRef directly toward the tracked centroid. Look = aim (pan), Follow = distance.
// Sets target* equal to value* so the downstream damp in smoothedRenderControls is a no-op.
function applyTrackingToCamera(
  camera: OrbitCamera,
  tracker: SwimmerTracker,
  tracking: TrackingControls,
  springs: { panX: Spring; panY: Spring; distance: Spring },
  aspect: number,
  dt: number
): void {
  const state = tracker.state;
  if (!state.active || (!tracking.follow && !tracking.look)) return;
  const anchor: Vec3 = [state.centroid[0], state.centroid[1] + tracking.followHeight, state.centroid[2]];
  const camMath: CameraMath3d = { yaw: camera.yaw, pitch: camera.pitch, distance: camera.distance, aspect };
  if (tracking.look) {
    const view = worldToCamera(anchor, camMath); // centered when pan = -view
    springs.panX.value = camera.panX;
    springs.panY.value = camera.panY;
    springStep(springs.panX, clamp(-view[0], minCameraPan, maxCameraPan), 1 / tracking.lookSpeed, tracking.lookSmoothing, dt);
    springStep(springs.panY, clamp(-view[1], minCameraPan, maxCameraPan), 1 / tracking.lookSpeed, tracking.lookSmoothing, dt);
    camera.panX = springs.panX.value;
    camera.panY = springs.panY.value;
    camera.targetPanX = camera.panX;
    camera.targetPanY = camera.panY;
  }
  if (tracking.follow) {
    springs.distance.value = camera.distance;
    springStep(springs.distance, clampCameraDistance(tracking.followDistance), 1 / tracking.followSpeed, tracking.followSmoothing, dt);
    camera.distance = springs.distance.value;
    camera.targetDistance = camera.distance;
  }
}

function smoothedRenderControls(controls: RenderControls, camera: OrbitCamera): RenderControls {
  camera.yaw = dampAngle(camera.yaw, camera.targetYaw, 0.26);
  camera.pitch = damp(camera.pitch, camera.targetPitch, 0.26);
  camera.distance = damp(camera.distance, camera.targetDistance, 0.22);
  camera.panX = damp(camera.panX, camera.targetPanX, 0.28);
  camera.panY = damp(camera.panY, camera.targetPanY, 0.28);
  camera.fov = damp(camera.fov, camera.targetFov, 0.18);
  return {
    ...controls,
    fov: camera.fov,
    cameraYaw: camera.yaw,
    cameraPitch: camera.pitch,
    cameraDistance: camera.distance,
    cameraPanX: camera.panX,
    cameraPanY: camera.panY
  };
}

function publishUiDiagnostics(callback: () => void, lastPublished: { current: number }): void {
  const now = performance.now();
  // ~3 Hz HUD refresh: enough for live readouts, but it roughly halves the per-second control-panel
  // React reconciles vs 180ms when the panel is open (and it's already gated off when hidden).
  if (now - lastPublished.current < 300) return;
  lastPublished.current = now;
  callback();
}

function publishFps(stats: { lastTime: number; frames: number; value: number }, callback: (fps: number) => void): void {
  const now = performance.now();
  if (stats.lastTime === 0) {
    stats.lastTime = now;
    return;
  }
  stats.frames += 1;
  const elapsed = now - stats.lastTime;
  if (elapsed < 500) return;
  stats.value = stats.frames * 1000 / elapsed;
  stats.frames = 0;
  stats.lastTime = now;
  callback(stats.value);
}

function liveVertexCount(diagnostics: LiveGpu3dDiagnostics | null, config: LiveGpu3dConfig): number {
  const particleCount = diagnostics?.particleCount ?? config.particleCount;
  if (diagnostics?.effectiveRenderLayer === "debug-voxels") {
    return diagnostics.voxelCount * 6;
  }
  return particleCount * 4;
}

function formatCount(value: number): string {
  if (!Number.isFinite(value)) return "...";
  return Math.max(0, Math.round(value)).toLocaleString("en-US");
}

function liveWebGpuConformance(live: LiveGpu3dDiagnostics | null): boolean {
  if (!live) return false;
  const renderModeOk =
    live.renderMode === "volume-raymarch+particle-splats" ||
    live.renderMode === "particle-splats" ||
    live.renderMode === "volume-density-raymarch";
  const fieldStatsOk = live.renderMode === "particle-splats" || live.fieldStats.nonzeroVoxels > 0;
  return live.renderer === "webgpu-live-fluoddity-3d" &&
    renderModeOk &&
    live.depositMode === "particle-atomic-fixed-point" &&
    live.timestep > 0 &&
    fieldStatsOk &&
    live.ruleEvidence.nonzero &&
    live.ruleEvidence.cohortDependent;
}

function buildSavedSettingsPreset(
  name: string,
  presetId: string,
  displayMode: DisplayMode,
  controls: RenderControls,
  liveConfig: LiveGpu3dConfig,
  ui: SavedUiState,
  audio: SavedAudioState,
  existing?: SavedSettingsPreset
): SavedSettingsPreset {
  const now = new Date().toISOString();
  const normalizedControls = normalizeRenderControlsForDisplayMode(
    displayMode,
    renderControlsWithModulationRangeOverrides(sanitizeRenderControls(controls), controls, audio.sliders)
  );
  return {
    version: 2,
    id: existing?.id ?? makeSettingsId(),
    name,
    createdAt: existing?.createdAt ?? now,
    updatedAt: now,
    presetId,
    displayMode,
    controls: normalizedControls,
    liveConfig: sanitizeLiveConfig(liveConfig),
    ui: sanitizeSavedUiState(ui),
    audio: sanitizeSavedAudioState(audio)
  };
}

function buildSavedAudioState(value: { panel: AudioPanelState; sliders: SliderModulationSettings }): SavedAudioState {
  return sanitizeSavedAudioState(value);
}

function renderControlsWithModulationRangeOverrides(
  controls: RenderControls,
  source: RenderControls,
  settings: SliderModulationSettings
): RenderControls {
  let next = controls;
  for (const [sliderKey, controlKey] of Object.entries(renderControlSliderTargets)) {
    const config = settings[sliderKey];
    if (!config || (!hasEnabledSliderModulation(config) && !hasEnabledMidiMapping(config.midi))) continue;
    const currentValue = controls[controlKey];
    if (typeof currentValue !== "number") continue;
    const rawValue = finiteNumber(source[controlKey], currentValue);
    if (rawValue === currentValue) continue;
    if (next === controls) next = { ...controls };
    (next as unknown as Record<keyof RenderControls, unknown>)[controlKey] = rawValue;
  }
  return next;
}

function buildSavedUiState(value: {
  playing: boolean;
  overlay: boolean;
  viewLocked: boolean;
  demoMode: boolean;
  frameCap: number;
  cameraOrbitEnabled: boolean;
  cameraOrbitSpeed: number;
  timelineEnabled: boolean;
  timeline: TimelineState;
  trackingControls: TrackingControls;
}): SavedUiState {
  return sanitizeSavedUiState({
    playing: value.playing,
    overlay: value.overlay,
    viewLocked: value.viewLocked,
    demoMode: value.demoMode,
    frameCap: value.frameCap,
    cameraOrbit: {
      enabled: value.cameraOrbitEnabled,
      speed: value.cameraOrbitSpeed
    },
    timeline: {
      enabled: value.timelineEnabled,
      state: value.timeline
    },
    trackingControls: value.trackingControls
  });
}

function defaultSettingsName(presetId: string): string {
  return `${getLivePreset(presetId).name} ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
}

async function saveSettingsPresetWithFilePicker(
  defaultName: string,
  buildPreset: (name: string) => SavedSettingsPreset
): Promise<SavedSettingsPreset | null> {
  const suggestedName = `${filenameSlug(defaultName)}.fluoddity3d.json`;
  const pickerWindow = window as FilePickerWindow;
  if (pickerWindow.showSaveFilePicker) {
    try {
      const handle = await pickerWindow.showSaveFilePicker({
        suggestedName,
        types: settingsFileTypes
      });
      const saved = buildPreset(settingsNameFromFileName(handle.name) || defaultName);
      const writable = await handle.createWritable();
      await writable.write(settingsPresetBlob(saved));
      await writable.close();
      return saved;
    } catch (error) {
      if (isAbortError(error)) return null;
      throw error;
    }
  }
  const saved = buildPreset(defaultName);
  downloadSettingsPreset(saved);
  return saved;
}

function settingsPresetBlob(settings: SavedSettingsPreset): Blob {
  return new Blob([`${JSON.stringify(settings, null, 2)}\n`], { type: "application/json" });
}

function downloadSettingsPreset(settings: SavedSettingsPreset): void {
  const url = URL.createObjectURL(settingsPresetBlob(settings));
  const link = document.createElement("a");
  link.href = url;
  link.download = `${filenameSlug(settings.name)}.fluoddity3d.json`;
  link.click();
  URL.revokeObjectURL(url);
}

// Performance captures are local browser downloads. The public app has no recording backend.
async function savePerformanceFile(filename: string, blob: Blob): Promise<string | null> {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
  return filename;
}

function savePerformanceRecording(recording: AutomationRecording): Promise<string | null> {
  const blob = new Blob([JSON.stringify(recording)], { type: "application/json" });
  return savePerformanceFile(`performance_${recording.frames.length}f.fluoddity-recording.json`, blob);
}

// Pairs with the recording JSON above (same frame-count stem) so the two files are obviously a
// set; feed the audio file to tools/hdr-export via --audio to mux it onto the offline HDR render.
function savePerformanceAudio(audio: PerformanceAudioRecording, frameCount: number): Promise<string | null> {
  return savePerformanceFile(`performance_${frameCount}f.audio.${audioExtensionFor(audio.mimeType)}`, audio.blob);
}

function publishSavedSettingsForAutomation(settings: SavedSettingsPreset): void {
  const automationWindow = window as unknown as Record<string, unknown>;
  for (const key of ["__audioModulationSaveWrites", "__volumeDensitySaveWrites", "__accumulationSaveWrites"]) {
    const writes = automationWindow[key];
    if (Array.isArray(writes)) {
      writes.push({ suggestedName: null, json: settings });
    }
  }
  const midiWrites = automationWindow.__midiMappingSaveWrites;
  if (Array.isArray(midiWrites)) {
    midiWrites.push(settings);
  }
}

function loadSavedSettings(): SavedSettingsPreset[] {
  return mergeSavedSettings(loadFileBackedSettings(), loadBrowserSavedSettings());
}

function loadFileBackedSettings(): SavedSettingsPreset[] {
  const presets: SavedSettingsPreset[] = [];
  for (const [path, json] of Object.entries(filePresetModules)) {
    try {
      const fileName = path.split("/").pop() ?? path;
      const name = settingsNameFromFileName(fileName);
      const parsed = JSON.parse(json) as unknown;
      const preset = settingsPresetFromUnknown(
        isRecord(parsed) ? { ...parsed, id: `file-${filenameSlug(fileName)}` } : parsed,
        name
      );
      if (preset) presets.push({ ...preset, fileBacked: true });
    } catch {
      // Ignore malformed repo preset files; the preset portability tests catch them.
    }
  }
  return presets;
}

function loadBrowserSavedSettings(): SavedSettingsPreset[] {
  try {
    const raw = localStorage.getItem(savedSettingsStorageKey);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .map((item, index) => settingsPresetFromUnknown(item, `Saved ${index + 1}`))
      .filter((item): item is SavedSettingsPreset => !!item)
      .map((item) => ({ ...item, fileBacked: false }));
  } catch {
    return [];
  }
}

function persistSavedSettings(settings: SavedSettingsPreset[]): void {
  localStorage.setItem(savedSettingsStorageKey, JSON.stringify(settings.filter((item) => !item.fileBacked)));
}

function upsertSavedSettings(settings: SavedSettingsPreset[], next: SavedSettingsPreset): SavedSettingsPreset[] {
  return mergeSavedSettings([next], settings.filter((item) => item.id !== next.id));
}

function mergeSavedSettings(primary: SavedSettingsPreset[], fallback: SavedSettingsPreset[]): SavedSettingsPreset[] {
  const byId = new Map<string, SavedSettingsPreset>();
  for (const settings of [...fallback, ...primary]) {
    const existing = byId.get(settings.id);
    if (!existing || savedSettingsTime(settings) >= savedSettingsTime(existing)) {
      byId.set(settings.id, settings);
    }
  }
  return Array.from(byId.values()).sort((a, b) =>
    savedSettingsTime(b) - savedSettingsTime(a) || a.name.localeCompare(b.name)
  );
}

function savedSettingsTime(settings: SavedSettingsPreset): number {
  const updated = Date.parse(settings.updatedAt || settings.createdAt || "");
  return Number.isFinite(updated) ? updated : 0;
}

function settingsPresetFromJson(json: string, fallbackName: string): SavedSettingsPreset {
  return settingsPresetFromUnknown(JSON.parse(json), fallbackName) ?? (() => {
    throw new Error("JSON preset is missing live settings");
  })();
}

function settingsPresetFromUnknown(value: unknown, fallbackName: string): SavedSettingsPreset | null {
  if (!isRecord(value)) return null;
  const controlsSource = isRecord(value.controls) ? value.controls : value;
  const liveSource = isRecord(value.liveConfig)
    ? value.liveConfig
    : isRecord(value.live)
      ? value.live
      : isRecord(value.config)
        ? value.config
        : null;
  if (!liveSource) return null;
  const now = new Date().toISOString();
  const name = typeof value.name === "string" && value.name.trim() ? value.name.trim() : fallbackName || "Imported";
  const presetId = typeof value.presetId === "string" && livePresets.some((preset) => preset.id === value.presetId)
    ? value.presetId
    : livePresets[0].id;
  const displayMode: DisplayMode = "live";
  const controls = normalizeRenderControlsForDisplayMode(displayMode, sanitizeRenderControls(controlsSource));
  return {
    version: 2,
    id: typeof value.id === "string" && value.id ? value.id : makeSettingsId(),
    name,
    createdAt: typeof value.createdAt === "string" ? value.createdAt : now,
    updatedAt: typeof value.updatedAt === "string" ? value.updatedAt : now,
    presetId,
    displayMode,
    controls,
    liveConfig: sanitizeLiveConfig(liveSource),
    ui: sanitizeSavedUiState(
      isRecord(value.ui) ? value.ui : value,
      fallbackPlayingForSavedUi(isRecord(value.ui) ? value.ui : value)
    ),
    audio: sanitizeSavedAudioState(isRecord(value.audio) ? value.audio : {})
  };
}

function sanitizeSavedAudioState(value: unknown): SavedAudioState {
  const source = isRecord(value) ? value : {};
  return {
    panel: sanitizeAudioPanelState(source.panel),
    sliders: sanitizeSliderModulationSettings(source.sliders)
  };
}

function fallbackPlayingForSavedUi(value: unknown): boolean {
  return hasExplicitPlaybackState(value) ? false : shouldStartPlaying(window.location.search);
}

function hasExplicitPlaybackState(value: unknown): boolean {
  if (!isRecord(value)) return false;
  if (typeof value.playing === "boolean") return true;
  const timelineSource = isRecord(value.timeline) ? value.timeline : null;
  if (!timelineSource) return false;
  if (typeof timelineSource.playing === "boolean") return true;
  const timelineStateSource = isRecord(timelineSource.state) ? timelineSource.state : timelineSource;
  return typeof timelineStateSource.playing === "boolean";
}

function sanitizeSavedUiState(value: unknown, fallbackPlaying = false): SavedUiState {
  const source = isRecord(value) ? value : {};
  const timelineSource = isRecord(source.timeline) ? source.timeline : {};
  const timelineStateSource = isRecord(timelineSource.state) ? timelineSource.state : timelineSource;
  const timelineState = sanitizeTimelineState(timelineStateSource, fallbackPlaying);
  const playing = typeof source.playing === "boolean"
    ? source.playing
    : timelineState.playing;
  const orbitSource = isRecord(source.cameraOrbit) ? source.cameraOrbit : {};
  const trackingSource = isRecord(source.trackingControls)
    ? source.trackingControls
    : isRecord(source.tracking)
      ? source.tracking
      : {};
  return {
    playing,
    overlay: typeof source.overlay === "boolean" ? source.overlay : false,
    viewLocked: typeof source.viewLocked === "boolean" ? source.viewLocked : false,
    demoMode: typeof source.demoMode === "boolean" ? source.demoMode : false,
    frameCap: clamp(finiteNumber(source.frameCap, 60), 0, 120),
    cameraOrbit: {
      enabled: typeof orbitSource.enabled === "boolean" ? orbitSource.enabled : defaultCameraOrbit.enabled,
      speed: clamp(finiteNumber(orbitSource.speed, defaultCameraOrbit.speed), orbitSpeedSliderRange.min, orbitSpeedSliderRange.max)
    },
    timeline: {
      enabled: typeof timelineSource.enabled === "boolean" ? timelineSource.enabled : false,
      state: { ...timelineState, playing }
    },
    trackingControls: sanitizeTrackingControls(trackingSource)
  };
}

function sanitizeTimelineState(value: unknown, fallbackPlaying = false): TimelineState {
  const source = isRecord(value) ? value : {};
  const { loopIn, loopOut } = normalizeLoop(
    finiteNumber(source.loopIn, 0),
    finiteNumber(source.loopOut, TIMELINE_TOTAL_FRAMES - 1)
  );
  return {
    currentFrame: Math.round(clamp(finiteNumber(source.currentFrame, 0), 0, TIMELINE_TOTAL_FRAMES - 1)),
    playing: typeof source.playing === "boolean" ? source.playing : fallbackPlaying,
    loopIn,
    loopOut,
    playbackSpeed: clamp(finiteNumber(source.playbackSpeed, 1), 0.05, 8)
  };
}

function sanitizeTrackingControls(value: unknown): TrackingControls {
  const source = isRecord(value) ? value : {};
  return {
    follow: typeof source.follow === "boolean" ? source.follow : trackingDefaults.follow,
    look: typeof source.look === "boolean" ? source.look : trackingDefaults.look,
    followSpeed: clamp(finiteNumber(source.followSpeed, trackingDefaults.followSpeed), trackingSliderRanges.followSpeed.min, trackingSliderRanges.followSpeed.max),
    followSmoothing: clamp(finiteNumber(source.followSmoothing, trackingDefaults.followSmoothing), trackingSliderRanges.followSmoothing.min, trackingSliderRanges.followSmoothing.max),
    lookSpeed: clamp(finiteNumber(source.lookSpeed, trackingDefaults.lookSpeed), trackingSliderRanges.lookSpeed.min, trackingSliderRanges.lookSpeed.max),
    lookSmoothing: clamp(finiteNumber(source.lookSmoothing, trackingDefaults.lookSmoothing), trackingSliderRanges.lookSmoothing.min, trackingSliderRanges.lookSmoothing.max),
    followDistance: clamp(finiteNumber(source.followDistance, trackingDefaults.followDistance), trackingSliderRanges.followDistance.min, trackingSliderRanges.followDistance.max),
    followHeight: clamp(finiteNumber(source.followHeight, trackingDefaults.followHeight), trackingSliderRanges.followHeight.min, trackingSliderRanges.followHeight.max),
    cohesion: clamp(finiteNumber(source.cohesion, trackingDefaults.cohesion), trackingSliderRanges.cohesion.min, trackingSliderRanges.cohesion.max)
  };
}

function sanitizeRenderControls(value: unknown): RenderControls {
  const source = isRecord(value) ? value : {};
  const palette = source.palette === "ember" || source.palette === "spectral" || source.palette === "aurora"
    ? source.palette
    : defaultControls.palette;
  const particleMinPx = clamp(finiteNumber(source.particleMinPx, defaultControls.particleMinPx), 0, 8);
  const particleMaxPx = clamp(finiteNumber(source.particleMaxPx, defaultControls.particleMaxPx), 0.2, 32);
  const particleStretch = clamp(finiteNumber(source.particleStretch, defaultControls.particleStretch), renderSliderRanges.particleStretch.min, renderSliderRanges.particleStretch.max);
  const particleStretchMin = Math.min(
    particleStretch,
    clamp(finiteNumber(source.particleStretchMin, defaultControls.particleStretchMin), renderSliderRanges.particleStretchMin.min, renderSliderRanges.particleStretchMin.max)
  );
  const variationSizeMin = clamp(finiteNumber(source.variationSizeMin, defaultControls.variationSizeMin), renderSliderRanges.variationSizeMin.min, renderSliderRanges.variationSizeMin.max);
  const variationSizeMax = clamp(finiteNumber(source.variationSizeMax, defaultControls.variationSizeMax), renderSliderRanges.variationSizeMax.min, renderSliderRanges.variationSizeMax.max);
  const variationBrightMin = clamp(finiteNumber(source.variationBrightMin, defaultControls.variationBrightMin), renderSliderRanges.variationBrightMin.min, renderSliderRanges.variationBrightMin.max);
  const variationBrightMax = clamp(finiteNumber(source.variationBrightMax, defaultControls.variationBrightMax), renderSliderRanges.variationBrightMax.min, renderSliderRanges.variationBrightMax.max);
  const variationOpacityMin = clamp(finiteNumber(source.variationOpacityMin, defaultControls.variationOpacityMin), renderSliderRanges.variationOpacityMin.min, renderSliderRanges.variationOpacityMin.max);
  const variationOpacityMax = clamp(finiteNumber(source.variationOpacityMax, defaultControls.variationOpacityMax), renderSliderRanges.variationOpacityMax.min, renderSliderRanges.variationOpacityMax.max);
  const variationColorMin = clamp(finiteNumber(source.variationColorMin, defaultControls.variationColorMin), renderSliderRanges.variationColorMin.min, renderSliderRanges.variationColorMin.max);
  const variationColorMax = clamp(finiteNumber(source.variationColorMax, defaultControls.variationColorMax), renderSliderRanges.variationColorMax.min, renderSliderRanges.variationColorMax.max);
  return {
    ...defaultControls,
    bloomStrength: clamp(finiteNumber(source.bloomStrength, defaultControls.bloomStrength), renderSliderRanges.bloomStrength.min, renderSliderRanges.bloomStrength.max),
    bloomThreshold: clamp(finiteNumber(source.bloomThreshold, defaultControls.bloomThreshold), renderSliderRanges.bloomThreshold.min, renderSliderRanges.bloomThreshold.max),
    bloomRadius: clamp(finiteNumber(source.bloomRadius, defaultControls.bloomRadius), renderSliderRanges.bloomRadius.min, renderSliderRanges.bloomRadius.max),
    colorSaturation: clamp(finiteNumber(source.colorSaturation, defaultControls.colorSaturation), renderSliderRanges.colorSaturation.min, renderSliderRanges.colorSaturation.max),
    colorContrast: clamp(finiteNumber(source.colorContrast, defaultControls.colorContrast), renderSliderRanges.colorContrast.min, renderSliderRanges.colorContrast.max),
    chromaticAberration: clamp(finiteNumber(source.chromaticAberration, defaultControls.chromaticAberration), renderSliderRanges.chromaticAberration.min, renderSliderRanges.chromaticAberration.max),
    vignetteStrength: clamp(finiteNumber(source.vignetteStrength, defaultControls.vignetteStrength), renderSliderRanges.vignetteStrength.min, renderSliderRanges.vignetteStrength.max),
    vignetteSoftness: clamp(finiteNumber(source.vignetteSoftness, defaultControls.vignetteSoftness), renderSliderRanges.vignetteSoftness.min, renderSliderRanges.vignetteSoftness.max),
    streakStrength: clamp(finiteNumber(source.streakStrength, defaultControls.streakStrength), renderSliderRanges.streakStrength.min, renderSliderRanges.streakStrength.max),
    streakLength: clamp(finiteNumber(source.streakLength, defaultControls.streakLength), renderSliderRanges.streakLength.min, renderSliderRanges.streakLength.max),
    streakVertical: clamp(finiteNumber(source.streakVertical, defaultControls.streakVertical), renderSliderRanges.streakVertical.min, renderSliderRanges.streakVertical.max),
    flareHeight: clamp(finiteNumber(source.flareHeight, defaultControls.flareHeight), renderSliderRanges.flareHeight.min, renderSliderRanges.flareHeight.max),
    flareCutoff: clamp(finiteNumber(source.flareCutoff, defaultControls.flareCutoff), renderSliderRanges.flareCutoff.min, renderSliderRanges.flareCutoff.max),
    ribbonFraction: clamp(finiteNumber(source.ribbonFraction, defaultControls.ribbonFraction), renderSliderRanges.ribbonFraction.min, renderSliderRanges.ribbonFraction.max),
    ribbonWidth: clamp(finiteNumber(source.ribbonWidth, defaultControls.ribbonWidth), renderSliderRanges.ribbonWidth.min, renderSliderRanges.ribbonWidth.max),
    ribbonTaper: clamp(finiteNumber(source.ribbonTaper, defaultControls.ribbonTaper), renderSliderRanges.ribbonTaper.min, renderSliderRanges.ribbonTaper.max),
    ribbonLength: clamp(finiteNumber(source.ribbonLength, defaultControls.ribbonLength), renderSliderRanges.ribbonLength.min, renderSliderRanges.ribbonLength.max),
    ribbonJoints: clamp(finiteNumber(source.ribbonJoints, defaultControls.ribbonJoints), renderSliderRanges.ribbonJoints.min, renderSliderRanges.ribbonJoints.max),
    ribbonFadeStart: clamp(finiteNumber(source.ribbonFadeStart, defaultControls.ribbonFadeStart), renderSliderRanges.ribbonFadeStart.min, renderSliderRanges.ribbonFadeStart.max),
    ribbonEdgeFade: clamp(finiteNumber(source.ribbonEdgeFade, defaultControls.ribbonEdgeFade), renderSliderRanges.ribbonEdgeFade.min, renderSliderRanges.ribbonEdgeFade.max),
    density: clamp(finiteNumber(source.density, defaultControls.density), 0.25, 2.8),
    exposure: clamp(finiteNumber(source.exposure, defaultControls.exposure), 0.4, 2.6),
    sceneBrightness: clamp(finiteNumber(source.sceneBrightness, defaultControls.sceneBrightness), 0, 2.6),
    fov: clamp(finiteNumber(source.fov, defaultControls.fov), 10, 170),
    aperture: clamp(finiteNumber(source.aperture, defaultControls.aperture), 0, 2),
    focusDistance: clamp(finiteNumber(source.focusDistance, defaultControls.focusDistance), renderSliderRanges.focusDistance.min, renderSliderRanges.focusDistance.max),
    dofBlur: clamp(finiteNumber(source.dofBlur, defaultControls.dofBlur), 0, 8),
    dofEnabled: typeof source.dofEnabled === "boolean" ? source.dofEnabled : defaultControls.dofEnabled,
    dofDebug: typeof source.dofDebug === "boolean" ? source.dofDebug : defaultControls.dofDebug,
    raySteps: Math.round(clamp(finiteNumber(source.raySteps, defaultControls.raySteps), 8, 512) / 8) * 8,
    rayResolution: clampRayResolution(finiteNumber(source.rayResolution, defaultControls.rayResolution)),
    fogTemporal: typeof source.fogTemporal === "boolean" ? source.fogTemporal : defaultControls.fogTemporal,
    fogRenderScale: clamp(finiteNumber(source.fogRenderScale, defaultControls.fogRenderScale), 0.25, 1),
    fogStepScale: clamp(finiteNumber(source.fogStepScale, defaultControls.fogStepScale), 0.125, 1),
    fogTemporalBlend: clamp(finiteNumber(source.fogTemporalBlend, defaultControls.fogTemporalBlend), 0, 0.96),
    fogBlueNoise: typeof source.fogBlueNoise === "boolean" ? source.fogBlueNoise : defaultControls.fogBlueNoise,
    fieldTextureSampling: typeof source.fieldTextureSampling === "boolean" ? source.fieldTextureSampling : defaultControls.fieldTextureSampling,
    emptySpaceSkipping: typeof source.emptySpaceSkipping === "boolean" ? source.emptySpaceSkipping : defaultControls.emptySpaceSkipping,
    emptySpaceThreshold: clamp(finiteNumber(source.emptySpaceThreshold, defaultControls.emptySpaceThreshold), 0, 0.2),
    emptySpaceStride: Math.round(clamp(finiteNumber(source.emptySpaceStride, defaultControls.emptySpaceStride), 1, 8)),
    particleSizePx: clamp(finiteNumber(source.particleSizePx, defaultControls.particleSizePx), 0.2, 24),
    particleMinPx: Math.min(particleMinPx, particleMaxPx),
    particleMaxPx: Math.max(particleMaxPx, particleMinPx),
    particleOpacity: clamp(finiteNumber(source.particleOpacity, defaultControls.particleOpacity), 0, 1),
    particleBrightness: clamp(finiteNumber(source.particleBrightness, defaultControls.particleBrightness), 0, 8),
    particleColorMode: coerceParticleColorMode(source.particleColorMode) ?? defaultControls.particleColorMode,
    particleVelocityStretch: typeof source.particleVelocityStretch === "boolean" ? source.particleVelocityStretch : defaultControls.particleVelocityStretch,
    particleStretch,
    particleStretchMin,
    particleStretchSpeed: clamp(finiteNumber(source.particleStretchSpeed, defaultControls.particleStretchSpeed), renderSliderRanges.particleStretchSpeed.min, renderSliderRanges.particleStretchSpeed.max),
    particleSpeedCutoff: clamp(finiteNumber(source.particleSpeedCutoff, defaultControls.particleSpeedCutoff), renderSliderRanges.particleSpeedCutoff.min, renderSliderRanges.particleSpeedCutoff.max),
    particleSlowCutoff: clamp(finiteNumber(source.particleSlowCutoff, defaultControls.particleSlowCutoff), renderSliderRanges.particleSlowCutoff.min, renderSliderRanges.particleSlowCutoff.max),
    particleGlowCore: clamp(finiteNumber(source.particleGlowCore, defaultControls.particleGlowCore), renderSliderRanges.particleGlowCore.min, renderSliderRanges.particleGlowCore.max),
    particleHotCore: clamp(finiteNumber(source.particleHotCore, defaultControls.particleHotCore), renderSliderRanges.particleHotCore.min, renderSliderRanges.particleHotCore.max),
    particleExponent: clamp(finiteNumber(source.particleExponent, defaultControls.particleExponent), renderSliderRanges.particleExponent.min, renderSliderRanges.particleExponent.max),
    particleBrightnessBoost: clamp(finiteNumber(source.particleBrightnessBoost, defaultControls.particleBrightnessBoost), renderSliderRanges.particleBrightnessBoost.min, renderSliderRanges.particleBrightnessBoost.max),
    particleSupportSmoothing: clamp(finiteNumber(source.particleSupportSmoothing, defaultControls.particleSupportSmoothing), renderSliderRanges.particleSupportSmoothing.min, renderSliderRanges.particleSupportSmoothing.max),
    particleHazeCull: clamp(finiteNumber(source.particleHazeCull, defaultControls.particleHazeCull), renderSliderRanges.particleHazeCull.min, renderSliderRanges.particleHazeCull.max),
    particleDespeckle: clamp(finiteNumber(source.particleDespeckle, defaultControls.particleDespeckle), renderSliderRanges.particleDespeckle.min, renderSliderRanges.particleDespeckle.max),
    particleBlendMode: parseParticleBlendMode(typeof source.particleBlendMode === "string" ? source.particleBlendMode : null) ?? defaultControls.particleBlendMode,
    particleDensityCutoff: clamp(finiteNumber(source.particleDensityCutoff, defaultControls.particleDensityCutoff), renderSliderRanges.particleDensityCutoff.min, renderSliderRanges.particleDensityCutoff.max),
    particleDensityRadius: clamp(finiteNumber(source.particleDensityRadius, defaultControls.particleDensityRadius), 0, 0.18),
    particleDensityNormalize: clamp(finiteNumber(source.particleDensityNormalize, defaultControls.particleDensityNormalize), renderSliderRanges.particleDensityNormalize.min, renderSliderRanges.particleDensityNormalize.max),
    particleDensitySoftness: clamp(finiteNumber(source.particleDensitySoftness, defaultControls.particleDensitySoftness), renderSliderRanges.particleDensitySoftness.min, renderSliderRanges.particleDensitySoftness.max),
    particleSupportMask: clamp(finiteNumber(source.particleSupportMask, defaultControls.particleSupportMask), renderSliderRanges.particleSupportMask.min, renderSliderRanges.particleSupportMask.max),
    particleSupportRadius: clamp(finiteNumber(source.particleSupportRadius, defaultControls.particleSupportRadius), renderSliderRanges.particleSupportRadius.min, renderSliderRanges.particleSupportRadius.max),
    particleSupportNeighbors: clamp(finiteNumber(source.particleSupportNeighbors, defaultControls.particleSupportNeighbors), renderSliderRanges.particleSupportNeighbors.min, renderSliderRanges.particleSupportNeighbors.max),
    particleSupportFlow: clamp(finiteNumber(source.particleSupportFlow, defaultControls.particleSupportFlow), renderSliderRanges.particleSupportFlow.min, renderSliderRanges.particleSupportFlow.max),
    fastParticleRender: typeof source.fastParticleRender === "boolean" ? source.fastParticleRender : defaultControls.fastParticleRender,
    fastNoBloomPost: typeof source.fastNoBloomPost === "boolean" ? source.fastNoBloomPost : defaultControls.fastNoBloomPost,
    particleCutoffPrepass: typeof source.particleCutoffPrepass === "boolean" ? source.particleCutoffPrepass : defaultControls.particleCutoffPrepass,
    densityLargeHalfRes: typeof source.densityLargeHalfRes === "boolean" ? source.densityLargeHalfRes : defaultControls.densityLargeHalfRes,
    densityPassStrength: clamp(finiteNumber(source.densityPassStrength, defaultControls.densityPassStrength), renderSliderRanges.densityPassStrength.min, renderSliderRanges.densityPassStrength.max),
    densitySmallScale: clamp(finiteNumber(source.densitySmallScale, defaultControls.densitySmallScale), 0.25, 5),
    densityLargeScale: clamp(finiteNumber(source.densityLargeScale, defaultControls.densityLargeScale), renderSliderRanges.densityLargeScale.min, renderSliderRanges.densityLargeScale.max),
    densityLargeThreshold: clamp(finiteNumber(source.densityLargeThreshold, defaultControls.densityLargeThreshold), renderSliderRanges.densityLargeThreshold.min, renderSliderRanges.densityLargeThreshold.max),
    densityContrastGain: clamp(finiteNumber(source.densityContrastGain, defaultControls.densityContrastGain), renderSliderRanges.densityContrastGain.min, renderSliderRanges.densityContrastGain.max),
    densityContrastBalance: clamp(finiteNumber(source.densityContrastBalance, defaultControls.densityContrastBalance), 0, 1.5),
    densityEmissionPower: clamp(finiteNumber(source.densityEmissionPower, defaultControls.densityEmissionPower), 0.25, 5),
    densityOcclusion: clamp(finiteNumber(source.densityOcclusion, defaultControls.densityOcclusion), 0, 1),
    accumulationStrength: clamp(finiteNumber(source.accumulationStrength, defaultControls.accumulationStrength), renderSliderRanges.accumulationStrength.min, renderSliderRanges.accumulationStrength.max),
    accumulationRadius: clamp(finiteNumber(source.accumulationRadius, defaultControls.accumulationRadius), renderSliderRanges.accumulationRadius.min, renderSliderRanges.accumulationRadius.max),
    accumulationCurve: clamp(finiteNumber(source.accumulationCurve, defaultControls.accumulationCurve), renderSliderRanges.accumulationCurve.min, renderSliderRanges.accumulationCurve.max),
    accumulationMemory: clamp(finiteNumber(source.accumulationMemory, defaultControls.accumulationMemory), renderSliderRanges.accumulationMemory.min, renderSliderRanges.accumulationMemory.max),
    accumulationNoiseReject: clamp(finiteNumber(source.accumulationNoiseReject, defaultControls.accumulationNoiseReject), renderSliderRanges.accumulationNoiseReject.min, renderSliderRanges.accumulationNoiseReject.max),
    trailOpacity: clamp(finiteNumber(source.trailOpacity, defaultControls.trailOpacity), 0, 2.5),
    fogBrightness: clamp(finiteNumber(source.fogBrightness, defaultControls.fogBrightness), 0, 8),
    trailThreshold: clamp(finiteNumber(source.trailThreshold, defaultControls.trailThreshold), 0, 0.28),
    trailColorMode: parseTrailColorMode(typeof source.trailColorMode === "string" ? source.trailColorMode : null) ?? defaultControls.trailColorMode,
    fogTint: sanitizeHexColor(source.fogTint, defaultControls.fogTint),
    particleTint: sanitizeHexColor(source.particleTint, defaultControls.particleTint),
    renderLayer: "particles",
    palette,
    filament: clamp(finiteNumber(source.filament, defaultControls.filament), 0, 1),
    cameraYaw: finiteNumber(source.cameraYaw, defaultControls.cameraYaw),
    cameraPitch: clamp(finiteNumber(source.cameraPitch, defaultControls.cameraPitch), minCameraPitch, maxCameraPitch),
    cameraDistance: clampCameraDistance(finiteNumber(source.cameraDistance, defaultControls.cameraDistance)),
    cameraPanX: clamp(finiteNumber(source.cameraPanX, defaultControls.cameraPanX), minCameraPan, maxCameraPan),
    cameraPanY: clamp(finiteNumber(source.cameraPanY, defaultControls.cameraPanY), minCameraPan, maxCameraPan),
    variationMaster: clamp(finiteNumber(source.variationMaster, defaultControls.variationMaster), renderSliderRanges.variationMaster.min, renderSliderRanges.variationMaster.max),
    variationDrift: clamp(finiteNumber(source.variationDrift, defaultControls.variationDrift), renderSliderRanges.variationDrift.min, renderSliderRanges.variationDrift.max),
    variationNoiseMix: clamp(finiteNumber(source.variationNoiseMix, defaultControls.variationNoiseMix), renderSliderRanges.variationNoiseMix.min, renderSliderRanges.variationNoiseMix.max),
    variationFreq: clamp(finiteNumber(source.variationFreq, defaultControls.variationFreq), renderSliderRanges.variationFreq.min, renderSliderRanges.variationFreq.max),
    variationOctaves: Math.round(clamp(finiteNumber(source.variationOctaves, defaultControls.variationOctaves), renderSliderRanges.variationOctaves.min, renderSliderRanges.variationOctaves.max)),
    variationGain: clamp(finiteNumber(source.variationGain, defaultControls.variationGain), renderSliderRanges.variationGain.min, renderSliderRanges.variationGain.max),
    variationLacunarity: clamp(finiteNumber(source.variationLacunarity, defaultControls.variationLacunarity), renderSliderRanges.variationLacunarity.min, renderSliderRanges.variationLacunarity.max),
    variationSizeAmount: clamp(finiteNumber(source.variationSizeAmount, defaultControls.variationSizeAmount), renderSliderRanges.variationSizeAmount.min, renderSliderRanges.variationSizeAmount.max),
    variationSizeCurve: clamp(finiteNumber(source.variationSizeCurve, defaultControls.variationSizeCurve), renderSliderRanges.variationSizeCurve.min, renderSliderRanges.variationSizeCurve.max),
    variationSizeMin: Math.min(variationSizeMin, variationSizeMax),
    variationSizeMax: Math.max(variationSizeMin, variationSizeMax),
    variationBrightAmount: clamp(finiteNumber(source.variationBrightAmount, defaultControls.variationBrightAmount), renderSliderRanges.variationBrightAmount.min, renderSliderRanges.variationBrightAmount.max),
    variationBrightCurve: clamp(finiteNumber(source.variationBrightCurve, defaultControls.variationBrightCurve), renderSliderRanges.variationBrightCurve.min, renderSliderRanges.variationBrightCurve.max),
    variationBrightMin: Math.min(variationBrightMin, variationBrightMax),
    variationBrightMax: Math.max(variationBrightMin, variationBrightMax),
    variationOpacityAmount: clamp(finiteNumber(source.variationOpacityAmount, defaultControls.variationOpacityAmount), renderSliderRanges.variationOpacityAmount.min, renderSliderRanges.variationOpacityAmount.max),
    variationOpacityCurve: clamp(finiteNumber(source.variationOpacityCurve, defaultControls.variationOpacityCurve), renderSliderRanges.variationOpacityCurve.min, renderSliderRanges.variationOpacityCurve.max),
    variationOpacityMin: Math.min(variationOpacityMin, variationOpacityMax),
    variationOpacityMax: Math.max(variationOpacityMin, variationOpacityMax),
    variationColorAmount: clamp(finiteNumber(source.variationColorAmount, defaultControls.variationColorAmount), renderSliderRanges.variationColorAmount.min, renderSliderRanges.variationColorAmount.max),
    variationColorCurve: clamp(finiteNumber(source.variationColorCurve, defaultControls.variationColorCurve), renderSliderRanges.variationColorCurve.min, renderSliderRanges.variationColorCurve.max),
    variationColorMin: Math.min(variationColorMin, variationColorMax),
    variationColorMax: Math.max(variationColorMin, variationColorMax)
  };
}

function sanitizeLiveConfig(value: unknown): LiveGpu3dConfig {
  const source = isRecord(value) ? value : {};
  const width = clampVolumeSize(finiteNumber(source.width, defaultLiveGpu3dConfig.width));
  const height = clampVolumeSize(finiteNumber(source.height, width));
  const depth = clampVolumeSize(finiteNumber(source.depth, width));
  const initialConditions = clampInteger(source.initialConditions, 0, 8) as LiveGpu3dConfig["initialConditions"];
  const boundaryMode = clampInteger(source.boundaryMode, 0, 6) as LiveGpu3dConfig["boundaryMode"];
  const domainShape = clampInteger(source.domainShape, 0, 3) as LiveGpu3dConfig["domainShape"];
  const absoluteOrientation = clampInteger(source.absoluteOrientation, 0, 5) as LiveGpu3dConfig["absoluteOrientation"];
  return normalizeTrailKernel({
    ...defaultLiveGpu3dConfig,
    // Emergent behaviors — clamped so saved presets round-trip; absent keys fall back to the
    // default (0/off) via the spread above, so old presets reproduce the classic engine exactly.
    mips: clamp(finiteNumber(source.mips, defaultLiveGpu3dConfig.mips), 0, 1),
    anisoFollow: clamp(finiteNumber(source.anisoFollow, defaultLiveGpu3dConfig.anisoFollow), 0, 1),
    flockAlign: clamp(finiteNumber(source.flockAlign, defaultLiveGpu3dConfig.flockAlign), 0, 1),
    flockSeparate: clamp(finiteNumber(source.flockSeparate, defaultLiveGpu3dConfig.flockSeparate), 0, 1),
    chemotaxis: clamp(finiteNumber(source.chemotaxis, defaultLiveGpu3dConfig.chemotaxis), -1, 1),
    quorumStrength: clamp(finiteNumber(source.quorumStrength, defaultLiveGpu3dConfig.quorumStrength), 0, 1),
    quorumThreshold: clamp(finiteNumber(source.quorumThreshold, defaultLiveGpu3dConfig.quorumThreshold), 0, 1),
    leniaStrength: clamp(finiteNumber(source.leniaStrength, defaultLiveGpu3dConfig.leniaStrength), 0, 1),
    leniaCenter: clamp(finiteNumber(source.leniaCenter, defaultLiveGpu3dConfig.leniaCenter), 0, 1),
    leniaWidth: clamp(finiteNumber(source.leniaWidth, defaultLiveGpu3dConfig.leniaWidth), 0.001, 0.2),
    speciesForce: clamp(finiteNumber(source.speciesForce, defaultLiveGpu3dConfig.speciesForce), 0, 1),
    predator: clamp(finiteNumber(source.predator, defaultLiveGpu3dConfig.predator), 0, 1),
    alarm: clamp(finiteNumber(source.alarm, defaultLiveGpu3dConfig.alarm), 0, 1),
    grayScott: clamp(finiteNumber(source.grayScott, defaultLiveGpu3dConfig.grayScott), 0, 1),
    gsFeed: clamp(finiteNumber(source.gsFeed, defaultLiveGpu3dConfig.gsFeed), 0, 0.1),
    gsKill: clamp(finiteNumber(source.gsKill, defaultLiveGpu3dConfig.gsKill), 0, 0.1),
    energy: clamp(finiteNumber(source.energy, defaultLiveGpu3dConfig.energy), 0, 1),
    energyDrain: clamp(finiteNumber(source.energyDrain, defaultLiveGpu3dConfig.energyDrain), 0, 1),
    seed: clampInteger(source.seed, 0, 0xffffffff),
    simulationSpeed: clampSimulationSpeed(finiteNumber(source.simulationSpeed, defaultLiveGpu3dConfig.simulationSpeed)),
    width,
    height,
    depth,
    particleCount: clampParticleCount(finiteNumber(source.particleCount, defaultLiveGpu3dConfig.particleCount)),
    dt: finiteNumber(source.dt, defaultLiveGpu3dConfig.dt),
    sensorGain: finiteNumber(source.sensorGain, defaultLiveGpu3dConfig.sensorGain),
    sensorAngle: finiteNumber(source.sensorAngle, defaultLiveGpu3dConfig.sensorAngle),
    sensorDistance: finiteNumber(source.sensorDistance, defaultLiveGpu3dConfig.sensorDistance),
    mutationScale: finiteNumber(source.mutationScale, defaultLiveGpu3dConfig.mutationScale),
    globalForceMult: finiteNumber(source.globalForceMult, defaultLiveGpu3dConfig.globalForceMult),
    drag: finiteNumber(source.drag, defaultLiveGpu3dConfig.drag),
    strafePower: finiteNumber(source.strafePower, defaultLiveGpu3dConfig.strafePower),
    strafeMomentum: clamp(finiteNumber(source.strafeMomentum, defaultLiveGpu3dConfig.strafeMomentum), liveSliderRanges.strafeMomentum.min, liveSliderRanges.strafeMomentum.max),
    axialForce: finiteNumber(source.axialForce, defaultLiveGpu3dConfig.axialForce),
    lateralForce: finiteNumber(source.lateralForce, defaultLiveGpu3dConfig.lateralForce),
    hazardRate: finiteNumber(source.hazardRate, defaultLiveGpu3dConfig.hazardRate),
    trailPersistence: finiteNumber(source.trailPersistence, defaultLiveGpu3dConfig.trailPersistence),
    trailDiffusion: finiteNumber(source.trailDiffusion, defaultLiveGpu3dConfig.trailDiffusion),
    depositRadius: finiteNumber(source.depositRadius, defaultLiveGpu3dConfig.depositRadius),
    depositTapRadius: finiteNumber(source.depositTapRadius, defaultLiveGpu3dConfig.depositTapRadius),
    depositMass: finiteNumber(source.depositMass, defaultLiveGpu3dConfig.depositMass),
    sigma: finiteNumber(source.sigma, defaultLiveGpu3dConfig.sigma),
    pulseDepth: clamp(finiteNumber(source.pulseDepth, defaultLiveGpu3dConfig.pulseDepth), 0, 1),
    pulseRate: clamp(finiteNumber(source.pulseRate, defaultLiveGpu3dConfig.pulseRate), 0, 4),
    restlessness: clamp(finiteNumber(source.restlessness, defaultLiveGpu3dConfig.restlessness), 0, 1),
    cohorts: clampInteger(source.cohorts, 1, 128),
    ruleSeed: finiteNumber(source.ruleSeed, defaultLiveGpu3dConfig.ruleSeed),
    hueSensitivity: finiteNumber(source.hueSensitivity, defaultLiveGpu3dConfig.hueSensitivity),
    colorByCohort: typeof source.colorByCohort === "boolean" ? source.colorByCohort : defaultLiveGpu3dConfig.colorByCohort,
    symmetryAxes: typeof source.symmetryAxes === "number" ? clampInteger(source.symmetryAxes, 0, 7) : (source.disableSymmetry === true ? 0 : 2),
    absoluteOrientation,
    orientationMix: finiteNumber(source.orientationMix, defaultLiveGpu3dConfig.orientationMix),
    initialConditions,
    boundaryMode,
    domainShape,
    rule: Array.isArray(source.rule) ? source.rule.slice(0, 80).map((value) => finiteNumber(value, 0)) : defaultLiveGpu3dConfig.rule,
    recycleCutoff: clamp(finiteNumber(source.recycleCutoff, defaultLiveGpu3dConfig.recycleCutoff), 0, 0.25),
    recycleEnabled: typeof source.recycleEnabled === "boolean" ? source.recycleEnabled : defaultLiveGpu3dConfig.recycleEnabled
  });
}

function finiteNumber(value: unknown, fallback: number): number {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : fallback;
}

function clampInteger(value: unknown, min: number, max: number): number {
  return Math.round(clamp(finiteNumber(value, min), min, max));
}

function makeSettingsId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `settings-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function filenameSlug(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "fluoddity-settings";
}

function settingsNameFromFileName(value: string): string {
  return value
    .replace(/\.fluoddity3d\.json$/i, "")
    .replace(/\.json$/i, "")
    .trim();
}

function isAbortError(error: unknown): boolean {
  return isRecord(error) && error.name === "AbortError";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function createInitialLiveState(): { presetId: string; config: LiveGpu3dConfig; controls: RenderControls } {
  const params = new URLSearchParams(window.location.search);
  const preset = getLivePreset(params.get("preset") ?? livePresets[0].id);
  const baseControls: RenderControls = { ...defaultControls, ...preset.renderControls };
  const particleCount = params.has("particles")
    ? clampParticleCount(Number(params.get("particles")))
    : preset.config.particleCount;
  const volumeSize = params.has("volume")
    ? clampVolumeSize(Number(params.get("volume")))
    : recommendedVolumeSize(particleCount, preset.config.width);
  const simulationSpeed = params.has("simSpeed")
    ? clampSimulationSpeed(Number(params.get("simSpeed")))
    : params.has("speed")
      ? clampSimulationSpeed(Number(params.get("speed")))
      : preset.config.simulationSpeed;
  const rayResolution = params.has("rayRes")
    ? clampRayResolution(Number(params.get("rayRes")))
    : baseControls.rayResolution;
  const raySteps = params.has("raySteps")
    ? clampRaySteps(Number(params.get("raySteps")))
    : baseControls.raySteps;
  const bloomStrength = params.has("bloom")
    ? clamp(finiteNumber(Number(params.get("bloom")), baseControls.bloomStrength), renderSliderRanges.bloomStrength.min, renderSliderRanges.bloomStrength.max)
    : baseControls.bloomStrength;
  const bloomThreshold = params.has("bloomThreshold")
    ? clamp(finiteNumber(Number(params.get("bloomThreshold")), baseControls.bloomThreshold), renderSliderRanges.bloomThreshold.min, renderSliderRanges.bloomThreshold.max)
    : baseControls.bloomThreshold;
  const bloomRadius = params.has("bloomRadius")
    ? clamp(finiteNumber(Number(params.get("bloomRadius")), baseControls.bloomRadius), renderSliderRanges.bloomRadius.min, renderSliderRanges.bloomRadius.max)
    : baseControls.bloomRadius;
  const colorSaturation = params.has("colorSaturation")
    ? clamp(finiteNumber(Number(params.get("colorSaturation")), baseControls.colorSaturation), renderSliderRanges.colorSaturation.min, renderSliderRanges.colorSaturation.max)
    : baseControls.colorSaturation;
  const colorContrast = params.has("colorContrast")
    ? clamp(finiteNumber(Number(params.get("colorContrast")), baseControls.colorContrast), renderSliderRanges.colorContrast.min, renderSliderRanges.colorContrast.max)
    : baseControls.colorContrast;
  const chromaticAberration = params.has("chromaticAberration")
    ? clamp(finiteNumber(Number(params.get("chromaticAberration")), baseControls.chromaticAberration), renderSliderRanges.chromaticAberration.min, renderSliderRanges.chromaticAberration.max)
    : baseControls.chromaticAberration;
  const vignetteStrength = params.has("vignetteStrength")
    ? clamp(finiteNumber(Number(params.get("vignetteStrength")), baseControls.vignetteStrength), renderSliderRanges.vignetteStrength.min, renderSliderRanges.vignetteStrength.max)
    : baseControls.vignetteStrength;
  const vignetteSoftness = params.has("vignetteSoftness")
    ? clamp(finiteNumber(Number(params.get("vignetteSoftness")), baseControls.vignetteSoftness), renderSliderRanges.vignetteSoftness.min, renderSliderRanges.vignetteSoftness.max)
    : baseControls.vignetteSoftness;
  const streakStrength = params.has("streakStrength")
    ? clamp(finiteNumber(Number(params.get("streakStrength")), baseControls.streakStrength), renderSliderRanges.streakStrength.min, renderSliderRanges.streakStrength.max)
    : baseControls.streakStrength;
  const streakLength = params.has("streakLength")
    ? clamp(finiteNumber(Number(params.get("streakLength")), baseControls.streakLength), renderSliderRanges.streakLength.min, renderSliderRanges.streakLength.max)
    : baseControls.streakLength;
  const streakVertical = params.has("streakVertical")
    ? clamp(finiteNumber(Number(params.get("streakVertical")), baseControls.streakVertical), renderSliderRanges.streakVertical.min, renderSliderRanges.streakVertical.max)
    : baseControls.streakVertical;
  const flareHeight = params.has("flareHeight")
    ? clamp(finiteNumber(Number(params.get("flareHeight")), baseControls.flareHeight), renderSliderRanges.flareHeight.min, renderSliderRanges.flareHeight.max)
    : baseControls.flareHeight;
  const flareCutoff = params.has("flareCutoff")
    ? clamp(finiteNumber(Number(params.get("flareCutoff")), baseControls.flareCutoff), renderSliderRanges.flareCutoff.min, renderSliderRanges.flareCutoff.max)
    : baseControls.flareCutoff;
  const ribbonFraction = params.has("ribbonFraction")
    ? clamp(finiteNumber(Number(params.get("ribbonFraction")), baseControls.ribbonFraction), renderSliderRanges.ribbonFraction.min, renderSliderRanges.ribbonFraction.max)
    : baseControls.ribbonFraction;
  const ribbonWidth = params.has("ribbonWidth")
    ? clamp(finiteNumber(Number(params.get("ribbonWidth")), baseControls.ribbonWidth), renderSliderRanges.ribbonWidth.min, renderSliderRanges.ribbonWidth.max)
    : baseControls.ribbonWidth;
  const ribbonTaper = params.has("ribbonTaper")
    ? clamp(finiteNumber(Number(params.get("ribbonTaper")), baseControls.ribbonTaper), renderSliderRanges.ribbonTaper.min, renderSliderRanges.ribbonTaper.max)
    : baseControls.ribbonTaper;
  const ribbonLength = params.has("ribbonLength")
    ? clamp(finiteNumber(Number(params.get("ribbonLength")), baseControls.ribbonLength), renderSliderRanges.ribbonLength.min, renderSliderRanges.ribbonLength.max)
    : baseControls.ribbonLength;
  const ribbonJoints = params.has("ribbonJoints")
    ? clamp(finiteNumber(Number(params.get("ribbonJoints")), baseControls.ribbonJoints), renderSliderRanges.ribbonJoints.min, renderSliderRanges.ribbonJoints.max)
    : baseControls.ribbonJoints;
  const ribbonFadeStart = params.has("ribbonFadeStart")
    ? clamp(finiteNumber(Number(params.get("ribbonFadeStart")), baseControls.ribbonFadeStart), renderSliderRanges.ribbonFadeStart.min, renderSliderRanges.ribbonFadeStart.max)
    : baseControls.ribbonFadeStart;
  const ribbonEdgeFade = params.has("ribbonEdgeFade")
    ? clamp(finiteNumber(Number(params.get("ribbonEdgeFade")), baseControls.ribbonEdgeFade), renderSliderRanges.ribbonEdgeFade.min, renderSliderRanges.ribbonEdgeFade.max)
    : baseControls.ribbonEdgeFade;
  const dofEnabled = parseBooleanParam(params.get("dofEnabled") ?? params.get("dof")) ?? baseControls.dofEnabled;
  const sceneBrightness = params.has("sceneBrightness")
    ? clamp(finiteNumber(Number(params.get("sceneBrightness")), baseControls.sceneBrightness), 0, 2.6)
    : baseControls.sceneBrightness;
  const fogTemporal = parseBooleanParam(params.get("fogTemporal")) ?? baseControls.fogTemporal;
  const fogBrightness = params.has("fogBrightness")
    ? clamp(finiteNumber(Number(params.get("fogBrightness")), baseControls.fogBrightness), 0, 8)
    : baseControls.fogBrightness;
  const fogRenderScale = params.has("fogScale")
    ? clamp(finiteNumber(Number(params.get("fogScale")), baseControls.fogRenderScale), 0.25, 1)
    : baseControls.fogRenderScale;
  const fogStepScale = params.has("fogStepScale")
    ? clamp(finiteNumber(Number(params.get("fogStepScale")), baseControls.fogStepScale), 0.125, 1)
    : baseControls.fogStepScale;
  const fogTemporalBlend = params.has("fogBlend")
    ? clamp(finiteNumber(Number(params.get("fogBlend")), baseControls.fogTemporalBlend), 0, 0.96)
    : baseControls.fogTemporalBlend;
  const fogBlueNoise = parseBooleanParam(params.get("fogBlueNoise")) ?? baseControls.fogBlueNoise;
  const fieldTextureSampling = parseBooleanParam(params.get("fieldTexture")) ?? baseControls.fieldTextureSampling;
  const emptySpaceSkipping = parseBooleanParam(params.get("emptySkip")) ?? baseControls.emptySpaceSkipping;
  const emptySpaceThreshold = params.has("emptySkipThreshold")
    ? clamp(finiteNumber(Number(params.get("emptySkipThreshold")), baseControls.emptySpaceThreshold), 0, 0.2)
    : baseControls.emptySpaceThreshold;
  const emptySpaceStride = params.has("emptySkipStride")
    ? Math.round(clamp(finiteNumber(Number(params.get("emptySkipStride")), baseControls.emptySpaceStride), 1, 8))
    : baseControls.emptySpaceStride;
  const particleBlendMode = parseParticleBlendMode(params.get("particleBlend")) ?? baseControls.particleBlendMode;
  const particleColorMode = coerceParticleColorMode(params.get("particleColor")) ?? baseControls.particleColorMode;
  const particleBrightness = params.has("particleBrightness")
    ? clamp(finiteNumber(Number(params.get("particleBrightness")), baseControls.particleBrightness), 0, 8)
    : baseControls.particleBrightness;
  const particleOpacity = params.has("particleOpacity")
    ? clamp(finiteNumber(Number(params.get("particleOpacity")), baseControls.particleOpacity), 0, 1)
    : baseControls.particleOpacity;
  const particleVelocityStretch = parseBooleanParam(params.get("particleVelocityStretch")) ?? baseControls.particleVelocityStretch;
  const particleStretch = params.has("particleStretch")
    ? clamp(finiteNumber(Number(params.get("particleStretch")), baseControls.particleStretch), 0, 6)
    : baseControls.particleStretch;
  const particleStretchMin = params.has("particleStretchMin")
    ? Math.min(particleStretch, clamp(finiteNumber(Number(params.get("particleStretchMin")), baseControls.particleStretchMin), renderSliderRanges.particleStretchMin.min, renderSliderRanges.particleStretchMin.max))
    : Math.min(baseControls.particleStretchMin, particleStretch);
  const particleStretchSpeed = params.has("particleStretchSpeed")
    ? clamp(finiteNumber(Number(params.get("particleStretchSpeed")), baseControls.particleStretchSpeed), renderSliderRanges.particleStretchSpeed.min, renderSliderRanges.particleStretchSpeed.max)
    : baseControls.particleStretchSpeed;
  const particleSpeedCutoff = params.has("particleSpeedCutoff")
    ? clamp(finiteNumber(Number(params.get("particleSpeedCutoff")), baseControls.particleSpeedCutoff), renderSliderRanges.particleSpeedCutoff.min, renderSliderRanges.particleSpeedCutoff.max)
    : baseControls.particleSpeedCutoff;
  const particleSlowCutoff = params.has("particleSlowCutoff")
    ? clamp(finiteNumber(Number(params.get("particleSlowCutoff")), baseControls.particleSlowCutoff), renderSliderRanges.particleSlowCutoff.min, renderSliderRanges.particleSlowCutoff.max)
    : baseControls.particleSlowCutoff;
  const trailColorMode = parseTrailColorMode(params.get("trailColor")) ?? baseControls.trailColorMode;
  const fogTint = sanitizeHexColor(params.get("fogTint"), baseControls.fogTint);
  const particleTint = sanitizeHexColor(params.get("particleTint"), baseControls.particleTint);
  const particleDensityCutoff = params.has("particleCutoff")
    ? clamp(finiteNumber(Number(params.get("particleCutoff")), baseControls.particleDensityCutoff), renderSliderRanges.particleDensityCutoff.min, renderSliderRanges.particleDensityCutoff.max)
    : baseControls.particleDensityCutoff;
  const particleDensityRadius = params.has("particleCutoffRadius")
    ? clamp(finiteNumber(Number(params.get("particleCutoffRadius")), baseControls.particleDensityRadius), 0, 0.18)
    : baseControls.particleDensityRadius;
  const particleDensityNormalize = params.has("particleDensityNormalize")
    ? clamp(finiteNumber(Number(params.get("particleDensityNormalize")), baseControls.particleDensityNormalize), renderSliderRanges.particleDensityNormalize.min, renderSliderRanges.particleDensityNormalize.max)
    : baseControls.particleDensityNormalize;
  const particleDensitySoftness = params.has("particleDensitySoftness")
    ? clamp(finiteNumber(Number(params.get("particleDensitySoftness")), baseControls.particleDensitySoftness), renderSliderRanges.particleDensitySoftness.min, renderSliderRanges.particleDensitySoftness.max)
    : baseControls.particleDensitySoftness;
  const particleSupportMask = params.has("particleSupport")
    ? clamp(finiteNumber(Number(params.get("particleSupport")), baseControls.particleSupportMask), renderSliderRanges.particleSupportMask.min, renderSliderRanges.particleSupportMask.max)
    : baseControls.particleSupportMask;
  const particleSupportRadius = params.has("particleSupportRadius")
    ? clamp(finiteNumber(Number(params.get("particleSupportRadius")), baseControls.particleSupportRadius), renderSliderRanges.particleSupportRadius.min, renderSliderRanges.particleSupportRadius.max)
    : baseControls.particleSupportRadius;
  const particleSupportNeighbors = params.has("particleSupportNeighbors")
    ? clamp(finiteNumber(Number(params.get("particleSupportNeighbors")), baseControls.particleSupportNeighbors), renderSliderRanges.particleSupportNeighbors.min, renderSliderRanges.particleSupportNeighbors.max)
    : baseControls.particleSupportNeighbors;
  const particleSupportFlow = params.has("particleSupportFlow")
    ? clamp(finiteNumber(Number(params.get("particleSupportFlow")), baseControls.particleSupportFlow), renderSliderRanges.particleSupportFlow.min, renderSliderRanges.particleSupportFlow.max)
    : baseControls.particleSupportFlow;
  const fastParticleRender = parseBooleanParam(params.get("fastParticles") ?? params.get("fastParticleRender")) ?? baseControls.fastParticleRender;
  const fastNoBloomPost = parseBooleanParam(params.get("fastNoBloomPost") ?? params.get("fastPost")) ?? baseControls.fastNoBloomPost;
  const particleCutoffPrepass = parseBooleanParam(params.get("cutoffPrepass")) ?? baseControls.particleCutoffPrepass;
  const densityLargeHalfRes = parseBooleanParam(params.get("densityHalfLarge")) ?? baseControls.densityLargeHalfRes;
  const densityPassStrength = params.has("densityStrength")
    ? clamp(finiteNumber(Number(params.get("densityStrength")), baseControls.densityPassStrength), renderSliderRanges.densityPassStrength.min, renderSliderRanges.densityPassStrength.max)
    : baseControls.densityPassStrength;
  const densitySmallScale = params.has("densitySmall")
    ? clamp(finiteNumber(Number(params.get("densitySmall")), baseControls.densitySmallScale), 0.25, 5)
    : baseControls.densitySmallScale;
  const densityLargeScale = params.has("densityLarge")
    ? clamp(finiteNumber(Number(params.get("densityLarge")), baseControls.densityLargeScale), renderSliderRanges.densityLargeScale.min, renderSliderRanges.densityLargeScale.max)
    : baseControls.densityLargeScale;
  const densityLargeThreshold = params.has("densityMask")
    ? clamp(finiteNumber(Number(params.get("densityMask")), baseControls.densityLargeThreshold), renderSliderRanges.densityLargeThreshold.min, renderSliderRanges.densityLargeThreshold.max)
    : baseControls.densityLargeThreshold;
  const densityContrastGain = params.has("densityContrast")
    ? clamp(finiteNumber(Number(params.get("densityContrast")), baseControls.densityContrastGain), renderSliderRanges.densityContrastGain.min, renderSliderRanges.densityContrastGain.max)
    : baseControls.densityContrastGain;
  const densityContrastBalance = params.has("densityBalance")
    ? clamp(finiteNumber(Number(params.get("densityBalance")), baseControls.densityContrastBalance), 0, 1.5)
    : baseControls.densityContrastBalance;
  const densityEmissionPower = params.has("densityCurve")
    ? clamp(finiteNumber(Number(params.get("densityCurve")), baseControls.densityEmissionPower), 0.25, 5)
    : baseControls.densityEmissionPower;
  const densityOcclusion = params.has("densityOcclusion")
    ? clamp(finiteNumber(Number(params.get("densityOcclusion")), baseControls.densityOcclusion), 0, 1)
    : baseControls.densityOcclusion;
  const accumulationStrength = params.has("accumulationStrength")
    ? clamp(finiteNumber(Number(params.get("accumulationStrength")), baseControls.accumulationStrength), renderSliderRanges.accumulationStrength.min, renderSliderRanges.accumulationStrength.max)
    : baseControls.accumulationStrength;
  const accumulationRadius = params.has("accumulationRadius")
    ? clamp(finiteNumber(Number(params.get("accumulationRadius")), baseControls.accumulationRadius), renderSliderRanges.accumulationRadius.min, renderSliderRanges.accumulationRadius.max)
    : baseControls.accumulationRadius;
  const accumulationCurve = params.has("accumulationCurve")
    ? clamp(finiteNumber(Number(params.get("accumulationCurve")), baseControls.accumulationCurve), renderSliderRanges.accumulationCurve.min, renderSliderRanges.accumulationCurve.max)
    : baseControls.accumulationCurve;
  const accumulationMemory = params.has("accumulationMemory")
    ? clamp(finiteNumber(Number(params.get("accumulationMemory")), baseControls.accumulationMemory), renderSliderRanges.accumulationMemory.min, renderSliderRanges.accumulationMemory.max)
    : baseControls.accumulationMemory;
  const accumulationNoiseReject = params.has("accumulationReject")
    ? clamp(finiteNumber(Number(params.get("accumulationReject")), baseControls.accumulationNoiseReject), renderSliderRanges.accumulationNoiseReject.min, renderSliderRanges.accumulationNoiseReject.max)
    : baseControls.accumulationNoiseReject;
  const probeControls = preset.id === "trail-probe"
    ? {
        particleSizePx: 7.5,
        particleMinPx: 1.2,
        particleMaxPx: 18,
        particleOpacity: 0.95,
        trailOpacity: 1.45,
        trailThreshold: 0,
        density: 1.65,
        raySteps: 64
      }
    : {};
  const config = normalizeTrailKernel({
    ...preset.config,
    simulationSpeed,
    particleCount,
    width: volumeSize,
    height: volumeSize,
    depth: volumeSize
  });
  const controls = normalizeRenderControlsForDisplayMode("live", {
    ...baseControls,
    ...probeControls,
    rayResolution,
    raySteps,
    bloomStrength,
    chromaticAberration,
    vignetteStrength,
    vignetteSoftness,
    streakStrength,
    streakLength,
    streakVertical,
    flareHeight,
    flareCutoff,
    ribbonFraction,
    ribbonWidth,
    ribbonTaper,
    ribbonLength,
    ribbonJoints,
    ribbonFadeStart,
    ribbonEdgeFade,
    bloomThreshold,
    bloomRadius,
    colorSaturation,
    colorContrast,
    dofEnabled,
    sceneBrightness,
    fogTemporal,
    fogBrightness,
    fogRenderScale,
    fogStepScale,
    fogTemporalBlend,
    fogBlueNoise,
    fieldTextureSampling,
    emptySpaceSkipping,
    emptySpaceThreshold,
    emptySpaceStride,
    particleBlendMode,
    particleColorMode,
    particleBrightness,
    particleOpacity,
    particleVelocityStretch,
    particleStretch,
    particleStretchMin,
    particleStretchSpeed,
    particleSpeedCutoff,
    particleSlowCutoff,
    trailColorMode,
    fogTint,
    particleTint,
    particleDensityCutoff,
    particleDensityRadius,
    particleDensityNormalize,
    particleDensitySoftness,
    particleSupportMask,
    particleSupportRadius,
    particleSupportNeighbors,
    particleSupportFlow,
    fastParticleRender,
    fastNoBloomPost,
    particleCutoffPrepass,
    densityLargeHalfRes,
    densityPassStrength,
    densitySmallScale,
    densityLargeScale,
    densityLargeThreshold,
    densityContrastGain,
    densityContrastBalance,
    densityEmissionPower,
    densityOcclusion,
    accumulationStrength,
    accumulationRadius,
    accumulationCurve,
    accumulationMemory,
    accumulationNoiseReject,
    renderLayer: "particles"
  });
  return {
    presetId: preset.id,
    config,
    controls
  };
}

function normalizeRenderControlsForDisplayMode(_displayMode: DisplayMode, controls: RenderControls): RenderControls {
  if (controls.renderLayer === "particles" && controls.ribbonFraction === 0) return controls;
  return { ...controls, renderLayer: "particles", ribbonFraction: 0 };
}

function shouldEnableGpuProfiling(params: URLSearchParams): boolean {
  return params.get("profileGpu") === "1" && (params.get("traceGpu") === "1" || params.get("debugGpu") === "1");
}

function parseParticleBlendMode(value: string | null): RenderControls["particleBlendMode"] | null {
  if (value === "additive" || value === "alpha" || value === "opaque") return value;
  return null;
}

function parseTrailColorMode(value: string | null): RenderControls["trailColorMode"] | null {
  if (value === "stable" || value === "flow" || value === "thermal" || value === "tint") return value;
  return null;
}

function coerceParticleColorMode(value: unknown): RenderControls["particleColorMode"] | null {
  return particleColorModes.includes(value as RenderControls["particleColorMode"])
    ? value as RenderControls["particleColorMode"]
    : null;
}

function sanitizeHexColor(value: unknown, fallback: string): string {
  return typeof value === "string" && /^#[0-9a-f]{6}$/i.test(value) ? value.toLowerCase() : fallback;
}

function parseBooleanParam(value: string | null): boolean | null {
  if (value === "1" || value === "true") return true;
  if (value === "0" || value === "false") return false;
  return null;
}

function damp(current: number, target: number, amount: number): number {
  const next = current + (target - current) * amount;
  return Math.abs(next - target) < 0.0001 ? target : next;
}

function dampAngle(current: number, target: number, amount: number): number {
  const delta = normalizeAngle(target - current);
  const next = current + delta * amount;
  return Math.abs(delta) < 0.0001 ? target : normalizeAngle(next);
}

function normalizeAngle(value: number): number {
  const fullTurn = Math.PI * 2;
  return ((((value + Math.PI) % fullTurn) + fullTurn) % fullTurn) - Math.PI;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function clampCameraDistance(value: number): number {
  return clamp(Number.isFinite(value) ? value : cameraDefaults.distance, minCameraDistance, maxCameraDistance);
}

function clampParticleCount(value: number): number {
  const clamped = clamp(Number.isFinite(value) ? value : minLiveParticles, minLiveParticles, maxLiveParticles);
  return clamp(Math.round(clamped / liveParticleStep) * liveParticleStep, minLiveParticles, maxLiveParticles);
}

function clampVolumeSize(value: number): number {
  const clamped = clamp(Number.isFinite(value) ? value : minVolumeSize, minVolumeSize, maxVolumeSize);
  return Math.round(clamped / volumeStep) * volumeStep;
}

function clampRaySteps(value: number): number {
  const clamped = clamp(Number.isFinite(value) ? value : defaultControls.raySteps, 8, 512);
  return Math.round(clamped / 8) * 8;
}

function clampSimulationSpeed(value: number): number {
  return clamp(Number.isFinite(value) ? value : defaultLiveGpu3dConfig.simulationSpeed, 0, MAX_SIMULATION_SPEED);
}

function randomSliderValue(range: SliderRange): number {
  const steps = Math.round((range.max - range.min) / range.step);
  const value = range.min + Math.floor(Math.random() * (steps + 1)) * range.step;
  return Number(clamp(value, range.min, range.max).toFixed(stepPrecision(range.step)));
}

function randomBoolean(): boolean {
  return Math.random() < 0.5;
}

function randomChoice<T>(values: readonly T[]): T {
  return values[Math.floor(Math.random() * values.length)];
}

function stepPrecision(step: number): number {
  const text = String(step);
  const decimalIndex = text.indexOf(".");
  return decimalIndex === -1 ? 0 : text.length - decimalIndex - 1;
}

function normalizeTrailKernel(config: LiveGpu3dConfig): LiveGpu3dConfig {
  const radius = clamp(config.depositRadius, minTrailRadius, maxSupportedTrailRadius(config));
  return {
    ...config,
    depositRadius: radius,
    sigma: sigmaForTrailRadius(radius, config.width),
    depositTapRadius: autoDepositTapRadius(radius, config.width, config.particleCount)
  };
}

function sigmaForTrailRadius(radius: number, volumeSize: number): number {
  return Math.max(0.00075, radius * 0.34);
}

function autoDepositTapRadius(radius: number, volumeSize: number, particleCount: number): number {
  const voxelSize = 2 / Math.max(1, volumeSize);
  return clamp(Math.ceil(radius / voxelSize), 1, maxDepositTapRadius(particleCount));
}

function maxDepositTapRadius(particleCount: number): number {
  if (particleCount >= 262144) return 1;
  if (particleCount >= 65536) return 2;
  if (particleCount >= 16384) return 3;
  return 6;
}

function maxSupportedTrailRadius(config: Pick<LiveGpu3dConfig, "width" | "particleCount">): number {
  const voxelSize = 2 / Math.max(1, config.width);
  return Math.max(minTrailRadius, Math.min(maxTrailRadiusCap, (maxDepositTapRadius(config.particleCount) + 0.5) * voxelSize));
}

function recommendedVolumeSize(particleCount: number, current: number): number {
  if (particleCount >= 1048576) return Math.max(current, 64);
  if (particleCount >= 524288) return Math.max(current, 56);
  if (particleCount >= 262144) return Math.max(current, 48);
  if (particleCount >= 131072) return Math.max(current, 40);
  return Math.max(current, minVolumeSize);
}
