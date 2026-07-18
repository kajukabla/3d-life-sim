import { useCallback, useEffect, useRef, useState, type MutableRefObject } from "react";
import type { RenderControls } from "./renderControls";
import {
  defaultLiveGpu3dConfig,
  type LiveGpu3dConfig,
  type LiveGpu3dDiagnostics,
  RealtimeGpuSim3d
} from "./realtimeGpuSim3d";
import {
  analyzeSimSpeedSamples,
  type SimSpeedLabReport,
  type SimSpeedLabSample,
  type SimSpeedLabSnapshot
} from "./simSpeedLab";

declare global {
  interface Window {
    __lifesimSimSpeedLab?: () => SimSpeedLabSnapshot;
  }
}

type LabLayer = "particles" | "trails";

const speedPresets = [1, 0.95, 0.55, 0.4, 0.05];
const maxSamples = 720;

const emptyReport = analyzeSimSpeedSamples([]);

export function SimSpeedDebugLab() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rendererRef = useRef<RealtimeGpuSim3d | null>(null);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);
  const previousPixelsRef = useRef<Uint8ClampedArray | null>(null);
  const samplesRef = useRef<SimSpeedLabSample[]>([]);
  const snapshotRef = useRef<SimSpeedLabSnapshot>({
    speed: initialSpeed(),
    scene: "straight-line-particles",
    layer: "particles",
    report: emptyReport,
    samples: []
  });
  const speedRef = useRef(snapshotRef.current.speed);
  const layerRef = useRef<LabLayer>("particles");
  const resetRequestedRef = useRef(false);
  const [speed, setSpeedState] = useState(speedRef.current);
  const [layer, setLayerState] = useState<LabLayer>("particles");
  const [snapshot, setSnapshot] = useState(snapshotRef.current);

  const publishSnapshot = useCallback((report: SimSpeedLabReport, latest?: SimSpeedLabSample) => {
    const next: SimSpeedLabSnapshot = {
      speed: speedRef.current,
      scene: "straight-line-particles",
      layer: layerRef.current,
      latest,
      report,
      samples: [...samplesRef.current]
    };
    snapshotRef.current = next;
    setSnapshot(next);
  }, []);

  const clearSamples = useCallback(() => {
    samplesRef.current = [];
    previousPixelsRef.current = null;
    publishSnapshot(emptyReport);
  }, [publishSnapshot]);

  const resetSimulation = useCallback(() => {
    clearSamples();
    resetRequestedRef.current = true;
  }, [clearSamples]);

  const setSpeed = useCallback((nextSpeed: number) => {
    speedRef.current = nextSpeed;
    setSpeedState(nextSpeed);
    clearSamples();
  }, [clearSamples]);

  const setLayer = useCallback((nextLayer: LabLayer) => {
    layerRef.current = nextLayer;
    setLayerState(nextLayer);
    clearSamples();
  }, [clearSamples]);

  useEffect(() => {
    window.__lifesimSimSpeedLab = () => snapshotRef.current;
    window.__lifesimDiagnostics = () => ({
      mode: "sim-speed-lab",
      lab: snapshotRef.current,
      renderer: snapshotRef.current.latest,
      conformance: {
        simSpeedLab: true,
        sampleCount: snapshotRef.current.report.sampleCount,
        flickerEvents: snapshotRef.current.report.flickerEvents
      }
    });
    return () => {
      delete window.__lifesimSimSpeedLab;
      delete window.__lifesimDiagnostics;
    };
  }, []);

  useEffect(() => {
    let active = true;
    let frameHandle = 0;
    const renderer = new RealtimeGpuSim3d();
    rendererRef.current = renderer;

    const tick = async () => {
      if (!active || !canvasRef.current) return;
      try {
        if (resetRequestedRef.current) {
          resetRequestedRef.current = false;
          await renderer.reset();
        }
        const diagnostics = await renderer.render(
          canvasRef.current,
          createLabControls(layerRef.current),
          false,
          true,
          createLabConfig(speedRef.current),
          false
        );
        const sample = buildSample(canvasRef.current, diagnostics, samplesRef.current.at(-1), offscreenRef, previousPixelsRef);
        samplesRef.current.push(sample);
        if (samplesRef.current.length > maxSamples) {
          samplesRef.current.splice(0, samplesRef.current.length - maxSamples);
        }
        if (sample.frame % 6 === 0) {
          publishSnapshot(analyzeSimSpeedSamples(samplesRef.current), sample);
        }
      } catch (error) {
        const nextSnapshot = {
          ...snapshotRef.current,
          report: {
            ...snapshotRef.current.report,
            renderErrorEvents: snapshotRef.current.report.renderErrorEvents + 1,
            events: [
              ...snapshotRef.current.report.events,
              {
                frame: snapshotRef.current.latest?.frame ?? 0,
                timestep: snapshotRef.current.latest?.timestep ?? 0,
                kind: "render-error" as const,
                value: -1,
                renderLerpT: snapshotRef.current.latest?.renderLerpT ?? 0
              }
            ].slice(-64)
          }
        };
        snapshotRef.current = nextSnapshot;
        if (snapshotRef.current.report.renderErrorEvents % 30 === 1) setSnapshot(nextSnapshot);
        console.warn("sim speed lab render failed", error);
      } finally {
        if (active) frameHandle = requestAnimationFrame(tick);
      }
    };

    frameHandle = requestAnimationFrame(tick);
    return () => {
      active = false;
      cancelAnimationFrame(frameHandle);
      void renderer.reset();
      rendererRef.current = null;
    };
  }, [publishSnapshot]);

  const latest = snapshot.latest;
  const report = snapshot.report;

  return (
    <main className="sim-speed-lab">
      <section className="sim-speed-lab__stage">
        <canvas ref={canvasRef} data-testid="sim-speed-lab-canvas" />
      </section>

      <aside className="sim-speed-lab__panel" data-testid="sim-speed-lab-panel">
        <div className="sim-speed-lab__header">
          <div>
            <span>Sim Speed Lab</span>
            <strong>{speed.toFixed(2)}x</strong>
          </div>
          <button type="button" onClick={resetSimulation}>Reset</button>
        </div>

        <div className="sim-speed-lab__controls">
          <div className="sim-speed-lab__speed-row">
            {speedPresets.map((preset) => (
              <button
                key={preset}
                type="button"
                className={Math.abs(speed - preset) < 0.0001 ? "is-active" : ""}
                onClick={() => setSpeed(preset)}
              >
                {preset.toFixed(preset < 1 ? 2 : 0)}
              </button>
            ))}
          </div>
          <label>
            <span>Speed</span>
            <input
              data-testid="sim-speed-lab-speed"
              type="range"
              min="0"
              max="1.25"
              step="0.01"
              value={speed}
              onChange={(event) => setSpeed(Number(event.currentTarget.value))}
            />
          </label>
          <label>
            <span>Layer</span>
            <select value={layer} onChange={(event) => setLayer(event.currentTarget.value as LabLayer)}>
              <option value="particles">Particles</option>
              <option value="trails">Trails</option>
            </select>
          </label>
        </div>

        <div className="sim-speed-lab__metrics">
          <Metric label="Frames" value={String(report.frameSpan)} />
          <Metric label="Ticks" value={String(report.tickSpan)} />
          <Metric label="Lerp" value={latest ? latest.renderLerpT.toFixed(3) : "..."} />
          <Metric label="Step" value={latest ? latest.simulationTimeAdvance.toFixed(2) : "..."} />
          <Metric label="Stats" value={String(report.statsReadEvents)} />
          <Metric label="Errors" value={String(report.renderErrorEvents)} />
          <Metric label="Flicker" value={String(report.flickerEvents)} />
          <Metric label="P95 Luma" value={report.p95AbsLumaDelta.toFixed(3)} />
          <Metric label="Max Luma" value={report.maxAbsLumaDelta.toFixed(3)} />
          <Metric label="P95 Frame" value={`${report.p95FrameTimeMs.toFixed(2)}ms`} />
          <Metric label="Max Frame" value={`${report.maxFrameTimeMs.toFixed(2)}ms`} />
          <Metric label="Cadence" value={report.likelyCadenceFrames == null ? "none" : `${report.likelyCadenceFrames}f`} />
          <Metric label="Phase" value={report.phaseResetEvents.toString()} />
        </div>

        <LumaSparkline samples={snapshot.samples} />

        <div className="sim-speed-lab__log" data-testid="sim-speed-lab-events">
          {report.events.length === 0 ? (
            <span>No flagged events</span>
          ) : report.events.slice(-12).map((event, index) => (
            <span key={`${event.kind}-${event.frame}-${index}`}>
              {event.kind} f{event.frame} t{event.timestep} v{event.value.toFixed(3)} r{event.renderLerpT.toFixed(2)}
            </span>
          ))}
        </div>
      </aside>
    </main>
  );
}

function Metric(props: { label: string; value: string }) {
  return (
    <div>
      <span>{props.label}</span>
      <strong>{props.value}</strong>
    </div>
  );
}

function LumaSparkline(props: { samples: readonly SimSpeedLabSample[] }) {
  const values = props.samples.slice(-120).map((sample) => Math.abs(sample.lumaDelta));
  const max = Math.max(0.001, ...values);
  const points = values.map((value, index) => {
    const x = values.length <= 1 ? 0 : (index / (values.length - 1)) * 100;
    const y = 34 - (value / max) * 30;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");
  return (
    <svg className="sim-speed-lab__sparkline" viewBox="0 0 100 36" preserveAspectRatio="none" aria-hidden="true">
      <polyline points={points} />
    </svg>
  );
}

function createLabConfig(speed: number): LiveGpu3dConfig {
  return {
    ...defaultLiveGpu3dConfig,
    seed: 3405691582,
    simulationSpeed: speed,
    width: 32,
    height: 32,
    depth: 32,
    particleCount: 64,
    dt: 1,
    sensorGain: 0,
    sensorAngle: 0,
    sensorDistance: 1,
    mutationScale: 0,
    globalForceMult: 0,
    drag: 1,
    strafePower: 0,
    axialForce: 0,
    lateralForce: 0,
    hazardRate: 0,
    trailPersistence: 0.992,
    trailDiffusion: 0.02,
    depositRadius: 0.052,
    depositTapRadius: 1,
    depositMass: 2.2,
    sigma: 0.016,
    cohorts: 8,
    hueSensitivity: 0.25,
    colorByCohort: true,
    symmetryAxes: 2,
    absoluteOrientation: 0,
    orientationMix: 0,
    initialConditions: 3,
    boundaryMode: 2,
    rule: [],
    recycleCutoff: 0,
    recycleEnabled: false
  };
}

function createLabControls(layer: LabLayer): RenderControls {
  return {
    density: 1,
    exposure: 1,
    fov: 45,
    aperture: 0,
    focusDistance: 2,
    dofBlur: 0,
    dofEnabled: false,
    dofDebug: false,
    sceneBrightness: 1.4,
    raySteps: 32,
    rayResolution: 0.5,
    fogTemporal: false,
    fogRenderScale: 0.5,
    fogStepScale: 0.5,
    fogTemporalBlend: 0,
    fogBlueNoise: false,
    fieldTextureSampling: false,
    emptySpaceSkipping: false,
    emptySpaceThreshold: 0,
    emptySpaceStride: 1,
    particleSizePx: 5.5,
    particleMinPx: 1,
    particleMaxPx: 8,
    particleOpacity: 0.58,
    particleBrightness: 1.6,
    particleColorMode: "gradient-rainbow",
    particleVelocityStretch: false,
    particleStretch: 0.05,
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
    particleDensityRadius: 0,
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
    densityPassStrength: 0,
    densitySmallScale: 1,
    densityLargeScale: 1,
    densityLargeThreshold: 0.35,
    densityContrastGain: 0,
    densityContrastBalance: 0.5,
    densityEmissionPower: 1,
    densityOcclusion: 0,
    accumulationStrength: 0,
    accumulationRadius: 1,
    accumulationCurve: 1,
    accumulationMemory: 0,
    accumulationNoiseReject: 0,
    trailOpacity: 1,
    fogBrightness: 1.2,
    trailThreshold: 0,
    trailColorMode: "flow",
    fogTint: "#ffffff",
    particleTint: "#ffffff",
    renderLayer: layer,
    palette: "spectral",
    filament: 0,
    bloomStrength: 0,
    bloomThreshold: 1,
    bloomRadius: 0,
    colorSaturation: 1,
    colorContrast: 1,
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
    cameraYaw: 0,
    cameraPitch: 0,
    cameraDistance: 2.25,
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
}

function buildSample(
  canvas: HTMLCanvasElement,
  diagnostics: LiveGpu3dDiagnostics,
  previous: SimSpeedLabSample | undefined,
  offscreenRef: MutableRefObject<HTMLCanvasElement | null>,
  previousPixelsRef: MutableRefObject<Uint8ClampedArray | null>
): SimSpeedLabSample {
  const luma = sampleCanvasLuma(canvas, offscreenRef, previousPixelsRef);
  return {
    frame: diagnostics.renderFrame,
    timestep: diagnostics.timestep,
    simulationTime: diagnostics.simulationTime,
    simulationTimeAdvance: diagnostics.simulationTimeAdvance,
    renderLerpT: diagnostics.renderLerpT,
    statsReadMs: diagnostics.timings.statsReadMs,
    statsReadPending: diagnostics.statsReadPending,
    frameTimeMs: diagnostics.frameTimeMs,
    cpuFrameTimeMs: diagnostics.cpuFrameTimeMs,
    renderErrorEvents: 0,
    meanLuma: luma.meanLuma,
    lumaDelta: previous ? luma.meanLuma - previous.meanLuma : 0,
    meanChannelDelta: luma.meanChannelDelta,
    maxChannelDelta: luma.maxChannelDelta,
    stepScales: diagnostics.simulationStepScales
  };
}

function sampleCanvasLuma(
  canvas: HTMLCanvasElement,
  offscreenRef: MutableRefObject<HTMLCanvasElement | null>,
  previousPixelsRef: MutableRefObject<Uint8ClampedArray | null>
): { meanLuma: number; meanChannelDelta: number; maxChannelDelta: number } {
  const offscreen = offscreenRef.current ?? document.createElement("canvas");
  offscreenRef.current = offscreen;
  offscreen.width = 96;
  offscreen.height = 72;
  const context = offscreen.getContext("2d", { willReadFrequently: true });
  if (!context) return { meanLuma: 0, meanChannelDelta: 0, maxChannelDelta: 0 };
  context.drawImage(canvas, 0, 0, offscreen.width, offscreen.height);
  const pixels = context.getImageData(0, 0, offscreen.width, offscreen.height).data;
  let lumaSum = 0;
  let channelDeltaSum = 0;
  let maxChannelDelta = 0;
  const previous = previousPixelsRef.current;
  for (let i = 0; i < pixels.length; i += 4) {
    lumaSum += 0.2126 * pixels[i] + 0.7152 * pixels[i + 1] + 0.0722 * pixels[i + 2];
    if (previous) {
      const dr = Math.abs(pixels[i] - previous[i]);
      const dg = Math.abs(pixels[i + 1] - previous[i + 1]);
      const db = Math.abs(pixels[i + 2] - previous[i + 2]);
      channelDeltaSum += (dr + dg + db) / 3;
      maxChannelDelta = Math.max(maxChannelDelta, dr, dg, db);
    }
  }
  previousPixelsRef.current = new Uint8ClampedArray(pixels);
  return {
    meanLuma: lumaSum / (pixels.length / 4),
    meanChannelDelta: channelDeltaSum / (pixels.length / 4),
    maxChannelDelta
  };
}

function initialSpeed(): number {
  const value = Number(new URLSearchParams(window.location.search).get("speed"));
  return Number.isFinite(value) ? Math.max(0, Math.min(1.25, value)) : 0.95;
}
