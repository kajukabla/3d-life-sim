// Records a live performance so it can be re-rendered deterministically offline (see
// tools/hdr-export). A browser cannot encode HDR video from a live canvas, and reading back
// full-res HDR floats every frame would stall the GPU and wreck a music-synced take. So we
// capture only the *performance as data* — cheap numbers, zero framerate cost — and replay it
// later through the same renderer with no real-time clock, reading HDR floats at leisure.
//
// What makes that replay faithful: the sim is a pure function of (seed, timestep, config
// history), so reset() + advanceSteps(config) reproduces it bit-for-bit on the same GPU. The
// only non-sim inputs that change the picture are the render controls (camera baked in via
// smoothedRenderControls), the live config, and the audio levels — so we snapshot all three
// per rendered frame. Wall-clock timestamps let the exporter resample our variable live
// cadence to a constant-fps video that stays in sync with the music. Mirrors cameraRecorder.ts.

import type { RenderControls } from "./cacheRenderer";
import type { LiveGpu3dConfig } from "./realtimeGpuSim3d";

export const AUTOMATION_RECORDING_VERSION = 1;

export type AutomationFrame = {
  // Milliseconds since recording start (wall clock). Drives constant-fps resampling so the
  // exported video matches real-time duration even when live frames were dropped.
  t: number;
  // Sim timestep reached at this frame. Replay steps the sim by the delta to the previous frame
  // (which may be 0 when paused or >1 when simulationSpeed > 1), reproducing the exact evolution.
  timestep: number;
  // Smoothed render controls actually played this frame — camera pose is already baked in by
  // smoothedRenderControls(), so replay needs no separate camera handling.
  controls: RenderControls;
  // Live sim/render config in effect this frame (includes audio/MIDI-modulated live.* values).
  config: LiveGpu3dConfig;
  // Audio band levels fed to the render uniforms this frame (visual modulation, not sim physics).
  audio: { low: number; mid: number; high: number };
};

export type AutomationHeader = {
  version: number;
  // Canvas backing-store size at record time (informational; the exporter sets its own res).
  width: number;
  height: number;
};

export type AutomationRecording = {
  header: AutomationHeader;
  frames: AutomationFrame[];
};

export type AutomationRecorder = {
  start(header: AutomationHeader, nowMs: number): void;
  record(frame: Omit<AutomationFrame, "t">, nowMs: number): void;
  stop(): AutomationRecording | null;
  isRecording(): boolean;
  frameCount(): number;
};

// A live/demo run can stream frames indefinitely. Cap the buffer so a forgotten recording can't
// grow without bound (~one flat controls+config record per frame). 200k frames is ~55 min at
// 60fps — far beyond any realistic take, and dropping the oldest keeps the most recent footage.
const MAX_RECORDED_FRAMES = 200000;

export function createAutomationRecorder(): AutomationRecorder {
  let header: AutomationHeader | null = null;
  let frames: AutomationFrame[] = [];
  let startMs = 0;
  let recording = false;

  return {
    start(nextHeader, nowMs) {
      header = { ...nextHeader, version: AUTOMATION_RECORDING_VERSION };
      frames = [];
      startMs = nowMs;
      recording = true;
    },
    record(frame, nowMs) {
      if (!recording) return;
      frames.push({
        t: nowMs - startMs,
        timestep: frame.timestep,
        // Defensive copies: the loop reuses/mutates these objects across frames.
        controls: { ...frame.controls },
        config: { ...frame.config },
        audio: { ...frame.audio }
      });
      if (frames.length > MAX_RECORDED_FRAMES) frames.shift();
    },
    stop() {
      recording = false;
      if (!header || frames.length === 0) return null;
      return { header, frames };
    },
    isRecording() {
      return recording;
    },
    frameCount() {
      return frames.length;
    }
  };
}
