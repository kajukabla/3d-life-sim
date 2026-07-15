import { sampleLightingGradient } from "./gradient";
import type { LightConfig, LightingColorSource, LightingState, LoopMode, RGB, StrobeConfig } from "./types";

// Pure per-frame color computation for the Hue lights. Given the live lighting
// state, the color source the lights follow, the per-light configs, and the
// current time, returns one RGB (0..255) per light. No side effects, no clock.
export function computeLightFrame(
  state: LightingState,
  source: LightingColorSource,
  lights: LightConfig[],
  timeSec: number
): RGB[] {
  const strobeBlack = state.strobe.enabled && !strobeOn(state.strobe, timeSec);
  return lights.map((light) => {
    if (!light.enabled || strobeBlack) return [0, 0, 0];
    const raw = lightPhase(state, light.channel, timeSec);
    const base = sampleLightingGradient(source, foldPhase(raw, state.loopMode));
    return applyMasterGamma(base, state.master * state.brightness, state.gamma);
  });
}

// The raw (unwrapped) phase accumulator for light `channel` at `timeSec`.
// raw = scrub + scrubRate*t + phaseSpread*h(ch) + freqSpread*h2(ch)*t.
// Both spreads 0 -> identical for every channel (fully synced). Folding into a
// gradient position happens in foldPhase so ping-pong can bounce at the ends.
export function lightPhase(state: LightingState, channel: number, timeSec: number): number {
  const t = Number.isFinite(timeSec) ? timeSec : 0;
  // Locked -> spreads are forced to 0 so every channel resolves to the identical
  // phase: a hard guarantee the lights stay in sync, independent of slider drift.
  const phaseSpread = state.locked ? 0 : state.phaseSpread;
  const freqSpread = state.locked ? 0 : state.freqSpread;
  const offset = phaseSpread * hash01(channel, 0x1);
  const extraFreq = freqSpread * hash01(channel, 0x2);
  return state.scrub + state.scrubRateHz * t + offset + extraFreq * t;
}

// Map the raw phase to a gradient position in [0,1]. "repeat" wraps (sawtooth,
// cutting end->start); "pingpong" mirrors (triangle) so the sweep bounces
// end->start->end and any gradient stays continuous across the loop.
export function foldPhase(raw: number, mode: LoopMode): number {
  if (!Number.isFinite(raw)) return 0;
  if (mode === "pingpong") {
    const f = wrap01(raw / 2);
    return 1 - Math.abs(2 * f - 1);
  }
  return wrap01(raw);
}

function strobeOn(strobe: StrobeConfig, timeSec: number): boolean {
  if (strobe.hz <= 0) return true;
  const duty = clamp(strobe.duty, 0, 1);
  return wrap01(timeSec * strobe.hz) < duty;
}

function applyMasterGamma(color: RGB, master: number, gamma: number): RGB {
  const m = clamp(master, 0, 1);
  const g = gamma > 0 ? gamma : 1;
  return [
    scale(color[0], m, g),
    scale(color[1], m, g),
    scale(color[2], m, g)
  ];
}

function scale(channel: number, master: number, gamma: number): number {
  const n = clamp(channel / 255, 0, 1);
  return Math.round(255 * master * Math.pow(n, gamma));
}

// Deterministic per-channel hash in [0,1). Stable across frames so spreads don't
// shimmer; uses no Math.random.
function hash01(channel: number, salt: number): number {
  let x = Math.imul((channel | 0) ^ salt ^ 0x9e3779b1, 0x85ebca6b);
  x ^= x >>> 13;
  x = Math.imul(x, 0xc2b2ae35);
  x ^= x >>> 16;
  return (x >>> 0) / 4294967296;
}

function wrap01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return ((value % 1) + 1) % 1;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, Number.isFinite(value) ? value : min));
}
