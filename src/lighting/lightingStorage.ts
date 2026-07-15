import { defaultLightingState, type LightingState } from "./types";

const STORAGE_KEY = "fluoddity-3d.lighting.v1";

export function sanitizeLightingState(value: unknown): LightingState {
  const base = defaultLightingState();
  if (!isRecord(value)) return base;
  const strobe = isRecord(value.strobe) ? value.strobe : {};
  return {
    scrub: clamp(num(value.scrub, base.scrub), 0, 1),
    scrubRateHz: clamp(num(value.scrubRateHz, base.scrubRateHz), -4, 4),
    phaseSpread: clamp(num(value.phaseSpread, base.phaseSpread), 0, 1),
    freqSpread: clamp(num(value.freqSpread, base.freqSpread), 0, 8),
    locked: typeof value.locked === "boolean" ? value.locked : true,
    loopMode: value.loopMode === "repeat" ? "repeat" : "pingpong",
    master: clamp(num(value.master, base.master), 0, 1),
    gamma: clamp(num(value.gamma, base.gamma), 0.1, 4),
    brightness: clamp(num(value.brightness, base.brightness), 0, 1),
    strobe: {
      enabled: typeof strobe.enabled === "boolean" ? strobe.enabled : base.strobe.enabled,
      hz: clamp(num(strobe.hz, base.strobe.hz), 0.1, 60),
      duty: clamp(num(strobe.duty, base.strobe.duty), 0.01, 1)
    }
  };
}

export function loadLightingState(): LightingState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? sanitizeLightingState(JSON.parse(raw)) : defaultLightingState();
  } catch {
    return defaultLightingState();
  }
}

export function saveLightingState(state: LightingState): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sanitizeLightingState(state)));
  } catch {
    // storage unavailable (private mode) — non-fatal
  }
}

function num(value: unknown, fallback: number): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
