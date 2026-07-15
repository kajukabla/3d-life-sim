import type { ParticleColorMode } from "../cacheRenderer";

// RGB triple, each channel 0..255 (integer after final packing).
export type RGB = [number, number, number];

export type LightingPalette = "aurora" | "ember" | "spectral";

// The current on-screen color settings the lights follow. Mirrors the fields the
// Lighting Editor reads from RenderControls so lights scrub the selected gradient.
export type LightingColorSource = {
  mode: ParticleColorMode;
  palette: LightingPalette;
  tint: string; // hex "#rrggbb", used only for the "solid" mode
};

export type LightConfig = {
  channel: number; // entertainment-area channel index (0..)
  name: string;
  enabled: boolean;
};

export type StrobeConfig = {
  enabled: boolean;
  hz: number; // flashes per second
  duty: number; // 0..1 fraction of each cycle that is "on"
};

// How the scrub position folds back when it runs past the end of the gradient.
// "repeat" cuts from the end color straight to the start (only seamless for
// cyclic gradients like rainbow); "pingpong" bounces end->start->end so any
// gradient stays continuous.
export type LoopMode = "repeat" | "pingpong";

// All the live, MIDI-mappable lighting parameters. Both spreads at 0 -> every
// light samples the identical gradient point (fully synced).
export type LightingState = {
  scrub: number; // 0..1 manual/MIDI position into the gradient
  scrubRateHz: number; // global auto-advance of the scrub position
  phaseSpread: number; // 0 = aligned; >0 gives each light a static offset
  freqSpread: number; // Hz; 0 = same rate; >0 gives each light its own rate (drifts over time)
  // Declarative sync lock. When true, EVERY light samples the exact same gradient
  // point regardless of the spread values — a robust guarantee that the colors
  // stay linked and can never drift out of sync. Uncheck to allow spread effects.
  locked: boolean;
  loopMode: LoopMode;
  master: number; // 0..1 engine master brightness (with gamma)
  gamma: number; // >0 perceptual curve on output
  brightness: number; // 0..1 final overall dimmer applied last, to every bulb
  strobe: StrobeConfig;
};

export function defaultLightingState(): LightingState {
  return {
    scrub: 0,
    scrubRateHz: 0,
    phaseSpread: 0,
    freqSpread: 0,
    locked: true,
    loopMode: "pingpong",
    master: 1,
    gamma: 1,
    brightness: 1,
    strobe: { enabled: false, hz: 8, duty: 0.5 }
  };
}

// The calm state the lights fall into when idle/demo mode engages, so unattended
// bulbs never sit in a harsh state: every light at its own point in the gradient
// (fully out of phase), the slowest gentle sweep, ping-pong so it never cuts, and
// strobe off. Brightness/gamma are preserved.
export function demoLightingState(prev: LightingState): LightingState {
  return {
    ...prev,
    scrub: 0,
    scrubRateHz: 0.01,
    phaseSpread: 1,
    freqSpread: 0.03,
    locked: false, // idle demo intentionally spreads the lights out of phase
    loopMode: "pingpong",
    strobe: { ...prev.strobe, enabled: false }
  };
}
