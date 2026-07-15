import type { ParticleColorMode } from "./cacheRenderer";

export type RGB = [number, number, number];

export type ParticleGradientKey =
  | "solid"
  | "gradient-inferno"
  | "gradient-magma"
  | "gradient-viridis"
  | "gradient-turbo"
  | "gradient-rainbow"
  | "gradient-spectral"
  | "gradient-plasma"
  | "gradient-cosmic"
  | "gradient-ice"
  | "gradient-ember"
  | "velocity"
  | "cohort"
  | "audio";

// Keep this table aligned with particleColorModeToUniform() and
// particle_gradient_color() in realtimeGpuSim3d.ts.
export function particleGradientKeyForMode(mode: ParticleColorMode | string): ParticleGradientKey {
  switch (mode) {
    case "solid":
      return "solid";
    case "velocity-inferno":
      return "gradient-inferno";
    case "velocity-viridis":
      return "gradient-viridis";
    case "velocity-spectral":
      return "gradient-spectral";
    case "velocity-cosmic":
      return "gradient-cosmic";
    case "velocity-ice":
      return "gradient-ice";
    case "audio-magma":
      return "gradient-magma";
    case "audio-viridis":
      return "gradient-viridis";
    case "audio-turbo":
      return "gradient-turbo";
    case "audio-cosmic":
      return "gradient-cosmic";
    case "audio-ice":
      return "gradient-ice";
    case "audio-ember":
      return "gradient-ember";
    case "audio-plasma":
      return "gradient-plasma";
    case "gradient-inferno":
    case "gradient-magma":
    case "gradient-viridis":
    case "gradient-turbo":
    case "gradient-rainbow":
    case "gradient-spectral":
    case "gradient-plasma":
    case "gradient-cosmic":
    case "gradient-ice":
    case "gradient-ember":
    case "velocity":
    case "cohort":
    case "audio":
      return mode;
    default:
      return "gradient-spectral";
  }
}

export function gradientColor(mode: ParticleColorMode | ParticleGradientKey | string, value: number): RGB {
  const t = clamp01(value);
  switch (particleGradientKeyForMode(mode)) {
    case "gradient-inferno":
      return ramp4(t, [0.02, 0.01, 0.08], [0.45, 0.05, 0.35], [0.91, 0.24, 0.12], [0.99, 0.96, 0.64]);
    case "gradient-magma":
      return ramp4(t, [0.01, 0, 0.03], [0.32, 0.07, 0.48], [0.82, 0.25, 0.32], [0.99, 0.87, 0.58]);
    case "gradient-viridis":
      return ramp4(t, [0.17, 0, 0.33], [0.10, 0.38, 0.55], [0.20, 0.70, 0.48], [0.99, 0.91, 0.14]);
    case "gradient-turbo":
      return ramp4(t, [0.19, 0.07, 0.23], [0.10, 0.48, 0.97], [0.51, 0.99, 0.37], [0.98, 0.17, 0.09]);
    case "gradient-rainbow":
      return hsv(t, 0.92, 1);
    case "gradient-plasma":
      return ramp4(t, [0.05, 0, 0.20], [0.50, 0, 0.60], [1.00, 0.30, 0.40], [1.00, 0.92, 0.62]);
    case "gradient-cosmic":
      return ramp4(t, [0.02, 0, 0.10], [0.20, 0.05, 0.50], [0.80, 0.10, 0.62], [0.60, 0.95, 1.00]);
    case "gradient-ice":
      return ramp4(t, [0, 0.05, 0.12], [0.05, 0.32, 0.62], [0.42, 0.82, 0.96], [0.92, 1.00, 1.00]);
    case "gradient-ember":
      return ramp4(t, [0.05, 0, 0], [0.62, 0.05, 0], [1.00, 0.42, 0.06], [1.00, 0.95, 0.72]);
    case "velocity":
      return ramp4(t, [0.05, 0.10, 0.55], [0.10, 0.75, 0.90], [0.95, 0.85, 0.20], [1.00, 0.18, 0.08]);
    case "cohort":
      return hsv(t, 0.85, 1);
    case "audio":
      return ramp4(t, [0.15, 0, 0.45], [0.95, 0.10, 0.55], [1.00, 0.65, 0.10], [0.40, 0.95, 1.00]);
    case "gradient-spectral":
    case "solid":
      return ramp4(t, [0.37, 0.31, 0.64], [0.15, 0.68, 0.50], [0.99, 0.87, 0.35], [0.83, 0.18, 0.14]);
  }
}

type RGB01 = readonly [number, number, number];

function ramp4(t: number, a: RGB01, b: RGB01, c: RGB01, d: RGB01): RGB {
  if (t < 1 / 3) return mixRgb(a, b, smoothstep(0, 1 / 3, t));
  if (t < 2 / 3) return mixRgb(b, c, smoothstep(1 / 3, 2 / 3, t));
  return mixRgb(c, d, smoothstep(2 / 3, 1, t));
}

function mixRgb(a: RGB01, b: RGB01, t: number): RGB {
  return [
    Math.round((a[0] + (b[0] - a[0]) * t) * 255),
    Math.round((a[1] + (b[1] - a[1]) * t) * 255),
    Math.round((a[2] + (b[2] - a[2]) * t) * 255)
  ];
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp01((x - edge0) / (edge1 - edge0));
  return t * t * (3 - 2 * t);
}

function hsv(h: number, s: number, v: number): RGB {
  const hue = ((h % 1) + 1) % 1;
  const i = Math.floor(hue * 6);
  const f = hue * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  const [r, g, b] = [
    [v, t, p],
    [q, v, p],
    [p, v, t],
    [p, q, v],
    [t, p, v],
    [v, p, q]
  ][i % 6];
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}
