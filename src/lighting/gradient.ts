import { gradientColor, particleGradientKeyForMode } from "../particleGradients";
import type { LightingColorSource, RGB } from "./types";

// Sample the color the lights should show at gradient position t (wrapped to 0..1).
export function sampleLightingGradient(source: LightingColorSource, t: number): RGB {
  if (source.mode === "solid") return hexToRgb255(source.tint);
  const key = particleGradientKeyForMode(source.mode);
  return gradientColor(key, wrap01(t));
}

function hexToRgb255(value: string): RGB {
  const match = /^#([0-9a-f]{6})$/i.exec(value);
  if (!match) return [255, 255, 255];
  const hex = match[1];
  return [
    parseInt(hex.slice(0, 2), 16),
    parseInt(hex.slice(2, 4), 16),
    parseInt(hex.slice(4, 6), 16)
  ];
}

function wrap01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return ((value % 1) + 1) % 1;
}
