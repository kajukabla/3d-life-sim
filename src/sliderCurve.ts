// Non-linear slider mapping. A linear range input wastes most of its travel when the useful
// values live near the bottom of the range (e.g. the particle density cutoff: useful ~0.0005-0.02
// on a 0-0.05 range). A power curve > 1 front-loads resolution: the slider POSITION (0..1) maps to
// value = min + (max-min) * pos^curve, so small values get most of the travel and large ones get
// compressed. curve === 1 is the identity (linear).

export function clamp01(x: number): number {
  if (!Number.isFinite(x)) return 0;
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

// Slider position (0..1) -> actual value.
export function sliderPosToValue(pos: number, min: number, max: number, curve: number): number {
  const p = clamp01(pos);
  const shaped = curve === 1 ? p : Math.pow(p, curve);
  return min + (max - min) * shaped;
}

// Actual value -> slider position (0..1). Inverse of sliderPosToValue.
export function sliderValueToPos(value: number, min: number, max: number, curve: number): number {
  if (max <= min) return 0;
  const norm = clamp01((value - min) / (max - min));
  return curve === 1 ? norm : Math.pow(norm, 1 / curve);
}
