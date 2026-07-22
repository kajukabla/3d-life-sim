import { liveParticleStep, maxLiveParticles, minLiveParticles } from "./particleLimits";

// Mobile GPUs can't sustain the desktop preset budgets, so presets boot at a tenth
// of their saved count on touch-primary devices. Explicit ?particles= still wins,
// and manual edits to the Particles field are never rescaled.
export function mobileParticleCount(count: number): number {
  const scaled = Math.round(count / 10 / liveParticleStep) * liveParticleStep;
  return Math.min(maxLiveParticles, Math.max(minLiveParticles, scaled));
}

// Touch-primary detection: matches phones and tablets, but not touchscreen laptops
// whose primary pointer is a mouse/trackpad.
export function isCoarsePointerClient(): boolean {
  return typeof window !== "undefined" && typeof window.matchMedia === "function"
    ? window.matchMedia("(pointer: coarse)").matches
    : false;
}
