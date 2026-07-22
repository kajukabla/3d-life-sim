import { liveParticleStep, maxLiveParticles, minLiveParticles } from "./particleLimits";

// Mobile GPUs can't sustain the desktop preset budgets, so preset counts are capped
// at ~200K on touch-primary devices (snapped to the particle step). Explicit
// ?particles= still wins, and manual edits to the Particles field are never rescaled.
export const mobileMaxParticles = Math.round(200000 / liveParticleStep) * liveParticleStep;

export function mobileParticleCount(count: number): number {
  const capped = Math.min(count, mobileMaxParticles);
  const snapped = Math.round(capped / liveParticleStep) * liveParticleStep;
  return Math.min(maxLiveParticles, Math.max(minLiveParticles, snapped));
}

// Touch-primary detection: matches phones and tablets, but not touchscreen laptops
// whose primary pointer is a mouse/trackpad.
export function isCoarsePointerClient(): boolean {
  return typeof window !== "undefined" && typeof window.matchMedia === "function"
    ? window.matchMedia("(pointer: coarse)").matches
    : false;
}
