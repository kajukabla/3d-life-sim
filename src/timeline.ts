export const TIMELINE_FPS = 60;
export const TIMELINE_TOTAL_FRAMES = 3600;

export type TimelineState = {
  currentFrame: number;
  playing: boolean;
  loopIn: number;
  loopOut: number;
  playbackSpeed: number;
};

export function clampFrame(frame: number, total: number = TIMELINE_TOTAL_FRAMES): number {
  if (!Number.isFinite(frame)) return 0;
  return Math.max(0, Math.min(total - 1, Math.round(frame)));
}

export function frameToTimecode(frame: number, fps: number = TIMELINE_FPS): string {
  const totalSeconds = Math.floor(clampFrame(frame, Number.MAX_SAFE_INTEGER) / fps);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

export function normalizeLoop(
  loopIn: number,
  loopOut: number,
  total: number = TIMELINE_TOTAL_FRAMES
): { loopIn: number; loopOut: number } {
  let lo = clampFrame(Math.min(loopIn, loopOut), total);
  let hi = clampFrame(Math.max(loopIn, loopOut), total);
  if (hi <= lo) {
    if (lo > 0) {
      lo = lo - 1;
    } else {
      hi = lo + 1;
    }
  }
  return { loopIn: lo, loopOut: hi };
}

export function advanceFrame(state: {
  currentFrame: number;
  loopIn: number;
  loopOut: number;
}): number {
  if (state.currentFrame < state.loopIn || state.currentFrame >= state.loopOut) {
    return state.loopIn;
  }
  return state.currentFrame + 1;
}

export function planSeek(
  currentFrame: number,
  targetFrame: number
): { needsReset: boolean; steps: number } {
  if (targetFrame >= currentFrame) {
    return { needsReset: false, steps: targetFrame - currentFrame };
  }
  return { needsReset: true, steps: targetFrame };
}
