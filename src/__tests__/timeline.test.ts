import { describe, expect, it } from "vitest";
import {
  TIMELINE_FPS,
  TIMELINE_TOTAL_FRAMES,
  advanceFrame,
  clampFrame,
  frameToTimecode,
  normalizeLoop,
  planSeek
} from "../timeline";

describe("timeline model", () => {
  it("anchors 1 minute at 60fps = 3600 frames", () => {
    expect(TIMELINE_FPS).toBe(60);
    expect(TIMELINE_TOTAL_FRAMES).toBe(3600);
  });

  it("clamps frames into [0, total-1]", () => {
    expect(clampFrame(-5)).toBe(0);
    expect(clampFrame(99999)).toBe(TIMELINE_TOTAL_FRAMES - 1);
    expect(clampFrame(1234)).toBe(1234);
  });

  it("formats frames as mm:ss", () => {
    expect(frameToTimecode(0)).toBe("0:00");
    expect(frameToTimecode(60)).toBe("0:01");
    expect(frameToTimecode(90)).toBe("0:01");
    expect(frameToTimecode(3599)).toBe("0:59");
  });

  it("normalizes a loop region to 0<=in<out<=total-1", () => {
    expect(normalizeLoop(100, 50)).toEqual({ loopIn: 50, loopOut: 100 });
    expect(normalizeLoop(-10, 99999)).toEqual({ loopIn: 0, loopOut: TIMELINE_TOTAL_FRAMES - 1 });
    expect(normalizeLoop(200, 200)).toEqual({ loopIn: 199, loopOut: 200 });
  });

  it("advances within the loop region and wraps loopOut -> loopIn", () => {
    expect(advanceFrame({ currentFrame: 10, loopIn: 0, loopOut: 100 })).toBe(11);
    expect(advanceFrame({ currentFrame: 100, loopIn: 20, loopOut: 100 })).toBe(20);
    expect(advanceFrame({ currentFrame: 5, loopIn: 20, loopOut: 100 })).toBe(20);
  });

  it("plans a forward seek as steps with no reset", () => {
    expect(planSeek(10, 40)).toEqual({ needsReset: false, steps: 30 });
    expect(planSeek(10, 10)).toEqual({ needsReset: false, steps: 0 });
  });

  it("plans a backward seek as reset + steps from zero", () => {
    expect(planSeek(40, 10)).toEqual({ needsReset: true, steps: 10 });
    expect(planSeek(40, 0)).toEqual({ needsReset: true, steps: 0 });
  });
});
