import { describe, expect, it } from "vitest";
import { initialTimelineState, timelineReducer } from "../useTimelineTransport";

describe("timeline transport reducer", () => {
  it("starts paused at frame 0 with full-range loop", () => {
    expect(initialTimelineState).toEqual({
      currentFrame: 0,
      playing: false,
      loopIn: 0,
      loopOut: 3599,
      playbackSpeed: 1
    });
  });

  it("toggles play", () => {
    const s = timelineReducer(initialTimelineState, { type: "toggle-play" });
    expect(s.playing).toBe(true);
  });

  it("scrubs and clamps", () => {
    expect(timelineReducer(initialTimelineState, { type: "scrub", frame: 500 }).currentFrame).toBe(500);
    expect(timelineReducer(initialTimelineState, { type: "scrub", frame: 99999 }).currentFrame).toBe(3599);
  });

  it("sets and normalizes the loop region", () => {
    const s = timelineReducer(initialTimelineState, { type: "set-loop", loopIn: 800, loopOut: 200 });
    expect(s).toMatchObject({ loopIn: 200, loopOut: 800 });
  });

  it("ticks within the loop and wraps", () => {
    const start = { ...initialTimelineState, currentFrame: 99, loopIn: 0, loopOut: 100 };
    expect(timelineReducer(start, { type: "tick" }).currentFrame).toBe(100);
    expect(timelineReducer({ ...start, currentFrame: 100 }, { type: "tick" }).currentFrame).toBe(0);
  });

  it("sets playback speed", () => {
    expect(timelineReducer(initialTimelineState, { type: "set-speed", speed: 2 }).playbackSpeed).toBe(2);
  });
});
