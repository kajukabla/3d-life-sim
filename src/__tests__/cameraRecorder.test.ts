import { describe, expect, it } from "vitest";
import { type CameraPose, createCameraRecorder, denseKeyframes } from "../cameraRecorder";

const pose = (yaw: number): CameraPose => ({
  yaw, pitch: -0.3, distance: 3, panX: 0, panY: 0,
  fov: 1.2, focusDistance: 3, aperture: 2.8, dofBlur: 0, dofEnabled: false
});

describe("camera recorder", () => {
  it("records and returns an exact (copied) pose per frame", () => {
    const rec = createCameraRecorder();
    const p = pose(0.5);
    rec.record(10, p);
    expect(rec.get(10)).toEqual(p);
    // returns a copy, not the same reference
    expect(rec.get(10)).not.toBe(p);
  });

  it("returns undefined for an unrecorded frame", () => {
    const rec = createCameraRecorder();
    expect(rec.get(7)).toBeUndefined();
  });

  it("carries time variation: different frames hold different poses (proves recording works)", () => {
    const rec = createCameraRecorder();
    rec.record(0, pose(0));
    rec.record(100, pose(1.0));
    expect(rec.get(0)).not.toEqual(rec.get(100));
    expect(rec.frames()).toEqual([0, 100]);
    expect(rec.size()).toBe(2);
  });

  it("bakes dense keyframes, holding the last pose across gaps", () => {
    const rec = createCameraRecorder();
    rec.record(0, pose(0));
    rec.record(2, pose(2));
    const kf = denseKeyframes(rec, 0, 3);
    expect(kf.map((p) => p.yaw)).toEqual([0, 0, 2, 2]);
    expect(kf).toHaveLength(4);
  });

  it("bakes an empty array when nothing was recorded", () => {
    const rec = createCameraRecorder();
    expect(denseKeyframes(rec, 0, 10)).toEqual([]);
  });
});
