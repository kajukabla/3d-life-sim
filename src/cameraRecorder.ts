// The procedural camera (SwimmerTracker) eases in wall-clock and re-clusters from
// async GPU readbacks, so its pose at frame N is NOT a pure function of N. To reproduce
// the animation exactly we RECORD the actual pose played at each frame and replay those
// exact poses. This module is the recording + dense-keyframe primitive.

export type CameraPose = {
  yaw: number;
  pitch: number;
  distance: number;
  panX: number;
  panY: number;
  fov: number;
  focusDistance: number;
  aperture: number;
  dofBlur: number;
  dofEnabled: boolean;
};

export type CameraRecorder = {
  record(frame: number, pose: CameraPose): void;
  get(frame: number): CameraPose | undefined;
  frames(): number[];
  size(): number;
  clear(): void;
};

// Cap the rolling pose history. The render loop records one pose PER FRAME unconditionally, so
// during a long live/demo run the frame counter climbs forever and an unbounded Map would leak one
// pose per frame (~0.7MB/min) — over an hour that makes GC progressively heavier and the viewport
// hitch. ~50k frames is ~14 min at 60fps (a few MB), beyond any realistic capture range.
const MAX_RECORDED_FRAMES = 50000;

export function createCameraRecorder(): CameraRecorder {
  const poses = new Map<number, CameraPose>();
  return {
    record(frame, pose) {
      poses.set(frame, { ...pose });
      if (poses.size > MAX_RECORDED_FRAMES) {
        // Map preserves insertion order, so the first key is the oldest-recorded frame.
        const oldest = poses.keys().next().value;
        if (oldest !== undefined) poses.delete(oldest);
      }
    },
    get(frame) {
      const p = poses.get(frame);
      return p ? { ...p } : undefined;
    },
    frames() {
      return [...poses.keys()].sort((a, b) => a - b);
    },
    size() {
      return poses.size;
    },
    clear() {
      poses.clear();
    }
  };
}

// Produce a pose for every frame in [from, to]. Gaps are held at the last recorded pose
// (a "step" curve), so a captured frame sequence always has a pose even if recording was
// sparse. Frames before the first recorded pose fall back to the first available pose.
export function denseKeyframes(recorder: CameraRecorder, from: number, to: number): CameraPose[] {
  const recorded = recorder.frames();
  const out: CameraPose[] = [];
  if (recorded.length === 0) return out;
  let held: CameraPose = recorder.get(recorded[0])!;
  for (let frame = from; frame <= to; frame += 1) {
    const exact = recorder.get(frame);
    if (exact) held = exact;
    out.push({ ...held });
  }
  return out;
}
