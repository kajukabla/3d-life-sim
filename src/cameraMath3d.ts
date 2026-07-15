export type Vec2 = readonly [number, number];
export type Vec3 = readonly [number, number, number];

export type CameraMath3d = {
  yaw: number;
  pitch: number;
  distance: number;
  aspect: number;
};

export const liveCameraFocal = 1.85 * 0.92;

export function worldToCamera(position: Vec3, camera: CameraMath3d): [number, number, number] {
  const cy = Math.cos(camera.yaw);
  const sy = Math.sin(camera.yaw);
  const cp = Math.cos(camera.pitch);
  const sp = Math.sin(camera.pitch);
  const rx = position[0] * cy - position[2] * sy;
  const rz = position[0] * sy + position[2] * cy;
  const ry = position[1] * cp - rz * sp;
  const rz2 = position[1] * sp + rz * cp;
  return [rx, ry, rz2];
}

export function cameraToWorld(position: Vec3, camera: CameraMath3d): [number, number, number] {
  const cy = Math.cos(camera.yaw);
  const sy = Math.sin(camera.yaw);
  const cp = Math.cos(camera.pitch);
  const sp = Math.sin(camera.pitch);
  const y = position[1] * cp + position[2] * sp;
  const rz = -position[1] * sp + position[2] * cp;
  const x = position[0] * cy + rz * sy;
  const z = -position[0] * sy + rz * cy;
  return [x, y, z];
}

export function projectWorldToNdc(position: Vec3, camera: CameraMath3d): [number, number, number] {
  const cameraPosition = worldToCamera(position, camera);
  const perspective = 1.85 / Math.max(0.62, camera.distance + cameraPosition[2]);
  return [
    cameraPosition[0] * perspective * 0.92 / camera.aspect,
    cameraPosition[1] * perspective * 0.92,
    perspective
  ];
}

export function rayForNdc(ndc: Vec2, camera: CameraMath3d): { origin: [number, number, number]; direction: [number, number, number] } {
  const origin = cameraToWorld([0, 0, -camera.distance], camera);
  const cameraDirection: [number, number, number] = [
    ndc[0] * camera.aspect / liveCameraFocal,
    ndc[1] / liveCameraFocal,
    1
  ];
  return { origin, direction: normalize(cameraToWorld(cameraDirection, camera)) };
}

export function rayPointForProjectedWorld(position: Vec3, camera: CameraMath3d): [number, number, number] {
  const projected = projectWorldToNdc(position, camera);
  const cameraPosition = worldToCamera(position, camera);
  const cameraDirection = normalize([
    projected[0] * camera.aspect / liveCameraFocal,
    projected[1] / liveCameraFocal,
    1
  ]);
  const distanceAlongRay = (camera.distance + cameraPosition[2]) / cameraDirection[2];
  return cameraToWorld([
    cameraDirection[0] * distanceAlongRay,
    cameraDirection[1] * distanceAlongRay,
    -camera.distance + cameraDirection[2] * distanceAlongRay
  ], camera);
}

function normalize(value: Vec3): [number, number, number] {
  const length = Math.hypot(value[0], value[1], value[2]);
  if (length <= 0) return [0, 0, 0];
  return [value[0] / length, value[1] / length, value[2] / length];
}

// DOF focus: the renderer encodes screen-space depth as DOF_DEPTH_K / view_depth (an
// FOV-independent metric, realtimeGpuSim3d.ts splat_vs), and the `focusDistance` uniform is
// compared in that same space. So the focus PLANE sits at a real camera-space distance of
// DOF_DEPTH_K / focusDistance. The stored `focusDistance` is therefore inverse-distance and
// non-intuitive; these convert to/from an actual world focal distance for the UI.
export const DOF_DEPTH_K = 1.85;

// World camera-space focal distance -> stored focusDistance uniform.
export function focalToFocus(focalDistance: number): number {
  return DOF_DEPTH_K / Math.max(1e-4, focalDistance);
}

// Stored focusDistance uniform -> world camera-space focal distance (for display).
export function focusToFocal(focusDistance: number): number {
  return DOF_DEPTH_K / Math.max(1e-4, focusDistance);
}
