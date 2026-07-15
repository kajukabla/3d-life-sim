export type CachePoint = {
  position: [number, number, number];
  density: number;
};

export type CacheCloud = {
  domain: [number, number, number];
  camera_position: [number, number, number];
  camera_target: [number, number, number];
  points: CachePoint[];
};

export type CloudMetrics = {
  pointCount: number;
  densitySum: number;
  maxDensity: number;
  centerOfMass: [number, number, number];
  bounds: [[number, number, number], [number, number, number]];
};

export async function loadCacheCloud(): Promise<CacheCloud> {
  const response = await fetch("./evidence/cache_points.json", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load cache_points.json: ${response.status}`);
  }
  const cloud = (await response.json()) as CacheCloud;
  validateCloud(cloud);
  return cloud;
}
export function validateCloud(cloud: CacheCloud): void {
  if (!Array.isArray(cloud.domain) || cloud.domain.length !== 3) {
    throw new Error("cache cloud missing domain");
  }
  if (!Array.isArray(cloud.points) || cloud.points.length === 0) {
    throw new Error("cache cloud has no points");
  }
  for (const point of cloud.points) {
    if (!Array.isArray(point.position) || point.position.length !== 3) {
      throw new Error("cache point missing position");
    }
    if (!Number.isFinite(point.density) || point.density <= 0) {
      throw new Error("cache point density must be positive");
    }
  }
}

export function cloudMetrics(cloud: CacheCloud): CloudMetrics {
  let densitySum = 0;
  let maxDensity = 0;
  let mx = 0;
  let my = 0;
  let mz = 0;
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (const point of cloud.points) {
    const density = point.density;
    densitySum += density;
    maxDensity = Math.max(maxDensity, density);
    mx += point.position[0] * density;
    my += point.position[1] * density;
    mz += point.position[2] * density;
    for (let axis = 0; axis < 3; axis += 1) {
      min[axis] = Math.min(min[axis], point.position[axis]);
      max[axis] = Math.max(max[axis], point.position[axis]);
    }
  }
  return {
    pointCount: cloud.points.length,
    densitySum,
    maxDensity,
    centerOfMass: [mx / densitySum, my / densitySum, mz / densitySum],
    bounds: [min, max]
  };
}
