import { type Vec3 } from "./cameraMath3d";

export const PARTICLE_STRIDE = 12;

export type Spring = { value: number; velocity: number };

// Semi-implicit damped harmonic oscillator. smoothTime≈1/speed (smaller = snappier),
// dampingRatio: 1 = ~critical, <1 underdamped (overshoot), >1 overdamped (sluggish).
export function springStep(
  spring: Spring,
  target: number,
  smoothTime: number,
  dampingRatio: number,
  dt: number
): void {
  const omega = 2 / Math.max(1e-4, smoothTime);
  const stiffness = omega * omega;
  const damping = 2 * dampingRatio * omega;
  const accel = stiffness * (target - spring.value) - damping * spring.velocity;
  spring.velocity += accel * dt;
  spring.value += spring.velocity * dt;
}

export type Ray = { origin: Vec3; direction: Vec3 };

// Returns the index of the frontmost particle whose perpendicular distance to the ray
// is within coneRadius*t (screen-consistent cone), or -1. direction must be normalized.
export function pickParticle(
  ray: Ray,
  particles: Float32Array,
  count: number,
  coneRadius: number
): number {
  const [ox, oy, oz] = ray.origin;
  const [dx, dy, dz] = ray.direction;
  let best = -1;
  let bestT = Infinity;
  for (let i = 0; i < count; i++) {
    const b = i * PARTICLE_STRIDE;
    const px = particles[b] - ox;
    const py = particles[b + 1] - oy;
    const pz = particles[b + 2] - oz;
    const t = px * dx + py * dy + pz * dz;
    if (t <= 0) continue;
    const cx = px - dx * t;
    const cy = py - dy * t;
    const cz = pz - dz * t;
    const perp = Math.sqrt(cx * cx + cy * cy + cz * cz);
    if (perp <= coneRadius * t && t < bestT) {
      bestT = t;
      best = i;
    }
  }
  return best;
}

export type Cluster = {
  members: number[];
  centroid: Vec3;
  velocity: Vec3;
  boundingRadius: number;
};

// cohesion: 0 = pure spatial clumping; 1 = candidate velocity must match the seed's almost
// exactly. The gate is skipped when either velocity is near-zero (MIN_COHESION_SPEED).
export type GrowParams = { eps: number; maxMembers: number; maxRadius: number; cohesion: number };

const MIN_COHESION_SPEED = 1e-4;

// Density + velocity-cohesion region-grow (BFS) over a uniform spatial-hash grid.
// Cohort-agnostic: an organism is a clump whose members also move together (cohesion>0).
export function growCluster(
  particles: Float32Array,
  count: number,
  seedIndex: number,
  params: GrowParams
): Cluster {
  const { eps, maxMembers, maxRadius, cohesion } = params;
  const cosThreshold = 2 * cohesion - 1;
  const svb = seedIndex * PARTICLE_STRIDE;
  const seedVx = particles[svb + 4], seedVy = particles[svb + 5], seedVz = particles[svb + 6];
  const seedSpeed = Math.sqrt(seedVx * seedVx + seedVy * seedVy + seedVz * seedVz);

  const grid = new Map<string, number[]>();
  for (let i = 0; i < count; i++) {
    const b = i * PARTICLE_STRIDE;
    const key = `${Math.floor(particles[b] / eps)},${Math.floor(particles[b + 1] / eps)},${Math.floor(particles[b + 2] / eps)}`;
    const bucket = grid.get(key);
    if (bucket) bucket.push(i);
    else grid.set(key, [i]);
  }

  const eps2 = eps * eps;
  const maxR2 = maxRadius * maxRadius;
  const sb = seedIndex * PARTICLE_STRIDE;
  const seed: Vec3 = [particles[sb], particles[sb + 1], particles[sb + 2]];
  const visited = new Set<number>([seedIndex]);
  const members: number[] = [];
  const queue: number[] = [seedIndex];

  while (queue.length > 0 && members.length < maxMembers) {
    const idx = queue.shift() as number;
    const b = idx * PARTICLE_STRIDE;
    const x = particles[b], y = particles[b + 1], z = particles[b + 2];
    const drx = x - seed[0], dry = y - seed[1], drz = z - seed[2];
    if (drx * drx + dry * dry + drz * drz > maxR2) continue;
    members.push(idx);
    const cx = Math.floor(x / eps), cy = Math.floor(y / eps), cz = Math.floor(z / eps);
    for (let gx = -1; gx <= 1; gx++)
      for (let gy = -1; gy <= 1; gy++)
        for (let gz = -1; gz <= 1; gz++) {
          const bucket = grid.get(`${cx + gx},${cy + gy},${cz + gz}`);
          if (!bucket) continue;
          for (const j of bucket) {
            if (visited.has(j)) continue;
            const jb = j * PARTICLE_STRIDE;
            const ddx = particles[jb] - x, ddy = particles[jb + 1] - y, ddz = particles[jb + 2] - z;
            if (ddx * ddx + ddy * ddy + ddz * ddz > eps2) continue;
            if (cosThreshold > -1 && seedSpeed > MIN_COHESION_SPEED) {
              const jvx = particles[jb + 4], jvy = particles[jb + 5], jvz = particles[jb + 6];
              const jSpeed = Math.sqrt(jvx * jvx + jvy * jvy + jvz * jvz);
              if (jSpeed > MIN_COHESION_SPEED) {
                const cos = (jvx * seedVx + jvy * seedVy + jvz * seedVz) / (jSpeed * seedSpeed);
                if (cos < cosThreshold) continue;
              }
            }
            visited.add(j);
            queue.push(j);
          }
        }
  }

  let sx = 0, sy = 0, sz = 0, svx = 0, svy = 0, svz = 0;
  for (const idx of members) {
    const b = idx * PARTICLE_STRIDE;
    sx += particles[b]; sy += particles[b + 1]; sz += particles[b + 2];
    svx += particles[b + 4]; svy += particles[b + 5]; svz += particles[b + 6];
  }
  const n = Math.max(1, members.length);
  const centroid: Vec3 = [sx / n, sy / n, sz / n];
  let boundingRadius = 0;
  for (const idx of members) {
    const b = idx * PARTICLE_STRIDE;
    const ex = particles[b] - centroid[0], ey = particles[b + 1] - centroid[1], ez = particles[b + 2] - centroid[2];
    boundingRadius = Math.max(boundingRadius, Math.sqrt(ex * ex + ey * ey + ez * ez));
  }
  return { members, centroid, velocity: [svx / n, svy / n, svz / n], boundingRadius };
}

export function nearestParticleIndex(particles: Float32Array, count: number, point: Vec3): number {
  let best = -1;
  let bestD2 = Infinity;
  for (let i = 0; i < count; i++) {
    const b = i * PARTICLE_STRIDE;
    const dx = particles[b] - point[0], dy = particles[b + 1] - point[1], dz = particles[b + 2] - point[2];
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 < bestD2) { bestD2 = d2; best = i; }
  }
  return best;
}

// Cluster link radius must exceed the mean particle spacing or no particle finds a neighbor
// and clusters never form. Mean spacing for N points in the [-1,1]^3 volume (8 units^3) is
// (8/N)^(1/3); we use ~1.6x that so neighbors reliably connect, clamped to a sane band.
export function epsForParticleCount(count: number): number {
  const spacing = Math.cbrt(8 / Math.max(1, count));
  return Math.min(0.22, Math.max(0.05, 1.6 * spacing));
}

export const DEFAULT_TRACKER_PARAMS = {
  eps: 0.08,
  maxMembers: 4000,
  maxRadius: 0.5,
  minMembers: 15,
  reacquireRadius: 0.35,
  reacquireTimeoutMs: 1500,
  cohesion: 0.5
} as const;

export type TrackerParams = {
  eps: number;
  maxMembers: number;
  maxRadius: number;
  minMembers: number;
  reacquireRadius: number;
  reacquireTimeoutMs: number;
  cohesion: number;
};

export type TrackState = {
  active: boolean;
  members: number;
  centroid: Vec3;
  velocity: Vec3;
};

function dist(a: Vec3, b: Vec3): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

export class SwimmerTracker {
  private params: TrackerParams;
  private grow: GrowParams;
  private activeFlag = false;
  private centroid: Vec3 = [0, 0, 0];
  private velocity: Vec3 = [0, 0, 0];
  private memberCount = 0;
  private lastMs = -1;
  private lostSinceMs = 0;

  constructor(params: TrackerParams) {
    this.params = params;
    this.grow = { eps: params.eps, maxMembers: params.maxMembers, maxRadius: params.maxRadius, cohesion: params.cohesion };
  }

  // Live-tunable from the Cohesion slider without re-locking.
  setCohesion(cohesion: number): void {
    this.params = { ...this.params, cohesion };
    this.grow = { ...this.grow, cohesion };
  }

  // Adapt the link radius to particle density so clusters form at any particle count.
  setEps(eps: number): void {
    this.params = { ...this.params, eps };
    this.grow = { ...this.grow, eps };
  }

  get state(): TrackState {
    return {
      active: this.activeFlag,
      members: this.memberCount,
      centroid: [this.centroid[0], this.centroid[1], this.centroid[2]],
      velocity: [this.velocity[0], this.velocity[1], this.velocity[2]]
    };
  }

  private adopt(cluster: Cluster): void {
    this.centroid = cluster.centroid;
    this.velocity = cluster.velocity;
    this.memberCount = cluster.members.length;
    this.lostSinceMs = 0;
  }

  lockFromSeed(particles: Float32Array, count: number, seedIndex: number): boolean {
    if (seedIndex < 0) return false;
    const cluster = growCluster(particles, count, seedIndex, this.grow);
    if (cluster.members.length < this.params.minMembers) {
      this.activeFlag = false;
      return false;
    }
    this.activeFlag = true;
    this.lastMs = -1;
    this.adopt(cluster);
    return true;
  }

  recluster(particles: Float32Array, count: number, nowMs: number): void {
    if (!this.activeFlag) return;
    const dtSec = this.lastMs < 0 ? 0 : Math.min(0.5, Math.max(0, (nowMs - this.lastMs) / 1000));
    this.lastMs = nowMs;

    const predicted: Vec3 = [
      this.centroid[0] + this.velocity[0] * dtSec,
      this.centroid[1] + this.velocity[1] * dtSec,
      this.centroid[2] + this.velocity[2] * dtSec
    ];
    let cluster = growCluster(particles, count, nearestParticleIndex(particles, count, predicted), this.grow);
    if (cluster.members.length >= this.params.minMembers) {
      this.adopt(cluster);
      return;
    }
    cluster = growCluster(particles, count, nearestParticleIndex(particles, count, this.centroid), this.grow);
    if (cluster.members.length >= this.params.minMembers && dist(cluster.centroid, this.centroid) <= this.params.reacquireRadius) {
      this.adopt(cluster);
      return;
    }
    if (this.lostSinceMs === 0) this.lostSinceMs = nowMs;
    else if (nowMs - this.lostSinceMs > this.params.reacquireTimeoutMs) this.release();
  }

  release(): void {
    this.activeFlag = false;
    this.memberCount = 0;
    this.lostSinceMs = 0;
    this.lastMs = -1;
  }
}
