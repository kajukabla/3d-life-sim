import { describe, it, expect } from "vitest";
import {
  springStep,
  pickParticle,
  growCluster,
  nearestParticleIndex,
  epsForParticleCount,
  SwimmerTracker,
  PARTICLE_STRIDE,
  type Spring,
  type Ray
} from "../cameraTracking";

function buildParticles(points: Array<[number, number, number]>): Float32Array {
  const buf = new Float32Array(points.length * PARTICLE_STRIDE);
  points.forEach((p, i) => {
    buf[i * PARTICLE_STRIDE + 0] = p[0];
    buf[i * PARTICLE_STRIDE + 1] = p[1];
    buf[i * PARTICLE_STRIDE + 2] = p[2];
  });
  return buf;
}

function buildFull(rows: Array<[number, number, number, number, number, number]>): Float32Array {
  const buf = new Float32Array(rows.length * PARTICLE_STRIDE);
  rows.forEach((r, i) => {
    const b = i * PARTICLE_STRIDE;
    buf[b] = r[0]; buf[b + 1] = r[1]; buf[b + 2] = r[2];
    buf[b + 4] = r[3]; buf[b + 5] = r[4]; buf[b + 6] = r[5];
  });
  return buf;
}

function blob(cx: number, cy: number, cz: number, n: number, vel: [number, number, number]) {
  const rows: Array<[number, number, number, number, number, number]> = [];
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2;
    rows.push([cx + 0.01 * Math.cos(a), cy + 0.01 * Math.sin(a), cz, vel[0], vel[1], vel[2]]);
  }
  return rows;
}

function settle(target: number, speed: number, damping: number, steps: number): Spring {
  const s: Spring = { value: 0, velocity: 0 };
  for (let i = 0; i < steps; i++) springStep(s, target, 1 / speed, damping, 1 / 60);
  return s;
}

function stepsTo90(target: number, speed: number, damping: number): number {
  const s: Spring = { value: 0, velocity: 0 };
  for (let i = 1; i <= 1000; i++) {
    springStep(s, target, 1 / speed, damping, 1 / 60);
    if (s.value >= 0.9 * target) return i;
  }
  return Infinity;
}

describe("springStep", () => {
  it("converges to the target", () => {
    const s = settle(5, 4, 1, 600);
    expect(Math.abs(s.value - 5)).toBeLessThan(0.01);
    expect(Math.abs(s.velocity)).toBeLessThan(0.01);
  });

  it("does not meaningfully overshoot at critical/over damping", () => {
    const s: Spring = { value: 0, velocity: 0 };
    let maxValue = 0;
    for (let i = 0; i < 600; i++) {
      springStep(s, 1, 1 / 4, 1, 1 / 60);
      maxValue = Math.max(maxValue, s.value);
    }
    expect(maxValue).toBeLessThanOrEqual(1.01);
  });

  it("higher speed converges in fewer steps", () => {
    expect(stepsTo90(5, 8, 1)).toBeLessThan(stepsTo90(5, 2, 1));
  });
});

describe("pickParticle", () => {
  const ray: Ray = { origin: [0, 0, -5], direction: [0, 0, 1] };

  it("picks the frontmost particle near the ray", () => {
    const particles = buildParticles([
      [0, 0, 2],
      [0, 0, -1],
      [3, 3, 0]
    ]);
    expect(pickParticle(ray, particles, 3, 0.1)).toBe(1);
  });

  it("returns -1 when nothing is within the cone", () => {
    const particles = buildParticles([[3, 3, 0], [-3, 2, 1]]);
    expect(pickParticle(ray, particles, 2, 0.1)).toBe(-1);
  });

  it("ignores particles behind the ray origin", () => {
    const particles = buildParticles([[0, 0, -9]]);
    expect(pickParticle(ray, particles, 1, 0.1)).toBe(-1);
  });
});

describe("growCluster", () => {
  const params = { eps: 0.05, maxMembers: 1000, maxRadius: 0.5, cohesion: 0 };

  it("captures only the seeded blob, not a far blob", () => {
    const rows = [...blob(-0.5, 0, 0, 60, [1, 0, 0]), ...blob(0.5, 0, 0, 60, [0, 0, 0])];
    const particles = buildFull(rows);
    const cluster = growCluster(particles, rows.length, 0, params);
    expect(cluster.members.length).toBe(60);
    expect(cluster.centroid[0]).toBeCloseTo(-0.5, 2);
    expect(cluster.velocity[0]).toBeCloseTo(1, 2);
  });

  it("semantic delta: merging blobs within eps grows membership", () => {
    const apart = buildFull([...blob(-0.2, 0, 0, 50, [0, 0, 0]), ...blob(0.2, 0, 0, 50, [0, 0, 0])]);
    const near = buildFull([...blob(-0.02, 0, 0, 50, [0, 0, 0]), ...blob(0.02, 0, 0, 50, [0, 0, 0])]);
    const a = growCluster(apart, 100, 0, params).members.length;
    const b = growCluster(near, 100, 0, params).members.length;
    expect(a).toBe(50);
    expect(b).toBe(100);
  });

  it("tracks a moved blob's centroid", () => {
    const moved = buildFull(blob(0.3, 0.4, -0.1, 40, [0, 0, 0]));
    const cluster = growCluster(moved, 40, 0, params);
    expect(cluster.centroid[0]).toBeCloseTo(0.3, 2);
    expect(cluster.centroid[1]).toBeCloseTo(0.4, 2);
  });

  it("cohesion gate: high cohesion does NOT merge adjacent blobs moving opposite ways", () => {
    const rows = [...blob(-0.02, 0, 0, 50, [1, 0, 0]), ...blob(0.02, 0, 0, 50, [-1, 0, 0])];
    const particles = buildFull(rows);
    const loose = growCluster(particles, 100, 0, { ...params, cohesion: 0 });
    const tight = growCluster(particles, 100, 0, { ...params, cohesion: 0.9 });
    expect(loose.members.length).toBe(100);
    expect(tight.members.length).toBe(50);
  });

  it("cohesion gate is skipped for near-stationary particles", () => {
    const rows = [...blob(-0.02, 0, 0, 50, [0, 0, 0]), ...blob(0.02, 0, 0, 50, [0, 0, 0])];
    const cluster = growCluster(buildFull(rows), 100, 0, { ...params, cohesion: 1 });
    expect(cluster.members.length).toBe(100);
  });
});

describe("epsForParticleCount", () => {
  // Regression: at the app's default 8192 particles, eps MUST exceed the mean particle
  // spacing or no cluster forms and locking silently fails (the real-world bug).
  it("exceeds mean particle spacing at low/default densities", () => {
    for (const n of [4096, 8192, 16384]) {
      const spacing = Math.cbrt(8 / n);
      expect(epsForParticleCount(n)).toBeGreaterThan(spacing);
    }
  });

  it("shrinks as density rises and stays in a sane band", () => {
    expect(epsForParticleCount(8192)).toBeGreaterThan(epsForParticleCount(262144));
    expect(epsForParticleCount(1_000_000)).toBeGreaterThanOrEqual(0.05);
    expect(epsForParticleCount(500)).toBeLessThanOrEqual(0.22);
  });

  it("forms a connected cluster from a uniform 8192-particle cloud (the failure case)", () => {
    // 8192 points on a jittered grid in [-1,1]^3, all moving together.
    const side = 20; // 20^3 = 8000 ≈ 8192
    const rows: Array<[number, number, number, number, number, number]> = [];
    for (let x = 0; x < side; x++)
      for (let y = 0; y < side; y++)
        for (let z = 0; z < side; z++) {
          rows.push([
            -1 + (2 * x) / side, -1 + (2 * y) / side, -1 + (2 * z) / side,
            1, 0, 0
          ]);
        }
    const particles = buildFull(rows);
    const eps = epsForParticleCount(rows.length);
    // Old hard-coded eps=0.06 < spacing(0.1) → singleton clusters → lock fails.
    expect(growCluster(particles, rows.length, 0, { eps: 0.06, maxMembers: 5000, maxRadius: 2, cohesion: 0 }).members.length).toBeLessThan(15);
    // Adaptive eps connects the cloud → a real cluster forms.
    expect(growCluster(particles, rows.length, 0, { eps, maxMembers: 5000, maxRadius: 2, cohesion: 0 }).members.length).toBeGreaterThan(100);
  });
});

describe("nearestParticleIndex", () => {
  it("finds the closest particle to a point", () => {
    const particles = buildFull([[0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0], [0.4, 0, 0, 0, 0, 0]]);
    expect(nearestParticleIndex(particles, 3, [0.45, 0, 0])).toBe(2);
  });
});

const PARAMS = { eps: 0.05, maxMembers: 1000, maxRadius: 0.5, minMembers: 30, reacquireRadius: 0.3, reacquireTimeoutMs: 1000, cohesion: 0 };

describe("SwimmerTracker", () => {
  it("locks onto a seed blob and reports state", () => {
    const t = new SwimmerTracker(PARAMS);
    const particles = buildFull(blob(0.2, 0, 0, 50, [0.5, 0, 0]));
    expect(t.lockFromSeed(particles, 50, 0)).toBe(true);
    expect(t.state.active).toBe(true);
    expect(t.state.members).toBe(50);
    expect(t.state.centroid[0]).toBeCloseTo(0.2, 2);
  });

  it("re-seeds toward the moved blob on recluster", () => {
    const t = new SwimmerTracker(PARAMS);
    t.lockFromSeed(buildFull(blob(0, 0, 0, 50, [1, 0, 0])), 50, 0);
    t.recluster(buildFull(blob(0.3, 0, 0, 50, [1, 0, 0])), 50, 100);
    expect(t.state.centroid[0]).toBeCloseTo(0.3, 2);
    expect(t.state.active).toBe(true);
  });

  it("re-acquires a reformed blob near the last centroid", () => {
    const t = new SwimmerTracker(PARAMS);
    t.lockFromSeed(buildFull(blob(0, 0, 0, 50, [0, 0, 0])), 50, 0);
    const reformed = buildFull([...blob(0, 0, 0, 5, [0, 0, 0]), ...blob(0.1, 0, 0, 50, [0, 0, 0])]);
    t.recluster(reformed, 55, 100);
    expect(t.state.active).toBe(true);
    expect(t.state.members).toBe(50);
  });

  it("auto-releases after the timeout when nothing qualifies nearby", () => {
    const t = new SwimmerTracker(PARAMS);
    t.lockFromSeed(buildFull(blob(0, 0, 0, 50, [0, 0, 0])), 50, 0);
    const sparse = buildFull(blob(0, 0, 0, 5, [0, 0, 0]));
    t.recluster(sparse, 5, 100);
    expect(t.state.active).toBe(true);
    t.recluster(sparse, 5, 100 + PARAMS.reacquireTimeoutMs + 1);
    expect(t.state.active).toBe(false);
  });
});
