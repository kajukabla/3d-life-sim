import type { DiscoveryCandidate } from "./discovery";
import { gradientColor } from "./particleGradients";

type PreviewParticle = {
  x: number;
  y: number;
  vx: number;
  vy: number;
  vz: number;
  cohort: number;
  id: number;
};

export type DiscoveryPreviewOptions = {
  frames?: number;
  size?: number;
  particleCount?: number;
};

export function drawDiscoveryPreview(
  canvas: HTMLCanvasElement,
  candidate: DiscoveryCandidate,
  options: DiscoveryPreviewOptions = {}
): void {
  const ctx = canvas.getContext("2d", { alpha: false, colorSpace: "srgb" });
  if (!ctx) return;
  const size = Math.max(96, Math.min(256, Math.round(options.size ?? 176)));
  const frames = Math.max(1, Math.round(options.frames ?? 180));
  const particleCount = Math.max(320, Math.min(7000, Math.round(options.particleCount ?? candidate.liveConfig.particleCount / 6)));
  const field = new Float32Array(size * size);
  const particles = createParticles(candidate, particleCount, size);
  for (let frame = 0; frame < frames; frame += 1) {
    stepPreview(candidate, particles, field, size, frame);
  }
  renderPreview(ctx, canvas.width, canvas.height, candidate, particles, field, size);
}

function createParticles(candidate: DiscoveryCandidate, particleCount: number, size: number): PreviewParticle[] {
  const rng = seededRng(candidate.seed);
  const particles: PreviewParticle[] = [];
  const cohorts = Math.max(1, candidate.liveConfig.cohorts);
  for (let id = 0; id < particleCount; id += 1) {
    const cohort = Math.floor((id / particleCount) * cohorts);
    const cohortPhase = (cohort / cohorts) * Math.PI * 2;
    const angle = rng() * Math.PI * 2;
    const radius = candidate.liveConfig.initialConditions === 2
      ? size * (0.24 + 0.08 * Math.sin(cohortPhase * 3))
      : Math.sqrt(rng()) * size * 0.42;
    const gridSide = Math.ceil(Math.sqrt(cohorts));
    const gx = cohort % gridSide;
    const gy = Math.floor(cohort / gridSide);
    const centerX = candidate.liveConfig.initialConditions === 0
      ? size * (0.18 + 0.64 * ((gx + 0.5) / gridSide))
      : size * 0.5;
    const centerY = candidate.liveConfig.initialConditions === 0
      ? size * (0.18 + 0.64 * ((gy + 0.5) / gridSide))
      : size * 0.5;
    const x = wrap(centerX + Math.cos(angle) * radius + randomRange(rng, -3, 3), size);
    const y = wrap(centerY + Math.sin(angle) * radius + randomRange(rng, -3, 3), size);
    const heading = angle + Math.PI * 0.5 + randomRange(rng, -0.9, 0.9) + cohortPhase * 0.15;
    particles.push({
      x,
      y,
      vx: Math.cos(heading),
      vy: Math.sin(heading),
      vz: 0,
      cohort,
      id
    });
  }
  return particles;
}

function stepPreview(
  candidate: DiscoveryCandidate,
  particles: PreviewParticle[],
  field: Float32Array,
  size: number,
  frame: number
): void {
  const config = candidate.liveConfig;
  const sensorDistance = 2 + config.sensorDistance * 7;
  const sensorAngle = config.sensorAngle * Math.PI * 0.72;
  const force = 0.18 + config.globalForceMult * 0.42 + Math.abs(config.axialForce) * 0.22;
  const turnGain = 0.018 + config.sensorGain * 0.006 + Math.abs(config.lateralForce) * 0.032;
  const strafe = config.strafePower * 0.9;
  const persistence = config.trailPersistence;
  const diffusion = config.trailDiffusion;
  const mass = config.depositMass * (0.035 + config.depositRadius * 0.22);
  const mutation = config.mutationScale;
  const rng = seededRng(candidate.seed + frame * 7919);

  for (let i = 0; i < field.length; i += 1) {
    field[i] *= persistence;
  }
  if (diffusion > 0.02 && frame % 2 === 0) {
    diffuse(field, size, diffusion * 0.08);
  }

  for (const particle of particles) {
    const forward = Math.atan2(particle.vy, particle.vx);
    const left = sample(field, particle.x + Math.cos(forward + sensorAngle) * sensorDistance, particle.y + Math.sin(forward + sensorAngle) * sensorDistance, size);
    const right = sample(field, particle.x + Math.cos(forward - sensorAngle) * sensorDistance, particle.y + Math.sin(forward - sensorAngle) * sensorDistance, size);
    const ahead = sample(field, particle.x + particle.vx * sensorDistance, particle.y + particle.vy * sensorDistance, size);
    const cohortBias = Math.sin((particle.cohort + 1) * 12.989 + candidate.seed * 0.0001) * mutation * 0.34;
    const jitter = randomRange(rng, -0.035, 0.035) * (1 + mutation * 6);
    const turn = (left - right) * turnGain + cohortBias + config.lateralForce * 0.018 + jitter;
    const cos = Math.cos(turn);
    const sin = Math.sin(turn);
    const vx = particle.vx * cos - particle.vy * sin;
    const vy = particle.vx * sin + particle.vy * cos;
    const length = Math.hypot(vx, vy) || 1;
    particle.vx = vx / length;
    particle.vy = vy / length;
    const sideX = -particle.vy * strafe;
    const sideY = particle.vx * strafe;
    const speed = force * (0.65 + ahead * 0.4) * (1 - Math.max(0, config.drag) * 0.18);
    particle.x = wrap(particle.x + particle.vx * speed + sideX, size);
    particle.y = wrap(particle.y + particle.vy * speed + sideY, size);
    deposit(field, size, particle.x, particle.y, mass, 1 + config.depositRadius * size * 0.18);
  }
}

function renderPreview(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  candidate: DiscoveryCandidate,
  particles: PreviewParticle[],
  field: Float32Array,
  size: number
): void {
  const image = ctx.createImageData(width, height);
  const maxValue = field.reduce((max, value) => Math.max(max, value), 0.0001);
  const palette = candidate.controls.palette;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const sx = Math.floor((x / width) * size);
      const sy = Math.floor((y / height) * size);
      const value = field[sy * size + sx] / maxValue;
      const signal = Math.pow(Math.max(0, value - candidate.controls.trailThreshold), 0.42) * candidate.controls.trailOpacity;
      const vignette = Math.max(0, 1 - Math.hypot(x / width - 0.5, y / height - 0.52) * 1.05);
      const color = trailColor(signal * vignette, palette, candidate.controls.trailColorMode);
      const i = (y * width + x) * 4;
      image.data[i] = color[0];
      image.data[i + 1] = color[1];
      image.data[i + 2] = color[2];
      image.data[i + 3] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);
  ctx.globalCompositeOperation = candidate.controls.particleBlendMode === "alpha" ? "source-over" : "lighter";
  const particleStride = Math.max(1, Math.floor(particles.length / 1800));
  for (let i = 0; i < particles.length; i += particleStride) {
    const particle = particles[i];
    const hue = particleGradientCoordinate(candidate, particle);
    const [r, g, b] = particleColor(candidate, hue);
    const px = (particle.x / size) * width;
    const py = (particle.y / size) * height;
    const radius = Math.max(candidate.controls.particleMinPx, Math.min(candidate.controls.particleMaxPx, candidate.controls.particleSizePx));
    const alpha = Math.max(0, Math.min(1, candidate.controls.particleOpacity * Math.max(0, candidate.controls.particleBrightness)));
    const speed = Math.hypot(particle.vx, particle.vy, particle.vz);
    const stretchT = smoothstep(0, Math.max(0.000001, candidate.controls.particleStretchSpeed), speed);
    const stretchAmount = candidate.controls.particleVelocityStretch
      ? candidate.controls.particleStretchMin + (Math.max(candidate.controls.particleStretchMin, candidate.controls.particleStretch) - candidate.controls.particleStretchMin) * stretchT
      : 0;
    const stretch = Math.max(1, 1 + stretchAmount);
    const angle = Math.atan2(particle.vy, particle.vx);
    ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`;
    ctx.beginPath();
    ctx.ellipse(px, py, radius * stretch, radius, angle, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalCompositeOperation = "source-over";
}

function particleGradientCoordinate(candidate: DiscoveryCandidate, particle: PreviewParticle): number {
  const base = particle.cohort / Math.max(1, candidate.liveConfig.cohorts);
  const sensitivity = Math.max(-1, Math.min(1, candidate.liveConfig.hueSensitivity));
  const velocityAngle = positiveModulo(Math.atan2(particle.vy, particle.vx) / (Math.PI * 2) + 0.5, 1);
  const velocitySpeed = Math.min(1, Math.hypot(particle.vx, particle.vy));
  const velocityHue = positiveModulo(velocityAngle + velocitySpeed * 0.23, 1);
  const directedVelocityHue = sensitivity < 0 ? 1 - velocityHue : velocityHue;
  const mixAmount = smoothstep(0, 1, Math.abs(sensitivity));
  return positiveModulo(base * (1 - mixAmount) + directedVelocityHue * mixAmount, 1);
}

function trailColor(signal: number, palette: string, mode: string): [number, number, number] {
  const value = Math.max(0, Math.min(1, signal));
  if (mode === "thermal") {
    return [
      Math.round(255 * Math.min(1, value * 1.7)),
      Math.round(210 * Math.pow(value, 0.74)),
      Math.round(120 * Math.pow(value, 1.6))
    ];
  }
  if (mode === "tint") {
    return [
      Math.round(120 * Math.pow(value, 0.55)),
      Math.round(210 * value),
      Math.round(255 * Math.pow(value, 0.75))
    ];
  }
  if (palette === "ember") {
    return [
      Math.round(245 * Math.pow(value, 0.55)),
      Math.round(125 * Math.pow(value, 0.8)),
      Math.round(70 * Math.pow(value, 1.2))
    ];
  }
  if (palette === "spectral") {
    return hsv(0.58 + value * 0.34, 0.62, value);
  }
  return [
    Math.round(95 * Math.pow(value, 0.64)),
    Math.round(225 * Math.pow(value, 0.72)),
    Math.round(215 * Math.pow(value, 0.55))
  ];
}

function particleColor(candidate: DiscoveryCandidate, hue: number): [number, number, number] {
  const mode = candidate.controls.particleColorMode;
  const tint = parseHexColor(candidate.controls.particleTint);
  const applyTint = (color: [number, number, number], amount: number): [number, number, number] => [
    Math.round(color[0] * (1 - amount + tint[0] * amount)),
    Math.round(color[1] * (1 - amount + tint[1] * amount)),
    Math.round(color[2] * (1 - amount + tint[2] * amount))
  ];
  if (mode === "solid") {
    return [Math.round(tint[0] * 255), Math.round(tint[1] * 255), Math.round(tint[2] * 255)];
  }
  return applyTint(gradientColor(mode, hue), 0.28);
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function positiveModulo(value: number, size: number): number {
  return ((value % size) + size) % size;
}

function parseHexColor(value: string): [number, number, number] {
  const match = /^#([0-9a-f]{6})$/i.exec(value);
  if (!match) return [1, 1, 1];
  const hex = match[1];
  return [
    parseInt(hex.slice(0, 2), 16) / 255,
    parseInt(hex.slice(2, 4), 16) / 255,
    parseInt(hex.slice(4, 6), 16) / 255
  ];
}

function hsv(h: number, s: number, v: number): [number, number, number] {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  const [r, g, b] = [
    [v, t, p],
    [q, v, p],
    [p, v, t],
    [p, q, v],
    [t, p, v],
    [v, p, q]
  ][i % 6];
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

function deposit(field: Float32Array, size: number, x: number, y: number, amount: number, radius: number): void {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const r = Math.max(1, Math.min(5, Math.ceil(radius)));
  for (let dy = -r; dy <= r; dy += 1) {
    for (let dx = -r; dx <= r; dx += 1) {
      const d2 = dx * dx + dy * dy;
      if (d2 > r * r) continue;
      const weight = Math.exp(-d2 / Math.max(0.6, r * 0.82));
      field[wrap(iy + dy, size) * size + wrap(ix + dx, size)] += amount * weight;
    }
  }
}

function diffuse(field: Float32Array, size: number, amount: number): void {
  const copy = new Float32Array(field);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const i = y * size + x;
      const neighbors =
        copy[y * size + wrap(x - 1, size)] +
        copy[y * size + wrap(x + 1, size)] +
        copy[wrap(y - 1, size) * size + x] +
        copy[wrap(y + 1, size) * size + x];
      field[i] = copy[i] * (1 - amount) + neighbors * amount * 0.25;
    }
  }
}

function sample(field: Float32Array, x: number, y: number, size: number): number {
  return field[wrap(Math.floor(y), size) * size + wrap(Math.floor(x), size)];
}

function wrap(value: number, size: number): number {
  return ((value % size) + size) % size;
}

function randomRange(rng: () => number, min: number, max: number): number {
  return min + (max - min) * rng();
}

function seededRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
