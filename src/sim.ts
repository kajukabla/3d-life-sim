import { wideGamut2dSettings } from "./webgpuCanvas";

export type Particle = {
  id: number;
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  species: number;
};

export type SimControls = {
  seed: number;
  timestep: number;
  particleCount: number;
  density: number;
  exposure: number;
  aperture: number;
  focusDistance: number;
  raySteps: number;
};

export type SimState = {
  frame: number;
  particles: Particle[];
  field: Float32Array;
  size: number;
};

export type SimMetrics = {
  frame: number;
  particleCount: number;
  averageVelocity: number;
  trailIntensitySum: number;
  maxTrailIntensity: number;
  nonzeroCells: number;
  centerOfMass: [number, number];
};

export const defaultControls: SimControls = {
  seed: 3405691582,
  timestep: 1,
  particleCount: 900,
  density: 1.15,
  exposure: 1.2,
  aperture: 0.42,
  focusDistance: 0.54,
  raySteps: 72
};

export function createState(controls: SimControls): SimState {
  const size = 96;
  const field = new Float32Array(size * size);
  const particles: Particle[] = [];
  for (let i = 0; i < controls.particleCount; i += 1) {
    const angle = randomRange(controls.seed, i, 0, 0, 0, Math.PI * 2);
    const radius = Math.sqrt(randomRange(controls.seed, i, 0, 1, 0, 1)) * size * 0.43;
    const cx = size * 0.5 + Math.cos(angle) * radius;
    const cy = size * 0.5 + Math.sin(angle) * radius;
    const z = randomRange(controls.seed, i, 0, 2, -1, 1);
    const heading = angle + Math.PI / 2 + randomRange(controls.seed, i, 0, 3, -0.7, 0.7);
    particles.push({
      id: i,
      x: wrap(cx, size),
      y: wrap(cy, size),
      z,
      vx: Math.cos(heading),
      vy: Math.sin(heading),
      vz: randomRange(controls.seed, i, 0, 4, -0.06, 0.06),
      species: i % 3
    });
  }
  return { frame: 0, particles, field, size };
}

export function stepState(state: SimState, controls: SimControls): SimState {
  const nextField = new Float32Array(state.field);
  const size = state.size;
  for (const particle of state.particles) {
    const ahead = sample(state.field, particle.x + particle.vx * 3, particle.y + particle.vy * 3, size);
    const left = sample(state.field, particle.x - particle.vy * 3, particle.y + particle.vx * 3, size);
    const right = sample(state.field, particle.x + particle.vy * 3, particle.y - particle.vx * 3, size);
    const speciesBias = (particle.species - 1) * 0.035;
    const turn = (left - right + speciesBias) * 0.22;
    const jitter = randomRange(controls.seed, particle.id, state.frame, 7, -0.025, 0.025);
    const cos = Math.cos(turn + jitter);
    const sin = Math.sin(turn + jitter);
    const vx = particle.vx * cos - particle.vy * sin;
    const vy = particle.vx * sin + particle.vy * cos;
    const speed = 0.42 + ahead * 0.19;
    particle.vx = vx;
    particle.vy = vy;
    particle.x = wrap(particle.x + vx * speed * controls.timestep, size);
    particle.y = wrap(particle.y + vy * speed * controls.timestep, size);
    particle.z = Math.max(-1, Math.min(1, particle.z + particle.vz));
    if (Math.abs(particle.z) >= 1) particle.vz *= -1;
    deposit(nextField, size, particle.x, particle.y, 0.035 + particle.species * 0.01);
  }
  diffuse(nextField, state.field, size);
  return {
    frame: state.frame + 1,
    particles: state.particles,
    field: nextField,
    size
  };
}

export function metrics(state: SimState): SimMetrics {
  let velocity = 0;
  let trailIntensitySum = 0;
  let maxTrailIntensity = 0;
  let nonzeroCells = 0;
  let mass = 0;
  let mx = 0;
  let my = 0;
  for (const particle of state.particles) {
    velocity += Math.hypot(particle.vx, particle.vy, particle.vz);
  }
  for (let y = 0; y < state.size; y += 1) {
    for (let x = 0; x < state.size; x += 1) {
      const value = state.field[y * state.size + x];
      trailIntensitySum += value;
      maxTrailIntensity = Math.max(maxTrailIntensity, value);
      if (value > 0.001) nonzeroCells += 1;
      mass += value;
      mx += x * value;
      my += y * value;
    }
  }
  return {
    frame: state.frame,
    particleCount: state.particles.length,
    averageVelocity: velocity / Math.max(1, state.particles.length),
    trailIntensitySum,
    maxTrailIntensity,
    nonzeroCells,
    centerOfMass: mass > 0 ? [mx / mass, my / mass] : [state.size / 2, state.size / 2]
  };
}

export function renderToCanvas(
  canvas: HTMLCanvasElement,
  state: SimState,
  controls: SimControls,
  overlay: boolean
): void {
  const ctx = canvas.getContext("2d", wideGamut2dSettings);
  if (!ctx) return;
  const width = canvas.width;
  const height = canvas.height;
  const image = ctx.createImageData(width, height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const sx = Math.floor((x / width) * state.size);
      const sy = Math.floor((y / height) * state.size);
      const value = Math.min(1, state.field[sy * state.size + sx] * controls.density * controls.exposure);
      const vignette = 1 - Math.hypot(x / width - 0.5, y / height - 0.52) * 0.95;
      const glow = Math.pow(value, 0.38) * Math.max(0.2, vignette);
      const i = (y * width + x) * 4;
      image.data[i] = Math.min(255, glow * 105 + value * 130);
      image.data[i + 1] = Math.min(255, glow * 78 + Math.sin(value * 7) * 35 + 70);
      image.data[i + 2] = Math.min(255, glow * 220);
      image.data[i + 3] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);
  for (const particle of state.particles) {
    const px = (particle.x / state.size) * width;
    const py = (particle.y / state.size) * height;
    const focus = Math.max(0.2, 1 - Math.abs(particle.z - (controls.focusDistance * 2 - 1)) * controls.aperture);
    ctx.fillStyle = particle.species === 0 ? `rgba(112,255,213,${focus})` : particle.species === 1 ? `rgba(255,126,88,${focus})` : `rgba(190,122,255,${focus})`;
    ctx.beginPath();
    ctx.arc(px, py, 0.7 + focus * 1.2, 0, Math.PI * 2);
    ctx.fill();
  }
  if (overlay) {
    ctx.save();
    ctx.globalCompositeOperation = "screen";
    ctx.strokeStyle = "rgba(255,255,255,0.42)";
    ctx.lineWidth = 2;
    ctx.strokeRect(24, 24, width - 48, height - 48);
    ctx.fillStyle = "rgba(255,240,190,0.10)";
    ctx.fillRect(24, 24, width - 48, height - 48);
    ctx.restore();
  }
}

function deposit(field: Float32Array, size: number, x: number, y: number, amount: number): void {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  for (let dy = -2; dy <= 2; dy += 1) {
    for (let dx = -2; dx <= 2; dx += 1) {
      const d2 = dx * dx + dy * dy;
      if (d2 > 4) continue;
      const px = wrap(ix + dx, size);
      const py = wrap(iy + dy, size);
      field[py * size + px] += amount * Math.exp(-d2 / 2.2);
    }
  }
}

function diffuse(field: Float32Array, previous: Float32Array, size: number): void {
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const i = y * size + x;
      const neighbors =
        previous[y * size + wrap(x - 1, size)] +
        previous[y * size + wrap(x + 1, size)] +
        previous[wrap(y - 1, size) * size + x] +
        previous[wrap(y + 1, size) * size + x];
      field[i] = Math.max(0, field[i] * 0.935 + neighbors * 0.014);
    }
  }
}

function sample(field: Float32Array, x: number, y: number, size: number): number {
  return field[wrap(Math.floor(y), size) * size + wrap(Math.floor(x), size)];
}

function wrap(value: number, size: number): number {
  return ((value % size) + size) % size;
}

function randomRange(seed: number, entity: number, timestep: number, stream: number, min: number, max: number): number {
  return min + (max - min) * randomUnit(seed, entity, timestep, stream);
}

function randomUnit(seed: number, entity: number, timestep: number, stream: number): number {
  let x = (seed ^ Math.imul(entity + 1, 0x9e3779b1) ^ Math.imul(timestep + 1, 0x85ebca6b) ^ Math.imul(stream + 1, 0xc2b2ae35)) >>> 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x7feb352d) >>> 0;
  x ^= x >>> 15;
  x = Math.imul(x, 0x846ca68b) >>> 0;
  x ^= x >>> 16;
  return (x >>> 0) / 0xffffffff;
}
