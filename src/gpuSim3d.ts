import { wideGamut2dSettings } from "./webgpuCanvas";
import { requestHighPerformanceWebGpuAdapter } from "./webgpu";

export type GpuSim3dConfig = {
  seed: number;
  width: number;
  height: number;
  depth: number;
  particleCount: number;
  steps: number;
  dt: number;
  speed: number;
  turn: number;
  sensorDistance: number;
  diffusion: number;
  decay: number;
  depositRadius: number;
  depositMass: number;
  sigma: number;
};

export type GpuSim3dSummary = {
  available: boolean;
  passed: boolean;
  mode: "webgpu-compute-3d" | "unavailable" | "failed";
  width: number;
  height: number;
  depth: number;
  particleCount: number;
  steps: number;
  cpuFieldSum: number;
  gpuFieldSum: number;
  gpuNonzeroVoxels: number;
  fieldMeanAbsError: number;
  fieldMaxAbsError: number;
  particlePositionMeanError: number;
  particlePositionMaxError: number;
  tolerances: {
    fieldMeanAbsError: number;
    fieldMaxAbsError: number;
    particlePositionMeanError: number;
    particlePositionMaxError: number;
  };
  error?: string;
};

export type GpuSim3dResult = GpuSim3dSummary & {
  gpuField: number[];
  cpuField: number[];
};

type Sim3dState = {
  particles: Float32Array;
  field: Float32Array;
};

export const defaultGpuSim3dConfig: GpuSim3dConfig = {
  seed: 3405691582,
  width: 4,
  height: 4,
  depth: 4,
  particleCount: 8,
  steps: 2,
  dt: 1,
  speed: 0.72,
  turn: 0.38,
  sensorDistance: 2.25,
  diffusion: 0.055,
  decay: 0.025,
  depositRadius: 2.2,
  depositMass: 0.04,
  sigma: 1.05
};

const conformanceTolerances = {
  fieldMeanAbsError: 0.035,
  fieldMaxAbsError: 0.75,
  particlePositionMeanError: 0.08,
  particlePositionMaxError: 0.32
};

export async function runWebGpuComputeConformance(
  config: GpuSim3dConfig = defaultGpuSim3dConfig
): Promise<GpuSim3dResult> {
  const cpu = runCpuReference3d(config);
  if (!navigator.gpu) {
    return failureResult(config, cpu, "navigator.gpu unavailable", "unavailable");
  }
  try {
    const adapter = await requestHighPerformanceWebGpuAdapter(navigator.gpu);
    if (!adapter) {
      return failureResult(config, cpu, "no WebGPU adapter returned", "unavailable");
    }
    const device = await adapter.requestDevice();
    const gpu = await withTimeout(runGpu3d(device, config), 10_000, "WebGPU 3D compute timed out");
    const summary = summarizeStates(config, cpu, gpu);
    device.destroy();
    return {
      ...summary,
      gpuField: Array.from(gpu.field),
      cpuField: Array.from(cpu.field)
    };
  } catch (error) {
    return failureResult(config, cpu, error instanceof Error ? error.message : String(error), "failed");
  }
}

export function runCpuReference3d(config: GpuSim3dConfig = defaultGpuSim3dConfig): Sim3dState {
  let state = createInitialState(config);
  for (let step = 0; step < config.steps; step += 1) {
    state = stepCpu3d(state, config);
  }
  return state;
}

export function drawFieldSlice(
  canvas: HTMLCanvasElement,
  field: number[] | Float32Array,
  config: Pick<GpuSim3dConfig, "width" | "height" | "depth">,
  z = Math.floor(config.depth / 2)
): void {
  const ctx = canvas.getContext("2d", wideGamut2dSettings);
  if (!ctx) return;
  const width = canvas.width;
  const height = canvas.height;
  const image = ctx.createImageData(width, height);
  let max = 0;
  const offset = z * config.width * config.height;
  for (let i = 0; i < config.width * config.height; i += 1) {
    max = Math.max(max, field[offset + i] ?? 0);
  }
  const scale = max > 0 ? 1 / max : 1;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const sx = Math.min(config.width - 1, Math.floor((x / width) * config.width));
      const sy = Math.min(config.height - 1, Math.floor((y / height) * config.height));
      const value = Math.min(1, (field[offset + sy * config.width + sx] ?? 0) * scale);
      const glow = Math.pow(value, 0.38);
      const i = (y * width + x) * 4;
      image.data[i] = Math.min(255, 24 + glow * 84 + value * 90);
      image.data[i + 1] = Math.min(255, 28 + glow * 180);
      image.data[i + 2] = Math.min(255, 36 + glow * 145 + value * 54);
      image.data[i + 3] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);
}

export function summarizeCompute3d(result: GpuSim3dResult): GpuSim3dSummary {
  const { gpuField: _gpuField, cpuField: _cpuField, ...summary } = result;
  return summary;
}

function createInitialState(config: GpuSim3dConfig): Sim3dState {
  const particles = new Float32Array(config.particleCount * 8);
  const field = new Float32Array(config.width * config.height * config.depth);
  for (let i = 0; i < config.particleCount; i += 1) {
    const x = randomRange(config.seed, i, 0, 0, 0.0, config.width);
    const y = randomRange(config.seed, i, 0, 1, 0.0, config.height);
    const z = randomRange(config.seed, i, 0, 2, 0.0, config.depth);
    const theta = randomRange(config.seed, i, 0, 3, 0.0, Math.PI * 2);
    const vz = randomRange(config.seed, i, 0, 4, -0.42, 0.42);
    const radial = Math.sqrt(Math.max(0.05, 1 - vz * vz));
    const base = i * 8;
    particles[base] = f32(x);
    particles[base + 1] = f32(y);
    particles[base + 2] = f32(z);
    particles[base + 3] = i % 3;
    particles[base + 4] = f32(Math.cos(theta) * radial);
    particles[base + 5] = f32(Math.sin(theta) * radial);
    particles[base + 6] = f32(vz);
    particles[base + 7] = i;
  }
  return { particles, field };
}

function stepCpu3d(state: Sim3dState, config: GpuSim3dConfig): Sim3dState {
  const nextParticles = new Float32Array(state.particles.length);
  for (let i = 0; i < config.particleCount; i += 1) {
    const base = i * 8;
    const position: Vec3 = [state.particles[base], state.particles[base + 1], state.particles[base + 2]];
    const species = state.particles[base + 3];
    const forward = normalize([
      state.particles[base + 4],
      state.particles[base + 5],
      state.particles[base + 6]
    ]);
    const side = normalize(cross(forward, [0.37, 0.71, 0.13]));
    const up = normalize(cross(side, forward));
    const left = sampleField(state.field, config, add(position, scale(side, config.sensorDistance)));
    const right = sampleField(state.field, config, add(position, scale(side, -config.sensorDistance)));
    const upper = sampleField(state.field, config, add(position, scale(up, config.sensorDistance)));
    const steer = f32((left - right + (species - 1) * 0.025) * config.turn);
    const lift = f32((upper - (left + right) * 0.5) * config.turn * 0.35);
    const velocity = normalize(add(forward, add(scale(side, steer), scale(up, lift))));
    const nextPosition: Vec3 = [
      wrapFloat(position[0] + velocity[0] * config.speed * config.dt, config.width),
      wrapFloat(position[1] + velocity[1] * config.speed * config.dt, config.height),
      wrapFloat(position[2] + velocity[2] * config.speed * config.dt, config.depth)
    ];
    nextParticles[base] = f32(nextPosition[0]);
    nextParticles[base + 1] = f32(nextPosition[1]);
    nextParticles[base + 2] = f32(nextPosition[2]);
    nextParticles[base + 3] = species;
    nextParticles[base + 4] = f32(velocity[0]);
    nextParticles[base + 5] = f32(velocity[1]);
    nextParticles[base + 6] = f32(velocity[2]);
    nextParticles[base + 7] = state.particles[base + 7];
  }

  const nextField = new Float32Array(state.field.length);
  for (let z = 0; z < config.depth; z += 1) {
    for (let y = 0; y < config.height; y += 1) {
      for (let x = 0; x < config.width; x += 1) {
        const index = fieldIndex(x, y, z, config);
        const center = state.field[index];
        const neighbors =
          state.field[fieldIndex(x - 1, y, z, config)] +
          state.field[fieldIndex(x + 1, y, z, config)] +
          state.field[fieldIndex(x, y - 1, z, config)] +
          state.field[fieldIndex(x, y + 1, z, config)] +
          state.field[fieldIndex(x, y, z - 1, config)] +
          state.field[fieldIndex(x, y, z + 1, config)];
        let value = Math.max(0, (1 - 6 * config.diffusion) * center + config.diffusion * neighbors) * (1 - config.decay);
        const voxel: Vec3 = [x + 0.5, y + 0.5, z + 0.5];
        for (let i = 0; i < config.particleCount; i += 1) {
          const base = i * 8;
          const dx = periodicDelta(voxel[0], nextParticles[base], config.width);
          const dy = periodicDelta(voxel[1], nextParticles[base + 1], config.height);
          const dz = periodicDelta(voxel[2], nextParticles[base + 2], config.depth);
          const d2 = dx * dx + dy * dy + dz * dz;
          if (d2 <= config.depositRadius * config.depositRadius) {
            value += config.depositMass * Math.exp(-d2 / (2 * config.sigma * config.sigma));
          }
        }
        nextField[index] = f32(value);
      }
    }
  }
  return { particles: nextParticles, field: nextField };
}

async function runGpu3d(device: GPUDevice, config: GpuSim3dConfig): Promise<Sim3dState> {
  const initial = createInitialState(config);
  const particleBytes = initial.particles.byteLength;
  const fieldBytes = initial.field.byteLength;
  const particleBuffers = [
    createGpuBuffer(device, particleBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC),
    createGpuBuffer(device, particleBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC)
  ];
  const fieldBuffers = [
    createGpuBuffer(device, fieldBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC),
    createGpuBuffer(device, fieldBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC)
  ];
  device.queue.writeBuffer(particleBuffers[0], 0, initial.particles);
  device.queue.writeBuffer(fieldBuffers[0], 0, initial.field);

  const uniformBuffer = createGpuBuffer(device, 64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const updatePipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: particleShaderSource }),
      entryPoint: "update_particles"
    }
  });
  const fieldPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: fieldShaderSource }),
      entryPoint: "deposit_diffuse_field"
    }
  });

  let readIndex = 0;
  let writeIndex = 1;
  for (let step = 0; step < config.steps; step += 1) {
    device.queue.writeBuffer(uniformBuffer, 0, encodeConfig(config, step));
    const updateBindGroup = device.createBindGroup({
      layout: updatePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: particleBuffers[readIndex] } },
        { binding: 1, resource: { buffer: fieldBuffers[readIndex] } },
        { binding: 2, resource: { buffer: particleBuffers[writeIndex] } },
        { binding: 3, resource: { buffer: uniformBuffer } }
      ]
    });
    const fieldBindGroup = device.createBindGroup({
      layout: fieldPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: particleBuffers[writeIndex] } },
        { binding: 1, resource: { buffer: fieldBuffers[readIndex] } },
        { binding: 2, resource: { buffer: fieldBuffers[writeIndex] } },
        { binding: 3, resource: { buffer: uniformBuffer } }
      ]
    });
    const encoder = device.createCommandEncoder();
    const updatePass = encoder.beginComputePass();
    updatePass.setPipeline(updatePipeline);
    updatePass.setBindGroup(0, updateBindGroup);
    updatePass.dispatchWorkgroups(Math.ceil(config.particleCount / 64));
    updatePass.end();
    const fieldPass = encoder.beginComputePass();
    fieldPass.setPipeline(fieldPipeline);
    fieldPass.setBindGroup(0, fieldBindGroup);
    fieldPass.dispatchWorkgroups(Math.ceil((config.width * config.height * config.depth) / 64));
    fieldPass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    [readIndex, writeIndex] = [writeIndex, readIndex];
  }

  const particles = await readBuffer(device, particleBuffers[readIndex], particleBytes);
  const field = await readBuffer(device, fieldBuffers[readIndex], fieldBytes);
  return {
    particles: new Float32Array(particles),
    field: new Float32Array(field)
  };
}

function summarizeStates(config: GpuSim3dConfig, cpu: Sim3dState, gpu: Sim3dState): GpuSim3dSummary {
  let fieldAbs = 0;
  let fieldMax = 0;
  let cpuFieldSum = 0;
  let gpuFieldSum = 0;
  let gpuNonzeroVoxels = 0;
  for (let i = 0; i < cpu.field.length; i += 1) {
    const cpuValue = cpu.field[i];
    const gpuValue = gpu.field[i];
    const diff = Math.abs(cpuValue - gpuValue);
    fieldAbs += diff;
    fieldMax = Math.max(fieldMax, diff);
    cpuFieldSum += cpuValue;
    gpuFieldSum += gpuValue;
    if (gpuValue > 0.0001) gpuNonzeroVoxels += 1;
  }
  let particlePositionSum = 0;
  let particlePositionMax = 0;
  for (let i = 0; i < config.particleCount; i += 1) {
    const base = i * 8;
    const dx = periodicDelta(cpu.particles[base], gpu.particles[base], config.width);
    const dy = periodicDelta(cpu.particles[base + 1], gpu.particles[base + 1], config.height);
    const dz = periodicDelta(cpu.particles[base + 2], gpu.particles[base + 2], config.depth);
    const error = Math.hypot(dx, dy, dz);
    particlePositionSum += error;
    particlePositionMax = Math.max(particlePositionMax, error);
  }
  const fieldMeanAbsError = fieldAbs / Math.max(1, cpu.field.length);
  const particlePositionMeanError = particlePositionSum / Math.max(1, config.particleCount);
  return {
    available: true,
    passed:
      fieldMeanAbsError <= conformanceTolerances.fieldMeanAbsError &&
      fieldMax <= conformanceTolerances.fieldMaxAbsError &&
      particlePositionMeanError <= conformanceTolerances.particlePositionMeanError &&
      particlePositionMax <= conformanceTolerances.particlePositionMaxError &&
      gpuFieldSum > 1 &&
      gpuNonzeroVoxels > 10,
    mode: "webgpu-compute-3d",
    width: config.width,
    height: config.height,
    depth: config.depth,
    particleCount: config.particleCount,
    steps: config.steps,
    cpuFieldSum,
    gpuFieldSum,
    gpuNonzeroVoxels,
    fieldMeanAbsError,
    fieldMaxAbsError: fieldMax,
    particlePositionMeanError,
    particlePositionMaxError: particlePositionMax,
    tolerances: conformanceTolerances
  };
}

function failureResult(
  config: GpuSim3dConfig,
  cpu: Sim3dState,
  error: string,
  mode: "unavailable" | "failed"
): GpuSim3dResult {
  return {
    available: false,
    passed: false,
    mode,
    width: config.width,
    height: config.height,
    depth: config.depth,
    particleCount: config.particleCount,
    steps: config.steps,
    cpuFieldSum: sum(cpu.field),
    gpuFieldSum: 0,
    gpuNonzeroVoxels: 0,
    fieldMeanAbsError: Number.POSITIVE_INFINITY,
    fieldMaxAbsError: Number.POSITIVE_INFINITY,
    particlePositionMeanError: Number.POSITIVE_INFINITY,
    particlePositionMaxError: Number.POSITIVE_INFINITY,
    tolerances: conformanceTolerances,
    error,
    gpuField: [],
    cpuField: Array.from(cpu.field)
  };
}

function createGpuBuffer(device: GPUDevice, size: number, usage: GPUBufferUsageFlags): GPUBuffer {
  return device.createBuffer({ size, usage });
}

function encodeConfig(config: GpuSim3dConfig, step: number): ArrayBuffer {
  const buffer = new ArrayBuffer(64);
  const u32 = new Uint32Array(buffer);
  const f32View = new Float32Array(buffer);
  u32[0] = config.width;
  u32[1] = config.height;
  u32[2] = config.depth;
  u32[3] = config.particleCount;
  f32View[4] = config.dt;
  f32View[5] = config.speed;
  f32View[6] = config.turn;
  f32View[7] = config.sensorDistance;
  f32View[8] = config.diffusion;
  f32View[9] = config.decay;
  f32View[10] = config.depositRadius;
  f32View[11] = config.depositMass;
  f32View[12] = config.sigma;
  f32View[13] = step;
  return buffer;
}

async function readBuffer(device: GPUDevice, source: GPUBuffer, size: number): Promise<ArrayBuffer> {
  const readback = device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, readback, 0, size);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const out = readback.getMappedRange().slice(0);
  readback.unmap();
  readback.destroy();
  return out;
}

type Vec3 = [number, number, number];

function sampleField(field: Float32Array, config: GpuSim3dConfig, position: Vec3): number {
  return field[fieldIndex(Math.floor(position[0]), Math.floor(position[1]), Math.floor(position[2]), config)];
}

function fieldIndex(x: number, y: number, z: number, config: Pick<GpuSim3dConfig, "width" | "height" | "depth">): number {
  const ix = wrapInt(x, config.width);
  const iy = wrapInt(y, config.height);
  const iz = wrapInt(z, config.depth);
  return (iz * config.height + iy) * config.width + ix;
}

function wrapInt(value: number, size: number): number {
  return ((value % size) + size) % size;
}

function wrapFloat(value: number, size: number): number {
  return f32(value - Math.floor(value / size) * size);
}

function periodicDelta(a: number, b: number, size: number): number {
  let delta = a - b;
  delta -= Math.round(delta / size) * size;
  return delta;
}

function add(a: Vec3, b: Vec3): Vec3 {
  return [f32(a[0] + b[0]), f32(a[1] + b[1]), f32(a[2] + b[2])];
}

function scale(a: Vec3, s: number): Vec3 {
  return [f32(a[0] * s), f32(a[1] * s), f32(a[2] * s)];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    f32(a[1] * b[2] - a[2] * b[1]),
    f32(a[2] * b[0] - a[0] * b[2]),
    f32(a[0] * b[1] - a[1] * b[0])
  ];
}

function normalize(a: Vec3): Vec3 {
  const length = Math.hypot(a[0], a[1], a[2]);
  if (length <= 0.000001) return [1, 0, 0];
  return [f32(a[0] / length), f32(a[1] / length), f32(a[2] / length)];
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

function sum(values: Float32Array): number {
  let total = 0;
  for (const value of values) total += value;
  return total;
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number, message: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => reject(new Error(message)), timeoutMs);
    promise.then(
      (value) => {
        window.clearTimeout(timeout);
        resolve(value);
      },
      (error: unknown) => {
        window.clearTimeout(timeout);
        reject(error);
      }
    );
  });
}

function f32(value: number): number {
  return Math.fround(value);
}

const commonShaderSource = /* wgsl */ `
struct Particle {
  pos_species: vec4f,
  vel_id: vec4f,
};

struct Config {
  width: u32,
  height: u32,
  depth: u32,
  particle_count: u32,
  dt: f32,
  speed: f32,
  turn: f32,
  sensor_distance: f32,
  diffusion: f32,
  decay: f32,
  deposit_radius: f32,
  deposit_mass: f32,
  sigma: f32,
  step_index: f32,
  _pad0: f32,
  _pad1: f32,
};
`;

const particleShaderSource = /* wgsl */ `
${commonShaderSource}

@group(0) @binding(0) var<storage, read> particles_in: array<Particle>;
@group(0) @binding(1) var<storage, read> field_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> particles_out: array<Particle>;
@group(0) @binding(3) var<uniform> config: Config;

fn wrap_i(value: i32, size: u32) -> u32 {
  let s = i32(size);
  return u32(((value % s) + s) % s);
}

fn wrap_f(value: f32, size: f32) -> f32 {
  return value - floor(value / size) * size;
}

fn index_xyz(x: i32, y: i32, z: i32) -> u32 {
  let ix = wrap_i(x, config.width);
  let iy = wrap_i(y, config.height);
  let iz = wrap_i(z, config.depth);
  return (iz * config.height + iy) * config.width + ix;
}

fn sample_field(position: vec3f) -> f32 {
  return field_in[index_xyz(i32(floor(position.x)), i32(floor(position.y)), i32(floor(position.z)))];
}

fn safe_normalize(v: vec3f) -> vec3f {
  let len = length(v);
  if (len <= 0.000001) {
    return vec3f(1.0, 0.0, 0.0);
  }
  return v / len;
}

@compute @workgroup_size(64)
fn update_particles(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= config.particle_count) {
    return;
  }
  let p = particles_in[i];
  let position = p.pos_species.xyz;
  let species = p.pos_species.w;
  let forward = safe_normalize(p.vel_id.xyz);
  let side = safe_normalize(cross(forward, vec3f(0.37, 0.71, 0.13)));
  let up = safe_normalize(cross(side, forward));
  let left = sample_field(position + side * config.sensor_distance);
  let right = sample_field(position - side * config.sensor_distance);
  let upper = sample_field(position + up * config.sensor_distance);
  let steer = (left - right + (species - 1.0) * 0.025) * config.turn;
  let lift = (upper - (left + right) * 0.5) * config.turn * 0.35;
  let velocity = safe_normalize(forward + side * steer + up * lift);
  let next = position + velocity * config.speed * config.dt;
  let wrapped = vec3f(
    wrap_f(next.x, f32(config.width)),
    wrap_f(next.y, f32(config.height)),
    wrap_f(next.z, f32(config.depth))
  );
  particles_out[i].pos_species = vec4f(wrapped, species);
  particles_out[i].vel_id = vec4f(velocity, p.vel_id.w);
}
`;

const fieldShaderSource = /* wgsl */ `
${commonShaderSource}

@group(0) @binding(0) var<storage, read> particles_in: array<Particle>;
@group(0) @binding(1) var<storage, read> field_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> field_out: array<f32>;
@group(0) @binding(3) var<uniform> config: Config;

fn wrap_i(value: i32, size: u32) -> u32 {
  let s = i32(size);
  return u32(((value % s) + s) % s);
}

fn index_xyz(x: i32, y: i32, z: i32) -> u32 {
  let ix = wrap_i(x, config.width);
  let iy = wrap_i(y, config.height);
  let iz = wrap_i(z, config.depth);
  return (iz * config.height + iy) * config.width + ix;
}

fn periodic_delta(a: f32, b: f32, size: f32) -> f32 {
  var delta = a - b;
  delta = delta - round(delta / size) * size;
  return delta;
}

@compute @workgroup_size(64)
fn deposit_diffuse_field(@builtin(global_invocation_id) gid: vec3u) {
  let flat = gid.x;
  let voxel_count = config.width * config.height * config.depth;
  if (flat >= voxel_count) {
    return;
  }
  let x = flat % config.width;
  let y = (flat / config.width) % config.height;
  let z = flat / (config.width * config.height);
  let center = field_in[flat];
  let neighbors =
    field_in[index_xyz(i32(x) - 1, i32(y), i32(z))] +
    field_in[index_xyz(i32(x) + 1, i32(y), i32(z))] +
    field_in[index_xyz(i32(x), i32(y) - 1, i32(z))] +
    field_in[index_xyz(i32(x), i32(y) + 1, i32(z))] +
    field_in[index_xyz(i32(x), i32(y), i32(z) - 1)] +
    field_in[index_xyz(i32(x), i32(y), i32(z) + 1)];
  var value = max(0.0, (1.0 - 6.0 * config.diffusion) * center + config.diffusion * neighbors) * (1.0 - config.decay);
  let voxel = vec3f(f32(x) + 0.5, f32(y) + 0.5, f32(z) + 0.5);
  let radius2 = config.deposit_radius * config.deposit_radius;
  let sigma2 = 2.0 * config.sigma * config.sigma;
  for (var i = 0u; i < config.particle_count; i = i + 1u) {
    let particle = particles_in[i].pos_species.xyz;
    let dx = periodic_delta(voxel.x, particle.x, f32(config.width));
    let dy = periodic_delta(voxel.y, particle.y, f32(config.height));
    let dz = periodic_delta(voxel.z, particle.z, f32(config.depth));
    let d2 = dx * dx + dy * dy + dz * dz;
    if (d2 <= radius2) {
      value = value + config.deposit_mass * exp(-d2 / sigma2);
    }
  }
  field_out[flat] = value;
}
`;
