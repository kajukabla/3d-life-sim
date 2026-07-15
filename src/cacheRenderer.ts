import { CacheCloud } from "./cacheCloud";
import { resizeCanvasToDisplayResolution } from "./renderTarget";
import { requestHighPerformanceWebGpuAdapter } from "./webgpu";
import { configureHdrWebGpuCanvas, wideGamut2dSettings, type WebGpuCanvasColor } from "./webgpuCanvas";

export type ParticleBlendMode = "additive" | "alpha" | "opaque";
export type TrailColorMode = "stable" | "flow" | "thermal" | "tint";

export type RenderControls = {
  density: number;
  exposure: number;
  fov: number;
  aperture: number;
  focusDistance: number;
  dofBlur: number;
  dofEnabled: boolean;
  dofDebug: boolean;
  sceneBrightness: number;
  raySteps: number;
  rayResolution: number;
  fogTemporal: boolean;
  fogRenderScale: number;
  fogStepScale: number;
  fogTemporalBlend: number;
  fogBlueNoise: boolean;
  fieldTextureSampling: boolean;
  emptySpaceSkipping: boolean;
  emptySpaceThreshold: number;
  emptySpaceStride: number;
  particleSizePx: number;
  particleMinPx: number;
  particleMaxPx: number;
  particleOpacity: number;
  particleBrightness: number;
  particleColorMode: ParticleColorMode;
  particleVelocityStretch: boolean;
  particleStretch: number;
  particleStretchMin: number;
  particleStretchSpeed: number;
  particleSpeedCutoff: number;
  particleSlowCutoff: number;
  particleGlowCore: number;
  particleHotCore: number;
  particleBlendMode: ParticleBlendMode;
  particleExponent: number;
  particleBrightnessBoost: number;
  particleSupportSmoothing: number;
  particleHazeCull: number;
  particleDespeckle: number;
  particleDensityCutoff: number;
  particleDensityRadius: number;
  particleDensityNormalize: number;
  particleDensitySoftness: number;
  particleSupportMask: number;
  particleSupportRadius: number;
  particleSupportNeighbors: number;
  particleSupportFlow: number;
  fastParticleRender: boolean;
  fastNoBloomPost: boolean;
  // Render-side optimization toggles (default OFF = baseline). particleCutoffPrepass evaluates
  // the per-particle density-cutoff signal once in a compute prepass instead of 4x in the vertex
  // shader; densityLargeHalfRes builds the volume-density large/blurred channel at half resolution.
  particleCutoffPrepass: boolean;
  densityLargeHalfRes: boolean;
  densityPassStrength: number;
  densitySmallScale: number;
  densityLargeScale: number;
  densityLargeThreshold: number;
  densityContrastGain: number;
  densityContrastBalance: number;
  densityEmissionPower: number;
  densityOcclusion: number;
  accumulationStrength: number;
  accumulationRadius: number;
  accumulationCurve: number;
  accumulationMemory: number;
  accumulationNoiseReject: number;
  trailOpacity: number;
  fogBrightness: number;
  trailThreshold: number;
  trailColorMode: TrailColorMode;
  fogTint: string;
  particleTint: string;
  renderLayer: "both" | "particles" | "trails" | "density" | "volume-density" | "accumulation" | "debug-voxels";
  palette: "aurora" | "ember" | "spectral";
  filament: number;
  bloomStrength: number;
  bloomThreshold: number;
  bloomRadius: number;
  colorSaturation: number;
  colorContrast: number;
  chromaticAberration: number;
  vignetteStrength: number;
  vignetteSoftness: number;
  streakStrength: number;
  streakLength: number;
  streakVertical: number;
  flareHeight: number;
  flareCutoff: number;
  ribbonFraction: number;
  ribbonWidth: number;
  ribbonTaper: number;
  ribbonLength: number;
  ribbonJoints: number;
  ribbonFadeStart: number;
  ribbonEdgeFade: number;
  cameraYaw: number;
  cameraPitch: number;
  cameraDistance: number;
  cameraPanX: number;
  cameraPanY: number;
  // Per-particle variation + fractal-noise system. All default to a visual no-op (amounts 0).
  variationMaster: number;
  variationDrift: number;
  variationNoiseMix: number;
  variationFreq: number;
  variationOctaves: number;
  variationGain: number;
  variationLacunarity: number;
  variationSizeAmount: number;
  variationSizeCurve: number;
  variationSizeMin: number;
  variationSizeMax: number;
  variationBrightAmount: number;
  variationBrightCurve: number;
  variationBrightMin: number;
  variationBrightMax: number;
  variationOpacityAmount: number;
  variationOpacityCurve: number;
  variationOpacityMin: number;
  variationOpacityMax: number;
  variationColorAmount: number;
  variationColorCurve: number;
  variationColorMin: number;
  variationColorMax: number;
};

export type ParticleColorMode =
  | "solid"
  | "gradient-inferno"
  | "gradient-magma"
  | "gradient-viridis"
  | "gradient-turbo"
  | "gradient-rainbow"
  | "gradient-spectral"
  | "gradient-plasma"
  | "gradient-cosmic"
  | "gradient-ice"
  | "gradient-ember"
  | "velocity"
  | "velocity-inferno"
  | "velocity-viridis"
  | "velocity-spectral"
  | "velocity-cosmic"
  | "velocity-ice"
  | "audio-magma"
  | "audio-viridis"
  | "audio-turbo"
  | "audio-cosmic"
  | "audio-ice"
  | "audio-ember"
  | "audio-plasma"
  | "cohort"
  | "audio";

export type RenderDiagnostics = {
  renderer: "webgpu-point-volume" | "canvas-fallback";
  webgpu: boolean;
  pointCount: number;
  projectedCenter: [number, number];
  canvasColor?: WebGpuCanvasColor;
};

type Uniforms = {
  resolution: [number, number];
  pointCount: number;
  time: number;
  density: number;
  exposure: number;
  focusDistance: number;
  aperture: number;
  dofBlur: number;
  dofEnabled: number;
  overlay: number;
  palette: number;
  filament: number;
  yaw: number;
  panX: number;
  panY: number;
};

export class CacheRenderer {
  private device?: GPUDevice;
  private pipeline?: GPURenderPipeline;
  private bindGroupLayout?: GPUBindGroupLayout;
  private pointBuffer?: GPUBuffer;
  private uniformBuffer?: GPUBuffer;
  private pointCapacity = 0;
  private failed = false;
  private format?: GPUTextureFormat;
  private canvasColor?: WebGpuCanvasColor;

  async render(
    canvas: HTMLCanvasElement,
    cloud: CacheCloud,
    controls: RenderControls,
    overlay: boolean,
    time: number
  ): Promise<RenderDiagnostics> {
    resizeCanvasToDisplayResolution(canvas, controls.rayResolution);
    if (!this.failed && navigator.gpu) {
      try {
        await this.renderWebGpu(canvas, cloud, controls, overlay, time);
        return {
          renderer: "webgpu-point-volume",
          webgpu: true,
          pointCount: cloud.points.length,
          projectedCenter: projectedCenter(cloud, controls),
          canvasColor: this.canvasColor
        };
      } catch (error) {
        console.warn("WebGPU renderer failed, using canvas fallback", error);
        this.failed = true;
      }
    }
    renderCacheCanvasPresentation(canvas, cloud, controls, overlay);
    return {
      renderer: "canvas-fallback",
      webgpu: false,
      pointCount: cloud.points.length,
      projectedCenter: projectedCenter(cloud, controls)
    };
  }

  private async renderWebGpu(
    canvas: HTMLCanvasElement,
    cloud: CacheCloud,
    controls: RenderControls,
    overlay: boolean,
    time: number
  ): Promise<void> {
    if (!this.device) {
      const adapter = await requestHighPerformanceWebGpuAdapter(navigator.gpu);
      if (!adapter) throw new Error("No WebGPU adapter");
      this.device = await adapter.requestDevice();
    }
    const device = this.device;
    const context = canvas.getContext("webgpu");
    if (!context) throw new Error("No WebGPU canvas context");
    const canvasColor = configureHdrWebGpuCanvas(context, device);
    const format = canvasColor.format;
    if (this.format && this.format !== canvasColor.format) {
      this.pipeline = undefined;
    }
    this.format = format;
    this.canvasColor = canvasColor;
    if (!this.pipeline) {
      this.bindGroupLayout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }
        ]
      });
      this.pipeline = device.createRenderPipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
        vertex: {
          module: device.createShaderModule({ code: shaderSource }),
          entryPoint: "vs"
        },
        fragment: {
          module: device.createShaderModule({ code: shaderSource }),
          entryPoint: "fs",
          targets: [{ format }]
        },
        primitive: { topology: "triangle-list" }
      });
      this.uniformBuffer = device.createBuffer({
        size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      });
    }
    this.uploadPoints(device, cloud);
    const uniformValues = encodeUniforms({
      resolution: [canvas.width, canvas.height],
      pointCount: cloud.points.length,
      time,
      density: controls.density,
      exposure: controls.exposure,
      focusDistance: controls.focusDistance,
      aperture: controls.aperture,
      dofBlur: controls.dofBlur,
      dofEnabled: controls.dofEnabled ? 1 : 0,
      overlay: overlay ? 1 : 0,
      palette: controls.palette === "aurora" ? 0 : controls.palette === "ember" ? 1 : 2,
      filament: controls.filament,
      yaw: controls.cameraYaw,
      panX: controls.cameraPanX,
      panY: controls.cameraPanY
    });
    device.queue.writeBuffer(this.uniformBuffer!, 0, uniformValues);
    const bindGroup = device.createBindGroup({
      layout: this.bindGroupLayout!,
      entries: [
        { binding: 0, resource: { buffer: this.pointBuffer! } },
        { binding: 1, resource: { buffer: this.uniformBuffer! } }
      ]
    });
    const encoder = device.createCommandEncoder();
    const textureView = context.getCurrentTexture().createView();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: textureView,
          clearValue: { r: 0.006, g: 0.008, b: 0.01, a: 1 },
          loadOp: "clear",
          storeOp: "store"
        }
      ]
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  private uploadPoints(device: GPUDevice, cloud: CacheCloud): void {
    const packed = new Float32Array(Math.max(256, cloud.points.length) * 4);
    const scale = 1 / Math.max(...cloud.domain);
    cloud.points.forEach((point, index) => {
      packed[index * 4] = point.position[0] * scale;
      packed[index * 4 + 1] = point.position[1] * scale;
      packed[index * 4 + 2] = point.position[2] * scale;
      packed[index * 4 + 3] = point.density;
    });
    if (!this.pointBuffer || this.pointCapacity < packed.byteLength) {
      this.pointBuffer = device.createBuffer({
        size: packed.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });
      this.pointCapacity = packed.byteLength;
    }
    device.queue.writeBuffer(this.pointBuffer, 0, packed);
  }
}

function encodeUniforms(uniforms: Uniforms): ArrayBuffer {
  const values = new ArrayBuffer(64);
  const f32 = new Float32Array(values);
  const u32 = new Uint32Array(values);
  f32[0] = uniforms.resolution[0];
  f32[1] = uniforms.resolution[1];
  u32[2] = uniforms.pointCount;
  f32[3] = uniforms.time;
  f32[4] = uniforms.density;
  f32[5] = uniforms.exposure;
  f32[6] = uniforms.focusDistance;
  f32[7] = uniforms.aperture;
  f32[8] = uniforms.dofBlur;
  u32[9] = uniforms.overlay;
  u32[10] = uniforms.palette;
  f32[11] = uniforms.filament;
  f32[12] = uniforms.yaw;
  f32[13] = uniforms.panX;
  f32[14] = uniforms.panY;
  u32[15] = uniforms.dofEnabled;
  return values;
}

export function renderCacheCanvasPresentation(
  canvas: HTMLCanvasElement,
  cloud: CacheCloud,
  controls: RenderControls,
  overlay: boolean
): void {
  resizeCanvasToDisplayResolution(canvas, controls.rayResolution);
  const ctx = canvas.getContext("2d", wideGamut2dSettings);
  if (!ctx) return;
  ctx.fillStyle = "#050708";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.globalCompositeOperation = "screen";
  for (const point of cloud.points) {
    const projected = project(point.position, controls);
    const x = (projected[0] * 0.5 + 0.5) * canvas.width;
    const y = (-projected[1] * 0.5 + 0.5) * canvas.height;
    const unclampedRadius = (point.density * 4.5 + controls.particleSizePx) * 0.5;
    const minRadius = Math.min(controls.particleMinPx, controls.particleMaxPx);
    const maxRadius = Math.max(controls.particleMinPx, controls.particleMaxPx);
    const radius = Math.max(0.1, Math.min(maxRadius, Math.max(minRadius, unclampedRadius)));
    const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
    gradient.addColorStop(0, paletteCss(controls.palette, point.density, 0.9, controls.particleTint));
    gradient.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalCompositeOperation = "source-over";
  if (overlay) {
    ctx.strokeStyle = "rgba(255, 244, 204, 0.72)";
    ctx.lineWidth = 2;
    ctx.strokeRect(36, 36, canvas.width - 72, canvas.height - 72);
  }
}

function projectedCenter(cloud: CacheCloud, controls: RenderControls): [number, number] {
  let mx = 0;
  let my = 0;
  let mass = 0;
  for (const point of cloud.points) {
    const p = project(point.position, controls);
    mx += p[0] * point.density;
    my += p[1] * point.density;
    mass += point.density;
  }
  return [mx / mass, my / mass];
}

function project(position: [number, number, number], controls: RenderControls): [number, number, number] {
  const yaw = controls.cameraYaw;
  const scale = 1 / 32;
  const x = position[0] * scale;
  const y = position[1] * scale;
  const z = position[2] * scale;
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const rx = x * cy - z * sy;
  const rz = x * sy + z * cy;
  const perspective = 1.15 / (1.55 + rz);
  return [(rx + controls.cameraPanX) * perspective * 1.35, (y + controls.cameraPanY) * perspective * 1.35, perspective];
}

function paletteCss(palette: RenderControls["palette"], density: number, alpha: number, tint = "#ffffff"): string {
  const [tr, tg, tb] = parseHexColor(tint);
  const tinted = (value: number, channel: number) => Math.round(value * channel);
  if (palette === "ember") {
    return `rgba(${tinted(255, tr)},${tinted(92 + density * 120, tg)},${tinted(36 + density * 48, tb)},${alpha})`;
  }
  if (palette === "spectral") {
    return `rgba(${tinted(150 + density * 85, tr)},${tinted(90 + density * 130, tg)},${tinted(255, tb)},${alpha})`;
  }
  return `rgba(${tinted(80 + density * 120, tr)},${tinted(190 + density * 65, tg)},${tinted(180 + density * 70, tb)},${alpha})`;
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

const shaderSource = /* wgsl */ `
struct Point {
  pos_density: vec4f,
};

struct Uniforms {
  resolution: vec2f,
  point_count: u32,
  time: f32,
  density: f32,
  exposure: f32,
  focus_distance: f32,
  aperture: f32,
  dof_blur: f32,
  overlay: u32,
  palette: u32,
  filament: f32,
  yaw: f32,
  pan_x: f32,
  pan_y: f32,
  dof_enabled: u32,
};

@group(0) @binding(0) var<storage, read> points: array<Point>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

struct VertexOut {
  @builtin(position) position: vec4f,
};

@vertex
fn vs(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -3.0),
    vec2f(-1.0, 1.0),
    vec2f(3.0, 1.0)
  );
  var out: VertexOut;
  out.position = vec4f(positions[vertex_index], 0.0, 1.0);
  return out;
}

fn palette(kind: u32, density: f32, depth: f32) -> vec3f {
  if (kind == 1u) {
    return vec3f(1.0, 0.34 + density * 0.42, 0.10 + depth * 0.14);
  }
  if (kind == 2u) {
    return vec3f(0.48 + density * 0.44, 0.30 + depth * 0.45, 1.0);
  }
  return vec3f(0.20 + depth * 0.34, 0.74 + density * 0.24, 0.68 + density * 0.28);
}

fn rotate_project(p: vec3f) -> vec3f {
  let c = cos(uniforms.yaw);
  let s = sin(uniforms.yaw);
  let rx = p.x * c - p.z * s;
  let rz = p.x * s + p.z * c;
  let perspective = 1.15 / (1.55 + rz);
  return vec3f((rx + uniforms.pan_x) * perspective * 2.0, (p.y + uniforms.pan_y) * perspective * 2.0, perspective);
}

@fragment
fn fs(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let aspect = uniforms.resolution.x / uniforms.resolution.y;
  var uv = position.xy / uniforms.resolution * 2.0 - vec2f(1.0, 1.0);
  uv.x = uv.x * aspect;
  uv.y = -uv.y;
  var color = vec3f(0.006, 0.009, 0.011);
  var transmittance = 1.0;
  let count = min(uniforms.point_count, 256u);
  for (var i = 0u; i < count; i = i + 1u) {
    let point = points[i].pos_density;
    let projected = rotate_project(point.xyz);
    let focus_depth = clamp(abs(projected.z - uniforms.focus_distance) * max(0.0, uniforms.aperture) * 8.0, 0.0, 1.0);
    let blur = select(0.0, focus_depth * max(0.0, uniforms.dof_blur) * 0.028, uniforms.dof_enabled == 1u);
    let radius = 0.024 + point.w * (0.036 + uniforms.filament * 0.018) + blur;
    let d = length(uv - projected.xy);
    let splat = exp(-(d * d) / max(0.00002, radius * radius));
    let density = splat * point.w * uniforms.density * 0.62;
    color = color + palette(uniforms.palette, point.w, projected.z) * density * transmittance;
    transmittance = max(0.24, transmittance - density * 0.006);
  }
  let vignette = 1.0 - smoothstep(0.58, 1.42, length(uv));
  color = color * uniforms.exposure;
  color = color * (0.25 + vignette * 0.85);
  if (uniforms.overlay == 1u) {
    let border = step(0.93, max(abs(uv.x / aspect), abs(uv.y)));
    color = mix(color, vec3f(1.0, 0.87, 0.52), border * 0.45);
  }
  return vec4f(pow(max(color, vec3f(0.0)), vec3f(1.0 / 2.2)), 1.0);
}
`;
