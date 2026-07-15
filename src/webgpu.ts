export type WebGpuDiagnostics = {
  checked: boolean;
  available: boolean;
  adapterOk: boolean;
  deviceOk: boolean;
  bufferRoundtripOk: boolean;
  adapterInfo: string;
  error?: string;
};

export const highPerformanceWebGpuAdapterOptions: GPURequestAdapterOptions = {
  powerPreference: "high-performance"
};

export function requestHighPerformanceWebGpuAdapter(gpu: GPU): Promise<GPUAdapter | null> {
  return gpu.requestAdapter(highPerformanceWebGpuAdapterOptions);
}

export function describeWebGpuAdapter(adapter: GPUAdapter): string {
  const info = adapter.info;
  const parts = [
    info?.description,
    info?.vendor,
    info?.architecture,
    info?.device
  ].filter((value): value is string => typeof value === "string" && value.trim().length > 0);
  return parts.length > 0 ? [...new Set(parts)].join(" / ") : "adapter-present";
}

export async function probeWebGpu(): Promise<WebGpuDiagnostics> {
  const gpu = navigator.gpu;
  if (!gpu) {
    return {
      checked: true,
      available: false,
      adapterOk: false,
      deviceOk: false,
      bufferRoundtripOk: false,
      adapterInfo: "navigator.gpu unavailable",
      error: "navigator.gpu unavailable"
    };
  }
  try {
    const adapter = await requestHighPerformanceWebGpuAdapter(gpu);
    if (!adapter) {
      return {
        checked: true,
        available: true,
        adapterOk: false,
        deviceOk: false,
        bufferRoundtripOk: false,
        adapterInfo: "no adapter returned",
        error: "no adapter returned"
      };
    }
    const device = await adapter.requestDevice();
    const input = new Uint32Array([0x46554f44, 7, 11, 13]);
    const buffer = device.createBuffer({
      size: input.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    const readback = device.createBuffer({
      size: input.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    device.queue.writeBuffer(buffer, 0, input);
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, readback, 0, input.byteLength);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const output = new Uint32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    return {
      checked: true,
      available: true,
      adapterOk: true,
      deviceOk: true,
      bufferRoundtripOk: output.every((value, index) => value === input[index]),
      adapterInfo: describeWebGpuAdapter(adapter)
    };
  } catch (error) {
    return {
      checked: true,
      available: true,
      adapterOk: false,
      deviceOk: false,
      bufferRoundtripOk: false,
      adapterInfo: "probe failed",
      error: error instanceof Error ? error.message : String(error)
    };
  }
}
