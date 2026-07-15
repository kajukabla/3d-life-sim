export const hdrCanvasFormat: GPUTextureFormat = "rgba16float";
export const hdrCanvasColorSpace: PredefinedColorSpace = "display-p3";
export const hdrCanvasToneMappingMode: GPUCanvasToneMappingMode = "extended";
export const wideGamut2dSettings: CanvasRenderingContext2DSettings = { alpha: false, colorSpace: hdrCanvasColorSpace };

export type WebGpuCanvasColor = {
  format: GPUTextureFormat;
  colorSpace: PredefinedColorSpace;
  toneMappingMode: GPUCanvasToneMappingMode;
  hdr: boolean;
  fallback: boolean;
};

export function configureHdrWebGpuCanvas(context: GPUCanvasContext, device: GPUDevice): WebGpuCanvasColor {
  // Add COPY_SRC so verification tests can read the canvas back to CPU as linear HDR
  // and avoid browser PNG conversion variance in brightness measurements.
  const readbackUsage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC;
  if (shouldUseSdrCanvas()) {
    const fallbackFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
      device,
      format: fallbackFormat,
      alphaMode: "opaque",
      colorSpace: "srgb",
      usage: readbackUsage
    });
    return readCanvasColor(context, fallbackFormat, true);
  }
  const hdrConfiguration: GPUCanvasConfiguration = {
    device,
    format: hdrCanvasFormat,
    alphaMode: "opaque",
    colorSpace: hdrCanvasColorSpace,
    toneMapping: { mode: hdrCanvasToneMappingMode },
    usage: readbackUsage
  };
  try {
    context.configure(hdrConfiguration);
    return readCanvasColor(context, hdrCanvasFormat, false);
  } catch (error) {
    console.warn("HDR WebGPU canvas configuration failed, falling back to preferred canvas format", error);
    const fallbackFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
      device,
      format: fallbackFormat,
      alphaMode: "opaque",
      colorSpace: hdrCanvasColorSpace,
      toneMapping: { mode: hdrCanvasToneMappingMode },
      usage: readbackUsage
    });
    return readCanvasColor(context, fallbackFormat, true);
  }
}

function shouldUseSdrCanvas(): boolean {
  if (typeof window === "undefined") return false;
  return new URLSearchParams(window.location.search).get("sdrCanvas") === "1";
}

function readCanvasColor(context: GPUCanvasContext, format: GPUTextureFormat, fallback: boolean): WebGpuCanvasColor {
  const configuration = context.getConfiguration();
  const colorSpace = configuration?.colorSpace ?? hdrCanvasColorSpace;
  const confirmedToneMappingMode = configuration?.toneMapping?.mode;
  const toneMappingMode = confirmedToneMappingMode ?? "standard";
  return {
    format,
    colorSpace,
    toneMappingMode,
    hdr: format === hdrCanvasFormat && colorSpace === hdrCanvasColorSpace && confirmedToneMappingMode === hdrCanvasToneMappingMode,
    fallback
  };
}
