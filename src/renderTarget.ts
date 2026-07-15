export const minRayResolution = 360;
export const maxRayResolution = 2160;
export const defaultRayResolution = 1080;
export const rayResolutionStep = 120;

export function clampRayResolution(value: number): number {
  const finite = Number.isFinite(value) ? value : defaultRayResolution;
  const clamped = Math.min(maxRayResolution, Math.max(minRayResolution, finite));
  return Math.round(clamped / rayResolutionStep) * rayResolutionStep;
}

// aspectOverride (width / height) forces the render aspect instead of deriving it from the canvas's
// CSS layout box. The offline HDR exporter uses it to render an exact 9:16 portrait regardless of the
// replay browser's window shape; live rendering passes nothing and follows the canvas as before.
export function resizeCanvasToDisplayResolution(
  canvas: HTMLCanvasElement,
  rayResolution: number,
  aspectOverride?: number
): [number, number] {
  const rect = canvas.getBoundingClientRect();
  const clientWidth = Math.max(1, rect.width || canvas.clientWidth || canvas.width || 1);
  const clientHeight = Math.max(1, rect.height || canvas.clientHeight || canvas.height || 1);
  const pixelRatio = Math.max(1, window.devicePixelRatio || 1);
  const hiddenOrProbeCanvas = clientWidth <= 8 || clientHeight <= 8;
  // A forced aspect always wins (even for a hidden/probe canvas) so offline replay is deterministic.
  const aspect = aspectOverride && aspectOverride > 0 ? aspectOverride : clientWidth / clientHeight;
  const height = hiddenOrProbeCanvas && !(aspectOverride && aspectOverride > 0)
    ? Math.max(1, Math.round(clientHeight * pixelRatio))
    : clampRayResolution(rayResolution);
  const width = hiddenOrProbeCanvas && !(aspectOverride && aspectOverride > 0)
    ? Math.max(1, Math.round(clientWidth * pixelRatio))
    : Math.max(1, Math.round(height * aspect));

  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;

  return [width, height];
}
