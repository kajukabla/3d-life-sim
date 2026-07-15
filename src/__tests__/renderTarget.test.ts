import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { resizeCanvasToDisplayResolution } from "../renderTarget";

// resizeCanvasToDisplayResolution reads window.devicePixelRatio and the canvas's CSS box. The unit
// env is "node" (no DOM), so we stub a minimal window + a fake canvas whose layout box we control.
function fakeCanvas(clientWidth: number, clientHeight: number) {
  return {
    width: 0,
    height: 0,
    clientWidth,
    clientHeight,
    getBoundingClientRect: () => ({ width: clientWidth, height: clientHeight })
  } as unknown as HTMLCanvasElement;
}

beforeEach(() => {
  (globalThis as { window?: unknown }).window = { devicePixelRatio: 1 };
});
afterEach(() => {
  delete (globalThis as { window?: unknown }).window;
});

describe("resizeCanvasToDisplayResolution", () => {
  it("derives width from the canvas CSS aspect when no override is given (live behavior)", () => {
    // 16:9 landscape box at rayResolution 1080 -> 1920x1080.
    const canvas = fakeCanvas(1600, 900);
    const [w, h] = resizeCanvasToDisplayResolution(canvas, 1080);
    expect([w, h]).toEqual([1920, 1080]);
    expect([canvas.width, canvas.height]).toEqual([1920, 1080]);
  });

  it("forces 9:16 portrait from an aspect override regardless of the CSS box", () => {
    // Landscape window, but a 9/16 override at height 1920 must yield exactly 1080x1920.
    const canvas = fakeCanvas(1600, 900);
    const [w, h] = resizeCanvasToDisplayResolution(canvas, 1920, 9 / 16);
    expect([w, h]).toEqual([1080, 1920]);
  });

  it("honors the override even for a hidden/probe-sized canvas (deterministic offline replay)", () => {
    // A <=8px canvas normally falls back to its pixel box; an override must still win so headless
    // export is deterministic.
    const canvas = fakeCanvas(1, 1);
    const [w, h] = resizeCanvasToDisplayResolution(canvas, 1920, 9 / 16);
    expect([w, h]).toEqual([1080, 1920]);
  });

  it("ignores a zero/negative override and follows the canvas", () => {
    const canvas = fakeCanvas(1600, 900);
    expect(resizeCanvasToDisplayResolution(canvas, 1080, 0)).toEqual([1920, 1080]);
  });
});
