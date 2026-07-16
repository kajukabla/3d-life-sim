import { describe, expect, it, vi } from "vitest";
import {
  createBrowserAudio,
  type BrowserAudioDependencies,
  type BrowserAudioSnapshot
} from "./micAudio";
import type { AudioAnalysisFrame } from "./audioReactive";

function analysisFrame(): AudioAnalysisFrame {
  return {
    version: 1,
    sequence: 7,
    timestampSec: 0.25,
    sampleRate: 48_000,
    rms: 0.4,
    peak: 0.6,
    bands: {
      low: { value: 0.5 },
      mid: { value: 0.25 },
      high: { value: 0.1 }
    }
  };
}

function harness() {
  const track = {
    readyState: "live",
    stop: vi.fn(),
    getSettings: vi.fn(() => ({ deviceId: "mic-2" }))
  };
  const stream = {
    getTracks: vi.fn(() => [track]),
    getAudioTracks: vi.fn(() => [track])
  };
  const source = { connect: vi.fn(), disconnect: vi.fn() };
  const gain = { gain: { value: 1 }, connect: vi.fn(), disconnect: vi.fn() };
  const port = { onmessage: null as ((event: MessageEvent<unknown>) => void) | null, start: vi.fn(), close: vi.fn() };
  const worklet = { port, connect: vi.fn(), disconnect: vi.fn() };
  const context = {
    state: "running",
    sampleRate: 48_000,
    destination: {},
    audioWorklet: { addModule: vi.fn(async () => undefined) },
    createMediaStreamSource: vi.fn(() => source),
    createGain: vi.fn(() => gain),
    resume: vi.fn(async () => undefined),
    close: vi.fn(async () => undefined)
  };
  const mediaDevices = {
    getUserMedia: vi.fn(async () => stream),
    enumerateDevices: vi.fn(async () => [
      { kind: "audioinput", deviceId: "default", label: "Default microphone", groupId: "group-1" },
      { kind: "audioinput", deviceId: "mic-2", label: "Studio input", groupId: "group-2" },
      { kind: "videoinput", deviceId: "camera", label: "Camera", groupId: "group-3" }
    ]),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn()
  };
  const dependencies: BrowserAudioDependencies = {
    mediaDevices: mediaDevices as unknown as MediaDevices,
    createContext: vi.fn(() => context as unknown as AudioContext),
    createWorkletNode: vi.fn(() => worklet as unknown as AudioWorkletNode),
    workletUrl: "/assets/audio-worklet.js"
  };
  return { context, dependencies, gain, mediaDevices, port, source, stream, track, worklet };
}

describe("browser audio controller", () => {
  it("starts a selected microphone on an AudioWorklet and publishes frames", async () => {
    const kit = harness();
    const frames: AudioAnalysisFrame[] = [];
    const states: BrowserAudioSnapshot[] = [];
    const controller = createBrowserAudio({
      onFrame: (frame) => frames.push(frame),
      onState: (state) => states.push(state)
    }, kit.dependencies);

    await controller.start("mic-2");

    expect(kit.mediaDevices.getUserMedia).toHaveBeenCalledWith({
      audio: expect.objectContaining({ deviceId: { exact: "mic-2" } })
    });
    expect(kit.context.audioWorklet.addModule).toHaveBeenCalledWith("/assets/audio-worklet.js");
    expect(kit.dependencies.createWorkletNode).toHaveBeenCalledWith(
      kit.context,
      "life-audio-analyzer"
    );
    expect(kit.source.connect).toHaveBeenCalledWith(kit.worklet);
    expect(kit.worklet.connect).toHaveBeenCalledWith(kit.gain);
    expect(kit.gain.gain.value).toBe(0);
    expect(controller.snapshot()).toMatchObject({
      status: "running",
      activeDeviceId: "mic-2",
      frameCount: 0,
      devices: [
        { id: "default", label: "Default microphone" },
        { id: "mic-2", label: "Studio input" }
      ]
    });

    kit.port.onmessage?.({ data: analysisFrame() } as MessageEvent<unknown>);
    expect(frames).toHaveLength(1);
    expect(controller.snapshot().frameCount).toBe(1);
    expect(controller.snapshot().lastSequence).toBe(7);
    expect(states.at(-1)?.status).toBe("running");
  });

  it("stops the previous graph before switching inputs", async () => {
    const kit = harness();
    const controller = createBrowserAudio({ onFrame: () => {} }, kit.dependencies);

    await controller.start("mic-2");
    await controller.start("default");

    expect(kit.track.stop).toHaveBeenCalled();
    expect(kit.context.close).toHaveBeenCalled();
    expect(kit.mediaDevices.getUserMedia).toHaveBeenLastCalledWith({
      audio: expect.objectContaining({ deviceId: { exact: "default" } })
    });
  });

  it("turns permission failures into a useful denied state", async () => {
    const kit = harness();
    kit.mediaDevices.getUserMedia.mockRejectedValueOnce(Object.assign(new Error("Permission denied"), {
      name: "NotAllowedError"
    }));
    const controller = createBrowserAudio({ onFrame: () => {} }, kit.dependencies);

    await controller.start();

    expect(controller.snapshot()).toMatchObject({
      status: "denied",
      error: "Microphone permission was denied."
    });
  });
});
