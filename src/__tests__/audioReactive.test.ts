import { describe, expect, it } from "vitest";
import {
  applyAudioReactiveFrame,
  connectAudioReactiveSocket,
  defaultAudioReactiveMapping,
  normalizeAudioSignal,
  parseAudioAnalysisFrame,
  parseAudioBackendMessage,
  type AudioAnalysisFrame,
  type AudioReactiveMapping
} from "../audioReactive";
import { getLivePreset } from "../livePresets";
import { defaultLiveGpu3dConfig } from "../realtimeGpuSim3d";

const baseRender = {
  bloomStrength: 0.5,
  particleBrightness: 1,
  fogBrightness: 0.6
};

function frame(value: number): AudioAnalysisFrame {
  const parsed = parseAudioAnalysisFrame({
    version: 1,
    sequence: 7,
    timestampSec: 1.25,
    sampleRate: 48_000,
    rms: value,
    peak: value,
    bands: {
      low: { value },
      bass: { value },
      mid: { value: 0.5 },
      high: { value: 1 - value }
    }
  });
  if (!parsed) throw new Error("test frame did not parse");
  return parsed;
}

describe("audio reactive mapping", () => {
  it("parses backend analysis frames into normalized signals", () => {
    const parsed = frame(0.4);

    expect(parsed?.version).toBe(1);
    expect(normalizeAudioSignal(parsed, "bass")).toBeCloseTo(0.4);
    expect(normalizeAudioSignal(parsed, "high")).toBeCloseTo(0.6);
    expect(normalizeAudioSignal(parsed, "rms")).toBeCloseTo(0.4);
    expect(normalizeAudioSignal(parsed, "missing")).toBe(0);
  });

  it("maps normalized band values onto render and live parameter ranges", () => {
    const mapping: AudioReactiveMapping = {
      id: "test",
      rules: [
        { source: "bass", target: "render.bloomStrength", range: [0.2, 1.2] },
        { source: "high", target: "live.depositMass", range: [0.5, 2.5], curve: 2 }
      ]
    };

    const result = applyAudioReactiveFrame(frame(0.5), mapping, {
      render: baseRender,
      live: defaultLiveGpu3dConfig
    });

    expect(result.render.bloomStrength).toBeCloseTo(0.7);
    expect(result.live.depositMass).toBeCloseTo(1);
  });

  it("uses attack and decay smoothing per target without mutating base state", () => {
    const mapping: AudioReactiveMapping = {
      id: "smooth",
      rules: [{ source: "bass", target: "render.bloomStrength", range: [0, 1], attackMs: 1, decayMs: 200 }]
    };
    const state = { render: baseRender, live: defaultLiveGpu3dConfig };

    const first = applyAudioReactiveFrame(frame(1), mapping, state, { previousValues: {}, dtSec: 1 / 60 });
    const second = applyAudioReactiveFrame(frame(0), mapping, state, {
      previousValues: first.nextValues,
      dtSec: 1 / 60
    });

    expect(first.render.bloomStrength).toBeGreaterThan(0.99);
    expect(second.render.bloomStrength).toBeGreaterThan(0.9);
    expect(baseRender.bloomStrength).toBe(0.5);
  });

  it("treats one millisecond attack and decay as raw target values", () => {
    const mapping: AudioReactiveMapping = {
      id: "raw",
      rules: [{ source: "bass", target: "render.bloomStrength", range: [0, 1], attackMs: 1, decayMs: 1 }]
    };
    const state = { render: baseRender, live: defaultLiveGpu3dConfig };

    const first = applyAudioReactiveFrame(frame(1), mapping, state, { previousValues: {}, dtSec: 1 / 750 });
    const second = applyAudioReactiveFrame(frame(0), mapping, state, {
      previousValues: first.nextValues,
      dtSec: 1 / 750
    });

    expect(first.render.bloomStrength).toBe(1);
    expect(second.render.bloomStrength).toBe(0);
  });

  it("ships a useful default low/mid/high mapping for the live app", () => {
    const result = applyAudioReactiveFrame(frame(0.8), defaultAudioReactiveMapping, {
      render: baseRender,
      live: defaultLiveGpu3dConfig
    });

    expect(result.render.bloomStrength).toBeGreaterThan(baseRender.bloomStrength);
    expect(result.live.depositMass).toBeGreaterThan(defaultLiveGpu3dConfig.depositMass);
  });

  it("allows curated presets to override the audio mapping", () => {
    const preset = getLivePreset("volume-tendrils");

    expect(preset.audioMapping?.id).toBe("volume-tendrils-audio");
    expect(preset.audioMapping?.rules.some((rule) => rule.target === "render.densityPassStrength")).toBe(true);
  });

  it("parses backend input devices separately from audio frames", () => {
    const message = parseAudioBackendMessage({
      type: "audioDevices",
      activeInput: "Universal Audio Thunderbolt",
      devices: [
        { id: "Universal Audio Thunderbolt", name: "Universal Audio Thunderbolt", isDefault: false, channels: 32, sampleRate: 48000, sampleFormat: "F32" },
        { id: "MacBook Pro Microphone", name: "MacBook Pro Microphone", isDefault: true, channels: 1, sampleRate: 48000, sampleFormat: "F32" }
      ]
    });

    expect(message?.type).toBe("audioDevices");
    if (message?.type !== "audioDevices") throw new Error("expected devices message");
    expect(message.activeInput).toBe("Universal Audio Thunderbolt");
    expect(message.devices).toHaveLength(2);
    expect(message.devices[0].channels).toBe(32);
    expect(parseAudioAnalysisFrame(message)).toBe(null);
  });

  it("requests devices and sends input switch commands over the socket", () => {
    const originalWebSocket = globalThis.WebSocket;
    const sockets: any[] = [];
    const seenDevices: string[][] = [];
    class FakeWebSocket {
      static OPEN = 1;
      readyState = FakeWebSocket.OPEN;
      sent: string[] = [];
      listeners: Record<string, Array<(event: any) => void>> = {};
      constructor(public url: string) {
        sockets.push(this);
      }
      addEventListener(type: string, listener: (event: any) => void) {
        this.listeners[type] = [...(this.listeners[type] ?? []), listener];
      }
      send(payload: string) {
        this.sent.push(payload);
      }
      close() {
        this.readyState = 3;
      }
      emit(type: string, event: any = {}) {
        for (const listener of this.listeners[type] ?? []) listener(event);
      }
    }
    globalThis.WebSocket = FakeWebSocket as unknown as typeof WebSocket;
    try {
      const controller = connectAudioReactiveSocket({
        url: "ws://127.0.0.1:47831",
        mapping: { id: "none", rules: [] },
        getState: () => ({ render: {}, live: {} }),
        apply: () => {},
        onDevices: (_active, devices) => seenDevices.push(devices.map((device) => device.name))
      });
      const socket = sockets[0];
      socket.emit("open");
      socket.emit("message", {
        data: JSON.stringify({
          type: "audioDevices",
          activeInput: "Universal Audio Thunderbolt",
          devices: [{ id: "Universal Audio Thunderbolt", name: "Universal Audio Thunderbolt", isDefault: false }]
        })
      });
      controller.setInput("MacBook Pro Microphone");
      controller.requestDevices();

      expect((socket.sent as string[]).map((value: string) => JSON.parse(value))).toEqual([
        { type: "listDevices" },
        { type: "setInput", input: "MacBook Pro Microphone" },
        { type: "listDevices" }
      ]);
      expect(seenDevices).toEqual([["Universal Audio Thunderbolt"]]);
      controller.close();
    } finally {
      globalThis.WebSocket = originalWebSocket;
    }
  });
});
