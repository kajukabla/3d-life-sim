import { afterEach, describe, expect, it } from "vitest";
import { audioExtensionFor, createPerformanceAudioRecorder } from "../performanceAudioRecorder";

describe("audioExtensionFor", () => {
  it("maps mime types to sensible file extensions", () => {
    expect(audioExtensionFor("audio/webm;codecs=opus")).toBe("webm");
    expect(audioExtensionFor("audio/ogg;codecs=opus")).toBe("ogg");
    expect(audioExtensionFor("audio/mp4")).toBe("m4a");
    expect(audioExtensionFor("")).toBe("webm");
  });
});

describe("createPerformanceAudioRecorder without MediaRecorder (node env)", () => {
  afterEach(() => {
    delete (globalThis as { MediaRecorder?: unknown }).MediaRecorder;
  });

  it("start() returns false and stays idle when MediaRecorder is unavailable", () => {
    const rec = createPerformanceAudioRecorder();
    const fakeStream = { getAudioTracks: () => [{ readyState: "live" }] } as unknown as MediaStream;
    expect(rec.start(fakeStream, 0)).toBe(false);
    expect(rec.isRecording()).toBe(false);
  });

  it("stop() resolves null when nothing was recorded", async () => {
    const rec = createPerformanceAudioRecorder();
    await expect(rec.stop(0)).resolves.toBeNull();
  });

  it("start() returns false for a stream with no audio tracks", () => {
    // Stub a minimal MediaRecorder so we get past the availability check to the track check.
    (globalThis as { MediaRecorder?: unknown }).MediaRecorder = class {
      static isTypeSupported() { return false; }
    };
    const rec = createPerformanceAudioRecorder();
    const noAudio = { getAudioTracks: () => [] } as unknown as MediaStream;
    expect(rec.start(noAudio, 0)).toBe(false);
  });
});
