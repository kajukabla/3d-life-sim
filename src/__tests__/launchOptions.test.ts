import { describe, expect, it } from "vitest";
import { audioReactiveUrlFromLaunch, chooseBootSettingsId, defaultAudioReactiveWsUrl, shouldStartPlaying } from "../launchOptions";

describe("launch options", () => {
  it("starts the simulation by default for normal and dev launch URLs", () => {
    expect(shouldStartPlaying("")).toBe(true);
    expect(shouldStartPlaying("?skipAppCompute=1")).toBe(true);
    expect(shouldStartPlaying("?skipAppCompute=1&playing=1")).toBe(true);
  });

  it("only pauses the simulation when explicitly requested", () => {
    expect(shouldStartPlaying("?playing=0")).toBe(false);
    expect(shouldStartPlaying("?skipAppCompute=1&playing=0")).toBe(false);
  });

  it("keeps backend audio opt-in while preserving explicit opt-out and env opt-in", () => {
    expect(audioReactiveUrlFromLaunch("", {})).toBeNull();
    expect(audioReactiveUrlFromLaunch("?skipAppCompute=1", {})).toBeNull();
    expect(audioReactiveUrlFromLaunch("?audio=0", {})).toBeNull();
    expect(audioReactiveUrlFromLaunch("?audioReactive=0", {})).toBeNull();
    expect(audioReactiveUrlFromLaunch("", { VITE_AUDIO_REACTIVE_DEFAULT: "0" })).toBeNull();
    expect(audioReactiveUrlFromLaunch("", { VITE_AUDIO_REACTIVE_DEFAULT: "1" })).toBe(defaultAudioReactiveWsUrl);
    expect(audioReactiveUrlFromLaunch("?audio=1", { VITE_AUDIO_REACTIVE_DEFAULT: "0" })).toBe(defaultAudioReactiveWsUrl);
  });

  it("uses a loopback custom audio websocket URL when supplied", () => {
    expect(audioReactiveUrlFromLaunch("?audio=1&audioWs=ws://127.0.0.1:49999", {})).toBe("ws://127.0.0.1:49999");
    expect(audioReactiveUrlFromLaunch("?audio=1&audioWs=ws://localhost:49999/audio", {})).toBe("ws://localhost:49999");
  });

  it("ignores non-loopback or non-ws audio websocket overrides", () => {
    expect(audioReactiveUrlFromLaunch("?audio=1&audioWs=wss://127.0.0.1:49999", {})).toBe(defaultAudioReactiveWsUrl);
    expect(audioReactiveUrlFromLaunch("?audio=1&audioWs=ws://example.com:49999", {})).toBe(defaultAudioReactiveWsUrl);
    expect(audioReactiveUrlFromLaunch("?audio=1&audioWs=notaurl", {})).toBe(defaultAudioReactiveWsUrl);
  });

  it("skips default saved settings for automation unless settings are explicit", () => {
    const candidates = [
      { id: "file-newdefault-json", name: "NewDefault" },
      { id: "file-ar10-json", name: "AR10" }
    ];

    expect(chooseBootSettingsId("", candidates)).toBe("file-newdefault-json");
    expect(chooseBootSettingsId("?skipAppCompute=1", candidates)).toBeNull();
    expect(chooseBootSettingsId("?profileGpu=1", candidates)).toBeNull();
    expect(chooseBootSettingsId("?skipAppCompute=1&settings=AR10", candidates)).toBe("file-ar10-json");
  });
});
