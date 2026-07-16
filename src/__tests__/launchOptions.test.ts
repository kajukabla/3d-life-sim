import { describe, expect, it } from "vitest";
import { audioMicFromLaunch, chooseBootSettingsId, shouldStartPlaying } from "../launchOptions";

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

  it("keeps microphone capture user-initiated unless explicitly requested", () => {
    expect(audioMicFromLaunch("")).toBe(false);
    expect(audioMicFromLaunch("?skipAppCompute=1")).toBe(false);
    expect(audioMicFromLaunch("?audio=0")).toBe(false);
    expect(audioMicFromLaunch("?audio=mic")).toBe(true);
    expect(audioMicFromLaunch("?audio=1")).toBe(true);
  });

  it("skips default saved settings for automation unless settings are explicit", () => {
    const candidates = [
      { id: "file-default-json", name: "Default" },
      { id: "file-viridian-aurora-json", name: "Viridian Aurora" }
    ];

    expect(chooseBootSettingsId("", candidates)).toBe("file-default-json");
    expect(chooseBootSettingsId("?skipAppCompute=1", candidates)).toBeNull();
    expect(chooseBootSettingsId("?profileGpu=1", candidates)).toBeNull();
    expect(chooseBootSettingsId("?skipAppCompute=1&settings=AR11", candidates)).toBe("file-viridian-aurora-json");
    expect(chooseBootSettingsId("?settings=Viridian%20Aurora", candidates)).toBe("file-viridian-aurora-json");
  });
});
