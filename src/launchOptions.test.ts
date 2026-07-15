import { describe, expect, it } from "vitest";
import {
  audioMicFromLaunch,
  audioReactiveUrlFromLaunch,
  chooseBootSettingsId,
  defaultAudioReactiveWsUrl,
  demoLaunchFromSearch,
  parallelPipelinesFromLaunch
} from "./launchOptions";

describe("parallelPipelinesFromLaunch", () => {
  it("defaults OFF (serial pipeline compile) without the param", () => {
    expect(parallelPipelinesFromLaunch("")).toBe(false);
    expect(parallelPipelinesFromLaunch("?particles=24000")).toBe(false);
  });

  it("?parallelPipelines=1 enables concurrent compilation", () => {
    expect(parallelPipelinesFromLaunch("?parallelPipelines=1")).toBe(true);
  });

  it("any other value stays OFF", () => {
    expect(parallelPipelinesFromLaunch("?parallelPipelines=0")).toBe(false);
    expect(parallelPipelinesFromLaunch("?parallelPipelines=yes")).toBe(false);
  });
});

describe("demoLaunchFromSearch", () => {
  it("?demo=instant enables both instant and idle demo", () => {
    expect(demoLaunchFromSearch("?demo=instant")).toEqual({ instant: true, idle: true });
  });

  it("no demo param leaves both off", () => {
    expect(demoLaunchFromSearch("")).toEqual({ instant: false, idle: false });
    expect(demoLaunchFromSearch("?preset=shimmer-orb")).toEqual({ instant: false, idle: false });
  });

  it("unknown demo value leaves both off", () => {
    expect(demoLaunchFromSearch("?demo=later")).toEqual({ instant: false, idle: false });
  });
});

describe("chooseBootSettingsId", () => {
  const candidates = [
    { id: "file-newdefault-json", name: "NewDefault" },
    { id: "file-ar10-json", name: "AR10" },
    { id: "file-ar11-json", name: "AR11" },
    { id: "abc-123", name: "GooWalls" }
  ];

  it("defaults to NewDefault with no params", () => {
    expect(chooseBootSettingsId("", candidates)).toBe("file-newdefault-json");
  });

  it("falls back to AR10 when NewDefault is absent", () => {
    const noNewDefault = candidates.filter((c) => c.name !== "NewDefault");
    expect(chooseBootSettingsId("", noNewDefault)).toBe("file-ar10-json");
  });

  it("?settings=AR11 selects AR11 by name (case-insensitive)", () => {
    expect(chooseBootSettingsId("?settings=AR11", candidates)).toBe("file-ar11-json");
    expect(chooseBootSettingsId("?settings=ar11", candidates)).toBe("file-ar11-json");
  });

  it("?settings can select by id too", () => {
    expect(chooseBootSettingsId("?settings=abc-123", candidates)).toBe("abc-123");
  });

  it("unknown or empty ?settings applies nothing (no AR10 fallback)", () => {
    expect(chooseBootSettingsId("?settings=Nope", candidates)).toBeNull();
    expect(chooseBootSettingsId("?settings=", candidates)).toBeNull();
  });

  it("?profileGpu skips any default", () => {
    expect(chooseBootSettingsId("?profileGpu", candidates)).toBeNull();
  });

  it("?settings wins even when profileGpu is absent and AR10 exists", () => {
    expect(chooseBootSettingsId("?settings=GooWalls", candidates)).toBe("abc-123");
  });
});

describe("audioMicFromLaunch", () => {
  it("?audio=mic enables the in-browser mic analyzer", () => {
    expect(audioMicFromLaunch("?audio=mic")).toBe(true);
  });

  it("other audio values do not (standalone)", () => {
    expect(audioMicFromLaunch("")).toBe(false);
    expect(audioMicFromLaunch("?audio=1")).toBe(false);
    expect(audioMicFromLaunch("?audio=0")).toBe(false);
  });

  it("embedded mode (?embed=1 from the bay scheduler) defaults to mic with no audio param", () => {
    expect(audioMicFromLaunch("?embed=1")).toBe(true);
    expect(audioMicFromLaunch("?embed=1&mode=viewer&demo=instant")).toBe(true);
  });

  it("embedded mode treats an explicit ?audio=1 as mic too — the helper is never reachable from a visitor", () => {
    expect(audioMicFromLaunch("?embed=1&audio=1")).toBe(true);
  });

  it("?audio=0 disables audio entirely, embedded or not", () => {
    expect(audioMicFromLaunch("?embed=1&audio=0")).toBe(false);
  });
});

describe("audioReactiveUrlFromLaunch — the Rust helper WebSocket", () => {
  it("?audio=mic never opens the helper WebSocket, even with the env default on", () => {
    expect(audioReactiveUrlFromLaunch("?audio=mic", { VITE_AUDIO_REACTIVE_DEFAULT: "1" })).toBeNull();
  });

  it("embedded mode never opens the helper WebSocket regardless of params or env default", () => {
    expect(audioReactiveUrlFromLaunch("?embed=1", { VITE_AUDIO_REACTIVE_DEFAULT: "1" })).toBeNull();
    expect(audioReactiveUrlFromLaunch("?embed=1&audio=1", { VITE_AUDIO_REACTIVE_DEFAULT: "1" })).toBeNull();
  });

  it("standalone ?audio=1 still uses the WebSocket backend", () => {
    expect(audioReactiveUrlFromLaunch("?audio=1", {})).toBe(defaultAudioReactiveWsUrl);
  });
});
