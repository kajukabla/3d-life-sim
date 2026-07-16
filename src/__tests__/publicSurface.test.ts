import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const appSource = readFileSync(new URL("../App.tsx", import.meta.url), "utf8");
const launchSource = readFileSync(new URL("../launchOptions.ts", import.meta.url), "utf8");
const audioSource = readFileSync(new URL("../audioReactive.ts", import.meta.url), "utf8");

describe("public particle-only surface", () => {
  it("does not expose cache, alternate render-mode, or Hue controls", () => {
    expect(appSource).toContain('renderLayer: "particles"');
    expect(appSource).toContain('return { ...controls, renderLayer: "particles", ribbonFraction: 0 };');
    expect(appSource).not.toContain("render-layer-select");
    expect(appSource).not.toContain('<ControlGroup title="Density">');
    expect(appSource).not.toContain('<ControlGroup title="Accumulation">');
    expect(appSource).not.toContain("lighting-editor");
    expect(appSource).not.toContain("useHueLighting");
    expect(appSource).not.toContain("loadCacheCloud");
  });

  it("starts every remaining cockpit section collapsed", () => {
    expect(appSource).toContain('<details className="control-group control-group-collapsible">');
    expect(appSource).toContain('<details className="audio-panel control-group-collapsible"');
    expect(appSource).toContain('<details className="midi-panel control-group-collapsible"');
    expect(appSource).not.toContain("function CollapsibleGroup");
  });

  it("keeps the cockpit header focused on built-in presets", () => {
    const settings = appSource.slice(
      appSource.indexOf('<ControlGroup title="Settings">'),
      appSource.indexOf('<ControlGroup title="Particles">')
    );

    expect(appSource).toContain('data-testid="preset-select"');
    expect(settings).toContain('data-testid="play-toggle"');
    expect(settings).toContain('data-testid="save-settings"');
    expect(settings).toContain('data-testid="import-settings"');
    expect(settings).not.toContain('data-testid="export-settings"');
    expect(appSource).not.toContain('data-testid="saved-settings-select"');
    for (const testId of [
      "timeline-toggle",
      "timeline-bar",
      "reset",
      "reset-view",
      "render-preview",
      "portrait-mode",
      "record-performance",
      "record-midi-learn",
      "randomize-all"
    ]) {
      expect(appSource).not.toContain(`data-testid="${testId}"`);
    }
  });

  it("ends the public cockpit at MIDI", () => {
    expect(appSource).not.toContain('<ControlGroup title="Track">');
    expect(appSource).not.toContain('data-testid="track-release"');
    expect(appSource).not.toContain('className="metrics-grid"');
    expect(appSource).not.toContain('className="diagnostics-panel"');
    expect(appSource).not.toContain('className="preset-json"');
  });

  it("uses browser-only audio without a native helper or WebSocket", () => {
    expect(appSource).toContain('data-testid="audio-capture-toggle"');
    expect(appSource).toContain("Start microphone");
    expect(appSource).not.toContain("connectAudioReactiveSocket");
    expect(appSource).not.toContain("audioReactiveUrlFromLaunch");
    expect(launchSource).not.toContain("audioWs");
    expect(launchSource).not.toContain("127.0.0.1:47831");
    expect(audioSource).not.toContain("new WebSocket");
  });

  it("ships only the renamed curated presets without MIDI mappings or camera locks", () => {
    const files = new Map([
      ["default.json", "Default"],
      ["electric-current.json", "Electric Current"],
      ["ice-flower.json", "Ice Flower"],
      ["viridian-aurora.json", "Viridian Aurora"]
    ]);

    for (const [fileName, name] of files) {
      const raw = readFileSync(new URL(`../../Presets/${fileName}`, import.meta.url), "utf8");
      const preset = JSON.parse(raw) as {
        name?: string;
        controls?: { renderLayer?: string; ribbonFraction?: number };
        ui?: { viewLocked?: boolean };
      };

      expect(preset.name, fileName).toBe(name);
      expect(preset.controls?.renderLayer, fileName).toBe("particles");
      expect(preset.controls?.ribbonFraction, fileName).toBe(0);
      expect(preset.ui?.viewLocked, fileName).toBe(false);
      expect(raw.toLowerCase(), fileName).not.toContain('"midi"');
      expect(raw.toLowerCase(), fileName).not.toContain("nanokontrol");
    }

    expect(appSource).toContain('viewLocked: false');
  });
});
