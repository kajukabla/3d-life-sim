import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const appSource = readFileSync(new URL("../App.tsx", import.meta.url), "utf8");

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

  it("ends the public cockpit at MIDI", () => {
    expect(appSource).not.toContain('<ControlGroup title="Track">');
    expect(appSource).not.toContain('data-testid="track-release"');
    expect(appSource).not.toContain('className="metrics-grid"');
    expect(appSource).not.toContain('className="diagnostics-panel"');
    expect(appSource).not.toContain('className="preset-json"');
  });

  it("ships the selected particle presets without MIDI mappings", () => {
    for (const fileName of ["vid1.json", "vid2.json", "AR11.json"]) {
      const raw = readFileSync(new URL(`../../Presets/${fileName}`, import.meta.url), "utf8");
      const preset = JSON.parse(raw) as { controls?: { renderLayer?: string; ribbonFraction?: number } };

      expect(preset.controls?.renderLayer, fileName).toBe("particles");
      expect(preset.controls?.ribbonFraction, fileName).toBe(0);
      expect(raw.toLowerCase(), fileName).not.toContain('"midi"');
      expect(raw.toLowerCase(), fileName).not.toContain("nanokontrol");
    }
  });
});
