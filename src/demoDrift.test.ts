import { describe, expect, it } from "vitest";
import { selectDemoDriftCandidates, DEMO_DRIFT_DENYLIST, type DriftRegistryEntry } from "./demoDrift";
import { defaultSliderModulation, type SliderModulationConfig, type SliderModulationSettings } from "./audioModulation";

// A MIDI-mapped config: hasEnabledMidiMapping requires enabled === true and a non-null control.
function mapped(min = 0, max = 1): SliderModulationConfig {
  const cfg = defaultSliderModulation(min, max);
  cfg.midi = {
    enabled: true,
    min,
    max,
    control: { inputId: "in", inputName: "pad", messageType: "cc", channel: 1, controller: 7 }
  };
  return cfg;
}

function entry(value = 0.5): DriftRegistryEntry {
  return { value, min: 0, max: 1, step: 0.01, onChange: () => {} };
}

function registryFor(keys: string[]): Record<string, DriftRegistryEntry> {
  return Object.fromEntries(keys.map((k) => [k, entry()]));
}

describe("selectDemoDriftCandidates", () => {
  it("returns every MIDI-mapped, non-denylisted slider when nothing is touched", () => {
    const modulations: SliderModulationSettings = {
      "trail-amount-slider": mapped(),
      "sensor-gain-slider": mapped(),
      "drag-slider": mapped()
    };
    const registry = registryFor(Object.keys(modulations));
    const result = selectDemoDriftCandidates(modulations, registry, new Set());
    expect(result.sort()).toEqual(["drag-slider", "sensor-gain-slider", "trail-amount-slider"]);
  });

  // Semantic delta: the ONLY change is touching one slider. That exact key must disappear and the
  // rest must remain. If this passed when `touched` were ignored, the feature would be a no-op.
  it("excludes exactly the touched slider, leaving the others to keep drifting", () => {
    const modulations: SliderModulationSettings = {
      "trail-amount-slider": mapped(),
      "sensor-gain-slider": mapped(),
      "drag-slider": mapped()
    };
    const registry = registryFor(Object.keys(modulations));

    const untouched = selectDemoDriftCandidates(modulations, registry, new Set());
    const touched = selectDemoDriftCandidates(modulations, registry, new Set(["sensor-gain-slider"]));

    expect(untouched).toContain("sensor-gain-slider");
    expect(touched).not.toContain("sensor-gain-slider");
    expect(touched.sort()).toEqual(["drag-slider", "trail-amount-slider"]);
    // The touch removed one and only one candidate.
    expect(touched.length).toBe(untouched.length - 1);
  });

  it("never drifts structural denylisted controls even when MIDI-mapped", () => {
    const modulations: SliderModulationSettings = {
      "seed-slider": mapped(),
      "particle-count-slider": mapped(),
      "orbit-speed-slider": mapped(),
      "volume-x-slider": mapped(),
      "trail-amount-slider": mapped()
    };
    const registry = registryFor(Object.keys(modulations));
    const result = selectDemoDriftCandidates(modulations, registry, new Set());
    expect(result).toEqual(["trail-amount-slider"]);
    for (const token of DEMO_DRIFT_DENYLIST) {
      expect(result.some((k) => k.includes(token))).toBe(false);
    }
  });

  // The instant demo kept flipping the deliberately-chosen palette over to green/solid render
  // states. Color-mode sliders are denylisted so the preset's color choice stays put; structural
  // and motion sliders still drift. Semantic delta: removing "color" from the denylist would let
  // the color-mode slider back into the candidate set.
  it("never drifts color-mode sliders, so the chosen palette sticks", () => {
    const modulations: SliderModulationSettings = {
      "particle-color-mode-slider": mapped(),
      "trail-color-mode-slider": mapped(),
      "force-slider": mapped(),
      "trail-amount-slider": mapped()
    };
    const registry = registryFor(Object.keys(modulations));
    const result = selectDemoDriftCandidates(modulations, registry, new Set());
    expect(result.sort()).toEqual(["force-slider", "trail-amount-slider"]);
    expect(result).not.toContain("particle-color-mode-slider");
    expect(result).not.toContain("trail-color-mode-slider");
    expect(DEMO_DRIFT_DENYLIST).toContain("color");
  });

  it("skips MIDI-mapped sliders that have no live registry entry", () => {
    const modulations: SliderModulationSettings = {
      "trail-amount-slider": mapped(),
      "unmounted-slider": mapped()
    };
    // Only the first is mounted in the registry.
    const registry = registryFor(["trail-amount-slider"]);
    const result = selectDemoDriftCandidates(modulations, registry, new Set());
    expect(result).toEqual(["trail-amount-slider"]);
  });

  it("skips sliders whose MIDI mapping is disabled", () => {
    const off = defaultSliderModulation(0, 1); // default midi.enabled === false
    const modulations: SliderModulationSettings = {
      "trail-amount-slider": mapped(),
      "muted-slider": off
    };
    const registry = registryFor(Object.keys(modulations));
    const result = selectDemoDriftCandidates(modulations, registry, new Set());
    expect(result).toEqual(["trail-amount-slider"]);
  });
});
