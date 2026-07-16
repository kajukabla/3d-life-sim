import { hasEnabledMidiMapping } from "./midiMapping";
import type { SliderModulationSettings } from "./audioModulation";

// Structural view of a live slider registry entry (matches App's SliderRegistryEntry without
// importing it, keeping this module free of the App graph so it stays unit-testable).
export type DriftRegistryEntry = {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
};

// Structural controls that demo never drives: seeds/rules/cohorts reseed or rebuild buffers, the
// volume/size params reallocate, and orbit-speed is driven by its own dedicated block in the loop.
export const DEMO_DRIFT_DENYLIST = [
  "seed",
  "cohort",
  "rule",
  "volume",
  "width",
  "height",
  "depth",
  "particle-input",
  "particle-count",
  "orbit-speed",
  // Color-mode sliders: the demo kept flipping the chosen palette over to green/solid render
  // states. Freeze color so the preset's audio-reactive palette stays put; motion still drifts.
  "color"
];

// Which MIDI-mapped sliders the idle/instant demo is allowed to drift this re-pick. The MIDI mapping
// is the opt-in: if the user mapped it, demo may drive it. Excludes anything without a live registry
// entry, anything matching the structural denylist, and -- the instant-demo addition -- anything the
// viewer has already grabbed (`touched`), so manual control always wins over the auto-drift.
export function selectDemoDriftCandidates(
  modulations: SliderModulationSettings,
  registry: Record<string, DriftRegistryEntry>,
  touched: ReadonlySet<string>,
  denylist: readonly string[] = DEMO_DRIFT_DENYLIST
): string[] {
  return Object.keys(modulations).filter((key) => {
    const cfg = modulations[key];
    const entry = registry[key];
    if (!entry || !hasEnabledMidiMapping(cfg.midi)) return false;
    if (touched.has(key)) return false;
    const lower = key.toLowerCase();
    return !denylist.some((token) => lower.includes(token));
  });
}
