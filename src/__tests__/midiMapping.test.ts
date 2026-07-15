import { describe, expect, it } from "vitest";
import {
  defaultSliderMidiMapping,
  hasEnabledMidiMapping,
  mapMidiValue,
  midiControlIndexKey,
  midiMappingFromControl,
  midiMappingMatches,
  midiMessageIndexKeys,
  parseMidiControlMessage,
  sanitizeSliderMidiMapping,
  type SliderMidiMapping
} from "../midiMapping";

describe("midi mapping model", () => {
  it("parses control change and maps normalized values across a range", () => {
    const message = parseMidiControlMessage("midi-1", "Launch Control", [0xb2, 74, 64]);

    expect(message).not.toBeNull();
    expect(message?.messageType).toBe("cc");
    expect(message?.channel).toBe(3);
    expect(message?.controller).toBe(74);
    expect(message?.value).toBeCloseTo(64 / 127, 5);
    expect(mapMidiValue(0.2, 0.8, message?.value ?? 0, 0)).toBeCloseTo(0.502, 3);
  });

  it("matches learned controls by channel, type, and controller", () => {
    const learned = parseMidiControlMessage("midi-1", "Launch Control", [0xb0, 21, 127]);
    const same = parseMidiControlMessage("midi-1", "Launch Control", [0xb0, 21, 32]);
    const otherController = parseMidiControlMessage("midi-1", "Launch Control", [0xb0, 22, 32]);
    if (!learned || !same || !otherController) throw new Error("expected MIDI messages");
    const mapping = midiMappingFromControl(learned, 0.25, 0.75);

    expect(hasEnabledMidiMapping(mapping)).toBe(true);
    expect(mapping.min).toBe(0.25);
    expect(mapping.max).toBe(0.75);
    expect(midiMappingMatches(mapping, same)).toBe(true);
    expect(midiMappingMatches(mapping, otherController)).toBe(false);
    // Index key ignores the input identity so it's portable across machines.
    expect(midiControlIndexKey(mapping.control!)).toBe("all:cc:1:21");
  });

  it("matches regardless of which MIDI input sent it (PC↔Mac preset portability)", () => {
    // Same physical fader (CC 16, ch 1), but Web MIDI reports a different inputId
    // and name on each machine. A mapping learned on the PC must still fire on the Mac.
    const learnedOnPc = parseMidiControlMessage("pc-usb-abc123", "nanoKONTROL2", [0xb0, 16, 100]);
    const sameFaderOnMac = parseMidiControlMessage("mac-iac-xyz789", "nanoKONTROL2 SLIDER/KNOB", [0xb0, 16, 64]);
    if (!learnedOnPc || !sameFaderOnMac) throw new Error("expected MIDI messages");
    const mapping = midiMappingFromControl(learnedOnPc, 0, 1);

    expect(midiMappingMatches(mapping, sameFaderOnMac)).toBe(true);
    // and the reverse index finds it (same bucket across inputs)
    expect(midiMessageIndexKeys(sameFaderOnMac)).toContain(midiControlIndexKey(mapping.control!));
    // a different controller number still does NOT match
    const otherCc = parseMidiControlMessage("mac-iac-xyz789", "nanoKONTROL2", [0xb0, 0, 64]);
    expect(midiMappingMatches(mapping, otherCc!)).toBe(false);
  });

  it("routes one CC to every param mapped to it (no exclusivity)", () => {
    const learned = parseMidiControlMessage("midi-1", "Launch Control", [0xb0, 74, 0]);
    if (!learned) throw new Error("expected MIDI message");
    // Two distinct params learn the SAME control, each with its own output range.
    const mappings: Record<string, SliderMidiMapping> = {
      "render.particleBrightness": midiMappingFromControl(learned, 0, 8),
      "render.bloomStrength": midiMappingFromControl(learned, 0, 1)
    };

    // Build the dispatch index exactly as App.tsx does: control key -> array of param keys.
    const index: Record<string, string[]> = {};
    for (const [key, mapping] of Object.entries(mappings)) {
      if (!hasEnabledMidiMapping(mapping) || !mapping.control) continue;
      const indexKey = midiControlIndexKey(mapping.control);
      index[indexKey] = [...(index[indexKey] ?? []), key];
    }
    // Both params share one index bucket — nothing overwrote the first.
    const bucketKey = midiControlIndexKey(learned);
    expect(index[bucketKey]).toEqual(["render.particleBrightness", "render.bloomStrength"]);

    // Dispatch a fresh CC74 value through the same fan-out the app uses.
    const incoming = parseMidiControlMessage("midi-1", "Launch Control", [0xb0, 74, 127]);
    if (!incoming) throw new Error("expected MIDI message");
    const targets = new Set<string>();
    for (const indexKey of midiMessageIndexKeys(incoming)) {
      for (const target of index[indexKey] ?? []) targets.add(target);
    }
    const driven: Record<string, number> = {};
    for (const key of targets) {
      const mapping = mappings[key];
      if (!midiMappingMatches(mapping, incoming)) continue;
      driven[key] = mapMidiValue(mapping.min, mapping.max, incoming.value, 0);
    }

    // BOTH params are driven, each on its own range.
    expect(driven["render.particleBrightness"]).toBeCloseTo(8, 3);
    expect(driven["render.bloomStrength"]).toBeCloseTo(1, 3);
  });

  it("sanitizes missing and legacy mapping values safely", () => {
    expect(defaultSliderMidiMapping()).toEqual({ enabled: false, min: 0, max: 1, control: null });
    expect(defaultSliderMidiMapping(0.2, 0.8)).toEqual({ enabled: false, min: 0.2, max: 0.8, control: null });
    expect(sanitizeSliderMidiMapping({ enabled: true }, 0.3, 0.9)).toEqual({ enabled: true, min: 0.3, max: 0.9, control: null });
    expect(sanitizeSliderMidiMapping({
      enabled: true,
      min: 0.2,
      max: 0.6,
      control: {
        inputId: "",
        inputName: "Any",
        messageType: "pitch-bend",
        channel: 99,
        controller: 12
      }
    })).toEqual({
      enabled: true,
      min: 0.2,
      max: 0.6,
      control: {
        inputId: null,
        inputName: "Any",
        messageType: "pitch-bend",
        channel: 16,
        controller: null
      }
    });
  });
});
