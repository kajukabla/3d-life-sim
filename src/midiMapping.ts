export const midiAllInputsId = "all";

export type MidiMessageType = "cc" | "pitch-bend" | "channel-pressure" | "poly-pressure";

export type MidiControlIdentity = {
  inputId: string | null;
  inputName: string;
  messageType: MidiMessageType;
  channel: number;
  controller: number | null;
};

export type MidiControlMessage = MidiControlIdentity & {
  inputId: string;
  value: number;
  rawValue: number;
  rawMax: number;
};

export type SliderMidiMapping = {
  enabled: boolean;
  min: number;
  max: number;
  control: MidiControlIdentity | null;
};

export function defaultSliderMidiMapping(min = 0, max = 1): SliderMidiMapping {
  return {
    enabled: false,
    min,
    max,
    control: null
  };
}

export function midiMappingFromControl(control: MidiControlMessage, min: number, max: number): SliderMidiMapping {
  return {
    enabled: true,
    min,
    max,
    control: {
      inputId: control.inputId,
      inputName: control.inputName,
      messageType: control.messageType,
      channel: control.channel,
      controller: control.controller
    }
  };
}

export function sanitizeSliderMidiMapping(value: unknown, fallbackMin = 0, fallbackMax = 1): SliderMidiMapping {
  const source = isRecord(value) ? value : {};
  const control = sanitizeMidiControlIdentity(source.control);
  return {
    enabled: typeof source.enabled === "boolean" ? source.enabled : !!control,
    min: finiteNumber(source.min, fallbackMin),
    max: finiteNumber(source.max, fallbackMax),
    control
  };
}

export function hasEnabledMidiMapping(mapping: SliderMidiMapping | undefined): boolean {
  return mapping?.enabled === true && mapping.control !== null;
}

export function parseMidiControlMessage(
  inputId: string,
  inputName: string,
  data: ArrayLike<number> | null | undefined
): MidiControlMessage | null {
  if (!data || data.length < 2) return null;
  const status = data[0] & 0xff;
  if (status < 0x80 || status >= 0xf0) return null;
  const command = status & 0xf0;
  const channel = (status & 0x0f) + 1;
  const safeInputId = inputId || inputName || "midi-input";
  const safeInputName = inputName || safeInputId;

  if (command === 0xb0 && data.length >= 3) {
    const controller = data[1] & 0x7f;
    const rawValue = data[2] & 0x7f;
    return controlMessage(safeInputId, safeInputName, "cc", channel, controller, rawValue, 127);
  }
  if (command === 0xe0 && data.length >= 3) {
    const rawValue = (data[1] & 0x7f) | ((data[2] & 0x7f) << 7);
    return controlMessage(safeInputId, safeInputName, "pitch-bend", channel, null, rawValue, 16383);
  }
  if (command === 0xd0) {
    const rawValue = data[1] & 0x7f;
    return controlMessage(safeInputId, safeInputName, "channel-pressure", channel, null, rawValue, 127);
  }
  if (command === 0xa0 && data.length >= 3) {
    const controller = data[1] & 0x7f;
    const rawValue = data[2] & 0x7f;
    return controlMessage(safeInputId, safeInputName, "poly-pressure", channel, controller, rawValue, 127);
  }
  return null;
}

// Matching deliberately IGNORES the input identity and keys only on
// type/channel/controller. Web MIDI's inputId (and even the input name) differ
// per machine/OS for the same physical controller, so honoring it would break
// every mapping the moment you move a preset from the PC to the Mac. The stored
// inputName is kept for display only. The live "active input" dropdown still lets
// you scope to one input this session (handled in App), but saved mappings match
// any input so they're portable across machines.
export function midiMappingMatches(mapping: SliderMidiMapping | undefined, message: MidiControlMessage): boolean {
  if (!mapping?.enabled || !mapping.control) return false;
  const control = mapping.control;
  return control.messageType === message.messageType &&
    control.channel === message.channel &&
    control.controller === message.controller;
}

export function mapMidiValue(min: number, max: number, value: number, fallback: number | undefined): number {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return fallback ?? 0;
  return min + (max - min) * unit(value);
}

// Index key ignores the input identity (see midiMappingMatches) so mappings
// collapse to one bucket per type/channel/controller regardless of which machine
// or input created them.
export function midiControlIndexKey(control: MidiControlIdentity): string {
  return [
    midiAllInputsId,
    control.messageType,
    String(control.channel),
    control.controller === null ? "-" : String(control.controller)
  ].join(":");
}

export function midiMessageIndexKeys(message: MidiControlMessage): string[] {
  return [midiControlIndexKey(message)];
}

export function midiControlLabel(control: MidiControlIdentity | null | undefined): string {
  if (!control) return "Unmapped";
  const source = control.inputName ? ` ${control.inputName}` : "";
  if (control.messageType === "cc") {
    return `CC ${control.controller ?? 0} ch ${control.channel}${source}`;
  }
  if (control.messageType === "pitch-bend") {
    return `Pitch ch ${control.channel}${source}`;
  }
  if (control.messageType === "channel-pressure") {
    return `Pressure ch ${control.channel}${source}`;
  }
  return `Poly ${control.controller ?? 0} ch ${control.channel}${source}`;
}

function controlMessage(
  inputId: string,
  inputName: string,
  messageType: MidiMessageType,
  channel: number,
  controller: number | null,
  rawValue: number,
  rawMax: number
): MidiControlMessage {
  return {
    inputId,
    inputName,
    messageType,
    channel,
    controller,
    rawValue,
    rawMax,
    value: rawMax > 0 ? unit(rawValue / rawMax) : 0
  };
}

function sanitizeMidiControlIdentity(value: unknown): MidiControlIdentity | null {
  if (!isRecord(value)) return null;
  const messageType = value.messageType;
  if (messageType !== "cc" && messageType !== "pitch-bend" && messageType !== "channel-pressure" && messageType !== "poly-pressure") {
    return null;
  }
  const channel = clamp(Math.round(finiteNumber(value.channel, 1)), 1, 16);
  const needsController = messageType === "cc" || messageType === "poly-pressure";
  const controller = needsController
    ? clamp(Math.round(finiteNumber(value.controller, 0)), 0, 127)
    : null;
  return {
    inputId: typeof value.inputId === "string" && value.inputId ? value.inputId : null,
    inputName: typeof value.inputName === "string" ? value.inputName : "",
    messageType,
    channel,
    controller
  };
}

function unit(value: number): number {
  return Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function finiteNumber(value: unknown, fallback: number): number {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : fallback;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
