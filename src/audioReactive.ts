export type AudioBandFrame = {
  value: number;
  rawDb?: number;
  rawMagnitude?: number;
};

export type AudioAnalysisFrame = {
  version: 1;
  sequence: number;
  timestampSec: number;
  sampleRate: number;
  rms: number;
  peak: number;
  rmsDb?: number;
  peakDb?: number;
  bands: Record<string, AudioBandFrame>;
};

export type AudioReactiveTarget = `render.${string}` | `live.${string}`;

export type AudioReactiveRule = {
  source: string;
  target: AudioReactiveTarget;
  range: readonly [number, number];
  curve?: number;
  attackMs?: number;
  decayMs?: number;
  smoothingMs?: number;
  invert?: boolean;
};

export type AudioReactiveMapping = {
  id: string;
  rules: AudioReactiveRule[];
};

export type AudioReactiveState = {
  render: Record<string, unknown>;
  live: Record<string, unknown>;
};

export type AudioReactiveApplyOptions = {
  previousValues?: Record<string, number>;
  dtSec?: number;
};

export type AudioReactiveApplyResult = {
  render: Record<string, number>;
  live: Record<string, number>;
  nextValues: Record<string, number>;
};

export type AudioReactiveSocketStatus = {
  connected: boolean;
  frameCount: number;
  lastSequence: number | null;
  url: string;
};

export type AudioInputDeviceInfo = {
  id: string;
  name: string;
  isDefault: boolean;
  channels?: number;
  sampleRate?: number;
  sampleFormat?: string;
  error?: string;
};

export type AudioBackendMessage =
  | { type: "audioDevices"; activeInput: string | null; devices: AudioInputDeviceInfo[] }
  | { type: "audioBackendError"; message: string };

export type AudioWsClientCommand =
  | { type: "listDevices" }
  | { type: "setInput"; input: string };

export type AudioReactiveSocketController = {
  close: () => void;
  requestDevices: () => void;
  setInput: (input: string) => void;
  status: () => AudioReactiveSocketStatus;
};

export type AudioReactiveMappingSource = AudioReactiveMapping | (() => AudioReactiveMapping);

export const defaultAudioReactiveMapping: AudioReactiveMapping = {
  id: "fluoddity-default-low-mid-high",
  rules: [
    { source: "low", target: "render.bloomStrength", range: [0.35, 1.35], curve: 1.4 },
    { source: "low", target: "live.depositMass", range: [0.78, 2.2], curve: 1.25 },
    { source: "mid", target: "live.sensorGain", range: [1.1, 3.8], curve: 1.1 },
    { source: "high", target: "render.particleBrightness", range: [0.85, 2.2], curve: 1.35 },
    { source: "high", target: "render.fogBrightness", range: [0.45, 2.2], curve: 1.2 }
  ]
};

export function parseAudioAnalysisFrame(value: unknown): AudioAnalysisFrame | null {
  if (!isRecord(value)) return null;
  const version = finiteNumber(value.version, NaN);
  const sequence = finiteNumber(value.sequence, NaN);
  const timestampSec = finiteNumber(value.timestampSec, NaN);
  const sampleRate = finiteNumber(value.sampleRate, NaN);
  const rms = unit(finiteNumber(value.rms, 0));
  const peak = unit(finiteNumber(value.peak, 0));
  if (version !== 1 || !Number.isFinite(sequence) || !Number.isFinite(timestampSec) || !Number.isFinite(sampleRate)) {
    return null;
  }
  const bands: Record<string, AudioBandFrame> = {};
  if (isRecord(value.bands)) {
    for (const [name, bandValue] of Object.entries(value.bands)) {
      if (!isRecord(bandValue)) continue;
      bands[name] = {
        value: unit(finiteNumber(bandValue.value, 0)),
        rawDb: optionalFiniteNumber(bandValue.rawDb),
        rawMagnitude: optionalFiniteNumber(bandValue.rawMagnitude)
      };
    }
  }
  return {
    version: 1,
    sequence,
    timestampSec,
    sampleRate,
    rms,
    peak,
    rmsDb: optionalFiniteNumber(value.rmsDb),
    peakDb: optionalFiniteNumber(value.peakDb),
    bands
  };
}

export function parseAudioBackendMessage(value: unknown): AudioBackendMessage | null {
  if (!isRecord(value) || typeof value.type !== "string") return null;
  if (value.type === "audioDevices") {
    return {
      type: "audioDevices",
      activeInput: typeof value.activeInput === "string" ? value.activeInput : null,
      devices: Array.isArray(value.devices) ? value.devices.map(parseAudioInputDevice).filter((device): device is AudioInputDeviceInfo => !!device) : []
    };
  }
  if (value.type === "audioBackendError") {
    return {
      type: "audioBackendError",
      message: typeof value.message === "string" ? value.message : "Audio backend error"
    };
  }
  return null;
}

export function normalizeAudioSignal(frame: AudioAnalysisFrame, source: string): number {
  if (source === "rms") return frame.rms;
  if (source === "peak") return frame.peak;
  const direct = frame.bands[source]?.value;
  if (direct !== undefined) return unit(direct);
  return unit(frame.bands[legacyBandFallback(source)]?.value ?? 0);
}

export function applyAudioReactiveFrame(
  frame: AudioAnalysisFrame,
  mapping: AudioReactiveMapping,
  state: AudioReactiveState,
  options: AudioReactiveApplyOptions = {}
): AudioReactiveApplyResult {
  const render: Record<string, number> = {};
  const live: Record<string, number> = {};
  const nextValues: Record<string, number> = { ...(options.previousValues ?? {}) };
  const dtSec = Math.max(0, finiteNumber(options.dtSec, 0));

  for (const rule of mapping.rules) {
    const target = splitTarget(rule.target);
    if (!target) continue;
    const current = finiteNumber(state[target.scope][target.key], rule.range[0]);
    const normalized = shapeSignal(normalizeAudioSignal(frame, rule.source), rule);
    const rawMapped = lerp(rule.range[0], rule.range[1], normalized);
    const previous = options.previousValues?.[rule.target] ?? current;
    const mapped = smoothValue(previous, rawMapped, rule, dtSec);
    nextValues[rule.target] = mapped;
    if (target.scope === "render") {
      render[target.key] = mapped;
    } else {
      live[target.key] = mapped;
    }
  }

  return { render, live, nextValues };
}

export function connectAudioReactiveSocket(options: {
  url: string;
  mapping: AudioReactiveMappingSource;
  getState: () => AudioReactiveState;
  apply: (patches: { render: Record<string, number>; live: Record<string, number> }) => void;
  onStatus?: (status: AudioReactiveSocketStatus) => void;
  onFrame?: (frame: AudioAnalysisFrame, dtSec: number) => void;
  onDevices?: (activeInput: string | null, devices: AudioInputDeviceInfo[]) => void;
  onBackendError?: (message: string) => void;
}): AudioReactiveSocketController {
  let closed = false;
  let frameCount = 0;
  let lastSequence: number | null = null;
  let previousTimestamp: number | null = null;
  let previousValues: Record<string, number> = {};
  const socket = new WebSocket(options.url);

  const emitStatus = (connected: boolean) => {
    options.onStatus?.({ connected, frameCount, lastSequence, url: options.url });
  };
  const sendCommand = (command: AudioWsClientCommand) => {
    if (!closed && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(command));
    }
  };

  socket.addEventListener("open", () => {
    emitStatus(true);
    sendCommand({ type: "listDevices" });
  });
  socket.addEventListener("close", () => emitStatus(false));
  socket.addEventListener("error", () => emitStatus(false));
  socket.addEventListener("message", (event) => {
    const payload = typeof event.data === "string" ? event.data : "";
    let json: unknown;
    try {
      json = JSON.parse(payload);
    } catch {
      return;
    }
    const backendMessage = parseAudioBackendMessage(json);
    if (backendMessage) {
      if (backendMessage.type === "audioDevices") {
        options.onDevices?.(backendMessage.activeInput, backendMessage.devices);
      } else {
        options.onBackendError?.(backendMessage.message);
      }
      return;
    }
    const frame = parseAudioAnalysisFrame(json);
    if (!frame) return;
    const dtSec = previousTimestamp === null ? 1 / 60 : Math.max(0, frame.timestampSec - previousTimestamp);
    previousTimestamp = frame.timestampSec;
    frameCount += 1;
    lastSequence = frame.sequence;
    options.onFrame?.(frame, dtSec);
    const result = applyAudioReactiveFrame(frame, resolveMapping(options.mapping), options.getState(), {
      previousValues,
      dtSec
    });
    previousValues = result.nextValues;
    options.apply({ render: result.render, live: result.live });
    emitStatus(socket.readyState === WebSocket.OPEN);
  });

  return {
    close: () => {
      closed = true;
      socket.close();
    },
    requestDevices: () => sendCommand({ type: "listDevices" }),
    setInput: (input: string) => sendCommand({ type: "setInput", input }),
    status: () => ({
      connected: !closed && socket.readyState === WebSocket.OPEN,
      frameCount,
      lastSequence,
      url: options.url
    })
  };
}

function parseAudioInputDevice(value: unknown): AudioInputDeviceInfo | null {
  if (!isRecord(value)) return null;
  const name = typeof value.name === "string" ? value.name : "";
  if (!name) return null;
  const id = typeof value.id === "string" && value.id ? value.id : name;
  const channels = optionalFiniteNumber(value.channels);
  const sampleRate = optionalFiniteNumber(value.sampleRate);
  return {
    id,
    name,
    isDefault: value.isDefault === true,
    channels: channels === undefined ? undefined : Math.max(0, Math.round(channels)),
    sampleRate: sampleRate === undefined ? undefined : Math.max(0, Math.round(sampleRate)),
    sampleFormat: typeof value.sampleFormat === "string" ? value.sampleFormat : undefined,
    error: typeof value.error === "string" ? value.error : undefined
  };
}

function resolveMapping(source: AudioReactiveMappingSource): AudioReactiveMapping {
  return typeof source === "function" ? source() : source;
}

function shapeSignal(value: number, rule: AudioReactiveRule): number {
  const base = rule.invert ? 1 - unit(value) : unit(value);
  const curve = finiteNumber(rule.curve, 1);
  return curve > 0 ? Math.pow(base, curve) : base;
}

function smoothValue(previous: number, target: number, rule: AudioReactiveRule, dtSec: number): number {
  const fallback = finiteNumber(rule.smoothingMs, 0);
  const attack = finiteNumber(rule.attackMs, fallback);
  const decay = finiteNumber(rule.decayMs, fallback);
  const smoothing = target >= previous ? attack : decay;
  if (smoothing <= 1 || dtSec <= 0) return target;
  const alpha = 1 - Math.exp(-dtSec / (smoothing / 1000));
  return previous + (target - previous) * unit(alpha);
}

function splitTarget(target: AudioReactiveTarget): { scope: "render" | "live"; key: string } | null {
  const dot = target.indexOf(".");
  const scope = target.slice(0, dot);
  const key = target.slice(dot + 1);
  if ((scope === "render" || scope === "live") && key.length > 0) {
    return { scope, key };
  }
  return null;
}

function lerp(min: number, max: number, value: number): number {
  return min + (max - min) * unit(value);
}

function unit(value: number): number {
  return Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0;
}

function finiteNumber(value: unknown, fallback: number): number {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : fallback;
}

function optionalFiniteNumber(value: unknown): number | undefined {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : undefined;
}

function legacyBandFallback(source: string): string {
  if (source === "sub" || source === "bass") return "low";
  if (source === "lowMid") return "mid";
  if (source === "presence") return "high";
  return source;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
