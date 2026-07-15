import type { RGB } from "./types";

// Browser -> helper control messages.
export type HueOutgoing =
  | { type: "discover" }
  | { type: "pair"; ip: string }
  | { type: "ensureArea" }
  | { type: "start" }
  | { type: "stop" }
  | { type: "frame"; channels: RGB[] };

// Helper -> browser messages. `clientkey` is never sent (it stays helper-side).
export type HueBridge = { ip: string; id: string };
export type HueLight = { id: string; name: string };
export type HueAreaChannel = { index: number; names: string[] };
export type HueConnectionState =
  | "idle"
  | "discovering"
  | "pairing"
  | "paired"
  | "streaming"
  | "error";

export type HueIncoming =
  | { type: "bridges"; bridges: HueBridge[] }
  | { type: "paired"; lights: HueLight[] }
  | { type: "area"; channels: HueAreaChannel[] }
  | { type: "status"; state: HueConnectionState; error?: string };

export function encodeOutgoing(message: HueOutgoing): string {
  return JSON.stringify(message);
}

export function parseIncoming(data: string): HueIncoming | null {
  let value: unknown;
  try {
    value = JSON.parse(data);
  } catch {
    return null;
  }
  if (!isRecord(value) || typeof value.type !== "string") return null;
  switch (value.type) {
    case "bridges":
      return { type: "bridges", bridges: asBridges(value.bridges) };
    case "paired":
      return { type: "paired", lights: asLights(value.lights) };
    case "area":
      return { type: "area", channels: asChannels(value.channels) };
    case "status":
      return {
        type: "status",
        state: asConnectionState(value.state),
        error: typeof value.error === "string" ? value.error : undefined
      };
    default:
      return null;
  }
}

function asBridges(value: unknown): HueBridge[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter(isRecord)
    .map((b) => ({ ip: String(b.ip ?? ""), id: String(b.id ?? "") }))
    .filter((b) => b.ip.length > 0);
}

function asLights(value: unknown): HueLight[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter(isRecord)
    .map((l) => ({ id: String(l.id ?? ""), name: String(l.name ?? "") }))
    .filter((l) => l.id.length > 0);
}

function asChannels(value: unknown): HueAreaChannel[] {
  if (!Array.isArray(value)) return [];
  return value.filter(isRecord).map((c) => ({
    index: Number(c.index) | 0,
    names: Array.isArray(c.names) ? c.names.map((n) => String(n)) : []
  }));
}

function asConnectionState(value: unknown): HueConnectionState {
  const states: HueConnectionState[] = ["idle", "discovering", "pairing", "paired", "streaming", "error"];
  return states.includes(value as HueConnectionState) ? (value as HueConnectionState) : "idle";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
