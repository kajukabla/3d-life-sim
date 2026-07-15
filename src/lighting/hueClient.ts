import {
  encodeOutgoing,
  parseIncoming,
  type HueAreaChannel,
  type HueBridge,
  type HueConnectionState,
  type HueLight,
  type HueOutgoing
} from "./protocol";
import type { RGB } from "./types";

export const defaultHueWsUrl = "ws://127.0.0.1:47832";

// Loopback-only guard mirroring launchOptions.loopbackAudioWsUrl: a remote page
// must never be able to point the helper WS at an arbitrary host.
export function loopbackHueWsUrl(value: string | null): string {
  if (!value) return defaultHueWsUrl;
  try {
    const url = new URL(value);
    const loopback = url.hostname === "127.0.0.1" || url.hostname === "localhost" || url.hostname === "::1";
    if (url.protocol !== "ws:" || !loopback) return defaultHueWsUrl;
    return `ws://${url.host}`;
  } catch {
    return defaultHueWsUrl;
  }
}

// The browser talks to the helper through an injectable socket so the client is
// testable without a real WebSocket.
export type SocketLike = {
  send: (data: string) => void;
  close: () => void;
  onopen: (() => void) | null;
  onclose: (() => void) | null;
  onerror: (() => void) | null;
  onmessage: ((data: string) => void) | null;
};

export type HueClientOptions = {
  url: string;
  createSocket: (url: string) => SocketLike;
  onClose?: () => void; // fired when the socket closes, so callers can reconnect
};

export type HueClientState = {
  connected: boolean;
  connection: HueConnectionState;
  bridges: HueBridge[];
  lights: HueLight[];
  channels: HueAreaChannel[];
  error: string | null;
};

function initialState(): HueClientState {
  return { connected: false, connection: "idle", bridges: [], lights: [], channels: [], error: null };
}

export class HueClient {
  private readonly options: HueClientOptions;
  private socket: SocketLike | null = null;
  private open = false;
  private state: HueClientState = initialState();
  private listeners = new Set<(state: HueClientState) => void>();

  constructor(options: HueClientOptions) {
    this.options = options;
  }

  connect(): void {
    if (this.socket) return;
    const socket = this.options.createSocket(this.options.url);
    this.socket = socket;
    socket.onopen = () => {
      this.open = true;
      this.patch({ connected: true, error: null });
    };
    socket.onclose = () => {
      this.open = false;
      this.socket = null;
      this.patch({ connected: false });
      this.options.onClose?.();
    };
    socket.onerror = () => {
      this.patch({ connection: "error", error: this.state.error ?? "connection error" });
    };
    socket.onmessage = (data) => this.handle(data);
  }

  disconnect(): void {
    this.socket?.close();
  }

  subscribe(listener: (state: HueClientState) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  getState(): HueClientState {
    return this.state;
  }

  discover(): void {
    this.send({ type: "discover" });
  }
  pair(ip: string): void {
    this.send({ type: "pair", ip });
  }
  ensureArea(): void {
    this.send({ type: "ensureArea" });
  }
  start(): void {
    this.send({ type: "start" });
  }
  stop(): void {
    this.send({ type: "stop" });
  }
  sendFrame(channels: RGB[]): void {
    this.send({ type: "frame", channels });
  }

  private send(message: HueOutgoing): void {
    if (!this.socket || !this.open) return;
    this.socket.send(encodeOutgoing(message));
  }

  private handle(data: string): void {
    const message = parseIncoming(data);
    if (!message) return;
    switch (message.type) {
      case "bridges":
        this.patch({ bridges: message.bridges });
        break;
      case "paired":
        this.patch({ lights: message.lights });
        break;
      case "area":
        this.patch({ channels: message.channels });
        break;
      case "status":
        this.patch({ connection: message.state, error: message.error ?? (message.state === "error" ? this.state.error : null) });
        break;
    }
  }

  private patch(partial: Partial<HueClientState>): void {
    this.state = { ...this.state, ...partial };
    for (const listener of this.listeners) listener(this.state);
  }
}

// Real-WebSocket adapter for production use.
export function browserSocketFactory(url: string): SocketLike {
  const ws = new WebSocket(url);
  const adapter: SocketLike = {
    send: (data) => ws.send(data),
    close: () => ws.close(),
    onopen: null,
    onclose: null,
    onerror: null,
    onmessage: null
  };
  ws.onopen = () => adapter.onopen?.();
  ws.onclose = () => adapter.onclose?.();
  ws.onerror = () => adapter.onerror?.();
  ws.onmessage = (event) => adapter.onmessage?.(typeof event.data === "string" ? event.data : "");
  return adapter;
}
