import { describe, expect, it, vi } from "vitest";
import { HueClient, defaultHueWsUrl, loopbackHueWsUrl, type SocketLike } from "../lighting/hueClient";

// Minimal fake socket we can drive from tests.
class FakeSocket implements SocketLike {
  sent: string[] = [];
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((data: string) => void) | null = null;
  closed = false;
  send(data: string) {
    this.sent.push(data);
  }
  close() {
    this.closed = true;
    this.onclose?.();
  }
  open() {
    this.onopen?.();
  }
  receive(data: string) {
    this.onmessage?.(data);
  }
}

function makeClient() {
  const socket = new FakeSocket();
  const client = new HueClient({ url: defaultHueWsUrl, createSocket: () => socket });
  client.connect();
  return { socket, client };
}

describe("loopbackHueWsUrl", () => {
  it("defaults when empty or non-loopback or non-ws", () => {
    expect(loopbackHueWsUrl(null)).toBe(defaultHueWsUrl);
    expect(loopbackHueWsUrl("ws://evil.example.com:47832")).toBe(defaultHueWsUrl);
    expect(loopbackHueWsUrl("http://127.0.0.1:47832")).toBe(defaultHueWsUrl);
  });
  it("accepts loopback ws urls", () => {
    expect(loopbackHueWsUrl("ws://127.0.0.1:49000")).toBe("ws://127.0.0.1:49000");
    expect(loopbackHueWsUrl("ws://localhost:47832")).toBe("ws://localhost:47832");
  });
});

describe("HueClient", () => {
  it("sends control messages as tagged JSON", () => {
    const { socket, client } = makeClient();
    socket.open();
    client.discover();
    client.pair("192.168.1.50");
    client.ensureArea();
    client.start();
    client.sendFrame([[255, 0, 0], [0, 128, 255]]);
    client.stop();
    expect(socket.sent.map((s) => JSON.parse(s))).toEqual([
      { type: "discover" },
      { type: "pair", ip: "192.168.1.50" },
      { type: "ensureArea" },
      { type: "start" },
      { type: "frame", channels: [[255, 0, 0], [0, 128, 255]] },
      { type: "stop" }
    ]);
  });

  it("buffers nothing before open (frames dropped, no throw)", () => {
    const { socket, client } = makeClient();
    // not opened yet
    expect(() => client.sendFrame([[1, 2, 3]])).not.toThrow();
    expect(socket.sent).toEqual([]);
  });

  it("reduces incoming messages into connection state, bridges, lights, area", () => {
    const { socket, client } = makeClient();
    socket.open();
    const changes = vi.fn();
    client.subscribe(changes);
    socket.receive(JSON.stringify({ type: "status", state: "discovering" }));
    socket.receive(JSON.stringify({ type: "bridges", bridges: [{ ip: "192.168.1.50", id: "abc" }] }));
    socket.receive(JSON.stringify({ type: "paired", lights: [{ id: "l1", name: "Lamp" }] }));
    socket.receive(JSON.stringify({ type: "area", channels: [{ index: 0, names: ["Lamp"] }] }));
    socket.receive(JSON.stringify({ type: "status", state: "streaming" }));
    const s = client.getState();
    expect(s.connection).toBe("streaming");
    expect(s.bridges).toEqual([{ ip: "192.168.1.50", id: "abc" }]);
    expect(s.lights).toEqual([{ id: "l1", name: "Lamp" }]);
    expect(s.channels).toEqual([{ index: 0, names: ["Lamp"] }]);
    expect(changes).toHaveBeenCalled();
  });

  it("captures error status", () => {
    const { socket, client } = makeClient();
    socket.open();
    socket.receive(JSON.stringify({ type: "status", state: "error", error: "link button not pressed" }));
    expect(client.getState().connection).toBe("error");
    expect(client.getState().error).toBe("link button not pressed");
  });

  it("marks disconnected when the socket closes", () => {
    const { socket, client } = makeClient();
    socket.open();
    expect(client.getState().connected).toBe(true);
    socket.close();
    expect(client.getState().connected).toBe(false);
  });

  it("fires onClose when the socket closes (so callers can reconnect)", () => {
    const socket = new FakeSocket();
    const onClose = vi.fn();
    const client = new HueClient({ url: defaultHueWsUrl, createSocket: () => socket, onClose });
    client.connect();
    socket.open();
    expect(onClose).not.toHaveBeenCalled();
    socket.close();
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
