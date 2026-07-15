import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { computeLightFrame } from "./lightingEngine";
import { browserSocketFactory, HueClient, type HueClientState } from "./hueClient";
import { loadLightingState, saveLightingState } from "./lightingStorage";
import { type LightConfig, type LightingColorSource, type LightingState } from "./types";

const FRAME_INTERVAL_MS = 20; // ~50Hz cap; helper keeps the DTLS stream alive between frames.

type UseHueLightingOptions = {
  enabled: boolean;
  wsUrl: string;
  getColorSource: () => LightingColorSource;
};

type UseHueLightingResult = {
  hueState: HueClientState;
  lightingState: LightingState;
  setLightingState: (updater: (prev: LightingState) => LightingState) => void;
  lightConfigs: LightConfig[];
  setLightEnabled: (channel: number, enabled: boolean) => void;
  actions: {
    discover: () => void;
    pair: (ip: string) => void;
    ensureArea: () => void;
    start: () => void;
    stop: () => void;
  };
};

const idleState: HueClientState = {
  connected: false,
  connection: "idle",
  bridges: [],
  lights: [],
  channels: [],
  error: null
};

// Wires the browser to the Hue helper: owns the WebSocket client, the live
// lighting parameters, and the ~50Hz frame loop that pushes computed colors to
// the helper while streaming. All color math is in the pure lightingEngine.
export function useHueLighting(options: UseHueLightingOptions): UseHueLightingResult {
  const { enabled, wsUrl, getColorSource } = options;
  const [hueState, setHueState] = useState<HueClientState>(idleState);
  const [lightingState, setLightingStateRaw] = useState<LightingState>(loadLightingState);
  const [lightEnabled, setLightEnabledMap] = useState<Record<number, boolean>>({});
  const clientRef = useRef<HueClient | null>(null);

  const lightConfigs = useMemo<LightConfig[]>(
    () =>
      hueState.channels.map((channel) => ({
        channel: channel.index,
        name: channel.names.join(", ") || `Channel ${channel.index}`,
        enabled: lightEnabled[channel.index] ?? true
      })),
    [hueState.channels, lightEnabled]
  );

  // Refs so the rAF loop reads current values without re-subscribing every change.
  const lightingRef = useRef(lightingState);
  const configsRef = useRef(lightConfigs);
  const colorSourceRef = useRef(getColorSource);
  const hueStateRef = useRef(hueState);
  useEffect(() => { lightingRef.current = lightingState; }, [lightingState]);
  useEffect(() => { configsRef.current = lightConfigs; }, [lightConfigs]);
  useEffect(() => { colorSourceRef.current = getColorSource; }, [getColorSource]);
  useEffect(() => { hueStateRef.current = hueState; }, [hueState]);
  useEffect(() => { saveLightingState(lightingState); }, [lightingState]);

  // Client lifecycle: connect while enabled and KEEP it connected. If the socket
  // ever closes (helper restart, dropped connection), retry until it's back — a
  // one-shot connection would otherwise leave the panel stuck in error/idle.
  useEffect(() => {
    if (!enabled) {
      setHueState(idleState);
      return;
    }
    let cancelled = false;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    const connect = () => {
      if (cancelled) return;
      const client = new HueClient({
        url: wsUrl,
        createSocket: browserSocketFactory,
        onClose: () => {
          if (cancelled) return;
          clientRef.current = null;
          retryTimer = setTimeout(connect, 1500);
        }
      });
      clientRef.current = client;
      client.subscribe((state) => { if (!cancelled) setHueState(state); });
      client.connect();
    };
    connect();
    return () => {
      cancelled = true;
      if (retryTimer) clearTimeout(retryTimer);
      clientRef.current?.disconnect();
      clientRef.current = null;
    };
  }, [enabled, wsUrl]);

  // Frame loop: only runs while enabled; sends a frame at most every ~20ms and
  // only while the helper reports it is streaming.
  useEffect(() => {
    if (!enabled) return;
    let raf = 0;
    let lastSent = 0;
    const loop = (nowMs: number) => {
      raf = requestAnimationFrame(loop);
      if (hueStateRef.current.connection !== "streaming") return;
      if (nowMs - lastSent < FRAME_INTERVAL_MS) return;
      lastSent = nowMs;
      const frame = computeLightFrame(
        lightingRef.current,
        colorSourceRef.current(),
        configsRef.current,
        nowMs / 1000
      );
      clientRef.current?.sendFrame(frame);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [enabled]);

  const setLightingState = useCallback((updater: (prev: LightingState) => LightingState) => {
    setLightingStateRaw(updater);
  }, []);

  const setLightEnabled = useCallback((channel: number, value: boolean) => {
    setLightEnabledMap((prev) => ({ ...prev, [channel]: value }));
  }, []);

  const actions = useMemo(
    () => ({
      discover: () => clientRef.current?.discover(),
      pair: (ip: string) => clientRef.current?.pair(ip),
      ensureArea: () => clientRef.current?.ensureArea(),
      start: () => clientRef.current?.start(),
      stop: () => clientRef.current?.stop()
    }),
    []
  );

  return { hueState, lightingState, setLightingState, lightConfigs, setLightEnabled, actions };
}
