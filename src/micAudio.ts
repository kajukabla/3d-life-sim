import audioWorkletUrl from "./micAudioWorklet.ts?worker&url";
import { parseAudioAnalysisFrame, type AudioAnalysisFrame } from "./audioReactive";

export {
  MicFilterbank,
  dbToUnit,
  defaultMicAnalysisOptions,
  linearToDb,
  type MicAnalysisOptions,
  type MicBandConfig
} from "./micAudioDsp";

export type BrowserAudioStatus = "idle" | "starting" | "running" | "denied" | "error" | "unsupported";

export type BrowserAudioInput = {
  id: string;
  label: string;
  isDefault: boolean;
};

export type BrowserAudioSnapshot = {
  status: BrowserAudioStatus;
  devices: BrowserAudioInput[];
  activeDeviceId: string | null;
  error: string | null;
  frameCount: number;
  lastSequence: number | null;
};

export type BrowserAudioController = {
  start: (deviceId?: string) => Promise<void>;
  stop: () => void;
  destroy: () => void;
  refreshDevices: () => Promise<void>;
  getStream: () => MediaStream | null;
  snapshot: () => BrowserAudioSnapshot;
};

export type BrowserAudioDependencies = {
  mediaDevices: MediaDevices;
  createContext: () => AudioContext;
  createWorkletNode: (context: AudioContext, processorName: string) => AudioWorkletNode;
  workletUrl: string;
};

type BrowserAudioHandlers = {
  onFrame: (frame: AudioAnalysisFrame, dtSec: number) => void;
  onState?: (snapshot: BrowserAudioSnapshot) => void;
};

const maxBrowserAudioInputs = 64;
const maxDeviceLabelChars = 160;

export function createBrowserAudio(
  handlers: BrowserAudioHandlers,
  injectedDependencies?: BrowserAudioDependencies
): BrowserAudioController {
  const dependencies = injectedDependencies ?? defaultBrowserAudioDependencies();
  let state: BrowserAudioSnapshot = {
    status: dependencies ? "idle" : "unsupported",
    devices: [],
    activeDeviceId: null,
    error: dependencies ? null : "This browser does not support AudioWorklet microphone capture.",
    frameCount: 0,
    lastSequence: null
  };
  let generation = 0;
  let destroyed = false;
  let stream: MediaStream | null = null;
  let context: AudioContext | null = null;
  let source: MediaStreamAudioSourceNode | null = null;
  let worklet: AudioWorkletNode | null = null;
  let silentGain: GainNode | null = null;
  let previousTimestamp: number | null = null;

  const publish = (patch: Partial<BrowserAudioSnapshot>) => {
    state = { ...state, ...patch };
    handlers.onState?.(snapshot());
  };

  const refreshDevices = async () => {
    if (!dependencies || destroyed) return;
    try {
      const available = await dependencies.mediaDevices.enumerateDevices();
      const devices = available
        .filter((device) => device.kind === "audioinput" && device.deviceId)
        .slice(0, maxBrowserAudioInputs)
        .map((device, index) => ({
          id: device.deviceId,
          label: boundedLabel(device.label) || `Microphone ${index + 1}`,
          isDefault: device.deviceId === "default"
        }));
      publish({ devices });
    } catch {
      // Device enumeration is supplemental; an active stream can continue without it.
    }
  };

  const handleDeviceChange = () => { void refreshDevices(); };
  dependencies?.mediaDevices.addEventListener?.("devicechange", handleDeviceChange);

  const teardownGraph = () => {
    worklet?.port.close();
    worklet?.disconnect();
    source?.disconnect();
    silentGain?.disconnect();
    stream?.getTracks().forEach((track) => track.stop());
    void context?.close();
    stream = null;
    context = null;
    source = null;
    worklet = null;
    silentGain = null;
    previousTimestamp = null;
  };

  const start = async (deviceId?: string) => {
    if (!dependencies || destroyed) {
      publish({
        status: "unsupported",
        error: "This browser does not support AudioWorklet microphone capture."
      });
      return;
    }
    const currentGeneration = ++generation;
    teardownGraph();
    publish({ status: "starting", error: null, frameCount: 0, lastSequence: null });

    let nextStream: MediaStream;
    try {
      nextStream = await dependencies.mediaDevices.getUserMedia({
        audio: {
          ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false
        }
      });
    } catch (error) {
      if (currentGeneration !== generation || destroyed) return;
      const denied = error instanceof DOMException
        ? error.name === "NotAllowedError" || error.name === "SecurityError"
        : error instanceof Error && (error.name === "NotAllowedError" || error.name === "SecurityError");
      publish({
        status: denied ? "denied" : "error",
        error: denied ? "Microphone permission was denied." : browserAudioError(error)
      });
      return;
    }
    if (currentGeneration !== generation || destroyed) {
      nextStream.getTracks().forEach((track) => track.stop());
      return;
    }

    let nextContext: AudioContext | null = null;
    try {
      nextContext = dependencies.createContext();
      await nextContext.audioWorklet.addModule(dependencies.workletUrl);
      if (currentGeneration !== generation || destroyed) {
        nextStream.getTracks().forEach((track) => track.stop());
        void nextContext.close();
        return;
      }
      const nextSource = nextContext.createMediaStreamSource(nextStream);
      const nextWorklet = dependencies.createWorkletNode(nextContext, "life-audio-analyzer");
      const nextSilentGain = nextContext.createGain();
      nextSilentGain.gain.value = 0;
      nextSource.connect(nextWorklet);
      nextWorklet.connect(nextSilentGain);
      nextSilentGain.connect(nextContext.destination);
      nextWorklet.port.onmessage = (event: MessageEvent<unknown>) => {
        const frame = parseAudioAnalysisFrame(event.data);
        if (!frame || currentGeneration !== generation || destroyed) return;
        const dtSec = previousTimestamp === null
          ? 1 / 60
          : Math.min(1, Math.max(0.001, frame.timestampSec - previousTimestamp));
        previousTimestamp = frame.timestampSec;
        state = {
          ...state,
          frameCount: state.frameCount + 1,
          lastSequence: frame.sequence
        };
        handlers.onFrame(frame, dtSec);
      };
      nextWorklet.port.start();
      if (nextContext.state === "suspended") await nextContext.resume();

      stream = nextStream;
      context = nextContext;
      source = nextSource;
      worklet = nextWorklet;
      silentGain = nextSilentGain;
      const trackDeviceId = nextStream.getAudioTracks()[0]?.getSettings().deviceId;
      publish({
        status: "running",
        activeDeviceId: trackDeviceId || deviceId || null,
        error: null
      });
      await refreshDevices();
    } catch (error) {
      nextStream.getTracks().forEach((track) => track.stop());
      void nextContext?.close();
      if (currentGeneration !== generation || destroyed) return;
      teardownGraph();
      publish({ status: "error", error: browserAudioError(error) });
    }
  };

  const stop = () => {
    generation += 1;
    teardownGraph();
    publish({ status: dependencies ? "idle" : "unsupported", error: null, frameCount: 0, lastSequence: null });
  };

  const destroy = () => {
    if (destroyed) return;
    destroyed = true;
    generation += 1;
    teardownGraph();
    dependencies?.mediaDevices.removeEventListener?.("devicechange", handleDeviceChange);
  };

  function snapshot(): BrowserAudioSnapshot {
    return {
      ...state,
      devices: state.devices.map((device) => ({ ...device }))
    };
  }

  return { start, stop, destroy, refreshDevices, getStream: () => stream, snapshot };
}

function defaultBrowserAudioDependencies(): BrowserAudioDependencies | null {
  if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia || typeof AudioContext === "undefined" || typeof AudioWorkletNode === "undefined") {
    return null;
  }
  return {
    mediaDevices: navigator.mediaDevices,
    createContext: () => new AudioContext({ latencyHint: "interactive" }),
    createWorkletNode: (context, processorName) => new AudioWorkletNode(context, processorName, {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
      channelCount: 1,
      channelCountMode: "explicit"
    }),
    workletUrl: audioWorkletUrl
  };
}

function boundedLabel(value: string): string {
  return Array.from(value)
    .slice(0, maxDeviceLabelChars)
    .join("")
    .replace(/[\u0000-\u001f\u007f-\u009f\u202a-\u202e\u2066-\u2069]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function browserAudioError(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  return boundedLabel(message) || "Microphone capture could not start.";
}
