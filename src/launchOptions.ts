export const defaultAudioReactiveWsUrl = "ws://127.0.0.1:47831";

type AudioReactiveLaunchEnv = {
  VITE_AUDIO_REACTIVE_DEFAULT?: string;
};

export function shouldStartPlaying(search: string): boolean {
  return new URLSearchParams(search).get("playing") !== "0";
}

// Init-time switch: ?parallelPipelines=1 makes the renderer compile its ~45 GPU pipelines
// concurrently instead of serially (faster cold start). Default off = current serial path.
export function parallelPipelinesFromLaunch(search: string): boolean {
  return new URLSearchParams(search).get("parallelPipelines") === "1";
}

// The website embed's iframes always carry ?embed=1 (appended by the bay scheduler).
// Embedded viewers can never reach the native Rust helper, so embed mode forbids the
// WebSocket path entirely and the helper-download gate can never trigger.
export function embedFromLaunch(search: string): boolean {
  return new URLSearchParams(search).get("embed") === "1";
}

export function audioReactiveUrlFromLaunch(search: string, env: AudioReactiveLaunchEnv): string | null {
  const params = new URLSearchParams(search);
  if (embedFromLaunch(search) || audioMicFromLaunch(search)) return null;
  const defaultEnabled = env.VITE_AUDIO_REACTIVE_DEFAULT === "1";
  const disabled = params.get("audio") === "0" || params.get("audioReactive") === "0";
  const enabled = !disabled && (defaultEnabled || params.get("audio") === "1" || params.get("audioReactive") === "1");
  return enabled ? loopbackAudioWsUrl(params.get("audioWs")) : null;
}

export function loopbackAudioWsUrl(value: string | null): string {
  if (!value) return defaultAudioReactiveWsUrl;
  try {
    const url = new URL(value);
    const loopback = url.hostname === "127.0.0.1" || url.hostname === "localhost" || url.hostname === "::1";
    if (url.protocol !== "ws:" || !loopback) return defaultAudioReactiveWsUrl;
    return `ws://${url.host}`;
  } catch {
    return defaultAudioReactiveWsUrl;
  }
}

// In-browser microphone analyzer (no native helper, no WebSocket). Explicit via
// ?audio=mic; embedded mode defaults to it unless audio is switched off outright.
export function audioMicFromLaunch(search: string): boolean {
  const audio = new URLSearchParams(search).get("audio");
  if (audio === "mic") return true;
  return embedFromLaunch(search) && audio !== "0";
}

export function demoLaunchFromSearch(search: string): { instant: boolean; idle: boolean } {
  const instant = new URLSearchParams(search).get("demo") === "instant";
  // The embed wants the idle demo armed too, so the sim re-enters demo even if a
  // viewer unchecks Instant Demo and walks away.
  return { instant, idle: instant };
}

export type BootSettingsCandidate = { id: string; name: string };

// Which saved-settings preset to apply on boot. `?settings=<name|id>` selects explicitly
// (unknown name → nothing, preserving the old "?settings skips the default" semantics);
// `?profileGpu`/`?skipAppCompute` automation skips any default; otherwise the repo
// NewDefault preset (AR10 fallback).
export function chooseBootSettingsId(search: string, candidates: BootSettingsCandidate[]): string | null {
  const params = new URLSearchParams(search);
  if (params.has("settings")) {
    const wanted = (params.get("settings") ?? "").trim().toLowerCase();
    if (!wanted) return null;
    const match = candidates.find((c) => c.id.toLowerCase() === wanted || c.name.trim().toLowerCase() === wanted);
    return match?.id ?? null;
  }
  if (params.has("profileGpu") || params.has("skipAppCompute")) return null;
  const def = candidates.find((c) => c.id === "file-newdefault-json" || c.name === "NewDefault")
    ?? candidates.find((c) => c.id === "file-ar10-json" || c.name === "AR10");
  return def?.id ?? null;
}
