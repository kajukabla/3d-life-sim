import { defaultCuratedPresetId, legacyCuratedPresetIds } from "./curatedPresets";

export function shouldStartPlaying(search: string): boolean {
  return new URLSearchParams(search).get("playing") !== "0";
}

// Init-time switch: ?parallelPipelines=1 makes the renderer compile its ~45 GPU pipelines
// concurrently instead of serially (faster cold start). Default off = current serial path.
export function parallelPipelinesFromLaunch(search: string): boolean {
  return new URLSearchParams(search).get("parallelPipelines") === "1";
}

// The website embed's iframes always carry ?embed=1 (appended by the bay scheduler).
export function embedFromLaunch(search: string): boolean {
  return new URLSearchParams(search).get("embed") === "1";
}

// Browser audio is normally started by the cockpit button. Embedded viewers and
// explicit legacy audio launch values can request an immediate permission prompt.
export function audioMicFromLaunch(search: string): boolean {
  const audio = new URLSearchParams(search).get("audio");
  if (audio === "0") return false;
  if (audio === "mic" || audio === "1") return true;
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
// curated Default preset. Legacy file names remain valid for old links.
export function chooseBootSettingsId(search: string, candidates: BootSettingsCandidate[]): string | null {
  const params = new URLSearchParams(search);
  if (params.has("settings")) {
    const wanted = (params.get("settings") ?? "").trim().toLowerCase();
    if (!wanted) return null;
    const legacyId = legacyCuratedPresetIds[wanted];
    const match = candidates.find((c) =>
      c.id.toLowerCase() === wanted ||
      c.name.trim().toLowerCase() === wanted ||
      c.id === legacyId
    );
    return match?.id ?? null;
  }
  if (params.has("profileGpu") || params.has("skipAppCompute")) return null;
  const def = candidates.find((c) => c.id === defaultCuratedPresetId);
  return def?.id ?? null;
}
