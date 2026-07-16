// Records the audio that drives a live performance so the offline HDR render can have the real
// music stitched onto it (see tools/hdr-export --audio). Audio is essentially free to capture in
// real time — unlike the HDR video frames, which is why the visuals are replayed offline and the
// audio is just recorded live and muxed on afterward. We record the SAME MediaStream that feeds the
// browser audio analyzer, so the recorded track and the modulation it produced are inherently in
// sync. A browser-visible virtual input can be selected when clean system-audio capture is needed.

export type PerformanceAudioRecording = {
  blob: Blob;
  mimeType: string;
  // Wall-clock ms from start() to stop(). Lets the exporter align/trim against the video duration.
  durationMs: number;
};

export type PerformanceAudioRecorder = {
  // Begin recording the given stream. nowMs is a performance.now() stamp taken next to the visual
  // recorder's start so the two share a t0. Returns false if MediaRecorder is unavailable/failed.
  start(stream: MediaStream, nowMs: number): boolean;
  // Finalize and resolve the captured audio (null if nothing was recorded). endMs pairs with the
  // start stamp to report duration without trusting MediaRecorder's own timing.
  stop(nowMs: number): Promise<PerformanceAudioRecording | null>;
  isRecording(): boolean;
};

// Prefer Opus in WebM (broad Chrome support, ffmpeg decodes it for the --audio mux); fall back
// through other containers, then to the UA default ('' lets MediaRecorder choose).
const PREFERRED_MIME_TYPES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/ogg;codecs=opus",
  "audio/mp4"
];

function pickMimeType(): string {
  const supports = typeof MediaRecorder !== "undefined" && typeof MediaRecorder.isTypeSupported === "function";
  if (!supports) return "";
  for (const type of PREFERRED_MIME_TYPES) {
    if (MediaRecorder.isTypeSupported(type)) return type;
  }
  return "";
}

export function createPerformanceAudioRecorder(): PerformanceAudioRecorder {
  let recorder: MediaRecorder | null = null;
  let chunks: BlobPart[] = [];
  let startMs = 0;
  let mimeType = "";

  return {
    start(stream, nowMs) {
      if (typeof MediaRecorder === "undefined") return false;
      const audioTracks = stream.getAudioTracks();
      if (audioTracks.length === 0) return false;
      try {
        mimeType = pickMimeType();
        // Record only the audio so a combined A/V stream (e.g. getDisplayMedia) doesn't drag video in.
        const audioOnly = new MediaStream(audioTracks);
        recorder = mimeType ? new MediaRecorder(audioOnly, { mimeType }) : new MediaRecorder(audioOnly);
        chunks = [];
        recorder.ondataavailable = (event) => {
          if (event.data && event.data.size > 0) chunks.push(event.data);
        };
        // Timeslice so long takes flush incrementally instead of buffering the whole clip in one Blob.
        recorder.start(1000);
        startMs = nowMs;
        mimeType = recorder.mimeType || mimeType;
        return true;
      } catch {
        recorder = null;
        return false;
      }
    },
    stop(nowMs) {
      return new Promise((resolve) => {
        const active = recorder;
        if (!active || active.state === "inactive") {
          recorder = null;
          resolve(null);
          return;
        }
        active.onstop = () => {
          const type = mimeType || active.mimeType || "audio/webm";
          const blob = new Blob(chunks, { type });
          recorder = null;
          chunks = [];
          resolve(blob.size > 0 ? { blob, mimeType: type, durationMs: Math.max(0, nowMs - startMs) } : null);
        };
        try {
          active.stop();
        } catch {
          recorder = null;
          resolve(null);
        }
      });
    },
    isRecording() {
      return recorder !== null && recorder.state === "recording";
    }
  };
}

// File extension for a recorded blob's mime type, for a sensible download name.
export function audioExtensionFor(mimeType: string): string {
  if (mimeType.includes("ogg")) return "ogg";
  if (mimeType.includes("mp4")) return "m4a";
  return "webm";
}
