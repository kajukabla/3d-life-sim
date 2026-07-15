import { TIMELINE_TOTAL_FRAMES, frameToTimecode } from "./timeline";
import type { TimelineState } from "./timeline";

type Props = {
  state: TimelineState;
  onTogglePlay: () => void;
  onScrub: (frame: number) => void;
  onSetLoop: (loopIn: number, loopOut: number) => void;
  onSetSpeed: (speed: number) => void;
};

const SPEEDS = [0.25, 0.5, 1, 2];

export function TimelineBar({ state, onTogglePlay, onScrub, onSetLoop, onSetSpeed }: Props) {
  const max = TIMELINE_TOTAL_FRAMES - 1;
  return (
    <div className="timeline-bar" data-testid="timeline-bar">
      <button data-testid="timeline-play" onClick={onTogglePlay}>
        {state.playing ? "Pause" : "Play"}
      </button>
      <input
        type="range"
        data-testid="timeline-scrub"
        min={0}
        max={max}
        value={state.currentFrame}
        onChange={(e) => onScrub(Number(e.target.value))}
        aria-label="Timeline scrubber"
      />
      <span className="timeline-readout" data-testid="timeline-readout">
        {frameToTimecode(state.currentFrame)} / {state.currentFrame}
      </span>
      <label className="timeline-loop">
        in
        <input
          type="range"
          data-testid="timeline-loop-in"
          min={0}
          max={max}
          value={state.loopIn}
          onChange={(e) => onSetLoop(Number(e.target.value), state.loopOut)}
          aria-label="Loop in point"
        />
      </label>
      <label className="timeline-loop">
        out
        <input
          type="range"
          data-testid="timeline-loop-out"
          min={0}
          max={max}
          value={state.loopOut}
          onChange={(e) => onSetLoop(state.loopIn, Number(e.target.value))}
          aria-label="Loop out point"
        />
      </label>
      <select
        data-testid="timeline-speed"
        value={state.playbackSpeed}
        onChange={(e) => onSetSpeed(Number(e.target.value))}
        aria-label="Playback speed"
      >
        {SPEEDS.map((s) => (
          <option key={s} value={s}>{s}x</option>
        ))}
      </select>
    </div>
  );
}
