import { useReducer } from "react";
import {
  TIMELINE_TOTAL_FRAMES,
  type TimelineState,
  advanceFrame,
  clampFrame,
  normalizeLoop
} from "./timeline";

export type TimelineAction =
  | { type: "toggle-play" }
  | { type: "play"; playing: boolean }
  | { type: "scrub"; frame: number }
  | { type: "set-loop"; loopIn: number; loopOut: number }
  | { type: "set-speed"; speed: number }
  | { type: "tick" };

export const initialTimelineState: TimelineState = {
  currentFrame: 0,
  playing: false,
  loopIn: 0,
  loopOut: TIMELINE_TOTAL_FRAMES - 1,
  playbackSpeed: 1
};

export function timelineReducer(state: TimelineState, action: TimelineAction): TimelineState {
  switch (action.type) {
    case "toggle-play":
      return { ...state, playing: !state.playing };
    case "play":
      return { ...state, playing: action.playing };
    case "scrub":
      return { ...state, currentFrame: clampFrame(action.frame) };
    case "set-loop": {
      const { loopIn, loopOut } = normalizeLoop(action.loopIn, action.loopOut);
      return { ...state, loopIn, loopOut, currentFrame: clampFrame(state.currentFrame) };
    }
    case "set-speed":
      return { ...state, playbackSpeed: Math.max(0.05, action.speed) };
    case "tick":
      return { ...state, currentFrame: advanceFrame(state) };
    default:
      return state;
  }
}

export function useTimelineTransport() {
  const [state, dispatch] = useReducer(timelineReducer, initialTimelineState);
  return { state, dispatch };
}
