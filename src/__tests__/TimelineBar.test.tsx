import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { TimelineBar } from "../TimelineBar";
import { initialTimelineState } from "../useTimelineTransport";

describe("TimelineBar", () => {
  it("renders play control, scrubber, readout, loop handles, speed", () => {
    const html = renderToStaticMarkup(
      <TimelineBar
        state={{ ...initialTimelineState, currentFrame: 90 }}
        onTogglePlay={() => {}}
        onScrub={() => {}}
        onSetLoop={() => {}}
        onSetSpeed={() => {}}
      />
    );
    expect(html).toContain('data-testid="timeline-play"');
    expect(html).toContain('data-testid="timeline-scrub"');
    expect(html).toContain('data-testid="timeline-loop-in"');
    expect(html).toContain('data-testid="timeline-loop-out"');
    expect(html).toContain('data-testid="timeline-speed"');
    expect(html).toContain("0:01 / 90");
  });
});
