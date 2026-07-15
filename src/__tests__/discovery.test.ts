import { describe, expect, it } from "vitest";
import {
  createDiscoveryBatch,
  defaultDiscoveryBatchSize,
  scoreDiscoveryCandidate,
  summarizeDiscoveryRun
} from "../discovery";

describe("discovery search model", () => {
  it("generates deterministic candidate batches from a seed", () => {
    const a = createDiscoveryBatch({ seed: 1234, count: defaultDiscoveryBatchSize, basePresetId: "angles" });
    const b = createDiscoveryBatch({ seed: 1234, count: defaultDiscoveryBatchSize, basePresetId: "angles" });

    expect(a).toEqual(b);
    expect(a).toHaveLength(defaultDiscoveryBatchSize);
    expect(new Set(a.map((candidate) => candidate.id)).size).toBe(defaultDiscoveryBatchSize);
    expect(a.every((candidate) => candidate.score.interestingness >= 0 && candidate.score.interestingness <= 1)).toBe(true);
  });

  it("mutates around a parent while preserving lineage", () => {
    const [parent] = createDiscoveryBatch({ seed: 222, count: 1, basePresetId: "web" });
    const children = createDiscoveryBatch({ seed: 333, count: 4, parent });

    expect(children).toHaveLength(4);
    expect(children.every((candidate) => candidate.parentId === parent.id)).toBe(true);
    expect(children.every((candidate) => candidate.generation === parent.generation + 1)).toBe(true);
    expect(children.some((candidate) => candidate.liveConfig.ruleSeed !== parent.liveConfig.ruleSeed)).toBe(true);
  });

  it("uses visual evidence to cool down blank candidates", () => {
    const [candidate] = createDiscoveryBatch({ seed: 444, count: 1 });
    const score = scoreDiscoveryCandidate(candidate, {
      image: {
        width: 128,
        height: 128,
        pixel_mean: 0.0004,
        pixel_variance: 0.00001,
        non_black_ratio: 0.001,
        edge_sharpness: 0.0002,
        bounding_box: null
      },
      temporal: {
        mean_absolute_error: 0.0001,
        ssim: 0.99,
        alignment_error: 0.2
      }
    });

    expect(score.temperature).toBe("cold");
    expect(score.reason).toContain("low motion");
    expect(score.interestingness).toBeLessThan(candidate.score.interestingness);
  });

  it("summarizes top candidates and temperature buckets", () => {
    const candidates = createDiscoveryBatch({ seed: 555, count: 6, basePresetId: "circuits" });
    const summary = summarizeDiscoveryRun(candidates);

    expect(summary.count).toBe(6);
    expect(summary.best).toBeTruthy();
    expect(summary.workable + summary.cold + summary.hot).toBe(6);
    expect(summary.best?.score.interestingness).toBe(
      Math.max(...candidates.map((candidate) => candidate.score.interestingness))
    );
  });
});
