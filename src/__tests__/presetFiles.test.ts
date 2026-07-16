import { describe, expect, it, vi } from "vitest";
import {
  maxPresetFileBytes,
  maxPresetNameLength,
  readPresetJsonFile,
  sanitizePresetName
} from "../presetFiles";

function presetFile(overrides: Partial<{ name: string; size: number; text: () => Promise<string> }> = {}) {
  return {
    name: "look.fluoddity3d.json",
    size: 128,
    text: vi.fn(async () => '{"name":"Look"}'),
    ...overrides
  };
}

describe("preset JSON files", () => {
  it("reads a bounded JSON preset", async () => {
    const file = presetFile();

    await expect(readPresetJsonFile(file)).resolves.toBe('{"name":"Look"}');
    expect(file.text).toHaveBeenCalledOnce();
  });

  it("rejects empty, oversized, and non-JSON files before reading them", async () => {
    for (const [file, message] of [
      [presetFile({ size: 0 }), "empty"],
      [presetFile({ size: maxPresetFileBytes + 1 }), "too large"],
      [presetFile({ name: "look.txt" }), "JSON file"]
    ] as const) {
      await expect(readPresetJsonFile(file)).rejects.toThrow(message);
      expect(file.text).not.toHaveBeenCalled();
    }
  });

  it("verifies the encoded payload length instead of trusting file metadata", async () => {
    const file = presetFile({
      size: 128,
      text: vi.fn(async () => "é".repeat(maxPresetFileBytes))
    });

    await expect(readPresetJsonFile(file)).rejects.toThrow("too large");
    expect(file.text).toHaveBeenCalledOnce();
  });

  it("trims and bounds user-controlled preset names", () => {
    expect(sanitizePresetName("  Aurora  ", "Imported")).toBe("Aurora");
    expect(sanitizePresetName("   ", "Imported")).toBe("Imported");
    expect(sanitizePresetName("Safe\u202e.json\u0000Name", "Imported")).toBe("Safe .json Name");
    expect(sanitizePresetName("x".repeat(maxPresetNameLength + 20), "Imported")).toHaveLength(maxPresetNameLength);
  });
});
