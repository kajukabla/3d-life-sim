export const maxPresetFileBytes = 1024 * 1024;
export const maxPresetNameLength = 80;

export type PresetJsonFile = Pick<File, "name" | "size" | "text">;

export async function readPresetJsonFile(file: PresetJsonFile): Promise<string> {
  if (!/\.json$/i.test(file.name.trim())) {
    throw new Error("Choose a JSON file.");
  }
  if (!Number.isSafeInteger(file.size) || file.size <= 0) {
    throw new Error("Preset JSON is empty.");
  }
  if (file.size > maxPresetFileBytes) {
    throw new Error("Preset JSON is too large (1 MB maximum).");
  }

  const json = await file.text();
  if (!json.trim()) {
    throw new Error("Preset JSON is empty.");
  }
  if (new TextEncoder().encode(json).byteLength > maxPresetFileBytes) {
    throw new Error("Preset JSON is too large (1 MB maximum).");
  }
  return json;
}

export function sanitizePresetName(value: unknown, fallback: string): string {
  const clean = (candidate: unknown) => typeof candidate === "string"
    ? candidate
      .replace(/[\u0000-\u001f\u007f-\u009f\u202a-\u202e\u2066-\u2069]/g, " ")
      .replace(/\s+/g, " ")
      .trim()
    : "";
  const normalized = clean(value) || clean(fallback) || "Imported";
  return Array.from(normalized).slice(0, maxPresetNameLength).join("");
}
