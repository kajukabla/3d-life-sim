export const curatedPresetIds = [
  "file-default-json",
  "file-electric-current-json",
  "file-ice-flower-json",
  "file-viridian-aurora-json"
] as const;

export const defaultCuratedPresetId = curatedPresetIds[0];

export const legacyCuratedPresetIds: Readonly<Record<string, string>> = {
  vid1: "file-default-json",
  vid2: "file-electric-current-json",
  vid3: "file-ice-flower-json",
  ar11: "file-viridian-aurora-json"
};
