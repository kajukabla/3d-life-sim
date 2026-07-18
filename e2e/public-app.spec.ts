import { expect, test } from "@playwright/test";

const presetNames = ["Default", "Electric Current", "Ice Flower", "Viridian Aurora"];

test.beforeEach(async ({ page }) => {
  await page.goto("/?particles=16384");
  await expect(page.getByTestId("preset-select")).toBeVisible();
});

test("boots the curated public cockpit with collapsed controls", async ({ page }) => {
  const preset = page.getByTestId("preset-select");
  await expect(preset.locator("option")).toHaveText(presetNames);
  await expect(preset).toHaveValue("file-default-json");
  await expect(page.locator("#control-panel details[open]")).toHaveCount(0);

  await page.locator("#control-panel summary", { hasText: "Settings" }).click();
  await expect(page.getByTestId("play-toggle")).toHaveText("Pause");
  await expect(page.getByTestId("save-settings")).toHaveText("Save JSON");
  await expect(page.getByTestId("import-settings")).toHaveText("Load JSON");

  await page.getByTestId("play-toggle").click();
  await expect(page.getByTestId("play-toggle")).toHaveText("Play");

  await page.keyboard.press("m");
  await expect(page.locator("main.app-shell")).toHaveClass(/panel-collapsed/);
  await page.keyboard.press("m");
  await expect(page.locator("main.app-shell")).not.toHaveClass(/panel-collapsed/);
});

test("keeps camera lock disabled across built-in and imported presets", async ({ page }) => {
  const preset = page.getByTestId("preset-select");
  await page.locator("#control-panel summary", { hasText: "Camera" }).click();
  const viewLock = page.getByTestId("view-lock-checkbox");
  await expect(viewLock).not.toBeChecked();

  for (const name of presetNames) {
    await preset.selectOption({ label: name });
    await expect(viewLock).not.toBeChecked();
  }

  await page.getByTestId("json-preset-input").setInputFiles({
    name: "locked-look.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify({
      name: "Imported Lock Attempt",
      liveConfig: { particleCount: 16384 },
      ui: { viewLocked: true }
    }))
  });

  await expect(viewLock).not.toBeChecked();
  await expect(page.getByTestId("particle-input")).toHaveValue("16384");
});

test("rejects oversized preset files in the browser", async ({ page }) => {
  const dialogPromise = page.waitForEvent("dialog").then(async (dialog) => {
    const message = dialog.message();
    await dialog.accept();
    return message;
  });
  await page.getByTestId("json-preset-input").setInputFiles({
    name: "oversized.json",
    mimeType: "application/json",
    buffer: Buffer.alloc(1024 * 1024 + 1, 32)
  });
  await expect(dialogPromise).resolves.toContain("too large");
});

test("runs microphone analysis entirely in the browser", async ({ page }) => {
  await page.getByTestId("audio-panel").locator("summary").click();
  const capture = page.getByTestId("audio-capture-toggle");

  await expect(capture).toHaveText("Start microphone");
  await capture.click();
  await expect(capture).toHaveText("Stop microphone");
  await expect(page.getByTestId("audio-capture-status")).toHaveText("Live");
  await expect(page.getByTestId("audio-input-select")).toBeEnabled();
  await expect.poll(() => page.evaluate(() => (
    (window as unknown as { __lifesimBrowserAudio?: () => { frameCount: number } })
      .__lifesimBrowserAudio?.().frameCount ?? 0
  ))).toBeGreaterThan(0);
});
