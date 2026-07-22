import { devices, expect, test, type Page } from "@playwright/test";

type Profile = {
  liveConfig: { particleCount: number };
  controls: { cameraYaw: number; cameraPitch: number; cameraDistance: number; cameraPanX: number; cameraPanY: number };
};

async function readProfile(page: Page): Promise<Profile> {
  return JSON.parse(await page.getByTestId("profile-json").innerText());
}

// defaultBrowserType cannot be overridden inside a describe group (it would force a
// new worker); the suite already runs on chromium, so drop it from the descriptor.
const { defaultBrowserType: _pixelBrowserType, ...pixel7 } = devices["Pixel 7"];

test.describe("mobile", () => {
  test.use(pixel7);

  test("shows the desktop-only screen instead of booting the sim", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByTestId("desktop-only")).toBeVisible();
    await expect(page.getByTestId("desktop-only")).toContainText("Desktop only");
    await expect(page.locator("main.app-shell")).toHaveCount(0);
    await expect(page.getByTestId("simulation-canvas")).toHaveCount(0);
  });
});

test.describe("desktop", () => {
  test("boots the full sim with the default particle budget", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByTestId("desktop-only")).toHaveCount(0);
    await expect(page.locator("main.app-shell")).toBeVisible();
    await expect.poll(async () => (await readProfile(page)).liveConfig.particleCount).toBe(3399680);
  });

  test("mouse drag still orbits and the wheel still zooms", async ({ page }) => {
    await page.goto("/?particles=16384");
    await expect(page.getByTestId("viewport-band")).toBeVisible();
    const before = (await readProfile(page)).controls;

    await page.mouse.move(300, 400);
    await page.mouse.down();
    await page.mouse.move(450, 350, { steps: 8 });
    await page.mouse.up();
    await expect.poll(async () => Math.abs((await readProfile(page)).controls.cameraYaw - before.cameraYaw)).toBeGreaterThan(0.2);

    await page.mouse.move(300, 400);
    await page.mouse.wheel(0, -400);
    await expect.poll(async () => (await readProfile(page)).controls.cameraDistance).toBeLessThan(before.cameraDistance);
  });
});
