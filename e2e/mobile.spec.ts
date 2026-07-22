import { devices, expect, test, type CDPSession, type Page } from "@playwright/test";

type Profile = {
  liveConfig: { particleCount: number };
  controls: {
    cameraYaw: number; cameraPitch: number; cameraDistance: number; cameraPanX: number; cameraPanY: number;
    particleGlowCore: number; particleSupportMask: number;
  };
};

async function readProfile(page: Page): Promise<Profile> {
  return JSON.parse(await page.getByTestId("profile-json").innerText());
}

type TouchPoint = { x: number; y: number };

async function dispatchTouch(cdp: CDPSession, type: "touchStart" | "touchMove" | "touchEnd", touchPoints: TouchPoint[]): Promise<void> {
  await cdp.send("Input.dispatchTouchEvent", { type, touchPoints });
}

// defaultBrowserType cannot be overridden inside a describe group (it would force a
// new worker); the suite already runs on chromium, so drop it from the descriptor.
const { defaultBrowserType: _pixelBrowserType, ...pixel7 } = devices["Pixel 7"];

test.describe("mobile cockpit", () => {
  test.use(pixel7);

  test("boots collapsed with a tenth of the desktop particle budget", async ({ page }) => {
    await page.goto("/");
    const shell = page.locator("main.app-shell");
    await expect(shell).toHaveClass(/is-mobile/);
    await expect(shell).toHaveClass(/panel-collapsed/);

    // Default preset holds 3,399,680 particles on desktop; mobile caps at ~200K,
    // with glow off and a lighter support mask.
    await expect.poll(async () => (await readProfile(page)).liveConfig.particleCount).toBe(199680);
    const controls = (await readProfile(page)).controls;
    expect(controls.particleGlowCore).toBe(0);
    expect(controls.particleSupportMask).toBe(0.2);
  });

  test("menu toggle opens a half-screen sheet and closes it again", async ({ page }) => {
    await page.goto("/");
    const shell = page.locator("main.app-shell");
    const toggle = page.getByTestId("panel-toggle");
    await expect(toggle).toBeVisible();

    await toggle.tap();
    await expect(shell).not.toHaveClass(/panel-collapsed/);

    const viewport = page.viewportSize();
    if (!viewport) throw new Error("viewport size unavailable");
    // Right-side drawer: full height, half the screen wide but never narrower
    // than 280px. Poll so the 180ms slide-in transition finishes first.
    const expectedWidth = Math.max(viewport.width / 2, 280);
    await expect(async () => {
      const box = await page.locator("#control-panel").boundingBox();
      if (!box) throw new Error("control panel has no bounding box");
      expect(Math.abs(box.width - expectedWidth)).toBeLessThanOrEqual(2);
      expect(Math.abs(box.height - viewport.height)).toBeLessThanOrEqual(2);
      expect(Math.abs(box.x + box.width - viewport.width)).toBeLessThanOrEqual(2);
    }).toPass();

    await toggle.tap();
    await expect(shell).toHaveClass(/panel-collapsed/);
  });

  test("one-finger drag orbits the camera", async ({ page }) => {
    await page.goto("/?particles=16384");
    await expect(page.getByTestId("viewport-band")).toBeVisible();
    const before = (await readProfile(page)).controls;

    const cdp = await page.context().newCDPSession(page);
    await dispatchTouch(cdp, "touchStart", [{ x: 200, y: 400 }]);
    for (let i = 1; i <= 6; i++) {
      await dispatchTouch(cdp, "touchMove", [{ x: 200 + i * 20, y: 400 - i * 10 }]);
    }
    await dispatchTouch(cdp, "touchEnd", []);

    // 120px horizontal drag at 0.0048 rad/px ≈ 0.58 rad of yaw.
    await expect.poll(async () => Math.abs((await readProfile(page)).controls.cameraYaw - before.cameraYaw)).toBeGreaterThan(0.2);
    await expect.poll(async () => Math.abs((await readProfile(page)).controls.cameraPitch - before.cameraPitch)).toBeGreaterThan(0.1);
  });

  test("pinch zooms the camera and two-finger drag pans", async ({ page }) => {
    await page.goto("/?particles=16384");
    await expect(page.getByTestId("viewport-band")).toBeVisible();
    const before = (await readProfile(page)).controls;

    const cdp = await page.context().newCDPSession(page);
    // Spread two fingers apart (span 100px -> 280px): zoom in.
    await dispatchTouch(cdp, "touchStart", [{ x: 156, y: 400 }]);
    await dispatchTouch(cdp, "touchStart", [{ x: 156, y: 400 }, { x: 256, y: 400 }]);
    for (let i = 1; i <= 6; i++) {
      await dispatchTouch(cdp, "touchMove", [{ x: 156 - i * 15, y: 400 }, { x: 256 + i * 15, y: 400 }]);
    }
    await dispatchTouch(cdp, "touchEnd", []);

    await expect.poll(async () => (await readProfile(page)).controls.cameraDistance).toBeLessThan(before.cameraDistance * 0.6);

    // Two-finger drag with constant span: pan.
    await dispatchTouch(cdp, "touchStart", [{ x: 130, y: 500 }]);
    await dispatchTouch(cdp, "touchStart", [{ x: 130, y: 500 }, { x: 230, y: 500 }]);
    for (let i = 1; i <= 6; i++) {
      await dispatchTouch(cdp, "touchMove", [{ x: 130, y: 500 - i * 20 }, { x: 230, y: 500 - i * 20 }]);
    }
    await dispatchTouch(cdp, "touchEnd", []);

    await expect.poll(async () => Math.abs((await readProfile(page)).controls.cameraPanY - before.cameraPanY)).toBeGreaterThan(0.01);
  });
});

test.describe("desktop cockpit", () => {
  test("keeps the full default particle budget", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("main.app-shell")).not.toHaveClass(/is-mobile/);
    await expect.poll(async () => (await readProfile(page)).liveConfig.particleCount).toBe(3399680);
    // Desktop keeps the preset's own render tuning.
    const controls = (await readProfile(page)).controls;
    expect(controls.particleGlowCore).toBe(1);
    expect(controls.particleSupportMask).toBe(1);
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
