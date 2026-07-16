import { defineConfig, devices } from "@playwright/test";

const localBaseUrl = "http://127.0.0.1:4173";
const externalBaseUrl = process.env.PLAYWRIGHT_BASE_URL?.replace(/\/$/, "");

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 2 : 0,
  workers: 1,
  reporter: process.env.CI ? [["line"], ["html", { open: "never" }]] : "line",
  timeout: 90_000,
  expect: { timeout: 15_000 },
  use: {
    baseURL: externalBaseUrl ?? localBaseUrl,
    permissions: ["microphone"],
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
    video: "retain-on-failure"
  },
  webServer: externalBaseUrl
    ? undefined
    : {
        command: "pnpm exec vite preview --host 127.0.0.1 --port 4173",
        url: localBaseUrl,
        reuseExistingServer: !process.env.CI,
        timeout: 30_000
      },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        launchOptions: {
          args: [
            "--enable-unsafe-webgpu",
            "--use-fake-device-for-media-stream",
            "--use-fake-ui-for-media-stream"
          ]
        }
      }
    }
  ]
});
