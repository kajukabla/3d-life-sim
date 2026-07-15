import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"]
  },
  server: {
    host: "127.0.0.1",
    port: 6197,
    strictPort: true
  }
});
