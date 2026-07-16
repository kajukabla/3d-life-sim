import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const headers = readFileSync(new URL("../../public/_headers", import.meta.url), "utf8");

describe("static deployment security headers", () => {
  it("applies a restrictive policy to every route", () => {
    expect(headers).toMatch(/^\/\*$/m);
    expect(headers).toContain("Content-Security-Policy:");
    for (const directive of [
      "default-src 'self'",
      "script-src 'self'",
      "worker-src 'self'",
      "style-src 'self'",
      "img-src 'self' data:",
      "connect-src 'self'",
      "object-src 'none'",
      "base-uri 'none'",
      "frame-ancestors 'none'"
    ]) {
      expect(headers).toContain(directive);
    }
    expect(headers).not.toContain("'unsafe-inline'");
    expect(headers).not.toContain("'unsafe-eval'");
    expect(headers).not.toContain("ws://");
  });

  it("sets browser hardening and feature policies", () => {
    expect(headers).toContain("X-Frame-Options: DENY");
    expect(headers).toContain("Referrer-Policy: no-referrer");
    expect(headers).toContain("X-Content-Type-Options: nosniff");
    expect(headers).toContain("Strict-Transport-Security: max-age=31536000; includeSubDomains");
    expect(headers).toContain("Permissions-Policy: microphone=(self), midi=(self), fullscreen=(self)");
  });
});
