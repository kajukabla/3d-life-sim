# Security policy

Please report suspected vulnerabilities through GitHub's private vulnerability reporting for this repository. Do not publish secrets or exploit details in a public issue.

The hosted application is a static browser build and does not require API keys or an application backend. Browser microphone and MIDI access require an explicit browser permission. The optional audio-helper WebSocket is disabled by default and restricted to loopback addresses.

Cloudflare Pages serves a checked-in content security policy and browser hardening headers. Preset imports accept JSON files up to 1 MB, normalize user-controlled names, sanitize settings into supported ranges, and always disable camera lock. Treat preset files as untrusted input and avoid committing private captures, credentials, environment files, or personal data.
