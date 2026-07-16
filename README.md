# 3D Life Sim

A real-time, browser-based 3D particle life simulation powered by WebGPU. Hundreds of thousands of particles respond to a configurable local field, creating evolving swarms, filaments, shells, and emergent structures.

## Live demo

<https://3d-life-sim.pages.dev>

## Requirements

- Node.js 22 or newer
- pnpm 10
- A WebGPU-capable browser

## Run locally

```bash
pnpm install
pnpm dev
```

Open the local URL Vite prints. The simulation starts automatically. Use the cockpit to change the physics, rendering, camera, presets, timeline, audio modulation, and MIDI mappings.

Useful shortcuts:

- `F` — toggle fullscreen
- `M` — hide or show the cockpit
- `R` — start or stop a performance recording

Browser microphone analysis is opt-in with `?audio=mic`. WebSocket audio input is also opt-in and accepts loopback connections only. The normal simulation does not contact a backend and needs no API keys.

## Development

```bash
pnpm check
```

This runs TypeScript checking, the unit test suite, and a production build.

## Deploy

The repository includes a Cloudflare Pages configuration:

```bash
pnpm deploy
```

## Project scope

This repository contains the browser simulation, a particle-only public render path, three curated particle presets (`vid1`, `vid2`, and `AR11`), and its tests. The copied presets contain no MIDI mappings. Cache playback, alternate render-mode controls and presets, Philips Hue integration, native helpers, offline render tools, generated captures, private development notes, and the original monorepo history are intentionally excluded.

## License and attribution

The project is MIT licensed. Portions are derived from Jesse Gelders' MIT-licensed [Fluoddity](https://github.com/aphid91/Fluoddity), and the post-processing shader includes an MIT-licensed AgX implementation by Missing Deadlines (Benjamin Wrensch). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
