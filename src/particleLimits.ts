export const minLiveParticles = 128;
// 2^23 particles * 8 floats * 4 bytes = 256 MiB, the WebGPU default maxBufferSize.
export const maxLiveParticles = 8388608;
export const liveParticleStep = 1024;
