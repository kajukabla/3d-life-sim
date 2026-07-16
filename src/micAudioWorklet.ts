import { MicFilterbank } from "./micAudioDsp";

declare const sampleRate: number;
declare const currentTime: number;
declare abstract class AudioWorkletProcessor {
  readonly port: MessagePort;
  abstract process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean;
}
declare function registerProcessor(name: string, processorCtor: typeof AudioWorkletProcessor): void;

const analysisWindowSize = 512;

class LifeAudioAnalyzerProcessor extends AudioWorkletProcessor {
  private readonly filterbank = new MicFilterbank(sampleRate);
  private readonly window = new Float32Array(analysisWindowSize);
  private writeOffset = 0;
  private sequence = 0;

  process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    for (const output of outputs[0] ?? []) output.fill(0);
    const input = inputs[0]?.[0];
    if (!input) return true;

    let readOffset = 0;
    while (readOffset < input.length) {
      const count = Math.min(input.length - readOffset, this.window.length - this.writeOffset);
      this.window.set(input.subarray(readOffset, readOffset + count), this.writeOffset);
      readOffset += count;
      this.writeOffset += count;
      if (this.writeOffset === this.window.length) {
        this.sequence += 1;
        this.port.postMessage(this.filterbank.analyzeWindow(this.window, this.sequence, currentTime));
        this.writeOffset = 0;
      }
    }
    return true;
  }
}

registerProcessor("life-audio-analyzer", LifeAudioAnalyzerProcessor);
