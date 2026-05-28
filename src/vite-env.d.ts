/// <reference types="vite/client" />

declare module "onnxruntime-web" {
  export interface Tensor {
    data: Float32Array;
  }
  export class Tensor {
    constructor(type: string, data: Float32Array, dims: number[]);
  }
  export namespace InferenceSession {
    function create(path: string): Promise<InferenceSession>;
  }
  export interface InferenceSession {
    run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>>;
  }
}
