/**
 * StepLoop — a requestAnimationFrame-driven stepping loop with play / pause /
 * single-step / speed control. Each logical step takes `baseIntervalMs / speed`
 * of wall-clock time; multiple steps are batched per frame at high speed (so we
 * never block longer than one frame), and reduced-motion users still get manual
 * stepping. The render callback runs once per frame after the steps.
 */
import { prefersReducedMotion } from "./base";

export interface StepLoopOptions {
  baseIntervalMs: number;
  maxStepsPerFrame?: number;
  onStep: () => boolean; // return false to stop (e.g. horizon reached)
  onFrame?: () => void; // render once per animated frame
}

export class StepLoop {
  private speed = 1;
  private playing = false;
  private accum = 0;
  private last = 0;
  private raf = 0;
  private readonly maxPerFrame: number;

  constructor(private opts: StepLoopOptions) {
    this.maxPerFrame = opts.maxStepsPerFrame ?? 64;
  }

  setSpeed(mult: number): void {
    this.speed = mult;
  }

  isPlaying(): boolean {
    return this.playing;
  }

  play(): void {
    if (this.playing) return;
    this.playing = true;
    this.last = performance.now();
    this.accum = 0;
    const tick = (now: number) => {
      if (!this.playing) return;
      const dt = now - this.last;
      this.last = now;
      const interval = this.opts.baseIntervalMs / this.speed;
      this.accum += dt;
      let steps = 0;
      while (this.accum >= interval && steps < this.maxPerFrame) {
        this.accum -= interval;
        steps++;
        if (!this.opts.onStep()) {
          this.pause();
          break;
        }
      }
      if (steps > 0) this.opts.onFrame?.();
      if (this.playing) this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }

  pause(): void {
    this.playing = false;
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = 0;
  }

  /** Execute one step immediately and render. */
  step(): void {
    this.pause();
    if (this.opts.onStep()) this.opts.onFrame?.();
  }

  /** Whether the environment prefers no animated tweens. */
  get reducedMotion(): boolean {
    return prefersReducedMotion();
  }

  destroy(): void {
    this.pause();
  }
}
