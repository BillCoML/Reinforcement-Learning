/**
 * Store — a minimal synchronous pub/sub store. The Convergence Lab (V5) keeps
 * all four panels in lockstep by having them subscribe to one of these; the
 * play loop and controls mutate it via set(), and every panel re-renders from
 * the single shared state. No framework, fully testable.
 */
export class Store<T extends object> {
  private subs = new Set<(s: T) => void>();
  private state: T;

  constructor(initial: T) {
    this.state = initial;
  }

  get(): T {
    return this.state;
  }

  set(patch: Partial<T>): void {
    this.state = { ...this.state, ...patch };
    for (const fn of this.subs) fn(this.state);
  }

  subscribe(fn: (s: T) => void): () => void {
    this.subs.add(fn);
    return () => this.subs.delete(fn);
  }
}
