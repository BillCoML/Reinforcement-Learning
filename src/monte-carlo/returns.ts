/** Compute discounted returns G_t via backward sweep. */
export function computeReturns(rewards: number[], gamma: number): number[] {
  const T = rewards.length;
  const Gs = new Array<number>(T).fill(0);
  let running = 0;
  for (let t = T - 1; t >= 0; t--) {
    running = rewards[t] + gamma * running;
    Gs[t] = running;
  }
  return Gs;
}
