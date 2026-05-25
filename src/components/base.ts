/** Shared helpers for the interactive components. */

export function prefersReducedMotion(): boolean {
  return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
}

/** Resolve a CSS custom property to its computed color string. */
export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** Per-arm color: optimal arm green, others a UCB-tinted ramp by index. */
export function armColor(i: number, isOptimal: boolean): string {
  if (isOptimal) return "var(--rl-algo-optimal)";
  const tints = ["#0e7490", "#3b82a6", "#6796ad"];
  return tints[i % tints.length];
}

/** Format a number with fixed decimals in mono context. */
export function fmt(x: number, dp = 3): string {
  return x.toFixed(dp);
}

export type Cleanup = () => void;
