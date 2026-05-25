/**
 * The single shared value-function color scale, used by V4–V7 and the
 * GridworldRenderer so that "0.729" reads as the same green everywhere.
 * Diverging green↔red around 0, fixed domain [-1.5, 1.0] (locked in the spec).
 */
import * as d3 from "d3";
import { cssVar } from "./base";

export const VALUE_DOMAIN: [number, number] = [-1.5, 1.0];

/** Build a diverging value→color scale. Reads the --mdp-value-* tokens live. */
export function makeValueColorScale(
  domain: [number, number] = VALUE_DOMAIN,
): (v: number) => string {
  const neg = cssVar("--mdp-value-neg") || "#b91c1c";
  const zero = cssVar("--mdp-value-zero") || "#f1ede4";
  const pos = cssVar("--mdp-value-pos") || "#15803d";
  const scale = d3
    .scaleLinear<string>()
    .domain([domain[0], 0, domain[1]])
    .range([neg, zero, pos])
    .interpolate(d3.interpolateLab)
    .clamp(true);
  return (v: number) => scale(v);
}

/** Readable text color (dark or light) for text sitting on `bg`. */
export function textColorOn(bg: string): string {
  const c = d3.color(bg)?.rgb();
  if (!c) return "var(--rl-ink)";
  // Relative luminance (sRGB approximation).
  const lum = (0.299 * c.r + 0.587 * c.g + 0.114 * c.b) / 255;
  return lum < 0.55 ? "#ffffff" : "#1c1e22";
}

/** Fixed 3-decimal formatting for value functions (curriculum convention). */
export function fmtV(v: number): string {
  return v.toFixed(3);
}
