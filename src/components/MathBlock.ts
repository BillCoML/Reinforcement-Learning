/**
 * KaTeX wrapper for rendering math *inside* components (status readouts, pick
 * reasons, axis labels). Prose math is handled separately by the auto-render
 * pass in main.ts; this is for dynamically-generated strings.
 */
import katex from "katex";

export function mathToHTML(tex: string, display = false): string {
  return katex.renderToString(tex, {
    throwOnError: false,
    displayMode: display,
  });
}

/** Render `tex` into `el` (replacing its contents). */
export function renderMathInto(el: HTMLElement, tex: string, display = false): void {
  katex.render(tex, el, { throwOnError: false, displayMode: display });
}
