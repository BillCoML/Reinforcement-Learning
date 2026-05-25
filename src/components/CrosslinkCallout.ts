/**
 * Cross-link callouts. Lesson 1 has no *incoming* links yet (it's first), and
 * its forward links point at lessons that don't exist. So forward links render
 * as styled, non-navigating callouts that degrade gracefully: a label, the
 * prose, and a muted "(coming in Lesson N)" marker instead of a live anchor.
 */

export interface ForwardCalloutOptions {
  /** Destination lesson label, e.g. "Lesson 10 — Max-Entropy RL". */
  destination: string;
  /** Whether the target exists yet (false for all Lesson 1 forward links). */
  ready?: boolean;
  /** Href for the link when ready (e.g. "#mdps"). Defaults to "#". */
  href?: string;
  /** Body HTML (may contain KaTeX `$…$`). */
  html: string;
}

/** Forward link → future lesson. Returns an HTML string for inlining in prose. */
export function forwardLink(opts: ForwardCalloutOptions): string {
  const marker = opts.ready
    ? `<a href="${opts.href ?? "#"}">${opts.destination}</a>`
    : `<span class="future-target" title="Not yet written">${opts.destination} · coming soon</span>`;
  return `<blockquote class="forward-link">
    <span class="label">Forward link → ${marker}</span>
    ${opts.html}
  </blockquote>`;
}

/** A thematic sidebar (violet), for asides like "Knowing what you don't know". */
export function sidebar(title: string, html: string): string {
  return `<blockquote class="sidebar">
    <span class="label">${title}</span>
    ${html}
  </blockquote>`;
}
