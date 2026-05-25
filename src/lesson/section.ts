/** A lesson section: an id/title plus a function that builds its DOM. */
export interface Section {
  id: string; // anchor id
  title: string;
  build(): HTMLElement;
}

/**
 * Small helper to turn an HTML string into a <section> element. KaTeX math
 * (delimited by $…$ / $$…$$) is rendered later, once the whole article is in
 * the DOM, by main.ts.
 */
export function sectionFromHTML(id: string, html: string): HTMLElement {
  const el = document.createElement("section");
  el.id = id;
  el.className = "lesson-section";
  el.innerHTML = html;
  return el;
}
