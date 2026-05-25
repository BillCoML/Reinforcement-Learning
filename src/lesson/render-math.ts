import renderMathInElement from "katex/contrib/auto-render";

/** Render all $…$ / $$…$$ math inside `root` with KaTeX. */
export function typesetMath(root: HTMLElement): void {
  renderMathInElement(root, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false },
    ],
    throwOnError: false,
    macros: {
      "\\KL": "\\mathrm{KL}",
      "\\E": "\\mathbb{E}",
      "\\muhat": "\\hat{\\mu}",
    },
  });
}
