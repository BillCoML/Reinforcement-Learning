// Fonts (bundled locally via @fontsource — no runtime Google Fonts).
import "@fontsource/ibm-plex-serif/400.css";
import "@fontsource/ibm-plex-serif/400-italic.css";
import "@fontsource/ibm-plex-serif/600.css";
import "@fontsource/inter/400.css";
import "@fontsource/inter/500.css";
import "@fontsource/inter/600.css";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/500.css";

// KaTeX + lesson styles.
import "katex/dist/katex.min.css";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/components.css";

import { lessonMeta } from "./lesson/meta";
import type { Section } from "./lesson/section";
import { typesetMath } from "./lesson/render-math";

import { section01 } from "./lesson/section-01-problem";
import { section02 } from "./lesson/section-02-regret";
import { section03 } from "./lesson/section-03-dilemma";
import { section04 } from "./lesson/section-04-epsgreedy";
import { section05 } from "./lesson/section-05-ucb";
import { section06 } from "./lesson/section-06-thompson";
import { section07 } from "./lesson/section-07-arena";
import { section08 } from "./lesson/section-08-roadmap";

// Register all interactive Web Components (self-registering on import).
import "./components/registry";

const sections: Section[] = [
  section01,
  section02,
  section03,
  section04,
  section05,
  section06,
  section07,
  section08,
];

function difficultyDots(n: number, max = 5): string {
  return "●".repeat(n) + "○".repeat(max - n);
}

function buildMasthead(): HTMLElement {
  const m = lessonMeta;
  const header = document.createElement("header");
  header.className = "masthead";

  const prereqs = m.prerequisites
    .map((p) => `<span class="chip">${p.label}</span>`)
    .join("");

  header.innerHTML = `
    <p class="eyebrow">RL Curriculum · Tier ${m.tier} · Lesson 1</p>
    <h1>${m.title}</h1>
    <p class="subtitle">${m.subtitle}</p>
    <div class="prereq-strip">
      <span class="chip rl-mono" title="difficulty">diff ${difficultyDots(m.difficulty)}</span>
      <span class="meta-dot">·</span>
      <span class="rl-mono">~${m.estimatedReadMinutes} min</span>
      <span class="meta-dot">·</span>
      <span style="color:var(--rl-ink-faint)">prereqs:</span>
      ${prereqs}
    </div>`;
  return header;
}

function mount(): void {
  const root = document.getElementById("app");
  if (!root) throw new Error("#app not found");

  const article = document.createElement("article");
  article.className = "lesson-root";
  article.appendChild(buildMasthead());

  for (const s of sections) {
    article.appendChild(s.build());
  }

  root.appendChild(article);

  // KaTeX must run after the DOM is populated.
  typesetMath(article);
}

mount();
