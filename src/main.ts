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
import "./styles/markov-tokens.css";
import "./styles/mdp-tokens.css";
import "./styles/base.css";
import "./styles/components.css";
import "./styles/markov-components.css";
import "./styles/mdp-components.css";
import "./styles/contractions-tokens.css";
import "./styles/dp-tokens.css";
import "./styles/is-tokens.css";
import "./styles/mc-tokens.css";
import "./styles/td-tokens.css";
import "./styles/dqn-tokens.css";
import "./styles/pg-tokens.css";
import "./styles/ppo-tokens.css";
import "./styles/maxent-tokens.css";
import "./styles/landing.css";

import { buildLanding } from "./landing";
import type { LessonMeta } from "./lesson/meta";
import type { Section } from "./lesson/section";
import { typesetMath } from "./lesson/render-math";

import { lessonMeta as banditsMeta } from "./lesson/meta";
import { section01 } from "./lesson/section-01-problem";
import { section02 } from "./lesson/section-02-regret";
import { section03 } from "./lesson/section-03-dilemma";
import { section04 } from "./lesson/section-04-epsgreedy";
import { section05 } from "./lesson/section-05-ucb";
import { section06 } from "./lesson/section-06-thompson";
import { section07 } from "./lesson/section-07-arena";
import { section08 } from "./lesson/section-08-roadmap";

import { markovMeta, markovSections } from "./lesson/markov-chains";
import { mdpMeta, mdpSections } from "./lesson/mdps";
import { contractionsMeta, contractionsSections } from "./lesson/contractions";
import { dpMeta, dpSections } from "./lesson/dp";
import { isMeta, isSections } from "./lesson/importance-sampling";
import { mcMeta, mcSections } from "./lesson/monte-carlo";
import { tdMeta, tdSections } from "./lesson/td-learning";
import { dqnMeta, dqnSections } from "./lesson/function-approximation";
import { pgMeta, pgSections } from "./lesson/policy-gradient";
import { ppoMeta, ppoSections } from "./lesson/l11-trpo-ppo";
import { maxentMeta, maxentSections } from "./lesson/l12-max-ent-rl";

// Register all interactive Web Components (self-registering on import).
import "./components/registry";
import "./components/markov-registry";
import "./components/mdp-registry";
import "./components/contractions-registry";
import "./components/dp-registry";
import "./components/is-registry";
import "./components/mc-registry";
import "./components/td-registry";
import "./components/dqn-registry";
import "./components/pg-registry";
import "./components/ppo-registry";
import "./components/maxent-registry";

interface LessonDef {
  /** URL slug, e.g. "bandits" → /bandits. */
  slug: string;
  /** Eyebrow label shown above the title, e.g. "Lesson 1" or "Prereq A". */
  eyebrow: string;
  meta: LessonMeta;
  sections: Section[];
}

const LESSONS: LessonDef[] = [
  { slug: "bandits", eyebrow: "Lesson 1", meta: banditsMeta, sections: [section01, section02, section03, section04, section05, section06, section07, section08] },
  { slug: "markov-chains", eyebrow: "Prereq A · before Lesson 2", meta: markovMeta, sections: markovSections },
  { slug: "mdps", eyebrow: "Lesson 2", meta: mdpMeta, sections: mdpSections },
  { slug: "contractions", eyebrow: "Prereq C · before Lesson 3", meta: contractionsMeta, sections: contractionsSections },
  { slug: "dynamic-programming", eyebrow: "Lesson 3", meta: dpMeta, sections: dpSections },
  { slug: "importance-sampling", eyebrow: "Lesson 6", meta: isMeta, sections: isSections },
  { slug: "monte-carlo", eyebrow: "Lesson 7", meta: mcMeta, sections: mcSections },
  { slug: "td-learning", eyebrow: "Lesson 8", meta: tdMeta, sections: tdSections },
  { slug: "function-approximation", eyebrow: "Lesson 9", meta: dqnMeta, sections: dqnSections },
  { slug: "policy-gradient", eyebrow: "Lesson 10", meta: pgMeta, sections: pgSections },
  { slug: "trpo-ppo", eyebrow: "Lesson 11", meta: ppoMeta, sections: ppoSections },
  { slug: "max-ent-rl", eyebrow: "Lesson 12", meta: maxentMeta, sections: maxentSections },
];

const DEFAULT_SLUG = "home";
const HOME_SLUG = "home";

function difficultyDots(n: number, max = 5): string {
  return "●".repeat(n) + "○".repeat(max - n);
}

/** Top navigation strip letting the reader switch lessons. */
function buildNav(activeSlug: string): HTMLElement {
  const nav = document.createElement("nav");
  nav.className = "lesson-nav";
  const entries: { slug: string; label: string }[] = [
    { slug: HOME_SLUG, label: "Home" },
    ...LESSONS.map((l) => ({ slug: l.slug, label: l.meta.title })),
  ];
  for (const e of entries) {
    const a = document.createElement("a");
    a.href = routeHref(e.slug);
    a.textContent = e.label;
    a.className = "lesson-nav__link" + (e.slug === activeSlug ? " is-active" : "");
    a.dataset.slug = e.slug;
    a.addEventListener("click", (ev) => {
      if (ev.metaKey || ev.ctrlKey || ev.shiftKey) return; // let new-tab work
      ev.preventDefault();
      navigateTo(e.slug);
    });
    nav.appendChild(a);
  }
  return nav;
}

function buildMasthead(def: LessonDef): HTMLElement {
  const m = def.meta;
  const header = document.createElement("header");
  header.className = "masthead";

  const prereqs = m.prerequisites.map((p) => `<span class="chip">${p.label}</span>`).join("");

  header.innerHTML = `
    <p class="eyebrow">RL Curriculum · Tier ${m.tier} · ${def.eyebrow}</p>
    <h1>${m.title}</h1>
    <p class="subtitle">${m.subtitle}</p>
    <div class="prereq-strip">
      <span class="chip rl-mono" title="difficulty">diff ${difficultyDots(m.difficulty)}</span>
      <span class="meta-dot">·</span>
      <span class="rl-mono">~${m.estimatedReadMinutes} min</span>
      <span class="meta-dot">·</span>
      <span style="color:var(--rl-ink-muted)">prereqs:</span>
      ${prereqs}
    </div>`;
  return header;
}

// ---- routing --------------------------------------------------------------

/**
 * Derive the active slug from the URL. Supports both path form (/markov-chains,
 * works in dev/preview via SPA fallback) and hash form (#markov-chains, robust
 * for static/file:// hosting and base-path deploys).
 */
function currentSlug(): string {
  const hash = location.hash.replace(/^#\/?/, "");
  if (hash === HOME_SLUG) return HOME_SLUG;
  if (LESSONS.some((l) => l.slug === hash)) return hash;
  const seg = location.pathname.replace(/\/+$/, "").split("/").pop() ?? "";
  if (seg === HOME_SLUG) return HOME_SLUG;
  if (LESSONS.some((l) => l.slug === seg)) return seg;
  return DEFAULT_SLUG;
}

/** Build a shareable href for a slug, preserving the hash-vs-path convention in use. */
function routeHref(slug: string): string {
  // If we arrived via a hash route (or can't use clean paths), keep hashes.
  if (location.hash || location.protocol === "file:") return `#${slug}`;
  const base = location.pathname.replace(/\/[^/]*$/, "/");
  return `${base}${slug}`;
}

function navigateTo(slug: string): void {
  history.pushState({ slug }, "", routeHref(slug));
  render();
}

let renderedSlug: string | null = null;

function render(): void {
  const root = document.getElementById("app");
  if (!root) throw new Error("#app not found");

  const slug = currentSlug();
  // Skip rebuild when the slug hasn't changed — lets in-page anchors
  // like #curriculum on the landing scroll naturally without a re-render.
  if (renderedSlug === slug) return;
  renderedSlug = slug;
  root.innerHTML = "";

  const article = document.createElement("article");
  article.className = "lesson-root";

  if (slug === HOME_SLUG) {
    document.title = "Reinforcement Learning · An Interactive Curriculum";
    article.classList.add("is-landing");
    article.appendChild(buildNav(HOME_SLUG));
    article.appendChild(buildLanding());
    root.appendChild(article);
    window.scrollTo(0, 0);
    return;
  }

  const def = LESSONS.find((l) => l.slug === slug) ?? LESSONS[0];
  document.title = `${def.meta.title} · RL Curriculum`;
  article.appendChild(buildNav(def.slug));
  article.appendChild(buildMasthead(def));

  for (const s of def.sections) article.appendChild(s.build());

  root.appendChild(article);
  window.scrollTo(0, 0);

  // KaTeX must run after the DOM is populated.
  typesetMath(article);

  mountRoadmapJump(article);
}

let roadmapObserver: IntersectionObserver | null = null;

/**
 * Floating "Roadmap" button in the bottom-right that scrolls to the lesson's
 * roadmap-mini panel. Hides itself when the panel is already on screen.
 */
function mountRoadmapJump(article: HTMLElement): void {
  roadmapObserver?.disconnect();
  roadmapObserver = null;

  const target = article.querySelector(
    "roadmap-mini, dqn-roadmap, maxent-roadmap-mini, ppo-roadmap-mini",
  ) as HTMLElement | null;
  if (!target) return;

  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "roadmap-jump";
  btn.setAttribute("aria-label", "Jump to roadmap");
  btn.innerHTML = `<span aria-hidden="true">↓</span> Roadmap`;
  btn.addEventListener("click", () => {
    target.scrollIntoView({ behavior: "smooth", block: "start" });
  });
  article.appendChild(btn);

  roadmapObserver = new IntersectionObserver(
    (entries) => {
      const visible = entries[0]?.isIntersecting ?? false;
      btn.classList.toggle("roadmap-jump--hidden", visible);
    },
    { threshold: 0.15 },
  );
  roadmapObserver.observe(target);
}

window.addEventListener("popstate", render);
// In-prose / roadmap links navigate by setting location.hash (#slug); re-render
// when it changes to a known lesson so cross-lesson links work without a full nav.
window.addEventListener("hashchange", render);
render();
