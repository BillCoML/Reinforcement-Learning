# Agent Build Brief — Lesson 1: Multi-Armed Bandits

> **You are a coding agent picking up this lesson cold.** Read this document
> end-to-end before touching any code. Then read `BANDITS_LESSON_README.md`
> end-to-end. Then, before writing any code, return to the user the short plan
> requested in §6 below.

---

## 1. What's already in the workspace

This is **Lesson 1 of a fresh RL curriculum project.** Treat the workspace as
empty unless told otherwise. There is no prior code to integrate with; you are
scaffolding the project as part of this build.

This RL project is **independent of the StatViz project** the user previously
worked on. Do not import code from StatViz. The visual/aesthetic direction here
is different (see §2 of the lesson README).

---

## 2. Documents to read, in order

1. **This document** (the build brief you are reading now).
2. **`BANDITS_LESSON_README.md`** — the full lesson specification. This is the
   authoritative source for prose, math, numerical examples, visualization specs,
   and acceptance criteria. When in doubt, the README wins.

You may also see one of these in the workspace later; *ignore them for now*
unless the user explicitly directs you:

- A future `RL_SYSTEM_AND_ROADMAP.md` catalog file (not yet written).
- Future lesson specs (Lesson 2 onward).

---

## 3. What you'll build

A standalone Vite + TypeScript site for Lesson 1. File layout:

```
project-root/
├── index.html                       # entry point, mounts the lesson
├── package.json
├── tsconfig.json                    # strict mode
├── vite.config.ts
├── scripts/
│   └── simulate_bandits.py          # offline simulation, produces JSON
├── public/
│   ├── data/bandits/
│   │   ├── regret_curves_default.json
│   │   └── beta_posterior_snapshots.json
│   └── fonts/                       # IBM Plex Serif, Inter, JetBrains Mono
├── src/
│   ├── main.ts                      # mounts the page
│   ├── styles/
│   │   ├── tokens.css               # CSS variables from spec §2
│   │   └── base.css                 # prose layout, headings, panel chrome
│   ├── lesson/
│   │   ├── meta.ts                  # LessonMeta object from spec §3
│   │   ├── section-01-problem.ts    # prose + V1
│   │   ├── section-02-regret.ts     # prose + V2
│   │   ├── section-03-dilemma.ts    # prose + V3
│   │   ├── section-04-epsgreedy.ts  # prose + V4
│   │   ├── section-05-ucb.ts        # prose + V5
│   │   ├── section-06-thompson.ts   # prose + V6
│   │   ├── section-07-arena.ts      # prose + V7 (centerpiece)
│   │   └── section-08-roadmap.ts    # prose + V8
│   ├── bandits/
│   │   ├── algorithms.ts            # EpsilonGreedy, UCB1, ThompsonBetaBernoulli
│   │   ├── env.ts                   # BernoulliBandit, GaussianBandit
│   │   ├── regret.ts                # RegretTracker
│   │   ├── stats.ts                 # laiRobbinsConstant, hoeffdingBound, sampleBeta
│   │   └── algorithms.test.ts       # Vitest tests from spec §5
│   └── components/
│       ├── BanditMachine.ts         # V1
│       ├── RegretDecomposition.ts   # V2
│       ├── TwoExtremeFailure.ts     # V3
│       ├── EpsilonGreedyExplorer.ts # V4
│       ├── UCBConfidenceBounds.ts   # V5
│       ├── ThompsonPosteriorEvolution.ts  # V6
│       ├── AlgorithmBattleArena.ts  # V7 (centerpiece)
│       ├── RoadmapMini.ts           # V8
│       ├── MathBlock.ts             # KaTeX wrapper
│       ├── CrosslinkCallout.ts      # for back/forward/sidebar links
│       └── PanelChrome.ts           # shared border/header for interactive panels
└── README.md                        # short "how to run" file
```

---

## 4. Tech stack — locked, do not deviate

- **Vite 5+** with TypeScript **strict mode** (`"strict": true` in tsconfig).
- **No framework.** No React, no Vue, no Svelte. Plain TS modules and Web
  Components where reusability matters. The prose pages are HTML strings
  assembled per section.
- **KaTeX** (npm `katex`) for math rendering. Auto-render on mount.
- **D3.js v7** for visualizations.
- **ml-matrix** is available for any linear algebra; light usage in Lesson 1.
- **Vitest** for tests. Configure `vitest.config.ts` with jsdom environment.
- **Python 3.11+** with **NumPy** for the offline simulation script.

Fonts: IBM Plex Serif (prose), Inter (UI), JetBrains Mono (mono/numbers). Use
@fontsource packages or local font files; do not load from Google Fonts at
runtime (privacy + slow).

---

## 5. File management rules (recap)

- All build output goes in `/dist/` (Vite default).
- The Python simulation script writes to `public/data/bandits/`. Run it manually
  during development; do not invoke it from the build pipeline yet (Vite build
  should not depend on Python).
- Numerical examples in the prose **must match** what the algorithms produce.
  If you change the algorithms' RNG or hyperparameters, re-run the Python
  simulation and update the table in section 7.
- Commit early, commit often. Each step in §7 below is a separate commit.

---

## 6. First action: read everything, then return a plan

**Before writing any code,** return to the user a short plan covering:

1. **Catalog state.** Confirm you've read both this brief and
   `BANDITS_LESSON_README.md`. Note any ambiguities you spotted.
2. **Dependency footprint.** Exact npm packages and versions you intend to install.
3. **Pre-simulation plan.** Confirm you'll run `scripts/simulate_bandits.py`
   once during scaffolding and check the resulting JSON into `public/data/`.
   Note expected file sizes (should be tens of KB, not MB).
4. **Centerpiece architecture.** A paragraph on how you'll structure
   `AlgorithmBattleArena` — what the data flow looks like (simulation engine →
   reactive store → D3 render), how you'll handle the 200-seed averaging
   efficiently in-browser, and how you'll keep it responsive at 60fps.
5. **Cross-link plan.** Lesson 1 has no incoming cross-links yet (it's the
   first lesson). The forward links named in spec §3's
   `forwardLinksWhenReady` should be implemented as placeholder callout
   components that render gracefully even though their targets don't exist yet.
6. **Open questions.** Anything you need a decision on before starting. Be
   specific.

**Word budget: 450–700 words.** Wait for the user to confirm or revise before
proceeding to §7.

---

## 7. Build sequence

Each step is a separate commit. Do not collapse steps. Each step has a clear
done-condition.

### Step 1 — Math modules and tests
**Files:** `src/bandits/{algorithms,env,regret,stats}.ts`,
`src/bandits/algorithms.test.ts`.

Implement everything in spec §5 (Algorithm / Math Implementation). Tests use the
numerical targets from spec §2, §5, §6:

- Lai-Robbins constant for `μ = (0.3, 0.5, 0.7)` → `3.4744` (3 decimals).
- UCB1 picks arm 2 at `t = 4` after rewards `(0, 1, 0)`.
- Beta-Bernoulli posterior after 7s/3f → `(α, β) = (8, 4)`.
- Hoeffding bound at `(n=100, ε=0.1)` → `0.270671`.

**Done when:** `npm test` shows all green.

### Step 2 — Scaffolding
**Files:** `index.html`, `vite.config.ts`, `tsconfig.json`, `package.json`,
`src/main.ts`, `src/styles/tokens.css`, `src/styles/base.css`,
empty section files.

Mount an empty lesson skeleton with the title, prereq strip (placeholder), and
empty section divs. Verify CSS variables from spec §2 load correctly and the
page reads in IBM Plex Serif on the warm off-white background.

**Done when:** `npm run dev` shows the lesson skeleton at localhost with all
fonts loaded and CSS variables visible in devtools.

### Step 3 — Offline simulation
**Files:** `scripts/simulate_bandits.py`,
`public/data/bandits/regret_curves_default.json`,
`public/data/bandits/beta_posterior_snapshots.json`.

Implement the simulation per spec §11. Run it. Verify the JSON contains:

- `algos.eps01.mean[4999]` close to `112.0` (within ±5).
- `algos.eps001.mean[4999]` close to `97.0`.
- `algos.ucb1.mean[4999]` close to `84.2`.
- `algos.thompson.mean[4999]` close to `17.4`.

(All values match the spec's §7 table. If your simulation diverges substantially,
investigate the RNG and the algorithm implementation before proceeding.)

**Done when:** the JSON files exist, sizes are reasonable, and the listed
checkpoint values check out.

### Step 4 — Sections 1–3 prose + V1, V2, V3
**Files:** `section-01-problem.ts`, `section-02-regret.ts`,
`section-03-dilemma.ts`, `BanditMachine.ts`, `RegretDecomposition.ts`,
`TwoExtremeFailure.ts`.

Paste the prose from spec §4 verbatim (with light HTML/KaTeX markup). Wire up
the math blocks. Implement V1, V2, V3 to spec.

V3 is mostly static and pulls data from `regret_curves_default.json`. V1 and V2
are smaller interactives.

**Done when:** sections 1-3 render, look polished, and the math typesets
cleanly.

### Step 5 — Sections 4–6 prose + V4, V5, V6
**Files:** `section-04-epsgreedy.ts`, `section-05-ucb.ts`,
`section-06-thompson.ts`, `EpsilonGreedyExplorer.ts`,
`UCBConfidenceBounds.ts`, `ThompsonPosteriorEvolution.ts`.

V5 (UCB confidence bounds) is the most polished of the three — spec budgets
~2 days for it. Make sure the range bars expand subtly on un-pulled arms
(reflecting `log t` growth) — this is the visual hook that makes UCB legible.

V6's PDF curves should tween smoothly (~300ms transition) when a posterior
updates. Use D3 transitions, not raw setInterval.

**Done when:** all three sections render, the interactives respond at 60fps
on a 5-year-old laptop, and the spec's numerical traces from §5 and §6 are
reproducible by clicking through V5 / V6.

### Step 6 — Section 7 + V7 (centerpiece)
**Files:** `section-07-arena.ts`, `AlgorithmBattleArena.ts`.

**This is the polish-budget sink.** Spend 3-4 days here.

The Battle Arena must:

- Run 200 seeds × 5000 steps × 5 algorithms in ≤ 3 seconds when "Run" is hit.
  Use Float64Array for state, avoid object allocation in the hot loop.
- Animate the regret curve smoothly even while simulating. Render in
  requestAnimationFrame; do not block the main thread for more than 16ms at a
  time. If 200 seeds is too slow, drop default to 100 seeds and expose 200 as
  a power-user option.
- The pull-distribution mini-charts at the bottom are non-negotiable. They are
  the single most important visualization in the lesson because they reveal
  *how* each algorithm differs, not just how well it scores.

Test the export-PNG button. Test the URL-share feature
(`?seed=42&algos=ucb,ts`).

**Done when:** the Battle Arena works for K=2 through K=8, all five algorithms
produce reasonable curves, and a fresh-eyes review (yours) says "this is the
component someone would screenshot and share."

### Step 7 — Section 8 + V8 + page polish
**Files:** `section-08-roadmap.ts`, `RoadmapMini.ts`.

V8 is a *placeholder* — the full roadmap doesn't exist yet (it comes when the
catalog file is written). Implement it as a simple node-and-arrow diagram with
"Lesson 1" highlighted and four arrows leading to grayed-out future lessons.
Hover should show the destination's title.

This is also when you do the page-level polish: reduced-motion media query,
mobile fallback notice for V4-V7, URL-shareable seeds.

**Done when:** the page reads top-to-bottom cleanly, the print stylesheet
collapses interactives to static images (their first-frame default), and a
keyboard-only user can navigate the lesson.

### Step 8 — Verification pass
- All Vitest tests still pass.
- All spec numerical examples (§2 gap decomposition, §5 UCB trace, §6 Beta
  parameters) reproduce when you click through the interactives.
- Lighthouse score ≥ 90 on Performance and ≥ 95 on Accessibility.
- No console errors.
- The `regret_curves_default.json` values match what the in-browser simulation
  produces (the offline Python and in-browser TS should agree to ~1% on means).

**Done when:** all of the above check out and you can write the user a 5-bullet
summary of what's shipped.

---

## 8. Verifying your work (operational checks)

For each step's done-condition above, do not move on until it is met. If you
hit a blocker, **stop and ask the user.** Do not invent your way around the
spec.

Specific gotchas:

- **D3 with TypeScript strict mode** has rough edges around selection generics.
  Use `d3.Selection<SVGGElement, unknown, null, undefined>` patterns. Don't
  try to be clever with type inference here.
- **KaTeX rendering inside Web Components** needs careful timing — `katex.render()`
  must run after the element is in the DOM. Use `connectedCallback`.
- **JSDOM does not implement SVG geometry** — your V1-V8 component tests should
  not depend on layout. Only test the data-transformation logic.
- **Beta sampling in TS**: the easiest correct implementation is via two Gamma
  samples (Beta(α,β) = Gamma(α) / (Gamma(α) + Gamma(β))). The naïve
  inverse-CDF approach is wrong for fractional α, β. Use the Marsaglia-Tsang
  Gamma sampler.

---

## 9. Constraints (locked decisions)

- The aesthetic (spec §2) is locked. Do not change colors, fonts, or layout
  conventions.
- The algorithm signature colors (ε-greedy = amber, UCB = teal, Thompson =
  violet, random = gray, optimal = green) are curriculum-wide. Treat them as
  immutable.
- The lesson is single-page. No tabs, no collapsible accordion sections, no
  hidden content. Everything is scrollable.
- The Battle Arena (V7) is the *only* component that may break out of the
  prose column to ~960px width. Everything else fits the 720-880 range.
- No analytics, no telemetry, no external CDN dependencies at runtime. The
  lesson must work offline once loaded.

---

## 10. When you finish

Final state of the world:

- A working Vite project that runs locally with `npm run dev`.
- All Vitest tests green.
- Eight visualizations rendering correctly.
- The Battle Arena is screenshot-worthy.
- A 5-bullet summary handed to the user describing what's shipped, plus any
  technical debt or open issues you want flagged.

Then stop. Lesson 2 will arrive as a separate spec.

---

## End of brief

If anything in this document conflicts with `BANDITS_LESSON_README.md`, the
README wins. If you spot the conflict, flag it to the user in your initial
plan (§6 above).
