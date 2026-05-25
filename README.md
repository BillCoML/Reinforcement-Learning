# Reinforcement-Learning

An interactive RL curriculum. **Lesson 1 — Multi-Armed Bandits**: exploration,
exploitation, and the geometry of regret.

A single-page Vite + TypeScript site (no UI framework) with eight D3
visualizations, KaTeX-typeset math, and an in-browser algorithm tournament
(the "Battle Arena").

## Run it

```bash
npm install
npm run dev        # dev server at http://localhost:5173
```

## Other commands

```bash
npm run build      # type-check + production build to dist/
npm run preview    # preview the production build
npm test           # Vitest unit tests (algorithms, stats, arena sim)
npm run simulate   # regenerate public/data/bandits/*.json (needs Python 3.11+ & NumPy)
```

## What's inside

| Code | Component | Section |
|------|-----------|---------|
| V1 | Bandit Machine | §1 The bandit problem |
| V2 | Regret Decomposition | §2 Regret |
| V3 | Two-Extreme Failure | §3 Exploration–exploitation |
| V4 | ε-Greedy Explorer | §4 ε-greedy |
| V5 | UCB Confidence Bounds | §5 UCB |
| V6 | Thompson Posterior Evolution | §6 Thompson sampling |
| V7 | **Algorithm Battle Arena** (centerpiece) | §7 Empirical comparison |
| V8 | Roadmap Mini | §8 Forward links |

- `src/bandits/` — algorithms, environments, regret, stats, and the Battle
  Arena simulation engine (with Vitest tests).
- `src/components/` — the eight Web Components plus shared chrome.
- `src/lesson/` — per-section prose + metadata.
- `scripts/simulate_bandits.py` — offline simulation that produces the
  pre-computed regret curves and posterior trace in `public/data/bandits/`.
- `docs/` — the lesson spec (`BANDITS_LESSON_README.md`) and build brief.

## Notes

- The Battle Arena (V7) accepts URL state, e.g. `?seed=42&algos=ucb,ts`.
- Heavy interactive panels (V4–V7) show a "view on desktop" notice on narrow
  screens; everything respects `prefers-reduced-motion` and degrades for print.
- Algorithm signature colors are curriculum-wide: ε-greedy = amber,
  UCB = teal, Thompson = violet, random = gray, optimal = green.
