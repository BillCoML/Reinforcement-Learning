/** Landing page composition for the RL curriculum. */
import { createHero } from "./hero";
import { createCurriculumMap } from "./curriculum-map";
import { buildTileGrid } from "./tiles";

export function buildLanding(): HTMLElement {
  const root = document.createElement("div");
  root.className = "landing-root";

  root.appendChild(createHero());
  root.appendChild(buildCredo());
  root.appendChild(createCurriculumMap());
  root.appendChild(buildTileGrid());
  root.appendChild(buildWhy());
  root.appendChild(buildFinalCTA());

  return root;
}

function buildCredo(): HTMLElement {
  const s = document.createElement("section");
  s.className = "landing-section landing-credo";
  s.innerHTML = `
    <div class="landing-credo__row">
      <div class="landing-credo__stat"><span>12</span> lessons</div>
      <div class="landing-credo__stat"><span>2</span> math prereqs</div>
      <div class="landing-credo__stat"><span>185</span> verified tests</div>
      <div class="landing-credo__stat"><span>0</span> black boxes</div>
    </div>
    <p class="landing-credo__line">
      Every formula derived. Every visualization live. Every result tested. No imports of "rl"; the algorithms are written in the browser, in TypeScript, in front of you.
    </p>
  `;
  return s;
}

function buildWhy(): HTMLElement {
  const s = document.createElement("section");
  s.className = "landing-section landing-why";
  s.innerHTML = `
    <h2 class="landing-section__title">Why this curriculum</h2>
    <div class="landing-why__grid">
      <div class="landing-why__card">
        <div class="landing-why__num">01</div>
        <h3>From first principles</h3>
        <p>We don't drop equations; we derive them. Bellman, contractions, the policy gradient theorem, GAE, the soft Bellman operator — every result is earned.</p>
      </div>
      <div class="landing-why__card">
        <div class="landing-why__num">02</div>
        <h3>Interactive end-to-end</h3>
        <p>Pull bandit arms, drive value iteration, sweep PPO's clip parameter, slide max-ent temperature — every concept has a knob you can turn.</p>
      </div>
      <div class="landing-why__card">
        <div class="landing-why__num">03</div>
        <h3>Verified by tests</h3>
        <p>185 unit tests check the math: convergence rates, variance bounds, soft-Bellman fixed points. If a formula is wrong, the tests catch it.</p>
      </div>
      <div class="landing-why__card">
        <div class="landing-why__num">04</div>
        <h3>Threads to modern RL</h3>
        <p>Each lesson ends pointing forward: TD(λ) → GAE → PPO; importance sampling → offline RL; entropy bonus → SAC, RL-as-inference, RLHF.</p>
      </div>
    </div>
  `;
  return s;
}

function buildFinalCTA(): HTMLElement {
  const s = document.createElement("section");
  s.className = "landing-section landing-final";
  s.innerHTML = `
    <h2>Ready to begin?</h2>
    <p>Start with multi-armed bandits — the simplest RL problem, the cleanest introduction to the exploration/exploitation dilemma.</p>
    <div class="landing-final__ctas">
      <a class="landing-cta landing-cta--primary landing-cta--lg" href="#bandits">
        Lesson 1 — Multi-Armed Bandits <span aria-hidden="true">→</span>
      </a>
      <a class="landing-cta landing-cta--ghost" href="#markov-chains">
        Skip ahead — Markov Chains
      </a>
    </div>
    <p class="landing-final__foot">
      Built in the open — TypeScript + D3 + KaTeX, no proprietary deps.
      <a href="https://github.com/BillCoML/Reinforcement-Learning" target="_blank" rel="noopener">Source on GitHub →</a>
    </p>
  `;
  return s;
}
