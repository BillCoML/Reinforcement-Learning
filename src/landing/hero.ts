/**
 * Landing hero: a live Q-learning gridworld training run as background, with
 * title, subtitle, and primary CTAs in the foreground. The agent learns from
 * scratch every ~25 seconds — heatmap, policy arrows, and trail update live.
 */
import { loop, mkCanvas, tok, lerp } from "./anim";

const GRID_W = 10;
const GRID_H = 7;
const START: [number, number] = [0, 0];
const GOAL: [number, number] = [GRID_W - 1, GRID_H - 1];
const WALLS = new Set([
  "2,1", "2,2", "2,3",
  "5,3", "5,4", "5,5",
  "7,0", "7,1",
  "4,6",
]);

type State = [number, number];
const actions: [number, number][] = [[1, 0], [-1, 0], [0, 1], [0, -1]];

function key(s: State): string {
  return `${s[0]},${s[1]}`;
}
function isWall(s: State): boolean {
  return WALLS.has(key(s));
}
function inBounds(s: State): boolean {
  return s[0] >= 0 && s[0] < GRID_W && s[1] >= 0 && s[1] < GRID_H;
}
function step(s: State, a: number): { ns: State; r: number; done: boolean } {
  const ns: State = [s[0] + actions[a][0], s[1] + actions[a][1]];
  if (!inBounds(ns) || isWall(ns)) return { ns: s, r: -0.04, done: false };
  if (ns[0] === GOAL[0] && ns[1] === GOAL[1]) return { ns, r: 1, done: true };
  return { ns, r: -0.02, done: false };
}

class QLearner {
  Q = new Float64Array(GRID_W * GRID_H * 4);
  s: State = [START[0], START[1]];
  ep = 0;
  stepsInEp = 0;
  trail: State[] = [];

  idx(s: State, a: number): number {
    return (s[1] * GRID_W + s[0]) * 4 + a;
  }
  maxQ(s: State): number {
    const b = (s[1] * GRID_W + s[0]) * 4;
    return Math.max(this.Q[b], this.Q[b + 1], this.Q[b + 2], this.Q[b + 3]);
  }
  argmax(s: State): number {
    const b = (s[1] * GRID_W + s[0]) * 4;
    let m = -Infinity, mi = 0;
    for (let a = 0; a < 4; a++) if (this.Q[b + a] > m) { m = this.Q[b + a]; mi = a; }
    return mi;
  }
  pick(s: State, eps: number): number {
    if (Math.random() < eps) return Math.floor(Math.random() * 4);
    return this.argmax(s);
  }
  tick(alpha: number, gamma: number, eps: number): void {
    const a = this.pick(this.s, eps);
    const { ns, r, done } = step(this.s, a);
    const target = done ? r : r + gamma * this.maxQ(ns);
    const i = this.idx(this.s, a);
    this.Q[i] += alpha * (target - this.Q[i]);
    this.trail.push([this.s[0], this.s[1]]);
    if (this.trail.length > 22) this.trail.shift();
    this.s = ns;
    this.stepsInEp++;
    if (done || this.stepsInEp > 180) {
      this.ep++;
      this.s = [START[0], START[1]];
      this.stepsInEp = 0;
    }
  }
  reset(): void {
    this.Q = new Float64Array(GRID_W * GRID_H * 4);
    this.s = [START[0], START[1]];
    this.ep = 0;
    this.stepsInEp = 0;
    this.trail = [];
  }
}

export function createHero(): HTMLElement {
  const hero = document.createElement("section");
  hero.className = "landing-hero";

  const stage = document.createElement("div");
  stage.className = "landing-hero__stage";
  hero.appendChild(stage);

  const W = 1200, H = 520;
  const { c, g } = mkCanvas(stage, W, H);
  c.classList.add("landing-hero__canvas");

  // Foreground overlay (title, CTAs)
  const overlay = document.createElement("div");
  overlay.className = "landing-hero__overlay";
  overlay.innerHTML = `
    <p class="landing-hero__eyebrow">An interactive curriculum · 12 lessons · 185 tests</p>
    <h1 class="landing-hero__title">Reinforcement Learning,<br/>from first principles.</h1>
    <p class="landing-hero__sub">
      From multi-armed bandits to soft actor-critic. Every concept derived,
      every visualization live, every result verified.
    </p>
    <div class="landing-hero__ctas">
      <a class="landing-cta landing-cta--primary" href="#bandits">
        Start with Lesson 1 <span aria-hidden="true">→</span>
      </a>
      <a class="landing-cta landing-cta--ghost" href="#curriculum">
        See the curriculum
      </a>
    </div>
    <p class="landing-hero__legend">
      <span class="dot dot--start"></span> start
      <span class="dot dot--goal"></span> goal
      <span class="dot dot--agent"></span> agent
      &nbsp;·&nbsp;tabular Q-learning, α = 0.5, γ = 0.95, ε decays
    </p>
  `;
  stage.appendChild(overlay);

  const learner = new QLearner();
  // Steps-per-frame scales: start fast, hold near optimum, then restart.
  let lastFrameT = 0;
  let cumT = 0;

  loop(stage, (t) => {
    const dt = t - lastFrameT;
    lastFrameT = t;
    cumT += dt;
    if (cumT > 25) { learner.reset(); cumT = 0; }

    // run multiple Q-learning steps per frame for visible training speed
    const stepsThisFrame = Math.min(40, 4 + Math.floor(cumT * 2));
    const eps = Math.max(0.05, 0.5 - cumT * 0.02);
    for (let k = 0; k < stepsThisFrame; k++) learner.tick(0.5, 0.95, eps);

    drawScene(g, W, H, learner, cumT);
  });

  return hero;
}

function drawScene(g: CanvasRenderingContext2D, W: number, H: number, L: QLearner, time: number) {
  g.clearRect(0, 0, W, H);

  // Background: soft warm gradient
  const grd = g.createLinearGradient(0, 0, 0, H);
  grd.addColorStop(0, tok("--rl-bg"));
  grd.addColorStop(1, tok("--rl-surface-2"));
  g.fillStyle = grd;
  g.fillRect(0, 0, W, H);

  // Grid placement: take up the right ~55% of the canvas so it sits behind overlay
  const gridLeftRatio = 0.42;
  const cellH = Math.min((H - 80) / GRID_H, (W * (1 - gridLeftRatio) - 60) / GRID_W);
  const gridW = cellH * GRID_W;
  const gridH = cellH * GRID_H;
  const ox = W - gridW - 40;
  const oy = (H - gridH) / 2;

  // Cells
  for (let r = 0; r < GRID_H; r++) {
    for (let c = 0; c < GRID_W; c++) {
      const s: State = [c, r];
      const wall = isWall(s);
      const q = wall ? 0 : Math.max(0, L.maxQ(s));
      const x = ox + c * cellH;
      const y = oy + r * cellH;
      if (wall) {
        g.fillStyle = "rgba(28,30,34,0.55)";
        g.fillRect(x, y, cellH - 2, cellH - 2);
      } else {
        // value-heatmap fill
        g.fillStyle = `rgba(14, 116, 144, ${0.06 + Math.min(0.7, q) * 0.55})`;
        g.fillRect(x, y, cellH - 2, cellH - 2);
      }
      // Goal & start markers
      if (c === GOAL[0] && r === GOAL[1]) {
        g.fillStyle = tok("--rl-algo-optimal");
        g.fillRect(x + 2, y + 2, cellH - 6, cellH - 6);
      } else if (c === START[0] && r === START[1]) {
        g.fillStyle = tok("--rl-algo-greedy");
        g.globalAlpha = 0.7;
        g.fillRect(x + 2, y + 2, cellH - 6, cellH - 6);
        g.globalAlpha = 1;
      }
    }
  }

  // Cell borders
  g.strokeStyle = "rgba(28,30,34,0.10)";
  g.lineWidth = 1;
  for (let r = 0; r <= GRID_H; r++) {
    g.beginPath();
    g.moveTo(ox, oy + r * cellH);
    g.lineTo(ox + gridW, oy + r * cellH);
    g.stroke();
  }
  for (let c = 0; c <= GRID_W; c++) {
    g.beginPath();
    g.moveTo(ox + c * cellH, oy);
    g.lineTo(ox + c * cellH, oy + gridH);
    g.stroke();
  }

  // Policy arrows
  for (let r = 0; r < GRID_H; r++) {
    for (let c = 0; c < GRID_W; c++) {
      const s: State = [c, r];
      if (isWall(s) || (c === GOAL[0] && r === GOAL[1])) continue;
      const q = L.maxQ(s);
      if (q < 0.03) continue;
      const a = L.argmax(s);
      const cx = ox + c * cellH + cellH / 2;
      const cy = oy + r * cellH + cellH / 2;
      drawArrow(g, cx, cy, actions[a][0], actions[a][1], Math.min(0.5, q) * cellH * 0.7);
    }
  }

  // Trail
  if (L.trail.length > 1) {
    g.strokeStyle = tok("--rl-algo-thompson");
    g.lineWidth = 2.4;
    g.beginPath();
    for (let i = 0; i < L.trail.length; i++) {
      const [c, r] = L.trail[i];
      const x = ox + c * cellH + cellH / 2;
      const y = oy + r * cellH + cellH / 2;
      g.globalAlpha = 0.15 + 0.7 * (i / L.trail.length);
      if (i === 0) g.moveTo(x, y);
      else g.lineTo(x, y);
    }
    g.stroke();
    g.globalAlpha = 1;
  }

  // Agent
  const [ac, ar] = L.s;
  const ax = ox + ac * cellH + cellH / 2;
  const ay = oy + ar * cellH + cellH / 2;
  g.fillStyle = tok("--rl-algo-thompson");
  g.beginPath();
  g.arc(ax, ay, Math.max(6, cellH * 0.22), 0, Math.PI * 2);
  g.fill();
  g.strokeStyle = "rgba(255,255,255,0.9)";
  g.lineWidth = 2;
  g.stroke();

  // Episode counter (lower-right)
  g.fillStyle = tok("--rl-ink-faint");
  g.font = '11px "JetBrains Mono", monospace';
  g.fillText(`episode ${L.ep}`, ox + gridW - 96, oy + gridH + 18);
  g.fillText(`ε = ${Math.max(0.05, 0.5 - time * 0.02).toFixed(2)}`, ox, oy + gridH + 18);
}

function drawArrow(g: CanvasRenderingContext2D, x: number, y: number, dx: number, dy: number, len: number) {
  if (len < 4) return;
  g.strokeStyle = "rgba(28,30,34,0.55)";
  g.fillStyle = "rgba(28,30,34,0.55)";
  g.lineWidth = 1.4;
  const x1 = x - (dx * len) / 2;
  const y1 = y - (dy * len) / 2;
  const x2 = x + (dx * len) / 2;
  const y2 = y + (dy * len) / 2;
  g.beginPath();
  g.moveTo(x1, y1);
  g.lineTo(x2, y2);
  g.stroke();
  g.beginPath();
  g.moveTo(x2, y2);
  g.lineTo(x2 - dx * 5 + dy * 3, y2 - dy * 5 - dx * 3);
  g.lineTo(x2 - dx * 5 - dy * 3, y2 - dy * 5 + dx * 3);
  g.closePath();
  g.fill();
}

// `lerp` is imported but only used in tiles.ts via re-export; keep it referenced
// here so the bundler doesn't drop the helper when this file is built solo.
void lerp;
