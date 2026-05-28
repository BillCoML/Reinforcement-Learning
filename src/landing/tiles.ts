/**
 * Twelve compact, looping animations — one per lesson — for the landing-page grid.
 * Each tile renders a self-contained Canvas/SVG vignette of the lesson's
 * central idea. Tiles are clickable and route via location.hash.
 */
import { loop, mkCanvas, tok, lerp, makeTile } from "./anim";

const TILE_W = 240;
const TILE_H = 140;

/* ── L1 Bandits ─────────────────────────────────────────────── 3 arms, ε-greedy
   Three reward bars on the right; an "agent" hand picks an arm each beat. */
function banditsTile(host: HTMLElement): void {
  const { g, W, H } = mkCanvas(host, TILE_W, TILE_H);
  const trueQ = [0.35, 0.55, 0.72];
  const N = [0, 0, 0];
  const Q = [0, 0, 0];
  let pick = 0;
  let pulse = 0;

  loop(host, (t) => {
    // ε-greedy step every ~0.35s
    const step = Math.floor(t / 0.35);
    if (step > N[0] + N[1] + N[2]) {
      pick = Math.random() < 0.25 ? Math.floor(Math.random() * 3) : Q.indexOf(Math.max(...Q));
      const r = trueQ[pick] + (Math.random() - 0.5) * 0.2;
      N[pick]++;
      Q[pick] += (r - Q[pick]) / N[pick];
      pulse = 1;
    }
    pulse = Math.max(0, pulse - 0.04);

    g.clearRect(0, 0, W, H);
    g.fillStyle = tok("--rl-surface-2");
    g.fillRect(0, 0, W, H);

    const barW = 36;
    const gap = 28;
    const startX = (W - (3 * barW + 2 * gap)) / 2;
    const baseY = H - 22;
    const maxH = H - 50;

    for (let i = 0; i < 3; i++) {
      const x = startX + i * (barW + gap);
      const h = Math.max(4, Q[i] * maxH);
      const isPick = i === pick;
      g.fillStyle = i === 0 ? tok("--rl-algo-greedy") : i === 1 ? tok("--rl-algo-ucb") : tok("--rl-algo-thompson");
      g.globalAlpha = isPick ? 1 : 0.55;
      g.fillRect(x, baseY - h, barW, h);
      g.globalAlpha = 1;
      // true-Q tick mark
      g.fillStyle = tok("--rl-ink-faint");
      g.fillRect(x - 3, baseY - trueQ[i] * maxH - 1, barW + 6, 1);
      // pulse on pick
      if (isPick && pulse > 0) {
        g.strokeStyle = tok("--rl-algo-ucb");
        g.globalAlpha = pulse;
        g.lineWidth = 2;
        g.strokeRect(x - 3, baseY - h - 3, barW + 6, h + 6);
        g.globalAlpha = 1;
      }
    }
    g.fillStyle = tok("--rl-ink-faint");
    g.font = '10px "JetBrains Mono", monospace';
    g.fillText(`n=${N[0] + N[1] + N[2]}`, 10, 14);
  });
}

/* ── Prereq A: Markov Chains ────── 4-node cycle, particle walks, histogram emerges */
function markovTile(host: HTMLElement): void {
  const { g, W, H } = mkCanvas(host, TILE_W, TILE_H);
  const cx = W / 2;
  const cy = H / 2 - 6;
  const R = 32;
  const nodes = [0, 1, 2, 3].map((i) => ({
    x: cx + R * Math.cos((i * Math.PI) / 2 - Math.PI / 2),
    y: cy + R * Math.sin((i * Math.PI) / 2 - Math.PI / 2),
  }));
  const P = [
    [0.1, 0.6, 0.2, 0.1],
    [0.2, 0.1, 0.6, 0.1],
    [0.1, 0.1, 0.1, 0.7],
    [0.6, 0.1, 0.2, 0.1],
  ];
  let cur = 0;
  let next = 1;
  let tStep = 0;
  const visits = [0, 0, 0, 0];

  loop(host, (t) => {
    const sStep = 0.45;
    const frame = (t / sStep) % 1;
    if (frame < tStep) {
      cur = next;
      visits[cur]++;
      let r = Math.random();
      for (let j = 0; j < 4; j++) {
        r -= P[cur][j];
        if (r <= 0) { next = j; break; }
      }
    }
    tStep = frame;

    g.clearRect(0, 0, W, H);
    g.fillStyle = tok("--rl-surface-2");
    g.fillRect(0, 0, W, H);

    // edges
    g.strokeStyle = tok("--rl-border");
    g.lineWidth = 1;
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        if (i !== j && P[i][j] > 0.15) {
          g.beginPath();
          g.moveTo(nodes[i].x, nodes[i].y);
          g.lineTo(nodes[j].x, nodes[j].y);
          g.stroke();
        }
      }
    }
    // nodes
    for (let i = 0; i < 4; i++) {
      g.fillStyle = i === cur ? tok("--rl-algo-ucb") : tok("--rl-surface");
      g.strokeStyle = tok("--rl-ink-muted");
      g.lineWidth = 1.2;
      g.beginPath();
      g.arc(nodes[i].x, nodes[i].y, 9, 0, Math.PI * 2);
      g.fill();
      g.stroke();
    }
    // particle interpolating cur → next
    const px = lerp(nodes[cur].x, nodes[next].x, frame);
    const py = lerp(nodes[cur].y, nodes[next].y, frame);
    g.fillStyle = tok("--rl-algo-thompson");
    g.beginPath();
    g.arc(px, py, 4, 0, Math.PI * 2);
    g.fill();

    // histogram bar at bottom
    const total = visits.reduce((a, b) => a + b, 0) || 1;
    const bw = 14;
    const bx0 = 14;
    const by = H - 6;
    for (let i = 0; i < 4; i++) {
      const h = (visits[i] / total) * 32;
      g.fillStyle = tok("--rl-algo-ucb");
      g.globalAlpha = 0.55;
      g.fillRect(bx0 + i * (bw + 4), by - h, bw, h);
      g.globalAlpha = 1;
    }
  });
}

/* ── L2 MDPs ──────── 4×4 gridworld with policy arrows, ghost agent traces path */
function mdpTile(host: HTMLElement): void {
  const { g, W, H } = mkCanvas(host, TILE_W, TILE_H);
  const N = 4;
  const cellH = Math.min(H - 24, (W - 80) / N);
  const ox = (W - cellH * N) / 2;
  const oy = (H - cellH * N) / 2;
  // hand-crafted V* heatmap with goal in bottom-right
  const V: number[][] = [];
  for (let r = 0; r < N; r++) {
    V[r] = [];
    for (let c = 0; c < N; c++) {
      const d = Math.abs(N - 1 - r) + Math.abs(N - 1 - c);
      V[r][c] = Math.pow(0.85, d);
    }
  }
  const goal = [N - 1, N - 1];
  // path: monotone right/down random
  let pathT = 0;

  loop(host, (t) => {
    pathT = (t / 4) % 1;

    g.clearRect(0, 0, W, H);
    g.fillStyle = tok("--rl-surface-2");
    g.fillRect(0, 0, W, H);

    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const v = V[r][c];
        g.fillStyle = `rgba(14, 116, 144, ${0.10 + v * 0.55})`;
        g.fillRect(ox + c * cellH, oy + r * cellH, cellH - 1, cellH - 1);
        // policy arrow
        if (r === goal[0] && c === goal[1]) {
          g.fillStyle = tok("--rl-algo-optimal");
          g.beginPath();
          g.arc(ox + c * cellH + cellH / 2, oy + r * cellH + cellH / 2, 4, 0, Math.PI * 2);
          g.fill();
        } else {
          const dr = r < N - 1 ? 1 : 0;
          const dc = c < N - 1 && (V[r][c + 1] ?? -1) >= (V[r + 1]?.[c] ?? -1) ? 1 : 0;
          const useRight = dc === 1 && r < N - 1 ? Math.random() < 0.0001 : dc === 1;
          drawArrow(g, ox + c * cellH + cellH / 2, oy + r * cellH + cellH / 2, useRight ? 1 : 0, useRight ? 0 : dr, 8);
        }
      }
    }

    // ghost agent walking from (0,0) along greedy path
    const steps = 2 * (N - 1);
    const stepF = pathT * steps;
    const k = Math.floor(stepF);
    const frac = stepF - k;
    let r = 0, c = 0;
    for (let i = 0; i < k; i++) {
      if (c + 1 < N && (V[r][c + 1] ?? -1) >= (V[r + 1]?.[c] ?? -1)) c++;
      else if (r + 1 < N) r++;
    }
    let nr = r, nc = c;
    if (c + 1 < N && (V[r][c + 1] ?? -1) >= (V[r + 1]?.[c] ?? -1)) nc = c + 1;
    else if (r + 1 < N) nr = r + 1;
    const ax = ox + (lerp(c, nc, frac) + 0.5) * cellH;
    const ay = oy + (lerp(r, nr, frac) + 0.5) * cellH;
    g.fillStyle = tok("--rl-algo-thompson");
    g.beginPath();
    g.arc(ax, ay, 5, 0, Math.PI * 2);
    g.fill();
  });
}

function drawArrow(g: CanvasRenderingContext2D, x: number, y: number, dx: number, dy: number, len: number) {
  if (dx === 0 && dy === 0) return;
  g.strokeStyle = tok("--rl-ink-muted");
  g.fillStyle = tok("--rl-ink-muted");
  g.lineWidth = 1.2;
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
  g.lineTo(x2 - dx * 3 + dy * 2, y2 - dy * 3 - dx * 2);
  g.lineTo(x2 - dx * 3 - dy * 2, y2 - dy * 3 + dx * 2);
  g.closePath();
  g.fill();
}

/* ── Prereq C: Contractions ───── |V_k - V*| decays geometrically; a fixed point on a line */
function contractionsTile(host: HTMLElement): void {
  const { g, W, H } = mkCanvas(host, TILE_W, TILE_H);
  const padL = 22, padR = 14, padT = 18, padB = 26;
  loop(host, (t) => {
    g.clearRect(0, 0, W, H);
    g.fillStyle = tok("--rl-surface-2");
    g.fillRect(0, 0, W, H);

    // axes
    g.strokeStyle = tok("--rl-border");
    g.beginPath();
    g.moveTo(padL, padT);
    g.lineTo(padL, H - padB);
    g.lineTo(W - padR, H - padB);
    g.stroke();
    g.fillStyle = tok("--rl-ink-faint");
    g.font = '9px "JetBrains Mono", monospace';
    g.fillText("‖V_k − V*‖", padL - 2, padT - 6);
    g.fillText("k", W - padR - 6, H - padB + 12);

    // geometric decay: c^k with c = 0.78
    const c = 0.78;
    const kMax = 24;
    const sweep = (t / 3.5) % 1;
    const visibleK = Math.min(kMax, Math.floor(sweep * kMax * 1.2));
    g.strokeStyle = tok("--rl-algo-ucb");
    g.lineWidth = 1.6;
    g.beginPath();
    for (let k = 0; k <= visibleK; k++) {
      const x = padL + (k / kMax) * (W - padL - padR);
      const y = H - padB - Math.pow(c, k) * (H - padT - padB);
      if (k === 0) g.moveTo(x, y);
      else g.lineTo(x, y);
    }
    g.stroke();

    // points
    g.fillStyle = tok("--rl-algo-ucb");
    for (let k = 0; k <= visibleK; k++) {
      const x = padL + (k / kMax) * (W - padL - padR);
      const y = H - padB - Math.pow(c, k) * (H - padT - padB);
      g.beginPath();
      g.arc(x, y, 2.2, 0, Math.PI * 2);
      g.fill();
    }

    // overlay: contraction factor
    g.fillStyle = tok("--rl-ink-muted");
    g.font = '10px "Inter", sans-serif';
    g.fillText(`γ = ${c}`, W - 56, padT + 12);
  });
}

/* ── L3 DP ──── value-iteration heatmap on 5×5 grid; cells fill smoothly */
function dpTile(host: HTMLElement): void {
  const { g, W, H } = mkCanvas(host, TILE_W, TILE_H);
  const N = 5;
  const cellH = Math.min(H - 16, (W - 40) / N);
  const ox = (W - cellH * N) / 2;
  const oy = (H - cellH * N) / 2;
  const goal = [N - 1, N - 1];
  const final: number[][] = [];
  for (let r = 0; r < N; r++) {
    final[r] = [];
    for (let c = 0; c < N; c++) {
      const d = Math.abs(N - 1 - r) + Math.abs(N - 1 - c);
      final[r][c] = Math.pow(0.85, d);
    }
  }

  loop(host, (t) => {
    const T = 4.5;
    const cycle = (t % T) / T;
    g.clearRect(0, 0, W, H);
    g.fillStyle = tok("--rl-surface-2");
    g.fillRect(0, 0, W, H);

    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        // wave outward from goal
        const d = Math.abs(N - 1 - r) + Math.abs(N - 1 - c);
        const phase = Math.max(0, Math.min(1, cycle * (2 * N) - d));
        const v = final[r][c] * phase;
        g.fillStyle = `rgba(14, 116, 144, ${0.08 + v * 0.6})`;
        g.fillRect(ox + c * cellH, oy + r * cellH, cellH - 1, cellH - 1);
        if (r === goal[0] && c === goal[1]) {
          g.fillStyle = tok("--rl-algo-optimal");
          g.beginPath();
          g.arc(ox + c * cellH + cellH / 2, oy + r * cellH + cellH / 2, 3.5, 0, Math.PI * 2);
          g.fill();
        }
      }
    }
    g.fillStyle = tok("--rl-ink-faint");
    g.font = '10px "JetBrains Mono", monospace';
    g.fillText(`k = ${Math.floor(cycle * 2 * N)}`, 8, 12);
  });
}

/* ── L6 Importance Sampling ─── two distributions, samples weighted by π/µ */
function isTile(host: HTMLElement): void {
  const { g, W, H } = mkCanvas(host, TILE_W, TILE_H);
  const padL = 14, padR = 14, padT = 14, padB = 18;
  const muMu = 0.4, muS = 0.18;
  const piMu = 0.65, piS = 0.12;
  const pdf = (x: number, m: number, s: number) =>
    Math.exp(-((x - m) ** 2) / (2 * s * s)) / (s * Math.sqrt(2 * Math.PI));
  const yScale = (H - padT - padB) / 3.5;

  type Dot = { x: number; w: number; born: number };
  let dots: Dot[] = [];
  let last = 0;

  loop(host, (t) => {
    if (t - last > 0.18) {
      last = t;
      const x = muMu + (Math.random() - 0.5) * 2 * muS;
      const w = pdf(x, piMu, piS) / Math.max(pdf(x, muMu, muS), 1e-6);
      dots.push({ x, w, born: t });
      if (dots.length > 40) dots.shift();
    }

    g.clearRect(0, 0, W, H);
    g.fillStyle = tok("--rl-surface-2");
    g.fillRect(0, 0, W, H);

    // pdf curves
    g.lineWidth = 1.4;
    for (const [m, s, color] of [[muMu, muS, tok("--rl-algo-greedy")], [piMu, piS, tok("--rl-algo-ucb")]] as const) {
      g.strokeStyle = color;
      g.beginPath();
      for (let i = 0; i <= 100; i++) {
        const xx = i / 100;
        const yy = pdf(xx, m, s);
        const px = padL + xx * (W - padL - padR);
        const py = H - padB - yy * yScale;
        if (i === 0) g.moveTo(px, py);
        else g.lineTo(px, py);
      }
      g.stroke();
    }

    // weighted dots
    for (const d of dots) {
      const age = Math.min(1, (t - d.born) / 1.0);
      const r = 1.5 + Math.min(d.w, 4) * 1.4;
      g.fillStyle = tok("--rl-algo-thompson");
      g.globalAlpha = 0.7 * (1 - age);
      g.beginPath();
      g.arc(padL + d.x * (W - padL - padR), H - padB + 2, r, 0, Math.PI * 2);
      g.fill();
    }
    g.globalAlpha = 1;
    g.fillStyle = tok("--rl-ink-faint");
    g.font = '9px "JetBrains Mono", monospace';
    g.fillText("µ", padL + muMu * (W - padL - padR) - 4, padT + 6);
    g.fillText("π", padL + piMu * (W - padL - padR) - 4, padT + 6);
  });
}

/* ── L7 Monte Carlo ──── many noisy samples; running mean converges to truth */
function mcTile(host: HTMLElement): void {
  const { g, W, H } = mkCanvas(host, TILE_W, TILE_H);
  const padL = 12, padR = 12, padT = 14, padB = 20;
  const trueMu = 0.62;
  type Sample = { x: number; y: number; born: number };
  let samples: Sample[] = [];
  let n = 0;
  let mean = 0;
  let last = 0;
  let nT = 0;

  loop(host, (t) => {
    if (t - last > 0.07) {
      last = t;
      nT++;
      const noise = (Math.random() - 0.5) * 0.6 + (Math.random() - 0.5) * 0.4;
      const y = Math.max(0.02, Math.min(0.98, trueMu + noise));
      samples.push({ x: nT, y, born: t });
      if (samples.length > 120) samples.shift();
      n++;
      mean += (y - mean) / n;
    }
    if (nT > 120) {
      nT = 0; n = 0; mean = 0; samples = [];
    }

    g.clearRect(0, 0, W, H);
    g.fillStyle = tok("--rl-surface-2");
    g.fillRect(0, 0, W, H);

    // true line
    g.strokeStyle = tok("--rl-algo-optimal");
    g.setLineDash([4, 3]);
    g.beginPath();
    g.moveTo(padL, H - padB - trueMu * (H - padT - padB));
    g.lineTo(W - padR, H - padB - trueMu * (H - padT - padB));
    g.stroke();
    g.setLineDash([]);

    // samples
    g.fillStyle = tok("--rl-ink-faint");
    for (const s of samples) {
      const px = padL + (s.x / 120) * (W - padL - padR);
      const py = H - padB - s.y * (H - padT - padB);
      g.globalAlpha = 0.4;
      g.beginPath();
      g.arc(px, py, 1.8, 0, Math.PI * 2);
      g.fill();
    }
    g.globalAlpha = 1;

    // running mean line
    let runMean = 0, runN = 0;
    g.strokeStyle = tok("--rl-algo-ucb");
    g.lineWidth = 1.6;
    g.beginPath();
    for (let i = 0; i < samples.length; i++) {
      runN++;
      runMean += (samples[i].y - runMean) / runN;
      const px = padL + (samples[i].x / 120) * (W - padL - padR);
      const py = H - padB - runMean * (H - padT - padB);
      if (i === 0) g.moveTo(px, py);
      else g.lineTo(px, py);
    }
    g.stroke();

    g.fillStyle = tok("--rl-ink-faint");
    g.font = '9px "JetBrains Mono", monospace';
    g.fillText(`x̄ = ${mean.toFixed(3)}`, 8, 12);
    g.fillText(`µ = ${trueMu.toFixed(2)}`, W - 56, 12);
  });
}

/* ── L8 TD Learning ──── chain of states; TD-error pulse propagates backward */
function tdTile(host: HTMLElement): void {
  const { g, W, H } = mkCanvas(host, TILE_W, TILE_H);
  const N = 6;
  const padX = 22;
  const dx = (W - padX * 2) / (N - 1);
  const yCenter = H / 2 + 4;
  const trueV = Array.from({ length: N }, (_, i) => i / (N - 1));
  const V = trueV.map(() => 0);

  loop(host, (t) => {
    // every 0.25s, do one backward-pass step
    const phase = (t / 4) % 1;
    // ε that decays over time
    for (let i = N - 1; i >= 0; i--) {
      const targetT = 1 - i / N;
      if (phase >= targetT && phase <= targetT + 0.15) {
        const target = i === N - 1 ? 1 : V[i + 1] * 0.9;
        V[i] += 0.18 * (target - V[i]);
      }
    }
    if (phase < 0.02) for (let i = 0; i < N; i++) V[i] = 0;

    g.clearRect(0, 0, W, H);
    g.fillStyle = tok("--rl-surface-2");
    g.fillRect(0, 0, W, H);

    // chain links
    g.strokeStyle = tok("--rl-border");
    g.beginPath();
    g.moveTo(padX, yCenter);
    g.lineTo(W - padX, yCenter);
    g.stroke();

    // states
    for (let i = 0; i < N; i++) {
      const x = padX + i * dx;
      const isGoal = i === N - 1;
      const v = V[i];
      g.fillStyle = isGoal ? tok("--rl-algo-optimal") : tok("--rl-surface");
      g.strokeStyle = tok("--rl-ink-muted");
      g.lineWidth = 1.2;
      g.beginPath();
      g.arc(x, yCenter, 11, 0, Math.PI * 2);
      g.fill();
      g.stroke();
      // value bar above each state
      const barH = v * 26;
      g.fillStyle = tok("--rl-algo-ucb");
      g.globalAlpha = 0.75;
      g.fillRect(x - 7, yCenter - 16 - barH, 14, barH);
      g.globalAlpha = 1;
    }
    g.fillStyle = tok("--rl-ink-faint");
    g.font = '9px "JetBrains Mono", monospace';
    g.fillText("V̂", 6, 14);
    g.fillText("Goal →", W - padX - 14, yCenter + 26);
  });
}

/* ── L9 Function Approx / DQN ───── tiny MLP, Q-values pulse, argmax highlighted */
function dqnTile(host: HTMLElement): void {
  const { g, W, H } = mkCanvas(host, TILE_W, TILE_H);
  const layers = [3, 5, 5, 4];
  const ox = 16;
  const span = W - 80;
  const cy = H / 2;

  loop(host, (t) => {
    const pulse = (Math.sin(t * 2) + 1) / 2;
    g.clearRect(0, 0, W, H);
    g.fillStyle = tok("--rl-surface-2");
    g.fillRect(0, 0, W, H);

    // edges
    g.strokeStyle = tok("--rl-border");
    g.lineWidth = 0.7;
    for (let l = 0; l < layers.length - 1; l++) {
      const x1 = ox + (l / (layers.length - 1)) * span;
      const x2 = ox + ((l + 1) / (layers.length - 1)) * span;
      const h1 = layers[l], h2 = layers[l + 1];
      for (let i = 0; i < h1; i++) {
        const y1 = cy + (i - (h1 - 1) / 2) * 14;
        for (let j = 0; j < h2; j++) {
          const y2 = cy + (j - (h2 - 1) / 2) * 14;
          g.globalAlpha = 0.3 + 0.4 * Math.abs(Math.sin(t * 1.3 + i + j));
          g.beginPath();
          g.moveTo(x1, y1);
          g.lineTo(x2, y2);
          g.stroke();
        }
      }
    }
    g.globalAlpha = 1;

    // neurons
    for (let l = 0; l < layers.length; l++) {
      const x = ox + (l / (layers.length - 1)) * span;
      const h = layers[l];
      for (let i = 0; i < h; i++) {
        const y = cy + (i - (h - 1) / 2) * 14;
        g.fillStyle = tok("--rl-algo-ucb");
        g.globalAlpha = 0.4 + 0.5 * Math.abs(Math.sin(t * 1.7 + i * 1.3 + l));
        g.beginPath();
        g.arc(x, y, 4, 0, Math.PI * 2);
        g.fill();
      }
    }
    g.globalAlpha = 1;

    // Q-bars on the right
    const Q = [
      0.4 + 0.15 * Math.sin(t * 0.9),
      0.6 + 0.2 * Math.sin(t * 1.1 + 1),
      0.78 + 0.1 * Math.sin(t * 0.7 + 2),
      0.3 + 0.12 * Math.sin(t * 1.3 + 3),
    ];
    const argmax = Q.indexOf(Math.max(...Q));
    const bx = W - 56;
    for (let i = 0; i < 4; i++) {
      const bw = Q[i] * 44;
      g.fillStyle = i === argmax ? tok("--rl-algo-optimal") : tok("--rl-ink-faint");
      g.globalAlpha = i === argmax ? 0.6 + 0.4 * pulse : 0.7;
      g.fillRect(bx, 22 + i * 22, bw, 14);
      g.globalAlpha = 1;
    }
    g.fillStyle = tok("--rl-ink-faint");
    g.font = '9px "JetBrains Mono", monospace';
    g.fillText("Q(s,a)", bx, 16);
  });
}

/* ── L10 Policy Gradient ─── action-prob distribution sharpens to point mass */
function pgTile(host: HTMLElement): void {
  const { g, W, H } = mkCanvas(host, TILE_W, TILE_H);
  const K = 4;
  const trueR = [0.3, 0.5, 0.85, 0.45];
  const logits = [0, 0, 0, 0];

  loop(host, (t) => {
    const T = 5;
    const cycle = (t % T) / T;
    if (cycle < 0.02) for (let i = 0; i < K; i++) logits[i] = 0;
    // softmax
    const m = Math.max(...logits);
    const ex = logits.map((l) => Math.exp(l - m));
    const Z = ex.reduce((a, b) => a + b, 0);
    const p = ex.map((e) => e / Z);
    // gradient step toward higher-reward action
    for (let i = 0; i < K; i++) {
      logits[i] += 0.03 * (trueR[i] - trueR.reduce((s, r, j) => s + r * p[j], 0)) * (1 - p[i]);
    }

    g.clearRect(0, 0, W, H);
    g.fillStyle = tok("--rl-surface-2");
    g.fillRect(0, 0, W, H);

    const padL = 24, padR = 12, padT = 18, padB = 22;
    const bw = (W - padL - padR) / K - 8;
    for (let i = 0; i < K; i++) {
      const x = padL + i * ((W - padL - padR) / K) + 4;
      const h = p[i] * (H - padT - padB);
      g.fillStyle = i === trueR.indexOf(Math.max(...trueR)) ? tok("--rl-algo-optimal") : tok("--rl-algo-ucb");
      g.fillRect(x, H - padB - h, bw, h);
      // reward tick on top
      g.fillStyle = tok("--rl-ink-faint");
      g.fillRect(x - 1, H - padB - trueR[i] * (H - padT - padB) - 1, bw + 2, 1);
    }
    g.fillStyle = tok("--rl-ink-faint");
    g.font = '9px "JetBrains Mono", monospace';
    g.fillText("π(a)", 8, 16);
  });
}

/* ── L11 TRPO / PPO ───── clipped surrogate L(r) — show clip zone + moving dot */
function ppoTile(host: HTMLElement): void {
  const { g, W, H } = mkCanvas(host, TILE_W, TILE_H);
  const eps = 0.2;
  const A = 1; // assume A > 0 for the visual

  loop(host, (t) => {
    g.clearRect(0, 0, W, H);
    g.fillStyle = tok("--rl-surface-2");
    g.fillRect(0, 0, W, H);

    const padL = 28, padR = 14, padT = 18, padB = 22;
    const xR = (r: number) => padL + ((r - 0.4) / 1.4) * (W - padL - padR);
    const yL = (l: number) => H - padB - ((l - 0) / 1.6) * (H - padT - padB);

    // axes
    g.strokeStyle = tok("--rl-border");
    g.beginPath();
    g.moveTo(padL, padT);
    g.lineTo(padL, H - padB);
    g.lineTo(W - padR, H - padB);
    g.stroke();

    // tinted clip region (1-ε, 1+ε)
    g.fillStyle = tok("--rl-ucb-tint");
    g.fillRect(xR(1 - eps), padT, xR(1 + eps) - xR(1 - eps), H - padT - padB);

    // surrogate: min(r*A, clip(r,1-ε,1+ε)*A)
    g.strokeStyle = tok("--rl-algo-ucb");
    g.lineWidth = 1.7;
    g.beginPath();
    for (let i = 0; i <= 120; i++) {
      const r = 0.4 + (i / 120) * 1.4;
      const clipped = Math.max(1 - eps, Math.min(1 + eps, r));
      const L = Math.min(r * A, clipped * A);
      const px = xR(r);
      const py = yL(L);
      if (i === 0) g.moveTo(px, py);
      else g.lineTo(px, py);
    }
    g.stroke();

    // moving dot on the curve
    const r = 0.6 + 0.6 * (1 + Math.sin(t * 1.2)) / 2;
    const clipped = Math.max(1 - eps, Math.min(1 + eps, r));
    const L = Math.min(r * A, clipped * A);
    g.fillStyle = tok("--rl-algo-thompson");
    g.beginPath();
    g.arc(xR(r), yL(L), 3.5, 0, Math.PI * 2);
    g.fill();

    g.fillStyle = tok("--rl-ink-faint");
    g.font = '9px "JetBrains Mono", monospace';
    g.fillText("L(r)", 6, 14);
    g.fillText("r = π/π_old", W - 70, H - 6);
  });
}

/* ── L12 Max-Entropy RL ─── Boltzmann over 4 actions; temperature sweeps */
function maxEntTile(host: HTMLElement): void {
  const { g, W, H } = mkCanvas(host, TILE_W, TILE_H);
  const Q = [0.2, 0.55, 0.85, 0.4];

  loop(host, (t) => {
    const T = 4.5;
    const cycle = (t % T) / T;
    // sweep alpha (temperature) from cold to hot and back
    const alpha = 0.04 + 0.55 * (1 - Math.cos(cycle * 2 * Math.PI)) / 2 + 1e-3;
    const ex = Q.map((q) => Math.exp(q / alpha));
    const Z = ex.reduce((a, b) => a + b, 0);
    const p = ex.map((e) => e / Z);

    g.clearRect(0, 0, W, H);
    g.fillStyle = tok("--rl-surface-2");
    g.fillRect(0, 0, W, H);

    const padL = 26, padR = 12, padT = 18, padB = 22;
    const k = 4;
    const slot = (W - padL - padR) / k;
    for (let i = 0; i < k; i++) {
      const x = padL + i * slot + 4;
      const h = p[i] * (H - padT - padB);
      g.fillStyle = i === Q.indexOf(Math.max(...Q)) ? tok("--rl-algo-optimal") : tok("--rl-algo-thompson");
      g.globalAlpha = 0.85;
      g.fillRect(x, H - padB - h, slot - 8, h);
      g.globalAlpha = 1;
      // Q tick
      g.fillStyle = tok("--rl-ink-faint");
      g.fillRect(x - 1, H - padB - Q[i] * (H - padT - padB) - 1, slot - 6, 1);
    }
    g.fillStyle = tok("--rl-ink-faint");
    g.font = '9px "JetBrains Mono", monospace';
    g.fillText(`α = ${alpha.toFixed(2)}`, 8, 14);
    g.fillText("π(a) ∝ exp(Q/α)", W - 110, 14);
  });
}

export function buildTileGrid(): HTMLElement {
  const wrap = document.createElement("section");
  wrap.className = "landing-section landing-tiles";

  const h = document.createElement("h2");
  h.className = "landing-section__title";
  h.textContent = "What you'll see in each lesson";
  wrap.appendChild(h);
  const sub = document.createElement("p");
  sub.className = "landing-section__sub";
  sub.textContent = "Every lesson is interactive — these are the live visualizations you'll get to drive.";
  wrap.appendChild(sub);

  const grid = document.createElement("div");
  grid.className = "landing-tile-grid";

  const tiles: { lesson: string; title: string; caption: string; slug: string; build: (h: HTMLElement) => void }[] = [
    { lesson: "Lesson 1",  title: "Multi-Armed Bandits", caption: "ε-greedy picks an arm; values converge to truth.", slug: "bandits", build: banditsTile },
    { lesson: "Prereq A",  title: "Markov Chains",        caption: "A particle walks a chain — visits trace the stationary distribution.", slug: "markov-chains", build: markovTile },
    { lesson: "Lesson 2",  title: "MDPs",                 caption: "Heatmap of V*; optimal policy arrows; ghost agent traces the greedy path.", slug: "mdps", build: mdpTile },
    { lesson: "Prereq C",  title: "Contractions",         caption: "‖V_k − V*‖ decays geometrically — Banach's theorem at work.", slug: "contractions", build: contractionsTile },
    { lesson: "Lesson 3",  title: "Dynamic Programming",  caption: "Value iteration sweeps outward from the goal.", slug: "dynamic-programming", build: dpTile },
    { lesson: "Lesson 6",  title: "Importance Sampling",  caption: "Behavior µ vs target π — samples reweighted by π/µ.", slug: "importance-sampling", build: isTile },
    { lesson: "Lesson 7",  title: "Monte Carlo",          caption: "Noisy returns; the running mean settles on the true value.", slug: "monte-carlo", build: mcTile },
    { lesson: "Lesson 8",  title: "TD Learning",          caption: "TD error propagates backward through the chain.", slug: "td-learning", build: tdTile },
    { lesson: "Lesson 9",  title: "Function Approx · DQN", caption: "Neurons fire; Q-values pulse; argmax wins.", slug: "function-approximation", build: dqnTile },
    { lesson: "Lesson 10", title: "Policy Gradient",      caption: "Action distribution sharpens onto the best arm.", slug: "policy-gradient", build: pgTile },
    { lesson: "Lesson 11", title: "TRPO / PPO",           caption: "Clipped surrogate — the (1−ε, 1+ε) zone keeps updates safe.", slug: "trpo-ppo", build: ppoTile },
    { lesson: "Lesson 12", title: "Max-Entropy RL",       caption: "Boltzmann policy; temperature α sweeps cold → hot.", slug: "max-ent-rl", build: maxEntTile },
  ];

  for (const tt of tiles) grid.appendChild(makeTile(tt));
  wrap.appendChild(grid);
  return wrap;
}
