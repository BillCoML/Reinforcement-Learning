/**
 * V1 — Parameter Space vs. Policy Space.
 * Two-panel side-by-side. Left: 2D PCA projection of the θ trajectory.
 * Right: action-probability space at state 0 (π(Up|s₀) vs π(Right|s₀)).
 */
import * as d3 from "d3";
import { buildGridworld } from "../mdp/gridworld";
import { mulberry32 } from "../td/helpers";
import { ppoUpdate, collectBatch } from "../ppo/ppo";

const mdp = buildGridworld({ slippery: false, gamma: 0.9 });

function pca2d(points: Float64Array[]): [number, number][] {
  if (points.length < 2) return points.map(() => [0, 0] as [number, number]);
  const D = points[0].length;
  const N = points.length;

  const mean = new Float64Array(D);
  for (const p of points) for (let d = 0; d < D; d++) mean[d] += p[d] / N;
  const centered = points.map((p) => {
    const c = new Float64Array(D);
    for (let d = 0; d < D; d++) c[d] = p[d] - mean[d];
    return c;
  });

  function powerIter(data: Float64Array[], deflate?: Float64Array): Float64Array {
    let v = new Float64Array(D);
    v[0] = 1;
    for (let iter = 0; iter < 60; iter++) {
      const Av = new Float64Array(D);
      for (const x of data) {
        let dot = 0;
        for (let d = 0; d < D; d++) dot += x[d] * v[d];
        for (let d = 0; d < D; d++) Av[d] += dot * x[d];
      }
      if (deflate) {
        let proj = 0;
        for (let d = 0; d < D; d++) proj += Av[d] * deflate[d];
        for (let d = 0; d < D; d++) Av[d] -= proj * deflate[d];
      }
      let norm = 0;
      for (let d = 0; d < D; d++) norm += Av[d] * Av[d];
      norm = Math.sqrt(norm);
      if (norm < 1e-12) break;
      for (let d = 0; d < D; d++) v[d] = Av[d] / norm;
    }
    return v;
  }

  const pc1 = powerIter(centered);
  const pc2 = powerIter(centered, pc1);

  return centered.map((c) => {
    let x = 0, y = 0;
    for (let d = 0; d < D; d++) { x += c[d] * pc1[d]; y += c[d] * pc2[d]; }
    return [x, y] as [number, number];
  });
}

function runAndCapture(lr: number): { paramPts: [number, number][]; policyPts: [number, number][] } {
  const config = {
    lrPolicy: lr,
    lrValue: Math.min(lr * 0.5, 0.5),
    clipEps: 0.2,
    gaeLambda: 0.95,
    epochs: 4,
    batchEpisodes: 10,
    entropyCoef: 0,
    valueCoef: 0.5,
    normalizeAdvantages: true,
  };

  const rng = mulberry32(42);
  const thetaSnapshots: Float64Array[] = [];
  let state = { theta: new Float64Array(mdp.nS * mdp.nA), V: new Float64Array(mdp.nS) };
  thetaSnapshots.push(state.theta.slice() as Float64Array);

  for (let i = 0; i < 50; i++) {
    const batch = collectBatch(state.theta, mdp, config.batchEpisodes, rng);
    const { state: newState } = ppoUpdate(state, batch, config, mdp);
    state = newState as typeof state;
    thetaSnapshots.push(state.theta.slice() as Float64Array);
  }

  const paramPts = pca2d(thetaSnapshots);

  const policyPts: [number, number][] = thetaSnapshots.map((theta) => {
    const base = 0;
    let maxV = -Infinity;
    for (let a = 0; a < mdp.nA; a++) if (theta[base + a] > maxV) maxV = theta[base + a];
    let sum = 0;
    const p = new Float64Array(mdp.nA);
    for (let a = 0; a < mdp.nA; a++) { p[a] = Math.exp(theta[base + a] - maxV); sum += p[a]; }
    for (let a = 0; a < mdp.nA; a++) p[a] /= sum;
    return [p[0], p[1]] as [number, number];
  });

  return { paramPts, policyPts };
}

function drawTrajectory(
  container: HTMLElement,
  pts: [number, number][],
  title: string,
  xlabel: string,
  ylabel: string,
): void {
  container.innerHTML = "";
  const W = 260, H = 200, pad = 36;

  const heading = document.createElement("p");
  heading.style.cssText = "font-size:11px;font-weight:600;text-align:center;margin:0 0 4px;color:var(--rl-ink-muted);";
  heading.textContent = title;
  container.appendChild(heading);

  const svg = d3.create("svg").attr("viewBox", `0 0 ${W} ${H}`)
    .style("width", "100%").style("height", "auto").style("max-width", `${W}px`);

  if (pts.length < 2) { container.appendChild(svg.node()!); return; }

  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const xExt = d3.extent(xs) as [number, number];
  const yExt = d3.extent(ys) as [number, number];
  const xPad = (xExt[1] - xExt[0]) * 0.12 || 0.05;
  const yPad = (yExt[1] - yExt[0]) * 0.12 || 0.05;

  const xScale = d3.scaleLinear().domain([xExt[0] - xPad, xExt[1] + xPad]).range([pad, W - 10]);
  const yScale = d3.scaleLinear().domain([yExt[0] - yPad, yExt[1] + yPad]).range([H - pad, 8]);

  svg.append("g").attr("transform", `translate(0,${H - pad})`).call(d3.axisBottom(xScale).ticks(4).tickSize(3));
  svg.append("g").attr("transform", `translate(${pad},0)`).call(d3.axisLeft(yScale).ticks(4).tickSize(3));

  svg.append("text").attr("x", W / 2).attr("y", H - 4).attr("text-anchor", "middle")
    .attr("font-size", 9).attr("fill", "var(--rl-ink-muted)").text(xlabel);
  svg.append("text").attr("transform", `rotate(-90)`).attr("x", -H / 2).attr("y", 11)
    .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", "var(--rl-ink-muted)").text(ylabel);

  const colorScale = d3.scaleSequential(d3.interpolatePlasma).domain([0, pts.length - 1]);

  for (let i = 1; i < pts.length; i++) {
    svg.append("line")
      .attr("x1", xScale(pts[i - 1][0])).attr("y1", yScale(pts[i - 1][1]))
      .attr("x2", xScale(pts[i][0])).attr("y2", yScale(pts[i][1]))
      .attr("stroke", colorScale(i)).attr("stroke-width", 1.5);
  }

  svg.append("circle").attr("cx", xScale(pts[0][0])).attr("cy", yScale(pts[0][1])).attr("r", 4)
    .attr("fill", "#b45309").attr("stroke", "#fff").attr("stroke-width", 1);
  svg.append("circle").attr("cx", xScale(pts[pts.length - 1][0])).attr("cy", yScale(pts[pts.length - 1][1])).attr("r", 4)
    .attr("fill", "#15803d").attr("stroke", "#fff").attr("stroke-width", 1);

  container.appendChild(svg.node()!);
}

customElements.define(
  "ppo-param-vs-policy-space",
  class extends HTMLElement {
    connectedCallback() { this.render(); }

    private render() {
      this.innerHTML = "";
      const W = 580;
      const wrapper = document.createElement("div");
      wrapper.style.cssText = `max-width:${W}px;margin:24px auto;background:var(--rl-surface);border:1px solid var(--rl-border);border-radius:8px;padding:16px;`;

      const title = document.createElement("p");
      title.style.cssText = "font-weight:700;font-size:14px;margin:0 0 8px;";
      title.textContent = "V1 — Parameter Space vs. Policy Space";
      wrapper.appendChild(title);

      const controls = document.createElement("div");
      controls.style.cssText = "display:flex;gap:12px;align-items:center;margin-bottom:12px;";
      const lbl = document.createElement("label");
      lbl.style.cssText = "font-size:13px;";
      lbl.innerHTML = "Learning rate: ";
      const sel = document.createElement("select");
      sel.style.cssText = "font-size:13px;padding:2px 6px;";
      [0.1, 0.5, 2.0, 5.0].forEach((lr) => {
        const opt = document.createElement("option");
        opt.value = String(lr);
        opt.textContent = `lr = ${lr}`;
        if (lr === 0.5) opt.selected = true;
        sel.appendChild(opt);
      });
      lbl.appendChild(sel);
      controls.appendChild(lbl);
      wrapper.appendChild(controls);

      const row = document.createElement("div");
      row.style.cssText = "display:flex;gap:16px;flex-wrap:wrap;";
      const leftDiv = document.createElement("div");
      leftDiv.style.cssText = "flex:1;min-width:220px;";
      const rightDiv = document.createElement("div");
      rightDiv.style.cssText = "flex:1;min-width:220px;";
      row.appendChild(leftDiv);
      row.appendChild(rightDiv);
      wrapper.appendChild(row);

      const note = document.createElement("p");
      note.style.cssText = "font-size:11px;color:var(--rl-ink-muted);margin:8px 0 0;";
      note.textContent = "In deep continuous RL, the policy panel would have no such simplex bound — probabilities can leave [0,1].";
      wrapper.appendChild(note);

      this.appendChild(wrapper);

      const update = (lr: number) => {
        const { paramPts, policyPts } = runAndCapture(lr);
        drawTrajectory(leftDiv, paramPts, "Parameter Space (θ — PCA)", "PC₁", "PC₂");
        drawTrajectory(rightDiv, policyPts, "Policy Space (π(Up|s₀) vs π(Right|s₀))", "π(Up)", "π(Right)");
      };

      sel.addEventListener("change", () => update(parseFloat(sel.value)));
      update(0.5);
    }
  },
);
