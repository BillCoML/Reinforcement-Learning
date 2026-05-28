/**
 * V2 — KL Geometry of the Softmax Policy.
 * 2D projection of the softmax parameter space at state 0.
 * Shows: L2 ball (circle), KL ellipse, natural gradient direction.
 * The Fisher for a 4-action softmax at a given logit is 4×4 rank-3;
 * we add a small diagonal regularizer 1e-4·I for numerical stability.
 */
import * as d3 from "d3";

/** Numerically stable softmax of a 4-vector. */
function sm4(t: [number, number, number, number]): [number, number, number, number] {
  const m = Math.max(...t);
  const e = t.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s) as [number, number, number, number];
}

/** Fisher information matrix for a 4-action softmax (4×4 matrix, row-major). */
function fisher4(p: [number, number, number, number]): number[][] {
  const F: number[][] = Array.from({ length: 4 }, () => new Array(4).fill(0));
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      // F_{ij} = sum_a p(a) score_i(a) score_j(a)
      // score_i(a) = 1[a=i] - p(i)
      F[i][j] = -p[i] * p[j];
      if (i === j) F[i][j] += p[i];
    }
  }
  // Add diagonal regularizer for numerical stability (rank-3 Fisher).
  const REG = 1e-4;
  for (let i = 0; i < 4; i++) F[i][i] += REG;
  return F;
}

/** Policy gradient at θ_old: approximate gradient of J using uniform policy estimate. */
function approxGradient(p: [number, number, number, number]): number[] {
  // Heuristic gradient for visualization: favor action 1 (Right) and 2 (Down) at state 0.
  return [p[1], 1 - p[1], p[2], 1 - p[2]];
}

customElements.define(
  "ppo-kl-geometry-plot",
  class extends HTMLElement {
    connectedCallback() { this.render(); }

    private render() {
      this.innerHTML = "";
      const W = 460, H = 320;
      const wrapper = document.createElement("div");
      wrapper.style.cssText = `max-width:${W + 32}px;margin:24px auto;background:var(--rl-surface);border:1px solid var(--rl-border);border-radius:8px;padding:16px;`;

      const title = document.createElement("p");
      title.style.cssText = "font-weight:700;font-size:14px;margin:0 0 8px;";
      title.textContent = "V2 — KL Geometry of the Softmax Policy";
      wrapper.appendChild(title);

      const desc = document.createElement("p");
      desc.style.cssText = "font-size:12px;color:var(--rl-ink-muted);margin:0 0 10px;";
      desc.textContent = "2D slice of the parameter space at state 0 (logit₀ vs logit₁, others fixed at 0). Circle = L2 distance; ellipse = KL contour; arrow = natural gradient direction.";
      wrapper.appendChild(desc);

      const controls = document.createElement("div");
      controls.style.cssText = "display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:8px;";

      const makeControl = (label: string, _id: string, min: number, max: number, step: number, defaultV: number) => {
        const wrap = document.createElement("label");
        wrap.style.cssText = "font-size:12px;display:flex;gap:6px;align-items:center;";
        const span = document.createElement("span");
        span.textContent = label;
        const input = document.createElement("input");
        input.type = "range";
        input.min = String(min);
        input.max = String(max);
        input.step = String(step);
        input.value = String(defaultV);
        input.style.cssText = "width:80px;";
        const val = document.createElement("span");
        val.style.cssText = "font-variant-numeric:tabular-nums;min-width:30px;";
        val.textContent = String(defaultV);
        input.addEventListener("input", () => { val.textContent = input.value; draw(); });
        wrap.appendChild(span);
        wrap.appendChild(input);
        wrap.appendChild(val);
        controls.appendChild(wrap);
        return input;
      };

      const t0Input = makeControl("θ₀:", "t0", -3, 3, 0.1, 0.5);
      const t1Input = makeControl("θ₁:", "t1", -3, 3, 0.1, 0.0);
      const deltaInput = makeControl("δ (radius):", "delta", 0.05, 0.5, 0.01, 0.1);
      wrapper.appendChild(controls);

      const svg = d3.create("svg").attr("viewBox", `0 0 ${W} ${H}`)
        .style("width", "100%").style("height", "auto");
      wrapper.appendChild(svg.node()!);

      const legend = document.createElement("div");
      legend.style.cssText = "display:flex;gap:16px;flex-wrap:wrap;margin-top:6px;font-size:11px;";
      [
        { color: "#0e7490", dash: "", label: "L2 ball (parameter space)" },
        { color: "#7c3aed", dash: "4,3", label: "KL contour (policy space)" },
        { color: "#b45309", dash: "", label: "Gradient ∇J" },
        { color: "#15803d", dash: "", label: "Natural gradient F⁻¹∇J" },
      ].forEach(({ color, dash, label }) => {
        const item = document.createElement("div");
        item.style.cssText = "display:flex;gap:6px;align-items:center;";
        const svg2 = d3.create("svg").attr("width", 28).attr("height", 8);
        svg2.append("line").attr("x1", 0).attr("y1", 4).attr("x2", 28).attr("y2", 4)
          .attr("stroke", color).attr("stroke-width", 2).attr("stroke-dasharray", dash || "none");
        item.appendChild(svg2.node()!);
        const lbl = document.createElement("span");
        lbl.style.cssText = "color:var(--rl-ink-muted);";
        lbl.textContent = label;
        item.appendChild(lbl);
        legend.appendChild(item);
      });
      wrapper.appendChild(legend);
      this.appendChild(wrapper);

      const draw = () => {
        const t0 = parseFloat(t0Input.value);
        const t1 = parseFloat(t1Input.value);
        const delta = parseFloat(deltaInput.value);
        this.drawGeometry(svg, t0, t1, delta, W, H);
      };
      draw();
    }

    private drawGeometry(
      svg: d3.Selection<SVGSVGElement, undefined, null, undefined>,
      t0: number,
      t1: number,
      delta: number,
      W: number,
      H: number,
    ) {
      svg.selectAll("*").remove();
      const pad = 40;
      const cx = W / 2, cy = H / 2;
      const scale = Math.min((W - 2 * pad), (H - 2 * pad)) / 2;

      const theta: [number, number, number, number] = [t0, t1, 0, 0];
      const p = sm4(theta);
      const F = fisher4(p);

      // KL ellipse in the (θ₀, θ₁) 2D subspace (other dims fixed at 0).
      // F is 4×4; we project to 2D by taking F's 2×2 top-left submatrix.
      const F2 = [[F[0][0], F[0][1]], [F[1][0], F[1][1]]];
      // Cholesky of 2×2 matrix.
      const l00 = Math.sqrt(Math.max(F2[0][0], 1e-12));
      const l10 = F2[1][0] / l00;
      const l11 = Math.sqrt(Math.max(F2[1][1] - l10 * l10, 1e-12));
      // Inverse of L: L^{-1} = [[1/l00, 0], [-l10/(l00*l11), 1/l11]]
      const linv00 = 1 / l00;
      const linv10 = -l10 / (l00 * l11);
      const linv11 = 1 / l11;
      const r = Math.sqrt(2 * delta);

      const ellipsePts: [number, number][] = [];
      for (let i = 0; i <= 64; i++) {
        const angle = (i / 64) * 2 * Math.PI;
        const u = Math.cos(angle), v = Math.sin(angle);
        // Map unit circle through L^{-T}: rotate by L^{-1}, scale by r.
        const x = r * (linv00 * u);
        const y = r * (linv10 * u + linv11 * v);
        ellipsePts.push([cx + x * scale, cy - y * scale]);
      }

      // L2 circle.
      const l2r = Math.sqrt(delta) * scale * 0.8;

      // Gradient and natural gradient.
      const grad2 = approxGradient(p).slice(0, 2) as [number, number];
      const gradNorm = Math.sqrt(grad2[0] ** 2 + grad2[1] ** 2) || 1;
      const gradDir: [number, number] = [grad2[0] / gradNorm, grad2[1] / gradNorm];

      // Natural gradient in 2D: F2^{-1} grad2.
      const detF2 = F2[0][0] * F2[1][1] - F2[0][1] * F2[1][0];
      const natGrad: [number, number] = [
        (F2[1][1] * grad2[0] - F2[0][1] * grad2[1]) / (detF2 || 1),
        (-F2[1][0] * grad2[0] + F2[0][0] * grad2[1]) / (detF2 || 1),
      ];
      const natNorm = Math.sqrt(natGrad[0] ** 2 + natGrad[1] ** 2) || 1;
      const natDir: [number, number] = [natGrad[0] / natNorm, natGrad[1] / natNorm];

      const arrowLen = scale * 0.45;

      // Grid lines.
      svg.append("line").attr("x1", pad).attr("y1", cy).attr("x2", W - pad).attr("y2", cy)
        .attr("stroke", "#ddd").attr("stroke-width", 1);
      svg.append("line").attr("x1", cx).attr("y1", pad).attr("x2", cx).attr("y2", H - pad)
        .attr("stroke", "#ddd").attr("stroke-width", 1);

      // L2 circle.
      svg.append("circle").attr("cx", cx).attr("cy", cy).attr("r", l2r)
        .attr("fill", "none").attr("stroke", "#0e7490").attr("stroke-width", 2);

      // KL ellipse.
      svg.append("polygon")
        .attr("points", ellipsePts.map((p) => p.join(",")).join(" "))
        .attr("fill", "rgba(124,58,237,0.06)")
        .attr("stroke", "#7c3aed")
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "5,3");

      // θ_old point.
      svg.append("circle").attr("cx", cx).attr("cy", cy).attr("r", 5)
        .attr("fill", "#1c1e22").attr("stroke", "#fff").attr("stroke-width", 1.5);
      svg.append("text").attr("x", cx + 7).attr("y", cy - 6)
        .attr("font-size", 11).attr("fill", "#1c1e22").text("θ_old");

      // Gradient arrow.
      const arrow = (x1: number, y1: number, dx: number, dy: number, color: string) => {
        const x2 = x1 + dx, y2 = y1 + dy;
        svg.append("line").attr("x1", x1).attr("y1", y1).attr("x2", x2).attr("y2", y2)
          .attr("stroke", color).attr("stroke-width", 2)
          .attr("marker-end", `url(#arrow-${color.replace(/[#,()]/g, "")})`);
      };

      // Define arrowhead markers.
      const defs = svg.append("defs");
      [["b45309", "#b45309"], ["15803d", "#15803d"]].forEach(([id, color]) => {
        const m = defs.append("marker").attr("id", `arrow-${id}`)
          .attr("viewBox", "0 0 6 6").attr("refX", 5).attr("refY", 3)
          .attr("markerWidth", 5).attr("markerHeight", 5).attr("orient", "auto");
        m.append("path").attr("d", "M0,0 L6,3 L0,6 Z").attr("fill", color);
      });

      arrow(cx, cy, gradDir[0] * arrowLen, -gradDir[1] * arrowLen, "#b45309");
      arrow(cx, cy, natDir[0] * arrowLen, -natDir[1] * arrowLen, "#15803d");

      // Labels.
      svg.append("text").attr("x", cx + gradDir[0] * arrowLen + 6).attr("y", cy - gradDir[1] * arrowLen)
        .attr("font-size", 10).attr("fill", "#b45309").text("∇J");
      svg.append("text").attr("x", cx + natDir[0] * arrowLen + 6).attr("y", cy - natDir[1] * arrowLen)
        .attr("font-size", 10).attr("fill", "#15803d").text("F⁻¹∇J");

      // Axis labels.
      svg.append("text").attr("x", W - pad - 4).attr("y", cy + 14)
        .attr("font-size", 10).attr("fill", "var(--rl-ink-muted)").attr("text-anchor", "end").text("θ₀");
      svg.append("text").attr("x", cx + 4).attr("y", pad + 10)
        .attr("font-size", 10).attr("fill", "var(--rl-ink-muted)").text("θ₁");
    }
  },
);
