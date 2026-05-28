/**
 * V4 — The Clipped Surrogate Curve.
 * Plots the PPO clipped surrogate as a function of μ_θ for a 1D Gaussian policy.
 * Three curves: unclipped r·A (amber), clip(r)·A (cyan), min/PPO (violet).
 * 10 discrete x-axis points. Gradient inset shows slope of violet (min) curve.
 */
import * as d3 from "d3";

const N_POINTS = 10;

interface CurvePoint {
  mu: number;
  ratio: number;
  unclipped: number;
  clipped: number;
  ppo: number;
  grad: number;
}

function computeCurve(
  muOld: number,
  sigma: number,
  aSample: number,
  Ahat: number,
  clipEps: number,
): CurvePoint[] {
  const muMin = muOld - 3 * sigma;
  const muMax = muOld + 3 * sigma;
  const muValues = Array.from({ length: N_POINTS }, (_, i) => muMin + (i / (N_POINTS - 1)) * (muMax - muMin));

  // Gaussian log-prob: log N(a; μ, σ²) = -0.5 * ((a-μ)/σ)² - log(σ√(2π))
  const logProbOld = -0.5 * ((aSample - muOld) / sigma) ** 2;

  const points = muValues.map((mu) => {
    const logProbNew = -0.5 * ((aSample - mu) / sigma) ** 2;
    const ratio = Math.exp(logProbNew - logProbOld);
    const rClip = Math.max(1 - clipEps, Math.min(1 + clipEps, ratio));
    const unclipped = ratio * Ahat;
    const clipped = rClip * Ahat;
    const ppo = Math.min(unclipped, clipped);
    return { mu, ratio, unclipped, clipped, ppo, grad: 0 };
  });

  // Numerical gradient of the ppo (min) curve.
  for (let i = 0; i < N_POINTS; i++) {
    const prev = points[Math.max(0, i - 1)];
    const next = points[Math.min(N_POINTS - 1, i + 1)];
    const dmu = next.mu - prev.mu;
    points[i].grad = dmu > 0 ? (next.ppo - prev.ppo) / dmu : 0;
  }

  return points;
}

customElements.define(
  "ppo-clipped-surrogate-curve",
  class extends HTMLElement {
    connectedCallback() { this.render(); }

    private render() {
      this.innerHTML = "";
      const W = 540, H = 320;

      const wrapper = document.createElement("div");
      wrapper.style.cssText = `max-width:${W + 32}px;margin:24px auto;background:var(--rl-surface);border:1px solid var(--rl-border);border-radius:8px;padding:16px;`;

      const title = document.createElement("p");
      title.style.cssText = "font-weight:700;font-size:14px;margin:0 0 4px;";
      title.textContent = "V4 — The Clipped Surrogate Curve";
      wrapper.appendChild(title);

      const desc = document.createElement("p");
      desc.style.cssText = "font-size:12px;color:var(--rl-ink-muted);margin:0 0 10px;";
      desc.textContent = "1D Gaussian policy π_θ(a|s) = N(a; μ_θ, σ²). Single sampled action. x-axis: μ_θ. The inset shows slope of the violet (min) curve — exactly 0 on the plateau.";
      wrapper.appendChild(desc);

      const controls = document.createElement("div");
      controls.style.cssText = "display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:10px;";

      const makeCtrl = (label: string, min: number, max: number, step: number, val: number) => {
        const wrap = document.createElement("label");
        wrap.style.cssText = "font-size:12px;display:flex;gap:5px;align-items:center;";
        wrap.appendChild(Object.assign(document.createElement("span"), { textContent: label }));
        const inp = document.createElement("input");
        inp.type = "range"; inp.min = String(min); inp.max = String(max);
        inp.step = String(step); inp.value = String(val);
        inp.style.cssText = "width:80px;";
        const valSpan = document.createElement("span");
        valSpan.style.cssText = "font-variant-numeric:tabular-nums;min-width:36px;";
        valSpan.textContent = String(val);
        inp.addEventListener("input", () => { valSpan.textContent = inp.value; this.redraw(mainSvg, insetSvg, controls2); });
        wrap.appendChild(inp); wrap.appendChild(valSpan);
        controls.appendChild(wrap);
        return inp;
      };

      const AhatCtrl = makeCtrl("Â:", -2, 2, 0.1, 1.0);
      const epsCtrl = makeCtrl("ε:", 0.05, 0.5, 0.05, 0.2);
      const sigmaCtrl = makeCtrl("σ:", 0.2, 2.0, 0.1, 0.8);
      wrapper.appendChild(controls);

      const controls2 = { Ahat: AhatCtrl, eps: epsCtrl, sigma: sigmaCtrl };

      const mainSvg = d3.create("svg").attr("viewBox", `0 0 ${W} ${H - 80}`)
        .style("width", "100%").style("height", "auto");
      wrapper.appendChild(mainSvg.node()!);

      const insetTitle = document.createElement("p");
      insetTitle.style.cssText = "font-size:11px;font-weight:600;color:var(--rl-ink-muted);margin:8px 0 2px;";
      insetTitle.textContent = "Gradient of min(r·Â, clip·Â) w.r.t. μ_θ";
      wrapper.appendChild(insetTitle);

      const insetSvg = d3.create("svg").attr("viewBox", `0 0 ${W} 80`)
        .style("width", "100%").style("height", "auto");
      wrapper.appendChild(insetSvg.node()!);

      // Legend.
      const legend = document.createElement("div");
      legend.style.cssText = "display:flex;gap:14px;flex-wrap:wrap;margin-top:6px;font-size:11px;";
      [
        { color: "#b45309", dash: "4,3", label: "r·Â (unclipped IS surrogate)" },
        { color: "#0e7490", dash: "", label: "clip(r)·Â (clipped term)" },
        { color: "#7c3aed", dash: "", label: "min(·) — PPO objective" },
      ].forEach(({ color, dash, label }) => {
        const item = document.createElement("div");
        item.style.cssText = "display:flex;gap:6px;align-items:center;";
        const svg2 = d3.create("svg").attr("width", 24).attr("height", 8);
        svg2.append("line").attr("x1", 0).attr("y1", 4).attr("x2", 24).attr("y2", 4)
          .attr("stroke", color).attr("stroke-width", 2).attr("stroke-dasharray", dash || "none");
        item.appendChild(svg2.node()!);
        item.appendChild(Object.assign(document.createElement("span"), {
          textContent: label,
          style: "color:var(--rl-ink-muted);",
        }));
        legend.appendChild(item);
      });
      wrapper.appendChild(legend);
      this.appendChild(wrapper);

      this.redraw(mainSvg, insetSvg, controls2);
    }

    private redraw(
      mainSvg: d3.Selection<SVGSVGElement, undefined, null, undefined>,
      insetSvg: d3.Selection<SVGSVGElement, undefined, null, undefined>,
      ctrls: { Ahat: HTMLInputElement; eps: HTMLInputElement; sigma: HTMLInputElement },
    ) {
      const Ahat = parseFloat(ctrls.Ahat.value);
      const clipEps = parseFloat(ctrls.eps.value);
      const sigma = parseFloat(ctrls.sigma.value);
      const muOld = 0; // behavior policy mean
      const aSample = sigma * 0.5; // fixed sampled action

      const pts = computeCurve(muOld, sigma, aSample, Ahat, clipEps);
      const W = 540, H = 240;
      const pad = { l: 52, r: 20, t: 16, b: 36 };

      mainSvg.selectAll("*").remove();
      insetSvg.selectAll("*").remove();

      const allY = pts.flatMap((p) => [p.unclipped, p.clipped, p.ppo]);
      const yMin = Math.min(...allY), yMax = Math.max(...allY);
      const yPad = (yMax - yMin) * 0.15 || 0.1;

      const xScale = d3.scaleLinear()
        .domain([pts[0].mu, pts[N_POINTS - 1].mu])
        .range([pad.l, W - pad.r]);
      const yScale = d3.scaleLinear()
        .domain([yMin - yPad, yMax + yPad])
        .range([H - pad.b, pad.t]);

      // Clip region highlight.
      const xL = xScale(muOld - Math.log(1 + clipEps) * sigma);
      const xR = xScale(muOld + Math.log(1 + clipEps) * sigma);
      mainSvg.append("rect")
        .attr("x", xL).attr("y", pad.t)
        .attr("width", Math.max(0, xR - xL))
        .attr("height", H - pad.t - pad.b)
        .attr("fill", "rgba(251,191,36,0.15)");

      // Clip region label.
      mainSvg.append("text").attr("x", (xL + xR) / 2).attr("y", pad.t + 10)
        .attr("font-size", 9).attr("fill", "#b45309").attr("text-anchor", "middle")
        .text(`clip region [1-ε, 1+ε]`);

      // Zero line.
      mainSvg.append("line").attr("x1", pad.l).attr("y1", yScale(0)).attr("x2", W - pad.r).attr("y2", yScale(0))
        .attr("stroke", "#ccc").attr("stroke-width", 1).attr("stroke-dasharray", "3,2");

      // Axes.
      mainSvg.append("g").attr("transform", `translate(0,${H - pad.b})`).call(d3.axisBottom(xScale).ticks(6));
      mainSvg.append("g").attr("transform", `translate(${pad.l},0)`).call(d3.axisLeft(yScale).ticks(5));
      mainSvg.append("text").attr("x", W / 2).attr("y", H).attr("text-anchor", "middle")
        .attr("font-size", 10).attr("fill", "var(--rl-ink-muted)").text("μ_θ (policy mean)");
      mainSvg.append("text").attr("transform", "rotate(-90)").attr("x", -(H / 2)).attr("y", 14)
        .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", "var(--rl-ink-muted)").text("Surrogate value");

      const line = (key: keyof CurvePoint, color: string, dash?: string) => {
        const path = d3.line<CurvePoint>().x((d) => xScale(d.mu)).y((d) => yScale(d[key] as number));
        mainSvg.append("path").datum(pts).attr("fill", "none")
          .attr("stroke", color).attr("stroke-width", 2.5)
          .attr("stroke-dasharray", dash || "none").attr("d", path);

        // Dots.
        pts.forEach((p) => {
          mainSvg.append("circle").attr("cx", xScale(p.mu)).attr("cy", yScale(p[key] as number)).attr("r", 3.5)
            .attr("fill", color).attr("stroke", "#fff").attr("stroke-width", 1);
        });
      };

      line("unclipped", "#b45309", "5,3");
      line("clipped", "#0e7490");
      line("ppo", "#7c3aed");

      // Inset: gradient of ppo curve.
      const IH = 80, Ipad = { l: 52, r: 20, t: 8, b: 28 };
      const gradVals = pts.map((p) => p.grad);
      const gExt = d3.extent(gradVals) as [number, number];
      const gPad = (gExt[1] - gExt[0]) * 0.2 || 0.1;
      const ixScale = d3.scaleLinear().domain([pts[0].mu, pts[N_POINTS - 1].mu]).range([Ipad.l, W - Ipad.r]);
      const iyScale = d3.scaleLinear().domain([gExt[0] - gPad, gExt[1] + gPad]).range([IH - Ipad.b, Ipad.t]);

      insetSvg.append("line").attr("x1", Ipad.l).attr("y1", iyScale(0)).attr("x2", W - Ipad.r).attr("y2", iyScale(0))
        .attr("stroke", "#ccc").attr("stroke-width", 1);
      insetSvg.append("g").attr("transform", `translate(0,${IH - Ipad.b})`).call(d3.axisBottom(ixScale).ticks(6));
      insetSvg.append("g").attr("transform", `translate(${Ipad.l},0)`).call(d3.axisLeft(iyScale).ticks(3));

      const gradPath = d3.line<CurvePoint>().x((d) => ixScale(d.mu)).y((d) => iyScale(d.grad));
      insetSvg.append("path").datum(pts).attr("fill", "none")
        .attr("stroke", "#7c3aed").attr("stroke-width", 2).attr("d", gradPath);
      pts.forEach((p) => {
        insetSvg.append("circle").attr("cx", ixScale(p.mu)).attr("cy", iyScale(p.grad)).attr("r", 3)
          .attr("fill", "#7c3aed").attr("stroke", "#fff").attr("stroke-width", 1);
      });
    }
  },
);
