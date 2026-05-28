/**
 * V5 — GAE λ Sweep.
 * Loads gae_lambda.json. Scatter plot of final V(start) vs λ with error bars.
 * Pink interpolator curve through the means.
 * Shows monotone trend: λ=0 lowest variance, λ=1 highest.
 */
import * as d3 from "d3";

interface LambdaResult {
  lambda: number;
  mean: number;
  std: number;
}

customElements.define(
  "ppo-gae-lambda-sweep",
  class extends HTMLElement {
    connectedCallback() { this.load(); }

    private async load() {
      this.innerHTML = `<p style="color:var(--rl-ink-muted);font-size:13px;">Loading GAE λ sweep data…</p>`;
      try {
        const resp = await fetch("/data/ppo/gae_lambda.json");
        const data = await resp.json();
        this.render(data);
      } catch (e) {
        this.innerHTML = `<p style="color:var(--ppo-diverged);">Failed to load gae_lambda.json. Run <code>python3 scripts/ppo_traces.py</code> first.</p>`;
      }
    }

    private render(data: { results: Record<string, { lambda: number; mean: number; std: number }> }) {
      this.innerHTML = "";
      const results: LambdaResult[] = Object.values(data.results)
        .map((r) => ({ lambda: r.lambda, mean: r.mean, std: r.std }))
        .sort((a, b) => a.lambda - b.lambda);

      const W = 500, H = 320;
      const pad = { l: 60, r: 20, t: 24, b: 48 };

      const wrapper = document.createElement("div");
      wrapper.style.cssText = `max-width:${W + 32}px;margin:24px auto;background:var(--rl-surface);border:1px solid var(--rl-border);border-radius:8px;padding:16px;`;

      const titleEl = document.createElement("p");
      titleEl.style.cssText = "font-weight:700;font-size:14px;margin:0 0 4px;";
      titleEl.textContent = "V5 — GAE λ Sweep (batch=5, 20 seeds)";
      wrapper.appendChild(titleEl);

      const desc = document.createElement("p");
      desc.style.cssText = "font-size:12px;color:var(--rl-ink-muted);margin:0 0 10px;";
      desc.textContent = "Final V(s₀) vs GAE λ. Error bars = 1 std over 20 seeds. On this small MDP, lower λ wins (critic is accurate; bootstrap bias is negligible).";
      wrapper.appendChild(desc);

      const svg = d3.create("svg").attr("viewBox", `0 0 ${W} ${H}`)
        .style("width", "100%").style("height", "auto");

      const yVals = results.flatMap((r) => [r.mean - r.std, r.mean + r.std]);
      const yMin = Math.min(...yVals);
      const yMax = Math.max(...yVals);
      const yPad = (yMax - yMin) * 0.15;

      const xScale = d3.scaleLinear().domain([-0.05, 1.05]).range([pad.l, W - pad.r]);
      const yScale = d3.scaleLinear().domain([yMin - yPad, yMax + yPad]).range([H - pad.b, pad.t]);

      svg.append("g").attr("transform", `translate(0,${H - pad.b})`).call(d3.axisBottom(xScale).ticks(5).tickValues([0, 0.5, 0.9, 0.95, 1.0]));
      svg.append("g").attr("transform", `translate(${pad.l},0)`).call(d3.axisLeft(yScale).ticks(5));

      svg.append("text").attr("x", W / 2).attr("y", H - 6).attr("text-anchor", "middle")
        .attr("font-size", 11).attr("fill", "var(--rl-ink-muted)").text("GAE λ");
      svg.append("text").attr("transform", "rotate(-90)").attr("x", -(H / 2)).attr("y", 16)
        .attr("text-anchor", "middle").attr("font-size", 11).attr("fill", "var(--rl-ink-muted)").text("Final V(s₀)");

      // Error bars.
      results.forEach((r) => {
        svg.append("line")
          .attr("x1", xScale(r.lambda)).attr("y1", yScale(r.mean - r.std))
          .attr("x2", xScale(r.lambda)).attr("y2", yScale(r.mean + r.std))
          .attr("stroke", "#db2777").attr("stroke-width", 1.5);
        [-1, 1].forEach((sign) => {
          svg.append("line")
            .attr("x1", xScale(r.lambda) - 5).attr("y1", yScale(r.mean + sign * r.std))
            .attr("x2", xScale(r.lambda) + 5).attr("y2", yScale(r.mean + sign * r.std))
            .attr("stroke", "#db2777").attr("stroke-width", 1.5);
        });
      });

      // Interpolator curve.
      const line = d3.line<LambdaResult>()
        .x((d) => xScale(d.lambda))
        .y((d) => yScale(d.mean))
        .curve(d3.curveMonotoneX);

      svg.append("path").datum(results).attr("fill", "none")
        .attr("stroke", "#db2777").attr("stroke-width", 2.5).attr("d", line);

      // Scatter dots.
      results.forEach((r) => {
        svg.append("circle").attr("cx", xScale(r.lambda)).attr("cy", yScale(r.mean)).attr("r", 6)
          .attr("fill", "#db2777").attr("stroke", "#fff").attr("stroke-width", 2);

        svg.append("text").attr("x", xScale(r.lambda) + 7).attr("y", yScale(r.mean) - 2)
          .attr("font-size", 9.5).attr("fill", "var(--rl-ink-muted)")
          .text(`${r.mean.toFixed(4)} ± ${r.std.toFixed(4)}`);
      });

      // Monotone trend annotation.
      svg.append("text").attr("x", xScale(0.5)).attr("y", pad.t + 2)
        .attr("font-size", 10).attr("fill", "#7c3aed").attr("text-anchor", "middle")
        .text("↑ λ → ↑ variance (critic easy on this MDP)");

      wrapper.appendChild(svg.node()!);

      // Bias-variance schematic.
      const schematic = document.createElement("div");
      schematic.style.cssText = "background:#f8f6f0;border-radius:6px;padding:10px;margin-top:10px;font-size:11px;color:var(--rl-ink-muted);";
      schematic.innerHTML = `
        <strong>Bias-variance trade:</strong><br>
        λ = 0 (TD residual): <span style="color:#0e7490">low variance</span>, some bias from imperfect critic.<br>
        λ = 1 (MC return): <span style="color:#dc2626">high variance</span>, no bootstrap bias.<br>
        On this MDP the critic converges easily → bias ≈ 0 → λ = 0 wins on variance.
        The canonical λ = 0.95 default comes from large continuous-control problems
        where the critic is inaccurate and bootstrap bias dominates.
      `;
      wrapper.appendChild(schematic);
      this.appendChild(wrapper);
    }
  },
);
