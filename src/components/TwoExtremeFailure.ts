/**
 * V3 — The Two-Extreme Failure. Two regret curves averaged over many seeds
 * (pure-greedy lock-in and uniform random), both visibly linear, with slope
 * annotations. The Lai–Robbins floor C·log t is overlaid — log-shaped and far
 * below — to motivate why §4–§6 exist. Static; pulls from the offline dataset.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";
import { loadRegretCurves } from "../lesson/data";

export class TwoExtremeFailure extends HTMLElement {
  private width = 720;
  private height = 320;
  private margin = { top: 18, right: 120, bottom: 38, left: 52 };

  connectedCallback(): void {
    void this.render();
  }

  private async render(): Promise<void> {
    this.innerHTML = "";
    const { panel, body, setStatus } = createPanel({ id: "two-extreme-failure" });
    this.appendChild(panel);

    const wrap = document.createElement("div");
    wrap.className = "chart-wrap";
    body.appendChild(wrap);

    let data;
    try {
      data = await loadRegretCurves();
    } catch (e) {
      wrap.innerHTML = `<p style="color:var(--rl-ink-faint)">Could not load regret data.</p>`;
      return;
    }

    setStatus(`avg of ${data.seeds} seeds · T=${data.T}`);
    this.draw(wrap, data.algos, data.laiRobbinsConstant, data.T);
  }

  private draw(
    wrap: HTMLElement,
    algos: Awaited<ReturnType<typeof loadRegretCurves>>["algos"],
    lrConst: number,
    T: number,
  ): void {
    const m = this.margin;
    const iw = this.width - m.left - m.right;
    const ih = this.height - m.top - m.bottom;

    const NS = "http://www.w3.org/2000/svg";
    const svgEl = document.createElementNS(NS, "svg");
    svgEl.setAttribute("viewBox", `0 0 ${this.width} ${this.height}`);
    svgEl.setAttribute("width", "100%");
    svgEl.classList.add("rl-svg");
    wrap.appendChild(svgEl);
    const svg = d3.select(svgEl as SVGSVGElement);

    const greedy = algos.greedy.mean;
    const random = algos.random.mean;
    const yMax = Math.max(random[random.length - 1], greedy[greedy.length - 1]) * 1.05;

    const x = d3.scaleLinear().domain([0, T]).range([0, iw]);
    const y = d3.scaleLinear().domain([0, yMax]).range([ih, 0]);

    const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);

    g.append("g")
      .attr("class", "rl-axis")
      .attr("transform", `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(6).tickFormat(d3.format("~s")));
    g.append("g").attr("class", "rl-axis").call(d3.axisLeft(y).ticks(5));
    g.append("text")
      .attr("class", "axis-label")
      .attr("x", iw / 2)
      .attr("y", ih + 32)
      .attr("text-anchor", "middle")
      .text("rounds  t");
    g.append("text")
      .attr("class", "axis-label")
      .attr("transform", "rotate(-90)")
      .attr("x", -ih / 2)
      .attr("y", -38)
      .attr("text-anchor", "middle")
      .text("cumulative regret  R_t");

    // downsample for path efficiency
    const stride = Math.max(1, Math.floor(random.length / 600));
    const toPts = (arr: number[]) =>
      arr.filter((_, i) => i % stride === 0 || i === arr.length - 1).map((v, i) => ({
        t: Math.min(i * stride, arr.length - 1),
        v,
      }));

    const line = d3
      .line<{ t: number; v: number }>()
      .x((d) => x(d.t))
      .y((d) => y(d.v));

    const floorPts = d3.range(1, T, Math.max(1, Math.floor(T / 400))).map((t) => ({
      t,
      v: lrConst * Math.log(t),
    }));

    const drawCurve = (arr: number[], color: string, label: string) => {
      const pts = toPts(arr);
      g.append("path")
        .datum(pts)
        .attr("fill", "none")
        .attr("stroke", color)
        .attr("stroke-width", 2)
        .attr("d", line);
      const final = arr[arr.length - 1];
      const slope = final / T;
      g.append("text")
        .attr("class", "annot")
        .attr("x", iw + 6)
        .attr("y", y(final) + 4)
        .attr("fill", color)
        .text(label);
      g.append("text")
        .attr("class", "annot")
        .attr("x", iw + 6)
        .attr("y", y(final) + 18)
        .attr("fill", "var(--rl-ink-faint)")
        .text(`slope ≈ ${slope.toFixed(3)}`);
    };

    drawCurve(random, "var(--rl-algo-random)", "random");
    drawCurve(greedy, "var(--rl-algo-greedy)", "greedy");

    // Lai-Robbins floor (log-shaped, far below)
    g.append("path")
      .datum(floorPts)
      .attr("fill", "none")
      .attr("stroke", "var(--rl-rule)")
      .attr("stroke-width", 1.4)
      .attr("stroke-dasharray", "5 4")
      .attr("d", line);
    g.append("text")
      .attr("class", "annot")
      .attr("x", x(T) + 6)
      .attr("y", y(lrConst * Math.log(T)) + 4)
      .attr("fill", "var(--rl-ink-muted)")
      .text(`${lrConst.toFixed(2)}·log t`);
  }
}

customElements.define("two-extreme-failure", TwoExtremeFailure);
