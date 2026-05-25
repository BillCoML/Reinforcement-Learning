/**
 * DistributionBars — horizontal probability bars in the categorical state
 * palette, the lesson's standard way of drawing a distribution vector. An
 * optional stationary overlay π is drawn as a thin green tick + outline on each
 * bar so "is the current distribution close to π?" reads at a glance. Reused by
 * V1, V2, V4 and V5.
 */
import * as d3 from "d3";

const NS = "http://www.w3.org/2000/svg";
const STATE_VARS = [
  "--mc-state-1", "--mc-state-2", "--mc-state-3", "--mc-state-4",
  "--mc-state-5", "--mc-state-6", "--mc-state-7", "--mc-state-8",
];

export interface DistributionBarsOptions {
  width?: number;
  rowHeight?: number;
  labels?: string[];
  colors?: string[];
  max?: number;
  /** Decimal places in the printed value. */
  dp?: number;
}

export class DistributionBars {
  private svg: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private opts: Required<Omit<DistributionBarsOptions, "labels" | "colors">> &
    Pick<DistributionBarsOptions, "labels" | "colors">;
  private W: number;
  private K = 0;

  constructor(container: HTMLElement, opts: DistributionBarsOptions = {}) {
    this.opts = {
      width: opts.width ?? 360,
      rowHeight: opts.rowHeight ?? 30,
      max: opts.max ?? 1,
      dp: opts.dp ?? 3,
      labels: opts.labels,
      colors: opts.colors,
    };
    this.W = this.opts.width;
    const svgEl = document.createElementNS(NS, "svg");
    svgEl.setAttribute("width", "100%");
    svgEl.classList.add("rl-svg");
    container.appendChild(svgEl);
    this.svg = d3.select(svgEl as SVGSVGElement);
  }

  private color(i: number): string {
    return this.opts.colors?.[i] ?? `var(${STATE_VARS[i % STATE_VARS.length]})`;
  }

  private label(i: number): string {
    return this.opts.labels?.[i] ?? String(i);
  }

  update(values: number[], overlay?: number[]): void {
    const K = values.length;
    const rowH = this.opts.rowHeight;
    const H = K * rowH + 6;
    if (K !== this.K) {
      this.svg.attr("viewBox", `0 0 ${this.W} ${H}`);
      this.K = K;
    }
    const labelW = 70;
    const valueW = 52;
    const x0 = labelW;
    const trackW = this.W - labelW - valueW;
    const max = this.opts.max;

    this.svg.selectAll("*").remove();

    values.forEach((v, i) => {
      const cy = i * rowH + rowH / 2 + 3;
      const barH = rowH * 0.56;
      const g = this.svg.append("g");
      // label
      g.append("text")
        .attr("x", labelW - 8)
        .attr("y", cy)
        .attr("text-anchor", "end")
        .attr("dominant-baseline", "central")
        .style("font-family", "var(--rl-font-ui)")
        .style("font-size", "12px")
        .style("fill", "var(--rl-ink-muted)")
        .text(this.label(i));
      // track
      g.append("rect")
        .attr("x", x0)
        .attr("y", cy - barH / 2)
        .attr("width", trackW)
        .attr("height", barH)
        .attr("rx", 3)
        .attr("fill", "var(--rl-surface-2)");
      // fill
      g.append("rect")
        .attr("x", x0)
        .attr("y", cy - barH / 2)
        .attr("width", Math.max(0, Math.min(1, v / max)) * trackW)
        .attr("height", barH)
        .attr("rx", 3)
        .attr("fill", this.color(i));
      // π overlay
      if (overlay) {
        const ox = x0 + Math.max(0, Math.min(1, overlay[i] / max)) * trackW;
        g.append("line")
          .attr("x1", ox)
          .attr("x2", ox)
          .attr("y1", cy - barH / 2 - 3)
          .attr("y2", cy + barH / 2 + 3)
          .attr("stroke", "var(--mc-stationary)")
          .attr("stroke-width", 2);
      }
      // value
      g.append("text")
        .attr("x", this.W - valueW + 6)
        .attr("y", cy)
        .attr("dominant-baseline", "central")
        .style("font-family", "var(--rl-font-mono)")
        .style("font-size", "11px")
        .style("fill", "var(--rl-ink)")
        .text(v.toFixed(this.opts.dp));
    });
  }
}
