/**
 * V7 — Double DQN vs Max-Bias.
 * Left: Sutton-Barto two-state MDP schematic.
 * Right: Learning curves showing DQN overestimation vs Double DQN correction.
 * Bottom: Q(A, left) estimate over episodes.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";
import { runMaxBias } from "../dqn/max-bias";

const W = 880;
const H = 440;

class DoubleDQNVsMaxBias extends HTMLElement {
  connectedCallback() {
    const { panel, body } = createPanel({ id: "double-dqn-vs-max-bias" });
    const wrap = document.createElement("div");
    wrap.style.overflowX = "auto";
    const svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg") as SVGSVGElement;
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svgEl.style.width = "100%"; svgEl.style.maxWidth = `${W}px`;
    wrap.appendChild(svgEl);
    body.appendChild(wrap);
    this.appendChild(panel);

    const svg = d3.select(svgEl);
    this.drawMDP(svg);
    this.drawCurves(svg);
  }

  private drawMDP(svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) {
    const g = svg.append("g").attr("transform", "translate(10,10)");
    const w = 260, h = 220;

    g.append("rect").attr("width", w).attr("height", h).attr("rx", 6)
      .attr("fill", "var(--fa-tabular)").attr("opacity", 0.06)
      .attr("stroke", "var(--fa-tabular)").attr("stroke-width", 1);

    g.append("text").attr("x", w / 2).attr("y", 18)
      .attr("text-anchor", "middle").attr("font-size", "11px").attr("font-weight", "600")
      .attr("fill", "var(--fa-tabular)").text("Maximization-Bias MDP");

    const defs = svg.append("defs");
    const addMarker = (id: string, color: string) =>
      defs.append("marker").attr("id", id)
        .attr("viewBox", "0 0 10 10").attr("refX", 9).attr("refY", 5)
        .attr("markerWidth", 5).attr("markerHeight", 5).attr("orient", "auto")
        .append("path").attr("d", "M 0 0 L 10 5 L 0 10 z").attr("fill", color);
    addMarker("arr-a", "var(--rl-ink)");
    addMarker("arr-b", "var(--dqn-replay)");
    addMarker("arr-t", "var(--rl-ink-muted)");

    // State circles
    const stateY = 110;
    const states = [
      { label: "A", x: 50, color: "var(--dqn-online)" },
      { label: "B", x: 140, color: "var(--dqn-replay)" },
      { label: "T", x: 220, color: "var(--rl-ink-muted)" },
    ];
    states.forEach(({ label, x, color }) => {
      g.append("circle").attr("cx", x).attr("cy", stateY).attr("r", 22)
        .attr("fill", color).attr("opacity", 0.15)
        .attr("stroke", color).attr("stroke-width", 2);
      g.append("text").attr("x", x).attr("y", stateY + 5)
        .attr("text-anchor", "middle").attr("font-size", "14px").attr("font-weight", "700")
        .attr("fill", color).text(label);
    });

    // Arrow A → T (right, r=0)
    g.append("line").attr("x1", 72).attr("y1", stateY - 10)
      .attr("x2", 198).attr("y2", stateY - 10)
      .attr("stroke", "var(--rl-ink-muted)").attr("stroke-width", 1.5)
      .attr("marker-end", "url(#arr-t)");
    g.append("text").attr("x", 135).attr("y", stateY - 14)
      .attr("text-anchor", "middle").attr("font-size", "9px")
      .attr("fill", "var(--rl-ink-muted)").text("right, r=0");

    // Arrow A → B (left, r=0)
    g.append("line").attr("x1", 72).attr("y1", stateY + 10)
      .attr("x2", 118).attr("y2", stateY + 10)
      .attr("stroke", "var(--dqn-replay)").attr("stroke-width", 1.5)
      .attr("marker-end", "url(#arr-b)");
    g.append("text").attr("x", 95).attr("y", stateY + 24)
      .attr("text-anchor", "middle").attr("font-size", "9px")
      .attr("fill", "var(--dqn-replay)").text("left, r=0");

    // Arrow B → T (10 actions, r~N(-0.1,1))
    g.append("line").attr("x1", 162).attr("y1", stateY)
      .attr("x2", 198).attr("y2", stateY)
      .attr("stroke", "var(--rl-ink)").attr("stroke-width", 1.5)
      .attr("marker-end", "url(#arr-a)");
    g.append("text").attr("x", 180).attr("y", stateY - 8)
      .attr("text-anchor", "middle").attr("font-size", "9px")
      .attr("fill", "var(--rl-ink)").text("10 acts");
    g.append("text").attr("x", 180).attr("y", stateY + 20)
      .attr("text-anchor", "middle").attr("font-size", "9px")
      .attr("fill", "var(--rl-ink-muted)").text("r~𝒩(−0.1,1)");

    // Annotation
    const ann = [
      "Optimal: always go right",
      "(E[r_right]=0 > E[r_B]=−0.1)",
      "",
      "DQN: max over 10 noisy Q(B,·)",
      "→ systematic overestimation",
      "→ incorrectly prefers left",
      "",
      "Double DQN: decouple",
      "selection from evaluation",
      "→ removes bias",
    ];
    ann.forEach((line, i) => {
      g.append("text").attr("x", 8).attr("y", 148 + i * 13)
        .attr("font-size", "9px")
        .attr("fill", i === 3 || i === 4 || i === 5 ? "var(--triad-warning)"
          : i === 7 || i === 8 || i === 9 ? "var(--double-dqn)"
          : "var(--rl-ink-muted)")
        .attr("font-weight", i === 0 ? "600" : "normal")
        .text(line);
    });
  }

  private drawCurves(svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) {
    const dqnResult = runMaxBias({ useDoubleDqn: false, nEpisodes: 300, nSeeds: 500 });
    const ddqnResult = runMaxBias({ useDoubleDqn: true, nEpisodes: 300, nSeeds: 500 });

    const ox = 290, oy = 10;
    const cw = W - ox - 20;
    const topH = (H - 30) / 2 - 20;
    const botH = (H - 30) / 2 - 20;

    const xScale = d3.scaleLinear().domain([0, 300]).range([0, cw]);

    // === Top panel: % left from A ===
    const yScaleTop = d3.scaleLinear().domain([0, 1]).range([topH, 0]);
    const gTop = svg.append("g").attr("transform", `translate(${ox},${oy + 20})`);

    gTop.append("text").attr("x", cw / 2).attr("y", -8)
      .attr("text-anchor", "middle").attr("class", "rl-label")
      .text("% left from A  (DQN vs Double DQN, 500-seed avg)");

    gTop.append("g").attr("transform", `translate(0,${topH})`).call(
      d3.axisBottom(xScale).ticks(6).tickFormat(d => `${d}`)
    );
    gTop.append("g").call(d3.axisLeft(yScaleTop).ticks(5).tickFormat(d3.format(".0%")));

    // Optimal reference (5% ≈ epsilon exploration)
    gTop.append("line").attr("x1", 0).attr("y1", yScaleTop(0.05))
      .attr("x2", cw).attr("y2", yScaleTop(0.05))
      .attr("stroke", "var(--rl-ink-muted)").attr("stroke-width", 1)
      .attr("stroke-dasharray", "4,4");
    gTop.append("text").attr("x", cw - 2).attr("y", yScaleTop(0.05) - 3)
      .attr("text-anchor", "end").attr("font-size", "9px")
      .attr("fill", "var(--rl-ink-muted)").text("optimal ~5%");

    const lineTop = d3.line<number>().x((_, i) => xScale(i)).y(d => yScaleTop(d)).curve(d3.curveMonotoneX);
    gTop.append("path").datum(dqnResult.leftFractionPerEpisode).attr("d", lineTop)
      .attr("fill", "none").attr("stroke", "var(--triad-warning)").attr("stroke-width", 2);
    gTop.append("path").datum(ddqnResult.leftFractionPerEpisode).attr("d", lineTop)
      .attr("fill", "none").attr("stroke", "var(--double-dqn)").attr("stroke-width", 2);

    // Legend top
    [
      { label: "DQN", color: "var(--triad-warning)" },
      { label: "Double DQN", color: "var(--double-dqn)" },
    ].forEach(({ label, color }, i) => {
      const lx = 8 + i * 110;
      gTop.append("line").attr("x1", lx).attr("y1", topH - 8).attr("x2", lx + 16).attr("y2", topH - 8)
        .attr("stroke", color).attr("stroke-width", 2);
      gTop.append("text").attr("x", lx + 20).attr("y", topH - 4)
        .attr("font-size", "10px").attr("fill", "var(--rl-ink)").text(label);
    });

    // === Bottom panel: Q(A, left) estimate ===
    const allQ = [...dqnResult.qALeftPerEpisode, ...ddqnResult.qALeftPerEpisode];
    const qMin = Math.min(-0.3, d3.min(allQ)!);
    const qMax = Math.max(0.5, d3.max(allQ)!);
    const yScaleBot = d3.scaleLinear().domain([qMin, qMax]).range([botH, 0]);

    const gBot = svg.append("g").attr("transform", `translate(${ox},${oy + topH + 60})`);

    gBot.append("text").attr("x", cw / 2).attr("y", -8)
      .attr("text-anchor", "middle").attr("class", "rl-label")
      .text("Q(A, left) estimate over training");

    gBot.append("g").attr("transform", `translate(0,${botH})`).call(
      d3.axisBottom(xScale).ticks(6).tickFormat(d => `${d}`)
    );
    gBot.append("g").call(d3.axisLeft(yScaleBot).ticks(5).tickFormat(d3.format(".2f")));

    // True Q* reference (reward 0 from right, so Q(A,left)=E[max Q(B,·)] which is biased; true = -0.1)
    gBot.append("line").attr("x1", 0).attr("y1", yScaleBot(-0.1))
      .attr("x2", cw).attr("y2", yScaleBot(-0.1))
      .attr("stroke", "var(--rl-ink-muted)").attr("stroke-width", 1)
      .attr("stroke-dasharray", "4,4");
    gBot.append("text").attr("x", cw - 2).attr("y", yScaleBot(-0.1) - 3)
      .attr("text-anchor", "end").attr("font-size", "9px")
      .attr("fill", "var(--rl-ink-muted)").text("true −0.1");

    const lineBot = d3.line<number>().x((_, i) => xScale(i)).y(d => yScaleBot(d)).curve(d3.curveMonotoneX);
    gBot.append("path").datum(dqnResult.qALeftPerEpisode).attr("d", lineBot)
      .attr("fill", "none").attr("stroke", "var(--triad-warning)").attr("stroke-width", 2);
    gBot.append("path").datum(ddqnResult.qALeftPerEpisode).attr("d", lineBot)
      .attr("fill", "none").attr("stroke", "var(--double-dqn)").attr("stroke-width", 2);

    gBot.append("text").attr("x", cw / 2).attr("y", botH + 20)
      .attr("text-anchor", "middle").attr("font-size", "10px").attr("fill", "var(--rl-ink-muted)")
      .text("episode");
    gBot.append("text").attr("x", -botH / 2).attr("y", -32).attr("transform", "rotate(-90)")
      .attr("text-anchor", "middle").attr("font-size", "10px").attr("fill", "var(--rl-ink-muted)")
      .text("Q(A, left)");
  }
}

customElements.define("dqn-double-vs-max-bias", DoubleDQNVsMaxBias);
