/**
 * V5 — n-step Backup Diagrams.
 * Left panel: vertical backup-diagram stacks for n=1,2,4,MC (selectable).
 * Right panel: RMSE vs n interpolation curve from pre-computed JSON data.
 */
import * as d3 from "d3";
import { createPanel } from "./PanelChrome";

interface NStepData {
  n_vals: number[];
  gridworld: { n: number; rmse: number }[];
  V_true_start: number;
}

async function loadData(): Promise<NStepData> {
  const res = await fetch("/data/td/n_step_interpolation.json");
  return res.json();
}

const COLORS: Record<number, string> = {
  1: "var(--td-td0)",
  2: "var(--td-nstep)",
  4: "var(--td-nstep)",
  8: "var(--td-lambda)",
  16: "var(--td-lambda)",
  100: "var(--td-mc)",
};

class NStepBackupDiagrams extends HTMLElement {
  private selected = 4;
  private data: NStepData | null = null;

  connectedCallback() {
    this.build();
    loadData().then(d => { this.data = d; this.redraw(); });
  }

  private svgEl: SVGSVGElement | null = null;

  private build() {
    this.innerHTML = "";
    const { panel, body, setStatus } = createPanel({ id: "n-step-backup-diagrams" });

    const nOptions = [1, 2, 4, 8, 16, "∞ (MC)"];
    const ctrl = document.createElement("div");
    ctrl.className = "rl-controls-row";
    ctrl.innerHTML = `
      <label class="rl-label">Show backup for n =</label>
      ${nOptions.map((n, i) => `<button class="rl-btn ${i === 2 ? "is-active" : ""}" data-n="${n}" id="ns-btn-${i}">${n}</button>`).join("")}
    `;
    body.appendChild(ctrl);

    const wrap = document.createElement("div");
    wrap.style.overflowX = "auto";
    const W = 860, H = 380;
    this.svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg") as SVGSVGElement;
    this.svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    this.svgEl.style.width = "100%";
    this.svgEl.style.maxWidth = `${W}px`;
    wrap.appendChild(this.svgEl);
    body.appendChild(wrap);
    this.appendChild(panel);

    ctrl.querySelectorAll("button[data-n]").forEach((btn) => {
      btn.addEventListener("click", () => {
        ctrl.querySelectorAll("button[data-n]").forEach(b => b.classList.remove("is-active"));
        btn.classList.add("is-active");
        const val = (btn as HTMLButtonElement).dataset.n!;
        this.selected = val === "∞ (MC)" ? 100 : +val;
        setStatus(`n=${this.selected}`);
        this.redraw();
      });
    });

    setStatus("n=4");
    this.redraw();
  }

  private redraw() {
    if (!this.svgEl) return;
    const svg = d3.select(this.svgEl);
    svg.selectAll("*").remove();
    const W = 860, H = 380;
    this.drawDiagram(svg, W, H);
  }

  private drawDiagram(svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, W: number, H: number) {
    const diagW = 340, diagX = 20, diagY = 20;

    // Left panel: backup diagram
    svg.append("text").attr("x", diagX + diagW / 2).attr("y", 14)
      .attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", 600)
      .attr("fill", "#1e293b").text(`n-step backup diagram (n=${this.selected === 100 ? "∞ MC" : this.selected})`);

    const n = Math.min(this.selected, 8); // cap visual depth
    const nodeR = 10, actionR = 5;
    const stepH = (H - diagY - 40) / (n + 1);
    const cx = diagX + diagW / 2;
    const color = COLORS[this.selected] ?? "var(--td-lambda)";

    for (let i = 0; i <= n; i++) {
      const y = diagY + 20 + i * stepH;
      const isLast = i === n;
      const isBootstrap = isLast && this.selected !== 100;

      if (i < n) {
        // Action node
        const ay = y + stepH * 0.4;
        svg.append("circle").attr("cx", cx + 24).attr("cy", ay).attr("r", actionR)
          .attr("fill", "#94a3b8").attr("stroke", "white").attr("stroke-width", 1);
        // Reward label
        svg.append("text").attr("x", cx + 40).attr("y", ay + 4)
          .attr("font-size", 9).attr("fill", "#64748b").text(`r${i > 0 ? `_${i}` : ""}`);
        // Line from state to action
        svg.append("line").attr("x1", cx).attr("y1", y + nodeR).attr("x2", cx + 24).attr("y2", ay - actionR)
          .attr("stroke", "#94a3b8").attr("stroke-width", 1.5);
        // Line from action to next state
        svg.append("line").attr("x1", cx + 24).attr("y1", ay + actionR).attr("x2", cx).attr("y2", y + stepH - nodeR)
          .attr("stroke", "#94a3b8").attr("stroke-width", 1.5);
      }

      // State node
      svg.append("circle").attr("cx", cx).attr("cy", y).attr("r", nodeR)
        .attr("fill", i === 0 ? "#1e293b" : isBootstrap ? color : "#475569")
        .attr("stroke", "white").attr("stroke-width", 1.5);
      svg.append("text").attr("x", cx).attr("y", y + 3.5)
        .attr("text-anchor", "middle").attr("font-size", 8).attr("fill", "white")
        .text(i === 0 ? "s_t" : `s+${i}`);

      if (isBootstrap) {
        // Bootstrap box
        svg.append("rect").attr("x", cx - 28).attr("y", y - 16).attr("width", 56).attr("height", 32)
          .attr("fill", "none").attr("stroke", color).attr("stroke-width", 2).attr("rx", 4);
        svg.append("text").attr("x", cx + 36).attr("y", y + 4)
          .attr("font-size", 10).attr("fill", color).attr("font-weight", 600).text(`V̂(s+${n})`);
      }
    }

    if (this.selected === 100) {
      // MC: terminal node at bottom
      const y = diagY + 20 + n * stepH;
      svg.append("text").attr("x", cx + 16).attr("y", y + 4)
        .attr("font-size", 9).attr("fill", "#64748b").text("terminal");
    }

    // Legend
    svg.append("text").attr("x", diagX).attr("y", H - 12)
      .attr("font-size", 9).attr("fill", "#94a3b8")
      .text(this.selected === 100 ? "MC: waits until terminal — no bootstrap" : `n=${this.selected}: bootstraps at s+${n} with V̂`);

    // Vertical divider
    svg.append("line").attr("x1", diagX + diagW + 20).attr("x2", diagX + diagW + 20)
      .attr("y1", 10).attr("y2", H - 10).attr("stroke", "#e2e8f0").attr("stroke-width", 1);

    // Right panel: RMSE vs n
    const rX = diagX + diagW + 40, rY = 20, rW = W - rX - 20, rH = H - 40;
    svg.append("text").attr("x", rX + rW / 2).attr("y", 14)
      .attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", 600)
      .attr("fill", "#1e293b").text("RMSE vs n — bias↑ variance↑ at large n");

    if (!this.data) {
      svg.append("text").attr("x", rX + rW / 2).attr("y", rY + rH / 2)
        .attr("text-anchor", "middle").attr("font-size", 11).attr("fill", "#94a3b8").text("loading…");
      return;
    }

    const pts = this.data.gridworld;
    const margin = { left: 40, right: 10, top: 15, bottom: 35 };
    const iW = rW - margin.left - margin.right;
    const iH = rH - margin.top - margin.bottom;

    const nVals = pts.map(p => p.n);
    const rmseVals = pts.map(p => p.rmse);
    const xS = d3.scaleLog([1, Math.max(...nVals)], [rX + margin.left, rX + margin.left + iW]);
    const yS = d3.scaleLinear([0, Math.max(...rmseVals) * 1.1], [rY + margin.top + iH, rY + margin.top]);

    // Axes
    svg.append("line").attr("x1", rX + margin.left).attr("x2", rX + margin.left + iW)
      .attr("y1", rY + margin.top + iH).attr("y2", rY + margin.top + iH)
      .attr("stroke", "#e2e8f0").attr("stroke-width", 1);
    svg.append("line").attr("x1", rX + margin.left).attr("x2", rX + margin.left)
      .attr("y1", rY + margin.top).attr("y2", rY + margin.top + iH)
      .attr("stroke", "#e2e8f0").attr("stroke-width", 1);

    // Grid lines
    for (const t of yS.ticks(4)) {
      svg.append("line").attr("x1", rX + margin.left).attr("x2", rX + margin.left + iW)
        .attr("y1", yS(t)).attr("y2", yS(t)).attr("stroke", "#f1f5f9").attr("stroke-width", 1);
      svg.append("text").attr("x", rX + margin.left - 3).attr("y", yS(t) + 3)
        .attr("text-anchor", "end").attr("font-size", 8).attr("fill", "#94a3b8").text(t.toFixed(3));
    }

    // X ticks
    nVals.forEach(n => {
      svg.append("text").attr("x", xS(n)).attr("y", rY + margin.top + iH + 14)
        .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", "#475569")
        .text(n === 100 ? "MC" : String(n));
    });

    // Connecting line
    svg.append("path").datum(pts)
      .attr("d", d3.line<{ n: number; rmse: number }>()
        .x(p => xS(p.n)).y(p => yS(p.rmse)).curve(d3.curveMonotoneX))
      .attr("fill", "none").attr("stroke", "#94a3b8").attr("stroke-width", 1.5).attr("stroke-dasharray", "4,2");

    // Points
    pts.forEach(p => {
      const isActive = (this.selected === 100 && p.n === 100) || p.n === this.selected;
      svg.append("circle").attr("cx", xS(p.n)).attr("cy", yS(p.rmse)).attr("r", isActive ? 6 : 4)
        .attr("fill", isActive ? (COLORS[this.selected] ?? "#64748b") : "#94a3b8")
        .attr("stroke", "white").attr("stroke-width", 1.5);
      if (isActive) {
        svg.append("text").attr("x", xS(p.n) + 8).attr("y", yS(p.rmse) - 3)
          .attr("font-size", 9).attr("fill", COLORS[this.selected] ?? "#64748b")
          .attr("font-weight", 600).text(`RMSE=${p.rmse.toFixed(3)}`);
      }
    });

    // Axis labels
    svg.append("text").attr("x", rX + margin.left + iW / 2).attr("y", rY + rH - 4)
      .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", "#94a3b8").text("n (log scale)");
    svg.append("text").attr("x", rX + margin.left - 28).attr("y", rY + margin.top + iH / 2)
      .attr("transform", `rotate(-90, ${rX + margin.left - 28}, ${rY + margin.top + iH / 2})`)
      .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", "#94a3b8").text("RMSE of V̂(s₀)");
  }
}

customElements.define("n-step-backup-diagrams", NStepBackupDiagrams);
