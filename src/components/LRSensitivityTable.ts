/**
 * V7 — LR Sensitivity Table.
 * Loads lr_sensitivity.json. Renders a table comparing PPO vs vanilla PG
 * across learning rates. Color-codes cells by performance.
 */
import * as d3 from "d3";

interface LREntry {
  lr: number;
  mean: number;
  std: number;
}

customElements.define(
  "ppo-lr-sensitivity-table",
  class extends HTMLElement {
    connectedCallback() { this.load(); }

    private async load() {
      this.innerHTML = `<p style="color:var(--rl-ink-muted);font-size:13px;">Loading LR sensitivity data…</p>`;
      try {
        const resp = await fetch("/data/ppo/lr_sensitivity.json");
        const data = await resp.json();
        this.render(data);
      } catch {
        this.innerHTML = `<p style="color:var(--ppo-diverged);">Failed to load lr_sensitivity.json.</p>`;
      }
    }

    private render(data: { ppo: Record<string, LREntry>; vanilla: Record<string, LREntry> }) {
      this.innerHTML = "";

      const lrKeys = Object.keys(data.ppo).sort((a, b) => Number(a) - Number(b));
      const vMax = 0.730;
      const vMin = -0.15;
      const colorScale = d3.scaleSequential(d3.interpolate("#fee2e2", "#dcfce7")).domain([vMin, vMax]);

      const wrapper = document.createElement("div");
      wrapper.style.cssText = "margin:16px 0;overflow-x:auto;";

      const table = document.createElement("table");
      table.style.cssText = "border-collapse:collapse;font-size:13px;width:100%;";

      const thead = table.createTHead();
      const hrow = thead.insertRow();
      ["Learning Rate", "PPO Mean ± Std", "Vanilla Mean ± Std", "PPO Std / Vanilla Std"].forEach((h) => {
        const th = document.createElement("th");
        th.style.cssText = "padding:6px 12px;border-bottom:2px solid #ccc;text-align:right;font-weight:700;font-size:12px;color:var(--rl-ink-muted);";
        th.textContent = h;
        hrow.appendChild(th);
      });
      hrow.cells[0].style.textAlign = "left";

      const tbody = table.createTBody();
      lrKeys.forEach((lrKey) => {
        const ppo = data.ppo[lrKey];
        const van = data.vanilla[lrKey];
        const row = tbody.insertRow();

        const cell = (text: string, bg: string, align = "right") => {
          const td = row.insertCell();
          td.style.cssText = `padding:5px 12px;border-bottom:1px solid #e5e5e5;text-align:${align};font-variant-numeric:tabular-nums;background:${bg};`;
          td.textContent = text;
          return td;
        };

        cell(`lr = ${lrKey}`, "transparent", "left");
        cell(`${ppo.mean.toFixed(4)} ± ${ppo.std.toFixed(4)}`, colorScale(ppo.mean));
        cell(`${van.mean.toFixed(4)} ± ${van.std.toFixed(4)}`, colorScale(van.mean));

        const ratio = van.std > 0.001 ? (ppo.std / van.std).toFixed(2) : "—";
        cell(ratio, "transparent");
      });

      wrapper.appendChild(table);

      // Caption.
      const caption = document.createElement("p");
      caption.style.cssText = "font-size:11px;color:var(--rl-ink-muted);margin:6px 0;";
      caption.textContent = "20 seeds, batch=10, 200 iters, λ=0.95, ε=0.2. Color = performance (green=high, red=low).";
      wrapper.appendChild(caption);

      this.appendChild(wrapper);
    }
  },
);
