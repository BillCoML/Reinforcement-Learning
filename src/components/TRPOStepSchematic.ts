/**
 * V3 — TRPO Step Schematic.
 * Static schematic of one TRPO iteration: vanilla gradient direction,
 * CG-corrected natural gradient, trust region ellipse, and line search.
 */
import * as d3 from "d3";

customElements.define(
  "ppo-trpo-step-schematic",
  class extends HTMLElement {
    connectedCallback() { this.render(); }

    private render() {
      this.innerHTML = "";
      const W = 420, H = 260;

      const wrapper = document.createElement("div");
      wrapper.style.cssText = `max-width:${W + 32}px;margin:24px auto;background:var(--rl-surface);border:1px solid var(--rl-border);border-radius:8px;padding:16px;`;

      const title = document.createElement("p");
      title.style.cssText = "font-weight:700;font-size:14px;margin:0 0 4px;";
      title.textContent = "V3 — TRPO Step Schematic";
      wrapper.appendChild(title);

      const desc = document.createElement("p");
      desc.style.cssText = "font-size:12px;color:var(--rl-ink-muted);margin:0 0 10px;";
      desc.textContent = "One TRPO iteration: vanilla gradient (amber), CG-corrected natural gradient (green), trust region ellipse (violet). Line search selects the largest step inside the ellipse.";
      wrapper.appendChild(desc);

      const svg = d3.create("svg").attr("viewBox", `0 0 ${W} ${H}`)
        .style("width", "100%").style("height", "auto");

      const cx = W * 0.35, cy = H * 0.55;

      // Background grid.
      for (let x = 0; x < W; x += 40) {
        svg.append("line").attr("x1", x).attr("y1", 0).attr("x2", x).attr("y2", H)
          .attr("stroke", "#f0f0f0").attr("stroke-width", 1);
      }
      for (let y = 0; y < H; y += 40) {
        svg.append("line").attr("x1", 0).attr("y1", y).attr("x2", W).attr("y2", y)
          .attr("stroke", "#f0f0f0").attr("stroke-width", 1);
      }

      // Trust region ellipse (tilted to show KL ≠ L2).
      svg.append("ellipse")
        .attr("cx", cx).attr("cy", cy)
        .attr("rx", 110).attr("ry", 70)
        .attr("transform", `rotate(-20,${cx},${cy})`)
        .attr("fill", "rgba(124,58,237,0.06)")
        .attr("stroke", "#7c3aed")
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "6,3");

      // θ_old point.
      svg.append("circle").attr("cx", cx).attr("cy", cy).attr("r", 6)
        .attr("fill", "#1c1e22").attr("stroke", "#fff").attr("stroke-width", 2);
      svg.append("text").attr("x", cx - 12).attr("y", cy + 20)
        .attr("font-size", 11).attr("fill", "#1c1e22").attr("text-anchor", "middle").text("θ_old");

      const defs = svg.append("defs");
      const addMarker = (id: string, color: string) => {
        const m = defs.append("marker").attr("id", id)
          .attr("viewBox", "0 0 6 6").attr("refX", 5).attr("refY", 3)
          .attr("markerWidth", 5).attr("markerHeight", 5).attr("orient", "auto");
        m.append("path").attr("d", "M0,0 L6,3 L0,6 Z").attr("fill", color);
      };
      addMarker("markerAmber", "#b45309");
      addMarker("markerGreen", "#15803d");
      addMarker("markerViolet", "#7c3aed");

      // Vanilla gradient direction (amber, goes outside ellipse).
      const vgEndX = cx + 160, vgEndY = cy - 60;
      svg.append("line").attr("x1", cx).attr("y1", cy).attr("x2", vgEndX).attr("y2", vgEndY)
        .attr("stroke", "#b45309").attr("stroke-width", 2.5)
        .attr("stroke-dasharray", "5,3").attr("marker-end", "url(#markerAmber)");
      svg.append("text").attr("x", vgEndX + 6).attr("y", vgEndY)
        .attr("font-size", 11).attr("fill", "#b45309").text("∇J (vanilla)");

      // Natural gradient (CG-corrected, tilted, stays inside ellipse).
      const ngEndX = cx + 100, ngEndY = cy - 95;
      svg.append("line").attr("x1", cx).attr("y1", cy).attr("x2", ngEndX).attr("y2", ngEndY)
        .attr("stroke", "#15803d").attr("stroke-width", 2.5)
        .attr("marker-end", "url(#markerGreen)");
      svg.append("text").attr("x", ngEndX + 6).attr("y", ngEndY)
        .attr("font-size", 11).attr("fill", "#15803d").text("F⁻¹∇J (natural)");

      // Line search steps along natural gradient direction.
      const dir = [(ngEndX - cx) / 100, (ngEndY - cy) / 100];
      [30, 55, 78].forEach((t, i) => {
        const accepted = i === 1;
        svg.append("circle")
          .attr("cx", cx + dir[0] * t).attr("cy", cy + dir[1] * t).attr("r", accepted ? 5 : 3.5)
          .attr("fill", accepted ? "#7c3aed" : "#ccc")
          .attr("stroke", accepted ? "#fff" : "none").attr("stroke-width", accepted ? 1.5 : 0);
      });

      // Arrow from θ_old to accepted step.
      const accX = cx + dir[0] * 55, accY = cy + dir[1] * 55;
      svg.append("line").attr("x1", cx).attr("y1", cy).attr("x2", accX).attr("y2", accY)
        .attr("stroke", "#7c3aed").attr("stroke-width", 2)
        .attr("marker-end", "url(#markerViolet)");
      svg.append("text").attr("x", accX - 30).attr("y", accY + 16)
        .attr("font-size", 10).attr("fill", "#7c3aed").text("θ_new (accepted)");

      // Trust region label.
      svg.append("text").attr("x", cx - 110).attr("y", cy + 10)
        .attr("font-size", 10).attr("fill", "#7c3aed").text("KL trust region");

      // "outside" label.
      svg.append("text").attr("x", vgEndX - 40).attr("y", vgEndY - 12)
        .attr("font-size", 9).attr("fill", "#dc2626").attr("font-style", "italic").text("violates KL ≤ δ");

      wrapper.appendChild(svg.node()!);
      this.appendChild(wrapper);
    }
  },
);
