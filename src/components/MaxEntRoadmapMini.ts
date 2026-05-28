/** V8 — Thin wrapper rendering RoadmapMini with active="max-ent-rl". */
customElements.define(
  "maxent-roadmap-mini",
  class extends HTMLElement {
    connectedCallback() {
      this.innerHTML = "";
      const rm = document.createElement("roadmap-mini");
      rm.setAttribute("active", "max-ent-rl");
      this.appendChild(rm);
    }
  },
);
