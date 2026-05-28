/** Thin wrapper that renders the RoadmapMini with active="trpo-ppo". */
customElements.define(
  "ppo-roadmap-mini",
  class extends HTMLElement {
    connectedCallback() {
      this.innerHTML = "";
      const rm = document.createElement("roadmap-mini");
      rm.setAttribute("active", "trpo-ppo");
      this.appendChild(rm);
    }
  },
);
