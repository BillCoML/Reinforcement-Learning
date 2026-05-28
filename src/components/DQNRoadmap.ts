/**
 * V8 — DQN Roadmap.
 * Thin wrapper that mounts <roadmap-mini active="function-approximation">.
 * Ensures the roadmap-mini custom element is defined before insertion.
 */
import "./RoadmapMini";

class DQNRoadmap extends HTMLElement {
  connectedCallback() {
    const roadmap = document.createElement("roadmap-mini") as HTMLElement;
    roadmap.setAttribute("active", "function-approximation");
    this.appendChild(roadmap);
  }
}

customElements.define("dqn-roadmap", DQNRoadmap);
