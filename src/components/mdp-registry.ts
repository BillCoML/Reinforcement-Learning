/**
 * Registry of MDP-lesson interactive Web Components. Each component
 * self-registers via customElements.define at the bottom of its own file;
 * importing it here runs that side effect. Imports are uncommented as the
 * components land across build steps 4–9.
 */

// GridworldRenderer / MDPEditor are plain classes (no custom element); imported
// directly by the components below, so they need no registry entry.
import "./MDPAnatomyExplorer"; // V1
import "./PolicyExplorer"; // V2
import "./ReturnComposer"; // V3
import "./ValueHeatmap"; // V4
import "./QQuadrantsAndAdvantage"; // V5
import "./BellmanBackupLab"; // V6 centerpiece
// import "./OptimalityExplorer"; // V7 — Step 8

export {};
