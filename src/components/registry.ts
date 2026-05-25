/**
 * Central registry of interactive Web Components. Importing this module
 * registers every custom element used by the lesson. Components self-register
 * via `customElements.define` at the bottom of their own file; we just import
 * them here so the side effects run.
 */

// Components are added here as they are implemented (Steps 4–7).
import "./BanditMachine"; // V1
import "./RegretDecomposition"; // V2
import "./TwoExtremeFailure"; // V3
import "./EpsilonGreedyExplorer"; // V4
import "./UCBConfidenceBounds"; // V5
import "./ThompsonPosteriorEvolution"; // V6
import "./AlgorithmBattleArena"; // V7 (centerpiece)
import "./RoadmapMini"; // V8

export {};
