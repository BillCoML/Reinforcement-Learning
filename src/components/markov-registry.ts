/**
 * Registry of Markov-lesson interactive Web Components. Each component
 * self-registers via customElements.define at the bottom of its own file;
 * importing it here runs that side effect. Populated across build steps 4–7.
 */

import "./WeatherChainExplorer"; // V1
import "./PowerOfPAnimator"; // V2
import "./StateClassificationInspector"; // V3

export {};
