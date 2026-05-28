import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const dqnSection06: Section = {
  id: "dqn-stability-lab",
  title: "DQN Stability Lab",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§6</span>DQN Stability Lab</h2>
<p class="tagline"><em>Run DQN with toggles for every component. Watch what each trick does.</em></p>

<p>Sections 1–5 introduced the components. This section makes them tangible. The DQN
Stability Lab runs all the variations side-by-side, with synchronized controls, and lets
the user discover the ablation table from §5 by direct manipulation.</p>

<p>The lab uses the <strong>$7 \\times 7$ gridworld</strong>: 49 states, four actions, walls
that make the value function genuinely non-linear in $(r, c)$, and two goals (reward $+1$
and reward $+0.5$). The Q-network is the small MLP from §4: $49 \\to 64 \\to 64 \\to 4$
with ReLU activations.</p>

<p>The training runs are pre-computed offline by <code>scripts/train_dqn.py</code> and
shipped as ONNX models plus training-trace JSON. The in-browser experience is playback
with synchronized timeline scrubbing across the ablation configurations.</p>

<p><strong>The lab's central diagnostic is Panel E: the parameter norm $\\|\\theta\\|$
over training.</strong> When $\\|\\theta\\|$ explodes, the Q-values become meaningless.
When $\\|\\theta\\|$ stays bounded, the Q-values converge to a reasonable approximation
of $Q^*$. Target networks and the replay buffer are what keeps $\\|\\theta\\|$ bounded.</p>

<dqn-stability-lab></dqn-stability-lab>
`);
  },
};
