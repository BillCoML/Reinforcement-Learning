import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const dqnSection07: Section = {
  id: "double-dqn",
  title: "DQN Family: Double DQN, Dueling, Prioritized, Distributional",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§7</span>DQN Family: Double DQN, Dueling, Prioritized, Distributional</h2>
<p class="tagline"><em>Each addressing a specific shortcoming of vanilla DQN. Double DQN cleans up max bias.</em></p>

<h3 id="double-dqn-section">Double DQN: max bias revisited</h3>

<p>Lesson 8's §5 documented Q-learning's <strong>maximization bias</strong>:
$\\mathbb{E}[\\max_a \\hat Q(s, a)] \\geq \\max_a \\mathbb{E}[\\hat Q(s, a)]$.
With function approximation the noise sources multiply, and the bias is amplified.</p>

<p>The Double DQN trick (van Hasselt, Guez, Silver 2015) decouples action selection from
action evaluation in the bootstrap target. Instead of</p>

$$\\text{target} = r + \\gamma \\max_{a'} Q_{\\theta^-}(s', a'),$$

<p>Double DQN uses</p>

$$\\text{target} = r + \\gamma Q_{\\theta^-}\\!\\bigl(s',\\, \\arg\\max_{a'} Q_\\theta(s', a')\\bigr).$$

<p>The online network $Q_\\theta$ selects the action; the target network $Q_{\\theta^-}$
evaluates it. Selection bias and evaluation bias are no longer correlated, and the
maximization bias shrinks substantially. The change is one line of code.</p>

<h3 id="dueling-dqn">Dueling DQN: separating state-value and advantage</h3>

<p>The action-value function decomposes as $Q^\\pi(s, a) = V^\\pi(s) + A^\\pi(s, a)$.
In many states, the specific action matters little — $V^\\pi(s)$ dominates. Dueling DQN
(Wang et al. 2016) gives the network an inductive bias for this decomposition:</p>

<pre class="rl-pre">Input → shared trunk → split:
                       ├─ V head → V(s)
                       └─ A head → A(s, ·)
Q(s,a) = V(s) + A(s,a) − (1/|𝒜|) Σ_{a'} A(s,a')   # mean-centering</pre>

<h3>Prioritized experience replay</h3>

<p>Instead of sampling uniformly from the buffer, prioritize transitions with high TD error.
Priority $\\propto |\\delta|^\\alpha$, with importance-sampling correction to maintain
unbiasedness (Schaul et al. 2015).</p>

<h3>Distributional DQN</h3>

<p>Vanilla DQN learns $\\mathbb{E}[Q(s, a)]$. The distributional family (Bellemare,
Dabney, Munos 2017) learns the full <em>distribution</em> over returns. $C51$ discretizes
the return support into 51 bins and learns a categorical distribution; $\\text{QR-DQN}$
and $\\text{IQN}$ generalize to quantile representations.</p>

<h3>Rainbow</h3>

<p>Hessel et al. 2018 combine seven DQN improvements (Double, Dueling, Prioritized,
Distributional, $n$-step returns, NoisyNets) into <strong>Rainbow</strong>. On Atari,
Rainbow is one of the strongest known value-based agents.</p>

<dqn-double-vs-max-bias></dqn-double-vs-max-bias>
`);
  },
};
