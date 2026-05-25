import type { Section } from "../section";
import { sectionFromHTML } from "../section";
import { forwardLink } from "../../components/CrosslinkCallout";

export const mdpSection04: Section = {
  id: "state-value-function",
  title: "State Value Function Vπ",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§4</span>State Value Function <span class="rl-mono">V<sup>π</sup></span></h2>
<p class="tagline">Expected return, given that you start here and follow π.</p>

<p>The return $G_t$ is a <em>random variable</em>: it depends on the stochasticity of both the
policy and the environment. To compare policies, we take expectations.</p>

<p>The <strong>state value function</strong> of policy $\\pi$ is</p>

$$\\boxed{V^\\pi(s) \\;:=\\; \\mathbb{E}_\\pi\\!\\left[G_t \\mid S_t = s\\right] \\;=\\; \\mathbb{E}_\\pi\\!\\left[\\sum_{k=0}^\\infty \\gamma^k R_{t+k+1} \\,\\middle|\\, S_t = s\\right]}$$

<p>The subscript $\\mathbb{E}_\\pi[\\cdot]$ means “expectation under the joint distribution
induced by $\\pi$ and the MDP dynamics”: the trajectory is drawn by following $\\pi$ and $P$.</p>

<ul>
  <li>$V^\\pi(s)$ is a function of $s$, not of $t$ — the MDP is stationary
  (time-homogeneous).</li>
  <li>$V^\\pi(s)$ is finite whenever $\\gamma \\in [0,1)$ and rewards are bounded:
  $|V^\\pi(s)| \\leq R_{\\max}/(1-\\gamma)$.</li>
  <li>For terminal states, $V^\\pi(s) = 0$ — no future rewards to collect.</li>
</ul>

<p><strong>The point of $V^\\pi$.</strong> It's how we compare <em>policies, not actions</em>. If
$V^{\\pi_1}(s) > V^{\\pi_2}(s)$ for all $s$ (strictly somewhere), then $\\pi_1$ dominates
$\\pi_2$. This <strong>partial order</strong> on policies is what makes “optimal” meaningful: an
optimal policy is one not dominated by any other.</p>

<h3>Worked example (pre-verified): V^π for the uniform-random policy</h3>
<p>With $\\pi(a|s) = 1/4$ and $\\gamma = 0.9$:</p>

<table class="numeric">
  <thead><tr><th>state</th><th>col 0</th><th>col 1</th><th>col 2</th></tr></thead>
  <tbody>
    <tr><td><strong>row 0</strong></td><td>−0.421</td><td>−0.514</td><td>−0.239</td></tr>
    <tr><td><strong>row 1</strong></td><td>−0.514</td><td>0.000</td><td>−0.069</td></tr>
    <tr><td><strong>row 2</strong></td><td>−0.239</td><td>−0.069</td><td>0.000</td></tr>
  </tbody>
</table>

<p>The pit and goal are terminal ($V = 0$). Every other cell is <strong>negative</strong>: a random
walker frequently falls into the pit (1/4 chance from each neighbour), and the pit's pull
dominates the goal's on this geometry. State $(0,1)$ is worst — adjacent to the pit and far
from the goal.</p>

<p><strong>Sanity-check $(1,2) \\approx -0.069$:</strong></p>
<ul>
  <li>Up → $(0,2)$, $V \\approx -0.239$: $0.25\\,(0 + 0.9\\cdot(-0.239)) = -0.054$.</li>
  <li>Right → wall, stay: $0.25\\,(0 + 0.9\\cdot(-0.069)) = -0.0155$.</li>
  <li>Down → $(2,2)$ <strong>goal</strong>: $0.25\\,(+1 + 0) = +0.250$.</li>
  <li>Left → $(1,1)$ <strong>pit</strong>: $0.25\\,(-1 + 0) = -0.250$.</li>
</ul>
<p>Sum: $-0.054 - 0.0155 + 0.250 - 0.250 = -0.0695 \\approx -0.069$ ✓. The goal-bonus and
pit-penalty cancel; the residual negative comes from the up-action's less-favourable
future.</p>

${forwardLink({
  destination: "Lessons 3 & 5 — DP / TD Learning",
  html: `<p>Computing $V^\\pi$ is <strong>policy evaluation</strong>. Lesson 3 does it iteratively (repeated
  Bellman backups); Lesson 5's TD(0) does it from <em>samples</em>, never forming $P^\\pi$. Both
  answer the same question: what does $V^\\pi$ equal?</p>`,
})}

<p>Select a policy and hit “Compute” to run the exact linear solve. The cell shading is the
shared value scale you'll see in every chart from here on.</p>

<value-heatmap></value-heatmap>`,
    );
  },
};
