import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const maxentSection07: Section = {
  id: "rl-as-inference",
  title: "RL as Inference",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§7</span>RL as Inference</h2>
<p class="tagline"><em>The soft-optimal policy is a posterior.</em></p>

<p>We end the theoretical content with a different lens on the same math.
The construction is due to Toussaint, Ziebart, and Levine; the most accessible
exposition is Levine 2018 (arXiv:1805.00909).</p>

<p>Define an <strong>optimality variable</strong> $O_t \\in \\{0,1\\}$ at each time step.
The probabilistic model is:</p>

<ul>
<li>States evolve as in the MDP: $p(s_{t+1} | s_t, a_t)$ from the dynamics.</li>
<li>Actions are drawn from a uniform "prior" policy: $p(a_t | s_t) = 1/|A|$.</li>
<li>The optimality variable is observed:
  $p(O_t = 1 | s_t, a_t) = \\exp(r(s_t, a_t) / \\alpha)$.</li>
</ul>

<p>Now we condition on observing $O_{0:\\infty} = 1$ — "the trajectory is optimal" —
and ask for the posterior over actions:</p>

$$p(a_t | s_t, O_{t:\\infty} = 1).$$

<p>By standard forward-backward inference on this graphical model, the answer is
<em>exactly</em> the Boltzmann policy from §4:</p>

$$p(a_t | s_t, O_{t:\\infty} = 1) = \\frac{\\exp(Q_\\alpha^*(s_t, a_t) / \\alpha)}{Z_\\alpha(s_t)}.$$

<p>The "backward message" $\\beta_t(s)$ corresponds to $\\exp(V_\\alpha^*(s)/\\alpha)$;
the soft Bellman equation is the backward recursion for this message; the partition
function $Z_\\alpha(s)$ is the marginal probability of optimality from state $s$.
Every quantity from soft VI corresponds to a quantity in the inference view.</p>

<h3>What this view buys us</h3>

<p><strong>A clean Bayesian story.</strong> "The agent is doing inference about what
action to take, given that the trajectory is optimal." For people from a probabilistic-ML
background, this is the natural framing. Mechanistically it is the same algorithm.</p>

<p><strong>Variational connections.</strong> Approximating the posterior with a
parameterized policy $\\pi_\\theta(a|s)$ and minimizing
$D_\\mathrm{KL}(\\pi_\\theta \\| p(\\cdot|s, O=1))$ gives a variational objective.
Maximizing the ELBO of this objective is equivalent to the entropy-regularized RL
objective. SAC's policy gradient (Lesson 13) can be derived from this ELBO directly.</p>

<p><strong>Soft Q as a log-posterior.</strong> $Q_\\alpha^*(s,a)/\\alpha$ behaves like a
log-posterior over actions — high-Q actions are "more likely" under the posterior.
This explains why the Boltzmann form is a softmax over $Q/\\alpha$: it is a posterior.</p>

<h3>Temperature as Bayesian confidence</h3>

<p>In the inference view, $\\alpha$ is the <strong>inverse confidence</strong> with which
we believe the trajectory is optimal. Small $\\alpha$ means "almost surely optimal" —
the likelihood sharpens to a delta on the highest-Q action. Large $\\alpha$ means
"weakly optimal" — the likelihood is nearly flat, and the posterior defaults to the
prior (uniform). The same temperature, but interpreted as a Bayesian belief rather
than a Lagrange multiplier.</p>

<p>The visualization shows the probabilistic graphical model: states, actions, and
optimality variables connected by the conditional dependencies above. Hover on any
node type to see the conditional distribution it encodes.</p>

<inference-graph></inference-graph>
`);
  },
};
