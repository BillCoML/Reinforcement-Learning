import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const ppoSection03: Section = {
  id: "trpo-algorithm",
  title: "TRPO: The Principled Ancestor",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§3</span>TRPO: The Principled Ancestor</h2>
<p class="tagline"><em>The principled ancestor.</em></p>

<p>TRPO (Trust Region Policy Optimization, Schulman et al. 2015) is the direct implementation
of the trust-region idea from §2. Each iteration:

<ol>
<li><strong>Collect</strong> a batch of trajectories under $\\pi_{\\theta_{\\mathrm{old}}}$.
Compute GAE advantages $\\hat{A}_t$.</li>
<li><strong>Compute</strong> the policy gradient $g = \\nabla_\\theta \\hat{\\mathcal{L}}(\\theta_{\\mathrm{old}})$.</li>
<li><strong>Solve</strong> $F x = g$ using conjugate gradient.
The solution $x = F^{-1}g$ is the <em>natural gradient direction</em>.
(In deep RL, $F$ is too large to invert directly — CG only needs Hessian-vector products.)</li>
<li><strong>Line search</strong> along $x$ for the largest step size $\\alpha$ such that
$D_{\\mathrm{KL}}(\\pi_{\\theta_{\\mathrm{old}}} \\| \\pi_{\\theta_{\\mathrm{old}} + \\alpha x}) \\le \\delta$
<em>and</em> the surrogate has actually improved.</li>
<li><strong>Update:</strong> $\\theta_{\\mathrm{new}} = \\theta_{\\mathrm{old}} + \\alpha x$.</li>
</ol>

<h3>The two heavy pieces</h3>

<p>The <strong>conjugate gradient solve</strong> is the first heavy piece.
For a neural policy with millions of parameters, the Fisher is never materialized;
instead, we compute Hessian-vector products $Fv$ using two backward passes.
The CG solver needs ~10 iterations per update, each requiring a backward pass.
This is expensive compared to the simple gradient descent used in vanilla PG.</p>

<p>The <strong>line search</strong> is the second heavy piece.
After computing the natural gradient direction, TRPO backtracks along it until
both the KL constraint and the objective improvement condition are satisfied.
Typical line searches require 5–10 function evaluations.
Combined with the CG iterations, TRPO can require 15–20× the compute of a vanilla PG step.</p>

<h3>TRPO's behavior on the gridworld</h3>

<p>On the 3×3 tabular gridworld, TRPO with $\\delta = 0.01$ gives convergence behavior
indistinguishable from PPO with $\\varepsilon = 0.2$.
Both algorithms make small, bounded policy updates;
the mechanism is different but the result is similar.
The dramatic difference between TRPO and vanilla PG shows up in continuous-action deep RL,
where TRPO's constrained updates prevent the policy collapse that vanilla PG can exhibit.</p>

<p>We treat TRPO mathematically and give the full algorithm walk-through,
but <strong>the implementation in this lesson is PPO only.</strong>
TRPO appears here as the conceptual ancestor that motivates PPO's design.
The question that spawned PPO: <em>can we get TRPO's stability without TRPO's machinery?</em></p>

<ppo-trpo-step-schematic></ppo-trpo-step-schematic>
`);
  },
};
