import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const tdSection07: Section = {
  id: "td-convergence",
  title: "Convergence Theory: Cashing in Contraction Mappings",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§7</span>Convergence Theory: Cashing in Contraction Mappings</h2>
<p class="tagline"><em>Why TD(0) converges: the Bellman operator is a contraction, and stochastic approximation handles the noise.</em></p>

<p>The convergence proofs for TD methods sit at the intersection of two threads we have
already developed.</p>

<p><strong>Thread 1: the Bellman operator is a $\\gamma$-contraction.</strong> From Lesson 4 we know
that the Bellman expectation operator $\\mathcal{T}^\\pi V$ and the Bellman optimality
operator $\\mathcal{T}V$ are $\\gamma$-contractions in the sup norm $\\|\\cdot\\|_\\infty$.
The Banach fixed-point theorem guarantees unique fixed points $V^\\pi$ and $V^*$
respectively.</p>

<p><strong>Thread 2: stochastic approximation handles noise.</strong> TD methods do not iterate the
Bellman operator exactly. They iterate noisy versions: each update is
$V(s) \\leftarrow V(s) + \\alpha(Y - V(s))$, where $Y$ is a sampled approximation of
$(\\mathcal{T}^\\pi V)(s)$. Under the Robbins-Monro conditions, this stochastic iteration
converges to the same fixed point as the deterministic iteration would.</p>

<p><strong>The combined result (Tsitsiklis 1994).</strong> TD(0) converges almost surely to $V^\\pi$
if every state is visited infinitely often, the Robbins-Monro conditions hold per-state,
and rewards have bounded variance. The proof combines the contraction argument (the
target $\\mathcal{T}^\\pi V$ doesn't shift faster than the estimator can chase it) with
the stochastic approximation argument (noise is averaged out by decaying step sizes).</p>

<p><strong>Q-learning convergence (Watkins and Dayan 1992).</strong> The same machinery works for
Q-learning: the bootstrap target $\\max_{a'} Q(s', a')$ is the Bellman <em>optimality</em>
operator applied to $Q$, also a $\\gamma$-contraction. Q-learning converges almost surely
to $Q^*$ under the analogous conditions.</p>

<h3>Where convergence breaks</h3>

<p>The clean convergence story falls apart in two important settings.</p>

<p>First, <strong>function approximation</strong> (Lesson 9): if $V$ is parameterized by a neural
network and updated by gradient descent, the projected Bellman operator may not be a
contraction in the relevant norm, and "TD divergence" can occur.</p>

<p>Second, <strong>off-policy with importance corrections</strong>: if importance weights are
unbounded, the stochastic approximation noise no longer averages out cleanly. The
"<strong>deadly triad</strong>" of function approximation + bootstrapping + off-policy is precisely
the intersection of these two issues, and it is the central technical challenge of
deep RL. The literature has spent thirty years patching this: target networks
(Mnih et al. 2015), conservative weight constraints, Polyak averaging. Each recovers
some of the tabular convergence guarantees in approximate form.</p>

<div class="callout callout--info">
  <strong>The clean tabular case is a reference point, not a deployment reality.</strong>
  Real systems use function approximation, real data is often off-policy, and the
  convergence proofs do not apply directly. The tabular case is what we know for sure;
  everything else is "approximately, with these caveats."
</div>

<div class="component-host">
  <maximization-bias-demo></maximization-bias-demo>
</div>
`);
  },
};
