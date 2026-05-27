import type { Section } from '../section';
import { sectionFromHTML } from '../section';
import { sidebar } from '../../components/CrosslinkCallout';

export const dpSection04: Section = {
  id: 'policy-iteration',
  title: 'Policy Iteration',
  build() {
    return sectionFromHTML(this.id, `<h2><span class="sec-num">§4</span>Policy Iteration</h2>
<p class="tagline">Alternate. Improve. Converge — in finitely many steps.</p>

<p>Combine §2 and §3 into a single loop:</p>

<pre class="dp-pseudocode">Policy Iteration

  Initialize: π₀ arbitrary (e.g. uniform random)
  for k = 0, 1, 2, ...
    V_{π_k} ← solve V = R^π + γ P^π V        <span class="dp-comment">// policy evaluation (full)</span>
    π_{k+1}(s) ← argmax_a [r(s,a) + γ Σ P(s'|s,a) V_{π_k}(s')]  <span class="dp-comment">// improvement</span>
    if π_{k+1} == π_k:
      return (π_k, V_{π_k})                   <span class="dp-comment">// converged</span></pre>

<h3>Properties</h3>

<ol>
  <li><strong>Monotonic improvement.</strong> By the policy improvement theorem,
    $V^{\\pi_{k+1}} \\geq V^{\\pi_k}$ pointwise. Iterates ratchet <em>up</em>.</li>
  <li><strong>Finite convergence.</strong> There are only finitely many deterministic
    policies ($|\\mathcal{A}|^{|\\mathcal{S}|}$ of them). Strict improvement visits each
    at most once. Hence PI terminates — in practice often in $O(|\\mathcal{S}|)$ steps
    or fewer.</li>
  <li><strong>Optimality at termination.</strong> When $\\pi_{k+1} = \\pi_k$, the policy is
    greedy w.r.t. its own value function, so it satisfies the Bellman
    <em>optimality</em> equation. By uniqueness of the fixed point, $V^{\\pi_k} = V^*$.</li>
</ol>

<h3>Numerical Trace — 3 Iterations on Our Gridworld</h3>

<p><strong>Iteration 0.</strong> Evaluate $\\pi_0$ (uniform): $V_{\\pi_0}(0,0) = -0.4205$.
Improve → $\\pi_1$ bounces at (0,0), takes good moves elsewhere.</p>

<p><strong>Iteration 1.</strong> Evaluate $\\pi_1$: $V_{\\pi_1}(0,0) = 0.000$ — because $\\pi_1$
at (0,0) bounces <em>forever</em>, earning nothing. Other cells improve substantially:
$V_{\\pi_1}(2,1) = 1.0$, $V_{\\pi_1}(1,2) = 1.0$, $V_{\\pi_1}(2,0) = 0.9$, etc.
Improve → $\\pi_2$: now (0,0) sees positive-valued neighbors (0.81), so greedy picks
Right or Down. <em>Unstuck.</em></p>

<p><strong>Iteration 2.</strong> Evaluate $\\pi_2$: $V_{\\pi_2}(0,0) = 0.729$.
Improve → $\\pi_3 = \\pi_2$. <strong>Converged.</strong></p>

<p>The monotone sequence at (0,0): $-0.4205 \\to 0.000 \\to 0.729$. The <em>pathological</em>
middle iteration — where the policy gets stuck bouncing at (0,0) — is the most
pedagogically interesting moment. It shows that one improvement step can produce a
value that <em>looks</em> strange (V(0,0) = 0, stuck forever) while still satisfying the
improvement theorem (V=0 > −0.4205). The next improvement immediately breaks the
stuckness.</p>

${sidebar('Connection to EM', `<p>Policy iteration has the same alternating-maximization structure as EM:
  evaluation = E-step (compute posterior / value), improvement = M-step (update
  parameter / policy). Both have monotone-improvement guarantees via similar
  telescoping arguments. The PI improvement theorem is the RL analog of EM's
  monotone-likelihood theorem.</p>`)}

<h3>Modified Policy Iteration</h3>

<p>Instead of <em>fully</em> evaluating $\\pi_k$ (running PE to convergence), do only $m$ PE
iterations before improving. With $m = 1$, this is <em>literally</em> value iteration
(next section). With $m = \\infty$, it's full PI. Modified PI with intermediate $m$
(often $m = 5$ or $m = 10$) is faster than full PI for large state spaces.</p>

<policy-iteration-trace></policy-iteration-trace>`);
  },
};
