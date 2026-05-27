import type { Section } from '../section';
import { sectionFromHTML } from '../section';
import { forwardLink } from '../../components/CrosslinkCallout';

export const dpSection06: Section = {
  id: 'asynchronous-dp',
  title: 'Asynchronous Dynamic Programming',
  build() {
    return sectionFromHTML(this.id, `<h2><span class="sec-num">§6</span>Asynchronous Dynamic Programming</h2>
<p class="tagline">Update in any order. Use updated values immediately. Save iterations.</p>

<p>So far, every backup has been <strong>synchronous</strong> (Jacobi-style): all states are
updated using the <em>old</em> $V_k$, producing a <em>new</em> $V_{k+1}$ in one batch. This is
conceptually clean but wasteful — after we've updated some states, <em>why not use the
new values</em> when updating others?</p>

<p><strong>Asynchronous DP</strong> updates one state at a time, in some order, <em>in place</em>.
Each update uses whatever values are currently in the array, old or new.
The simplest variant is <strong>Gauss-Seidel value iteration</strong>:</p>

<pre class="dp-pseudocode">Gauss-Seidel VI

  Initialize: V ≡ 0
  loop:
    for s in sweep_order:
      V[s] ← max_a [r(s,a) + γ Σ P(s'|s,a) V[s']]   <span class="dp-comment">in-place: V[s'] may already be updated</span>
    if max change &lt; ε(1−γ)/γ:
      return V</pre>

<p><strong>Convergence.</strong> Async DP converges to $V^*$ as long as <strong>every state is
updated infinitely often</strong>. Sweep order doesn't affect <em>whether</em> you converge —
only <em>how fast</em>.</p>

<h3>Sweep Order Matters — A Lot</h3>

<p>On our gridworld with three sweep orders:</p>

<div class="dp-table-wrap">
<table class="dp-table">
  <thead><tr><th>Sweep order</th><th>Iterations to converge</th></tr></thead>
  <tbody>
    <tr><td>Jacobi (synchronous)</td><td>5</td></tr>
    <tr><td>Forward (s = 0, 1, …, 8)</td><td>5 (no improvement over Jacobi)</td></tr>
    <tr><td><strong>Reverse (s = 8, 7, …, 0)</strong></td><td><strong>2</strong></td></tr>
    <tr><td>Smart (near-goal first)</td><td>2</td></tr>
  </tbody>
</table>
</div>

<p><strong>Reverse sweep is 2.5× faster.</strong> Why? The goal is at state 8 (highest index),
and value <em>propagates from the goal backward</em>. A reverse sweep updates state 8 first,
then state 7 (which immediately sees the updated neighbor via the in-place update), etc.
By the end of the first sweep, most cells already have the value-information that
Jacobi needs additional iterations to propagate.</p>

<p>The forward sweep doesn't gain anything because it propagates information in the
<em>wrong direction</em> — by the time it reaches state 8, the early states have already
been updated with stale information.</p>

<h3>Prioritized Sweeping</h3>

<p>A natural extension: prioritize states whose <em>neighbors recently changed a lot</em>.
Maintain a priority queue keyed by the magnitude of recent neighbor-changes.
This is called <strong>prioritized sweeping</strong> (Moore & Atkeson, 1993) and it's a
precursor to modern priority-based replay buffers.</p>

${forwardLink({
  destination: 'Lesson 7 — Deep Q-Networks',
  html: `<p>Prioritized Experience Replay (Schaul et al. 2015, used in Rainbow DQN, Lesson 7)
    is <em>exactly</em> prioritized sweeping with deep function approximation. The priority
    key is the absolute TD error, which plays the role of "neighbor-change magnitude."
    Same idea, different implementation.</p>`,
})}

<async-sweep-comparator></async-sweep-comparator>`);
  },
};
