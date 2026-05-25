import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const mcSection01: Section = {
  id: "markov-property",
  title: "The Markov Property",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§1</span>The Markov Property</h2>
<p class="tagline">Memorylessness, with one caveat.</p>

<p>Imagine tracking the weather day by day in three categories: sunny, cloudy,
rainy. We notice an empirical pattern. If today is sunny, tomorrow is sunny
with probability 0.7, cloudy with 0.2, rainy with 0.1. If today is cloudy,
tomorrow is sunny with 0.3, cloudy with 0.4, rainy with 0.3. Similar entries
hold for rainy. The full table:</p>

<table class="numeric">
  <thead>
    <tr><th></th><th>→ sunny</th><th>→ cloudy</th><th>→ rainy</th></tr>
  </thead>
  <tbody>
    <tr><td><strong>from sunny</strong></td><td>0.70</td><td>0.20</td><td>0.10</td></tr>
    <tr><td><strong>from cloudy</strong></td><td>0.30</td><td>0.40</td><td>0.30</td></tr>
    <tr><td><strong>from rainy</strong></td><td>0.20</td><td>0.30</td><td>0.50</td></tr>
  </tbody>
</table>

<p>This is a <strong>transition matrix</strong> — let's call it $P$. The entry $P_{ij}$ is
$\\Pr(\\text{next state} = j \\mid \\text{current state} = i)$. By construction, rows sum to one.
Notice something we've quietly assumed: tomorrow's weather depends <em>only on
today's</em>, not on the entire past history. This is the <strong>Markov property</strong>:</p>

$$\\boxed{\\Pr(X_{t+1} = j \\mid X_t = i,\\, X_{t-1}, X_{t-2}, \\ldots) \\;=\\; \\Pr(X_{t+1} = j \\mid X_t = i)}$$

<p>You may rightly object that real weather isn't Markov — a week of sun
genuinely raises the chance of a heat wave above what "today is sunny" alone
predicts. The Markov property is a <strong>modeling assumption</strong>, not a physical
truth. Its value is mathematical tractability: when the assumption holds (or
approximately holds), an enormous toolkit becomes available. Most of this
lesson is that toolkit.</p>

<p>A <strong>Markov chain</strong> is a stochastic process $(X_0, X_1, X_2, \\ldots)$ taking
values in a state space $\\mathcal{S}$ that satisfies the Markov property. In
this lesson we restrict to <strong>finite, time-homogeneous, discrete-time</strong> chains:</p>

<ul>
  <li><em>Finite:</em> $|\\mathcal{S}| = K < \\infty$. We index states $1, \\ldots, K$.</li>
  <li><em>Time-homogeneous:</em> the transition probabilities $P_{ij}$ don't depend on
  the time $t$.</li>
  <li><em>Discrete-time:</em> steps are integers $t = 0, 1, 2, \\ldots$.</li>
</ul>

<p>Each of these can be relaxed (countable state spaces, time-inhomogeneous
chains, continuous-time chains, controlled chains where transitions depend
on an action) — and in fact Lesson 2's MDP is exactly "finite Markov chain
+ a controllable action at each step." Every restriction we make here gets
loosened later in a principled way.</p>

<p><strong>Initial distribution.</strong> A chain is fully specified by $(P, \\mu_0)$ where
$\\mu_0$ is the distribution over the starting state $X_0$. From these two
ingredients we can compute the distribution of $X_t$ for any $t$. We'll see
how in §2.</p>

<figure class="numeric">
<p><strong>Numerical example (pre-verified).</strong> With $P$ as above and $\\mu_0 = (1, 0, 0)$
(start sunny), the probability of "sunny on day 2" is</p>

$$\\Pr(X_2 = \\text{sunny}) \\;=\\; \\sum_k P_{0k} P_{k0}
\\;=\\; 0.7 \\cdot 0.7 + 0.2 \\cdot 0.3 + 0.1 \\cdot 0.2
\\;=\\; \\mathbf{0.57}.$$

<p>This is just $(P^2)_{00}$, the top-left entry of $P$ squared. We'll formalize
this in the next section.</p>
</figure>

<p>Run the chain below. "Step" samples one transition and the marker hops; the
bars track how often each state has been visited. Run it long enough and the
visit frequencies settle near $(0.457, 0.283, 0.261)$ — the stationary
distribution we'll derive in §4.</p>

<weather-chain-explorer></weather-chain-explorer>`,
    );
  },
};
