import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const tdSection06: Section = {
  id: "td-lambda",
  title: "TD(λ) and Eligibility Traces",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§6</span>TD($\\lambda$) and Eligibility Traces</h2>
<p class="tagline"><em>A geometric mixture of all n-step returns. Implemented efficiently with eligibility traces.</em></p>

<p>n-step TD with a fixed $n$ picks a particular point on the bias-variance trade-off
curve. <strong>TD($\\lambda$)</strong> takes a different approach: instead of picking one $n$, take a
weighted geometric average of <em>all</em> $n$-step returns. The <strong>$\\lambda$-return</strong> is</p>

$$G_t^\\lambda \\;:=\\; (1 - \\lambda) \\sum_{n=1}^\\infty \\lambda^{n-1} G_t^{(n)},$$

<p>with $\\lambda \\in [0, 1]$. The weights $(1 - \\lambda)\\lambda^{n-1}$ form a geometric
distribution over $n$. With $\\lambda = 0$, all weight is on $n = 1$: TD(0). With
$\\lambda = 1$, all weight goes to $n = \\infty$: MC. For intermediate $\\lambda$, the
$\\lambda$-return mixes near and far bootstraps, with farther bootstraps weighted less.</p>

<h3>The backward view: eligibility traces</h3>

<p>TD($\\lambda$) can be implemented <strong>online</strong> using <strong>eligibility traces</strong>. The trace
$e_t(s)$ measures how recently and often each state has been visited:</p>

$$e_t(s) \\;:=\\; \\begin{cases}
  \\gamma\\lambda\\, e_{t-1}(s) + 1 & \\text{if } s = s_t \\\\
  \\gamma\\lambda\\, e_{t-1}(s)     & \\text{otherwise}
\\end{cases}$$

<p>The trace decays by $\\gamma\\lambda$ at every step and bumps by 1 every time its state
is visited. The TD($\\lambda$) update is then</p>

$$V(s) \\;\\leftarrow\\; V(s) + \\alpha\\, \\delta_t\\, e_t(s) \\quad \\forall s,$$

<p>where $\\delta_t = r_{t+1} + \\gamma V(s_{t+1}) - V(s_t)$ is the one-step TD error.
Every state updates proportionally to its current trace. The trace concentrates updates
on recently-visited states and gradually distributes them backward through the trajectory.</p>

<p><strong>Traces reset to zero at the start of each episode.</strong> Letting traces persist across
episodes is a common bug that causes "ghost updates" to states not visited in the
current episode and produces strange convergence behavior.</p>

<h3>TD($\\lambda$) as a unifying framework</h3>

<p>The $\\lambda$ parameter unifies TD(0) and MC into a single algorithm with a single dial.
$\\lambda = 0$ → TD(0); $\\lambda = 1$ → MC; $\\lambda = 0.5$ → a practical sweet spot.
The trace-based implementation gives $O(|S|)$ work per step (updating every state's
trace and value once), the same order as TD(0) — there is no extra computational cost
to using $\\lambda > 0$.</p>

<p>The $\\lambda$-return generalizes to action-values, giving SARSA($\\lambda$), and to
off-policy settings with importance corrections, giving the family of off-policy
TD($\\lambda$) algorithms (Retrace, V-trace) that underlie IMPALA and related systems.
These appear in Lessons 9 and 15.</p>

<div class="component-host">
  <td-lambda-trace-visualizer></td-lambda-trace-visualizer>
</div>
`);
  },
};
