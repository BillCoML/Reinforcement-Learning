import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const tdSection01: Section = {
  id: "bootstrap",
  title: "From Monte Carlo to TD: The Bootstrap Move",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§1</span>From Monte Carlo to TD: The Bootstrap Move</h2>
<p class="tagline"><em>Don't wait for the end of the episode. Plug in your current estimate.</em></p>

<p>Lesson 7 left us with the Monte Carlo policy-evaluation algorithm: sample an episode
under $\\pi$, compute the return $G_t$ at every step, and use $G_t$ as a sample of
$V^\\pi(s_t)$. The algorithm's defining property is that it <em>waits</em> for the episode to
end before it can compute any target. The full return is needed; partial returns are not used.</p>

<p>There is a different choice. Suppose we want to estimate $V^\\pi(s_t)$ at the moment we
take a single step — observe reward $r_{t+1}$ and next state $s_{t+1}$ — and we are not
willing to wait for the rest of the episode. We could use the <strong>Bellman expectation
equation</strong> as a guide. The equation says</p>

$$V^\\pi(s) \\;=\\; \\mathbb{E}_\\pi\\!\\left[ R_{t+1} + \\gamma V^\\pi(S_{t+1}) \\,\\bigg|\\, S_t = s \\right].$$

<p>The right-hand side splits the unknown $V^\\pi(s)$ into two pieces: the immediate reward
$R_{t+1}$, which we just observed; and the value of the next state $V^\\pi(S_{t+1})$,
which we do not know. But we have a <strong>current estimate</strong>
$\\hat V^\\pi(s_{t+1})$. If we substitute that estimate for the true $V^\\pi(s_{t+1})$,
we get a usable target:</p>

$$\\text{TD target:}\\quad r_{t+1} + \\gamma \\hat V^\\pi(s_{t+1}).$$

<p>This is the <strong>bootstrap</strong> move. We are using our own current estimate
$\\hat V^\\pi(s')$ as a stand-in for the unknown true $V^\\pi(s')$, which lets us turn a
one-step observation into an update target. The update rule is then</p>

$$\\hat V^\\pi(s_t) \\;\\leftarrow\\; \\hat V^\\pi(s_t) + \\alpha \\left[
  \\underbrace{r_{t+1} + \\gamma \\hat V^\\pi(s_{t+1})}_{\\text{TD target}} - \\hat V^\\pi(s_t)
\\right].$$

<p>The quantity in brackets is the <strong>TD error</strong>, traditionally written</p>

$$\\delta_t \\;:=\\; r_{t+1} + \\gamma \\hat V^\\pi(s_{t+1}) - \\hat V^\\pi(s_t).$$

<p>It measures how surprising the one-step observation was relative to the current estimate.
The update nudges $\\hat V^\\pi(s_t)$ in the direction that would reduce that surprise.</p>

<h3>The trade-off, viewed structurally</h3>

<p>MC's target $G_t$ is a sample of $V^\\pi(s_t)$: it is unbiased, but its variance comes
from the entire trajectory's randomness — many rewards, many transitions, all summed.
TD's target $r_{t+1} + \\gamma \\hat V^\\pi(s_{t+1})$ uses only one reward and one
bootstrap, so its variance is much smaller per step; but the bootstrap is <em>biased</em>
— $\\hat V^\\pi$ is not yet correct, and that incorrectness propagates into every TD target.</p>

<p>Two large structural consequences flow from this. First, TD updates can happen
<strong>online</strong>: the moment we see $(s, a, r, s')$, we can update. We do not need to
wait for the episode to end. Second, TD applies to <strong>continuing tasks</strong> with no
terminal state at all. Pure MC fundamentally cannot: there is no episode end at which
to compute a return. TD's bootstrap lets us learn value functions for systems that go
forever.</p>

<p>The variance reduction is real but <strong>horizon-dependent</strong>. On long-horizon problems
with many per-step rewards, the MC variance grows with horizon and TD's per-step
variance does not — TD wins decisively. On short-horizon problems, the MC variance is
small to begin with, and TD's bootstrap bias can be the dominant error source.
Lesson 7's gridworld is in the second regime; we will see the empirical consequence
in Section 2.</p>

<div class="crosslink-callout">
  <strong>Forward link · Function approximation.</strong>
  The bootstrap move generalizes immediately to function approximation (Lesson 9): replace
  $\\hat V^\\pi(s')$ with $V_{\\theta^-}(s')$, where $\\theta^-$ is a periodically-updated
  target network. The same bootstrap, now with the target network providing the "current
  estimate." The bias-variance trade-off survives intact; the "deadly triad"
  (function approximation + bootstrapping + off-policy) is the stability cost.
</div>

<div class="component-host">
  <bootstrap-move></bootstrap-move>
</div>
`);
  },
};
