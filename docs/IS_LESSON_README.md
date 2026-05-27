# Lesson 6 — Importance Sampling

> **The trick for evaluating one policy using data from another.** A two-line
> algebraic identity that powers off-policy Monte Carlo (Lesson 7), off-policy
> Temporal-Difference learning (Lesson 8), the policy ratio in PPO (Lesson 11),
> and the entire offline-RL story (Lesson 15). The identity is deceptively
> simple. The whole game is variance: when does the importance-weighted
> estimator have a reasonable chance of producing the right answer, and when
> does it blow up?

> **Where this slots in.** Between Lesson 5 (Dynamic Programming) and Lesson 7
> (Monte Carlo). The role is that of a small, focused tool lesson — similar in
> scope to Lesson 4 (Contractions). It does not introduce any new RL setting;
> it introduces one statistical tool that the next four lessons need. The
> implementation effort is light because most of the code is short estimator
> functions and visualizations that reuse the gridworld machinery already built
> in Lessons 3 and 5.

---

## 0. Pedagogical Philosophy

Five commitments specific to this lesson.

The first commitment is that the identity is trivial and the variance is
everything. We will derive the importance sampling identity in roughly a page,
showing both the ordinary and the weighted estimators. The remaining ninety
percent of the lesson concerns variance: when it is finite, when it is
infinite, how to diagnose problems, and what the variance grows like as a
function of horizon. This ratio of attention is intentional. The variance
problem is what makes off-policy reinforcement learning hard, and learners who
absorb only the identity without absorbing the variance gotchas leave
unprepared for what comes next.

The second commitment is to show the failure modes explicitly. When the
finite-variance condition fails, the estimator behaves pathologically. We
demonstrate this on a Gaussian example (proposal narrower than target leads to
infinite variance) and on the gridworld (uniform behavior policy estimating a
deterministic target policy gives roughly one non-zero contribution per two
hundred and fifty-six samples). Naming the failure mode while showing it
produces understanding that abstract theorems cannot.

The third commitment is that the centerpiece is the reinforcement-learning
application. Importance sampling on real-valued integrals is a fine warm-up,
but the lesson's centerpiece visualization is trajectory importance sampling
on the gridworld, exactly the setup that Lesson 7 will use. We are not
teaching importance sampling as a general statistical tool; we are teaching
the flavor of it that reinforcement learning uses, with the right examples and
the right diagnostics.

The fourth commitment is to keep the lesson short. Total length around nine
hundred lines, five sections, five visualizations. This is a tool lesson, not
a destination. The polish budget should be modest, with the centerpiece
absorbing most of it.

The fifth commitment is to point forward at the lessons that abuse this
machinery. PPO's clipped surrogate objective (Lesson 11) is a heuristic for
controlling importance-ratio variance. Offline RL (Lesson 15) is in many ways
defined by the importance-sampling variance problem. We pre-load both by
naming them explicitly in the closing section.

---

## 1. Tech Stack

The tech stack is identical to all prior lessons. Vite plus TypeScript in
strict mode, KaTeX for math, D3 version 7 for visualizations, the `ml-matrix`
package for linear algebra (lightly used here), and Vitest for tests. The
trajectory sampling on the gridworld reuses Lesson 3's MDP machinery; the
ground-truth value function used as a reference line reuses Lesson 5's
`policyEvaluationExact`. The estimator code itself is short, around eighty
lines of TypeScript.

A small Python script, `scripts/is_traces.py` at around fifty lines,
pre-computes the Gaussian variance sweep and the gridworld trajectory
statistics. Its outputs land as JSON in `public/data/is/` for instant initial
render of the visualizations before the user interacts.

---

## 2. Visual and Aesthetic Direction

The curriculum aesthetic continues unchanged. The lesson adds a small set of
tokens specific to the importance-sampling theme.

```css
:root {
  /* Distribution roles */
  --is-target:        #2563eb;   /* blue-600   | target distribution p */
  --is-proposal:      #ea580c;   /* orange-600 | proposal or behavior q */
  --is-weight:        #15803d;   /* green-700  | the importance weight w */
  --is-truth:         #1c1e22;   /* black ref  | true expectation */

  /* Estimator semantics */
  --is-ordinary:      #0e7490;   /* cyan-700   | unbiased, possibly high-variance */
  --is-weighted:      #6d28d9;   /* violet-700 | biased, lower-variance */
  --is-explosion:     #b91c1c;   /* red-700    | variance blow-up zone */
}
```

The blue-and-orange pairing for target-and-proposal carries forward to PPO's
old-versus-new policy ratio in Lesson 11 and to the behavior-versus-target
distinction in Lesson 15's offline-RL material. The cyan-and-violet pairing
for ordinary-versus-weighted rhymes with the algorithm-signature colors from
the Bandits lesson, where cyan was UCB (estimation with uncertainty) and
violet was Thompson (the principled-but-biased Bayesian alternative). The
same intuition transfers: cyan is the unbiased estimator with the variance
problem, violet is the biased estimator with the variance solution.

---

## 3. Lesson Metadata

```ts
export const lessonMeta: LessonMeta = {
  id: "importance-sampling",
  title: "Importance Sampling",
  subtitle: "Evaluating one distribution with samples from another",
  tier: 1,
  difficulty: 3,
  estimatedReadMinutes: 40,
  role: "critical-path",
  prerequisites: [
    { external: true, label: "Probability: expectations, variance, change of variables" },
    { lesson: "mdps", anchor: "state-value-function" },
    { lesson: "dynamic-programming", anchor: "iterative-policy-evaluation" },
  ],
  exportedAnchors: [
    "is-identity",
    "ordinary-is-estimator",
    "weighted-is-estimator",
    "is-variance",
    "infinite-variance-condition",
    "trajectory-is",
    "per-decision-is",
    "effective-sample-size",
  ],
  centerpieceComponent: "TrajectoryISExplorer",
  forwardLinksWhenReady: [
    { to: "monte-carlo",      anchor: "off-policy-mc" },
    { to: "td-learning",      anchor: "off-policy-td" },
    { to: "trust-region",     anchor: "ppo-clipped-ratio" },
    { to: "offline-rl",       anchor: "distribution-shift" },
  ],
};
```

---

## 4. Section-by-Section Plan

### Section 1 — The Importance Sampling Identity

**Tagline:** *Change of measure with a multiplicative correction.*
**Length:** ~550 words.
**Anchor:** `is-identity`.

---

**Prose:**

Suppose we want to compute or estimate `\mathbb{E}_p[f(X)]`, the expectation
of some function `f` of a random variable distributed according to `p`. The
obvious approach is to sample `X_i \sim p`, compute each `f(X_i)`, and average.
By the law of large numbers,

$$
\frac{1}{N} \sum_{i=1}^N f(X_i) \;\xrightarrow{a.s.}\; \mathbb{E}_p[f(X)].
$$

Sometimes we cannot sample from `p`. The reasons vary, but they include: `p`
is a target policy we want to evaluate, and the only trajectories we have
available were collected under some other behavior policy; `p` is
computationally expensive to sample from, while a related proposal `q` is
cheap; `p` is defined only up to a normalization constant (this case will
recur in Lesson 12 when we look at energy-based policies). In each of these
cases we have samples from a different distribution `q` and want to use them
to estimate an expectation under `p`.

The **importance sampling identity** is the elementary algebraic fact that
makes this possible. For any two probability distributions `p` and `q` such
that `q(x) > 0` wherever `p(x) f(x) \neq 0`,

$$
\boxed{\mathbb{E}_p[f(X)] \;=\; \int f(x) \, p(x) \, dx \;=\; \int f(x) \, \frac{p(x)}{q(x)} \, q(x) \, dx \;=\; \mathbb{E}_q\!\left[ f(X) \, \frac{p(X)}{q(X)} \right].}
$$

The trick is a multiplication-and-division by `q(x)`. The new integrand is
`f(x) \cdot w(x)` evaluated under `q`, where `w(x) := p(x)/q(x)` is called the
**importance weight**. The identity says that we can compute an expectation
under `p` by sampling under `q` and reweighting each sample by `w`.

The single requirement is that `q` "cover" the support of `p` on the regions
where `f` is non-zero. Formally: `q(x) > 0` whenever `p(x) f(x) \neq 0`. If
this fails — if there are regions where `p` has mass but `q` does not — then
the IS identity is technically infinite, because we would be dividing by zero
on a non-negligible region. In reinforcement learning, this requirement is
called the **coverage condition**, and it has a clean policy-language
translation: a behavior policy that assigns zero probability to an action
that the target policy might take cannot be used to evaluate that target
policy. Coverage is the single most important assumption in off-policy
methods.

---

**The two canonical estimators.** Given `N` samples `X_1, \ldots, X_N \sim q`,
there are two natural ways to turn the identity into an estimator. The first
is the **ordinary importance sampling estimator**:

$$
\boxed{\hat\mu_{\text{ord}} \;:=\; \frac{1}{N} \sum_{i=1}^N w(X_i) \, f(X_i).}
$$

This estimator is **unbiased** in a strong sense: by linearity of expectation
under `q`, we have `\mathbb{E}_q[\hat\mu_{\text{ord}}] = \mathbb{E}_p[f(X)]`
exactly, with no error term that vanishes only asymptotically.

The second estimator is the **weighted** or **self-normalized importance
sampling estimator**:

$$
\boxed{\hat\mu_{\text{wt}} \;:=\; \frac{\sum_{i=1}^N w(X_i) \, f(X_i)}{\sum_{i=1}^N w(X_i)}.}
$$

The denominator is an estimator of `\mathbb{E}_q[w(X)] = \int (p/q) q = \int p = 1`,
so for large `N` the denominator is close to 1 and the two estimators agree.
For small or moderate `N`, the denominator can differ noticeably from 1, and
the two estimators behave differently. This difference is what Section 2 is
about.

---

**Visualization V1 — IS Identity Demonstrator.** A side-by-side comparison
panel. On the left, a histogram of samples from `p = \mathcal{N}(0, 1)` and
the empirical average of `f(X) = X^2`. The histogram is rendered in the blue
of `--is-target`. On the right, a histogram of samples from
`q = \mathcal{N}(0, \sigma_q)` with a slider for `\sigma_q \in [0.3, 3]`,
rendered in the orange of `--is-proposal`. Each bar in the right histogram is
re-scaled by `w(x_i)`, so that the bar heights show the
importance-reweighted contribution rather than the raw count. Above each
histogram, the running estimate of `\mathbb{E}_p[f(X)]`: the empirical mean
from `p` directly on the left, the ordinary IS estimator from `q` on the
right. A horizontal green line at value `1.0` shows the true expectation.
A small inset on the right panel shows the weight histogram so the learner
can see `w(x)` concretely. Slider for sample count `N \in [50, 5000]`. The
pedagogical point is that both estimators converge to the same value but
the paths each takes through the data are different. Width 800, height 400.

---

### Section 2 — Variance: When IS Works and When It Fails

**Tagline:** *The estimator is unbiased. The variance is the problem.*
**Length:** ~800 words.
**Anchor:** `is-variance`.

---

**Prose:**

The ordinary IS estimator is unbiased. So what is the catch? Variance. The
variance of `\hat\mu_{\text{ord}}` is

$$
\text{Var}_q[\hat\mu_{\text{ord}}] \;=\; \frac{1}{N} \, \text{Var}_q[w(X) \, f(X)] \;=\; \frac{1}{N} \left( \mathbb{E}_q[w(X)^2 f(X)^2] - \mathbb{E}_p[f(X)]^2 \right).
$$

If `\mathbb{E}_q[w(X)^2 f(X)^2]` is finite, the IS estimator has finite
variance, the law of large numbers applies in the usual way, and the
estimator converges at the standard `1/\sqrt{N}` rate. If
`\mathbb{E}_q[w(X)^2 f(X)^2]` is infinite, the IS estimator has infinite
variance. The central limit theorem fails, sample averages can wander
arbitrarily far from the true expectation, and no amount of additional
sampling can rescue the situation in the way we are used to.

The condition for finite variance can be written as

$$
\mathbb{E}_q[w(X)^2 f(X)^2] \;=\; \int \frac{p(x)^2}{q(x)} f(x)^2 \, dx \;<\; \infty.
$$

In words: the importance weight `p(x)/q(x)` and the integrand `f(x)` should
not simultaneously be large on regions where `q` has appreciable mass. The
single most dangerous case is when `q` has *lighter tails* than `p`. In the
tails of `p`, the density of `q` becomes tiny, while `p` still has meaningful
mass; the weight `w(x) = p(x)/q(x)` then explodes. If `f(x)` also grows in
the tails, the product `w(x)^2 f(x)^2` is integrable only if the tails of
`q` are heavy enough.

A useful rule of thumb is that the proposal `q` should always have heavier
tails than the target `p`. Using a wider Gaussian to estimate a narrower
Gaussian is safe. Using a narrower Gaussian to estimate a wider one is a
recipe for variance blow-up. The variance problem here is not a small
numerical issue; it can be catastrophic.

---

**A worked example.** Take `p = \mathcal{N}(0, 1)` and `f(X) = X^2`. The
true expectation is `\mathbb{E}_p[X^2] = 1` (the variance of the standard
normal). We compare three choices of proposal `q = \mathcal{N}(0, \sigma_q^2)`,
running fifty trials each with `N = 1000` samples.

| `\sigma_q` | Ordinary IS standard deviation | Weighted IS standard deviation | Notes                              |
|-----------:|-------------------------------:|-------------------------------:|------------------------------------|
| 1.0        | 0.039                          | 0.039                          | no IS needed; `q = p`              |
| 2.0        | 0.021                          | 0.033                          | wider proposal; variance drops     |
| 0.5        | 1.460                          | 0.479                          | narrower proposal; near-infinite-variance regime |

The theoretical threshold for finite variance in this setup is
`\sigma_q > 1/\sqrt{2} \approx 0.707`. With `\sigma_q = 0.5`, the IS
variance is mathematically infinite, and the empirical estimator is
correspondingly unstable. The ordinary IS standard deviation of `1.46` is
larger than the true value of `1.0`, which is the definition of useless.

Notice that the weighted IS estimator is also unstable in the
`\sigma_q = 0.5` case, but markedly less so. Weighted IS is biased (the
denominator introduces a correlation with the numerator), but its bias is
typically `O(1/N)` and vanishes asymptotically; in exchange it often
delivers a constant-factor variance reduction. In the `\sigma_q = 0.5` case
the variance reduction is roughly three-fold.

---

**Ordinary versus weighted: the trade-off.**

|                                | Ordinary IS                          | Weighted IS                  |
|--------------------------------|--------------------------------------|------------------------------|
| Bias                           | exactly zero                          | `O(1/N)`, vanishes asymptotically |
| Variance                       | possibly infinite                    | usually finite, often much smaller |
| Behavior with one large weight | dominated by that one sample         | dominated by that one sample but rescaled |
| Use when                       | small variance, low-stakes bias       | large variance, can tolerate bias |

In practice, weighted IS is strongly preferred in reinforcement learning
settings, even though it is biased. The bias vanishes asymptotically; the
variance reduction is a constant factor that can be ten to a hundred times.
For sample-limited problems, that factor matters more than the asymptotic
bias.

---

**Effective sample size.** A useful diagnostic for whether your IS is
producing trustworthy estimates is the effective sample size:

$$
\boxed{N_{\text{eff}} \;:=\; \frac{\left( \sum_i w(X_i) \right)^2}{\sum_i w(X_i)^2}.}
$$

When all weights are equal, `N_{\text{eff}} = N`. When one sample has all
the weight, `N_{\text{eff}} = 1`. Values in between measure roughly how many
of your `N` samples are effectively contributing to the estimator. The
practical rule of thumb is that if `N_{\text{eff}} / N` falls below ten
percent, your IS estimator is on shaky ground; ninety percent or more of
your samples are being wasted, and the remaining samples may not be
sufficient to estimate the quantity of interest at the required precision.

> **Forward link to PPO** — The clipped surrogate objective in PPO
> (Lesson 11), which has the form
> `\min(r_t A_t, \text{clip}(r_t, 1 - \epsilon, 1 + \epsilon) A_t)`, is a
> heuristic that keeps importance ratios in `[1 - \epsilon, 1 + \epsilon]`
> and thereby controls the IS variance. It is not theoretically principled
> in the way TRPO's KL constraint is, but it works empirically because
> uncontrolled IS variance is the dominant failure mode of off-policy policy
> optimization. Recognizing the clipping for what it is — variance control
> rather than a "weird trick" — is one of the payoffs of this lesson.

---

**Visualization V2 — Variance Explorer.** A control panel with two sliders,
`\sigma_q` and `N`. In the center, a histogram of weights `w(X_i)` for
samples `X_i \sim q`. As `\sigma_q` decreases below `1/\sqrt{2}`, the
histogram develops a long right tail showing the variance blow-up. To the
right, a running-average plot showing `\hat\mu_{\text{ord}}` as samples
accumulate; for `\sigma_q < 1/\sqrt{2}`, the running average jumps every
time an outlier weight arrives, a visible signature of infinite-variance
behavior. Below, live readouts: estimator value, sample variance, effective
sample size as a fraction of `N`. The `\sigma_q` slider has a threshold
marker at `1/\sqrt{2} \approx 0.707`, with a red zone below it labeled
"infinite-variance regime." Width 800, height 440.

---

### Section 3 — Trajectory Importance Sampling in Reinforcement Learning (Centerpiece)

**Tagline:** *Estimate the value of one policy using trajectories sampled under another.*
**Length:** ~900 words.
**Anchor:** `trajectory-is`.

---

**Prose:**

We now arrive at the application that justifies this lesson. Suppose we have
a target policy `\pi_t` and we want to estimate `V^{\pi_t}(s_0)`, the value
of `\pi_t` starting from state `s_0`. We do not have the ability to roll out
`\pi_t` directly. Perhaps we are debugging an algorithm that decides which
policy to deploy and we want to know how well a candidate policy would have
performed before deploying it; perhaps we are in the offline setting of
Lesson 15 where the dataset was collected under some unknown behavior policy
and there is no possibility of new sampling. Whatever the reason, we have
trajectories from a behavior policy `\pi_b`, and we need to extract a
`V^{\pi_t}` estimate from them.

A trajectory `\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \ldots, s_T)` has joint
probability under any policy `\pi`:

$$
\Pr_\pi(\tau) \;=\; \mu_0(s_0) \prod_{t=0}^{T-1} \pi(a_t \mid s_t) \, P(s_{t+1} \mid s_t, a_t).
$$

The return is `G_0 := \sum_{t=0}^{T-1} \gamma^t r_{t+1}`. By the
importance-sampling identity applied to trajectories under `\pi_b` versus
`\pi_t`:

$$
\mathbb{E}_{\pi_t}[G_0] \;=\; \mathbb{E}_{\pi_b}\!\left[ \frac{\Pr_{\pi_t}(\tau)}{\Pr_{\pi_b}(\tau)} G_0 \right].
$$

The ratio of the two trajectory probabilities simplifies in a useful way. The
initial state distribution `\mu_0` and the transition kernel `P` are
properties of the environment, not the policy, so they appear in both numerator
and denominator and cancel. What remains is the product of per-step policy
ratios:

$$
\boxed{\rho_{0:T-1} \;:=\; \prod_{t=0}^{T-1} \frac{\pi_t(a_t \mid s_t)}{\pi_b(a_t \mid s_t)}.}
$$

We call this the **trajectory importance weight**. The identity now reads

$$
\boxed{V^{\pi_t}(s_0) \;=\; \mathbb{E}_{\pi_b}\!\left[ \rho_{0:T-1} \cdot G_0 \right].}
$$

This is trajectory-level importance sampling for reinforcement learning. It
is exact: the expectation on the right equals `V^{\pi_t}(s_0)` precisely.
The estimator that averages this expression across `N` independently sampled
trajectories is unbiased. The variance, however, is where the story begins.

---

**A worked example on the gridworld.** Take the running gridworld from
Lessons 3 through 5. The target policy `\pi_t` is the deterministic optimal
policy (specifically, the one that goes right-right-down-down from the start
state). The behavior policy `\pi_b` is uniform random over the four
actions. The discount factor is `\gamma = 0.9`. The true value
`V^{\pi_t}(0,0)` is `\gamma^3 = 0.729` (we computed this in Lesson 5).

The optimal trajectory from `(0,0)` to the goal under this deterministic
`\pi_t` is unique and has length four. Under the uniform behavior policy,
the probability that a sampled trajectory matches the optimal sequence
step-for-step is `(1/4)^4 = 1/256 \approx 0.0039`. Any deviation from the
optimal action at any step makes the corresponding per-step ratio zero, which
kills the entire trajectory weight by the product structure.

When a trajectory does match, the weight is `(1 / (1/4))^4 = 4^4 = 256`. So
we are computing an estimator where roughly 99.6 percent of trajectories
contribute zero and roughly 0.4 percent contribute `256 \times G_0 = 256 \times 0.729 \approx 186.6`.

The estimator behavior at three sample sizes, averaged over fifty trials:

| `N` (trajectories) | Non-zero / N | Ordinary IS (mean ± SD) | Weighted IS (mean ± SD) |
|------------------:|:------------:|:------------------------|:------------------------|
| 100               | 0.3 / 100    | 0.63 ± 0.96             | 0.23 ± 0.34 (huge bias) |
| 1,000             | 3.7 / 1000   | 0.68 ± 0.38             | 0.70 ± 0.14             |
| 10,000            | 38 / 10000   | 0.71 ± 0.13             | **0.7290 ± 0.0000**     |

The last row deserves attention. At `N = 10000`, weighted IS converges
exactly to `0.7290` with zero sample variance across the fifty trials. Why?
Because all matching trajectories under this configuration are identical:
they all take the same path, the same length, the same return, and have the
same weight. The weighted-IS numerator and denominator are perfectly
proportional, and the ratio is exact. This is a remarkable property unique
to deterministic targets — when `\pi_t` is deterministic, the few
contributing trajectories are all the same trajectory, and weighted IS
becomes perfectly accurate once you have enough hits to overcome the
denominator's discreteness.

For stochastic targets the lesson is less dramatic but the qualitative
picture is the same: weighted IS dominates ordinary IS by a substantial
margin.

---

**The variance of `\rho_{0:T-1}` is the entire problem.** For an episodic
task with horizon `T`, the trajectory weight is a product of `T` independent
random ratios. The variance of a product grows roughly multiplicatively in
`T`. So trajectory IS has variance that grows *exponentially in the horizon*.
This is why off-policy methods become brittle as episodes get longer, and
why we will work hard in Lesson 8 to compute per-step backups (which require
only one ratio at a time and avoid the exponential blow-up).

> **Practical implication.** Off-policy Monte Carlo with a behavior policy
> that barely overlaps with the target policy needs enormous sample sizes.
> In our toy example, getting a reliable estimate required ten thousand
> trajectories. This is a serious limitation. Lessons 8 (TD) and 11 (PPO)
> will partially address it by working at the per-step level rather than at
> the trajectory level, where short ratios are dramatically more stable than
> long products.

---

**Visualization V3 — Trajectory IS Explorer. This is the centerpiece.**

The polish-budget sink of the lesson; allocate roughly two to three days.

A four-panel synchronized layout. Panel A in the top left shows the two
policies as gridworld renderings side by side, the left for `\pi_b` and the
right for `\pi_t`. Each has a dropdown for selecting among uniform random,
deterministic optimal, ε-soft (with a slider for ε), all-down, and all-right.
Below the panel, the pre-computed probability that a sampled trajectory
matches `\pi_t` is displayed prominently in monospace; for the default
config (uniform behavior to deterministic optimal target), this displays as
`1/256 ≈ 0.0039`.

Panel B in the top right shows one sampled trajectory as an animated agent
moving cell to cell on the gridworld. Each step shows the per-step ratio
`\pi_t(a \mid s) / \pi_b(a \mid s)` updating the running weight `\rho_{0:t}`.
For the default configuration, per-step ratios are either zero (target says
no, trajectory rejected) or four (target matches behavior's possible choice
at uniform 1/4 probability). When a ratio is zero, the trajectory's running
weight goes to zero permanently; the animation can stop here, highlighting
the rejection in red. When a ratio is four, the running weight multiplies by
four, highlighted in green. At the end of the trajectory, the total return
`G_0` and the full trajectory weight `\rho` are shown.

Panel C in the bottom left shows the estimator over many sampled
trajectories. A live histogram of weighted contributions `\rho_i \cdot G_{0,i}`
across the samples taken so far; most are at zero, a few at the
characteristic value of approximately 186.6 for the default configuration.
Two running-average lines: the ordinary IS estimator in cyan and the
weighted IS estimator in violet, both converging toward the true
`V^{\pi_t}(s_0)` shown as a horizontal green dashed reference line. Live
readouts: current estimator values, effective sample size as a fraction of
`N`, number of non-zero weights. Buttons to run 100, 1000, or 10000
trajectories at speeds the user controls.

Panel D in the bottom right shows the estimator distribution across fifty
independent trials at the current `N`. Box-and-whisker plots, with one box
for ordinary IS and one for weighted IS. The true value
`V^{\pi_t}(s_0)` is shown as a horizontal reference. Bias and standard
deviation are labeled. This panel illustrates the spec's main numerical
table — it is the empirical proof of the bias-variance trade-off, presented
at the same time the learner has been able to feel the trade-off through
interaction.

Controls: behavior policy and target policy selectors (driving Panel A), an
`N` slider on log scale ranging through 100, 1000, 10000, a `\gamma` slider,
and a reset button.

The visualization should be the moment where the learner feels what
off-policy estimation costs. The transition from "empty histogram with a few
zeros" at `N = 100` to "populated histogram with dozens of identical
non-zero spikes" at `N = 10000` is the entire content of Lessons 7 and 8's
off-policy difficulties, conveyed visually.

Width 960 pixels (centerpiece breakout). Height 720 pixels.

---

### Section 4 — Per-Decision Importance Sampling

**Tagline:** *Each reward only needs the ratios up to that point.*
**Length:** ~500 words.
**Anchor:** `per-decision-is`.

---

**Prose:**

The trajectory IS estimator weights the entire return `G_0 = r_1 + \gamma r_2 + \gamma^2 r_3 + \cdots`
by the full trajectory weight `\rho_{0:T-1}`. But the reward `r_{t+1}` was
determined by the state-action pair `(s_t, a_t)` and the random transition
to `s_{t+1}`. It does not depend on the later actions
`a_{t+1}, a_{t+2}, \ldots, a_{T-1}`. So multiplying `r_{t+1}` by `\rho_{t+1:T-1}`
(the ratios from steps after `t`) is adding noise without adding signal.

The **per-decision importance sampling estimator** weights each reward by
only the ratios up to its own time step:

$$
\boxed{\hat V^{\pi_t}(s_0) \;=\; \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T_i - 1} \gamma^t \, r_{t+1}^{(i)} \, \rho_{0:t-1}^{(i)}.}
$$

We define `\rho_{0:-1} := 1` (the empty product) so the immediate reward
gets weight one. This estimator is still unbiased (each term has the same
expectation under the per-step algebra as the corresponding term in
trajectory IS), but the variance is strictly lower or equal because the
unnecessary noise from future ratios has been removed.

---

**Why doesn't per-decision IS help in our running gridworld example?**

Because the only non-zero reward in our gridworld arrives at the terminal
step. All trajectories that reach the goal do so at exactly `t = T - 1`,
where `\rho_{0:T-2}` equals `\rho_{0:T-1}` (since the next ratio at the
terminal state is trivial). Per-decision IS collapses to trajectory IS in
this special case of sparse-terminal rewards.

For environments with **dense rewards** (rewards at every step), per-decision
IS gives substantial variance reduction. The reduction grows with the
horizon — in horizon-100 settings, ten to a hundred times variance reduction
is typical. Lesson 8 will show this on a dense-reward problem; here the
mechanism is introduced for later use.

---

**Numerical comparison.** On the sparse-reward gridworld with uniform
behavior and deterministic optimal target:

| `N` | Ordinary IS standard deviation | Per-decision IS standard deviation |
|----:|-------------------------------:|----------------------------------:|
| 1,000  | 0.380 | 0.380 (identical) |
| 10,000 | 0.129 | 0.129 (identical) |

Identical, because the sparse-reward structure makes per-decision IS reduce
exactly to trajectory IS. The toggle to a "dense-reward" gridworld variant
(adding a small `-0.01` per step) exposes the per-decision IS advantage
clearly in the visualization.

> **Forward link** — Per-decision IS is the basis of *truncated* importance
> sampling estimators and **V-trace** (the algorithm used by IMPALA from
> Espeholt et al. 2018). V-trace clips per-step ratios at runtime to control
> variance, and is the backbone of large-scale distributed deep RL. We will
> meet it in Lesson 10's discussion of distributed actor-critic methods.

---

**Visualization V4 — Per-Decision IS Step-by-Step.** A trajectory animation
showing the per-step ratio accumulation `\rho_{0:t}` for `t = 0, 1, 2, \ldots`,
side by side with two contribution traces: the trajectory IS contribution
(each reward weighted by the full `\rho_{0:T-1}`) and the per-decision IS
contribution (each reward weighted by its own `\rho_{0:t-1}`). A toggle
switches between the sparse-reward default gridworld and a dense-reward
variant; the dense-reward toggle is where the per-decision IS advantage
becomes visible. Width 800, height 380.

---

### Section 5 — Where You'll See This Again

**Tagline:** *Four downstream lessons depend on this prereq, and one of them gets clipped.*
**Length:** ~400 words.
**Anchor:** `is-forward-links`.

---

**Prose:**

Four subsequent lessons depend directly on this one. Naming them now
pre-loads the recognition that will happen in each lesson when the
importance-sampling machinery first appears.

The first is **Lesson 7 (Monte Carlo)**. Monte Carlo methods sample
trajectories under a fixed policy and estimate the value function by
averaging the returns. The on-policy version requires no importance
sampling, since the trajectories are sampled under the same policy being
evaluated. The off-policy version is exactly the trajectory IS apparatus we
just built, applied directly. Lesson 7's off-policy MC section will reference
the trajectory IS estimator, the weighted IS estimator, and the
effective-sample-size diagnostic by name.

The second is **Lesson 8 (Temporal-Difference Learning)**. TD methods do
one-step backups, replacing trajectories with `(s, a, r, s')` tuples. The
importance ratio for off-policy TD therefore degrades from a length-`T`
product to a single per-step ratio `\pi(a \mid s) / \mu(a \mid s)`. This is
vastly lower variance than trajectory IS but brings its own subtleties (the
deadly triad of bootstrapping plus off-policy plus function approximation,
which we will meet in Lesson 9). Q-learning is a clever scheme that
maximally collapses the IS apparatus by re-deriving the off-policy target as
a max over actions, avoiding the ratio entirely.

The third is **Lesson 11 (Trust Region and Proximal Methods)**. When
optimizing a parametric policy `\pi_\theta`, the surrogate objective uses
trajectories collected under the previous parameter setting and weights them
by the policy ratio `\pi_\theta(a \mid s) / \pi_{\theta_{\text{old}}}(a \mid s)`.
This is importance sampling in disguise. TRPO's KL constraint and PPO's
ratio clipping are both techniques for bounding the IS variance — keeping
the ratio close to 1 so the estimator stays trustworthy. Without this
prereq, both algorithms read like incomprehensible heuristics.

The fourth is **Lesson 15 (Offline RL)**. Offline RL has a fixed dataset
collected under some unknown behavior policy and the goal is to evaluate or
improve a different policy. Evaluating any policy requires importance
sampling, and the variance problem is catastrophic in offline settings.
Almost the entire offline-RL toolkit (conservative Q-learning, behavior
regularization, density-ratio estimation, BCQ's action constraint) is
working around the IS variance in some way. Lesson 15 will spend a section
diagnosing exactly which IS variance issue each method targets.

---

**Visualization V5 — Roadmap Mini.** The curriculum's lesson-graph thumbnail
with Importance Sampling now marked as shipped. Outgoing arrows from this
lesson go to Lesson 7 (Monte Carlo), Lesson 8 (TD), Lesson 11 (TRPO/PPO),
and Lesson 15 (Offline RL). Each arrow's hover popover shows the specific
application within that downstream lesson. Width 720, height 240.

---

## 5. Algorithm and Math Implementation

The TypeScript module `src/importance-sampling/` is small and self-contained,
around eighty lines total.

```ts
/** Importance weight w(x) = p(x) / q(x). */
export function importanceWeight(
  pPdf: (x: number) => number,
  qPdf: (x: number) => number,
  x: number,
): number {
  const q = qPdf(x);
  if (q === 0) return Infinity;   // coverage condition violated
  return pPdf(x) / q;
}

/** Ordinary IS estimator (1/N) Σ w(x_i) f(x_i). */
export function ordinaryIS(
  samples: number[],
  f: (x: number) => number,
  w: (x: number) => number,
): number {
  return samples.reduce((s, x) => s + w(x) * f(x), 0) / samples.length;
}

/** Weighted (self-normalized) IS estimator. */
export function weightedIS(
  samples: number[],
  f: (x: number) => number,
  w: (x: number) => number,
): number {
  let num = 0, den = 0;
  for (const x of samples) {
    const wi = w(x);
    num += wi * f(x);
    den += wi;
  }
  return den === 0 ? 0 : num / den;
}

/** Effective sample size N_eff = (Σw)² / Σw². */
export function effectiveSampleSize(weights: number[]): number {
  let s = 0, s2 = 0;
  for (const w of weights) { s += w; s2 += w * w; }
  return s2 === 0 ? 0 : (s * s) / s2;
}

/** Trajectory-level IS weight ρ_{0:T-1} for an episode. */
export function trajectoryISWeight(
  traj: { s: number; a: number }[],
  piTarget: (s: number, a: number) => number,
  piBehavior: (s: number, a: number) => number,
): number {
  let rho = 1.0;
  for (const { s, a } of traj) {
    const pb = piBehavior(s, a);
    if (pb === 0) return 0;       // should not happen if behavior was used to sample
    rho *= piTarget(s, a) / pb;
    if (rho === 0) return 0;       // target says zero, no need to continue
  }
  return rho;
}

/** Per-decision IS contribution for one trajectory. */
export function perDecisionIS(
  traj: { s: number; a: number; r: number }[],
  piTarget: (s: number, a: number) => number,
  piBehavior: (s: number, a: number) => number,
  gamma: number,
): number {
  let total = 0, rho = 1.0;
  for (let t = 0; t < traj.length; t++) {
    const { s, a, r } = traj[t];
    total += Math.pow(gamma, t) * r * rho;
    rho *= piTarget(s, a) / piBehavior(s, a);
    if (rho === 0) return total;
  }
  return total;
}
```

**Vitest test targets** (from pre-verified numerics):

```ts
test('Ordinary IS is unbiased for Gaussian f(X)=X^2', () => {
  // p = N(0,1), q = N(0,2). True E_p[X^2] = 1.
  const N = 200000;
  const rng = seedrandom(123);
  const samples = Array.from({ length: N }, () => 2 * rngNormal(rng));
  const est = ordinaryIS(
    samples,
    x => x * x,
    x => normalPdf(x, 0, 1) / normalPdf(x, 0, 2),
  );
  expect(est).toBeCloseTo(1.0, 1);
});

test('Trajectory IS weight = 4^L for uniform→deterministic match', () => {
  const trajLength4 = [
    { s: 0, a: 1 }, { s: 1, a: 1 }, { s: 2, a: 2 }, { s: 5, a: 2 },
  ];
  const piT = (s: number, a: number) => {
    const opt: Record<number, number> = { 0: 1, 1: 1, 2: 2, 5: 2 };
    return opt[s] === a ? 1 : 0;
  };
  const piB = () => 0.25;
  expect(trajectoryISWeight(trajLength4, piT, piB)).toBe(256);
});

test('Trajectory IS weight = 0 on a single mismatch', () => {
  const traj = [
    { s: 0, a: 1 }, { s: 1, a: 3 /* WRONG */ }, { s: 0, a: 1 },
  ];
  const piT = (s: number, a: number) =>
    (s === 0 && a === 1) || (s === 1 && a === 1) ? 1 : 0;
  const piB = () => 0.25;
  expect(trajectoryISWeight(traj, piT, piB)).toBe(0);
});

test('Effective sample size N_eff = N when weights equal', () => {
  expect(effectiveSampleSize([2, 2, 2, 2, 2])).toBe(5);
});

test('Effective sample size N_eff = 1 when one weight dominates', () => {
  expect(effectiveSampleSize([256, 0, 0, 0, 0, 0, 0, 0])).toBe(1);
});
```

---

## 6. Component Catalog

| Code | Component                       | Section | Polish budget |
|-----|---------------------------------|---------|--------------|
| V1  | `<ISIdentityDemonstrator>`      | §1      | 1 day        |
| V2  | `<VarianceExplorer>`            | §2      | 1.5 days     |
| V3  | `<TrajectoryISExplorer>`        | §3      | **2-3 days** (centerpiece) |
| V4  | `<PerDecisionISStepwise>`       | §4      | 1 day        |
| V5  | `<RoadmapMini>` (update)        | §5      | 0.5 day      |

Total polish budget around six to seven days, in line with Lesson 4
(Contractions). This is a tool lesson, not a destination.

**Reuse from prior lessons:** `GridworldRenderer`, `MDPEditor`, `MathBlock`,
`CrosslinkCallout`, `PanelChrome`, `RoadmapMini`, all of `src/mdp/`, and
`policyEvaluationExact` from `src/dp/`. New code is small: five functions in
`src/importance-sampling/` plus the visualizations.

---

## 7. Page-Level User Experience

Same conventions as prior lessons. Single-page scroll, prereq strip at top,
reduced-motion support is especially important for V3 since the trajectory
animation can be motion-heavy. The centerpiece (V3) is the only component
that breaks out to 960 pixels in width.

---

## 8. Acceptance Criteria

After completing this lesson, a learner should be able to do the following.

State the importance sampling identity precisely and verify it algebraically.
Write down both the ordinary and weighted IS estimators and explain why
weighted IS is preferred in practice despite its bias. Identify the coverage
condition for a given pair of distributions and recognize when it fails.
State the finite-variance condition `E_q[(p/q)^2 f^2] < \infty` and identify
the most common failure case (proposal lighter-tailed than target). Compute
the trajectory IS weight `\rho_{0:T-1}` for a given trajectory and pair of
policies. Derive the per-decision IS estimator from the trajectory IS
estimator by canceling future ratios. Compute the effective sample size for
a set of weights. Explain why off-policy MC variance grows exponentially
with the horizon.

A concrete acceptance check: hand the learner a three-state chain MDP from
Lesson 3's exercises, a behavior policy of "uniform random," and a target
policy of "always go right." Ask them to compute the probability that a
length-`L` trajectory has non-zero weight, the expected weight conditional
on it being non-zero, and to predict how the trajectory-IS estimator's
variance scales with `L`. V3 lets them check their predictions on a small
example.

---

## 9. Stretch Goals (post-MVP)

The doubly-robust estimator combines importance sampling with a model-based
value estimate to reduce variance further; it sees use in offline RL and
could appear as a fifth estimator alongside ordinary and weighted IS in the
visualizations. The Owen-Zhou bound on weighted IS bias is `O(1/N)` with
explicit constants and could be quoted in section 2. V-trace clipping could
appear as a fourth estimator in V4's side-by-side comparison.

---

## 10. Out of Scope (intentionally)

Sequential Monte Carlo and particle filters are closely related but solve a
different problem (state estimation rather than expectation estimation) and
are out of scope. Multilevel Monte Carlo is not commonly used in
mainstream RL and is out of scope. The boundary between IS and variational
inference (sometimes called variational importance sampling) is interesting
but tangential to the curriculum.

---

## 11. Training Notebook

Not applicable. No models are trained for this lesson. The script
`scripts/is_traces.py` (around fifty lines) pre-computes the Gaussian
variance sweep, the gridworld estimator statistics, and the per-decision
comparison, all into JSON files under `public/data/is/` for instant initial
render. The in-browser TypeScript implementation reproduces these
computations live when the user interacts.

---

## End of specification

Total length: roughly nine hundred lines. Slightly shorter than the
Contractions lesson, similar shape: short, focused, ending squarely in
reinforcement-learning territory with the centerpiece visualization carrying
the polish weight. The empty histogram at `N=100` in V3, transitioning to
the populated histogram at `N=10000`, is the visual story that should land
hardest for the learner.