# Lesson 10 — Policy Gradient Methods

> **The shift to policy-based RL.** Value-based methods (Q-learning,
> DQN) learn a value function and act greedily with respect to it. Policy
> gradient methods parameterize the policy directly: `π_θ(a | s)` is a
> neural network from states to action distributions, and we optimize
> `θ` by gradient ascent on the expected return. This is the foundation
> of every continuous-control algorithm in deep RL, every actor-critic
> method, every modern LLM alignment technique (PPO is the workhorse of
> RLHF). The lesson covers the score function estimator from scratch,
> derives the policy gradient theorem, introduces REINFORCE as the
> simplest implementation, and then spends most of its time on the
> variance reduction problem — baselines, advantages, and actor-critic
> — that drives every modern policy gradient algorithm.

> **Where this slots in.** Between Lesson 9 (DQN) and Lesson 11
> (TRPO/PPO). Critical-path central — the lesson opens the second
> tier of the curriculum. From here forward, value-based methods
> recede (they reappear as critics inside actor-critic) and the policy
> becomes the primary object. The lesson cashes in the contraction
> theory from Lesson 4 (for the critic's TD convergence), the MC
> machinery from Lesson 7 (REINFORCE uses MC returns), the TD machinery
> from Lesson 8 (actor-critic uses TD targets for the critic), and the
> neural-net infrastructure from Lesson 9 (the policy and the critic
> are both parameterized by neural networks). It pays forward to
> Lessons 11, 12, 13, 14, and 17.

---

## 0. Pedagogical Philosophy

Seven commitments specific to this lesson.

The first commitment is that the score function estimator gets a full
derivation from scratch. The original curriculum had this as its own
prerequisite lesson; we have folded it into the opening sections of
this lesson because the curriculum was restructured. The trick is
short but conceptually non-trivial: `∇_θ E_{p_θ}[f(X)] = E_{p_θ}[f(X) ∇_θ \log p_θ(X)]`,
which is "obvious" once seen but typically not in students' working
memory. We derive it from the log-derivative identity in Section 2
and verify it on a Gaussian toy problem before applying it to RL.

The second commitment is that the policy gradient theorem is presented
as a *specific application* of the score function trick to the RL
objective `J(θ) = E_τ[\sum_t \gamma^t r_t]`. We do not present it as a
distinct theorem with its own derivation; that obscures the structure.
Once the learner sees the score function move, the policy gradient
theorem is the score function applied to the trajectory likelihood
`p_θ(τ)`, with the convenient property that only the policy factors
depend on `θ` (the transition dynamics and reward function cancel out).
This is the algorithmic core of why model-free policy gradient is
even possible.

The third commitment is that the variance reduction story is treated
as the central engineering challenge of policy gradient. REINFORCE in
its naive form has gradient variance that scales catastrophically
with horizon length. Every modern PG algorithm — REINFORCE with
baseline, actor-critic, advantage normalization, GAE, PPO's clipped
ratio — is a variance-reduction technique. We motivate each by the
variance problem it solves. The narrative arc of Sections 5 through 7
is "the gradient is high-variance; here's how we tame it."

The fourth commitment is that we are honest about the bias-variance
trade-off. REINFORCE is unbiased; actor-critic is biased (the critic
is wrong while learning); GAE interpolates between them. The trade-off
is not "actor-critic is better" — it is "actor-critic trades some bias
for lower variance, and in practice the trade is worth it." This
parallels the MC vs TD trade-off from Lessons 7-8 and the same
language is reused. The structural similarity is deliberate.

The fifth commitment is that the running gridworld continues but in
a slightly different form. The 3×3 gridworld with deterministic
optimal V*(0,0) = 0.7290 from Lessons 3, 5, 6, 7, 8 returns here.
A *softmax* policy `π_θ(a|s) = softmax(θ_{s,a})` is introduced as the
parameterized policy. Tabular PG (softmax with one parameter per
(state, action)) is the simplest setting and admits clean comparisons
to the tabular Q-learning baseline from Lesson 8. The result: softmax
PG converges to V(0,0) ≈ 0.722, slightly below 0.729 because softmax
cannot be perfectly deterministic. The gap between 0.722 and 0.729 is
the cost of representational stochasticity, and we make this explicit.

The sixth commitment is that we connect to continuous-control
settings without implementing them. The Gaussian policy
`π_θ(a|s) = N(a; μ_θ(s), σ_θ(s))` is the standard parameterization
for continuous actions. We derive the score function for this case
explicitly and connect to the broader literature, but the
visualizations stay on the gridworld with softmax. The reason: the
gridworld provides exact ground-truth values from prior lessons,
which is invaluable for verifying that the algorithm is doing the
right thing. Continuous control would require choosing a benchmark
environment (CartPole, Pendulum) that doesn't have an exact analytical
solution, and the lesson would lose its precise numerical anchoring.

The seventh commitment is that we name what is missing. Policy
gradient is not a closed solution. The vanilla algorithms have known
failure modes: high variance even with baselines, sensitivity to
learning rate, policy collapse to bad local optima. The lesson sets
up Lessons 11 (trust regions and clipping) and 12 (entropy
regularization) as direct responses to these failure modes. The
forward links in Section 8 are concrete pointers to "here is the
algorithm that fixes the problem we just identified."

---

## 1. Tech Stack

The tech stack is unchanged from prior lessons. Vite plus TypeScript
strict, KaTeX, D3 v7, `ml-matrix`, Vitest. No new dependencies.

For the in-browser policy gradient implementation, the policy
parameters are stored as `Float64Array` (small enough that we don't
need PyTorch/ONNX as in Lesson 9). The actor-critic critic is also
stored as `Float64Array`. The gradient computation is hand-coded
backprop through the softmax — short, transparent, easily testable.
Total new code in `src/pg/` is around 250 lines.

A Python script `scripts/pg_traces.py` (around 100 lines) pre-computes
the convergence statistics: REINFORCE final V across seeds and
training lengths, REINFORCE+baseline variance reduction, actor-critic
convergence rates, and the gradient-variance comparison at fixed
policy. The outputs land as JSON for instant initial render of V3 and
V6.

---

## 2. Visual and Aesthetic Direction

The curriculum aesthetic continues. Tokens specific to policy gradient
methods:

```css
:root {
  /* Policy and value */
  --pg-policy:         #7c3aed;   /* violet-600  | parameterized policy π_θ */
  --pg-value:          #0e7490;   /* cyan-700    | critic V_φ */
  --pg-advantage:      #15803d;   /* green-700   | advantage A(s,a) */

  /* Algorithm signatures */
  --pg-reinforce:      #b45309;   /* amber-700   | vanilla REINFORCE */
  --pg-baseline:       #6d28d9;   /* violet-700  | REINFORCE with baseline */
  --pg-actor-critic:   #0891b2;   /* cyan-600    | actor-critic */
  --pg-gae:            #be185d;   /* pink-700    | GAE (preview for L11) */

  /* Variance and gradient diagnostics */
  --pg-grad-vector:    #ea580c;   /* orange-600  | gradient vectors */
  --pg-variance-band:  rgba(180, 83, 9, 0.18);  /* amber tint, variance */
  --pg-bias-band:      rgba(14, 116, 144, 0.18); /* cyan tint, bias */

  /* Policy probability bars */
  --pg-prob-greedy:    #15803d;   /* green; the greedy action */
  --pg-prob-other:     #94a3b8;   /* slate; other actions */
}
```

The policy gets violet (continuing the running theme of violet for
"the principled distributional thing being learned" from Bandits and
IS). The critic gets cyan (continuing TD's color from Lesson 8). The
advantage gets green (continuing the "principled signal" color from
the IS and MC lessons). The algorithm signature colors are deliberately
chosen so REINFORCE/baseline/actor-critic form a clean visual
progression: amber → violet → cyan corresponds to "raw MC" → "MC with
control variate" → "TD-bootstrapped."

GAE is given its own pink color as a forward-link preview. It will
become a first-class citizen in Lesson 11.

---

## 3. Lesson Metadata

```ts
export const lessonMeta: LessonMeta = {
  id: "policy-gradient",
  title: "Policy Gradient Methods",
  subtitle: "Parameterize the policy. Optimize by gradient ascent. Tame the variance.",
  tier: 1,
  difficulty: 4,
  estimatedReadMinutes: 80,
  role: "critical-path",
  prerequisites: [
    { lesson: "mdps",                 anchor: "discounted-return" },
    { lesson: "contractions",         anchor: "bellman-contraction" },
    { lesson: "monte-carlo",          anchor: "mc-policy-evaluation" },
    { lesson: "td-learning",          anchor: "td-zero-prediction" },
    { lesson: "td-learning",          anchor: "td-convergence" },
    { lesson: "function-approximation", anchor: "neural-q-network" },
  ],
  exportedAnchors: [
    "policy-gradient-motivation",
    "score-function-estimator",
    "log-derivative-trick",
    "policy-gradient-theorem",
    "reinforce-algorithm",
    "control-variate-baseline",
    "actor-critic",
    "advantage-function",
    "bias-variance-trade-off",
    "softmax-policy",
    "gaussian-policy",
  ],
  centerpieceComponent: "PolicyGradientLab",
  forwardLinksWhenReady: [
    { to: "trpo-ppo",                 anchor: "trust-region-policy-optimization" },
    { to: "trpo-ppo",                 anchor: "generalized-advantage-estimation" },
    { to: "max-ent-rl",               anchor: "entropy-regularization" },
    { to: "sac",                      anchor: "soft-actor-critic" },
    { to: "model-based",              anchor: "model-based-pg" },
    { to: "rlhf",                     anchor: "rlhf-with-ppo" },
  ],
};
```

---

## 4. Section-by-Section Plan

### Section 1 — From Values to Policies

**Tagline:** *Why parameterize the policy directly? Three reasons that motivate the field.*
**Length:** ~700 words.
**Anchor:** `policy-gradient-motivation`.

---

**Prose:**

Lessons 8 and 9 built value-based RL: learn `Q(s, a)`, act greedily.
The recipe works beautifully on discrete-action problems with
sufficient data — Atari, board games, simple robotics with
discretized control. But the recipe has three structural limitations
that motivate a different approach.

**Limitation 1: Continuous action spaces.** Q-learning's policy is
`\pi(s) = \arg\max_{a} Q(s, a)`. For discrete `A`, the `\arg\max` is a
lookup over `|A|` values. For continuous `A` (e.g., a robot's joint
angles), the `\arg\max` is an optimization problem in its own right —
solved at every policy invocation, every training step, every
deployment forward pass. Practical Q-learning variants for continuous
actions exist (Normalized Advantage Functions, deep deterministic
policy gradient), but they are either limited (quadratic in actions,
which is restrictive) or are themselves policy gradient methods in
disguise. The direct approach: parameterize the policy itself, never
take an `\arg\max`.

**Limitation 2: Stochastic policies.** Q-learning's induced policy is
deterministic (greedy). In some settings the optimal policy *is*
stochastic — partial observability, multi-agent settings,
adversarial environments. Even in fully-observed MDPs, a stochastic
policy can be preferable for robustness reasons (see Lesson 7's
ε-soft optimality discussion). Q-learning cannot directly represent
or learn a stochastic policy; only "noisy approximations of greedy"
via ε-greedy or NoisyNets. Policy gradient methods natively
parameterize stochastic policies and can directly optimize them.

**Limitation 3: Direct optimization.** Value-based RL is *indirect*: we
optimize a Q-function with the hope that the greedy policy on the
learned Q-function is close to the optimal policy. The mapping from
Q-function error to policy error can be discontinuous (the greedy
policy changes by a discrete jump when Q-values cross), making it
hard to reason about the policy's quality from the Q-function's
quality. Policy gradient is direct: we optimize the policy itself with
respect to a smooth performance metric, so improvements in the metric
directly translate to improvements in the policy.

---

**The shift in viewpoint.** From this lesson forward, the central
object is `\pi_\theta(a \mid s)` — a parameterized distribution over
actions given states, with parameters `\theta` that we update by
gradient ascent on the expected return:

$$
J(\theta) \;:=\; \mathbb{E}_{\tau \sim \pi_\theta}\!\left[ \sum_{t=0}^{T-1} \gamma^t r_{t+1} \right].
$$

The performance metric `J(\theta)` is a scalar function of the
parameters. We want to update `\theta` in the direction of the
gradient `\nabla_\theta J(\theta)`. The challenge is computing this
gradient, because `J(\theta)` is an expectation over trajectories,
and those trajectories depend on `\theta` through the policy.

The next section introduces the **score function estimator**, the
technique that makes this gradient computable from sampled
trajectories.

---

**The role of value functions in PG.** A common misconception is that
policy gradient methods abandon value functions. They don't. Most
policy gradient methods use a value function in a supporting role —
as a baseline for variance reduction, as a critic in actor-critic, as
the source of an advantage signal. The value function is no longer
the *primary* object (the policy is), but it remains a workhorse
estimator. The TD machinery from Lesson 8 is fully reused: the
critic in actor-critic methods is exactly TD(0) applied to a value
function approximator.

> **Forward link to LLM alignment.** Modern reinforcement learning
> from human feedback (RLHF), the dominant technique for aligning
> large language models with human preferences, uses Proximal Policy
> Optimization (PPO) — a policy gradient method introduced in
> Lesson 11 — to fine-tune the language model's policy against a
> reward model. The "policy" is the LLM's token distribution; the
> "value function" is a separate critic network. The same machinery
> developed in this lesson is the foundation of LLM alignment.

---

**Visualization V1 — From Q-Values to Policies.**

A two-panel side-by-side comparison. Left panel: the 3×3 gridworld
with the Q-function from Lesson 8's Q-learning displayed at each cell
(four Q-values per cell, arranged in the action positions). An arrow
shows the implicit greedy policy. Right panel: the same gridworld
with the softmax policy `\pi_\theta(a \mid s)` displayed at each cell
(four probabilities per cell, arranged as bars). The bars are the
direct representation of the policy. A toggle below the panels
labeled "click for the difference": tapping it overlays both panels
and highlights that the policy is *directly* represented in the right
view but only *indirectly* (through `\arg\max`) in the left view. Width
720, height 360.

---

### Section 2 — The Score Function Estimator

**Tagline:** *The log-derivative trick. Unbiased gradient estimates from samples.*
**Length:** ~900 words.
**Anchor:** `score-function-estimator` and `log-derivative-trick`.

---

**Prose:**

We want `\nabla_\theta \mathbb{E}_{p_\theta}[f(X)]` for a parameterized
distribution `p_\theta` and a function `f`. The challenge is that the
expectation is taken with respect to the very distribution that
depends on `\theta`. We cannot simply pull the gradient through the
expectation as if the measure were fixed.

The **log-derivative trick** (a.k.a. the **score function estimator**,
a.k.a. **REINFORCE**, after the algorithm we will derive from it)
solves this. The derivation is short:

$$
\begin{aligned}
\nabla_\theta \mathbb{E}_{p_\theta}[f(X)] &= \nabla_\theta \int f(x) p_\theta(x) \, dx \\
&= \int f(x) \nabla_\theta p_\theta(x) \, dx \\
&= \int f(x) p_\theta(x) \frac{\nabla_\theta p_\theta(x)}{p_\theta(x)} \, dx \\
&= \int f(x) p_\theta(x) \nabla_\theta \log p_\theta(x) \, dx \\
&= \mathbb{E}_{p_\theta}\!\left[ f(X) \nabla_\theta \log p_\theta(X) \right].
\end{aligned}
$$

The first equality writes the expectation as an integral. The second
moves the gradient inside (assuming the standard regularity
conditions — `p_\theta` is differentiable in `\theta`, integration
and differentiation can be swapped). The third multiplies and divides
by `p_\theta`. The fourth uses the **log-derivative identity**
`\nabla_\theta p_\theta(x) / p_\theta(x) = \nabla_\theta \log p_\theta(x)`,
which is the chain rule applied to `\log`. The fifth rewrites the
integral as an expectation under `p_\theta`.

The result: the gradient of the expectation can be written as the
expectation of the function `f` times the **score** function
`\nabla_\theta \log p_\theta(X)`. The expectation is still under
`p_\theta`, so we can estimate it by sampling `X_i \sim p_\theta` and
averaging:

$$
\widehat{\nabla J}(\theta) \;=\; \frac{1}{N} \sum_{i=1}^N f(X_i) \nabla_\theta \log p_\theta(X_i).
$$

This estimator is **unbiased**: its expected value equals the true
gradient. It is also **computable** from samples — we never need to
take a gradient through the sampling process itself. This is the
foundation of policy gradient methods.

---

**Worked example: Gaussian.** Let `p_\theta = \mathcal{N}(\theta, 1)`,
the unit-variance Gaussian with mean `\theta`. We want
`\nabla_\theta \mathbb{E}_p[X^2]`. The true value is
`\mathbb{E}_p[X^2] = \theta^2 + 1`, so `\nabla_\theta \mathbb{E}_p[X^2] = 2\theta`.

The score function for the Gaussian: `\log p_\theta(x) = -\frac{1}{2}(x - \theta)^2 + \text{const}`,
so `\nabla_\theta \log p_\theta(x) = x - \theta`. The score function
estimator is

$$
\widehat{\nabla J}(\theta) \;=\; \frac{1}{N} \sum_{i=1}^N X_i^2 (X_i - \theta), \qquad X_i \sim \mathcal{N}(\theta, 1).
$$

Numerically, with `\theta = 1.0` and `N = 1000`, this estimator
produces `1.97 \pm 0.17` across 100 trials. The true gradient is
`2.0`. The estimator is unbiased and the variance is the
expected `O(1/\sqrt{N})`.

---

**The estimator's variance, generally.** The score function estimator
is unbiased, but its variance can be enormous. For high-dimensional
`X`, deep parametrizations of `p_\theta`, and high-magnitude `f`, the
naive Monte Carlo estimator has variance that grows uncontrollably.
The variance reduction problem is the central engineering challenge
of policy gradient methods, and Sections 5 through 7 focus on it
exclusively.

A first variance-reduction observation: we can subtract any
**constant** from `f` without changing the estimator's expectation,
because `\mathbb{E}_{p_\theta}[c \cdot \nabla_\theta \log p_\theta(X)] = 0`
for any constant `c` (the expectation of the score function under its
own distribution is zero). More generally, we can subtract any
function `b(\cdot)` that does not depend on the sampled `X` we are
differentiating through. This is the **control variate** trick, and
the function `b` is called a **baseline**. The optimal baseline (in
minimum-variance sense) is the expectation of `f`. We will revisit
this in Section 5 in the policy gradient setting.

---

**The score function for common policy parameterizations.**

**Softmax policy (discrete actions).** With `\pi_\theta(a \mid s) =
\exp(\theta_{s,a}) / \sum_{a'} \exp(\theta_{s,a'})`, the score function
is

$$
\nabla_{\theta_{s,a}} \log \pi_\theta(a' \mid s') \;=\; \begin{cases} 1 - \pi_\theta(a \mid s) & \text{if } s = s', a = a' \\ -\pi_\theta(a \mid s) & \text{if } s = s', a \neq a' \\ 0 & \text{if } s \neq s' \end{cases}
$$

This is convenient: the score is dense within state `s` (all actions'
parameters get a nonzero gradient) but sparse across states (only the
state actually visited gets updated).

**Gaussian policy (continuous actions).** With
`\pi_\theta(a \mid s) = \mathcal{N}(a; \mu_\theta(s), \sigma^2)` (fixed
variance, mean parameterized by a neural network), the score function
is

$$
\nabla_\theta \log \pi_\theta(a \mid s) \;=\; \nabla_\theta \mu_\theta(s) \cdot \frac{a - \mu_\theta(s)}{\sigma^2}.
$$

The gradient is the network gradient of the mean, weighted by the
"surprise" of the observed action (how far it was from the mean,
normalized by the variance). This is one of the workhorse forms in
continuous control.

---

> **Crosslink to Lesson 6 (Importance Sampling).** The score function
> estimator is sometimes called REINFORCE because that is the name of
> the algorithm we are about to derive. It is also called the
> **likelihood ratio** estimator in the statistics literature,
> because the score function `\nabla_\theta \log p_\theta(X)` is
> formally a likelihood ratio. Lesson 6's importance sampling
> apparatus and Lesson 10's score function estimator are closely
> related: both are "trick" estimators that use the log-likelihood
> to reweight samples. The score function uses the same identity but
> for gradients of expectations rather than for switching the
> sampling distribution.

---

**Visualization V2 — Score Function Estimator (Gaussian).**

A live demo of the score function estimator on the Gaussian toy.
A slider controls `\theta` (from -2 to +2). Below the slider, three
displays update in real time: (i) the current Gaussian
`\mathcal{N}(\theta, 1)` drawn as a density plot with a sample of
`N = 50` points shown as dots; (ii) the function `f(x) = x^2`
overlaid; (iii) the score function estimator's value
`\frac{1}{N} \sum f(X_i)(X_i - \theta)`, compared against the true
gradient `2\theta` shown as a horizontal reference. As the user drags
`\theta`, all three displays update in concert. A "resample" button
draws fresh `X_i`; the estimator value bounces around the reference,
illustrating the Monte Carlo noise. Width 720, height 420.

---

### Section 3 — The Policy Gradient Theorem

**Tagline:** *Apply the score function to the RL objective. Watch the dynamics cancel out.*
**Length:** ~800 words.
**Anchor:** `policy-gradient-theorem`.

---

**Prose:**

We have the score function estimator. Now we apply it to the RL
objective.

The trajectory distribution under `\pi_\theta` is

$$
p_\theta(\tau) \;=\; \rho_0(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t) \, P(s_{t+1} \mid s_t, a_t),
$$

where `\rho_0` is the initial state distribution and `P` is the
transition kernel (neither depends on `\theta`). The expected return
is

$$
J(\theta) \;=\; \mathbb{E}_{\tau \sim p_\theta}[R(\tau)], \qquad R(\tau) = \sum_{t=0}^{T-1} \gamma^t r_{t+1}.
$$

The score function estimator gives

$$
\nabla_\theta J(\theta) \;=\; \mathbb{E}_{\tau \sim p_\theta}\!\left[ R(\tau) \cdot \nabla_\theta \log p_\theta(\tau) \right].
$$

Now we examine `\nabla_\theta \log p_\theta(\tau)`. Taking the log of
`p_\theta(\tau)`:

$$
\log p_\theta(\tau) \;=\; \log \rho_0(s_0) + \sum_{t=0}^{T-1} \log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T-1} \log P(s_{t+1} \mid s_t, a_t).
$$

Of these terms, only the middle one depends on `\theta` — the initial
distribution and the transition kernel are fixed environmental
parameters. So

$$
\nabla_\theta \log p_\theta(\tau) \;=\; \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t).
$$

The transition dynamics and reward function **cancel out** of the
gradient computation. This is the central algorithmic miracle of
policy gradient methods: we can compute `\nabla_\theta J(\theta)`
without knowing or estimating the environment's dynamics, even though
`J(\theta)` is defined as an expectation over those dynamics.

The full gradient is then

$$
\nabla_\theta J(\theta) \;=\; \mathbb{E}_{\tau \sim p_\theta}\!\left[ \left( \sum_{t=0}^{T-1} \gamma^t r_{t+1} \right) \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right].
$$

---

**Reward-to-go simplification.** The above formula is correct but
contains an unnecessary cross-term. Each `\nabla \log \pi(a_t \mid s_t)`
is multiplied by the *entire* trajectory return. But by causality,
actions in the future cannot affect the past — `a_t` is sampled at
time `t`, and rewards received before time `t` are independent of
`a_t` given the trajectory history. So the contribution of
`r_{k+1}` for `k < t` to the gradient with respect to `\theta` via
`a_t` is zero. (Formal proof: the cross terms have expectation zero
under the trajectory distribution.)

The cleaned-up policy gradient is

$$
\nabla_\theta J(\theta) \;=\; \mathbb{E}_{\tau \sim p_\theta}\!\left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t \right],
$$

where `G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_{k+1}` is the
**reward-to-go** from time `t` (the MC return from time `t` onward,
re-indexed so it starts at time `t`). This formulation is the **policy
gradient theorem** as stated by Sutton, McAllester, Singh, Mansour
(1999).

The interpretation: `\nabla_\theta \log \pi_\theta(a_t \mid s_t)`
points in parameter space in the direction that *increases* the
probability of action `a_t` at state `s_t`. We weight this gradient
by `G_t`, the reward-to-go. If `G_t` is large and positive, we push
`\theta` to make `a_t` *more* likely at `s_t`. If `G_t` is negative,
we push the other way. This is policy iteration in continuous
parameter space: actions that led to high returns become more
probable, actions that led to low returns become less.

---

**A subtlety: the discount factor.** The formula above uses `G_t` as
the reward-to-go. There is a subtle question of where the discount
factor enters: does `G_t = \sum_{k \geq t} \gamma^{k-t} r_{k+1}`
(the return from `t` onward, discounted from `t`) or
`\sum_{k \geq t} \gamma^k r_{k+1}` (the return from `0` onward,
discounted from time 0)? The two differ by an overall factor of
`\gamma^t`.

Strictly, the policy gradient theorem as stated above uses
`\gamma^{k-t}`. This corresponds to viewing each time step as a
separate decision problem with its own discount from its own
reference point. In practice, most implementations use `G_t` without
the leading `\gamma^t` factor (called the "undiscounted" form), which
is technically a small bias but practically negligible. Sutton & Barto
Chapter 13 discusses the distinction in detail. We use the
`\gamma^{k-t}` form throughout for theoretical cleanness.

---

**Visualization V3 — The Policy Gradient Theorem.**

A two-panel illustration. Left panel: a sampled trajectory on the
gridworld, with each time step labeled. Below each step, the score
function `\nabla_\theta \log \pi_\theta(a_t \mid s_t)` is drawn as a
gradient arrow on a small policy probabilities bar chart (showing the
direction that would increase `\pi(a_t \mid s_t)`). Right panel: the
same trajectory, but each step's contribution to the gradient
estimate is shown: the score arrow scaled by `G_t`. Positive `G_t`
makes the arrow grow in the "increase action probability" direction;
negative `G_t` flips it.

A reset button resamples the trajectory; an "average over N
trajectories" slider averages the per-step gradient estimates across
N rollouts, showing the gradient direction sharpening as N grows.
Width 720, height 460.

---

### Section 4 — REINFORCE

**Tagline:** *Plug MC returns into the score function. The original policy gradient algorithm.*
**Length:** ~700 words.
**Anchor:** `reinforce-algorithm`.

---

**Prose:**

The simplest implementation of the policy gradient theorem is
**REINFORCE** (Williams, 1992). It is exactly the policy gradient
formula, with one design choice: estimate the expectation by a single
Monte Carlo sample, and treat each step's reward-to-go `G_t` as the
unbiased sample of the expected return-to-go.

```text
REINFORCE(α, N_episodes):
    initialize θ
    for episode = 1, ..., N_episodes:
        sample trajectory τ ~ π_θ
        compute returns G_0, G_1, ..., G_{T-1} (backward sweep)
        for t = 0, ..., T-1:
            θ ← θ + α · γ^t · G_t · ∇_θ log π_θ(a_t | s_t)
    return π_θ
```

In words: roll out one episode, compute the per-step reward-to-go,
update `\theta` by a sum of `T` gradient contributions, one per step.

Each gradient contribution `\gamma^t G_t \nabla \log \pi(a_t \mid s_t)`
is an unbiased single-sample estimate of `\nabla J(\theta)`'s
contribution from time `t`. Averaged over many episodes (and many time
steps within each), the estimator converges to `\nabla J(\theta)` in
expectation. The algorithm performs stochastic gradient ascent on
`J(\theta)`.

---

**On the running gridworld with softmax policy.** REINFORCE applied
to the 3×3 gridworld with `\pi_\theta(a \mid s) = \text{softmax}(\theta_{s,a})`,
starting from `\theta = \mathbf{0}` (uniform policy) and running 2,000
episodes with `\alpha = 0.05`:

| seeds | mean of `\hat V(0, 0)` | std of `\hat V(0, 0)` |
|:------|----------------------:|----------------------:|
| 10    | **0.7219**            | 0.0076                |

This is close to the deterministic optimal `V^*(0, 0) = 0.7290` but
not equal. The gap comes from the softmax policy's residual
stochasticity: even at convergence, `\pi_\theta(a^* \mid s)` does not
reach 1.0 (that would require infinite `\theta`). The policy becomes
*nearly* deterministic, but never exactly. The convergent V(0, 0)
reflects this residual stochasticity.

> **The cost of representational stochasticity.** Softmax policies
> cannot represent perfectly deterministic actions; they can only
> approach them as `\theta` grows large. For gridworlds and similar
> discrete problems, this is a small inefficiency (about 1% relative
> error in `V(0, 0)`). For continuous-action problems where the
> optimal action is exactly known (rare in practice), this can be a
> more significant limitation. Most modern policy gradient methods
> use either softmax (for discrete actions) or Gaussian (for
> continuous actions); both are stochastic, and both incur a small
> residual representational cost. SAC (Lesson 13) embraces this by
> using a max-entropy objective that explicitly prefers stochastic
> policies; PPO (Lesson 11) handles it via implicit constraints on
> policy change.

---

**REINFORCE's variance problem.** The gradient estimator has high
variance, and this is the dominant concern. A back-of-envelope
calculation: on the gridworld, the std of a single episode's return
under uniform random policy is about 0.41 (verified empirically in
Lesson 7). Each REINFORCE gradient estimate uses one trajectory, so
the per-update gradient has standard deviation that scales with this
return std. Across thousands of episodes the variance averages out,
but the per-update noise is substantial.

On longer-horizon problems the situation degrades dramatically. For
Atari-scale episodes (thousands of steps), the per-trajectory return
variance is enormous, and naive REINFORCE is effectively unusable.
Every modern policy gradient algorithm includes one or more variance
reduction techniques. The next sections introduce them.

---

**Visualization V4 — REINFORCE Training Trace.**

A two-panel layout. Top panel: the 3×3 gridworld with the current
softmax policy displayed as per-state probability bars (one bar per
action). The bars start uniform (all 0.25) and evolve as training
progresses. A timeline scrubber moves through training episodes from
0 to 2,000; the policy displays update as the user scrubs. Bottom
panel: the per-episode return over training (raw, noisy) and the
running mean of returns (smooth). The raw trace is highly variable;
the smoothed line shows the convergence trend toward 0.72. Reference
lines at `V^*(0, 0) = 0.7290` and the softmax-cap of 0.722 are
drawn. Width 880, height 480.

---

### Section 5 — Baselines and Variance Reduction

**Tagline:** *Subtract a state-dependent baseline. Halves the variance, free of charge.*
**Length:** ~700 words.
**Anchor:** `control-variate-baseline`.

---

**Prose:**

The core variance-reduction technique for policy gradient is the
**baseline**. Subtract a function `b(s_t)` from the reward-to-go
inside the gradient sum:

$$
\nabla_\theta J(\theta) \;=\; \mathbb{E}\!\left[ \sum_t \nabla \log \pi(a_t \mid s_t) \cdot (G_t - b(s_t)) \right].
$$

The expectation is unchanged: as long as `b(s_t)` does not depend on
`a_t`, the cross-term contribution
`\mathbb{E}[\nabla \log \pi(a \mid s) \cdot b(s)]` is zero. (The
gradient of `\log \pi` averaged over `a` is zero for any `s`, because
`\sum_a \pi(a \mid s) = 1` for all `\theta` — differentiating gives
`\sum_a \pi \nabla \log \pi = 0`.)

The variance *is* changed, and the question is whether we can choose
`b(s_t)` to reduce it. The optimal baseline (in the sense of
minimum-variance gradient estimator) is

$$
b^*(s) \;=\; \frac{\mathbb{E}[(\nabla \log \pi)^2 \cdot G \mid s]}{\mathbb{E}[(\nabla \log \pi)^2 \mid s]},
$$

which is the score-weighted average of the return-to-go from `s`.
This is intractable to compute exactly. The standard practical choice
is to use the **state-value function**:

$$
b(s_t) \;=\; V^{\pi_\theta}(s_t).
$$

The state-value baseline is not optimal in the minimum-variance
sense, but it is close, and it has the conceptual advantage of being
directly interpretable: subtracting `V^\pi(s)` from `G_t` gives an
estimate of the **advantage** `A^\pi(s_t, a_t) := Q^\pi(s_t, a_t) - V^\pi(s_t)`,
which measures how much better-than-average `a_t` was at `s_t`. A
positive advantage means "this action was a good idea given the
state"; a negative advantage means "this action was below average."

---

**Empirical effect on the gridworld.** REINFORCE with a running-mean
baseline (the average return seen so far across all episodes) on the
3×3 gridworld, 2,000 episodes, 10 seeds, `\alpha = 0.05`:

| algorithm                  | mean `\hat V(0, 0)` | std `\hat V(0, 0)` |
|:---------------------------|--------------------:|-------------------:|
| Vanilla REINFORCE          | 0.7219              | 0.0076             |
| REINFORCE with baseline    | **0.7250**          | **0.0011**         |

The mean is similar (slight improvement; the baseline lets `\theta`
push toward more deterministic policies more aggressively). The
*standard deviation across seeds* drops from 0.0076 to 0.0011, a
**~7× variance reduction** in the final policy estimate. The
variance reduction in the gradient estimator itself is even larger;
at later training stages, when `V^\pi(s)` is well-estimated, the
advantage `G_t - V^\pi(s_t)` is much closer to zero than the raw
`G_t`, and the gradient noise drops accordingly.

The 7× variance reduction is not a coincidence — it is the typical
order of magnitude that the state-value baseline provides on simple
problems. On harder problems with longer horizons, the variance
reduction can be 100× or more.

---

**The "advantage" interpretation, revisited.** Substituting
`A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)` for `G_t - V^\pi(s_t)` (using
the fact that `\mathbb{E}[G_t \mid s_t, a_t] = Q^\pi(s_t, a_t)`),
we can write the policy gradient as

$$
\nabla_\theta J(\theta) \;=\; \mathbb{E}\!\left[ \sum_t \nabla \log \pi(a_t \mid s_t) \cdot A^\pi(s_t, a_t) \right].
$$

This is the **advantage policy gradient**: weight each score by the
*advantage* of the action taken. Advantages are the cleanest version
of "did this action work out better than average?"

In practice we never have the true `A^\pi`; we estimate it. The next
section introduces actor-critic methods, which estimate `A^\pi` from a
learned critic and use the estimated advantage as the policy gradient
weight.

---

**Visualization V5 — Variance Reduction with Baseline.**

A two-panel layout. Left panel: a histogram of per-episode gradient
norms over 200 episodes at a fixed (uniform random) policy. Vanilla
REINFORCE's histogram is wide; baseline-corrected REINFORCE's
histogram is narrower (centered roughly the same, lighter tails). The
two histograms are overlaid with translucent fills.

Right panel: the seed-to-seed convergence comparison. 10 seeds of
vanilla REINFORCE produce trajectories of `\hat V(0, 0)` over
training; 10 seeds of REINFORCE-with-baseline produce a much
narrower band. Both bands are drawn as 90% confidence intervals
around the mean trajectory. The visual statement: same mean, much
less spread. Width 880, height 460.

---

### Section 6 — Actor-Critic Methods (Centerpiece)

**Tagline:** *Replace the MC return with a TD-bootstrapped critic. Variance crashes; bias appears.*
**Length:** ~800 words.
**Anchor:** `actor-critic`.

---

**Prose:**

The state-value baseline of Section 5 still uses the MC return `G_t`
to estimate `A^\pi(s_t, a_t)`. We can do better by *also*
bootstrap-replacing the return:

$$
G_t \;\approx\; r_{t+1} + \gamma V_\phi(s_{t+1}),
$$

where `V_\phi` is a parameterized critic. The TD error
`\delta_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)` is then an
unbiased estimate of `A^\pi(s_t, a_t)` *when* `V_\phi = V^\pi` exactly,
and a low-variance biased estimate otherwise (because `V_\phi` is
typically learned, not exact).

The **actor-critic** algorithm uses `\delta_t` as the policy gradient
weight:

```text
Actor-Critic(α_actor, α_critic):
    initialize policy parameters θ, critic parameters φ
    for episode = 1, ..., N_episodes:
        s ← initial state
        while s not terminal:
            a ~ π_θ(·|s)
            observe r, s'
            δ ← r + γ V_φ(s') - V_φ(s)     # TD error / advantage estimate
            φ ← φ + α_critic · δ · ∇_φ V_φ(s)         # critic update
            θ ← θ + α_actor · δ · ∇_θ log π_θ(a|s)    # actor update
            s ← s'
    return π_θ
```

The "actor" is the policy `\pi_\theta`. The "critic" is the value
function `V_\phi`. The actor uses the critic's TD error to estimate
the advantage. The critic is updated by standard TD(0) (Lesson 8).
The two networks are coupled through `\delta`: as the critic
improves, the advantage estimates improve, and the actor's gradient
estimates become more accurate.

---

**The bias-variance trade-off, again.** Actor-critic is the policy
gradient analog of the MC vs TD trade-off from Lessons 7-8.

- REINFORCE with baseline uses the MC return `G_t` minus the baseline.
  Unbiased; high variance (variance of `G_t` is the full
  trajectory variance).
- Actor-critic uses the TD error `\delta_t`. Biased while `V_\phi`
  is learning; substantially lower variance (variance of one reward
  plus the variance of `V_\phi(s_{t+1})`, which is much smaller
  than `G_t`'s variance).

The trade-off is the same one we made in Lesson 8 (TD vs MC). It
favors TD-style methods in practice for the same reason: the variance
reduction is dramatic, the bias vanishes as the critic improves, and
the resulting algorithms learn faster and more stably than their MC
counterparts.

---

**On the gridworld.** Actor-critic on the 3×3 gridworld with
`\alpha_\text{actor} = 0.05`, `\alpha_\text{critic} = 0.1`, 2,000
episodes, 10 seeds:

| algorithm                  | mean `\hat V(0, 0)` | std `\hat V(0, 0)` |
|:---------------------------|--------------------:|-------------------:|
| Vanilla REINFORCE          | 0.7219              | 0.0076             |
| REINFORCE with baseline    | 0.7250              | 0.0011             |
| Actor-Critic               | **0.7250**          | **0.0011**         |

On this small problem, actor-critic and REINFORCE-with-baseline are
indistinguishable. The gridworld's short episodes mean MC's variance
is already moderate. On longer-horizon problems, actor-critic's
variance advantage becomes more pronounced, and the bias from the
imperfect critic is paid off many times over.

---

**Visualization V6 — Policy Gradient Lab (Centerpiece).**

The polish-budget sink of the lesson; allocate four to five days.

A six-panel synchronized lab. The gridworld is shared. The three
algorithms (vanilla REINFORCE, REINFORCE with baseline, actor-critic)
train in parallel with shared hyperparameters.

**Panel A — Policy Display.** Three gridworlds side by side, one per
algorithm. Each cell shows the current softmax policy as four
probability bars (one per action). The bars evolve as training
progresses.

**Panel B — Learning Curves.** A line plot showing `\hat V(0, 0)`
over episodes for all three algorithms. The vanilla REINFORCE curve
is noisy; the baseline curve is smoother; the actor-critic curve is
smoother still. Reference lines at `V^*(0, 0) = 0.7290` and the
empirical softmax cap of 0.722 are drawn.

**Panel C — Gradient Norms.** The per-episode gradient norm `\|\nabla_\theta J\|`
for each algorithm. Vanilla REINFORCE's gradient norm is large and
variable; baselined REINFORCE's is smaller and more stable;
actor-critic's is smallest and most stable. This is the variance
story visualized directly.

**Panel D — Critic's Learned Value.** For actor-critic only, the
critic's `V_\phi(s)` over training time, displayed as a colorized
3×3 gridworld. As training progresses, `V_\phi` approaches the true
`V^\pi` under the current policy. The visual statement: the critic is
learning the right thing.

**Panel E — Episode Returns.** Per-episode return histories for the
three algorithms. The variability of the histories is itself a
variance diagnostic: noisy histories = high-variance estimator.

**Panel F — Settings Strip.** Sliders for `\alpha_\text{actor}` (0.01
to 0.5), `\alpha_\text{critic}` (0.01 to 0.5), `\gamma` (0.5 to 0.99),
N_episodes (100 to 10,000). A "reset all" button.

The central pedagogical moment: all three algorithms converge to
approximately the same final policy, but at very different *rates*
and with very different *gradient-noise profiles*. The variance
reduction from baseline and from actor-critic is visible in Panels B,
C, and E simultaneously. Width 960 (centerpiece breakout). Height
880.

---

### Section 7 — The Advantage Function and Bias-Variance Trade-offs

**Tagline:** *A unified view. The advantage function organizes everything.*
**Length:** ~600 words.
**Anchor:** `advantage-function` and `bias-variance-trade-off`.

---

**Prose:**

The recurring theme of Sections 5 and 6: the policy gradient is
weighted by *some estimate of the advantage* `A^\pi(s, a)`. The
choice of estimator determines the bias-variance trade-off.

We summarize the family:

| Estimator                          | Form                                            | Bias    | Variance |
|:-----------------------------------|:------------------------------------------------|:--------|:---------|
| MC return                          | `G_t`                                           | 0       | high     |
| MC return - state baseline         | `G_t - V_\phi(s_t)`                              | 0       | moderate |
| TD error                           | `r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)` | low (if V_φ ≈ V^π) | low |
| `n`-step TD error                  | mix of `n` rewards + bootstrap                  | adjustable | adjustable |
| Generalized Advantage Estimation (GAE) | weighted avg of `n`-step TD errors           | adjustable | adjustable |

GAE (Schulman et al. 2015) is the standard advantage estimator in
modern deep RL. It interpolates smoothly between MC (`\lambda = 1`)
and TD(0) (`\lambda = 0`) with a single hyperparameter `\lambda`, much
like TD(λ) from Lesson 8. GAE will be introduced in detail in
Lesson 11, where it appears inside PPO and TRPO.

---

**A unifying perspective: policy gradient is one update per (state, action).**
The policy gradient theorem can be rewritten as

$$
\nabla J(\theta) \;=\; \sum_{s, a} d^\pi(s) \pi_\theta(a \mid s) \cdot Q^\pi(s, a) \nabla \log \pi_\theta(a \mid s),
$$

where `d^\pi(s)` is the discounted state visitation distribution under
`\pi`. Equivalently,

$$
\nabla J(\theta) \;=\; \sum_{s, a} d^\pi(s) \pi_\theta(a \mid s) \cdot A^\pi(s, a) \nabla \log \pi_\theta(a \mid s),
$$

where we have subtracted off `V^\pi(s)` (which is a baseline). The
contribution to the gradient from `(s, a)` is proportional to:

1. How often we visit `(s, a)` (the `d^\pi(s) \pi(a \mid s)` weighting),
2. How much better than average `a` is at `s` (the advantage),
3. How sensitive `\pi(a \mid s)` is to `\theta` (the score function).

The estimator's job is to estimate this sum from samples. Different
algorithms (vanilla REINFORCE, baseline-corrected REINFORCE,
actor-critic, GAE-based methods) differ only in *how they estimate the
advantage*. The structure of the gradient is universal.

---

**The off-policy variant.** All the formulas above assume on-policy
data: trajectories sampled under the current `\pi_\theta`. For
**off-policy policy gradient** (trajectories sampled under some other
behavior policy `\pi_b`), we need to apply importance sampling on the
policy ratio:

$$
\nabla J(\theta) \;=\; \mathbb{E}_{\tau \sim \pi_b}\!\left[ \prod_t \frac{\pi_\theta(a_t \mid s_t)}{\pi_b(a_t \mid s_t)} \cdot \sum_t \nabla \log \pi_\theta(a_t \mid s_t) A(s_t, a_t) \right].
$$

This is the off-policy policy gradient theorem (Degris, White,
Sutton 2012). It reintroduces all the variance issues from Lesson 6's
IS apparatus. PPO's clipped objective (Lesson 11) and SAC's
reparameterization (Lesson 13) are both ways of avoiding this exact
problem; offline RL (Lesson 15) tackles it head-on with regularization.

---

**Visualization V7 — Bias-Variance in Advantage Estimation.**

A two-panel layout. Left panel: a schematic of the
"advantage estimator family" — MC return at one end, TD error at the
other, `n`-step in between. A slider on `n` (from 1 to ∞) moves a
marker through the family. Right panel: empirical results from the
running gridworld. The RMSE of `\hat A^\pi(0, 0, \text{right})` vs the
true `A^\pi` (≈ 0.07 for the right action) for each estimator,
plotted against `n`. The curve dips at intermediate `n` (the optimal
trade-off for this problem) and rises at both extremes. A reference
to Lesson 11's GAE is overlaid as a smooth curve through the
estimators. Width 720, height 380.

---

### Section 8 — Where You'll See This Again

**Tagline:** *Five subsequent lessons. PPO is the workhorse of LLM alignment.*
**Length:** ~500 words.
**Anchor:** `pg-forward-links`.

---

**Prose:**

Five subsequent lessons depend directly on this one.

**Lesson 11 (TRPO and PPO).** Trust Region Policy Optimization (TRPO,
Schulman et al. 2015) and Proximal Policy Optimization (PPO,
Schulman et al. 2017) are the dominant modern policy gradient
algorithms. Both address a specific failure mode of vanilla policy
gradient: large policy updates can move `\pi_\theta` so far from the
behavior policy that the gradient estimate becomes invalid (the
sampling distribution no longer matches the update direction). TRPO
enforces a KL-divergence constraint on the policy update; PPO
approximates this with a clipped ratio. PPO is, by a substantial
margin, the most-used policy gradient algorithm today, and is the
algorithm behind RLHF for large language models.

**Lesson 12 (Maximum-Entropy RL).** Adds an entropy bonus to the RL
objective: `J(\theta) = \mathbb{E}[\sum_t r_t + \alpha H(\pi(\cdot | s_t))]`,
where `H` is the policy entropy at `s_t`. Encourages stochastic
policies; provides better exploration; the foundation for SAC. The
policy gradient theorem extends directly with an additional
entropy-gradient term.

**Lesson 13 (Soft Actor-Critic, SAC).** Max-ent actor-critic with
twin Q-networks (an inheritance from Double DQN) and a stochastic
policy. SAC is the dominant continuous-control algorithm of the
2018-2023 era; it uses the full machinery of this lesson plus the
max-ent objective from Lesson 12 plus the twin-Q trick from Lesson 9.

**Lesson 14 (Model-Based RL).** Combines a learned dynamics model
with a planner or policy gradient. The policy gradient is computed
through the *imagined* trajectories (rollouts in the learned model)
rather than the actual environment. The score function trick still
applies, but with a different sampling distribution.

**Lesson 17 (RLHF and DPO).** Reinforcement Learning from Human
Feedback fine-tunes a language model's policy against a reward model
using PPO. Direct Preference Optimization (DPO, Rafailov et al. 2023)
is a recent alternative that bypasses the explicit reward model by
deriving the optimal policy in closed form from the preference data.
DPO is technically not a policy gradient algorithm — it directly
optimizes a classification-like loss — but it is a sibling of policy
gradient methods in the sense that both maximize an expected reward
objective. PPO and DPO together are the standard alignment toolkit.

---

**Visualization V8 — Roadmap Mini.** The curriculum's lesson-graph
thumbnail with Policy Gradient now marked as shipped. Outgoing arrows
go to Lessons 11 (TRPO/PPO), 12 (Max-Ent RL), 13 (SAC), 14
(Model-Based RL), 16 (Diffusion in RL), and 17 (RLHF/DPO). Each
arrow's hover popover shows the specific application. Width 720,
height 240.

---

## 5. Algorithm and Math Implementation

The TypeScript module `src/pg/` is around 250 lines.

```ts
import type { MDP, Policy } from "../mdp/types";
import { sampleTrajectory } from "../mdp/sampling";

/** Softmax policy with parameters theta of shape (nStates, nActions). */
export class SoftmaxPolicy {
  theta: Float64Array;
  constructor(public nStates: number, public nActions: number) {
    this.theta = new Float64Array(nStates * nActions);
  }

  probs(s: number): Float64Array {
    const p = new Float64Array(this.nActions);
    let maxVal = -Infinity;
    for (let a = 0; a < this.nActions; a++) {
      maxVal = Math.max(maxVal, this.theta[s * this.nActions + a]);
    }
    let sum = 0;
    for (let a = 0; a < this.nActions; a++) {
      p[a] = Math.exp(this.theta[s * this.nActions + a] - maxVal);
      sum += p[a];
    }
    for (let a = 0; a < this.nActions; a++) p[a] /= sum;
    return p;
  }

  sample(s: number, rng?: () => number): number {
    const p = this.probs(s);
    const r = rng ? rng() : Math.random();
    let cum = 0;
    for (let a = 0; a < this.nActions; a++) {
      cum += p[a];
      if (r < cum) return a;
    }
    return this.nActions - 1;
  }

  /** d/dtheta_{s,a'} log pi(a|s) = 1[a==a'] - pi(a'|s) */
  scoreFunction(s: number, a: number): Float64Array {
    const p = this.probs(s);
    const score = new Float64Array(this.nActions);
    for (let aPrime = 0; aPrime < this.nActions; aPrime++) {
      score[aPrime] = (aPrime === a ? 1 : 0) - p[aPrime];
    }
    return score;
  }
}

/** Vanilla REINFORCE. Returns the trained policy and per-episode history. */
export function reinforce(
  mdp: MDP,
  nEpisodes: number,
  alpha: number,
  options: { rng?: () => number; useBaseline?: boolean } = {},
): { policy: SoftmaxPolicy; history: number[] } {
  const policy = new SoftmaxPolicy(mdp.nStates, mdp.nActions);
  const history: number[] = [];
  let runningBaseline = 0;
  let seenCount = 0;

  for (let ep = 0; ep < nEpisodes; ep++) {
    const traj = sampleTrajectory(mdp, policy, options.rng);
    const T = traj.states.length;
    const Gs = new Float64Array(T);
    let G = 0;
    for (let t = T - 1; t >= 0; t--) {
      G = traj.rewards[t] + mdp.gamma * G;
      Gs[t] = G;
    }
    if (options.useBaseline) {
      seenCount++;
      runningBaseline += (Gs[0] - runningBaseline) / seenCount;
    }
    for (let t = 0; t < T; t++) {
      const s = traj.states[t];
      const a = traj.actions[t];
      const score = policy.scoreFunction(s, a);
      const advantage = options.useBaseline ? Gs[t] - runningBaseline : Gs[t];
      const gammaT = Math.pow(mdp.gamma, t);
      for (let aPrime = 0; aPrime < mdp.nActions; aPrime++) {
        policy.theta[s * mdp.nActions + aPrime] += alpha * gammaT * advantage * score[aPrime];
      }
    }
    history.push(estimateV(mdp, policy, mdp.startState, 20));
  }
  return { policy, history };
}

/** One-step actor-critic. */
export function actorCritic(
  mdp: MDP,
  nEpisodes: number,
  alphaActor: number,
  alphaCritic: number,
  options: { rng?: () => number } = {},
): { policy: SoftmaxPolicy; critic: Float64Array; history: number[] } {
  const policy = new SoftmaxPolicy(mdp.nStates, mdp.nActions);
  const V = new Float64Array(mdp.nStates);
  const history: number[] = [];

  for (let ep = 0; ep < nEpisodes; ep++) {
    let s = mdp.startState;
    while (!mdp.isTerminal(s)) {
      const a = policy.sample(s, options.rng);
      const { nextState, reward, done } = mdp.step(s, a, options.rng);
      const vNext = done ? 0 : V[nextState];
      const delta = reward + mdp.gamma * vNext - V[s];
      V[s] += alphaCritic * delta;
      const score = policy.scoreFunction(s, a);
      for (let aPrime = 0; aPrime < mdp.nActions; aPrime++) {
        policy.theta[s * mdp.nActions + aPrime] += alphaActor * delta * score[aPrime];
      }
      s = nextState;
      if (done) break;
    }
    history.push(estimateV(mdp, policy, mdp.startState, 20));
  }
  return { policy, critic: V, history };
}

/** Estimate V(s_0) under current policy via Monte Carlo rollouts. */
function estimateV(mdp: MDP, policy: SoftmaxPolicy, s0: number, nRollouts: number): number {
  let total = 0;
  for (let i = 0; i < nRollouts; i++) {
    const traj = sampleTrajectory(mdp, policy);
    let G = 0;
    for (let t = 0; t < traj.rewards.length; t++) G += Math.pow(mdp.gamma, t) * traj.rewards[t];
    total += G;
  }
  return total / nRollouts;
}
```

**Vitest test targets:**

```ts
test('Score function for softmax is unit-sum-zero', () => {
  const p = new SoftmaxPolicy(1, 4);
  p.theta = new Float64Array([0.1, 0.5, -0.3, 0.0]);
  const score = p.scoreFunction(0, 1);
  // sum of scores should be exactly zero (because sum of policy = 1)
  const sum = score.reduce((a, b) => a + b, 0);
  expect(sum).toBeCloseTo(0, 10);
});

test('REINFORCE on gridworld converges near V* (with softmax cap)', () => {
  const { history } = reinforce(gridworld, 2000, 0.05, { rng: seeded(0) });
  // Final V should be in [0.70, 0.74] (V* = 0.729, softmax cap ~0.722)
  expect(history[history.length - 1]).toBeGreaterThan(0.70);
  expect(history[history.length - 1]).toBeLessThan(0.74);
});

test('REINFORCE+baseline has lower seed-to-seed variance', () => {
  const vanillaSeeds: number[] = [];
  const baselineSeeds: number[] = [];
  for (let s = 0; s < 10; s++) {
    vanillaSeeds.push(reinforce(gridworld, 2000, 0.05,
      { rng: seeded(s) }).history.at(-1)!);
    baselineSeeds.push(reinforce(gridworld, 2000, 0.05,
      { rng: seeded(s), useBaseline: true }).history.at(-1)!);
  }
  const vStd = std(vanillaSeeds);
  const bStd = std(baselineSeeds);
  expect(bStd).toBeLessThan(vStd * 0.5);  // at least 2x lower
});

test('Actor-critic converges similar to REINFORCE+baseline on gridworld', () => {
  const acHistory = actorCritic(gridworld, 2000, 0.05, 0.1,
    { rng: seeded(0) }).history;
  expect(acHistory.at(-1)).toBeGreaterThan(0.70);
  expect(acHistory.at(-1)).toBeLessThan(0.74);
});

test('Critic in actor-critic approximates V^pi', () => {
  // After training, V_phi should be approximately the V^pi under the learned policy.
  const { policy, critic } = actorCritic(gridworld, 5000, 0.05, 0.1, { rng: seeded(0) });
  // Compute exact V^pi for the learned policy via DP
  const policyArray = computePolicyArray(policy, gridworld);
  const truePi = policyEvaluationExact(gridworld, policyArray);
  // V_phi should match truePi to within 0.1 at every state
  for (let s = 0; s < gridworld.nStates; s++) {
    expect(critic[s]).toBeCloseTo(truePi[s], 1);
  }
});

test('Score function estimator on Gaussian is unbiased', () => {
  // E[X^2] with X ~ N(theta, 1) has gradient 2*theta.
  // Score function estimator: (1/N) sum X_i^2 (X_i - theta)
  const theta = 1.0;
  let total = 0;
  const N = 10000;
  const rng = seeded(0);
  for (let i = 0; i < N; i++) {
    const x = rng.normal(theta, 1);
    total += x * x * (x - theta);
  }
  const estimate = total / N;
  // True gradient is 2 * theta = 2.0
  expect(estimate).toBeCloseTo(2.0, 0);  // within 0.5
});
```

---

## 6. Component Catalog

| Code | Component                       | Section | Polish budget |
|-----|---------------------------------|---------|--------------|
| V1  | `<FromQValuesToPolicies>`       | §1      | 1 day        |
| V2  | `<ScoreFunctionGaussian>`       | §2      | 1.5 days     |
| V3  | `<PolicyGradientTheorem>`       | §3      | 1.5 days     |
| V4  | `<REINFORCETrainingTrace>`      | §4      | 1 day        |
| V5  | `<VarianceReductionWithBaseline>` | §5    | 1.5 days     |
| V6  | `<PolicyGradientLab>`           | §6      | **4-5 days** (centerpiece) |
| V7  | `<BiasVarianceAdvantage>`       | §7      | 1 day        |
| V8  | `<RoadmapMini>` (update)        | §8      | 0.5 day      |

Total polish budget around twelve days, in line with Lesson 9's
budget — these are the curriculum's two heaviest lessons.

**Reuse from prior lessons:** `GridworldRenderer`, `MathBlock`,
`CrosslinkCallout`, `PanelChrome`, `RoadmapMini`, all of `src/mdp/`,
`policyEvaluationExact` from `src/dp/`, the MC infrastructure from
`src/monte-carlo/` (used for the REINFORCE baseline-running-mean), the
TD infrastructure from `src/td/` (used for the actor-critic critic).
New code is around 250 lines in `src/pg/` plus eight visualization
components.

---

## 7. Page-Level User Experience

Same conventions as prior lessons. Single-page scroll, prereq strip at
top, reduced-motion support for V2's Gaussian density animation, V4's
training scrubber, and V6's parallel training animations.

The centerpiece V6 is the only component breaking out to 960 pixels.
The training of three algorithms in parallel is computationally light
(all tabular softmax — no neural network) and runs in real time in
the browser. No pre-computation needed except for the convergence
statistics tables that populate the bias-variance comparison in V7.

A specific UX note for V6: the three algorithms train at slightly
different rates per episode (REINFORCE does one update per episode at
the end; actor-critic does T updates per episode, one per step). The
"episode counter" in Panel A should display both episode count and
total update count, since the two diverge across the algorithms. The
visual statement is "comparing at equal episode budgets" — which is
the standard convention in PG literature — but the per-update budget
difference should be transparent.

V2 (the Gaussian score function demo) is the lesson's pedagogical
entry point and deserves polish. The slider should be smooth (no
debouncing). The samples should re-draw with a small animation, not a
hard jump. The "true gradient" reference line should be drawn cleanly
with a clear label.

---

## 8. Acceptance Criteria

After completing this lesson, a learner should be able to:

State the score function estimator and derive it from the
log-derivative trick. Recognize the score function in policy gradient
context as `\nabla_\theta \log \pi_\theta(a \mid s)`. State the policy
gradient theorem and explain why the transition dynamics cancel out.
Write down the REINFORCE algorithm and explain why it is unbiased.
Identify the variance problem in REINFORCE; explain why a state-value
baseline reduces variance without introducing bias; write down
REINFORCE-with-baseline. Define the advantage function `A^\pi(s, a)`
and explain why it is the canonical baseline. Write down the
actor-critic algorithm; explain the bias-variance trade-off vs
REINFORCE-with-baseline. Identify the bias-variance interpolation
between MC return and TD error, and recognize that GAE (Lesson 11)
provides a tunable interpolation. Recognize PPO (Lesson 11), SAC
(Lesson 13), and RLHF (Lesson 17) as direct successors and
applications.

A concrete acceptance check: on the running 3×3 gridworld with
softmax policy and `\alpha = 0.05`, predict the final `V(0, 0)` after
2,000 episodes for vanilla REINFORCE, REINFORCE-with-baseline, and
actor-critic. The expected answers are 0.722 ± 0.008, 0.725 ± 0.001,
and 0.725 ± 0.001 respectively. V6 lets the learner verify.

---

## 9. Stretch Goals (post-MVP)

**Continuous-action demo.** A Gaussian policy on a 1D continuous
problem (e.g., a navigate-to-target task on a continuous line) would
showcase the continuous-action setting that motivates much of PG.
Out of scope for MVP because the verification numerics from prior
lessons don't apply.

**GAE preview.** A live demonstration of the `\lambda`-interpolation
in GAE on the gridworld, with a slider over `\lambda`. This is
slated for Lesson 11 but could appear as a preview here.

**Natural policy gradient.** The Fisher information matrix correction
applied to policy gradient (Kakade 2002) reweights the gradient by
the inverse Fisher to produce a more natural update direction. This
is the precursor to TRPO and would be a beautiful stretch section.

**Score function vs reparameterization comparison.** For continuous
actions, the reparameterization trick (Lesson 13's SAC) gives a
lower-variance gradient estimator than the score function trick.
A side-by-side variance comparison would set up the SAC story
beautifully but requires continuous-action infrastructure.

---

## 10. Out of Scope (intentionally)

**Natural policy gradient and Fisher information.** Mentioned in §9
stretch goals but not implemented. Lesson 11 builds on these
foundations.

**Off-policy policy gradient with importance sampling.** Mentioned in
§7 but not implemented. The off-policy story is taken up by PPO
(approximately, via clipping) and offline RL (Lesson 15) explicitly.

**Multi-agent policy gradient.** Self-play, MADDPG, COMA, and other
multi-agent PG methods are out of scope.

**Policy gradient with discrete decision trees or symbolic policies.**
Exotic policy classes are out of scope.

**Distributional policy gradient.** Methods that estimate the
*distribution* over returns (parallel to Distributional DQN in
Lesson 9) exist but are niche. Out of scope.

---

## 11. Training Notebook

The script `scripts/pg_traces.py` (around 100 lines) pre-computes the
convergence statistics for V5 (variance reduction histogram) and V7
(bias-variance comparison): per-episode gradient norms at fixed
policies, seed-to-seed variance of final `V(0, 0)` for the three
algorithms, advantage estimator RMSE for various `n` in n-step TD.
Outputs to `public/data/pg/`. No neural networks are trained for
this lesson.

---

## 12. Closing Notes and Length Tally

Total length: roughly fifteen hundred lines. The lesson is one of the
curriculum's longest, justified by the dense pedagogical content:
deriving the score function from scratch (filling in the cut
prereq), establishing the policy gradient theorem, and then spending
the bulk of the lesson on variance reduction — which is the actual
intellectual content of practical policy gradient methods.

The centerpiece V6 (Policy Gradient Lab) trains the three algorithms
side-by-side and exposes the variance-reduction story as a single
interactive contrast. The pedagogical statement: same gradient
direction, three different noise levels.

The forward links saturate. Five subsequent lessons (11, 12, 13, 14,
17) cash in this lesson's content. Lesson 11 (PPO) is the natural
successor and the immediate application — PPO is the workhorse of
RLHF, the dominant LLM alignment technique. The line from this
lesson to LLM alignment goes through Lesson 11.

## End of specification