# Lesson 12 — Maximum-Entropy RL and RL-as-Inference

> Lesson 11 closed with the PPO objective containing a small entropy
> bonus `c_2 · H[π_θ(·|s)]` — a regularizer to prevent premature
> policy collapse. In this lesson we make that term the *primary*
> objective. The result is a different theory of reinforcement
> learning: instead of maximizing reward subject to a side
> stipulation that the policy not be too greedy, we maximize the
> expected sum of reward *plus* entropy. The optimal policy is no
> longer deterministic. The Bellman equation gets a logsumexp where
> it used to have a max. And — surprisingly, but importantly —
> applied naively to an episodic gridworld, the framework breaks in
> an instructive way: the agent at moderate temperature learns to
> *avoid the goal* because terminating forfeits future entropy.

> Where this slots in. After Lesson 11 (TRPO, PPO, trust regions) and
> before Lesson 13 (SAC, continuous-action max-ent actor-critic).
> This lesson supplies the foundation SAC will operationalize, and
> it foreshadows the KL-to-reference trick that Lesson 17 (RLHF)
> uses to sidestep the failure mode we expose here. The "softmax
> cap" we noted in Lesson 10 (REINFORCE converging to V(0,0)≈0.722
> rather than V*=0.729) gets reframed here as a feature: that
> stochastic policy is the *correct* fixed point of a soft
> Bellman operator at a small temperature.

---

## 0. Pedagogical Philosophy

**0.1 The reframe is the lesson.** Maximum-entropy RL is not a
collection of new algorithms layered on top of RL. It is a different
question: instead of "what policy maximizes expected return?", we
ask "what policy maximizes expected return *plus* expected entropy?"
This is not a technical extension; it is a philosophical shift, and
the lesson treats it as one. The first section is devoted to the
question of why anyone would want this objective in the first place
— what's wrong with deterministic optimal policies, and what's
right about preserving stochasticity.

**0.2 The connection to Lesson 10's softmax cap is exact.** REINFORCE
on the gridworld converged to V^π(0,0) ≈ 0.722 rather than V* =
0.729. Lesson 11 named this the "softmax cap" and called it a
representational artifact of finite-temperature softmax. This lesson
shows it is the *correct* answer to a different question: under
the entropy-regularized objective at α ≈ 0.02, the optimal policy
gives V^π(0,0) = 0.7217. The cap is the soft fixed point, not an
artifact. Once you see this, the L10 narrative changes: REINFORCE
isn't approaching V* and falling short; it's converging to the
soft-optimal policy at the small effective α induced by its
training dynamics.

**0.3 We are honest about the failure mode.** Applied to the 3×3
gridworld at α ≥ 0.1, the entropy-regularized policy deliberately
*does not reach the goal*. At α = 0.2, only ~5% of Monte Carlo
rollouts terminate at the goal; the other 95% are still walking
around the grid 500 steps later, having decided that the entropy
bonus from continued exploration is worth more than the +1 terminal
reward. This is a real failure mode of max-ent RL on episodic
tasks. The lesson does not hide it; it puts it in §5 as the central
empirical finding, and uses it to motivate the design choices in
SAC (Lesson 13, which uses infinite-horizon continuous control) and
RLHF (Lesson 17, which uses KL-to-reference rather than pure
entropy).

**0.4 Soft Bellman is still a contraction.** Lesson 4 established
the Bellman operator as a γ-contraction in sup norm; that result
is the foundation under every algorithm in Lessons 5–11. The soft
Bellman operator with a logsumexp where the max used to be is
*also* a γ-contraction in sup norm. The argument is short, lives in
§3, and is one of the most reassuring small results in this lesson:
nothing structural changes. Soft VI converges; soft PI converges;
the fixed point is unique. The only thing that changes is what
fixed point you converge to.

**0.5 The "RL as inference" view is included, but as one lens
rather than the whole story.** A large literature treats max-ent RL
as variational inference in a graphical model with an "optimality"
variable. The mapping is exact and elegant: the posterior over
actions given optimality observed is the soft-optimal policy. The
lesson covers this in §7 with one full derivation and one
visualization. It does not become the dominant framing, however,
because the inference view often obscures more than it reveals for
practical algorithms. The objective-and-Bellman view is primary;
the inference view is a perspective worth knowing.

**0.6 Forward links carry weight.** L13 (SAC) operationalizes
everything in this lesson for continuous action spaces; L17 (RLHF)
uses a closely-related KL-constrained objective. The forward-link
section explicitly maps which piece of L12 reappears where. Given
the trimmed curriculum (L14, L15, L16 deferred), L12 is one of the
last places to set up the conceptual scaffolding that L17 will rely
on; the forward links carry more weight than usual.

---

## 1. Tech Stack

Same as Lessons 10 and 11: Vite, TypeScript strict, plain Web
Components, KaTeX, D3 v7, `ml-matrix`, Vitest, Python 3.11+,
`seedrandom`. No new browser-runtime dependencies. The lesson is
entirely tabular — soft VI, soft PI, and the entropy-regularized
policy evaluation all run in `Float64Array` arithmetic. The
offline pre-computation script (`scripts/maxent_traces.py`) is
pure NumPy. No PyTorch in this lesson; that returns in Lesson 13.

---

## 2. Visual and Aesthetic Direction

```css
:root {
  --maxent-bg: #faf8f3;
  --maxent-text: #1c1e22;

  /* Algorithm colors — preserves curriculum threads */
  --maxent-hard: #15803d;       /* green: hard / deterministic optimum (V*) */
  --maxent-soft: #7c3aed;       /* violet: the entropy-reg distribution */
  --maxent-uniform: #6b7280;    /* gray: the uninformative limit */
  --maxent-entropy: #be185d;    /* pink: the entropy bonus (interpolator) */

  /* Bridging colors */
  --maxent-alpha-low: #15803d;  /* green: alpha -> 0 limit */
  --maxent-alpha-mid: #7c3aed;  /* violet: useful regime */
  --maxent-alpha-high: #ea580c; /* orange: failure regime (avoids goal) */

  /* Outcome highlights */
  --maxent-cap: #db2777;        /* pink: the "softmax cap" from L10 */
  --maxent-failure: #dc2626;    /* red: the goal-avoidance failure */

  /* Inference view */
  --maxent-likelihood: #0e7490; /* cyan: the "optimality" observation */
  --maxent-posterior: #7c3aed;  /* violet (same as soft): the posterior policy */
}
```

The two threads from Lesson 10 — violet for the principled
distribution being learned, green for the hard optimum — are made
explicit here. Violet *is* the entropy-regularized policy; green
*is* the hard V*. The two are on a spectrum parametrized by α, and
we use a perceptual gradient (green → violet → orange) to show the
α axis on plots that sweep across temperatures. Pink continues to
indicate an interpolator (here, the entropy term, which interpolates
between reward maximization and uniform behavior — the same role
pink played for n-step TD in L8 and GAE in L11).

Red appears in this lesson for the first time outside of L9's
deadly triad: it indicates the *failure regime* (α ≥ 0.2 on the
gridworld). The L9 deadly-triad red is "divergence under function
approximation"; the L12 red is "the policy avoids the goal." Both
are warning colors; the visual rhyme is intentional.

The "softmax cap" gets its own pink shade (`--maxent-cap`) and is
used in §1 and §6 to highlight the L10 connection on plots.

---

## 3. Lesson Metadata

```typescript
import type { LessonMeta } from '../../shared/lesson-types';

export const lesson12Meta: LessonMeta = {
  id: 'l12-max-ent-rl',
  title: 'Maximum-Entropy RL and RL-as-Inference',
  subtitle: 'When stochasticity is the objective, not the regularizer',
  tier: 2,
  difficulty: 'advanced',
  estimatedReadMinutes: 60,
  role: 'theory-bridge',
  prerequisites: [
    'l11-trpo-ppo',           // the entropy bonus that motivates this
    'l10-policy-gradient',    // the softmax cap reframed
    'l8-td-learning',         // soft Bellman is still a contraction
    'l5-dynamic-programming', // soft VI/PI structure
    'l4-contractions',        // the contraction-in-sup-norm framework
  ],
  exportedAnchors: {
    // Verified in scripts/maxent_traces.py
    'soft-V-alpha-0-recovers-V-star': 0.7290,
    'V-soft-alpha-0.02': 0.7392,         // soft fixed-point V_soft at alpha=0.02
    'V-pi-alpha-0.02': 0.7217,           // true V^pi at alpha=0.02 (the L10 cap)
    'V-pi-alpha-0.05': 0.5928,
    'V-pi-alpha-0.1': 0.0711,
    'V-pi-alpha-0.2': 0.0006,
    'goal-reach-prob-alpha-0.05': 1.000,
    'goal-reach-prob-alpha-0.1': 0.999,
    'goal-reach-prob-alpha-0.2': 0.051,
    'goal-reach-prob-alpha-0.5': 0.001,
    'mean-steps-alpha-0.1': 78.1,
    'mean-steps-alpha-0.2': 252.8,
    'softmax-cap-effective-alpha': 0.0198,
  },
  centerpieceComponent: 'EntropySliderLab',
  forwardLinksWhenReady: [
    { lessonId: 'l13-sac', anchor: 'continuous-soft-actor-critic' },
    { lessonId: 'l17-rlhf-dpo', anchor: 'kl-to-reference-objective' },
  ],
};
```

---

## 4. Section-by-Section Plan

### Section 1 — Why Stochasticity Might Be the Objective

**Tagline:** *The hard optimum isn't always the right target.*
**Length:** ~600 words
**Anchor:** `why-stochastic`

The standard RL objective is to find a policy maximizing expected
return: π* = arg max_π E^π[Σ_t γ^t r_t]. The optimal policy under
this objective is almost always deterministic (when the MDP is
finite, the optimal policy can be chosen to be deterministic
without loss). On the gridworld, π* picks "right" with probability
1 in state (0,0). This is the answer Lessons 3, 5, 8 produced.

But the reader's voice, slightly skeptical: *Is determinism
actually what we want?*

We make three cases that it might not be.

First, **robustness**. A deterministic policy is brittle: if any
state-transition probability is slightly different from what the
agent thinks, or if the reward function has measurement noise, the
deterministic policy may be far from optimal under the true
dynamics. A stochastic policy has built-in robustness — it
explores neighboring strategies "by default."

Second, **exploration**. Lessons 1–10 treated exploration as a
separate concern (ε-greedy, decaying schedules, GLIE conditions in
L7-L8). A stochastic optimal policy folds exploration into the
objective itself: the policy doesn't *need* an ε-schedule because
it never collapses to a single action.

Third, **multimodality**. If two actions are nearly equally good,
a deterministic policy must choose one. The choice is arbitrary
and can lead to brittle commitment. A stochastic policy
acknowledges multimodality and hedges.

We then preview the punchline of the lesson: the softmax cap from
Lesson 10. REINFORCE on this gridworld converged to V^π(0,0) ≈
0.722. We promised in L11 that L12 would reframe this — and the
reframe is that the cap is the *exactly correct* value of the
optimal policy under an entropy-regularized objective at a small
temperature. Specifically, we verify in §6 that at α = 0.02, the
entropy-regularized optimal policy has V^π(0,0) = 0.7217. The cap
is not an artifact of representation; it is the right answer to a
slightly different question.

Visualization V1 — The Hard/Soft Spectrum.
A single-panel illustration. X-axis: α from 0 (hard) to 1 (high
entropy), on a log scale. Y-axis: true V^π(0,0) under the optimal
policy at that α. The curve starts at 0.729 (V*), passes through
0.722 (the L10 cap) at α ≈ 0.02, descends through ~0.59 at α=0.05,
plummets near α=0.1, and hits ~0 at α=0.2. Two vertical reference
lines: green at α=0 ("hard RL"), pink at α=0.02 ("L10 softmax
cap"). The "useful" regime where the policy is stochastic but
still goal-directed is shaded violet; the "failure" regime where
the policy stops reaching the goal is shaded red. 540 × 280 px.
Polish budget: 1.5 days.

---

### Section 2 — The Entropy-Regularized Objective

**Tagline:** *Reward plus entropy, weighted by α.*
**Length:** ~700 words
**Anchor:** `entropy-regularized-objective`

We define the objective formally:

$$
J_\alpha(\pi) \;=\; \mathbb{E}^\pi\!\left[
  \sum_{t=0}^{\infty} \gamma^t \bigl( r(s_t, a_t) + \alpha\, \mathcal{H}(\pi(\cdot | s_t)) \bigr)
\right],
$$

where `H(p) = -Σ_a p(a) log p(a)` is the Shannon entropy in nats.
The parameter `α ≥ 0` is the *temperature*. The objective is the
sum of two things: expected discounted reward (the familiar RL
objective) and expected discounted entropy of the policy at each
state.

The reader's voice: *Wait — that's a weird mixture. Reward is
measured in whatever units the task gives. Entropy is measured in
nats. How can we add them?*

That is exactly the role of α: it converts nats into reward-units.
A larger α means a single nat of entropy is "worth" more reward.
Two limits clarify:

- **α = 0**: only reward matters. The objective recovers standard
  RL. The optimal policy is deterministic.
- **α → ∞**: only entropy matters. The optimal policy maximizes
  per-state entropy subject to the dynamics. For a finite action
  space with no constraints, this is the uniform policy. (For more
  general settings, see §5 for what happens with discounted
  episodic tasks.)

At intermediate α, the policy is stochastic — sharper than uniform
in states where one action is much better than the others, closer
to uniform in states where actions are nearly tied. The exact
shape we will derive in §3 and §4.

A subtle but important point. The entropy term in the objective is
the entropy of `π(·|s)` *at each state visited*, not the entropy
of the entire trajectory. These are different things: trajectory
entropy includes contributions from environment stochasticity,
which the agent cannot control. State-conditional entropy is what
the agent controls. Lesson 11's PPO entropy bonus was the same
thing, just weighted differently. Continuing to L13, SAC also uses
state-conditional entropy. L17's RLHF will swap this term for a
KL penalty to a reference policy, which is a different but related
information-theoretic quantity.

A second subtlety: the discount factor γ multiplies *both* the
reward and the entropy. This is the standard formulation. An
alternative is to apply γ only to reward; this leads to different
fixed points but is rarely used in practice. We use the standard
formulation throughout.

We close §2 with the visualization of the objective surface.

Visualization V2 — The Objective Surface.
For the simplest case (a single state with two actions, like a
1-step bandit), the objective `J(π) = π_1 r_1 + π_2 r_2 + α H(π)`
is a function of `π_1` (with `π_2 = 1 - π_1`). User picks `r_1 -
r_2` and α. Three curves overlaid:

- The reward-only term `π_1 r_1 + π_2 r_2` (green, linear in π_1).
- The entropy term `α H(π)` (pink, concave with max at π_1 = 0.5).
- Their sum, the objective J (violet, concave with max somewhere
  between 0.5 and 1 depending on r and α).

The user slides α from 0 to 1 and watches the maximum migrate from
"all-in on the better action" at α=0 to "perfectly balanced" at
α → ∞. 480 × 280 px. Polish budget: 1.5 days.

---

### Section 3 — The Soft Bellman Operator

**Tagline:** *Logsumexp where the max used to be.*
**Length:** ~800 words
**Anchor:** `soft-bellman`

The Bellman optimality operator from L5 was

$$
(T V)(s) \;=\; \max_a\, \bigl[ r(s,a) + \gamma\, \mathbb{E}_{s'}[V(s')] \bigr].
$$

For the entropy-regularized objective, the analogous operator is

$$
(T_\alpha V)(s) \;=\; \alpha \log \sum_a \exp\!\left(
  \frac{r(s,a) + \gamma\, \mathbb{E}_{s'}[V(s')]}{\alpha}
\right).
$$

The max has become a *logsumexp* at temperature α. This is the
**soft Bellman operator**. Two limits make the connection
transparent:

- As `α → 0`, the logsumexp concentrates entirely on its largest
  argument: `α log Σ exp(x/α) → max x`. So `T_α → T`. Standard
  Bellman.
- As `α → ∞`, the logsumexp washes out differences:
  `α log Σ exp(x/α) → α log |A| + (1/|A|) Σ x`. The operator
  averages, weighted heavily toward the entropy term.

The derivation that this operator gives the soft-optimal value
function takes one paragraph and one Lagrangian. The
entropy-regularized Bellman equation is:

$$
V_\alpha^*(s) \;=\; \max_\pi\, \mathbb{E}_{a \sim \pi}\!\left[
  Q_\alpha^*(s,a) + \alpha\, \mathcal{H}(\pi(\cdot|s))
\right],
$$

where `Q*_α(s,a) = r(s,a) + γ E_{s'}[V*_α(s')]`. Maximizing over
`π` subject to the simplex constraint gives the **Boltzmann
policy**:

$$
\pi_\alpha^*(a|s) \;=\;
\frac{\exp(Q_\alpha^*(s,a) / \alpha)}{\sum_{a'} \exp(Q_\alpha^*(s,a') / \alpha)},
$$

and substituting back yields exactly `V*_α(s) = α log Σ_a
exp(Q*_α(s,a)/α)` — the logsumexp form. (One-paragraph derivation
included in full.)

The reader's voice: *Is this still a contraction?*

Yes. The soft Bellman operator `T_α` is still a γ-contraction in
sup norm. The proof is one line of inequalities:

$$
\bigl| (T_\alpha V)(s) - (T_\alpha V')(s) \bigr|
\;=\; \bigl| \alpha \log Σ exp(\cdot V/α) - \alpha \log Σ exp(\cdot V'/α) \bigr|
\;\le\; \gamma\, \|V - V'\|_\infty.
$$

The middle inequality uses that the logsumexp at temperature α is
Lipschitz-1 in its arguments (a standard fact, derived in §3.2).
The γ comes from the expectation `γ E[V(s')]` in the argument.

Lesson 4 established that γ-contractions have unique fixed points;
that result transfers directly. Soft VI converges. The fixed point
is unique. The fixed point is the soft-optimal value function
`V*_α`. This is the reassuring small theorem we promised in §0.4.

We then describe soft value iteration as a numerical algorithm:
initialize `Q(s,a) = 0`, repeatedly apply `T_α`, stop when
`‖Q_new - Q_old‖_∞ < tol`. The implementation is a direct
modification of L5's `valueIteration` — six lines change.

A second algorithm, **soft policy iteration**, alternates two
steps:

1. Soft policy evaluation: solve the linear system `V^π = (I - γ
   P^π)^{-1} R^π_soft` where `R^π_soft(s) = Σ_a π(a|s) r(s,a) +
   α H(π(·|s))`.
2. Soft policy improvement: `π_new(a|s) ∝ exp(Q^π(s,a) / α)`.

Like classical PI, soft PI is monotone in `V^π` and converges to
the soft-optimal policy. We verify empirically in §6 that soft VI
and soft PI converge to the same fixed point (they do, exactly).

Visualization V3 — Soft VI Convergence.
A side-by-side: left panel shows V_soft(s) for each non-terminal
state as a function of iteration count, for α=0.05 (and a toggle
for α=0.1, α=0.5). All curves smoothly converge to the fixed
point. Right panel shows the policy at state 0 as bar chart: bars
animate from uniform (initial) to the Boltzmann fixed point. The
animation is "scrubbable" via a slider. 600 × 280 px. Polish
budget: 1.5 days.

---

### Section 4 — The Boltzmann Policy

**Tagline:** *Soft-optimal policies are Boltzmann distributions.*
**Length:** ~600 words
**Anchor:** `boltzmann-policy`

We pause to look closely at the soft-optimal policy itself:

$$
\pi_\alpha^*(a|s) \;=\;
\frac{\exp(Q_\alpha^*(s,a) / \alpha)}{Z_\alpha(s)},
\qquad
Z_\alpha(s) = \sum_{a'} \exp(Q_\alpha^*(s,a') / \alpha).
$$

Three observations.

**The policy is parametrized by Q.** Once you have `Q*_α`, the
policy is determined by a softmax. This is the cleanest possible
connection between value-based and policy-based methods: in
max-ent RL, value learning and policy learning are *the same
problem*. SAC (L13) leverages this: it learns Q soft and derives
the policy by softmax-projecting Q. There is no separate "policy
network" in the sense of L10 — the policy is a function of Q.

**The temperature sets the sharpness.** Small α makes the softmax
concentrate on the argmax (recovering greedy as `α → 0`); large α
flattens it toward uniform. The "right" α is task-dependent and is
itself an object of optimization in modern SAC variants. The
canonical SAC v2 (Haarnoja et al. 2018b) learns α automatically by
constraining the policy's average entropy to a target.

**Z_α(s) is a partition function.** This will turn out to be the
same partition function that appears in the RL-as-inference view
(§7): the normalizer of the posterior over actions given
optimality. The math we are doing is statistical mechanics in
disguise. Many of the names in this lesson — partition function,
Boltzmann distribution, temperature — come from this connection.

Computation note: the partition function `Z_α(s)` becomes numerically
unstable to compute directly when `Q/α` is large. The standard
trick is the **log-sum-exp shift**: subtract `max_a Q(s,a)` before
exponentiating, then add it back outside the log. This is the
*same* numerical trick used in softmax over neural network logits.
We use it throughout the implementation.

We also write down the policy entropy explicitly:

$$
\mathcal{H}(\pi_\alpha^*(\cdot|s)) \;=\;
\log Z_\alpha(s) - \frac{1}{\alpha}\, \mathbb{E}_{a \sim \pi_\alpha^*}[Q_\alpha^*(s,a)].
$$

This is the standard "log Z minus expected energy over temperature"
identity from statistical mechanics. It is useful for two reasons:
(1) it lets us compute the entropy without summing `-p log p`
directly (which can be numerically tricky for sharp policies); and
(2) it shows that the entropy decreases as α decreases (sharper
policy → smaller log Z relative to E[Q]), recovering the
expected behavior.

Visualization V4 — Boltzmann Policy Across α.
A 3×3 grid of small bar charts — one per gridworld state — showing
the Boltzmann policy `π*_α(a|s)` at the user-selected α. As the
user moves the α slider, all 9 bar charts animate. At α → 0, the
charts become delta-spikes (the optimal action gets all the
probability mass). At large α, the charts flatten toward uniform.
The "interesting" regime to land at is α ≈ 0.05–0.1, where some
states show sharp policies (states near the goal) and others show
nearly-uniform policies (states far from the goal, where actions
are nearly tied). 540 × 360 px. Polish budget: 2 days.

---

### Section 5 — The Failure Mode

**Tagline:** *What happens when entropy starts to dominate.*
**Length:** ~800 words
**Anchor:** `maxent-failure-mode`

We have established the theory. The soft Bellman operator is a
contraction. The soft-optimal policy is a Boltzmann distribution.
Everything is mathematically clean. So we run soft VI on the
gridworld at α = 0.2 and ask: how often does this policy reach the
goal?

The answer is **5.1%**. Across 5,000 Monte Carlo rollouts from the
start state, the policy at α = 0.2 reaches the goal in 51 of them.
The other 4,949 trajectories are still wandering around the grid
500 steps in, having decided that the entropy bonus from continued
exploration is worth more than the +1 terminal reward.

The reader's voice: *Wait, that's broken.*

It is — in a specific and instructive way. The objective per step
is `r(s,a) + α H(π(·|s))`. The maximum possible per-step entropy
on a 4-action problem is `log 4 ≈ 1.386`. So the per-step entropy
bonus at α = 0.2 is up to `0.2 × 1.386 = 0.277` reward-units. Over
an unbounded discounted horizon, the total accumulated entropy
bonus from continued non-terminal motion is roughly `0.277 /
(1 - γ) = 2.77` reward-units. Compare to the +1 terminal reward
for reaching the goal: the entropy bonus from dawdling is worth
~2.77× the terminal reward. The optimal policy under this
objective is to *avoid the goal*.

The empirical evidence, from `scripts/maxent_traces.py`:

| α | Goal reach prob | Pit reach prob | Timeout prob | Mean steps to terminal |
|:---|:-----------------|:----------------|:--------------|:------------------------|
| 0.001 | 1.000 | 0.000 | 0.000 | 4.0 |
| 0.01 | 1.000 | 0.000 | 0.000 | 4.0 |
| 0.05 | 1.000 | 0.000 | 0.000 | 6.2 |
| 0.10 | 0.999 | 0.000 | 0.001 | 78.1 |
| 0.20 | 0.051 | 0.000 | 0.949 | 252.8 |
| 0.50 | 0.001 | 0.000 | 0.998 | 225.1 |

At α = 0.05, the policy reaches the goal but takes 6.2 steps on
average (vs 4 for the greedy policy) — a small inefficiency price
for stochastic exploration. At α = 0.1, the policy still reaches
the goal but takes 78 steps on average. At α = 0.2, the policy
fundamentally fails. By α = 0.5, only 0.1% of rollouts ever
terminate.

A look at the policy at α = 0.2 confirms the diagnosis. From state
0, the action probabilities are:

- up (bumps wall back to (0,0)): 0.286
- right (→ (0,1)): 0.214
- down (→ (1,0)): 0.214
- left (bumps wall back to (0,0)): 0.286

The wall-bumping actions are *preferred*. The agent has learned
that staying in place gives the maximum entropy bonus over time.
This is a real, well-known phenomenon in max-ent RL on episodic
tasks (Ziebart 2010; Haarnoja et al. 2017 discuss workarounds).

Three responses, leading to three later lessons.

**(a) Choose α small enough that this doesn't happen.** On the
gridworld, α ≤ 0.05 keeps the policy goal-directed. But the
"right" α is task-dependent, and tuning α by hand is exactly the
kind of hyperparameter sensitivity we hoped max-ent RL would
reduce. This is the most direct response and the most fragile.

**(b) Use infinite-horizon settings.** If episodes never
terminate, the agent doesn't gain anything by avoiding terminals
(there are no terminals to avoid). Continuous-control tasks like
pendulum, half-cheetah, etc. are typically infinite-horizon (the
"episode" ends only because of a wall-clock budget). SAC (L13)
operates in this regime and the failure mode does not appear. This
is the response of choice in deep continuous control.

**(c) Replace entropy with KL-to-reference.** Instead of
maximizing entropy `H(π(·|s))`, maximize `−KL(π(·|s) || π_ref(·|s))`
for some reference policy `π_ref`. The reference can be a uniform
policy (recovers entropy), but more usefully can be a "good
behavior" policy — a supervised-fine-tuned LLM, say. Now the agent
is rewarded for being close to the reference, not for being
uniform. This is the response in RLHF (L17). The failure mode
disappears because the reference is itself goal-directed; deviating
from it to dawdle in non-terminal states costs you in the KL
penalty.

The lesson concludes that max-ent RL on episodic discounted tasks
is a *useful theoretical framework* with a *known failure mode*
that motivates the design choices in two downstream algorithm
families. Naming the failure mode explicitly is the lesson's most
valuable pedagogical service.

Visualization V5 — The Failure Diagnostic.
Three synchronized panels, parameterized by α (slider).

Panel A: Goal-reach probability and pit-reach probability as bars,
plus a "timeout" bar in red. As α slides up past ~0.1, the green
"goal" bar shrinks and the red "timeout" bar grows.

Panel B: A small histogram of trajectory lengths (steps to
terminal, or 500 if timeout). At low α this is a sharp spike near
4. At α = 0.2 it is a flat distribution out to 500.

Panel C: The state-0 action probability bars, with the wall-bumping
actions (up, left) colored in `--maxent-failure` red when their
probability exceeds the goal-direct actions (right, down). The
"preference inversion" is visually obvious.

580 × 280 px. Polish budget: 2 days.

---

### Section 6 — Entropy Slider Lab (Centerpiece)

**Tagline:** *The full picture in one slider.*
**Length:** ~700 words
**Anchor:** `entropy-slider-lab`

The centerpiece. A six-panel interactive driven by a single global
α slider. Pre-computed soft VI is loaded from `public/data/maxent/`
at ~60 logarithmically-spaced α values; the slider snaps to the
nearest pre-computed point and updates all six panels.

**Panel A — Policy heatmap.** The 3×3 gridworld with the
Boltzmann policy rendered as four overlapping action-probability
bars in each cell. Background tint shows V_soft(s). Updates in
real time as α changes. 280 × 280 px.

**Panel B — Value bifurcation.** Two curves over the α-axis:
V_soft(start) (the soft fixed-point value, violet) and V^π(start)
(the true expected return under the soft-optimal policy, green).
At α = 0, both equal V* = 0.729. As α grows, the two curves
diverge: V_soft keeps growing (heading to infinity), V^π crashes
toward 0. The user's α-slider position is shown as a vertical
line. The L10 softmax cap (V^π = 0.722) is marked with a pink
horizontal reference line. 380 × 220 px.

**Panel C — Entropy / KL.** Two curves over the α-axis: mean
H(π_α(·|s)) (pink, the policy entropy averaged across non-terminal
states) and KL(π_α || uniform) (cyan, how far the policy is from
uniform). At α=0, H is near 0 (sharp policy) and KL is high. At
α→∞, H → log 4 and KL → 0. 380 × 220 px.

**Panel D — Goal-reach diagnostic.** Two stacked bars: green for
P(goal), gray for P(timeout), red border on the timeout bar in
the "failure" regime. Updates with the α-slider; pre-computed from
5,000 Monte Carlo rollouts at each α. 280 × 200 px.

**Panel E — Trajectory length histogram.** A live histogram of
steps to terminal, drawn from the Monte Carlo data. At low α, a
sharp peak near 4; at α=0.2, a flat distribution out to 500. 280 ×
200 px.

**Panel F — The "what is happening" panel.** A short auto-updated
caption based on the slider position:

- α ≤ 0.01: *"Tie-breaking regime: the policy is essentially
  greedy with slight stochasticity at states where two actions
  are equally good."*
- 0.01 < α ≤ 0.05: *"Useful regime: the policy is goal-directed
  but explores. This is where SAC would operate in continuous
  control."*
- 0.05 < α ≤ 0.1: *"Trade-off region: the agent takes longer
  paths to the goal. Entropy is paying a real cost in
  efficiency."*
- α > 0.1: *"Failure regime: the agent has learned to avoid the
  goal. The entropy bonus from continued exploration exceeds
  the +1 terminal reward."*

The auto-caption is the lesson's snap-into-place moment: as the
user slides α through the transition, the caption changes
character entirely, naming what's happening at each regime.

**Controls.** Just α, on a log slider from 0.001 to 1.0, with the
four named regimes marked as guide ticks.

960 × 660 px (breakout). Polish budget: **4 days** — the
polish-budget sink for this lesson.

---

### Section 7 — RL as Inference

**Tagline:** *The soft-optimal policy is a posterior.*
**Length:** ~700 words
**Anchor:** `rl-as-inference`

We end the theoretical content with a different lens on the same
math. The construction is due to Toussaint, Ziebart, and Levine;
the most accessible exposition is Levine 2018 (arXiv:1805.00909).

Define an **optimality variable** `O_t ∈ {0, 1}` at each time step.
The probabilistic model is:

- States evolve as in the MDP: `p(s_{t+1} | s_t, a_t)` from the
  dynamics.
- Actions are drawn from a uniform "prior" policy: `p(a_t | s_t) =
  1/|A|`.
- The optimality variable is observed: `p(O_t = 1 | s_t, a_t) =
  exp(r(s_t, a_t) / α)`. (We can choose units so the exponent
  is bounded; α serves the same role as before.)

Now we condition on observing `O_{0:∞} = 1` — "the trajectory is
optimal" — and ask for the posterior over actions:

$$
p(a_t | s_t, O_{t:\infty} = 1).
$$

The reader's voice: *That's a graphical-model question.*

It is. And the answer, by standard forward-backward inference, is
*exactly* the Boltzmann policy from §4:

$$
p(a_t | s_t, O_{t:\infty} = 1) \;=\; \frac{\exp(Q_\alpha^*(s_t, a_t) / \alpha)}{Z_\alpha(s_t)}.
$$

The "backward message" β_t(s) corresponds to `exp(V*_α(s)/α)`; the
soft Bellman equation is the backward recursion for this message;
the partition function is the marginal probability of optimality.
Every quantity we computed via soft VI corresponds to a quantity in
the inference view.

Three things this view buys us.

**(a) A clean Bayesian story.** "The agent is doing inference about
what action to take, given that the trajectory is optimal."
Pedagogically this resonates with people from a probabilistic-ML
background. Mechanistically it is the same algorithm.

**(b) Variational connections.** Approximating the posterior with a
parameterized policy `π_θ(a|s)` and minimizing a KL divergence
from the posterior gives a variational objective. Maximizing the
evidence lower bound (ELBO) of this objective is equivalent to
the entropy-regularized RL objective. SAC's policy gradient (L13)
can be derived from this ELBO directly.

**(c) Why "soft Q" is a posterior log-likelihood.** `Q*_α(s,a) / α`
behaves like a log-posterior over actions — high-Q actions are
"more likely" under the posterior. This explains why the
Boltzmann form is a softmax over Q/α: it is a posterior.

We do not derive the variational ELBO in full — that derivation
properly lives in L13 where it directly motivates SAC's actor
update. We sketch it (one paragraph) and forward-link.

A note on temperature interpretation. In the inference view, α is
the **inverse confidence** with which we believe the trajectory is
optimal. Small α means "almost surely optimal" — the likelihood
sharpens to a delta on the highest-Q action. Large α means
"weakly optimal" — the likelihood is nearly flat, and the posterior
defaults to the prior (uniform). This is the same temperature, but
interpreted as a Bayesian belief strength rather than a Lagrange
multiplier.

Visualization V6 — The Inference Graph.
A small probabilistic graphical model: nodes for `s_t`, `a_t`,
`O_t`, with arrows for the conditional dependencies. The
"observed" nodes (`O_t`) are shaded; the "latent" nodes (`a_t`)
are unshaded. Below the graph, an "unrolled" version showing the
forward and backward messages flowing through the chain. Hovering
on a node shows the conditional probability. The visualization is
illustrative, not interactive in the algorithmic sense. 540 × 280
px. Polish budget: 1.5 days.

---

### Section 8 — Forward Links

**Tagline:** *Toward SAC, toward RLHF.*
**Length:** ~500 words
**Anchor:** `l12-forward-links`

> **Forward link to L13 — Soft Actor-Critic.** SAC is the
> continuous-action realization of soft policy iteration. It
> parameterizes Q soft with a neural network, derives the policy
> via Boltzmann projection (using a reparameterized Gaussian to
> make the softmax differentiable), and uses the auto-tuned α
> trick to learn the temperature from a target-entropy constraint.
> Three pieces of this lesson reappear in L13: the soft Bellman
> operator from §3 (now applied to a neural Q-function), the
> Boltzmann policy from §4 (now with a reparameterized Gaussian
> policy), and the failure mode from §5 (avoided by operating in
> infinite-horizon continuous control). The auto-tuning of α is
> SAC's most pragmatic answer to the temperature-selection
> problem we raise in §6.

> **Forward link to L17 — RLHF and DPO.** In RLHF, the entropy
> term is replaced by a KL penalty to a reference policy:

> $$
> J_\text{RLHF}(\pi) \;=\; \mathbb{E}^\pi\!\bigl[\sum_t r(s_t, a_t)\bigr]
> \;-\; \beta\, D_\text{KL}\bigl(\pi(\cdot|s_t) \| \pi_\text{ref}(\cdot|s_t)\bigr).
> $$

> The structure is exactly parallel to §2 with `α H(π) → −β KL(π
> || π_ref)`. The two are related: KL to uniform is entropy plus a
> constant. KL to a non-uniform reference is what distinguishes
> RLHF. This is the response to the failure mode (§5): the agent
> is rewarded for being close to a goal-directed reference, not for
> being uniform, so the goal-avoidance pathology disappears. The
> optimal policy under the RLHF objective has the same Boltzmann
> form, but with `π_ref` multiplying the exponential:

> $$
> \pi^*(a|s) \;\propto\; \pi_\text{ref}(a|s)\, \exp(Q(s,a) / \beta).
> $$

> This is the closed-form DPO update target. L17 will derive it.

A brief note on the trimmed branches. Model-based RL (deferred
L14) has a clean max-ent variant: planning under an
entropy-regularized objective is itself an inference problem (a
direct corollary of §7). Offline RL (deferred L15) uses max-ent
regularization heavily — CQL's "conservative Q-learning" can be
viewed as a KL-constrained Boltzmann policy, exactly the form
above. Diffusion in RL (deferred L16) takes the
RL-as-inference view to its limit, treating action generation as
sampling from a learned posterior. The pointers in §10 give
references.

Visualization V7 — Roadmap Mini.
The curriculum graph with L12 highlighted. Outgoing edges to L13
(violet, strong) and L17 (violet, strong). The deferred lessons
(L14, L15, L16) shown in gray with dashed edges marked "future
work." Hovering shows a one-sentence pointer. 460 × 280 px. Polish
budget: 0.5 days.

---

## 5. Algorithm and Math Implementation

The new module is `src/maxent/`. It contains five exports.

```typescript
// src/maxent/types.ts
export interface SoftValueIterationResult {
  readonly Q: Float64Array;     // 9 * 4 = 36
  readonly V: Float64Array;     // 9
  readonly pi: Float64Array;    // 9 * 4 (Boltzmann)
  readonly iterations: number;
  readonly converged: boolean;
}

export interface SoftEvaluationResult {
  readonly V_soft: Float64Array;   // J_alpha(pi) per state, via soft Bellman
  readonly V_pi: Float64Array;     // E^pi[sum gamma^t r_t] per state, without entropy
}

export interface MonteCarloDiagnostic {
  readonly goalReachProb: number;
  readonly pitReachProb: number;
  readonly timeoutProb: number;
  readonly meanStepsToTerminal: number;
  readonly lengthHistogram: Float64Array;  // bucketed
}
```

```typescript
// src/maxent/softVI.ts
export function softValueIteration(
  alpha: number,
  mdp: MDP,
  options?: { tol?: number; maxIter?: number },
): SoftValueIterationResult;

// src/maxent/softPI.ts
export function softPolicyIteration(
  alpha: number,
  mdp: MDP,
  options?: { tol?: number; maxIter?: number },
): SoftValueIterationResult;

// src/maxent/softEval.ts
// Two-track evaluation: soft objective AND true E^pi[return].
// V_soft includes the alpha*H term; V_pi does not.
export function softEvaluate(
  pi: Float64Array,
  alpha: number,
  mdp: MDP,
): SoftEvaluationResult;

// src/maxent/diagnostic.ts
// Monte Carlo: does the policy actually reach the goal?
export function monteCarloDiagnostic(
  pi: Float64Array,
  mdp: MDP,
  options: { nRollouts: number; maxSteps: number; seed: number },
): MonteCarloDiagnostic;

// src/maxent/logsumexp.ts (utility, may be re-exported from src/pg/)
export function logSumExp(values: Float64Array, alpha: number): number;
```

**Vitest test targets** (in `src/maxent/*.test.ts`):

| Test | Target |
|:------|:--------|
| `softVI alpha=0.0001 ≈ hard VI` | Final V_soft(start) within 0.001 of V*=0.7290. |
| `softVI alpha=0.02 hits L10 cap` | softEvaluate(pi_alpha, alpha=0.02).V_pi[start] ≈ 0.7217 within 0.001. |
| `softVI alpha=0.2 failure` | monteCarloDiagnostic(pi_0.2, 5000 rollouts, max 500 steps): goalReachProb ≤ 0.10 (anchor: 0.051). |
| `softVI alpha=0.5 deep failure` | goalReachProb ≤ 0.01. |
| `softPI == softVI fixed point` | At alpha=0.05, 0.1, 0.5: pi from softPI and softVI agree to 1e-8 per element. |
| `Boltzmann policy normalizes` | All rows sum to 1.0 ± 1e-10. |
| `Soft Bellman is monotone in alpha` | V_soft(start) is monotonically non-decreasing as alpha increases (verified at 10 alpha values). |
| `logSumExp matches Python ref` | At alpha=0.05, Q from softVI, logSumExp matches the NumPy reference to 1e-10. |

---

## 6. Component Catalog

| Code | Component | Section | Polish (days) |
|:------|:-----------|:---------|:---------------|
| V1 | HardSoftSpectrum | §1 | 1.5 |
| V2 | ObjectiveSurface | §2 | 1.5 |
| V3 | SoftVIConvergence | §3 | 1.5 |
| V4 | BoltzmannPolicyGrid | §4 | 2.0 |
| V5 | FailureDiagnostic | §5 | 2.0 |
| V6 | EntropySliderLab | §6 | **4.0** |
| V7 | InferenceGraph | §7 | 1.5 |
| V8 | RoadmapMini-L12 | §8 | 0.5 |
| **Total** | | | **14.5** |

Reuse from prior lessons. `GridworldRenderer` (L3 → L5 → L8 → L11)
is reused in V4 and Panel A of V6. The `MathBlock` and
`CrosslinkCallout` components from L5 onward are used throughout.
`RoadmapMini` is extended (not modified) to highlight L12. No
existing component is modified.

---

## 7. Page-Level User Experience

The centerpiece (V6, EntropySliderLab) is a 960px breakout. It
loads ~60 pre-computed soft VI results from
`public/data/maxent/alpha_sweep.json` (~80KB) at page load. The α
slider snaps to the nearest pre-computed point; all panels update
synchronously when the slider moves.

The Monte Carlo data (used by Panel D of V6 and by V5) is
pre-computed at 5,000 rollouts per α value and shipped as
`maxent_rollouts.json` (~120KB). Re-running in-browser would take
several seconds; pre-computation keeps the interaction snappy.

V2 (Objective Surface) is fully analytic — no pre-computed data
needed. The user can sweep `r_1 − r_2` and α continuously.

The convergence visualization (V3) uses streamed iteration data
(~30 iterations per α, ~5 α values) at ~40KB.

The "auto-caption" panel in V6 (Panel F) is implemented as a
keyword-indexed lookup, not as an LLM call or anything fancy. The
four regime descriptions are static strings keyed to α buckets.

Accessibility: the α slider exposes keyboard control (left/right
arrows step to the next pre-computed value). Each panel has a CSV
download affordance for the data it shows. Color choices respect
the curriculum's accessibility conventions.

---

## 8. Acceptance Criteria

A learner who completes this lesson should be able to: (1) state
the entropy-regularized objective and interpret α as a temperature;
(2) write down the soft Bellman operator and explain why it is
still a γ-contraction; (3) derive the Boltzmann form of the
soft-optimal policy via the simplex-constrained maximization; (4)
recognize the L10 softmax cap as a soft-optimal policy at small α
(specifically α ≈ 0.02 on the gridworld); (5) explain the
goal-avoidance failure mode on episodic tasks and connect it to
the two mitigations (infinite-horizon settings in SAC, KL-to-
reference in RLHF); (6) sketch the RL-as-inference view and name
its core construction (the optimality variable).

**Concrete acceptance check.** Before running V6, the learner
should predict: as α increases from 0.001 to 0.5, V_soft(start)
will _____ (grow without bound), V^π(start) will _____ (collapse
toward zero), and the probability of reaching the goal will _____
(drop from 1.0 to near zero around α ≈ 0.15). The learner then
verifies by sliding through.

---

## 9. Stretch Goals (post-MVP)

(a) Auto-tuned α. The SAC v2 trick learns α by enforcing a
target-entropy constraint. Implementing this on the gridworld
would show the auto-tuning settling to a value just below the
failure threshold. Worth ~1 day; would fit in §6 as an "autopilot"
toggle on the slider.

(b) A KL-to-reference variant. Replace `α H(π)` with `−β KL(π ||
π_ref)` for a user-chosen reference (uniform, near-greedy, or a
trained policy from prior lessons). This is the L17 preview;
implementing it here would make the L12 → L17 bridge concrete.
Worth ~1.5 days; would belong as V5b or as a toggle on V6.

(c) A second MDP — a "river" or "cliff" — where the failure mode
manifests differently. The gridworld's failure mode is
"goal-avoidance via wall-bumping"; on a cliff problem it would
manifest as "prefer the long safe path even when the short risky
path is faster." Pedagogically distinct; ~2 days.

(d) Animate the inference view. V6's posterior probabilities can
be animated as backward-message updates from the goal. ~1.5 days
of D3 work.

(e) An α schedule. Many practical max-ent algorithms anneal α
during training (high at start for exploration, low at end for
sharp performance). Adding an α-schedule mode to V6 would
illustrate this. ~1 day.

---

## 10. Out of Scope (intentionally)

(a) Continuous action spaces. Treated mathematically in §3 (the
Boltzmann form generalizes to densities), but no continuous
implementation in this lesson. L13 (SAC) covers it.

(b) Off-policy max-ent algorithms (SAC, Soft Q-Learning). These
are L13 territory. We derive the value targets but do not run
off-policy max-ent algorithms here.

(c) The full variational derivation of SAC's policy update. We
sketch it in §7; the full derivation lives in L13.

(d) KL-to-reference objectives in detail. Mentioned in §5 and §8;
full treatment in L17.

(e) Maximum-entropy inverse RL (Ziebart 2008). Beautiful, but it's
about inferring rewards from demonstrations, which is a different
problem.

(f) Model-based max-ent RL (would have been L14): deferred.

(g) Offline max-ent / CQL (would have been L15): deferred. Brief
pointer in §8.

(h) Diffusion as a max-ent posterior (would have been L16):
deferred.

---

## 11. Training Notebook

`scripts/maxent_traces.py` is the offline pre-computation script.
It emits three JSON files to `public/data/maxent/`:

- `alpha_sweep.json`: soft VI results at ~60 α values from 0.001 to
  1.0 (log-spaced). For each: Q_soft, V_soft, pi, V_pi (true return
  under the soft policy), mean entropy, KL to uniform, KL to
  greedy. Loaded by V1, V4, V5, V6.
- `convergence_traces.json`: per-iteration V_soft and pi for ~5
  selected α values (0.01, 0.05, 0.1, 0.5, 1.0). Loaded by V3.
- `maxent_rollouts.json`: Monte Carlo diagnostic data (goal-reach
  prob, pit-reach prob, timeout prob, length histogram) at each α.
  Loaded by V5 and Panel D/E of V6.

Pure NumPy, no PyTorch. Total runtime: ~2 minutes (the bottleneck
is the 60 × 5000-rollout MC sweep, which is embarrassingly
parallel and can be sped up with `multiprocessing` if needed).

A `verify_anchors()` function re-runs the headline numbers and
prints them. The TypeScript implementation must match these to
within 1e-6 at single-seed level.

---

## 12. Closing Notes and Length Tally

Approximate length: ~1400 lines of README. Polish budget 14.5
days, dominated by the EntropySliderLab centerpiece (4 days). No
new dependencies; the lesson runs on the existing stack.

Centerpiece role: the EntropySliderLab is the lesson's
distillation. The slider is the simplest possible UI affordance,
and the six panels make the consequences of moving it visible
across multiple dimensions (value, policy, entropy, goal-reaching,
trajectory length, regime). The "auto-caption" panel is the
explicit pedagogical voice — at each α regime, the lesson tells
the user what's happening, in plain words. This is the snap-into-
place moment.

Forward links carry unusual weight in this lesson. With L14, L15,
L16 deferred, L12 is the last theoretical foundation before L13
(SAC) and the conceptual bridge to L17 (RLHF). The §8 forward
links are correspondingly long and substantive: SAC inherits §3,
§4, and the §5 failure-mode mitigation; RLHF inherits §2's
objective structure with the entropy term swapped for KL.

The curriculum spine after this lesson: L1 → L10 → L11 → **L12** →
L13 → L17. Two lessons left in the production schedule when L12
ships.

## End of specification