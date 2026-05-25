# Lesson 2 — Markov Decision Processes

> **The formal model of decision-making under uncertainty.** Bandits gave us
> exploration vs exploitation in one state. Markov chains gave us the
> long-run behaviour of uncontrolled dynamics. MDPs put them together: a
> chain you partially control, with rewards along the way. From this single
> object — the MDP — we'll derive value functions, the Bellman equations,
> and the notion of an optimal policy. Every subsequent lesson is a way of
> solving or generalizing this object.

---

## 0. Pedagogical Philosophy

1. **Build the object, then the equations.** First two sections introduce
   the MDP and policies — just structure, no calculus. The Bellman
   equations don't appear until §6, by which point the learner has already
   reasoned about returns, values, and Q-values informally. The equations
   then *click into* a place that's already prepared for them.

2. **One running example, computed exhaustively.** A 3×3 gridworld with
   one pit, one goal, deterministic transitions, γ = 0.9. Small enough that
   every numerical claim is hand-verifiable in five minutes; rich enough
   that the optimal policy is non-trivial. All eight visualizations use
   this gridworld unless explicitly noted.

3. **The Bellman backup, not the Bellman equation, is the centerpiece.**
   The equation is static; the backup is animated. The learner watches the
   value function ripple outward from terminal states, iteration by
   iteration, until it stabilizes. This visceral experience is what makes
   the equation meaningful in Lesson 3 when we *iterate* it to convergence.

4. **Vocabulary over algorithms.** This lesson does *not* implement
   policy iteration or value iteration — those are Lesson 3. We do
   compute V^π via direct linear solve (one matrix inversion), and we do
   show iterative Bellman backups for visualization purposes, but the
   *algorithms* live next door. Resist the temptation to spill over.

5. **End every section pointing forward.** Q-values point to Q-learning
   (Lesson 5), advantage points to actor-critic (Lesson 8), stochastic
   policies point to policy gradient (Lesson 8) and max-entropy RL
   (Lesson 10). The lesson is the spine that holds the whole curriculum
   together; every forward link is a vertebra.

---

## 1. Tech Stack

Identical to Lessons 1 and 2 (Markov Chains). Vite + TypeScript strict +
KaTeX + D3 v7 + ml-matrix. Vitest. No PyTorch.

This lesson exercises ml-matrix more heavily than Markov Chains did — we'll
solve `(I - γ P^π) V = R^π` via direct linear solve for V^π, which means
matrix inversion (or LU-decomp; ml-matrix handles this).

The Python script `scripts/mdp_gridworld.py` (~50 lines) pre-computes the
running example's V*, Q*, V^π for several policies, and a Bellman-backup
trace. Output JSON goes to `public/data/mdp/`.

---

## 2. Visual / Aesthetic Direction

We extend the established curriculum aesthetic. New conventions specific
to MDPs, locked in here for re-use across the rest of the curriculum:

```css
:root {
  /* Value-function heatmap palette: diverging green ↔ red around 0 */
  --mdp-value-pos:   #15803d;   /* green-700, high value */
  --mdp-value-zero:  #f1ede4;   /* neutral, ≈ surface-2 */
  --mdp-value-neg:   #b91c1c;   /* red-700, negative value */

  /* Reward semantics */
  --mdp-reward-pos:  #15803d;   /* goal / positive reward */
  --mdp-reward-neg:  #b91c1c;   /* pit / negative reward */
  --mdp-terminal:    #1c1e22;   /* terminal-state border */

  /* Policy arrows */
  --mdp-action-up:    #2563eb;
  --mdp-action-right: #ea580c;
  --mdp-action-down:  #16a34a;
  --mdp-action-left:  #9333ea;
  /* (Each action's color is reused across V5, V6, V7 for consistency.) */

  /* Bellman backup highlight */
  --mdp-backup-source:  #d97706;   /* state being backed up */
  --mdp-backup-input:   #0891b2;   /* states feeding into the backup */
}
```

**Gridworld rendering convention.** Cells drawn as rounded squares (radius
4px). Terminal cells have a dark border. The pit shows a small skull glyph
or `−1` label; the goal shows a flag or `+1`. Value heatmap fills the cell
interior; policy arrows overlay it. Hovering a cell pops a tooltip with
`V(s)`, `Q(s,a)` for all actions, and the policy `π(·|s)`.

**Action arrows.** Four arrows per cell, pointing in four directions, with
opacity proportional to `π(a|s)`. Deterministic policies show one solid
arrow; uniform random shows four equally-faded arrows.

---

## 3. Lesson Metadata

```ts
export const lessonMeta: LessonMeta = {
  id: "mdps",
  title: "Markov Decision Processes",
  subtitle: "The formal model of sequential decision-making",
  tier: 1,
  difficulty: 3,
  estimatedReadMinutes: 70,
  role: "critical-path",
  prerequisites: [
    { lesson: "bandits",        anchor: "regret-definition" },
    { lesson: "markov-chains",  anchor: "policy-induced-chain" },
  ],
  exportedAnchors: [
    "mdp-tuple",
    "policy",
    "return-discount",
    "state-value-function",
    "action-value-function",
    "advantage-function",
    "bellman-expectation",
    "bellman-optimality",
    "optimal-policy-exists-deterministic",
    "matrix-form-policy-evaluation",
  ],
  centerpieceComponent: "BellmanBackupLab",
  forwardLinksWhenReady: [
    { to: "dynamic-programming", anchor: "policy-iteration" },
    { to: "td-learning",         anchor: "sample-based-bellman" },
    { to: "policy-gradient",     anchor: "advantage-actor-critic" },
    { to: "max-ent-rl",          anchor: "soft-bellman" },
  ],
};
```

---

## 4. Section-by-Section Plan

### §1 — From Bandits + Chains to MDPs
**Tagline:** *Add a control knob to a Markov chain. Watch it become decision-making.*
**Length:** ~600 words.
**Anchor:** `mdp-tuple`.

---

**Prose:**

Two lessons ago, the agent chose between K arms in a single state — Bandits.
The world didn't move; only the agent's beliefs did. One lesson ago, a
system evolved on its own via a transition matrix `P` — Markov Chains. The
agent didn't exist; the world moved without input.

The Markov Decision Process is the synthesis. The agent observes a state `s`,
chooses an action `a`, and the world responds: it transitions to a new state
`s'` and delivers a scalar reward `r`. The transition distribution depends
on both the current state *and* the chosen action.

> **Back-link** — Bandits is the special case `|\mathcal{S}| = 1`. There is no
> "next state" because there's only one. The Bellman equation we'll derive
> in §6 collapses to `V^\pi = \sum_a \pi(a) r(a)`: the policy's expected
> immediate reward. That's exactly the quantity ε-greedy and UCB estimate.

Formally, a **finite Markov Decision Process** is a tuple

$$
\boxed{\mathcal{M} \;=\; (\mathcal{S},\, \mathcal{A},\, P,\, r,\, \gamma)}
$$

with:

- `\mathcal{S}`: a finite **state space**, `|\mathcal{S}| = K`.
- `\mathcal{A}`: a finite **action space**, `|\mathcal{A}| = m`. (We assume
  the same action set is available in every state, but everything generalizes
  to state-dependent action sets `\mathcal{A}(s)` if needed.)
- `P`: a **transition kernel**, `P(s' \mid s, a) = \Pr(S_{t+1} = s' \mid S_t = s, A_t = a)`.
  For each `(s, a)`, the row `P(\cdot \mid s, a)` is a probability distribution
  over `\mathcal{S}`.
- `r`: a **reward function**, `r(s, a) = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a]`.
  We take the expected immediate reward as a function of the *current* state
  and action.
- `γ ∈ [0, 1)`: the **discount factor**. Determines how much the agent values
  future reward vs immediate reward. We'll explain why `γ < 1` is the default
  in §3.

**The Markov property carries over from Lesson Prereq A**: the transition and
reward depend only on the current `(s, a)`, not on the full history. This is
what makes the whole apparatus tractable.

---

**Running example: the 3×3 gridworld.**

We'll use this single example throughout the lesson. Memorize it.

```
   col 0    col 1    col 2
 ┌────────┬────────┬────────┐
 │ START  │        │        │   row 0
 ├────────┼────────┼────────┤
 │        │  PIT   │        │   row 1
 │        │   -1   │        │
 ├────────┼────────┼────────┤
 │        │        │ GOAL   │   row 2
 │        │        │  +1    │
 └────────┴────────┴────────┘
```

- `\mathcal{S}` = the nine grid cells, indexed `(r, c)` with `r, c ∈ {0, 1, 2}`.
- `\mathcal{A} = \{\text{Up}, \text{Right}, \text{Down}, \text{Left}\}`.
- Transitions are **deterministic** in the basic version: the agent moves
  exactly one cell in the chosen direction. Walls bounce: the agent stays
  in place. (A stochastic variant appears in §6's centerpiece.)
- Rewards: `+1` upon entering the goal, `−1` upon entering the pit, `0`
  otherwise. The reward is collected on the transition *into* a state, not
  for being in it.
- Pit and goal are **terminal**: once entered, the episode ends.
- Discount `γ = 0.9`.

This is small enough (9 states, 4 actions, 36 (s,a) pairs) that the full
transition table fits on one screen, and yet the optimal policy must
*navigate around the pit* — a genuine planning problem.

---

**Visualization V1 — MDP Anatomy Explorer.**

- The 3×3 gridworld drawn with full styling: start marker, pit (red, skull
  glyph), goal (green, flag glyph), wall borders.
- Click any non-terminal cell to "select" it. Four action buttons appear:
  Up, Right, Down, Left.
- Click an action: a transparent overlay shows the target cell highlighted,
  with the reward and probability displayed (`P(s'|s,a) = 1.0`, `r = 0`).
- A small side panel shows the MDP tuple symbolically with the selected
  `(s, a, s', r)` filled in.
- "Stochastic mode" toggle: with 80%-10%-10% slip probabilities, the action
  arrow fans out into three target cells with their probabilities.
- Width 720, height 400.

The point of V1 is *tactile mastery of the dynamics*. The learner should
leave this section knowing the gridworld cold.

---

### §2 — Policies
**Tagline:** *A policy is a rule for living.*
**Length:** ~550 words.
**Anchor:** `policy`.

---

**Prose:**

The agent doesn't get to control transitions directly; they're set by `P`.
What the agent *does* control is how it chooses actions. A **policy**
specifies this choice.

**Deterministic policy.** A map `π: \mathcal{S} \to \mathcal{A}`. From state
`s`, the agent takes action `π(s)`, always.

**Stochastic policy.** A conditional distribution `π(\cdot \mid s)` over
actions, given the state. From state `s`, the agent samples
`a \sim π(\cdot \mid s)`. Deterministic policies are a special case where
the distribution is a delta.

Two policies on our gridworld worth keeping in mind:

- **Uniform random:** `π(a \mid s) = 1/4` for all `(s, a)`. The agent
  flails. This is our worst-realistic baseline.
- **An optimal policy:** at state `(0,0)`, go right; at `(0,1)`, go right;
  at `(0,2)`, go down; etc. There are actually *multiple* optimal policies
  in this gridworld (two symmetric shortest paths around the pit). We'll
  compute one in §7.

Why ever consider *stochastic* policies if a deterministic one suffices? Three
reasons that will matter downstream.

1. **Exploration.** A deterministic policy in an unknown MDP will never
   discover state-actions it has never tried. Adding randomness — ε-greedy
   (from Bandits), Boltzmann, Gaussian noise — is the simplest way to keep
   exploring. Lessons 5 and 7 lean on this.

2. **Differentiability.** When we parameterize policies as neural networks
   `π_θ(a \mid s)` and want to compute `∇_θ J(θ)`, the gradient must flow
   through the action distribution. A deterministic argmax has zero gradient
   almost everywhere; a stochastic policy does not. Lesson 8 (policy
   gradient) cannot exist without stochastic policies.

3. **Optimality under partial observability and adversarial settings.**
   In games against an adversary or under partial observation, the
   *optimal* policy can be genuinely stochastic (think rock-paper-scissors).
   Even in fully-observable MDPs, optimal *exploration* policies are
   stochastic.

For finite MDPs with full observability — the setting of this entire
lesson — there *always exists* a deterministic optimal policy. We'll
prove this in §7. So if you only care about exploitation, deterministic
is enough. The above three reasons explain why every lesson from Lesson 7
onward will use stochastic policies anyway.

> **Back-link to Bandits.** Bandits used stochastic policies for exploration
> reason 1 (ε-greedy, Thompson). The other two reasons didn't apply because
> there was no gradient to compute and no adversary.

---

**Visualization V2 — Policy Explorer.**

- The 3×3 gridworld with policy arrows overlaid on each non-terminal cell.
- Four arrows per cell (Up/Right/Down/Left) with opacity ∝ `π(a|s)`.
- Policy mode toggle:
  - **Uniform random:** all four arrows at 25% opacity.
  - **Deterministic optimal:** one solid arrow per cell.
  - **ε-soft optimal:** with slider `ε ∈ [0, 1]`, the optimal action has
    weight `1 - ε + ε/4`, others `ε/4`. The arrows update live.
  - **Custom:** click an arrow to increment that action's weight by 0.1;
    other actions in that cell renormalize.
- Below the grid: a small panel showing `π(·|s)` for the currently-selected
  cell as a bar chart.
- "Save policy" button writes the current `π` to a small in-memory store;
  V4 can then visualize V^π for it.
- Width 720, height 440.

---

### §3 — Returns and Discounting
**Tagline:** *Adding up rewards over time, with a thumb on the scale.*
**Length:** ~700 words.
**Anchor:** `return-discount`.

---

**Prose:**

The agent collects rewards `R_1, R_2, R_3, \ldots` over time. To talk about
"performance" we need to aggregate them into a single number. The standard
choice is the **discounted return** starting from time `t`:

$$
\boxed{G_t \;:=\; R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots \;=\; \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}}
$$

with `γ ∈ [0, 1]`. (Notation: rewards are indexed with `t+1` to emphasize
they're received *after* the action at time `t`. Some authors index them with
`t`; the choice doesn't matter as long as you're consistent.)

**Why discount?** Five answers, each illuminating.

1. **Mathematical convergence.** For continuing tasks (no terminal state),
   undiscounted returns `G_t = \sum_{k=0}^{\infty} R_{t+k+1}` can be infinite,
   meaning every policy has the same expected return and no comparison is
   possible. `γ < 1` guarantees finite returns whenever rewards are bounded.

2. **Computational stability.** The Bellman operator (§6) is a contraction
   with modulus `γ`. Iterating the Bellman backup converges geometrically;
   it diverges if `γ = 1` and the chain isn't absorbing.

3. **Economic interpretation.** A reward received 10 steps from now is
   worth `γ^{10}` of a reward right now. This is a standard discount in
   economics; in RL it captures "preference for sooner".

4. **Uncertainty about the future.** If `γ = 0.99`, you behave as if the
   world has a `1\%` chance of ending each step. Discounting bakes in some
   pessimism about the horizon.

5. **Modeling continuing tasks as if episodic.** A continuing task with
   discount `γ` has the same expected return as an episodic task that ends
   randomly at each step with probability `1 - γ`. This duality is
   sometimes useful.

The case `γ = 1` ("undiscounted") only makes sense for **episodic tasks**
where every trajectory reaches a terminal state in finite time with
probability 1. Our gridworld is episodic (every reasonable policy hits the
goal or pit eventually), so we *could* use `γ = 1`. We use `γ = 0.9`
anyway because it gives us numerical room to distinguish "close to goal"
from "far from goal" — see the V^* table coming up.

---

**Numerical example (pre-verified).** Consider a trajectory in our
gridworld where the agent goes from start to goal via four steps:

`(0,0) → (1,0) → (2,0) → (2,1) → (2,2) [goal]`

Rewards collected: `R_1 = R_2 = R_3 = 0`, `R_4 = +1` (received on
entering goal). The return at time 0 is

$$
G_0 \;=\; \gamma^0 \cdot 0 + \gamma^1 \cdot 0 + \gamma^2 \cdot 0 + \gamma^3 \cdot 1 \;=\; \gamma^3.
$$

For various discount factors:

| `γ`   | `G_0 = γ³`   |
|-------|-------------:|
| 0.0   | 0.0000       |
| 0.5   | 0.1250       |
| 0.9   | 0.7290       |
| 0.95  | 0.8574       |
| 0.99  | 0.9703       |
| 1.0   | 1.0000       |

The discount factor is a knob controlling how much the agent cares about
distant rewards. With `γ = 0`, the agent is purely myopic: only the next
reward matters. With `γ = 1`, all future rewards are equally weighted. At
`γ = 0.9`, a reward 4 steps away is worth 73% of an immediate reward —
strong but not extreme preference for sooner.

> **Forward link** — In Lesson 10 (Max-Entropy RL) we'll *augment* the
> reward with the policy's entropy: `\tilde{r}(s,a) = r(s,a) + \alpha \mathcal{H}[\pi(\cdot|s)]`.
> The discounted *augmented* return becomes the standard objective of
> max-ent RL. The discounting machinery doesn't change; only the reward
> definition.

---

**Visualization V3 — Return Composer.**

- Top half: a strip showing a sampled trajectory in the gridworld
  `s_0, a_0, r_1, s_1, a_1, r_2, \ldots`, with discrete colored boxes for
  each `(s, a, r, s')` step.
- Bottom half: a stacked bar chart of the discounted return contributions
  `γ^k R_{k+1}` for `k = 0, 1, 2, \ldots`, with the running total `G_0`
  to the right.
- Slider for `γ ∈ [0, 1]`. As γ changes, the bars rescale in real time.
- "Resample trajectory" button samples a new trajectory under a chosen
  policy (uniform random or current saved policy from V2).
- A second bar (faded) shows `r_k` (un-discounted) underneath for contrast.
- Width 800, height 380.

---

### §4 — State Value Function V^π
**Tagline:** *Expected return, given that you start here and follow π.*
**Length:** ~700 words.
**Anchor:** `state-value-function`.

---

**Prose:**

The return `G_t` is a *random variable*: it depends on the stochasticity of
both the policy and the environment. To compare policies, we need to take
expectations.

The **state value function** of policy `π` is

$$
\boxed{V^\pi(s) \;:=\; \mathbb{E}_\pi\!\left[G_t \mid S_t = s\right] \;=\; \mathbb{E}_\pi\!\left[\sum_{k=0}^\infty \gamma^k R_{t+k+1} \,\middle|\, S_t = s\right]}
$$

The subscript `\mathbb{E}_\pi[\cdot]` is shorthand for "expectation under
the joint distribution induced by `π` and the MDP dynamics" — that is, the
trajectory `(s_t, a_t, s_{t+1}, a_{t+1}, \ldots)` is drawn by following `π`
and `P` for `k = t, t+1, \ldots`.

A few important properties:

- `V^\pi(s)` is a function of `s`, not of `t`. The MDP is stationary, so
  "expected return starting from state `s`" doesn't depend on what time it
  is. (This is the *time-homogeneity* of the MDP at work.)
- `V^\pi(s)` is well-defined and finite whenever `γ ∈ [0,1)` and rewards
  are bounded: `|V^\pi(s)| \leq R_{\max} / (1 - \gamma)`.
- For terminal states `s`, `V^\pi(s) = 0` by convention — no future rewards
  to collect.

**The point of V^π.** It's how we compare *policies, not actions*. If
`V^{π_1}(s) > V^{π_2}(s)` for all `s` (with strict inequality somewhere),
then `π_1` dominates `π_2`. The **partial order** on policies induced by
component-wise comparison of `V^π` is what makes "optimal" meaningful: an
optimal policy is one that's not dominated by any other.

---

**Worked example (pre-verified): V^π for the uniform-random policy on our
gridworld.** With `π(a|s) = 1/4` for all `(s,a)` and `γ = 0.9`:

| state | (col 0) | (col 1) | (col 2) |
|------:|--------:|--------:|--------:|
| **row 0** | −0.421 | −0.514 | −0.239 |
| **row 1** | −0.514 |  0.000 | −0.069 |
| **row 2** | −0.239 | −0.069 |  0.000 |

The pit and goal are terminal (`V = 0`). Every other cell is **negative**,
which deserves a moment of thought. The reason: a random walker frequently
falls into the pit (1/4 chance from each of its four neighbors), incurring
the `−1` reward; the goal is also reachable but the pit's pull dominates
on this geometry. State `(0,1)` is the worst: it's *adjacent to the pit*
(going down lands in it with 25% probability) and the goal is far.

**Sanity-check.** State `(1, 2)` has `V^π ≈ −0.069`. Let's see why:
- Going up takes us to `(0,2)`, V ≈ −0.239 → contribution `0.25·(0 + 0.9·(-0.239)) = -0.054`
- Going right hits the wall, stay at `(1,2)`, → `0.25·(0 + 0.9·(-0.069)) = -0.0155`
- Going down to `(2,2)` is the **goal**, → `0.25·(+1 + 0.9·0) = +0.250`
- Going left to `(1,1)` is the **pit**, → `0.25·(-1 + 0.9·0) = -0.250`

Sum: `-0.054 + -0.0155 + 0.250 + -0.250 = -0.0695 ≈ -0.069` ✓.

The goal-bonus and the pit-penalty exactly cancel because they're
symmetric in this cell, and the residual negative comes from "expected
future of the up-action," which leads to less-favourable territory.

> **Forward link** — Computing `V^π` is called **policy evaluation**. In
> Lesson 3 we'll meet *iterative* policy evaluation (the linear solve via
> repeated Bellman backups). In Lesson 5, TD(0) does the same job from
> *samples*, without ever forming the matrix `P^π`. Both are answering the
> same question: what does `V^π` equal?

---

**Visualization V4 — V^π Heatmap.**

- The 3×3 gridworld with each cell shaded by `V^π(s)`, using the diverging
  green↔red palette. Numerical `V` value printed inside each cell in mono.
- Policy selector dropdown: "uniform random", "deterministic optimal",
  "go-down-then-right" (a hand-coded policy that achieves the goal but
  walks a long way), "custom (from V2)".
- Side panel: a bar chart of `V^π` across all 9 cells, sorted.
- "Compute" button runs the linear solve `(I - γP^π)V = R^π` and shows
  the computed values inline.
- Below: small text showing `||V^π||_∞`, max, min, mean.
- Width 800, height 460.

---

### §5 — Action Value Q^π and Advantage A^π
**Tagline:** *V, conditioned on the first move. And how much better that move is than average.*
**Length:** ~700 words.
**Anchor:** `action-value-function`.

---

**Prose:**

Sometimes we want to know not "what's the expected return starting from
state `s` under policy `π`," but "what's the expected return if I take
action `a` *now*, and follow `π` thereafter." That's the **action value
function**:

$$
\boxed{Q^\pi(s, a) \;:=\; \mathbb{E}_\pi\!\left[G_t \mid S_t = s,\, A_t = a\right]}
$$

The relationship to `V^π` is immediate. The state value is the action-value
*averaged over the policy's action choice*:

$$
V^\pi(s) \;=\; \sum_a \pi(a \mid s) \, Q^\pi(s, a) \;=\; \mathbb{E}_{a \sim \pi(\cdot|s)}\!\left[Q^\pi(s, a)\right].
$$

Conversely, `Q^π` decomposes into immediate reward plus discounted next-state
value:

$$
\boxed{Q^\pi(s, a) \;=\; r(s, a) + \gamma \sum_{s'} P(s' \mid s, a)\, V^\pi(s')}
$$

These two equations are baby versions of the Bellman expectation equation
(§6). Notice that the second one expresses `Q^π` purely in terms of `V^π` —
the action-value and state-value are not independent quantities.

**Why have two value functions?** Three reasons:

1. **Action selection.** To act greedily — `\pi'(s) = \arg\max_a Q^\pi(s, a)` —
   you need `Q^π`, not `V^π`. Knowing `V^π` and the model `P, r` is
   equivalent, but if you don't know `P, r`, `Q^π` is the actionable thing.

2. **Q-learning.** When we learn from samples (Lesson 5), we typically
   learn `Q` directly because samples give us `(s, a, r, s')` tuples — and
   `Q(s, a)` is what those tuples are about.

3. **Function approximation.** When `Q` is parameterized by a neural network,
   `Q_θ(s, a)` is computed in one forward pass; you do *not* need to know
   `P` or sum over `s'`. This is what DQN (Lesson 7) exploits.

---

**The advantage function.**

$$
\boxed{A^\pi(s, a) \;:=\; Q^\pi(s, a) - V^\pi(s)}
$$

The advantage measures *how much better than average* action `a` is at state
`s`, where "average" means "the policy's expected action." Some
observations:

- `\sum_a \pi(a|s) A^\pi(s, a) = 0`: the policy-weighted advantage is zero
  *by construction*. (Subtract `V^π` from both sides of the
  `V^π = E_{a∼π}[Q^π]` identity.)
- For deterministic policies, `A^π(s, \pi(s)) = 0` and the advantage of
  *other* actions is negative (since `V^π` equals the value of the
  chosen action). For stochastic policies, the picture is richer.
- The advantage is **invariant to a state-dependent baseline** — adding
  `b(s)` to `Q^π(s, a)` for all `a` leaves `A^π` unchanged. This is
  exactly why advantage estimators are used in policy gradient (Lesson 8)
  to *reduce variance* without biasing the gradient.

---

**Worked example (pre-verified): Q* and A* at state (1, 0) on the
gridworld, under the **optimal** policy `π*`.**

State `(1, 0)` is the leftmost cell in the middle row. `V^*((1,0)) = 0.81`.
The Q-values for each action:

| action | landing in | reward | `Q* = r + γ V*(s')` |
|:------:|:----------:|:------:|--------------------:|
| Up    | (0, 0)     | 0      | `0 + 0.9 · 0.729 = 0.6561`     |
| Right | (1, 1) = pit | −1   | `−1 + 0.9 · 0 = −1.0000`       |
| Down  | (2, 0)     | 0      | `0 + 0.9 · 0.9 = 0.8100` ← argmax |
| Left  | (1, 0) bounce | 0   | `0 + 0.9 · 0.81 = 0.7290`      |

So `V^*((1,0)) = max = 0.81` (consistent with what we got in §4). The
advantages:

| action | `A*((1,0), a) = Q* − V*` |
|:------:|------------------------:|
| Up    | `−0.1539`               |
| Right | `−1.8100`  ← *very* bad |
| Down  | `0.0000`   ← optimal    |
| Left  | `−0.0810`               |

The advantage `A* = −1.81` for "Right" is a beautifully horrible number:
it's saying *that single action costs you nearly two future-reward units
of total expected return.* (One unit from the immediate `−1` reward, plus
`0.81` lost from giving up the (1,0) → (2,0) → (2,1) → (2,2) trajectory.)
The advantage exposes *how much each action choice matters*, which is
exactly what an actor-critic algorithm needs to direct its policy gradient.

> **Forward link** — In Lesson 8 (Policy Gradient), the policy-gradient
> theorem will be expressed as
> `\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^\pi,\, a \sim \pi}[\nabla_\theta \log \pi_\theta(a|s) \cdot A^\pi(s,a)]`.
> The advantage is *exactly* what gets multiplied by the score function.
> Replacing `A^π` with `Q^π` is unbiased but higher-variance; using `r`
> alone (REINFORCE) is even higher-variance. The advantage is the
> variance-optimal baseline.

---

**Visualization V5 — Q-Quadrants and Advantage.**

- Each cell of the 3×3 gridworld is divided into four triangular quadrants
  (top/right/bottom/left), one per action.
- Each quadrant shaded by `Q^π(s, a)` (same diverging palette as V4).
- Numerical `Q` value printed in each quadrant in mono.
- Toggle: "Show Q^π" vs "Show A^π = Q^π − V^π". When showing A^π, the
  optimal action's quadrant is white (since A* = 0 there) and others are
  red-shaded by how much worse they are.
- Policy selector (same as V4).
- Hover any quadrant to see expanded info: action, `Q(s,a)`, `A(s,a)`,
  next-state distribution.
- Width 720, height 460.

This is the *single most informative* visualization for understanding the
relationship between V, Q, π. The four-quadrants-per-cell layout makes the
optimal action obvious at a glance (greenest quadrant) and reveals the
"avoid the pit" structure of the gridworld via deep red on the cells
adjacent to it.

---

### §6 — The Bellman Expectation Equations (CENTERPIECE)
**Tagline:** *V^π satisfies a recursive identity. Iterate the identity, get the value.*
**Length:** ~1100 words.
**Anchor:** `bellman-expectation`.

---

**Prose:**

We've defined `V^π` as an expectation over infinite trajectories. That's
fine for theory but useless for computation — you can't enumerate infinite
trajectories. The **Bellman expectation equation** rewrites `V^π` as a
*recursion*, breaking the infinite sum into "one step + discounted rest."

**Derivation.** Start with the definition:

$$
V^\pi(s) \;=\; \mathbb{E}_\pi\!\left[\sum_{k=0}^\infty \gamma^k R_{t+k+1} \,\Big|\, S_t = s\right].
$$

Peel off the first reward:

$$
\begin{aligned}
V^\pi(s) &= \mathbb{E}_\pi\!\left[R_{t+1} + \gamma \sum_{k=0}^\infty \gamma^k R_{t+k+2} \,\Big|\, S_t = s\right] \\[2pt]
&= \mathbb{E}_\pi\!\left[R_{t+1} + \gamma\, G_{t+1} \,\Big|\, S_t = s\right] \\[2pt]
&= \mathbb{E}_\pi\!\left[R_{t+1} \mid S_t = s\right] + \gamma\, \mathbb{E}_\pi\!\left[G_{t+1} \mid S_t = s\right].
\end{aligned}
$$

The first term is easy: `\mathbb{E}_\pi[R_{t+1} | S_t = s] = \sum_a \pi(a|s) r(s,a)`.
The second term needs the **tower property** (a.k.a. law of total
expectation) — conditioning the inner expectation on `(A_t, S_{t+1})`:

$$
\mathbb{E}_\pi[G_{t+1} \mid S_t = s] \;=\; \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \, \mathbb{E}_\pi[G_{t+1} | S_{t+1} = s'] \;=\; \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \, V^\pi(s').
$$

(The last equality used the Markov property: `G_{t+1}` conditioned on
`S_{t+1}` doesn't depend on the prior state.) Combining everything:

$$
\boxed{V^\pi(s) \;=\; \sum_a \pi(a \mid s) \!\left[\, r(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \, V^\pi(s') \,\right]}
$$

This is the **Bellman expectation equation** for `V^π`. The corresponding
equation for `Q^π` is

$$
\boxed{Q^\pi(s, a) \;=\; r(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \sum_{a'} \pi(a' \mid s')\, Q^\pi(s', a')}
$$

— derivable by analogous calculation.

---

**Matrix form.** Define the policy-induced transition matrix
`P^\pi \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{S}|}` with entries

$$
(P^\pi)_{s, s'} \;:=\; \sum_a \pi(a \mid s)\, P(s' \mid s, a)
$$

and the policy-induced reward vector
`R^\pi \in \mathbb{R}^{|\mathcal{S}|}` with entries

$$
(R^\pi)_s \;:=\; \sum_a \pi(a \mid s)\, r(s, a).
$$

The Bellman equation in matrix form is then

$$
\boxed{V^\pi \;=\; R^\pi + \gamma P^\pi V^\pi \;\;\Longleftrightarrow\;\; (I - \gamma P^\pi) V^\pi = R^\pi.}
$$

> **Back-link to Markov Chains** — `P^π` is exactly the policy-induced
> chain from Lesson Prereq A §7. Its stationary distribution `d^π`
> describes the on-policy state visitation. *Same matrix, two uses.*

The matrix `(I - γP^π)` is invertible for any `γ ∈ [0, 1)`. (Reason: the
spectral radius of `γP^π` is `γ \rho(P^π) = γ \cdot 1 = γ < 1`, so the
Neumann series `\sum_n (γP^π)^n = (I - γP^π)^{-1}` converges.) Hence

$$
V^\pi \;=\; (I - \gamma P^\pi)^{-1} R^\pi.
$$

This is **policy evaluation by direct solve**. For our 9-state gridworld
this is a 9×9 linear system; instant. For an MDP with `10^6` states, this
matrix is too big to invert directly, and we'd instead use iterative
methods (Lesson 3) or sample-based methods (Lesson 5).

---

**Iterative interpretation.** Let `T^π` denote the **Bellman operator**
`T^π V := R^π + γ P^π V`. Starting from any `V_0` (often `V_0 = 0`),
define `V_{k+1} = T^π V_k`. Then `V_k \to V^π` as `k \to \infty`, and the
convergence is geometric at rate `γ`:

$$
\|V_{k+1} - V^\pi\|_\infty \;\leq\; \gamma \|V_k - V^\pi\|_\infty.
$$

`T^π` is a **γ-contraction in the supremum norm**. (Proof: a single
application of `T^π` adds the same `R^π`, then multiplies by `γ P^π`;
`P^π` is row-stochastic so its `\|\cdot\|_\infty`-induced operator norm is
1, and multiplying by `γ` gives the contraction factor `γ`.)

The Banach fixed-point theorem says: a contraction on a complete metric
space has a unique fixed point, and iteration converges to it from any
starting value. Hence `V^π` is the unique fixed point of `T^π`, and the
iterative procedure converges. This is the same fixed-point machinery that
will power value iteration in Lesson 3, with `T^π` replaced by the
Bellman *optimality* operator `T^*` (next section).

---

**Numerical demonstration (pre-verified).** Iterating `T^π` for the
uniform-random policy on our gridworld, starting from `V_0 = 0`:

| iteration `k` | `V_k(0,0)` |
|--------------:|-----------:|
| 0             | 0.0000     |
| 5             | −0.288     |
| 10            | −0.386     |
| 20            | −0.418     |
| ∞             | −0.4205    |

It takes about 100 iterations to reach machine precision. The geometric
rate `γ = 0.9` is *visible*: errors shrink by ~10× every ~22 iterations.

---

**Visualization V6 — Bellman Backup Lab. THIS IS THE CENTERPIECE.**

A four-panel synchronized dashboard. Budget: 3-4 days of polish.

**Panel A (top-left): Backup at a single state, expanded.**
- User selects a state `s` by clicking a cell in a small 3×3 gridworld.
- The selected cell highlights in `--mdp-backup-source`.
- A schematic appears below showing the backup:
  - For each action `a`: a row showing `π(a|s) × [r(s,a) + γ V(s')]`.
  - The contributions sum, vertically, to `V_{new}(s)`.
  - Successor cells `s'` are highlighted in `--mdp-backup-input` in the
    mini-gridworld at the top.
- Numerical values for each term are shown in mono.

**Panel B (top-right): Full grid view.**
- The full 3×3 gridworld colored by `V_k`.
- A frame counter `k = N` and "step" / "play" / "reset" controls.
- "Mode" toggle: **expectation** (`T^π V`) vs **optimality** (`T^* V`, max
  instead of sum). When optimality is selected, V converges to V*.

**Panel C (bottom-left): Convergence trace.**
- Line plot of `V_k(s)` over `k` for a few selected states. Asymptotes
  visibly approach `V^π(s)`.
- Log-scale plot of `||V_k - V^π||_∞` showing geometric convergence with
  rate `γ`.

**Panel D (bottom-right): Policy panel.**
- The `π` the agent is using (uniform random by default, switchable).
- For optimality mode: shows the *greedy* policy `π_{V_k}(s) = \arg\max_a [r(s,a) + γ Σ P(s'|s,a) V_k(s')]`
  emerging from the value function as iterations proceed.

**Controls.**
- Policy dropdown (uniform / optimal / saved-from-V2).
- Mode toggle (expectation / optimality).
- γ slider [0.5, 0.99].
- Stochasticity toggle: deterministic transitions vs slippery (80%-10%-10%).
- Speed control 0.5×–16×.

The Bellman Backup Lab is the single most important visualization in the
entire RL curriculum so far. It makes one equation — the Bellman equation —
into an animated story about *how value propagates from terminal states
outward through the state space*. Once a learner has seen this, they can
read the algorithms of Lesson 3 (policy iteration, value iteration) as
"do this, but smarter." Without it, those algorithms are just
pseudocode.

**Width:** ~960px (centerpiece breakout). **Height:** ~640px.

---

### §7 — Bellman Optimality
**Tagline:** *Replace the sum-over-π with a max. Get V*. Greedy w.r.t. V* is optimal.*
**Length:** ~850 words.
**Anchor:** `bellman-optimality`.

---

**Prose:**

`V^π` is policy-specific. The **optimal value function** `V^*` is
policy-free — it's the best you could possibly do from each state:

$$
V^*(s) \;:=\; \max_\pi V^\pi(s).
$$

(The max is over all policies, possibly stochastic, in some appropriate
class.) Similarly, `Q^*(s, a) := \max_\pi Q^\pi(s, a)`.

A non-obvious fact: `V^*` and `Q^*` satisfy their own Bellman equations,
the **Bellman optimality equations**:

$$
\boxed{V^*(s) \;=\; \max_a \!\left[\, r(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \, V^*(s') \,\right]}
$$

$$
\boxed{Q^*(s, a) \;=\; r(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \max_{a'} Q^*(s', a')}
$$

The only difference from the Bellman *expectation* equation is the `\max`
in place of `\sum_a \pi(a|\cdot)`. The intuition: under the optimal
policy, you take the best action available — not the policy-weighted
average.

These equations are *nonlinear* (the max operator), so we can't solve
them with `(I - γP)^{-1} R`. We can still iterate them — the **Bellman
optimality operator** `T^* V := \max_a [r(\cdot, a) + γ P(\cdot|\cdot, a) V]`
is *also* a γ-contraction (same proof: it's a max-of-row-stochastic-mappings
shifted by `R`), and iterating it converges to `V^*`. This is *value
iteration*, Lesson 3.

---

**The fundamental theorem of MDPs.** A few consequences of the
Bellman optimality equations:

1. **Existence:** For finite MDPs with bounded rewards and `γ ∈ [0, 1)`,
   `V^*` exists and is finite, and is the unique fixed point of `T^*`.

2. **A deterministic optimal policy exists.** Define `π^*(s)` as any
   action achieving the max:

$$
\pi^*(s) \;\in\; \arg\max_a \!\left[\, r(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \, V^*(s') \,\right].
$$

   Then `V^{\pi^*} = V^*` — this deterministic policy is optimal. (Proof
   is a short induction; the policy achieves the Bellman optimality
   equation when plugged in, so by uniqueness of fixed points its `V` is
   `V^*`.)

3. **Greedy is optimal w.r.t. V^***. Once you have `V^*`, the optimal
   policy is *one-step greedy*: at each state, pick the action that
   maximizes immediate reward plus discounted next-state value. No
   long-term planning required.

This last point is striking. Once you've absorbed all the future structure
into `V^*`, every decision becomes a *myopic, immediate-reward-plus-value
maximization*. This is the deep reason value functions are useful: they
*replace planning with arithmetic*.

> **Forward link** — Q-learning (Lesson 5) and DQN (Lesson 7) learn `Q^*`
> directly from samples. Once `Q^*` is learned, `π^*(s) = \arg\max_a Q^*(s, a)`
> requires no model `P, r` — no planning, no sampling. This is what makes
> DQN "model-free": you never need to know transitions explicitly.

---

**Worked example: V\* for the gridworld** (pre-verified, computed via
value iteration to convergence in 5 iterations).

`V^*` table:

| | col 0 | col 1 | col 2 |
|-:|-:|-:|-:|
| row 0 | 0.729 | 0.810 | 0.900 |
| row 1 | 0.810 | 0.000 (pit) | 1.000 |
| row 2 | 0.900 | 1.000 | 0.000 (goal) |

Reading this table by cell:

- `(0, 0) → 0.729 = γ^3`. The agent is 3 steps from the goal under any
  optimal route; the discounted reward at the end gives `γ^3 \cdot 1`.
- `(2, 1) → 1.0`. One step from the goal: `0 + γ \cdot 0 = 0`, but the
  immediate reward of `+1` is collected on transition into the goal, so
  `V^*((2,1)) = 1.0`.
- `(0, 1) → 0.81 = γ^2`. Two steps from the goal under the optimal
  policy that routes through `(0,2) → (1,2) → (2,2)`.
- The pit's and goal's V values are 0 because they're terminal.

**Optimal policy** (one of two symmetric ones — multiple optimal policies
exist):

```
[ → ] [ → ] [ ↓ ]
[ ↓ ] [ . ] [ ↓ ]
[ → ] [ → ] [ . ]
```

Every arrow points along the optimal path to the goal, *avoiding the pit*.
The fact that both `(0,0)→(0,1)→(0,2)→(1,2)→(2,2)` and `(0,0)→(1,0)→(2,0)→(2,1)→(2,2)`
have the same return (`γ^3`) is what produces the policy non-uniqueness:
any tie-broken `\arg\max` is fine.

---

**Visualization V7 — Optimality Explorer.**

- The 3×3 gridworld with `V^*(s)` displayed as cell color + numerical label.
- Below it, the greedy policy arrows are auto-drawn.
- A side panel showing the Bellman optimality equation symbolically with
  the currently-hovered state's terms filled in.
- A "compare to V^π" mode: toggles a difference heatmap showing
  `V^*(s) - V^π(s)` for the chosen `π` (uniform / from-V2). This makes
  "regret of policy π" pop out visually.
- "Number of optimal actions per cell" badge: cells where multiple actions
  achieve the max get a special highlight, exposing policy non-uniqueness.
- Width 760, height 440.

---

### §8 — Where You'll See This Again
**Tagline:** *Every algorithm in the RL canon solves, generalizes, or learns one of these equations.*
**Length:** ~500 words.
**Anchor:** `mdps-forward-links`.

---

**Prose:**

The MDP and its Bellman equations sit at the center of everything that
follows. A field map:

**Lesson 3 (Dynamic Programming): Iterate the operators when the model is
known.** Policy iteration alternates policy evaluation
(`V^π = (I - γP^π)^{-1} R^π`) and policy improvement
(`π'(s) = \arg\max_a Q^π(s,a)`). Value iteration applies `T^*` until
convergence. Both are *exact* methods requiring full knowledge of `(P, r)`.

**Lesson 4 (Monte Carlo): Estimate V^π from sampled returns.** Don't use
the Bellman equation at all — just average observed `G_t` values across
trajectories. Robust, model-free, but slow.

**Lesson 5 (TD Learning): Estimate V^π and Q^π from *sampled Bellman
backups*.** Replace `\sum_{s'} P(s'|s,a) V(s')` with the single sample
`r + γ V(s')` from a `(s,a,r,s')` tuple. This is the *single most
important idea* in classical RL. Bias up, variance down vs Monte Carlo,
and you don't need a model.

**Lesson 6 (Function Approximation): Replace tabular V with V_θ.**
Bellman equations now hold only approximately. The "deadly triad"
(bootstrapping + off-policy + function approximation) can diverge.

**Lesson 7 (DQN): Q^* with a neural net.** Learn `Q_θ ≈ Q^*` via samples,
with target networks and replay buffers to stabilize the loss
`(r + γ \max_{a'} Q_{\bar{θ}}(s', a') - Q_θ(s, a))^2`. The loss is the
squared Bellman optimality residual.

**Lesson 8 (Policy Gradient): Learn π_θ directly.** No value function in
the loop *necessarily*, though actor-critic methods learn `V_θ` (or `Q_θ`)
as a baseline. The policy gradient theorem is
`\nabla_θ J(θ) = \mathbb{E}_{s ∼ d^π, a ∼ π}[\nabla_θ \log π_θ(a|s) \cdot A^π(s,a)]`
— and `A^π = Q^π - V^π` is from this lesson.

**Lesson 10+ (Max-Ent RL, RL as Inference):** Add an entropy bonus to the
reward; the Bellman equation acquires a `\log Z` term and the optimal
policy becomes a softmax over Q-values. Same equations, with a temperature.

**Lesson 16 (Diffusion in RL):** Even diffusion-policy methods, which look
nothing like Bellman backups, are training a sampler that maximizes
`E_{π}[Q^π(s, a)]` — i.e., they're optimizing the same value function we
just defined.

The MDP isn't a starting point you forget. It's the **center of gravity**
of the curriculum. Every lesson going forward is either a way to solve
the Bellman equation, a way to *approximately* solve it, or a way to
relax the assumptions and re-derive a generalized form.

---

**Visualization V8 — Roadmap Mini (updated).**

- The same small roadmap thumbnail from Lessons 1 and Prereq A, now with
  Lesson 2 (MDPs) highlighted.
- Forward links: outgoing arrows from MDPs to Lessons 3, 4, 5, 7, 8, 10, 16.
- Each forward arrow's hover shows the anchor in MDPs it connects from
  ("policy iteration uses §6 matrix form", "DQN learns §7's Q^* directly",
  etc.).
- This is the moment in the curriculum where the roadmap *fills out
  visibly*. The branching nature of the field becomes apparent.
- Width 760, height 280.

---

## 5. Algorithm / Math Implementation

TypeScript module `src/mdp/`. Core types:

```ts
export interface MDP {
  readonly nS: number;
  readonly nA: number;
  readonly gamma: number;
  /** P[s][a][s'] — transition probability. */
  readonly P: number[][][];
  /** r[s][a] — expected immediate reward. */
  readonly r: number[][];
  /** terminals[s] — true if s is terminal. */
  readonly terminals: boolean[];
}

export interface Policy {
  /** pi[s][a] — probability of taking action a in state s. */
  readonly pi: number[][];
}
```

Key functions:

```ts
import { Matrix, solve } from 'ml-matrix';

/** Build the policy-induced transition matrix P^π. */
export function pPi(mdp: MDP, policy: Policy): Matrix { /* ... */ }

/** Build the policy-induced reward vector R^π. */
export function rPi(mdp: MDP, policy: Policy): number[] { /* ... */ }

/** Solve V^π = (I - γP^π)^{-1} R^π via direct linear solve. */
export function policyEvaluationExact(mdp: MDP, policy: Policy): number[] {
  const Ppi = pPi(mdp, policy);
  const Rpi = Matrix.columnVector(rPi(mdp, policy));
  const A = Matrix.eye(mdp.nS).sub(Ppi.mul(mdp.gamma));
  return solve(A, Rpi).getColumn(0);
}

/** Compute Q^π from V^π using the one-step Bellman relation. */
export function qFromV(mdp: MDP, V: number[]): number[][] { /* ... */ }

/** Advantage A^π(s,a) = Q^π(s,a) − V^π(s). */
export function advantage(Q: number[][], V: number[]): number[][] { /* ... */ }

/** One iteration of the Bellman expectation operator T^π. */
export function bellmanExpectationBackup(mdp: MDP, policy: Policy, V: number[]): number[] {
  const Vnew = new Array(mdp.nS).fill(0);
  for (let s = 0; s < mdp.nS; s++) {
    if (mdp.terminals[s]) { Vnew[s] = 0; continue; }
    let v = 0;
    for (let a = 0; a < mdp.nA; a++) {
      let qsa = mdp.r[s][a];
      for (let sp = 0; sp < mdp.nS; sp++) {
        qsa += mdp.gamma * mdp.P[s][a][sp] * V[sp];
      }
      v += policy.pi[s][a] * qsa;
    }
    Vnew[s] = v;
  }
  return Vnew;
}

/** One iteration of the Bellman optimality operator T^*. */
export function bellmanOptimalityBackup(mdp: MDP, V: number[]): number[] { /* ... */ }

/** Sample one trajectory under a policy. */
export function rollout(mdp: MDP, policy: Policy, s0: number, maxSteps: number,
                        rng: () => number = Math.random): { s: number; a: number; r: number; sp: number }[] { /* ... */ }
```

The gridworld is constructed by `buildGridworld(opts: GridworldOpts): MDP`
in `src/mdp/gridworld.ts`. Options include deterministic vs slippery
transitions, custom pit/goal positions, custom γ.

**Vitest targets** (from pre-verified numerics):

```ts
test('V^π for uniform policy on gridworld', () => {
  const mdp = buildGridworld({ slippery: false, gamma: 0.9 });
  const uniform = uniformPolicy(mdp);
  const V = policyEvaluationExact(mdp, uniform);
  // V(0,0) ≈ -0.4205
  expect(V[idx(0,0)]).toBeCloseTo(-0.4205, 3);
  // V(0,1) ≈ -0.5139
  expect(V[idx(0,1)]).toBeCloseTo(-0.5139, 3);
});

test('V^* equals γ^3 at start of deterministic gridworld', () => {
  const mdp = buildGridworld({ slippery: false, gamma: 0.9 });
  let V = new Array(9).fill(0);
  for (let k = 0; k < 100; k++) V = bellmanOptimalityBackup(mdp, V);
  expect(V[idx(0,0)]).toBeCloseTo(0.729, 4);  // γ^3
  expect(V[idx(2,1)]).toBeCloseTo(1.0, 4);
});

test('Q*((1,0), right) = -1 (steps into pit)', () => {
  const mdp = buildGridworld({ slippery: false, gamma: 0.9 });
  let V = new Array(9).fill(0);
  for (let k = 0; k < 100; k++) V = bellmanOptimalityBackup(mdp, V);
  const Q = qFromV(mdp, V);
  expect(Q[idx(1,0)][RIGHT]).toBeCloseTo(-1.0, 6);
  expect(Q[idx(1,0)][DOWN]).toBeCloseTo(0.81, 4);
});

test('Advantage of optimal action is 0', () => {
  const mdp = buildGridworld({ slippery: false, gamma: 0.9 });
  let V = new Array(9).fill(0);
  for (let k = 0; k < 100; k++) V = bellmanOptimalityBackup(mdp, V);
  const Q = qFromV(mdp, V);
  const A = advantage(Q, V);
  // At (1,0), down is optimal, A = 0
  expect(A[idx(1,0)][DOWN]).toBeCloseTo(0, 6);
  expect(A[idx(1,0)][RIGHT]).toBeCloseTo(-1.81, 4);
});

test('V*(0,0) scales as γ^3', () => {
  for (const gamma of [0.5, 0.7, 0.9, 0.99]) {
    const mdp = buildGridworld({ slippery: false, gamma });
    let V = new Array(9).fill(0);
    for (let k = 0; k < 200; k++) V = bellmanOptimalityBackup(mdp, V);
    expect(V[idx(0,0)]).toBeCloseTo(Math.pow(gamma, 3), 4);
  }
});
```

---

## 6. Component Catalog

| Code | Component                       | Section | Polish |
|-----|---------------------------------|---------|--------|
| V1  | `<MDPAnatomyExplorer>`          | §1      | 1.5 days |
| V2  | `<PolicyExplorer>`              | §2      | 1 day  |
| V3  | `<ReturnComposer>`              | §3      | 1 day  |
| V4  | `<ValueHeatmap>`                | §4      | 1.5 days |
| V5  | `<QQuadrantsAndAdvantage>`      | §5      | 1.5 days |
| V6  | `<BellmanBackupLab>`            | §6      | **3-4 days** (centerpiece) |
| V7  | `<OptimalityExplorer>`          | §7      | 1.5 days |
| V8  | `<RoadmapMini>` (update)        | §8      | 0.5 day |

Total polish budget: ~11-12 days. Roughly equal to Bandits — appropriate for
a critical-path lesson. The Bellman Backup Lab is the protected investment.

**Shared from Lesson 2 (Markov Chains).** Reuse:
- `MathBlock`, `CrosslinkCallout`, `PanelChrome`, `RoadmapMini` chrome.
- `TransitionGraph` for the small MDP-as-chain views in V1 if needed.

**New shared primitives in this lesson** (worth promoting to
`src/components/` proper for Lesson 3+ reuse):
- `<GridworldRenderer>` — the core 3×3 grid drawing with cell shading,
  policy arrows, action quadrants. Used by V1, V2, V4, V5, V6, V7. The
  single most reused component. Get its API right.
- `<MDPEditor>` — a side-panel widget for tweaking γ, transition
  stochasticity, pit/goal positions. Used by V1, V6, V7.

---

## 7. Page-Level UX

Single-page scroll, same as Lessons 1 and Prereq A. The Bellman Backup
Lab (V6) is the only component that breaks out to 960px width.

A new consideration: **value-function color scales should be consistent
across V4, V5, V6, V7**. Use the same domain `[-1.5, +1.0]` and the same
diverging palette everywhere. Otherwise the learner reads `0.729` as
"green-ish" in one chart and "yellow-ish" in another, which is confusing.
Have a single shared `valueColorScale(v)` function.

---

## 8. Acceptance Criteria

After this lesson the learner can:

1. State the MDP tuple `(\mathcal{S}, \mathcal{A}, P, r, \gamma)` precisely
   and identify each component in an example.
2. Write down deterministic vs stochastic policies.
3. Compute the return `G_0` for a given trajectory and `γ`.
4. State the definitions of `V^π`, `Q^π`, and `A^π`, and relate them.
5. Derive the Bellman expectation equation from the definition of `V^π`.
6. Express policy evaluation as a linear solve `(I - γP^π)V = R^π`.
7. State the Bellman optimality equation and explain why a *deterministic*
   optimal policy exists.
8. Hand-compute `V^*` for a small (≤ 4 states) deterministic MDP.

Concrete check: hand them a 4-state chain MDP (states 0,1,2,3; actions
left/right; deterministic transitions; rewards `+1` on entering state 3,
`-1` on entering state 0; γ = 0.9). Ask them to (a) write down the MDP
tuple, (b) compute `V^π` for the policy "always right" by hand, (c)
compute `V^*` and identify the optimal policy, (d) verify both with V6's
Bellman Backup Lab using a custom MDP.

---

## 9. Stretch Goals (post-MVP)

- **Continuous-state MDPs**: a brief appendix introducing `\mathcal{S} = \mathbb{R}^d`
  via a continuous gridworld (1D lake). All equations carry over with
  integrals replacing sums.
- **Average-reward MDPs**: when `γ = 1` and the chain is recurrent, the
  natural objective is `\lim_{T \to \infty} \frac{1}{T} E[\sum_t R_t]`. This is
  the *average reward criterion*, used in some advanced RL papers.
- **POMDPs preview**: what changes when the agent observes `o_t` (not `s_t`)?
  A small inline note pointing to belief-state MDPs.
- **Per-state action set `\mathcal{A}(s)`**: generalize to state-dependent
  action availability.

---

## 10. Out of Scope (intentionally)

- **Policy iteration and value iteration** — these are Lesson 3.
- **Learning from samples** (TD, MC) — these are Lessons 4-5.
- **Function approximation** — Lesson 6.
- **Constrained MDPs**, **multi-objective RL**, **risk-sensitive RL**.
  All are downstream specializations.
- **Infinite or continuous state spaces** beyond a brief mention.

---

## 11. Training Notebook

**Not applicable.** No models are trained. The script
`scripts/mdp_gridworld.py` (~50 lines) pre-computes V*, Q*, V^π for a
small set of policies, and a 40-step Bellman backup trace, into
`public/data/mdp/`.

```
public/data/mdp/
  gridworld_v_star.json
  gridworld_v_pi_uniform.json
  gridworld_q_star.json
  gridworld_backup_trace.json
```

The TS implementation reproduces these in-browser as well. JSON is for
fast initial render before the first user interaction.

---

## End of spec

Total length: ~1500 lines. Comparable to Bandits — appropriate for a
critical-path lesson with heavy vocabulary load. The Bellman Backup Lab
is the centerpiece; everything else feeds into making it intelligible.
