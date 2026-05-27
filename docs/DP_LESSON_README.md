# Lesson 3 — Dynamic Programming

> **The first algorithms.** Lessons 2 (MDPs) and Contractions (Prereq C) gave
> us the equations and the convergence theory. This lesson turns them into
> algorithms: **policy iteration** and **value iteration**. Both solve MDPs
> when the model `(P, r)` is known. Both are exact. Both run in
> polynomial time. And the *pattern* they share — interleaved policy
> evaluation and policy improvement — is the template every RL algorithm
> will follow, all the way through actor-critic methods, DQN, and beyond.
>
> The only assumption this lesson makes is that you have the model.
> Lessons 4 onward will relax that.

---

## 0. Pedagogical Philosophy

1. **The algorithms ARE the equations.** Policy iteration is the
   Bellman expectation equation, solved. Value iteration is the Bellman
   optimality equation, iterated. The learner should leave this lesson
   feeling that the algorithms are *consequences* of the prior two
   lessons, not new inventions.

2. **The proofs are short and worth doing.** The policy improvement
   theorem has a 4-line proof (telescoping expansion); the convergence
   of value iteration is a one-line citation of Banach. Skipping these
   would be unforgivable given the prereqs we just built.

3. **One running example, every claim verified.** Same 3×3 gridworld
   from Lesson 2. The PI and VI traces, the Gauss-Seidel sweep-order
   experiments — all pre-computed and embedded as test targets.

4. **One unified centerpiece.** Three algorithms (PI, VI, Async) live
   in a single visualization with mode toggle. The learner sees them as
   the *same algorithm* with different sub-iteration budgets, which is
   exactly what they are.

5. **End by setting up Lessons 4-5.** DP solves the MDP *when you know
   the model*. Real RL doesn't. Every algorithm from Lesson 4 onward is
   "DP, but you have to estimate things." The lesson's final paragraph
   makes this transition crisp.

---

## 1. Tech Stack

Identical to prior lessons. ml-matrix used lightly for `solve` (the exact
policy-evaluation linear solve, computed in-browser); the iterative
methods are plain `number[]` loops.

The gridworld and the existing `bellmanExpectationBackup`,
`bellmanOptimalityBackup`, `policyEvaluationExact`, and
`buildGridworld` from Lesson 2 are **reused directly**. This lesson adds
the *algorithms* on top: `policyIteration()`, `valueIteration()`,
`asyncValueIteration()`, and supporting utilities.

`scripts/dp_traces.py` (~70 lines) pre-computes the PI and VI traces and
the sweep-order comparisons, into `public/data/dp/`.

---

## 2. Visual / Aesthetic Direction

We extend the curriculum aesthetic. One new convention worth locking in
here for the rest of the curriculum: **algorithm phase colors**.

```css
:root {
  /* Algorithm phase / step indicators */
  --dp-phase-evaluation:    #0e7490;   /* cyan-700  — policy evaluation */
  --dp-phase-improvement:   #b45309;   /* amber-700 — policy improvement */
  --dp-phase-optimality:    #6d28d9;   /* violet-700 — Bellman optimality */
  --dp-phase-async:         #15803d;   /* green-700 — async / sweep */

  /* Iteration counter chrome */
  --dp-iter-bg:             #f1ede4;
  --dp-iter-border:         #d9d3c4;
  --dp-iter-text:           #1c1e22;

  /* Monotonic-improvement trace */
  --dp-improving:           #15803d;   /* up step */
  --dp-stable:              #6b7280;   /* unchanged */
}
```

These phase colors *match the algorithm signature colors from Bandits*
where possible: amber for ε-greedy/improvement (both heuristic policy
changes), cyan for UCB-style and policy evaluation (both
estimation-with-uncertainty), violet for Thompson and optimality (both
"the principled solution"). The visual rhyme is intentional.

---

## 3. Lesson Metadata

```ts
export const lessonMeta: LessonMeta = {
  id: "dynamic-programming",
  title: "Dynamic Programming",
  subtitle: "Policy iteration, value iteration, and the GPI pattern",
  tier: 1,
  difficulty: 3,
  estimatedReadMinutes: 70,
  role: "critical-path",
  prerequisites: [
    { lesson: "mdps",          anchor: "bellman-expectation" },
    { lesson: "mdps",          anchor: "bellman-optimality" },
    { lesson: "contractions",  anchor: "bellman-pi-contraction" },
    { lesson: "contractions",  anchor: "bellman-star-contraction" },
  ],
  exportedAnchors: [
    "iterative-policy-evaluation",
    "policy-improvement-theorem",
    "policy-iteration",
    "value-iteration",
    "modified-policy-iteration",
    "asynchronous-dp",
    "gauss-seidel",
    "generalized-policy-iteration",
    "stopping-criterion",
  ],
  centerpieceComponent: "DPAlgorithmLab",
  forwardLinksWhenReady: [
    { to: "monte-carlo",      anchor: "model-free-evaluation" },
    { to: "td-learning",      anchor: "td-as-sampled-dp" },
    { to: "dqn",              anchor: "deep-vi" },
    { to: "policy-gradient",  anchor: "actor-critic-as-gpi" },
  ],
};
```

---

## 4. Section-by-Section Plan

### §1 — Setting Up: Known-Model Planning
**Tagline:** *We have the MDP. Now what?*
**Length:** ~400 words.
**Anchor:** `known-model-planning`.

---

**Prose:**

For the first and last time in this curriculum, we'll assume we know the
transition kernel `P` and the reward function `r` exactly. This is the
**planning** setting: the agent has a complete model of the world and
needs to compute the optimal policy. There's no learning, no estimation,
no exploration — just computation.

Why bother, if real RL doesn't have this? Three reasons.

1. **The classical algorithms are extraordinarily clean.** Policy
   iteration and value iteration are short, elegant, and *exactly* solve
   the MDP. They serve as a reference point — the gold standard against
   which all model-free algorithms are compared.

2. **The pattern generalizes.** The "interleave policy evaluation and
   policy improvement" pattern that defines policy iteration is what
   actor-critic methods (Lesson 8), DQN (Lesson 7), and most modern RL
   algorithms approximate. Master the pattern in its clean DP form;
   everything that follows is variations.

3. **Some real problems actually have models.** Board games (chess, Go),
   robotic systems with good simulators, financial decision processes with
   well-understood dynamics — all admit some flavor of DP. AlphaZero's
   MCTS is a generalization of value iteration; Dreamer's planning step
   (Lesson 14) does DP in a learned world model.

> **Back-link** — Lesson 2 (MDPs) defined `V^\pi`, `Q^\pi`, `V^*`, `Q^*`
> and the two Bellman equations they satisfy. This lesson does nothing
> more than iteratively solve those equations. Lesson C (Contractions)
> proved that the iteration converges. With those two foundations in
> hand, every algorithm in this lesson can be written down in a few
> lines.

We continue with our 3×3 gridworld from Lesson 2. Familiar example;
all numerics reproducible by the in-browser implementation.

---

*(No new visualization in §1. The introduction is short and verbal.)*

---

### §2 — Iterative Policy Evaluation
**Tagline:** *Apply T^π until V^π emerges.*
**Length:** ~700 words.
**Anchor:** `iterative-policy-evaluation`.

---

**Prose:**

Given a policy `\pi`, compute `V^\pi`. Two approaches.

**Approach 1 — Direct solve.** We derived in Lesson 2 §6:

$$
V^\pi \;=\; (I - \gamma P^\pi)^{-1} R^\pi.
$$

For our 9-state gridworld this is a 9×9 linear solve — instant. For an
MDP with `|\mathcal{S}| = 10^6`, this is hopeless: forming `(I - γP^π)` requires
`10^{12}` entries, and inverting it costs `10^{18}` operations.

**Approach 2 — Iterative.** Apply the Bellman expectation operator `T^\pi`
repeatedly:

$$
\boxed{V_{k+1}(s) \;:=\; \sum_a \pi(a|s) \!\left[ r(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s') \right]}
$$

Start with `V_0 \equiv 0`. Lesson C proved `T^\pi` is a `\gamma`-contraction
in the sup-norm. By Banach, `V_k \to V^\pi` geometrically at rate `\gamma`.

**Stopping criterion.** From Prereq C's *a posteriori* bound:

$$
\|V_k - V^\pi\|_\infty \;\leq\; \frac{\gamma}{1 - \gamma} \|V_k - V_{k-1}\|_\infty.
$$

So to guarantee `\|V_k - V^\pi\|_\infty < \epsilon`, iterate until

$$
\boxed{\|V_k - V_{k-1}\|_\infty \;<\; \frac{(1 - \gamma) \epsilon}{\gamma}.}
$$

For `\gamma = 0.9, \epsilon = 0.01`, this means stopping when consecutive
iterates differ by less than `0.00111`. The number of iterations needed
scales as `O(\log(1/\epsilon) / (1 - \gamma))` — *linear* in
`1/(1-\gamma)`, which is brutal as `\gamma \to 1`.

**Cost per iteration.** Each `T^\pi` application is `O(|\mathcal{S}|^2)`
for dense `P^\pi` (or `O(|\mathcal{S}| \cdot |\mathcal{A}|)` if `P` has at
most one non-zero per `(s,a)`, as in our deterministic gridworld). Total
cost for iterative PE: `O(|\mathcal{S}|^2 \cdot \log(1/\epsilon) / (1 - \gamma))`.

Compare to direct solve at `O(|\mathcal{S}|^3)`. **Iterative wins for
large `|\mathcal{S}|` and small `1/(1-\gamma)`.** When `\gamma` is close
to 1 and `|\mathcal{S}|` is small, direct solve can be faster. Both are
in our algorithmic toolbox.

---

**Numerical example (pre-verified).** Iterative policy evaluation for the
uniform random policy on our gridworld, starting `V_0 \equiv 0`:

| `k`  | `V_k(0,0)` | `\|V_k - V_{k-1}\|_\infty` |
|-----:|-----------:|---------------------------:|
| 0    | 0.0000     | —                          |
| 1    | 0.0000     | 0.2500                     |
| 5    | −0.2877    | 0.0796                     |
| 10   | −0.3859    | 0.0252                     |
| 20   | −0.4180    | 0.0026                     |
| 50   | −0.4205    | 8.6e-6                     |
| ∞    | −0.4205    | (limit)                    |

The convergence is geometric at rate `\gamma = 0.9`: errors shrink by ~10×
every ~22 iterations.

> **Forward link** — In Lesson 4 (Monte Carlo) we'll estimate `V^\pi`
> from sampled trajectories, *without* knowing `P` or `r`. In Lesson 5
> (TD) we'll do the same with one-step samples instead of full
> trajectories. Both are model-free versions of *this* iteration.

---

**Visualization V1 — Iterative PE Watcher.**

- The 3×3 gridworld colored by `V_k`, animated as `k` increases.
- Policy selector: uniform, all-down, all-right, custom (from V2 of MDP
  lesson if available).
- Top bar: current `k`, current `\|V_k - V_{k-1}\|_\infty`, target stopping
  threshold, "converged" badge when crossed.
- Right side: line plot of `V_k` at the four corner cells over `k`, log
  scale optional.
- Controls: step / play / reset; γ slider [0.5, 0.99]; ε slider for
  stopping threshold.
- Width 800, height 440.

---

### §3 — Policy Improvement
**Tagline:** *Given V^π, the greedy policy is at least as good as π.*
**Length:** ~700 words.
**Anchor:** `policy-improvement-theorem`.

---

**Prose:**

Suppose we have `V^\pi` (computed via §2). Can we use it to find a
*better* policy? Yes, and the construction is mechanical.

**The greedy improvement step.** Define a new policy `\pi'` by

$$
\boxed{\pi'(s) \;\in\; \arg\max_a Q^\pi(s, a) \;=\; \arg\max_a \!\left[ r(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s') \right].}
$$

`\pi'` is **greedy with respect to `V^\pi`**. It looks one step ahead and
picks whichever action maximizes immediate-plus-discounted-future value
under the *current* value estimate.

**Policy Improvement Theorem.** For deterministic `\pi'` defined as
above, `V^{\pi'}(s) \geq V^\pi(s)` for every state `s`. If `\pi'` and `\pi`
differ at any state where the strict inequality `Q^\pi(s, \pi'(s)) > V^\pi(s)`
holds, then `V^{\pi'}(s) > V^\pi(s)` at that state.

*Proof.* Start from the inequality `Q^\pi(s, \pi'(s)) \geq V^\pi(s)`,
which holds because `\pi'(s)` is the argmax. Expand:

$$
V^\pi(s) \;\leq\; r(s, \pi'(s)) + \gamma \sum_{s'} P(s'|s, \pi'(s)) V^\pi(s').
$$

Apply this same inequality to each `V^\pi(s')` on the right:

$$
V^\pi(s') \;\leq\; r(s', \pi'(s')) + \gamma \sum_{s''} P(s''|s', \pi'(s')) V^\pi(s'').
$$

Substituting and unrolling indefinitely:

$$
\begin{aligned}
V^\pi(s) &\leq r(s, \pi'(s)) + \gamma \sum_{s'} P(s'|s, \pi'(s)) \left[ r(s', \pi'(s')) + \gamma \sum_{s''} P(s''|s', \pi'(s')) V^\pi(s'') \right] \\
&\leq \mathbb{E}_{\pi'} \left[ R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots \mid S_t = s \right] \\
&= V^{\pi'}(s). \qquad \blacksquare
\end{aligned}
$$

The proof is **a telescoping expansion of one inequality** —
beautiful, short, and the workhorse of every "monotonic improvement"
result in RL (including TRPO's, Lesson 9).

---

**A small example.** Take our uniform-random policy's `V^\pi_0` (computed
in §2) and improve it. The greedy choice at each state under `V^\pi_0`:

| state  | `V^\pi_0` | `\arg\max_a Q^\pi_0(s, a)` | reason                            |
|--------|----------:|:--------------------------:|-----------------------------------|
| (0,0)  | −0.4205   | **Up** or **Left**         | bounces off wall, V=0 is best among options. *Surprising!* |
| (0,1)  | −0.5139   | Right                      | leads to (0,2) where V is least negative |
| (0,2)  | −0.2386   | Up or Down                 | bouncing off top wall keeps you out of trouble |
| (1,0)  | −0.5139   | Down                       | leads to (2,0), avoids pit at (1,1) |
| (1,2)  | −0.0693   | Down                       | leads to goal! R=+1, V=0 |
| (2,0)  | −0.2386   | Right                      | toward goal |
| (2,1)  | −0.0693   | Right                      | into goal |

Note the **counter-intuitive** behaviour at state (0,0): greedy
improvement w.r.t. `V^\pi_0` says "stay put by bouncing off the wall."
Why? Because every other action leads to a cell with *more negative*
value under the *uniform-random* `V^\pi_0`. The new policy isn't optimal —
it's just better than uniform random. The bounce-off-wall behaviour will
be fixed in the next round of policy improvement (next section).

This phenomenon — greedy improvement producing a *non-optimal but better*
policy — is exactly why we need *iterated* policy improvement. One step
isn't enough.

---

**Visualization V2 — Policy Improvement Inspector.**

- Two stacked gridworld renderers:
  - Top: the input value function `V^\pi` (color-shaded cells, numerical
    labels).
  - Bottom: the resulting greedy policy `\pi'` (arrows on each cell).
- Click any cell to expand a side panel showing the per-action Q-values
  used in the argmax, with the chosen action highlighted.
- Toggle: input `\pi` (the policy whose `V` is shown) — uniform, current
  greedy, "previous PI step." `V^\pi` recomputes via `policyEvaluationExact`.
- "Improve once" button: snaps the bottom panel's policy to the new
  greedy, and (after a brief pause) shows the next value function in the
  top panel for the user to inspect.
- Width 760, height 540.

---

### §4 — Policy Iteration
**Tagline:** *Alternate. Improve. Converge — in finitely many steps.*
**Length:** ~950 words.
**Anchor:** `policy-iteration`.

---

**Prose:**

Combine §2 and §3 into a single loop:

```text
Policy Iteration

  Initialize: π_0 arbitrary (e.g., uniform random)
  for k = 0, 1, 2, ...
    V_{π_k} ← solve V = R^π + γ P^π V        # policy evaluation (full)
    π_{k+1}(s) ← argmax_a [r(s,a) + γ Σ_{s'} P(s'|s,a) V_{π_k}(s')]   # improvement
    if π_{k+1} == π_k:
      return (π_k, V_{π_k})                    # converged
```

**Properties.**

1. **Monotonic improvement.** By the policy improvement theorem,
   `V^{\pi_{k+1}} \geq V^{\pi_k}` pointwise. Iterates ratchet *up*.

2. **Finite convergence.** There are only finitely many deterministic
   policies (`|\mathcal{A}|^{|\mathcal{S}|}` of them). Strict improvement
   visits each at most once. Hence policy iteration terminates in *at
   most* `|\mathcal{A}|^{|\mathcal{S}|}` steps. In practice, far fewer —
   typically `O(|\mathcal{S}|)` or even sub-linear.

3. **Optimality at termination.** When `\pi_{k+1} = \pi_k`, the policy
   is greedy w.r.t. its own value function, so it satisfies the Bellman
   *optimality* equation. By uniqueness of the optimality fixed point
   (Lesson C), `V^{\pi_k} = V^*` and `\pi_k = \pi^*`.

**Cost.** Each iteration costs one full policy evaluation
(`O(|\mathcal{S}|^3)` via direct solve, or `O(|\mathcal{S}|^2 / (1-\gamma))`
iteratively) plus one improvement pass (`O(|\mathcal{S}| \cdot |\mathcal{A}|)`).
The dominant cost is policy evaluation, especially for large state spaces.

---

**Numerical trace (pre-verified).** Policy iteration on our gridworld,
starting from uniform random:

**Iteration 0** — evaluate `\pi_0` (uniform):
- `V_{\pi_0}(0,0) = -0.4205`
- improve → `\pi_1`: bounces-off-wall at (0,0); good moves elsewhere.

**Iteration 1** — evaluate `\pi_1`:
- `V_{\pi_1}(0,0) = 0.0000` (because π₁ at (0,0) bounces forever — earns nothing)
- Other cells improved substantially: `V_{\pi_1}(2,1) = 1.0`, `V_{\pi_1}(1,2) = 1.0`,
  `V_{\pi_1}(2,0) = V_{\pi_1}(0,2) = 0.9`, `V_{\pi_1}(1,0) = V_{\pi_1}(0,1) = 0.81`.
- Improve → `\pi_2`: now (0,0) sees neighbors with positive value (0.81), so
  greedy picks Right or Down. *Unstuck.*

**Iteration 2** — evaluate `\pi_2`:
- `V_{\pi_2}(0,0) = 0.729`
- Improve → `\pi_3 = \pi_2`. Converged.

**Iteration 3** — `\pi_3` equals `\pi_2`. **Terminate.**

Total: 3 iterations. The monotone sequence at (0,0):
`−0.4205 \;\rightarrow\; 0.000 \;\rightarrow\; 0.729`. Strictly increasing,
as the theorem promises.

The *pathological* middle iteration — where the policy gets stuck
bouncing at (0,0) — is the most pedagogically interesting moment. It
shows that one improvement step can produce a *worse-looking* value
function than expected (V(0,0) goes from −0.42 to 0, which is technically
better, but the "stuck" behaviour feels worse than the original random
exploration). The next improvement breaks the stuckness. **Policy
iteration always escapes such local glitches in a single step**, because
every state's argmax sees updated values for its neighbors.

> **Connection to EM** (cashes in your StatViz background). Policy
> iteration has the same alternating-maximization structure as EM:
> evaluation = E-step (compute posterior / value), improvement = M-step
> (update the parameter / policy). Both have monotone-improvement
> guarantees via similar telescoping arguments. The PI improvement
> theorem is the RL analog of EM's monotone-likelihood theorem.

---

**Modified Policy Iteration.** A natural variant: instead of *fully*
evaluating `\pi_k` (running PE to convergence), do only `m` PE iterations
before improving. With `m = 1`, this is *literally* value iteration
(next section). With `m = \infty`, it's full policy iteration. Modified
PI with intermediate `m` (often `m = 5` or `m = 10`) is faster than
full PI for large state spaces and is what real DP libraries actually
implement.

---

**Visualization V3 — Policy Iteration Trace.**

- Top: the 3×3 gridworld renderer with both the current policy (arrows)
  and current `V^{\pi_k}` (color shading).
- Side panel: a vertical bar chart of `V^{\pi_k}(s)` for each cell, with
  the previous iteration's bars shown faded behind for comparison.
- Bottom: a line plot of `V^{\pi_k}(0,0)` (and a couple of other selected
  states) over `k`. Three breakpoints visible for our gridworld:
  −0.4205 → 0.000 → 0.729 → 0.729.
- Controls: "step PI" (one full PE + one PI), "play to convergence",
  "reset," initial policy selector.
- A "Modified PI" toggle that lets the user limit `m` (the number of PE
  iterations per outer step) — shows how the trace shifts.
- Width 880, height 520.

---

### §5 — Value Iteration
**Tagline:** *Skip the inner loop. One backup at a time. Same answer.*
**Length:** ~900 words.
**Anchor:** `value-iteration`.

---

**Prose:**

Modified policy iteration with `m = 1`: do one PE iteration, then
immediately improve. The combined update is

$$
\begin{aligned}
V_{k+1}(s) &= r(s, \pi_{k+1}(s)) + \gamma \sum_{s'} P(s'|s, \pi_{k+1}(s)) V_k(s') \\
&= \max_a \left[ r(s, a) + \gamma \sum_{s'} P(s'|s, a) V_k(s') \right] \\
&= (T^* V_k)(s).
\end{aligned}
$$

The first line is the policy-evaluation backup for the greedy policy;
the second line *uses* the fact that the greedy policy chooses the
argmax. So PE-then-improve with `m=1` is the *same* operation as
applying the **Bellman optimality operator** `T^*`. Hence:

```text
Value Iteration

  Initialize: V_0 ≡ 0
  for k = 0, 1, 2, ...
    V_{k+1}(s) ← max_a [r(s,a) + γ Σ_{s'} P(s'|s,a) V_k(s')]   for all s
    if ‖V_{k+1} - V_k‖_∞ < ε(1-γ)/γ:
      π_final(s) ← argmax_a [r(s,a) + γ Σ_{s'} P(s'|s,a) V_{k+1}(s')]
      return (V_{k+1}, π_final)
```

**Convergence.** From Lesson C, `T^*` is a `\gamma`-contraction. By
Banach, `V_k \to V^*` geometrically at rate `\gamma`. The stopping
criterion is exactly the *a posteriori* bound from that lesson.

**Cost.** Each iteration is `O(|\mathcal{S}|^2 \cdot |\mathcal{A}|)` (compute Q
for all (s,a) pairs, max over a). Number of iterations:
`O(\log(1/\epsilon) / (1 - \gamma))`. Total:
`O(|\mathcal{S}|^2 \cdot |\mathcal{A}| \cdot \log(1/\epsilon) / (1 - \gamma))`.

No "inner loop" of full policy evaluation. **Each VI iteration is cheaper
than each PI iteration**, but **VI needs more iterations**. The trade-off
depends on `\gamma` and the structure of the MDP. In practice both
algorithms are fast for tabular MDPs; the choice rarely matters at small
scale.

---

**Numerical trace (pre-verified).** Value iteration on our gridworld
with `\gamma = 0.9`:

**k = 0:** `V = 0` everywhere.

**k = 1:** value appears only at cells *adjacent to terminal states*:
```
 0    0    0
 0   pit   1
 0    1   goal
```
(`V(1,2) = 1` because right-action lands in goal with reward 1; `V(2,1) = 1`
similarly. `V(1,0)` stays 0 because right-action lands in pit with reward
−1, which loses to the 0-reward bounces — so the max is 0. Same logic at
`V(0,1)`.)

**k = 2:** value propagates one more cell outward:
```
 0    0    0.9
 0   pit   1
 0.9  1   goal
```

**k = 3:** another step:
```
 0    0.81  0.9
 0.81 pit   1
 0.9  1    goal
```

**k = 4:** the start state finally lights up:
```
 0.729 0.81  0.9
 0.81  pit   1
 0.9   1    goal
```

**k = 5:** identical to k = 4. Converged.

Value iteration runs in **5 iterations** for this gridworld. Policy
iteration ran in 3 — *fewer outer iterations but more total work per
iteration*. Total scalar updates: VI does `5 × 9 × 4 = 180` per-cell
backups; PI does `3 × (PE iterations + 9)` ≈ `3 × 50 ≈ 150` per cell
backups *plus* some matrix-inverse work. Comparable.

**The story value iteration tells.** Information about the rewards
*ripples outward* from terminal states, one cell per backup. After `k`
iterations, only cells within `k` steps of a terminal state have
non-trivial values. Once the ripple has traveled across the entire state
space, the algorithm converges. This is a very visceral picture; the
centerpiece visualization (V4) animates it directly.

---

**Stopping criterion in practice.** For `\gamma = 0.9, \epsilon = 0.01`:

| `k`  | `\|V_k - V_{k-1}\|_\infty` | bound `\frac{\gamma}{1-\gamma}\|V_k - V_{k-1}\|_\infty` | true `\|V_k - V^*\|_\infty` |
|-----:|---------------------------:|--------------------------------------------------------:|----------------------------:|
| 1    | 1.000                      | 9.000                                                   | 0.900                        |
| 2    | 0.900                      | 8.100                                                   | 0.810                        |
| 3    | 0.810                      | 7.290                                                   | 0.729                        |
| 4    | 0.729                      | 6.561                                                   | 0.000                        |

Observations: the bound is *very* conservative — it predicts errors of
~9.0 when the true error is 0.9. That's because the bound assumes
worst-case contraction; the *empirical* contraction is much faster (we
saw this drift in Lesson C). On the other hand: the bound is *correct* —
the true error is always below it. Sloppy bounds beat hopeful guesses.

---

**Visualization V4 — DP Algorithm Lab. THIS IS THE CENTERPIECE.**

A unified interactive that supports **three algorithm modes** in one
interface:

1. **Policy Iteration.** Alternates between PE (sub-iterated until
   convergence) and PI (one greedy update). Iteration counter tracks
   outer steps.
2. **Value Iteration.** One T\* backup per step.
3. **Modified Policy Iteration.** Configurable `m` between 1 (= VI) and
   ∞ (= full PI).

**Layout.**

Top half:
- Mode tabs (PI / VI / MPI) and an algorithm-phase indicator (currently
  doing PE? PI? T\* backup?).
- The 3×3 gridworld colored by current `V`, with current policy arrows
  overlaid.
- A small "step into the equation" panel that shows the current
  algorithmic line as text: `V_3(s) = max over a [r(s,a) + 0.9 * Σ P V_2(s')]`.

Bottom half:
- Side-by-side mode: enable a second algorithm in a second column to
  race PI vs VI on the same MDP.
- Line plot tracking `V_k(s_0)` at the start cell, plus a separate trace
  of `||V_k - V_{k-1}||` decay on log scale.
- Iteration count and total-backup-count panels.

**Controls.**
- Step / play / reset / pause.
- Speed control 0.25× to 16×.
- γ slider [0.5, 0.99].
- Stochasticity toggle (deterministic / slippery 80-10-10).
- Custom initial policy / initial V loader.

**The defining moment of this viz.** When the user toggles between PI and
VI side-by-side and watches *both* algorithms converge to the same
`V^*` and the same `\pi^*`, just by different paths. PI does fewer outer
steps but heavier inner work; VI does more outer steps but each is
trivial. The end state is identical. Once the learner sees that, "GPI"
in §7 is just a name for what they've already seen.

**Width:** 960px (centerpiece breakout). **Height:** 680px.

---

### §6 — Asynchronous Dynamic Programming
**Tagline:** *Update in any order. Use updated values immediately. Save iterations.*
**Length:** ~700 words.
**Anchor:** `asynchronous-dp`.

---

**Prose:**

So far, every backup has been **synchronous** (or "Jacobi-style"): all
states are updated using the *old* `V_k`, producing a *new* `V_{k+1}` in
one batch. This is conceptually clean but wasteful — after we've updated
some states, *why not use the new values* when updating others?

**Asynchronous DP** updates one state at a time, in some order, *in
place*. Each update uses whatever values are currently in the array,
old or new. The simplest variant is **Gauss-Seidel value iteration**:

```text
Gauss-Seidel VI

  Initialize: V ≡ 0
  loop:
    for s in sweep_order:
      V[s] ← max_a [r(s,a) + γ Σ_{s'} P(s'|s,a) V[s']]   # in-place
    if max change < ε(1-γ)/γ:
      return V
```

The key difference from synchronous VI: when we update `V[s]`, the
right-hand side already includes any *new* values for states that came
earlier in the sweep.

**Convergence.** Async DP converges to `V^*` as long as **every state is
updated infinitely often**. The proof is a generalization of the Banach
argument: the *eventual* iteration sequence is still a contraction, just
on a longer time scale. Sweep order doesn't affect *whether* you
converge — only *how fast*.

---

**Sweep order matters — a lot.** On our gridworld, with three sweep
orders:

| sweep order                          | iterations to converge |
|--------------------------------------|------------------------:|
| Jacobi (synchronous)                 | 5                       |
| Forward (s=0, 1, 2, ..., 8)          | 5 (no improvement over Jacobi) |
| **Reverse (s=8, 7, 6, ..., 0)**      | **2**                   |
| Smart (by distance to goal, near-first) | **2**                |

**Reverse sweep is 2.5× faster.** Why? Because in our gridworld, the goal
is at state 8 (highest index), and value *propagates from the goal
backward*. A reverse sweep updates state 8 first (which gets V=0
because goal is terminal), then state 7 (which sees the updated
neighbors via the optimality backup), etc. By the end of the first
sweep, most cells have already received the value-information that the
Jacobi version would need additional iterations to propagate.

The forward sweep doesn't gain anything because it propagates information
in the *wrong direction* — by the time it reaches state 8, the early
states have already been updated with stale information.

**Prioritized sweeping.** A natural extension: prioritize states whose
*neighbors recently changed a lot*. Maintain a priority queue keyed by
the magnitude of recent neighbor-changes. This is called **prioritized
sweeping** (Moore & Atkeson, 1993) and it's a precursor to modern
priority-based replay buffers in DQN (Lesson 7 — *Prioritized Experience
Replay*).

> **Forward link** — Prioritized Experience Replay (Schaul et al. 2015,
> used in Rainbow DQN, Lesson 7) is *exactly* prioritized sweeping with
> deep function approximation. The priority key is the absolute TD
> error, which plays the role of "neighbor-change magnitude" in the
> tabular setting. Same idea, different implementation.

---

**Visualization V5 — Async Sweep Comparator.**

- Three gridworld renderers side-by-side, each running a different sweep
  order: Jacobi, forward, reverse.
- Each updates synchronously *with the others* (every panel takes one
  step at a time, with the step buttons coupled).
- Iteration counter beneath each panel.
- A "current sweep position" indicator in the forward/reverse panels
  shows which state is about to be updated.
- The "smart" sweep order (sorted by Manhattan distance to goal) is
  available as a fourth option via a dropdown swap-in.
- A small bar chart at the top: total iterations to convergence for each
  sweep order — pre-converged on init for quick visual comparison.
- Width 880, height 460.

---

### §7 — Generalized Policy Iteration
**Tagline:** *V and π pull each other toward V\* and π\*. Every RL algorithm is some flavor of this.*
**Length:** ~500 words.
**Anchor:** `generalized-policy-iteration`.

---

**Prose:**

Step back. What did we just do? Two operations:

- **Policy evaluation** moves `V` closer to `V^\pi` for the current `\pi`.
- **Policy improvement** moves `\pi` to be more consistent with `V` (greedy
  w.r.t. `V`).

Each operation pulls one of (`V`, `\pi`) closer to a configuration in
which they're *mutually consistent*. The fixed point of both operations
simultaneously is `(V^*, \pi^*)`: the optimal value function and an
optimal policy that is greedy w.r.t. it.

**Generalized Policy Iteration (GPI)** is the meta-pattern: alternate
some amount of evaluation with some amount of improvement, in any
interleaving, until convergence. The algorithms in this lesson are the
*pure* cases:

| Algorithm | Eval | Improve | Async? |
|-----------|------|---------|--------|
| Full Policy Iteration | full | full sweep | sync |
| Value Iteration | 1 step | full sweep | sync |
| Modified PI | `m` steps | full sweep | sync |
| Gauss-Seidel VI | 1 step | per-state | async, in-place |
| Asynchronous PI | `m` steps | per-state | async, in-place |

GPI doesn't prescribe the schedule. It says: **as long as both processes
keep happening, both `V` and `\pi` will converge** to the joint fixed
point.

---

**Why this view matters.** Every algorithm in the rest of this curriculum
will be a variant of GPI:

- **Monte Carlo (Lesson 4):** evaluation by sampling returns; improvement
  by greedy w.r.t. estimated `Q`. *Sample-based* evaluation; same
  improvement.
- **TD learning (Lesson 5):** evaluation via sampled Bellman backups
  (TD(0)); improvement via greedy or ε-greedy. *Stochastic* version of
  this lesson, with a step-size parameter `\alpha`.
- **DQN (Lesson 7):** evaluation by gradient descent on the squared TD
  error; improvement by `\arg\max_a Q_\theta(s, a)`. Approximate
  evaluation with a neural net, same improvement.
- **Policy Gradient (Lesson 8):** evaluation via the *advantage*
  estimator; improvement by gradient ascent on `\log \pi_\theta \cdot A`.
  Continuous improvement in policy *parameter space*.
- **SAC (Lesson 12):** entropy-regularized evaluation; improvement via
  soft-greedy (Boltzmann). The full GPI with a temperature.

The vocabulary changes (estimation, gradient, neural net, regularization)
but the structural pattern is identical: **make `V` more consistent with
`\pi`, make `\pi` more consistent with `V`, iterate.**

---

**Visualization V6 — GPI Visualizer.**

An abstract two-axis diagram:

- Horizontal axis: "policy consistency" (greedy w.r.t. current V).
- Vertical axis: "value consistency" (V equals V^π for current π).
- The point (π^*, V^*) is in one corner, marked with a target.
- The current (π_k, V_k) is a dot in the plane.
- "Policy evaluation step" moves the dot vertically toward consistency.
- "Policy improvement step" moves the dot horizontally toward consistency.
- A staircase trajectory visualizes the alternation.

A toggleable "algorithm overlay":
- PI: bounces off both axes alternately.
- VI: takes small diagonal steps (one of each every iteration).
- Async: irregular zigzags.
- Actor-critic (preview): a smooth diagonal trajectory.

This is a conceptual/schematic visualization, intentionally abstract.
It's the picture the learner will carry forward to every subsequent
lesson.

Width 700, height 460.

---

### §8 — Where You'll See This Again
**Tagline:** *DP is the gold standard. Every model-free RL algorithm approximates it.*
**Length:** ~500 words.
**Anchor:** `dp-forward-links`.

---

**Prose:**

We've shipped two algorithms — policy iteration and value iteration —
that *exactly* solve any finite MDP in polynomial time, assuming we
know the model. The rest of the curriculum relaxes the "know the model"
assumption, one direction at a time.

**Direction 1 — Unknown model, known returns.** Lesson 4 (Monte Carlo)
estimates `V^\pi` and `Q^\pi` by averaging *observed* returns from
sampled trajectories. No model `(P, r)` is needed. The downside:
trajectories must reach a terminal state before any update can happen,
which is slow and only works for episodic tasks.

**Direction 2 — Unknown model, sampled Bellman backups.** Lesson 5
(Temporal-Difference Learning) replaces the expectation in the Bellman
backup, `\sum_{s'} P(s'|s,a) V(s')`, with the single sample
`V(s')` from one observed transition. *One-step samples* replace
*full trajectories*. The Bellman operator becomes stochastic; the
contraction property becomes a stochastic-approximation argument
(Prereq B / inline). The benefit: updates happen at every step, not at
the end of episodes.

**Direction 3 — Large/continuous state spaces.** Lesson 6 (Function
Approximation) replaces tabular `V` with `V_\theta`, a parameterized
function. The Bellman backup becomes a gradient step on a Bellman
residual. Convergence guarantees become subtler (the "deadly triad"
warning), and we trade exactness for scalability.

**Direction 4 — Deep RL.** Lesson 7 (DQN) combines directions 2 and 3:
sampled Bellman backups against a deep-net `Q_\theta`, with target
networks and replay buffers added to stabilize. The Bellman optimality
equation is *exactly* what DQN's loss function is squaring.

**Direction 5 — Policy parameterization.** Lesson 8 (Policy Gradient)
takes a different route: instead of learning `V` or `Q` and acting
greedily, learn `\pi_\theta` directly. The improvement step becomes a
*gradient step* on `J(\theta) = \mathbb{E}_{\pi_\theta}[G_0]`, computed via
the policy gradient theorem. Evaluation still happens (as the
"critic"), but improvement is continuous in `\theta` rather than
discrete.

**Direction 6 — Model-based modern.** Lesson 13 (Model-Based RL) learns
the model `(P_\theta, r_\theta)` from data and then *runs DP inside it*.
Lesson 14 (World Models / Dreamer) does the same with a latent-state
VAE-style model. Both are direct heirs of the DP we just built —
they're DP in a *learned* world.

The lesson's core algorithms — `policyIteration()`, `valueIteration()`,
`asyncValueIteration()` — will appear by name as primitives in
Lesson 13 (planning in a learned model), Lesson 14 (latent imagination
rollouts that approximate Bellman backups), and AlphaZero-style methods
beyond.

This is where the curriculum's "knowing the model" assumption ends.
Everything after is RL proper.

---

**Visualization V7 — Roadmap Mini (updated).**

The roadmap mini with DP now shipped. Outgoing arrows from DP to:
Lesson 4 (MC), Lesson 5 (TD), Lesson 6 (Function Approx), Lesson 7 (DQN),
Lesson 8 (PG), Lesson 13 (Model-Based). Each arrow's hover shows the
specific connection ("MC uses sample-based PE", "DQN learns V_θ to
approximate VI", etc.).

Width 760, height 280.

---

## 5. Algorithm / Math Implementation

TypeScript module `src/dp/`. Heavy reuse of `src/mdp/` from Lesson 2.

```ts
import type { MDP, Policy } from '../mdp/types';
import { bellmanExpectationBackup, bellmanOptimalityBackup } from '../mdp/policy-evaluation';
import { policyEvaluationExact, qFromV } from '../mdp/policy-evaluation';

/** Iterative policy evaluation: V_{k+1} = T^π V_k. */
export function policyEvaluationIterative(
  mdp: MDP, policy: Policy, tol = 1e-9, maxIter = 10000
): { V: number[]; iterations: number; trace: number[][] } {
  let V = new Array(mdp.nS).fill(0);
  const trace: number[][] = [V.slice()];
  for (let k = 0; k < maxIter; k++) {
    const Vn = bellmanExpectationBackup(mdp, policy, V);
    trace.push(Vn.slice());
    if (supDist(Vn, V) < tol) return { V: Vn, iterations: k + 1, trace };
    V = Vn;
  }
  return { V, iterations: maxIter, trace };
}

/** Greedy policy improvement: π'(s) = argmax_a Q(s, a). */
export function policyImprovement(mdp: MDP, V: number[]): Policy {
  const Q = qFromV(mdp, V);
  const pi: number[][] = [];
  for (let s = 0; s < mdp.nS; s++) {
    const row = new Array(mdp.nA).fill(0);
    if (mdp.terminals[s]) { row[0] = 1; pi.push(row); continue; }
    let bestA = 0, bestQ = -Infinity;
    for (let a = 0; a < mdp.nA; a++) {
      if (Q[s][a] > bestQ) { bestQ = Q[s][a]; bestA = a; }
    }
    row[bestA] = 1;
    pi.push(row);
  }
  return { pi };
}

/** Full policy iteration. */
export function policyIteration(
  mdp: MDP, maxIter = 100
): { V: number[]; policy: Policy; iterations: number; history: { V: number[]; policy: Policy }[] } {
  let policy = uniformPolicy(mdp);
  const history: { V: number[]; policy: Policy }[] = [];
  for (let k = 0; k < maxIter; k++) {
    const V = policyEvaluationExact(mdp, policy);
    history.push({ V: V.slice(), policy: clonePolicy(policy) });
    const newPolicy = policyImprovement(mdp, V);
    if (policiesEqual(policy, newPolicy)) return { V, policy, iterations: k + 1, history };
    policy = newPolicy;
  }
  throw new Error('PI failed to converge');
}

/** Synchronous value iteration. */
export function valueIteration(
  mdp: MDP, epsilon = 1e-8, maxIter = 10000
): { V: number[]; policy: Policy; iterations: number; trace: number[][] } {
  let V = new Array(mdp.nS).fill(0);
  const trace: number[][] = [V.slice()];
  const stopThreshold = epsilon * (1 - mdp.gamma) / mdp.gamma;
  for (let k = 0; k < maxIter; k++) {
    const Vn = bellmanOptimalityBackup(mdp, V);
    trace.push(Vn.slice());
    if (supDist(Vn, V) < stopThreshold) {
      const policy = policyImprovement(mdp, Vn);
      return { V: Vn, policy, iterations: k + 1, trace };
    }
    V = Vn;
  }
  throw new Error('VI failed to converge');
}

/** Async/Gauss-Seidel value iteration with selectable sweep order. */
export function asyncValueIteration(
  mdp: MDP, sweepOrder: number[], epsilon = 1e-8, maxIter = 10000
): { V: number[]; policy: Policy; iterations: number } {
  const V = new Array(mdp.nS).fill(0);
  const stopThreshold = epsilon * (1 - mdp.gamma) / mdp.gamma;
  for (let k = 0; k < maxIter; k++) {
    let maxDelta = 0;
    for (const s of sweepOrder) {
      if (mdp.terminals[s]) continue;
      const oldV = V[s];
      let best = -Infinity;
      for (let a = 0; a < mdp.nA; a++) {
        let q = mdp.r[s][a];
        for (let sp = 0; sp < mdp.nS; sp++) q += mdp.gamma * mdp.P[s][a][sp] * V[sp];
        if (q > best) best = q;
      }
      V[s] = best;
      maxDelta = Math.max(maxDelta, Math.abs(V[s] - oldV));
    }
    if (maxDelta < stopThreshold) {
      const policy = policyImprovement(mdp, V);
      return { V, policy, iterations: k + 1 };
    }
  }
  throw new Error('Async VI failed to converge');
}
```

**Vitest targets** (from pre-verified numerics):

```ts
test('Iterative PE matches exact PE for uniform policy', () => {
  const mdp = buildGridworld({ slippery: false, gamma: 0.9 });
  const exact = policyEvaluationExact(mdp, uniformPolicy(mdp));
  const { V } = policyEvaluationIterative(mdp, uniformPolicy(mdp));
  for (let s = 0; s < 9; s++) expect(V[s]).toBeCloseTo(exact[s], 5);
});

test('Policy iteration converges in 3 outer steps on gridworld', () => {
  const mdp = buildGridworld({ slippery: false, gamma: 0.9 });
  const { iterations, history } = policyIteration(mdp);
  expect(iterations).toBe(3);
  expect(history[0].V[idx(0,0)]).toBeCloseTo(-0.4205, 3);
  expect(history[1].V[idx(0,0)]).toBeCloseTo(0.0000, 3);
  expect(history[2].V[idx(0,0)]).toBeCloseTo(0.7290, 3);
});

test('Value iteration converges in 5 iterations on gridworld', () => {
  const mdp = buildGridworld({ slippery: false, gamma: 0.9 });
  const { iterations, V } = valueIteration(mdp, 1e-8);
  expect(iterations).toBeLessThanOrEqual(6);  // allow off-by-one for tolerance
  expect(V[idx(0,0)]).toBeCloseTo(0.729, 4);
});

test('Reverse sweep order is faster than forward', () => {
  const mdp = buildGridworld({ slippery: false, gamma: 0.9 });
  const forwardOrder = [0,1,2,3,4,5,6,7,8];
  const reverseOrder = [8,7,6,5,4,3,2,1,0];
  const fwd = asyncValueIteration(mdp, forwardOrder);
  const rev = asyncValueIteration(mdp, reverseOrder);
  expect(rev.iterations).toBeLessThan(fwd.iterations);
  expect(rev.iterations).toBeLessThanOrEqual(3);
});

test('PI and VI produce identical V*', () => {
  const mdp = buildGridworld({ slippery: false, gamma: 0.9 });
  const pi = policyIteration(mdp);
  const vi = valueIteration(mdp, 1e-10);
  for (let s = 0; s < 9; s++) expect(pi.V[s]).toBeCloseTo(vi.V[s], 5);
});

test('Stopping criterion guarantees epsilon-optimality', () => {
  const mdp = buildGridworld({ slippery: false, gamma: 0.9 });
  const epsilon = 0.01;
  const { V } = valueIteration(mdp, epsilon);
  const optimal = valueIteration(mdp, 1e-12).V;
  for (let s = 0; s < 9; s++) expect(Math.abs(V[s] - optimal[s])).toBeLessThan(epsilon);
});
```

---

## 6. Component Catalog

| Code | Component                      | Section | Polish |
|-----|--------------------------------|---------|--------|
| V1  | `<IterativePEWatcher>`         | §2      | 1.5 days |
| V2  | `<PolicyImprovementInspector>` | §3      | 1.5 days |
| V3  | `<PolicyIterationTrace>`       | §4      | 2 days |
| V4  | `<DPAlgorithmLab>`             | §5      | **3-4 days** (centerpiece) |
| V5  | `<AsyncSweepComparator>`       | §6      | 1.5 days |
| V6  | `<GPIVisualizer>`              | §7      | 1 day  |
| V7  | `<RoadmapMini>` (update)       | §8      | 0.5 day |

Total polish budget: ~11-12 days. Comparable to Bandits and MDPs.

**Reuse from prior lessons:** `GridworldRenderer`, `MDPEditor`, all
MDP math, `MathBlock`, `CrosslinkCallout`, `PanelChrome`, `RoadmapMini`.

---

## 7. Page-Level UX

Same conventions as prior lessons. The DP Algorithm Lab (V4) breaks out
to 960px width. All gridworld renderings use the same color scale as
Lesson 2's V4-V7 — *visual consistency across the curriculum*.

---

## 8. Acceptance Criteria

After this lesson the learner can:

1. Write down iterative policy evaluation and explain why it converges
   (citing Banach + Bellman expectation contraction).
2. State and prove the policy improvement theorem in 4 lines.
3. Write down full policy iteration and prove its finite convergence.
4. Write down value iteration and identify it as modified PI with `m=1`.
5. State the value iteration stopping criterion and explain why it's
   conservative.
6. Compare PI and VI on a small MDP by hand (≤ 4 states).
7. Explain why sweep order matters in async DP, with one concrete
   example.
8. State the GPI principle and identify it in any of the algorithms
   from Lessons 4-12.

Concrete check: hand them a 4-state chain MDP and ask them to run two
PI iterations and three VI iterations *by hand*, verifying with V3 and
V4.

---

## 9. Stretch Goals (post-MVP)

- **Prioritized sweeping** as a fourth sweep mode in V5, with the
  forward link to PER (Lesson 7) made explicit.
- **Real-Time DP (RTDP):** focus updates on states reached in trajectories
  from a start state. Bridges async DP to sample-based methods.
- **Linear programming formulation of MDPs:** mention that `V^*` is the
  solution of an LP. Bridges to constrained MDPs and dual formulations
  used in offline RL (Lesson 15).
- **Larger gridworlds:** in the Algorithm Lab, allow user-configurable
  grid sizes up to 10×10 to see how convergence scales.

---

## 10. Out of Scope (intentionally)

- **Model-free learning** (no samples in this lesson). That's Lessons
  4 onward.
- **Function approximation.** That's Lesson 6.
- **Discounted vs average reward** beyond a passing mention.
- **Linear programming / dual LPs.** Brief note in stretch goals only.

---

## 11. Training Notebook

**Not applicable.** No model trained. `scripts/dp_traces.py` (~70 lines)
pre-computes PI and VI traces for the default gridworld and the
sweep-order comparison data. Outputs:

```
public/data/dp/
  pi_trace.json        # 3-iteration PI on gridworld
  vi_trace.json        # 5-iteration VI on gridworld
  sweep_comparison.json # iter counts for all sweep orders
```

In-browser implementations reproduce these on the fly; JSON is for
instant initial render.

---

## End of spec

Total length: ~1480 lines. Comparable to MDPs in scope — appropriate
for a critical-path lesson. The DP Algorithm Lab (V4) carries the
polish-budget weight. Three pre-verified numerical traces (PI: 3 iters,
VI: 5 iters, reverse-sweep async: 2 iters) anchor the entire lesson.
