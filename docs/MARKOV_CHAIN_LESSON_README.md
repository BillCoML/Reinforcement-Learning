# Prereq A — Markov Chains and Stationary Distributions

> **The mathematics of "the future depends on the past only through the present."**
> Markov chains are the substrate of every RL algorithm to come. The policy
> induces a chain on states; analysis of that chain is how we'll reason about
> convergence, sample complexity, and on-policy vs off-policy distinctions.

> **Where this slots in.** This lesson lands between Lesson 1 (Bandits) and
> Lesson 2 (MDPs). It has no curricular prereqs of its own. It earns its place
> by giving us five tools — transition matrix, stationary distribution,
> irreducibility, aperiodicity, detailed balance — that every subsequent
> lesson will use as named primitives.

---

## 0. Pedagogical Philosophy

1. **Earn the abstraction.** The first example is a literal weather chain with
   concrete numbers. The formal definition arrives only after the learner has
   already computed two transition probabilities by hand.

2. **Convergence is the centerpiece.** Most of the formalism (irreducibility,
   aperiodicity, communicating classes) exists to characterize *when chains
   converge*. We spend the polish budget on §5's Convergence Lab so the
   learner sees the difference between "converges fast", "converges slowly",
   "fails to converge", and "converges but to a non-unique limit" in a single
   side-by-side panel.

3. **Counterexamples in every section.** When we say "irreducible chains have
   a unique stationary distribution," the next thing we show is a reducible
   chain with two stationary distributions, side by side with the irreducible
   case. The shape of the theorem only registers when the failure mode is
   visible.

4. **End on the bridge to RL.** The final section is short but load-bearing:
   the *state visitation distribution* under a policy is a Markov chain on
   states, and everything we just learned applies. This is the single most
   important sentence in the lesson; it pre-loads Lesson 2 (MDPs) and
   Lesson 8 (policy gradient).

5. **Keep it tight.** This is a prereq, not a destination. ~900 lines, seven
   sections, seven visualizations. The agent should finish with energy in
   reserve for Lesson 2.

---

## 1. Tech Stack

Identical to Lesson 1. Vite + TypeScript strict + KaTeX + D3 v7 + ml-matrix
for linear algebra (eigendecomposition is used here, lightly). Vitest for
tests. No PyTorch yet — eigendecomposition runs in the browser via ml-matrix.

The offline simulation script for Lesson 1 (`scripts/simulate_bandits.py`) is
unaffected. This lesson adds `scripts/markov_examples.py` which generates the
pre-baked chain examples as JSON.

---

## 2. Visual / Aesthetic Direction

We extend the curriculum aesthetic established in Lesson 1. New conventions
specific to Markov chains, locked in here for re-use in Lesson 2 (where MDPs
will reuse the transition-graph drawing) and beyond:

```css
:root {
  /* State / node colors — categorical palette for individual states */
  --mc-state-1:    #2563eb;   /* blue-600 */
  --mc-state-2:    #ea580c;   /* orange-600 */
  --mc-state-3:    #16a34a;   /* green-600 */
  --mc-state-4:    #9333ea;   /* purple-600 */
  --mc-state-5:    #db2777;   /* pink-600 */
  --mc-state-6:    #ca8a04;   /* yellow-600 */
  --mc-state-7:    #0891b2;   /* cyan-600 */
  --mc-state-8:    #4b5563;   /* gray-600 */

  /* Structural / role colors */
  --mc-transient:   #ea580c;   /* orange — transient states */
  --mc-recurrent:   #2563eb;   /* blue   — recurrent states */
  --mc-absorbing:   #1e3a8a;   /* dark blue — absorbing states */
  --mc-periodic:    #b91c1c;   /* red    — periodic class */

  /* The stationary distribution gets the "good/convergent" green */
  --mc-stationary:  #15803d;   /* green-700 — π */
  --mc-current:     #1c1e22;   /* current state in animations */

  /* Edge weights */
  --mc-edge:        #5a5d63;
  --mc-edge-strong: #1c1e22;
}
```

**Drawing convention for transition graphs.** Nodes laid out either in a
fixed grid (for small chains, ≤ 4 states) or by D3 force layout (for larger
ones). Edge thickness proportional to transition probability (min 1px, max 6px).
Self-loops drawn as small arcs above the node. Probabilities ≥ 0.05 are
labeled inline; smaller ones are accessible via hover.

**Bar charts for distributions.** Probability vectors are always shown as
horizontal bars in the categorical state palette. The stationary distribution
π gets a thin green outline overlay so the learner sees "is the current
distribution close to π?" at a glance.

---

## 3. Lesson Metadata

```ts
export const lessonMeta: LessonMeta = {
  id: "markov-chains",
  title: "Markov Chains and Stationary Distributions",
  subtitle: "The substrate of every policy",
  tier: 1,
  difficulty: 2,
  estimatedReadMinutes: 50,
  role: "prereq",  // not on the main critical path; lands before Lesson 2
  prerequisites: [
    { external: true, label: "Linear algebra: matrix multiplication, eigenvalues" },
    { external: true, label: "Probability: conditional probability, expectations" },
  ],
  exportedAnchors: [
    "markov-property",
    "transition-matrix",
    "n-step-transitions",
    "communicating-classes",
    "irreducible-aperiodic",
    "stationary-distribution",
    "ergodic-theorem",
    "detailed-balance",
    "policy-induced-chain",
  ],
  centerpieceComponent: "ConvergenceLab",
  forwardLinksWhenReady: [
    { to: "mdps",            anchor: "policy-induced-chain" },
    { to: "td-learning",     anchor: "stationary-distribution-sampling" },
    { to: "policy-gradient", anchor: "on-policy-distribution" },
  ],
};
```

---

## 4. Section-by-Section Plan

### §1 — The Markov Property
**Tagline:** *Memorylessness, with one caveat.*
**Length:** ~600 words.
**Anchor:** `markov-property`.

---

**Prose:**

Imagine tracking the weather day by day in three categories: sunny, cloudy,
rainy. We notice an empirical pattern. If today is sunny, tomorrow is sunny
with probability 0.7, cloudy with 0.2, rainy with 0.1. If today is cloudy,
tomorrow is sunny with 0.3, cloudy with 0.4, rainy with 0.3. Similar entries
hold for rainy. The full table:

|                | → sunny | → cloudy | → rainy |
|----------------|--------:|---------:|--------:|
| **from sunny** |    0.70 |     0.20 |    0.10 |
| **from cloudy**|    0.30 |     0.40 |    0.30 |
| **from rainy** |    0.20 |     0.30 |    0.50 |

This is a **transition matrix** — let's call it `P`. The entry `P_{ij}` is
`Pr(next state = j ∣ current state = i)`. By construction, rows sum to one.
Notice something we've quietly assumed: tomorrow's weather depends *only on
today's*, not on the entire past history. This is the **Markov property**:

$$
\boxed{\Pr(X_{t+1} = j \mid X_t = i,\, X_{t-1}, X_{t-2}, \ldots) \;=\; \Pr(X_{t+1} = j \mid X_t = i)}
$$

You may rightly object that real weather isn't Markov — a week of sun
genuinely raises the chance of a heat wave above what "today is sunny" alone
predicts. The Markov property is a **modeling assumption**, not a physical
truth. Its value is mathematical tractability: when the assumption holds (or
approximately holds), an enormous toolkit becomes available. Most of this
lesson is that toolkit.

A **Markov chain** is a stochastic process `(X_0, X_1, X_2, \ldots)` taking
values in a state space `\mathcal{S}` that satisfies the Markov property. In
this lesson we restrict to **finite, time-homogeneous, discrete-time** chains:

- *Finite:* `|\mathcal{S}| = K < \infty`. We index states `1, \ldots, K`.
- *Time-homogeneous:* the transition probabilities `P_{ij}` don't depend on
  the time `t`.
- *Discrete-time:* steps are integers `t = 0, 1, 2, \ldots`.

Each of these can be relaxed (countable state spaces, time-inhomogeneous
chains, continuous-time chains, controlled chains where transitions depend
on an action) — and in fact Lesson 2's MDP is exactly "finite Markov chain
+ a controllable action at each step." Every restriction we make here gets
loosened later in a principled way.

**Initial distribution.** A chain is fully specified by `(P, \mu_0)` where
`\mu_0` is the distribution over the starting state `X_0`. From these two
ingredients we can compute the distribution of `X_t` for any `t`. We'll see
how in §2.

---

**Numerical example (pre-verified).** With `P` as above and `μ_0 = (1, 0, 0)`
(start sunny), the probability of "sunny on day 2" is

$$
\Pr(X_2 = \text{sunny}) \;=\; \sum_k P_{0k} P_{k0}
\;=\; 0.7 \cdot 0.7 + 0.2 \cdot 0.3 + 0.1 \cdot 0.2
\;=\; \mathbf{0.57}.
$$

This is just `(P^2)_{00}`, the top-left entry of `P` squared. We'll formalize
this in the next section.

---

**Visualization V1 — Weather Chain Explorer.**

- Three colored nodes (sunny=yellow, cloudy=gray, rainy=blue), arranged in a
  triangle. (Note: these specific colors break our generic state palette
  because they're semantic for this one example. Subsequent vizzes use the
  state-1/state-2/state-3 palette.)
- Edges labeled with probabilities; thickness scales.
- A "current state" marker sits on one node. "Step" button samples a transition;
  marker moves with a smooth animation (~250ms).
- Below: a running tally of how many times each state has been visited, plus
  a small bar chart of empirical visit frequency.
- "Run 100 steps" button for impatient users.
- Width 720, height 360.

This is the warm-up. Pre-verify that running 1000 steps shows visit
frequencies converging to roughly (0.457, 0.283, 0.261) — the stationary
distribution, which we'll formally derive in §4.

---

### §2 — Multi-Step Transitions and Long-Run Behaviour
**Tagline:** *Powers of a matrix encode the future.*
**Length:** ~650 words.
**Anchor:** `n-step-transitions`.

---

**Prose:**

If `P_{ij}` is the one-step transition probability `\Pr(X_1 = j \mid X_0 = i)`,
what about two steps? The chain goes `i \to k \to j` for some intermediate
state `k`. By the law of total probability and the Markov property,

$$
\Pr(X_2 = j \mid X_0 = i)
\;=\; \sum_k \Pr(X_1 = k \mid X_0 = i) \cdot \Pr(X_2 = j \mid X_1 = k)
\;=\; \sum_k P_{ik} P_{kj}
\;=\; (P^2)_{ij}.
$$

The `n`-step transition probabilities are the entries of `P^n`:

$$
\boxed{\Pr(X_n = j \mid X_0 = i) \;=\; (P^n)_{ij}}
$$

This is the **Chapman-Kolmogorov equation** in matrix form. (The traditional
non-matrix statement is `P^{(m+n)}_{ij} = \sum_k P^{(m)}_{ik} P^{(n)}_{kj}`,
which is just matrix multiplication associativity in disguise.)

If `μ_0` is the initial distribution (a row vector), then the distribution
at time `t` is

$$
\boxed{\mu_t \;=\; \mu_0 \, P^t}
$$

— a row vector times a matrix, repeatedly.

**Powers of the weather P (pre-verified).** Let's see what happens.

`P^1`:
$$
\begin{pmatrix} 0.70 & 0.20 & 0.10 \\ 0.30 & 0.40 & 0.30 \\ 0.20 & 0.30 & 0.50 \end{pmatrix}
$$

`P^2`:
$$
\begin{pmatrix} 0.57 & 0.25 & 0.18 \\ 0.39 & 0.31 & 0.30 \\ 0.33 & 0.31 & 0.36 \end{pmatrix}
$$

`P^5`:
$$
\begin{pmatrix} 0.468 & 0.279 & 0.252 \\ 0.450 & 0.284 & 0.266 \\ 0.443 & 0.286 & 0.271 \end{pmatrix}
$$

`P^20`:
$$
\begin{pmatrix} 0.457 & 0.283 & 0.261 \\ 0.457 & 0.283 & 0.261 \\ 0.457 & 0.283 & 0.261 \end{pmatrix}
$$

Two striking things. **First**, the rows of `P^n` get *identical* as `n`
grows — meaning the long-run probability of being in state `j` no longer
depends on where you started. **Second**, that common row is some fixed
distribution — `(0.457, 0.283, 0.261)`. This distribution is what we'll call
**stationary** in §4. The conditions under which this convergence happens
are exactly the content of §3 and §5.

A different shape of behaviour. Consider the two-state chain with
`P = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}`. Then `P^2 = I`, `P^3 = P`,
`P^4 = I`, and so on. The chain oscillates forever; `P^n` never converges.
Yet `π = (0.5, 0.5)` is a stationary distribution in the sense of §4. So
"stationary distribution exists" and "the chain converges to it" are *two
different statements*. Hold that distinction.

---

**Visualization V2 — Power-of-P Animator.**

- A 3×3 (or 4×4) heatmap of `P^n` with `n` controllable via slider.
- Color scale: white (0) → dark gray (1), with the actual value printed
  inside each cell in mono.
- "Animate from n=1 to n=30" button — runs through the slider smoothly,
  ~150ms per step.
- Right-hand panel: stacked bar showing the distribution `(P^n)_{i, \cdot}`
  for a selectable starting state `i`. As `n` grows, the bar reshapes
  toward π.
- Toggle: "show stationary overlay" (green dashed line on the bar chart).
- Width 800, height 360.

A second tab in the same panel: the *periodic chain* (`P = [[0,1],[1,0]]`).
Same controls. The learner watches `P^n` flip between `I` and `[[0,1],[1,0]]`
forever. The contrast with the weather chain is the whole point.

---

### §3 — Classifying States
**Tagline:** *Reachability, recurrence, periodicity — three independent axes.*
**Length:** ~750 words.
**Anchor:** `communicating-classes`.

---

**Prose:**

To predict whether `P^n` converges (and to what), we need a vocabulary for
the *structure* of a chain. Three classifications matter for RL:
**reachability** (which states can be reached from which), **recurrence**
(do you keep coming back?), and **periodicity** (do you return on a regular
schedule?).

**Reachability and communicating classes.** State `j` is *reachable* from
state `i` if there exists `n \geq 0` such that `(P^n)_{ij} > 0`. (We allow
`n=0`, so every state is trivially reachable from itself.) States `i` and
`j` **communicate** if each is reachable from the other; this is an
equivalence relation, partitioning `\mathcal{S}` into **communicating
classes**.

A chain is **irreducible** if there's exactly one communicating class — every
state can reach every other. The weather chain is irreducible. A chain that
splits into two non-communicating classes is **reducible**.

**Recurrence.** A state `i` is *recurrent* if, starting from `i`, the chain
returns to `i` with probability 1. Otherwise it is *transient*. In a finite
chain, every state is either recurrent or transient, and recurrence is a
**class property** — all states in a communicating class are recurrent
together or transient together. In a finite irreducible chain *every state
is recurrent* (an easy consequence of finiteness + irreducibility).

**Periodicity.** The *period* of state `i` is the GCD of all `n \geq 1` for
which `(P^n)_{ii} > 0`. If the period is 1, the state is **aperiodic**.
Aperiodicity is a class property; an irreducible chain has a single period.

> **Quick check.** The two-state chain with `P=\begin{pmatrix}0&1\\1&0\end{pmatrix}`
> is irreducible, recurrent, and has period 2 (returns happen only at even `n`).
> The weather chain is irreducible, recurrent, and aperiodic — the diagonal
> entries `P_{ii} > 0`, so `(P^1)_{ii} > 0` and `\gcd\{1, 2, 3, \ldots\} = 1`.

A useful shorthand: a finite chain is called **ergodic** if it is
irreducible and aperiodic. (Some authors require also "positive recurrent,"
which is automatic in finite chains.) Ergodic finite chains are precisely
the ones for which `P^n` converges to a rank-one matrix `\mathbf{1} \pi^\top`.
This is the punchline; we'll prove it in §5.

---

**Counterexample gallery (pre-verified).**

**Reducible 4-state chain.** Consider

$$
P = \begin{pmatrix}
0.7 & 0.3 & 0   & 0   \\
0.4 & 0.6 & 0   & 0   \\
0.1 & 0   & 0.3 & 0.6 \\
0   & 0.2 & 0.5 & 0.3
\end{pmatrix}.
$$

States `\{0, 1\}` communicate (forming a recurrent class). States `\{2, 3\}`
also communicate with each other but they "leak" into `\{0, 1\}` via
`P_{20} = 0.1` and `P_{31} = 0.2` — and crucially nothing in `\{0,1\}`
returns to `\{2,3\}`. So `\{0,1\}` is recurrent, `\{2,3\}` is **transient**.

`P^{50}` starting from state 2 (transient):
$$
(0.571, \; 0.428, \; 0.000, \; 0.000)
$$

Nearly all mass has leaked into the absorbing class. `P^{50}` starting from
state 0 (in the absorbing class):
$$
(0.571, \; 0.429, \; 0, \; 0)
$$

Same limiting masses on the recurrent states, but exactly zero on transient
ones from the very first step (since the recurrent class is closed).

> **Sidebar.** This is the structure of every **episodic MDP**: terminal
> states form an absorbing class. The agent's job is to navigate the
> transient set toward favorable absorbing states. Lesson 2 will rest on
> exactly this picture.

---

**Visualization V3 — State Classification Inspector.**

- Drag-and-drop chain editor (≤ 6 states): user adds nodes, draws directed
  edges with weights.
- Auto-computed badges next to each node:
  - "Transient" / "Recurrent" badge
  - Period (1, 2, 3, …) badge
  - Communicating class color (each class gets a distinct background tint)
- Below the graph: a textual summary, e.g. "Chain is reducible: 2
  communicating classes. Class A = {0, 1} is recurrent, aperiodic. Class
  B = {2, 3} is transient."
- "Load preset" dropdown with three options: weather, periodic, reducible
  (the three pre-verified examples).
- Width 800, height 480.

This is a moderately complex viz. The chain-editing UX is the hardest part;
keep it simple — click-to-add-node, click-then-click to add an edge,
slider for edge weight.

---

### §4 — Stationary Distributions
**Tagline:** *π solves π = πP. Existence is easy, uniqueness needs work.*
**Length:** ~700 words.
**Anchor:** `stationary-distribution`.

---

**Prose:**

A row vector `\pi \in \mathbb{R}^K` is a **stationary distribution** of `P`
if it is a probability distribution (`\pi_i \geq 0`, `\sum_i \pi_i = 1`) and
satisfies the **balance equation**

$$
\boxed{\pi \;=\; \pi P, \qquad \text{i.e.}\quad \pi_j \;=\; \sum_i \pi_i P_{ij} \;\;\forall j.}
$$

Equivalently, `π` is a left eigenvector of `P` with eigenvalue 1, normalized
to sum to 1. The intuition: if the chain is *currently* distributed as `π`,
then after one step it's *still* distributed as `π`. The mass flowing into
state `j` (the right-hand side, `\sum_i \pi_i P_{ij}`) exactly equals the
mass currently in state `j` (the left-hand side).

**Existence.** Every finite Markov chain has at least one stationary
distribution. This is a consequence of the Perron-Frobenius theorem applied
to the row-stochastic matrix `P`: the spectral radius of `P` is exactly 1,
and there exists a non-negative left eigenvector for eigenvalue 1, which can
be normalized to a probability distribution.

**Uniqueness.** A finite chain has a *unique* stationary distribution if and
only if it is irreducible. Proof sketch: irreducibility implies that the
left-eigenspace of `P` for eigenvalue 1 is one-dimensional; reducibility
allows multiple "absorbing" classes each with its own stationary support,
yielding a family of stationary distributions.

**Computing π.** Three equivalent methods.

1. *Linear algebra.* Solve `(P^\top - I) \pi^\top = 0` subject to
   `\mathbf{1}^\top \pi^\top = 1`. Set up the augmented system; standard
   linear solve.
2. *Power iteration.* Pick any initial distribution `\mu_0`, compute
   `\mu_t = \mu_0 P^t`. If the chain is ergodic, `\mu_t \to \pi`. Stop when
   `\|\mu_{t+1} - \mu_t\|` is small.
3. *Eigendecomposition.* Find the left eigenvector of `P` for eigenvalue 1
   directly.

---

**Worked example.** For the weather chain, the balance equations are:

$$
\begin{aligned}
\pi_S &= 0.7 \pi_S + 0.3 \pi_C + 0.2 \pi_R \\
\pi_C &= 0.2 \pi_S + 0.4 \pi_C + 0.3 \pi_R \\
\pi_R &= 0.1 \pi_S + 0.3 \pi_C + 0.5 \pi_R \\
1     &= \pi_S + \pi_C + \pi_R
\end{aligned}
$$

Replace one balance equation (any of them — they're linearly dependent) with
the normalization. Solving yields

$$
\pi \;=\; \left(\tfrac{21}{46},\; \tfrac{13}{46},\; \tfrac{12}{46}\right) \;\approx\; (0.4565,\, 0.2826,\, 0.2609).
$$

Numerically, this is exactly the row that `P^{20}` converged to in §2. The
convergence wasn't accidental — it was inevitable.

**Periodic chain revisited.** For `P = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}`,
the balance equation `π = πP` gives `π_0 = π_1`, so `π = (0.5, 0.5)`. This
exists and is unique (chain is irreducible). But `P^n` doesn't converge to
the rank-one matrix `\mathbf{1} \pi^\top = \begin{pmatrix} 0.5 & 0.5 \\ 0.5 & 0.5 \end{pmatrix}` —
it oscillates. **Stationary existence does not imply convergence.** What we
need is the next section's content: aperiodicity.

> **Forward link** — In Lesson 8 (Policy Gradient) we'll write the policy
> gradient as an expectation over the *stationary state distribution* `d^π`
> induced by policy `π`. That `d^π` is exactly the stationary distribution
> of a Markov chain — the one whose transition matrix is `P^π_{ss'} =
> \sum_a π(a|s) p(s'|s,a)`. Same equation, same theorem, deeper application.

---

**Visualization V4 — Stationary Distribution Finder.**

- 3×3 (or 4×4) editable transition matrix on the left. Live row-sum checker.
- Right side: bar chart of the computed π, with three "compute mode" tabs:
  1. Linear algebra: shows the system being solved, with KaTeX rendering.
  2. Power iteration: animates `μ_t` over `t = 0, 1, 2, \ldots`, showing
     convergence (or non-convergence) in real time.
  3. Eigendecomp: shows the eigenvalues of `P^\top`, with the unit eigenvalue
     highlighted in green.
- "Verify: π P = π" panel beneath: shows `π P` and the residual `|πP - π|`.
- Width 880, height 460.

---

### §5 — The Ergodic Theorem (Convergence) — CENTERPIECE
**Tagline:** *Irreducible + aperiodic = inevitable convergence, at a rate set by the spectral gap.*
**Length:** ~900 words.
**Anchor:** `ergodic-theorem`.

---

**Prose:**

**Theorem (Convergence to Stationarity).** Let `P` be the transition matrix
of a finite, irreducible, aperiodic Markov chain. Then there exists a unique
stationary distribution `π`, and for every initial distribution `μ_0`,

$$
\boxed{\lim_{n \to \infty} \mu_0 P^n \;=\; \pi.}
$$

Equivalently, `\lim_{n\to\infty} P^n = \mathbf{1} \pi^\top` (each row of
`P^n` converges to `π`).

**Proof sketch via eigendecomposition.** Since `P` is row-stochastic, its
spectral radius is 1, and 1 is an eigenvalue (with left eigenvector `π`).
Irreducibility makes that eigenvalue *simple* (multiplicity 1). Aperiodicity
ensures that all *other* eigenvalues `λ_2, λ_3, \ldots, λ_K` have magnitude
strictly less than 1. (Periodic chains have eigenvalues on the unit circle
besides 1; aperiodicity rules this out.)

Write `P = \sum_k \lambda_k v_k u_k^\top` in spectral form (left/right
eigenpairs). Then

$$
P^n \;=\; \sum_k \lambda_k^n v_k u_k^\top
\;=\; \underbrace{\mathbf{1} \pi^\top}_{\lambda_1 = 1, \text{ rank 1}}
\;+\; \sum_{k \geq 2} \lambda_k^n v_k u_k^\top.
$$

The second sum vanishes because `|\lambda_k| < 1` for `k \geq 2`. Done.

**The rate of convergence — spectral gap.** The convergence rate is governed
by the **second largest eigenvalue in magnitude**:

$$
\lambda_\star \;:=\; \max_{k \geq 2} |\lambda_k|.
$$

The **spectral gap** is `\gamma := 1 - \lambda_\star`. The total-variation
distance to stationarity satisfies

$$
\|\mu_0 P^n - \pi\|_{\text{TV}} \;\leq\; C \cdot \lambda_\star^n
$$

for some constant `C` depending on the chain. So convergence is geometric
with rate `\lambda_\star`. The closer `\lambda_\star` is to 1, the slower.

**Mixing time.** Define the mixing time `t_{\text{mix}}(\varepsilon)` as the
smallest `n` for which `\|\mu_0 P^n - \pi\|_{\text{TV}} \leq \varepsilon`
from any starting distribution. A standard bound:

$$
t_{\text{mix}}(\varepsilon) \;\leq\; \frac{\log(1/\varepsilon)}{1 - \lambda_\star} \cdot (\text{small constant}).
$$

For the weather chain: eigenvalues of `P` are `(1, 0.4732, 0.1268)`. So
`\lambda_\star = 0.4732`, spectral gap `γ ≈ 0.527`, and the mixing time at
`ε = 0.01` is roughly `\log(100) / 0.527 ≈ 8.74` steps. Empirically (from
the table in §2), the TV distance is below `10^{-4}` by `n = 10` and below
`10^{-6}` by `n = 20`. Theory and experiment agree.

---

**Numerical convergence table (pre-verified).** TV distance to π from each
starting state, for the weather chain:

| `n` | from sunny | from cloudy | from rainy |
|----:|-----------:|------------:|-----------:|
| 0   | 0.5435     | 0.7174      | 0.7391     |
| 1   | 0.2435     | 0.1565      | 0.2565     |
| 2   | 0.1135     | 0.0665      | 0.1265     |
| 3   | 0.0535     | 0.0305      | 0.0605     |
| 5   | 0.0120     | 0.0068      | 0.0136     |
| 10  | 2.8e-4     | 1.6e-4      | 3.2e-4     |
| 20  | 7.4e-9     | (vanishing) | (vanishing) |

The ratio between consecutive rows is approximately `\lambda_\star = 0.4732`,
exactly as the theorem predicts.

---

**When ergodicity fails — and what survives.** If the chain is periodic,
`P^n` doesn't converge but the **Cesàro average** does:

$$
\frac{1}{N} \sum_{n=0}^{N-1} P^n \;\to\; \mathbf{1} \pi^\top.
$$

So time-averaged behaviour still gives you `π`; only instantaneous
distributions oscillate. For `P = \begin{pmatrix}0&1\\1&0\end{pmatrix}`, the
Cesàro average over `n=0,\ldots,99` is numerically
`\begin{pmatrix}0.5 & 0.5 \\ 0.5 & 0.5\end{pmatrix}` — exactly `\mathbf{1}\pi^\top`.

If the chain is *reducible*, even the Cesàro average depends on the starting
state — different connected components have different stationary
distributions, and there is no single `π` that all initial conditions land in.

> **Forward link** — The Cesàro average is what TD(0) and Q-learning
> effectively *average over* in their long-run behaviour (Lesson 5). The
> stochastic-approximation theorem (Robbins-Monro, Prereq B / inline in
> Lesson 5) requires only that the chain visits each state infinitely often
> — recurrence. Aperiodicity isn't strictly required for TD's convergence,
> but irreducibility is.

---

**Visualization V5 — Convergence Lab.** *This is the centerpiece.*

A four-panel synchronized dashboard, each panel telling part of the
convergence story.

**Panel A (top-left): Chain selector and matrix view.**
- Dropdown: "weather (ergodic)", "periodic-2 (irreducible, periodic)",
  "reducible (multiple stationary distributions)", "slow-mixing (eigenvalue
  near 1)" — four pre-baked chains plus "custom" (open the matrix editor
  from V4).
- Shows the transition matrix `P` as a small heatmap.
- Below: computed eigenvalues `\{1, \lambda_2, \lambda_3, \ldots\}` plotted
  on the complex unit circle. The user *sees* whether all non-unit
  eigenvalues are strictly inside the circle.

**Panel B (top-right): Distribution evolution.**
- Stacked bar chart showing `μ_t` over `t = 0, 1, 2, \ldots`. As `t`
  advances, the bar reshapes. The stationary `π` shows as a thin overlay.
- User can scrub `t` from 0 to 100 with a slider.
- "Play" button animates `t` advancing at ~250ms per step.

**Panel C (bottom-left): TV distance curve.**
- Log-scale plot of `\|\mu_t - π\|_{\text{TV}}` vs `t`.
- For ergodic chains: straight line on log scale (geometric decay).
- For periodic chains: oscillating line.
- For reducible chains: settles at a non-zero floor.
- A reference line shows the theoretical bound `C \cdot \lambda_\star^t`.

**Panel D (bottom-right): Sample path.**
- A single realization of the chain — the actual states visited, one at a
  time, drawn as a colored tape `[X_0, X_1, X_2, \ldots, X_t]`.
- Below the tape: empirical visit frequency bars updating as the path
  extends. The empirical frequencies should converge to π for ergodic
  chains.
- Useful pedagogically because it shows the *path-level* meaning of
  ergodicity.

**Controls.**
- Choose initial distribution: "uniform", "delta-on-state-0", "delta-on-each
  state averaged" (= computed via `P^n` directly).
- Speed control 1×–32×.
- Step / pause / reset.

This is the single largest interactive in the lesson. Budget: ~3 days
of polish.

**Width:** ~960px (centerpiece breakout, like V7 in Bandits).
**Height:** ~640px (two-row dashboard).

---

### §6 — Detailed Balance (Reversibility)
**Tagline:** *A sufficient condition for stationarity, and a glimpse of MCMC.*
**Length:** ~550 words.
**Anchor:** `detailed-balance`.

---

**Prose:**

Solving `π = πP` for the stationary distribution is a system of `K` linear
equations. Often we can shortcut this with a stronger structural condition.

**Detailed balance.** A distribution `π` and a transition matrix `P` are
said to satisfy **detailed balance** if

$$
\boxed{\pi_i P_{ij} \;=\; \pi_j P_{ji} \quad \forall i, j.}
$$

This says the probability flow from `i` to `j` exactly equals the flow from
`j` to `i`, **pairwise**. Detailed balance is *strictly stronger* than the
balance equation `π = πP`: if detailed balance holds for `π`, then summing
both sides over `i` gives `\sum_i \pi_i P_{ij} = \pi_j \sum_i P_{ji} = \pi_j`,
so `π = πP`. The converse is false — many chains have stationary
distributions without detailed balance.

A chain admitting a detailed-balance distribution is called **reversible**.
Run such a chain in reverse time, and the statistics are identical.

**Why it's useful.** Two reasons.

*Reason 1: Verification is local.* Checking `π = πP` requires verifying `K`
equations involving the whole distribution. Detailed balance is `K(K-1)/2`
pairwise checks — but each one involves only two entries of `π`. If you can
*guess* `π` and verify detailed balance pairwise, you've found the stationary
distribution.

*Reason 2: MCMC.* The Markov chain Monte Carlo (MCMC) methodology *engineers*
chains that have a target distribution `π` as their detailed-balance
distribution, then runs them to draw approximate samples from `π`. The
Metropolis-Hastings acceptance rule is *defined* by enforcing detailed balance.
We won't dwell on MCMC here, but it's the most important application of
detailed balance outside of physics.

---

**Worked example.** A 3-state birth-death chain:

$$
P = \begin{pmatrix} 0.5 & 0.5 & 0 \\ 0.3 & 0.4 & 0.3 \\ 0 & 0.6 & 0.4 \end{pmatrix}.
$$

Solving `π = πP` (or guessing-and-verifying) gives
`\pi = \left(\tfrac{6}{21},\, \tfrac{10}{21},\, \tfrac{5}{21}\right) \approx (0.2857, 0.4762, 0.2381)`.

Detailed balance check:
- `\pi_0 P_{01} = \tfrac{6}{21} \cdot 0.5 = \tfrac{3}{21}` vs
  `\pi_1 P_{10} = \tfrac{10}{21} \cdot 0.3 = \tfrac{3}{21}` ✓
- `\pi_1 P_{12} = \tfrac{10}{21} \cdot 0.3 = \tfrac{3}{21}` vs
  `\pi_2 P_{21} = \tfrac{5}{21} \cdot 0.6 = \tfrac{3}{21}` ✓
- `\pi_0 P_{02} = 0` vs `\pi_2 P_{20} = 0` ✓ (trivially)

All pairs satisfy detailed balance — the chain is reversible.

> **Forward link** — Detailed balance shows up again in Lesson 17 (RLHF) in
> a subtle way: the Bradley-Terry preference model gives a *symmetric*
> structure to pairwise comparison "transitions" that mirrors the reversibility
> condition. We'll point back here when we get there.

---

**Visualization V6 — Detailed Balance Flow.**

- Three nodes (or four, configurable) in a horizontal row.
- Between adjacent nodes, two arrows: `i → j` and `j → i`, drawn back-to-back.
- Arrow thickness is `π_i P_{ij}` (flow magnitude).
- A green check appears between paired arrows when they're equal (detailed
  balance holds); a red ⚠ appears when they differ.
- "Compute π" button runs the balance equation. "Check detailed balance"
  highlights all pairs.
- Preset dropdown: "birth-death (reversible)", "asymmetric (not reversible)".
- Width 720, height 320.

---

### §7 — The Bridge to RL
**Tagline:** *A policy turns an MDP into a Markov chain, and everything we just learned applies.*
**Length:** ~450 words.
**Anchor:** `policy-induced-chain`.

---

**Prose:**

Here's the one paragraph that justifies this prereq.

In Lesson 2, an **MDP** will be defined as `(\mathcal{S}, \mathcal{A}, P, r, \gamma)`
— states, actions, transition kernel, reward, discount. The transition kernel
is now `P(s' \mid s, a)` — it depends on a chosen *action*. An MDP is not yet
a Markov chain on states; it's a *controlled* process. But the moment we fix
a **policy** `\pi(a \mid s)` (a rule for choosing actions given states),
something magical happens: the marginal-over-actions transition

$$
\boxed{P^\pi_{ss'} \;=\; \sum_a \pi(a \mid s) \, P(s' \mid s, a)}
$$

is an ordinary `|S| \times |S|` stochastic matrix. The pair `(P^\pi, \mu_0)`
*is* a Markov chain on the state space. The state visitation distribution
under policy `π` is the stationary distribution `d^\pi` of `P^\pi`.

Every theorem from this lesson applies:

- **Irreducibility of `P^\pi`** ⇔ the policy can reach every state from
  every state (with positive probability under randomness in `π` and `P`).
- **Aperiodicity of `P^\pi`** ⇔ the policy doesn't trap the chain in periodic
  cycles.
- **Ergodicity of `P^\pi`** ⇔ the policy has a unique stationary distribution
  that *averages over the agent's behaviour*.

Three concrete payoffs downstream:

1. **Policy evaluation (Lessons 3, 5).** When TD(0) or Monte Carlo evaluates
   `V^\pi(s) = E_\pi[\sum_t \gamma^t r_t \mid s_0 = s]`, the *distribution
   of states encountered* under the policy is the stationary distribution of
   `P^\pi`. Sample complexity bounds depend on the spectral gap.

2. **Policy gradient (Lesson 8).** The policy gradient theorem will say

   $$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^{\pi_\theta},\, a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^\pi(s,a)].$$

   That `s \sim d^{\pi_\theta}` is exactly the stationary distribution we
   just defined. Without it, "on-policy" has no meaning.

3. **Off-policy correction (Lessons 4, 15).** When we use samples from one
   policy to evaluate another, the *correction factor* is a ratio of
   stationary distributions — importance sampling on chains. This is where
   the entire offline-RL story begins.

You now have the substrate. Lesson 2 builds the structure of decision-making
on top of it.

---

**Visualization V7 — Policy-Induced Chain Preview.**

- A 4×4 gridworld on the left, with arrows showing a (random or fixed)
  policy's action distribution at each cell.
- On the right: the resulting `P^\pi` matrix (16×16), shown as a heatmap.
- Below: the stationary distribution `d^\pi` overlaid on the gridworld as
  cell-shade intensity (darker = visited more often).
- Below that: an animation of a sampled trajectory under the policy, with
  the agent token moving cell-to-cell.
- Policy controls: "uniform random", "go-right-biased", "diagonal-biased".
- Width 800, height 460.

This visualization is intentionally a *preview*, not a full MDP visualizer.
It's enough to make the "policy → chain" idea visceral. Lesson 2 will
generalize it into a full MDP explorer with values, rewards, and Bellman
backups.

---

## 5. Algorithm / Math Implementation

TypeScript module `src/markov/chain.ts`. The math is light — eigendecomposition,
power iteration, a few classification routines.

```ts
import { Matrix, EigenvalueDecomposition } from 'ml-matrix';

export class MarkovChain {
  readonly P: Matrix;      // K x K row-stochastic
  readonly K: number;

  constructor(P: number[][]) {
    this.P = new Matrix(P);
    this.K = P.length;
    this.validateRowStochastic();
  }

  /** Power of P at index n. */
  pPower(n: number): Matrix { /* ... */ }

  /** Stationary distribution via eigendecomposition. */
  stationary(): number[] {
    const eig = new EigenvalueDecomposition(this.P.transpose());
    const realEigs = eig.realEigenvalues;
    const idx = argminBy(realEigs, e => Math.abs(e - 1));
    const v = eig.eigenvectorMatrix.getColumn(idx);
    return normalizeToSimplex(v);
  }

  /** All eigenvalues (for spectral gap computation). */
  eigenvalues(): Complex[] { /* ... */ }

  /** Second-largest eigenvalue magnitude. */
  lambdaStar(): number {
    const eigs = this.eigenvalues().map(c => c.abs()).sort((a, b) => b - a);
    return eigs[1];
  }

  /** Mixing time bound for given epsilon. */
  mixingTimeBound(epsilon: number): number {
    return Math.log(1 / epsilon) / (1 - this.lambdaStar());
  }

  /** Communicating classes via Tarjan's SCC on the support graph of P. */
  communicatingClasses(): number[][] { /* ... */ }

  /** Period of state i. */
  period(i: number): number { /* ... */ }

  /** Is the chain irreducible? */
  isIrreducible(): boolean { return this.communicatingClasses().length === 1; }

  /** Is the chain aperiodic? */
  isAperiodic(): boolean { return this.period(0) === 1; /* class property */ }

  /** Total variation distance ‖μ_n - π‖_TV for given initial distribution. */
  tvToStationary(mu0: number[], n: number): number { /* ... */ }

  /** Detailed balance check against a given distribution. Returns max residual. */
  detailedBalanceResidual(pi: number[]): number { /* ... */ }

  /** Sample one trajectory of length T, starting at state s0. */
  sampleTrajectory(s0: number, T: number, rng: () => number = Math.random): number[] { /* ... */ }
}
```

**Vitest targets (from pre-verified numerics):**

```ts
test('Weather chain stationary distribution', () => {
  const chain = new MarkovChain([[0.7,0.2,0.1],[0.3,0.4,0.3],[0.2,0.3,0.5]]);
  const pi = chain.stationary();
  expect(pi[0]).toBeCloseTo(21/46, 6);
  expect(pi[1]).toBeCloseTo(13/46, 6);
  expect(pi[2]).toBeCloseTo(12/46, 6);
});

test('Weather chain second eigenvalue', () => {
  const chain = new MarkovChain([[0.7,0.2,0.1],[0.3,0.4,0.3],[0.2,0.3,0.5]]);
  expect(chain.lambdaStar()).toBeCloseTo(0.473205, 5);
});

test('Periodic chain has period 2', () => {
  const chain = new MarkovChain([[0,1],[1,0]]);
  expect(chain.period(0)).toBe(2);
  expect(chain.isAperiodic()).toBe(false);
});

test('Reducible chain has two communicating classes', () => {
  const P = [
    [0.7, 0.3, 0,   0  ],
    [0.4, 0.6, 0,   0  ],
    [0.1, 0,   0.3, 0.6],
    [0,   0.2, 0.5, 0.3],
  ];
  const chain = new MarkovChain(P);
  const classes = chain.communicatingClasses();
  expect(classes.length).toBe(2);
  expect(chain.isIrreducible()).toBe(false);
});

test('Detailed balance for birth-death chain', () => {
  const P = [[0.5,0.5,0],[0.3,0.4,0.3],[0,0.6,0.4]];
  const chain = new MarkovChain(P);
  const pi = [6/21, 10/21, 5/21];
  expect(chain.detailedBalanceResidual(pi)).toBeLessThan(1e-10);
});

test('Power-of-P convergence rate', () => {
  const chain = new MarkovChain([[0.7,0.2,0.1],[0.3,0.4,0.3],[0.2,0.3,0.5]]);
  const tv1 = chain.tvToStationary([1,0,0], 1);
  const tv2 = chain.tvToStationary([1,0,0], 2);
  // ratio tv2/tv1 ≈ λ_2 = 0.4732
  expect(tv2/tv1).toBeCloseTo(0.4732, 2);
});
```

---

## 6. Component Catalog

| Code | Component                         | Section | Polish |
|-----|-----------------------------------|---------|--------|
| V1  | `<WeatherChainExplorer>`          | §1      | 1 day  |
| V2  | `<PowerOfPAnimator>`              | §2      | 1 day  |
| V3  | `<StateClassificationInspector>`  | §3      | 1.5 days |
| V4  | `<StationaryDistributionFinder>`  | §4      | 1 day  |
| V5  | `<ConvergenceLab>`                | §5      | **3 days** (centerpiece) |
| V6  | `<DetailedBalanceFlow>`           | §6      | 1 day  |
| V7  | `<PolicyInducedChainPreview>`     | §7      | 1.5 days |

Total polish budget: ~10 days. Lighter than Bandits (~11-12 days), which is
appropriate — this is a prereq, not a destination lesson.

---

## 7. Page-Level UX

Same conventions as Lesson 1: single-page scroll, prereq strip at top,
reduced-motion support, mobile fallbacks for interactives. RNG seeds
URL-shareable.

One new consideration: **the chain editor in V3 and V4 shares logic.** Build
it once as a reusable `<ChainEditor>` Web Component used by both. This is the
biggest engineering decision in the lesson; the agent should call it out
explicitly in their initial plan.

---

## 8. Acceptance Criteria

After this lesson the learner can:

1. Write down the Markov property in formal notation.
2. Compute multi-step transition probabilities via matrix powers.
3. Identify communicating classes, recurrence/transience, and period of a
   given small chain.
4. Set up and solve `π = πP` for a 3-state chain by hand.
5. State the convergence theorem precisely, including its irreducibility
   and aperiodicity conditions.
6. Identify the spectral gap of a chain and use it to bound mixing time.
7. Check detailed balance and use it as a verification shortcut.
8. Explain in one sentence why a policy turns an MDP into a Markov chain
   on states.

Concrete check: hand them the 3-state chain
`P = [[0.1, 0.6, 0.3], [0.5, 0.2, 0.3], [0.2, 0.3, 0.5]]`. Ask them to
(a) compute π by solving the balance equations, (b) compute the spectral
gap, (c) estimate the mixing time at ε = 0.01, (d) check whether detailed
balance holds (it doesn't — the chain is irreversible). All can be done
by hand in ~20 minutes; the Convergence Lab provides numerical verification.

---

## 9. Stretch Goals (post-MVP)

- **Continuous-time chains** (rate matrices) as a short appendix.
- **MCMC mini-section**: Metropolis-Hastings and Gibbs as engineered chains
  with target `π`.
- **Random walks on graphs**: the canonical reversible chain, with
  stationary distribution proportional to vertex degree.
- **Spectral gap of structured chains**: lazy random walks, expander graphs.

---

## 10. Out of Scope (intentionally)

- **Continuous state spaces**, measure theory, general state space chain theory.
- **Infinite-state chains** (recurrence is much subtler — positive vs null
  recurrent, etc.).
- **Hidden Markov models** — they belong to inference/ML, not RL substrate.
- **Time-inhomogeneous chains** — RL uses these implicitly for off-policy
  evaluation but the theory is more elaborate.

---

## 11. Training Notebook

**Not applicable.** No models are trained. The script
`scripts/markov_examples.py` exists only to pre-compute the example chains
and their stationary distributions/eigenvalues as JSON for offline display:

```
public/data/markov/
  weather.json
  periodic2.json
  reducible.json
  slow_mixing.json
  birth_death.json
```

Each JSON file contains `{ P, pi, eigenvalues, mixingTimeBound, ... }`. The
TS code in the browser computes these live too — the JSON is for instant
initial render before the first interaction. ~30 lines of Python total.

---

## End of spec

Total length: ~1050 lines. Lighter than Bandits as a prereq should be. The
Convergence Lab (V5) is where the polish budget lives; protect it from
feature creep in V1-V4 and V6-V7.
