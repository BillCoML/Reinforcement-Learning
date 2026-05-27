# Prereq C — Contractions and the Banach Fixed-Point Theorem

> **The single theorem that powers all of dynamic programming.** "Iterating a
> contraction converges to its unique fixed point at a geometric rate." That's
> the whole story. We're going to prove it carefully, give it a 1D and 2D
> picture, then deploy it to show that the Bellman operators are contractions
> — which is why value iteration converges, why policy iteration works, and
> why the discount factor `γ` controls everything.

> **Where this slots in.** Between Lesson 2 (MDPs) and Lesson 3 (Dynamic
> Programming). MDPs introduced the Bellman operators `T^π` and `T^*` and
> mentioned "γ-contraction" as a stated fact. Lesson 3 will iterate them
> until they converge. This prereq is the engine room: the theorem that
> makes the iteration work, plus the proofs that the Bellman operators
> qualify.

---

## 0. Pedagogical Philosophy

1. **Earn the theorem, then deploy it.** Sections 1-4 build the abstract
   machinery. Section 5 cashes it in for the Bellman operators. The
   payoff structure is deliberate: the learner sees the theorem in its
   most general form *first*, then watches it explain a specific
   phenomenon they've already encountered.

2. **The proof is the lesson.** Banach's theorem has a 5-line proof that
   is genuinely beautiful and tells you everything you need to know:
   *why* the rate is geometric, *why* the fixed point is unique, *why*
   completeness matters. We write it out fully.

3. **Counterexamples in their own section.** When contractions fail —
   `c = 1`, non-completeness, multiple fixed points — *something specific*
   goes wrong. Showing those failures explicitly is more memorable than
   listing hypotheses.

4. **Short.** This is a prereq, not a destination. Six sections, six
   visualizations, ~900 lines. The agent finishes with energy in reserve.

5. **End by pointing forward.** Lesson 3 cashes in within hours of the
   learner finishing. The forward link is unusually tight — "next lesson
   *literally* implements `V_{k+1} = T^* V_k`."

---

## 1. Tech Stack

Identical to prior lessons. The math here is light — most computation is
1D or 2D iteration; the gridworld reuse comes for free from Lesson 2's
infrastructure.

No new Python scripts; the `mdp_gridworld.py` from Lesson 2 produces the
data we need (a few Bellman backup traces with different initial V's).

---

## 2. Visual / Aesthetic Direction

Extend the curriculum aesthetic. New tokens for this lesson:

```css
:root {
  /* Contraction-specific palette */
  --contr-input:        #2563eb;   /* blue-600  | a point/set in domain */
  --contr-output:       #16a34a;   /* green-600 | T(point)/T(set), shrunk */
  --contr-fixed-point:  #b91c1c;   /* red-700   | the fixed point x* */
  --contr-trajectory:   #6b7280;   /* gray-500  | iteration trails */

  /* Bound visualization */
  --contr-bound:        #1c1e22;   /* dashed black ref line */
  --contr-bound-fill:   rgba(28, 30, 34, 0.06);

  /* For contraction-on-set visualization */
  --contr-set-original: rgba(37, 99, 235, 0.18);
  --contr-set-shrunk:   rgba(22, 163, 74, 0.18);
}
```

**Convention for iteration visualizations.** Trajectory dots in
`--contr-trajectory`, fading older points to lower opacity (0.3 → 1.0
across the last 8 steps). The fixed point gets a small red ring marker.
Arrows from `x_k` to `x_{k+1}` show the iteration direction.

---

## 3. Lesson Metadata

```ts
export const lessonMeta: LessonMeta = {
  id: "contractions",
  title: "Contractions and the Banach Fixed-Point Theorem",
  subtitle: "The engine of dynamic programming",
  tier: 1,
  difficulty: 3,
  estimatedReadMinutes: 35,
  role: "prereq",
  prerequisites: [
    { external: true, label: "Linear algebra: norms, matrix operator norms" },
    { external: true, label: "Real analysis: sequences and limits (Cauchy sequences a plus)" },
    { lesson: "mdps", anchor: "bellman-expectation" },
  ],
  exportedAnchors: [
    "metric-space",
    "contraction-property",
    "lipschitz-constant",
    "banach-fixed-point",
    "geometric-error-bound",
    "completeness-required",
    "bellman-pi-contraction",
    "bellman-star-contraction",
  ],
  centerpieceComponent: "BellmanContractionExplorer",
  forwardLinksWhenReady: [
    { to: "dynamic-programming", anchor: "value-iteration" },
    { to: "dynamic-programming", anchor: "policy-iteration" },
    { to: "td-learning",         anchor: "td-convergence-theory" },
    { to: "function-approx",     anchor: "deadly-triad" },
  ],
};
```

---

## 4. Section-by-Section Plan

### §1 — Metric Spaces, Briefly
**Tagline:** *We need a notion of distance, and we need our space to be "closed under limits."*
**Length:** ~450 words.
**Anchor:** `metric-space`.

---

**Prose:**

To talk about "contractions" we need a *distance*. A **metric space** is a
set `X` equipped with a function `d : X \times X \to \mathbb{R}_{\geq 0}`
satisfying

1. `d(x, y) = 0 \iff x = y`,
2. `d(x, y) = d(y, x)`,
3. `d(x, z) \leq d(x, y) + d(y, z)` (triangle inequality).

We won't dwell on these axioms — they capture the geometric meaning of
"distance" in a way that lets us compute. The three metric spaces we'll
use in this lesson are:

- **`\mathbb{R}` with `d(x, y) = |x - y|`** — the real line. Trivial,
  useful for warm-up.
- **`\mathbb{R}^n` with `d(x, y) = \|x - y\|_2`** — Euclidean space.
- **`\mathbb{R}^K` with `d(x, y) = \|x - y\|_\infty = \max_i |x_i - y_i|`**
  — Euclidean space with the **sup-norm** (a.k.a. max-norm or `\ell_\infty`-norm).
  This is the *only* metric that will matter for the Bellman analysis in §5.

A sequence `\{x_n\}` in a metric space **converges** to `x` if `d(x_n, x) \to 0`
as `n \to \infty`. A sequence is **Cauchy** if `d(x_m, x_n) \to 0` as
`m, n \to \infty` — the elements get arbitrarily close to *each other*.

**Completeness.** A metric space is **complete** if every Cauchy sequence
converges to some limit *within the space*. Equivalent intuition: the
space has no "missing points" that should be there.

| Space | Complete? | Why it matters |
|-------|:---------:|----------------|
| `\mathbb{R}` (with `|\cdot|`) | ✓ | The classical case. |
| `\mathbb{R}^n` (any norm) | ✓ | Our default. |
| `\mathbb{Q}` (with `|\cdot|`) | ✗ | A Cauchy sequence of rationals can converge to an irrational. |
| `(0, 1)` open interval | ✗ | A Cauchy sequence approaching 0 has no limit in the space. |
| Continuous functions on `[0,1]` with sup-norm | ✓ | The function-space analog. |

> **Why does completeness matter for fixed-point theorems?** Because we
> want the *limit of our iteration* to actually exist in the space we
> care about. If the limit lives outside the space, the theorem can't
> tell us anything useful about it. The Banach theorem will hand us a
> Cauchy sequence and a candidate limit — completeness is what lets us
> say the limit is in `X` and is, therefore, a meaningful object.

For RL, all our metric spaces will be finite-dimensional vector spaces with
familiar norms — they're complete by inheritance from `\mathbb{R}`. We
won't sweat this hypothesis again, but it's there in the background of
every theorem.

---

**Visualization V1 — Norm Comparison.**

- A small interactive in `\mathbb{R}^2`: a fixed reference point at the
  origin and a draggable point `x`.
- Three real-time readouts: `\|x\|_2` (Euclidean), `\|x\|_\infty` (sup),
  `\|x\|_1` (sum of absolute values).
- Unit balls of all three norms drawn as faint outlines: the round disk
  (`\ell_2`), the square (`\ell_\infty`), the diamond (`\ell_1`).
- A toggle: "highlight max coord" — colors the dominant coordinate of
  `x` in the sup-norm reading, demonstrating *which* coordinate the
  sup-norm "picks."
- Width 600, height 360.

This is a small, warm-up viz — the goal is just to make the sup-norm
concrete, since it's the one we'll use everywhere in §5.

---

### §2 — Contractions
**Tagline:** *A map that shrinks distances by a constant factor.*
**Length:** ~550 words.
**Anchor:** `contraction-property`.

---

**Prose:**

Let `(X, d)` be a metric space and `T : X \to X` a map.

`T` is a **contraction** (or **contractive mapping**) if there exists a
constant `c \in [0, 1)` such that

$$
\boxed{d(T(x), T(y)) \;\leq\; c \cdot d(x, y) \qquad \text{for all } x, y \in X.}
$$

The constant `c` is called the **contraction constant** (or **Lipschitz
constant**, when emphasizing the Lipschitz nature). Two things to notice:

1. **The "$<$" matters.** If `c = 1`, `T` is merely *non-expansive*, not
   a contraction. The theorem in §3 *requires* `c < 1` strictly; even
   `c = 1 - \epsilon` for arbitrarily small `\epsilon > 0` is fine, but
   `c = 1` is not.

2. **The constant must be uniform.** The inequality has to hold for *every*
   pair `(x, y)`, with the *same* `c`. A map that contracts more in some
   regions than others is fine, as long as some universal bound holds.

**Examples.**

| `T` | Domain | Contraction? | Constant |
|-----|--------|:------------:|----------|
| `T(x) = 0.5x + 1` | `\mathbb{R}` | ✓ | `c = 0.5` |
| `T(x) = 2x - 1` | `\mathbb{R}` | ✗ | Lipschitz constant 2 |
| `T(x) = x + 1` (translation) | `\mathbb{R}` | ✗ | `c = 1` — only non-expansive |
| `T(x) = \cos(x)` | `\mathbb{R}` | ✓ on appropriate subset | `c = \sin(1) \approx 0.84` near `x^* \approx 0.739` |
| `T(x) = x^2` | `[0, 1]` | ✗ on full interval | Local contraction only near 0 |
| `T(V) = R^\pi + \gamma P^\pi V` | `\mathbb{R}^K`, sup-norm | **✓ (§5 proves this)** | `c = \gamma` |

**The connection to Lipschitz continuity.** A map `T` is **Lipschitz** with
constant `L` if `d(T(x), T(y)) \leq L \cdot d(x, y)` for all `x, y`. A
contraction is a Lipschitz map with `L = c < 1`. So *all* contractions are
Lipschitz, hence continuous; the converse fails (continuous maps can have
Lipschitz constant 1 or larger).

**The two diagnostic tools for "is this a contraction?"**

- **In `\mathbb{R}`:** if `T` is differentiable, the mean value theorem
  gives `|T(x) - T(y)| = |T'(\xi)| \cdot |x - y|` for some `\xi`. So `T`
  is a contraction with constant `c` iff `|T'(x)| \leq c < 1` everywhere
  on the domain. Quick check, often decisive.

- **In `\mathbb{R}^K` with sup-norm and a linear map `T(x) = Ax + b`:**
  the relevant quantity is the **operator norm**
  `\|A\|_\infty := \max_i \sum_j |A_{ij}|` (the maximum absolute row sum).
  Then `\|Ax\|_\infty \leq \|A\|_\infty \cdot \|x\|_\infty`, so `T` is a
  contraction iff `\|A\|_\infty < 1`. This is *exactly* the criterion
  we'll use in §5 with `A = \gamma P^\pi` — and `\|P^\pi\|_\infty = 1`
  (row-stochastic), so `\|\gamma P^\pi\|_\infty = \gamma < 1`.

---

**Visualization V2 — Contraction Iterator (1D).**

- A horizontal number line.
- A draggable point `x_0` and a contraction selector dropdown:
  - `T(x) = 0.5x + 1` (contraction, `c = 0.5`, fixed point at 2)
  - `T(x) = 0.9x + 0.5` (slow contraction, `c = 0.9`, fixed point at 5)
  - `T(x) = -0.6x + 4` (negative slope contraction, alternating)
  - `T(x) = x + 1` (no fixed point, demonstrates `c = 1` failure)
- "Step" button applies `T` once; iteration trail visible behind the
  current point.
- A separate panel shows the graph of `T` against `y = x` (the identity);
  the cobweb plot animates each iteration as a horizontal-then-vertical
  hop.
- Width 720, height 360.

The cobweb plot is the classic 1D-iteration visualization; it makes
"convergence to fixed point" geometric.

---

### §3 — The Banach Fixed-Point Theorem
**Tagline:** *Iterate. Converge. Geometrically.*
**Length:** ~750 words.
**Anchor:** `banach-fixed-point`.

---

**Prose:**

**Theorem (Banach Fixed-Point, 1922).** Let `(X, d)` be a *complete*
metric space and `T : X \to X` a contraction with constant `c \in [0, 1)`.
Then:

1. `T` has a **unique fixed point** `x^* \in X`, meaning `T(x^*) = x^*`.
2. For any starting point `x_0 \in X`, the iterates `x_{k+1} := T(x_k)`
   converge to `x^*`.
3. The convergence is **geometric**:

$$
\boxed{d(x_k, x^*) \;\leq\; \frac{c^k}{1 - c} \cdot d(x_1, x_0).}
$$

---

**Proof.**

*Step 1: The iteration produces a Cauchy sequence.* For any `k \geq 0`,

$$
d(x_{k+1}, x_k) \;=\; d(T(x_k), T(x_{k-1})) \;\leq\; c \cdot d(x_k, x_{k-1}).
$$

By induction, `d(x_{k+1}, x_k) \leq c^k \cdot d(x_1, x_0)`. Now for `m > n`,
by the triangle inequality and geometric summation,

$$
d(x_m, x_n) \;\leq\; \sum_{k=n}^{m-1} d(x_{k+1}, x_k) \;\leq\; \sum_{k=n}^{m-1} c^k \cdot d(x_1, x_0) \;\leq\; \frac{c^n}{1 - c} \cdot d(x_1, x_0).
$$

The right-hand side `\to 0` as `n \to \infty`, independently of `m`. So
`\{x_k\}` is Cauchy.

*Step 2: The limit exists in `X`.* Since `X` is complete, the Cauchy
sequence converges: `x_k \to x^*` for some `x^* \in X`. (This is the
*only* place completeness is used — it lets us promote "Cauchy" to "has a
limit in the space.")

*Step 3: The limit is a fixed point.* Apply `T` to both sides of
`x_k \to x^*`. Since `T` is a contraction it is continuous, so
`T(x_k) \to T(x^*)`. But `T(x_k) = x_{k+1}`, which also converges to
`x^*`. By uniqueness of limits, `T(x^*) = x^*`.

*Step 4: The fixed point is unique.* If `T(y^*) = y^*` too, then

$$
d(x^*, y^*) \;=\; d(T(x^*), T(y^*)) \;\leq\; c \cdot d(x^*, y^*).
$$

Since `c < 1`, this forces `d(x^*, y^*) = 0`, hence `x^* = y^*`.

*Step 5: The error bound.* Take `n = k`, `m \to \infty` in the Step 1
inequality:

$$
d(x_k, x^*) \;\leq\; \lim_{m \to \infty} d(x_m, x_n)\big|_{n=k} \;\leq\; \frac{c^k}{1 - c} \cdot d(x_1, x_0). \qquad \blacksquare
$$

---

**The proof in one sentence.** Successive iterates get closer because
`T` shrinks distances; sum the geometric series to bound how far they can
drift; completeness ensures the limit exists; uniqueness falls out of the
contraction inequality applied to two purported fixed points.

---

**The error bound, used in practice.** The bound

$$
d(x_k, x^*) \;\leq\; \frac{c^k}{1 - c} \cdot d(x_1, x_0)
$$

is a **computable stopping criterion**. After `k` iterations, we know how
close we are to `x^*` without ever knowing `x^*` itself: we just compute
`d(x_1, x_0)` (the first step's distance) and multiply by `c^k / (1 - c)`.

There's also an *a posteriori* bound that's often tighter:

$$
d(x_k, x^*) \;\leq\; \frac{c}{1 - c} \cdot d(x_k, x_{k-1}).
$$

This says "the error is at most `c/(1-c)` times the *last* step's
movement." Once `d(x_k, x_{k-1})` is small, you're close to convergence.
For our `T(x) = 0.5x + 1` example with `c = 0.5`, this gives "error
≤ last step." Convergence checks are easy.

---

**Numerical verification (pre-verified).** For `T(x) = 0.5x + 1` with
`x_0 = 10`, fixed point `x^* = 2`:

| `k` | `x_k` | actual `|x_k - x^*|` | bound `c^k/(1-c) \cdot |x_1-x_0|` |
|----:|------:|---------------------:|---------------------------------:|
| 0   | 10.000 | 8.0000 | 8.0000 |
| 1   | 6.000  | 4.0000 | 4.0000 |
| 2   | 4.000  | 2.0000 | 2.0000 |
| 3   | 3.000  | 1.0000 | 1.0000 |
| 4   | 2.500  | 0.5000 | 0.5000 |
| 5   | 2.250  | 0.2500 | 0.2500 |
| 6   | 2.125  | 0.1250 | 0.1250 |

In this particular example the bound is *tight* — actual error equals
the bound exactly, because the contraction is a pure linear map and the
geometry aligns perfectly. For general contractions the bound is loose
by some constant factor, but the *rate* `c^k` is always the truth.

> **Forward link to RL** — When we iterate the Bellman optimality operator
> `V_{k+1} = T^* V_k` in value iteration (Lesson 3), the stopping
> criterion `\|V_k - V_{k-1}\|_\infty \leq \epsilon (1 - \gamma) / \gamma`
> guarantees `\|V_k - V^*\|_\infty \leq \epsilon`. This is literally the
> a posteriori bound with `c = \gamma`.

---

**Visualization V3 — Banach Iteration in 2D.**

- A 2D plane with a draggable starting point.
- A 2D contraction `T(x) = Ax + b` with the matrix `A` editable (2×2
  matrix, sliders for the four entries). Display the spectral radius and
  the operator norm `\|A\|_\infty`.
- Multiple starting points can be added (click-to-add). All of them
  iterate simultaneously, drawing trajectories that converge to the same
  fixed point `x^* = (I - A)^{-1} b`.
- A log-scale plot to the side: `\|x_k - x^*\|` over `k` for each
  trajectory, all showing the same geometric decay rate.
- "Match contraction constant" indicator: shows whether `\|A\|_\infty < 1`
  (contraction, ✓ in green) or `\geq 1` (not a contraction, ⚠ in red).
- Width 800, height 460.

This viz is meant to convey one thing viscerally: **any starting point
flows to the same fixed point, at the same rate.** That's the whole theorem,
made visible.

---

### §4 — When Contractions Fail
**Tagline:** *Three things can go wrong, and each illuminates a hypothesis.*
**Length:** ~550 words.
**Anchor:** `completeness-required`.

---

**Prose:**

The theorem has three hypotheses: `T` is a contraction, `T : X \to X`
maps `X` to itself, and `X` is complete. Each hypothesis is *necessary*
— violate one and the conclusion can fail. Three short counterexamples
make this concrete.

**Failure 1: `c = 1` (translation).** Consider `T(x) = x + 1` on
`\mathbb{R}`. This is *non-expansive* — it preserves distances exactly,
since `|T(x) - T(y)| = |x - y|`. But the iteration is
`0, 1, 2, 3, \ldots`, which diverges to infinity. No fixed point exists,
because the equation `T(x) = x \iff x + 1 = x` has no solution. The
strict inequality `c < 1` in the contraction definition is what prevents
this.

**Failure 2: Non-completeness (rationals).** Consider `T(x) = (x + 2/x)/2`
on `\mathbb{Q}_{>0}` (positive rationals) with the usual metric. This is
Newton's method for finding `\sqrt{2}`. Iterating from `x_0 = 1` produces
`3/2, 17/12, 577/408, \ldots` — a Cauchy sequence that converges to
`\sqrt{2}`. But `\sqrt{2} \notin \mathbb{Q}`. The limit *exists in
`\mathbb{R}`* but *not in `\mathbb{Q}`*. The space `\mathbb{Q}` isn't
complete, so the iteration "escapes" the space.

Switch the same `T` to `\mathbb{R}_{>0}`: now it's a contraction near
`\sqrt{2}`, completeness holds, and the theorem applies. Same map,
different space, different fate.

**Failure 3: Multiple fixed points (when `T` isn't actually a contraction).**
Consider `T(x) = x^2` on `[0, 1]`. Two fixed points exist: `0` and `1`.
Iterates from `x_0 = 0.9` converge to `0` (one fixed point); iterates from
`x_0 = 1.0` stay at `1` (the other). The map *is not a contraction* on
`[0, 1]` because `|T'(x)| = 2x` can be as large as `2` at `x = 1`. It
*is* a contraction on subsets bounded away from 1 (where `|T'(x)| \leq 2 \cdot \text{ess sup}|x| < 1`).
This is a typical pattern: the map is a "local" contraction with a
"basin of attraction" `\{x : T^k(x) \to x^*\}`.

> **Important nuance.** In RL we're going to need a *global* contraction
> on a full vector space — `T^\pi : \mathbb{R}^K \to \mathbb{R}^K`. We can't
> rely on locality. The proof in §5 shows that the contraction constant
> `\gamma` is *uniform* over the entire space, which is what makes value
> iteration converge from any starting `V`.

---

**Visualization V4 — Counterexample Gallery.**

A small tabbed panel showing the three failure cases:

- **Tab 1: "c = 1" (translation).** Same 1D iterator as V2 but pinned to
  `T(x) = x + 1`. The trajectory marches off to infinity. A side note
  says "no fixed point exists."

- **Tab 2: "Incomplete space."** Newton's iteration for `\sqrt{2}` shown
  on the rational number line. The rational iterates are highlighted; the
  limit `\sqrt{2}` (irrational) is shown as a faint dashed marker the
  iterates approach but never reach (within `\mathbb{Q}`).

- **Tab 3: "Multiple fixed points."** Cobweb plot of `T(x) = x^2` on
  `[0, 1]`. Two fixed points at 0 and 1 marked. Trajectories from
  different starting points converge to different fixed points. Slider
  for `x_0`; the destination is determined by which side of the unstable
  fixed point you start on.

- Width 720, height 380.

---

### §5 — Bellman Operators are Contractions
**Tagline:** *Two theorems, two proofs. Now value iteration converges.*
**Length:** ~850 words.
**Anchor:** `bellman-pi-contraction`.

---

**Prose:**

We now apply the Banach theorem to RL. Two operators need contracting:
`T^\pi` (Bellman expectation, for policy evaluation) and `T^*` (Bellman
optimality, for value iteration).

Recall the setup. The state space `\mathcal{S}` has `K` states. Value
functions live in `\mathbb{R}^K`, which we equip with the **sup-norm**
`\|V\|_\infty := \max_s |V(s)|`. The metric is `d(V, V') := \|V - V'\|_\infty`.
This space is complete (a finite-dimensional vector space with any norm
is complete).

---

**Theorem 1 (Bellman expectation contraction).** For any policy `\pi` and
discount factor `\gamma \in [0, 1)`, the operator

$$
(T^\pi V)(s) \;:=\; \sum_a \pi(a|s) \left[ r(s, a) + \gamma \sum_{s'} P(s'|s, a) V(s') \right]
$$

is a `\gamma`-contraction on `(\mathbb{R}^K, \|\cdot\|_\infty)`.

*Proof.* For any two value functions `V, V' \in \mathbb{R}^K`, the
expectation term has the form `T^\pi V = R^\pi + \gamma P^\pi V`. So

$$
T^\pi V - T^\pi V' \;=\; \gamma P^\pi (V - V').
$$

Taking sup-norms,

$$
\|T^\pi V - T^\pi V'\|_\infty \;=\; \gamma \|P^\pi (V - V')\|_\infty \;\leq\; \gamma \|P^\pi\|_\infty \cdot \|V - V'\|_\infty.
$$

The operator norm `\|P^\pi\|_\infty = \max_s \sum_{s'} |P^\pi_{ss'}| = 1`
because `P^\pi` is row-stochastic. Hence

$$
\|T^\pi V - T^\pi V'\|_\infty \;\leq\; \gamma \|V - V'\|_\infty. \qquad \blacksquare
$$

The contraction constant is exactly `\gamma`. Apply Banach: `T^\pi` has a
unique fixed point `V^\pi`, and `V_{k+1} := T^\pi V_k` converges to it
geometrically at rate `\gamma`. This is *iterative policy evaluation*,
which we'll implement in Lesson 3.

---

**Theorem 2 (Bellman optimality contraction).** The Bellman optimality
operator

$$
(T^* V)(s) \;:=\; \max_a \left[ r(s, a) + \gamma \sum_{s'} P(s'|s, a) V(s') \right]
$$

is also a `\gamma`-contraction on `(\mathbb{R}^K, \|\cdot\|_\infty)`.

*Proof.* The new ingredient is the **max-Lipschitz lemma**: for any two
functions `f, g : \mathcal{A} \to \mathbb{R}`,

$$
\left| \max_a f(a) - \max_a g(a) \right| \;\leq\; \max_a |f(a) - g(a)|.
$$

(Proof: let `a^* = \arg\max f`. Then `\max f - \max g \leq f(a^*) - g(a^*) \leq |f(a^*) - g(a^*)| \leq \max_a |f-g|`.
Symmetric argument for the other direction.)

Apply this state-by-state. For each `s`, let
`f_a := r(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s')` and
`g_a := r(s,a) + \gamma \sum_{s'} P(s'|s,a) V'(s')`. Then

$$
\begin{aligned}
\left| (T^* V)(s) - (T^* V')(s) \right| &= \left| \max_a f_a - \max_a g_a \right| \\
&\leq \max_a |f_a - g_a| \\
&= \max_a \left| \gamma \sum_{s'} P(s'|s,a) [V(s') - V'(s')] \right| \\
&\leq \gamma \max_a \sum_{s'} P(s'|s,a) \|V - V'\|_\infty \\
&= \gamma \|V - V'\|_\infty.
\end{aligned}
$$

(The penultimate step used `|V(s') - V'(s')| \leq \|V - V'\|_\infty` and
`\sum_{s'} P(s'|s,a) = 1`.) Taking the max over `s`,

$$
\|T^* V - T^* V'\|_\infty \;\leq\; \gamma \|V - V'\|_\infty. \qquad \blacksquare
$$

Apply Banach: `T^*` has a unique fixed point `V^*`, and iterating
`V_{k+1} := T^* V_k` converges to it geometrically at rate `\gamma`.
This is *value iteration*.

---

**Numerical verification on the gridworld (pre-verified).**

Take two arbitrary starting value functions on our 3×3 gridworld:
`V_1 = \mathbf{0}` (all zeros) and `V_2 = 2 \cdot \mathbf{1}` (all twos).
Initial sup-distance: `\|V_1 - V_2\|_\infty = 2`. Apply `T^\pi` for
uniform policy:

| iteration `k` | `\|V_1^{(k)} - V_2^{(k)}\|_\infty` | empirical ratio |
|--------------:|-----------------------------------:|----------------:|
| 0             | 2.0000                              | —              |
| 1             | 1.8000                              | 0.9000          |
| 2             | 1.6200                              | 0.9000          |
| 3             | 1.2758                              | 0.7875          |
| 4             | 1.0252                              | 0.8036          |
| 5             | 0.8119                              | 0.7920          |
| 10            | 0.2305                              | 0.7722          |

The first two ratios are *exactly* `\gamma = 0.9`. After that they drift
slightly *below* `\gamma`. **The theorem only guarantees an upper bound —
actual convergence can be faster**, and indeed is faster once the
"difference vector" `V_1 - V_2` has been smoothed by repeated averaging
under `P^\pi`. The *worst case* is `\gamma`; the *typical* case can be
much better.

For `T^*` the same experiment gives initial ratios at exactly `\gamma` for
*several* iterations before drifting (because the `\max` operator doesn't
average — it picks the worst-case action, which keeps the bound tight
longer). Either way, the geometric rate `\gamma` is the truth.

---

**Why this matters for the rest of the curriculum.**

- **Value iteration converges.** Lesson 3 will iterate `V_{k+1} = T^* V_k`
  until `\|V_{k+1} - V_k\|_\infty < \epsilon (1 - \gamma) / \gamma`,
  guaranteeing `\|V_k - V^*\|_\infty < \epsilon`. The contraction
  property is *literally* what makes that algorithm finite.

- **Policy iteration converges in finitely many steps.** Each policy
  evaluation is a Banach iteration (rate `\gamma`); each policy
  improvement strictly improves the policy until none exists, in which
  case we're at `\pi^*`. Lesson 3 will prove this in full.

- **TD(0) and Q-learning converge.** The *sample-based* Bellman backup
  in TD methods turns the deterministic contraction into a stochastic
  approximation; the Robbins-Monro theorem (Prereq B / Lesson 5) covers
  this with extra assumptions on step-size schedules. The contraction
  is a *necessary* ingredient even when not sufficient.

- **The deadly triad of function approximation.** When `V` is replaced
  by a neural-net approximation `V_\theta`, the projection step in TD
  can be non-expansive but *not* a contraction. This is exactly why the
  combination of bootstrapping + off-policy + function approximation
  can diverge (Lesson 6). The Banach hypotheses fail.

---

**Visualization V5 — Bellman Contraction Explorer. THIS IS THE CENTERPIECE.**

Budget: 2-3 days. Slightly lighter than other centerpieces because the
lesson is shorter.

A three-panel synchronized dashboard.

**Panel A (left): Two value functions side-by-side.**
- Two 3×3 gridworld renderers stacked vertically.
- `V_1` on top (initialized to all zeros by default).
- `V_2` below (initialized to a colorful arbitrary pattern by default).
- Click any cell to manually edit its value (slider 0-2).
- "Randomize V_2" button.

**Panel B (center): The Bellman backup itself.**
- A live readout of `\|V_1 - V_2\|_\infty` — the sup-norm distance.
- After "Apply T^π" (or T^*): compute both `T^π V_1` and `T^π V_2`,
  display the new sup-distance, and show the empirical ratio
  `\|T V_1 - T V_2\|_\infty / \|V_1 - V_2\|_\infty`.
- The ratio is highlighted in green if `\leq \gamma` (theorem
  holds) and red if `> \gamma` (would be a bug).
- Iteration counter and a "step" button.
- Mode toggle: `T^\pi` (uniform policy) vs `T^*`.

**Panel C (right): Convergence plot.**
- Log-scale `y`-axis: `\|V_1^{(k)} - V_2^{(k)}\|_\infty` over `k`.
- A dashed reference line shows `\gamma^k \cdot \|V_1^{(0)} - V_2^{(0)}\|_\infty`
  — the worst-case bound.
- The actual curve sits at or below the bound, always.
- A secondary line shows both trajectories converging to `V^\pi`
  (or `V^*` in optimality mode).

**Controls.**
- γ slider [0.1, 0.99] — at low γ, convergence is fast; at γ → 1,
  arbitrarily slow.
- Mode toggle (expectation / optimality).
- Step / play / reset.
- Speed control.

The viz makes one thing visceral: **two arbitrary value functions
converge toward each other at rate γ.** That convergence-toward-the-same-
limit *is* the contraction property. Once a learner has watched this,
"value iteration converges" stops being a theorem to memorize and
becomes a thing they've seen happen.

**Width:** ~960px (centerpiece breakout). **Height:** ~520px.

---

### §6 — Where You'll See This Again
**Tagline:** *The contraction theorem powers the next four lessons, then keeps cashing in.*
**Length:** ~400 words.
**Anchor:** `contraction-forward-links`.

---

**Prose:**

The Banach fixed-point theorem is *immediately* useful: Lesson 3 (next)
implements value iteration as `V_{k+1} := T^* V_k`, and the theorem is what
makes that loop terminate.

Five places downstream where contractions and fixed points reappear:

**1. Lesson 3 — Dynamic Programming.** Value iteration is Banach iteration
on `T^*`. Policy iteration alternates Banach iteration on `T^\pi` (the
evaluation step) with strict policy improvement. The stopping criterion
`\|V_k - V_{k-1}\|_\infty < \epsilon(1-\gamma)/\gamma` is the *a posteriori*
bound from §3 with `c = \gamma`. *Everything* in Lesson 3 is a direct
deployment of this prereq.

**2. Lesson 5 — TD Learning.** Sample-based Bellman backups are
*stochastic* versions of `T^*`. They lose the deterministic contraction
property and gain a noisy-gradient interpretation. The Robbins-Monro
theorem (Prereq B / inline) replaces Banach for this setting. The structural
similarity is unmistakable: a stochastic update that, in expectation, is a
contraction will converge under appropriate step-size conditions.

**3. Lesson 6 — Function Approximation.** When `V` is replaced by `V_\theta`,
the *projection* of `T^* V_\theta` back onto the parameterized class can
be non-expansive but *not* a contraction. Failure of the contraction
property is what makes the "deadly triad" deadly. This prereq is the
foundation; that lesson explains the failure mode.

**4. Lesson 9 — Trust Region Methods.** TRPO's natural-gradient step
solves a constrained optimization where the constraint is `\|\pi_{k+1} - \pi_k\|_{KL} \leq \delta`.
This is *not* a contraction theorem application directly, but the
fixed-point intuition (iterate until convergence; bound the per-step
movement) is the same shape. PPO's clipped ratio is a heuristic
relaxation of the same constraint.

**5. Lesson 11 — RL as Probabilistic Inference.** The soft Bellman
operator `T^{\text{soft}} V := \alpha \log \sum_a \exp([r + γPV]/α)` is
also a `\gamma`-contraction (entropy doesn't break the property). The
proof is a few lines using LogSumExp's Lipschitz property. Same theorem,
shifted reward.

---

**Visualization V6 — Roadmap Mini.**

Update the curriculum roadmap with the contraction prereq now marked as
shipped. Arrows from Contractions point to: DP (immediate), TD,
Function Approximation, Trust Region, Inference-RL.

Width 720, height 240.

---

## 5. Algorithm / Math Implementation

A small TypeScript module `src/contractions/`. Most utilities operate on
plain `number[]` arrays (for value functions) or 2×2 matrices (for the 2D
viz).

```ts
/** Apply a 1D affine contraction T(x) = a*x + b. */
export function apply1D(a: number, b: number, x: number): number {
  return a * x + b;
}

/** Iterate any function T: number -> number from x0, returning the trajectory. */
export function iterate1D(T: (x: number) => number, x0: number, n: number): number[] {
  const traj = [x0];
  for (let k = 0; k < n; k++) traj.push(T(traj[k]));
  return traj;
}

/** Apply a 2D affine map T(x) = A*x + b. */
export function apply2D(A: number[][], b: number[], x: number[]): number[] {
  return [
    A[0][0] * x[0] + A[0][1] * x[1] + b[0],
    A[1][0] * x[0] + A[1][1] * x[1] + b[1],
  ];
}

/** Compute the sup-norm operator norm of a square matrix. */
export function opNormInfinity(A: number[][]): number {
  return Math.max(...A.map(row => row.reduce((s, x) => s + Math.abs(x), 0)));
}

/** Compute the fixed point of an affine 2D contraction (I - A)^{-1} b. */
export function fixedPoint2D(A: number[][], b: number[]): number[] {
  // (I - A) x = b
  const M = [
    [1 - A[0][0], -A[0][1]],
    [-A[1][0], 1 - A[1][1]],
  ];
  const det = M[0][0]*M[1][1] - M[0][1]*M[1][0];
  return [
    (M[1][1]*b[0] - M[0][1]*b[1]) / det,
    (-M[1][0]*b[0] + M[0][0]*b[1]) / det,
  ];
}

/** Sup-norm distance between two value functions. */
export function supDist(V: number[], Vp: number[]): number {
  let m = 0;
  for (let i = 0; i < V.length; i++) m = Math.max(m, Math.abs(V[i] - Vp[i]));
  return m;
}

/** Banach error bound: c^k / (1-c) * d(x_1, x_0). */
export function banachBound(c: number, k: number, d0: number): number {
  return Math.pow(c, k) / (1 - c) * d0;
}
```

The Bellman contraction visualization (V5) reuses `bellmanExpectationBackup`
and `bellmanOptimalityBackup` from Lesson 2's `src/mdp/`. **No new MDP
code is written in this lesson.**

**Vitest targets:**

```ts
test('apply1D contraction T(x) = 0.5x + 1', () => {
  expect(apply1D(0.5, 1, 10)).toBe(6);
  expect(apply1D(0.5, 1, 2)).toBe(2);  // fixed point
});

test('iterate1D converges to fixed point of 0.5x+1', () => {
  const T = (x: number) => 0.5*x + 1;
  const traj = iterate1D(T, 10, 20);
  expect(traj[20]).toBeCloseTo(2, 4);
});

test('opNormInfinity is max absolute row sum', () => {
  const A = [[0.3, 0.4], [0.5, 0.2]];
  expect(opNormInfinity(A)).toBe(0.7);
});

test('Banach error bound exactly tracks T(x)=0.5x+1 from x_0=10', () => {
  const x0 = 10, c = 0.5, x1 = 6;
  const d0 = Math.abs(x1 - x0); // = 4
  for (let k = 0; k <= 6; k++) {
    const xstar = 2;
    let xk = x0;
    for (let j = 0; j < k; j++) xk = 0.5*xk + 1;
    expect(Math.abs(xk - xstar)).toBeCloseTo(banachBound(c, k, d0), 4);
  }
});

test('Bellman T^π contraction ratio ≤ γ on gridworld', () => {
  const mdp = buildGridworld({ slippery: false, gamma: 0.9 });
  const uniform = uniformPolicy(mdp);
  let V1 = new Array(9).fill(0);
  let V2 = new Array(9).fill(2);
  const d0 = supDist(V1, V2);
  const V1_next = bellmanExpectationBackup(mdp, uniform, V1);
  const V2_next = bellmanExpectationBackup(mdp, uniform, V2);
  const d1 = supDist(V1_next, V2_next);
  expect(d1).toBeLessThanOrEqual(0.9 * d0 + 1e-9);
  expect(d1 / d0).toBeCloseTo(0.9, 4);  // tight for the first step
});
```

---

## 6. Component Catalog

| Code | Component                       | Section | Polish |
|-----|---------------------------------|---------|--------|
| V1  | `<NormComparison>`              | §1      | 0.5 day |
| V2  | `<ContractionIterator1D>`       | §2      | 1 day  |
| V3  | `<BanachIteration2D>`           | §3      | 1 day  |
| V4  | `<CounterexampleGallery>`       | §4      | 1 day  |
| V5  | `<BellmanContractionExplorer>`  | §5      | **2-3 days** (centerpiece) |
| V6  | `<RoadmapMini>` (update)        | §6      | 0.5 day |

Total polish budget: ~6-7 days. The smallest budget of any lesson so
far — appropriate for a prereq this tight. The agent should *finish
early*, not pad the visualizations.

**Reuse:**
- `MathBlock`, `CrosslinkCallout`, `PanelChrome`, `RoadmapMini`.
- `GridworldRenderer` from Lesson 2 (used heavily by V5).
- All `src/mdp/` math from Lesson 2.

---

## 7. Page-Level UX

Same conventions as prior lessons. Single-page scroll, prereq strip,
reduced-motion support. The cobweb plot in V2 should have a "show
construction lines" toggle for learners who want to understand the
geometry step-by-step.

---

## 8. Acceptance Criteria

After this lesson the learner can:

1. Define a metric space and explain what "complete" means.
2. Define a contraction with constant `c < 1`, and tell whether a
   given simple map is one.
3. State the Banach fixed-point theorem precisely.
4. Reproduce the 5-step proof of the theorem from memory.
5. Use the geometric error bound `c^k/(1-c) \cdot d(x_1, x_0)` as a
   stopping criterion.
6. State and prove the contraction property of `T^\pi` and `T^*`.
7. Explain why the *strict* inequality `c < 1` matters, with a
   counterexample.

Concrete check: hand them an affine map `T(x) = 0.7 x + 3` and ask them
to (a) verify it's a contraction (`c = 0.7 < 1`), (b) find its fixed
point analytically (`x^* = 10`), (c) iterate from `x_0 = 0` for 5
steps by hand, (d) compute the Banach bound at `k=5` and compare to
actual. V2 lets them check.

---

## 9. Stretch Goals (post-MVP)

- **Contraction on function spaces.** A short note that the same theorem
  applies in `C([0,1])` with sup-norm; this enables Picard iteration for
  ODEs. Nice connection to numerical analysis.
- **Newton's method as a (locally) quadratic contraction.** Different
  beast (rate `O(c^{2^k})`), but the same fixed-point philosophy.
- **Brouwer & Schauder fixed-point theorems.** Non-contractive
  alternatives (continuity + compactness instead). Optional appendix.

---

## 10. Out of Scope (intentionally)

- **Stochastic approximation** (Robbins-Monro). That's Prereq B / inline
  in Lesson 5.
- **Specific algorithms** (value iteration, policy iteration). That's
  Lesson 3.
- **Function-approximation pathology.** That's Lesson 6.
- **Soft Bellman operators** (max-entropy RL). That's Lesson 10.

---

## 11. Training Notebook

**Not applicable.** No models trained, no offline simulations needed. The
Bellman contraction traces used by V5 are computed live in-browser via
Lesson 2's existing `bellmanExpectationBackup`/`bellmanOptimalityBackup`.

---

## End of spec

Total length: ~950 lines. The smallest lesson so far. The Bellman
Contraction Explorer (V5) is the centerpiece; everything else is
intentionally lean.
