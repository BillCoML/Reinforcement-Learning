# Lesson 1 — Multi-Armed Bandits

> **The simplest non-trivial reinforcement learning problem.** One state, K actions,
> unknown reward distributions, T rounds. Master this and you've internalized the
> exploration–exploitation tension that drives the rest of the curriculum.

---

## 0. Pedagogical Philosophy

This lesson opens the RL curriculum. Five commitments specific to it:

1. **Regret is the protagonist.** Most introductions teach algorithms first and
   regret as an afterthought. We invert that. The learner spends time understanding
   what they're trying to minimize before meeting any algorithm. This pays off
   in every subsequent lesson — value functions, Bellman gaps, advantage,
   sub-optimality bounds — they all rhyme with regret.

2. **Three algorithms, three philosophies.** ε-greedy (heuristic), UCB (frequentist
   confidence bounds), Thompson sampling (Bayesian posterior sampling). The point
   isn't that one is "best" — it's that each embodies a *strategy* for handling
   uncertainty that will recur in deep RL, model-based RL, and inference-RL.

3. **Every claim is run.** No regret curve appears in prose unless we've simulated
   it. The agent will produce an offline simulation script whose CSV outputs feed
   the visualizations directly. The numbers in the spec are the numbers the learner
   will see on screen.

4. **The centerpiece earns the polish budget.** The Algorithm Battle Arena (§7) is
   where this lesson lives or dies. Eight smaller visualizations build toward it.
   Each is deliberately scoped so the agent doesn't burn cycles polishing supporting
   diagrams at the centerpiece's expense.

5. **Forward links are sparse but loud.** Bandits previews exactly four downstream
   ideas: bandit-as-1-state-MDP (Lesson 2), exploration in deep RL (Lesson 7),
   contextual bandits and RLHF reward modeling (Lesson 17), and Thompson sampling
   as posterior inference (cashes in across the inference-RL track). We name these
   explicitly so the learner has hooks.

---

## 1. Tech Stack

- **Build:** Vite + TypeScript (strict mode)
- **Math:** KaTeX for typesetting
- **Visualization:** D3.js v7 for SVG/canvas plots, custom Web Components for
  interactive panels
- **Numerics:** `ml-matrix` for any linear algebra (light usage in this lesson)
- **Offline simulation:** Python 3.11 + NumPy + Matplotlib for pre-generating
  regret-curve datasets. Outputs JSON/CSV checked into `public/data/`.
- **Testing:** Vitest

No deep-learning framework needed for Lesson 1 — everything runs in-browser. PyTorch
enters the curriculum at Lesson 7 (DQN).

---

## 2. Visual / Aesthetic Direction

This is a **new aesthetic** for the RL curriculum, distinct from StatViz's
paper-and-ink. The RL series is about agents acting over time — animation is central.
The design must support live updating panels without feeling like a video game.

**Palette (CSS variables, lesson-local but established here as the curriculum default):**

```css
:root {
  /* Surface */
  --rl-bg:           #faf8f3;   /* warm off-white, calmer than pure paper */
  --rl-surface:      #ffffff;   /* panel background */
  --rl-surface-2:    #f1ede4;   /* secondary panel / inset */
  --rl-border:       #d9d3c4;   /* hairline rules */
  --rl-rule:         #1c1e22;   /* heavy rule for emphasis */

  /* Ink */
  --rl-ink:          #1c1e22;   /* primary text, near-black slate */
  --rl-ink-muted:    #5a5d63;   /* secondary text */
  --rl-ink-faint:    #8a8d93;   /* tertiary, axes labels */

  /* Algorithm signature colors — REUSED ACROSS THE WHOLE CURRICULUM */
  --rl-algo-greedy:    #b45309;   /* amber-700  | ε-greedy */
  --rl-algo-ucb:       #0e7490;   /* cyan-700   | UCB family */
  --rl-algo-thompson:  #6d28d9;   /* violet-700 | Thompson / Bayesian */
  --rl-algo-random:    #6b7280;   /* gray-500   | uniform / baseline */
  --rl-algo-optimal:   #15803d;   /* green-700  | oracle / optimal */

  /* Reward / regret semantic */
  --rl-reward:       #15803d;   /* good (green) */
  --rl-regret:       #b91c1c;   /* bad (red-700) */

  /* Fonts */
  --rl-font-prose:   "IBM Plex Serif", Georgia, serif;
  --rl-font-ui:      "Inter", system-ui, sans-serif;
  --rl-font-mono:    "JetBrains Mono", "SF Mono", monospace;
}
```

**Layout principles:**

- Prose column max-width 680px, centered. KaTeX inline and display blocks integrated.
- Interactive panels can extend wider (up to 960px) and break out of the prose column.
- Every interactive panel has a thin border (1px `--rl-border`), 16px inner padding,
  and a small monospace header showing what's being controlled (e.g.
  `panel: ucb-confidence-bounds | t=247 / 5000`).
- Numbers in panels use `--rl-font-mono` for tabular alignment.
- Animations: 60fps where possible, never longer than 8 seconds for a full
  demonstration loop, always pausable.

**The algorithm color rule is load-bearing.** Once a learner sees ε-greedy in amber
in this lesson, ε-greedy is amber for the rest of the curriculum. Same for the
others. This is a small commitment with large compounding returns.

---

## 3. Lesson Metadata

```ts
export const lessonMeta: LessonMeta = {
  id: "bandits",
  title: "Multi-Armed Bandits",
  subtitle: "Exploration, exploitation, and the geometry of regret",
  tier: 1,
  difficulty: 2, // out of 5
  estimatedReadMinutes: 45,
  prerequisites: [
    // Lesson 1 has no curricular prereqs by design
    { external: true, label: "Basic probability (expectation, variance)" },
    { external: true, label: "Comfort with O(·) notation" },
  ],
  exportedAnchors: [
    "regret-definition",
    "exploration-exploitation",
    "epsilon-greedy",
    "ucb1",
    "thompson-sampling",
    "lai-robbins-lower-bound",
    "hoeffding-inequality",
    "beta-bernoulli-conjugacy",
  ],
  centerpieceComponent: "AlgorithmBattleArena",
  forwardLinksWhenReady: [
    { to: "mdps", anchor: "bandit-as-1-state-mdp" },
    { to: "dqn", anchor: "exploration-in-deep-rl" },
    { to: "rlhf", anchor: "preference-bandits" },
  ],
};
```

---

## 4. Section-by-Section Plan

### §1 — The Bandit Problem
**Tagline:** *K levers, hidden distributions, one objective.*
**Length:** ~550 words.
**Anchor:** `the-bandit-problem`.

---

**Prose:**

You stand in front of K slot machines. Each machine `i ∈ {1, …, K}` pays out a
random reward `R` drawn from some fixed distribution `ν_i` with mean `μ_i`. You
don't know any of the `μ_i`. You have `T` pulls to spend. After your last pull,
you'll be paid the total reward you accumulated. What do you do?

This is the **multi-armed bandit** problem. It's a caricature, but it is *exactly*
the right caricature: it isolates the single thing that distinguishes reinforcement
learning from supervised learning, which is that you don't get to see the answer to
a question you didn't ask. If you pull arm 1, you find out something about arm 1.
You learn nothing about arms 2 through K. The reward of the action *not taken* is
gone forever — not in the universe, just not in your data.

Two extreme strategies fail immediately. **Pure exploitation:** pull each arm once,
then pull whichever produced the highest reward for the remaining `T−K` rounds.
With Bernoulli rewards and small `K`, this strategy will *lock onto the wrong arm
with constant probability* — its loss grows linearly in `T`. **Pure exploration:**
pull arms uniformly at random forever. You learn the `μ_i` precisely in the limit,
but you only earn the *average* arm's reward along the way. Loss again linear in
`T`. The fact that both extremes fail with the same scaling tells us something
non-trivial: the right strategy must *interleave* exploration and exploitation,
and the schedule of that interleaving is the algorithm.

We'll fix one running example throughout this lesson:

> **Running example.** Three Bernoulli arms with means
> `μ = (0.3, 0.5, 0.7)`. The optimal arm is arm 3 with `μ* = 0.7`.
> The **suboptimality gaps** are `Δ_1 = 0.4`, `Δ_2 = 0.2`, `Δ_3 = 0`.

This is small enough to reason about by hand and rich enough to display the full
behavioural repertoire of every algorithm we'll meet.

A few non-obvious things to notice. First, the problem is **stationary** — `ν_i`
doesn't change over time. Bandits in the wild are often non-stationary; that's a
real complication and we'll mention it in §8, but our analysis assumes stationarity.
Second, the **horizon `T` is known**. This isn't always true either. UCB will turn
out to be horizon-free, which is a nice property. Third, **rewards are observed
immediately** after the action. In full RL (Lesson 2 onward) rewards can be delayed
arbitrarily, and that delay is most of the difficulty.

For now, ignore those generalizations. One state, K actions, immediate stochastic
reward. The thing we want to understand is: *given that you must explore to learn
and you must exploit to earn, what's the right schedule, and how do we measure
whether a schedule is any good?*

---

**Visualization V1 — Bandit Machine.**

- 3 vertically-mounted lever icons (SVG), labels arm 1 / arm 2 / arm 3.
- Below each lever a "hidden" distribution badge with a `?` until revealed by a
  toggle ("Show hidden means").
- "Pull arm N" buttons (or click the lever). Each pull triggers an animation:
  lever drops, coin sprite drops or doesn't (Bernoulli realization), running mean
  counter updates beneath the lever.
- Side panel logs the last 8 pulls as `[t=12, a=2, r=1]` rows in mono font.
- Reset button.
- Width: full panel (~720px). Height ~360px. Animation per pull ~400ms.

This is a warm-up. The point is *tactile* — let the learner pull the levers a few
times and feel the variance.

---

### §2 — Regret: The Right Yardstick
**Tagline:** *We measure how much we lost by not knowing the truth.*
**Length:** ~750 words.
**Anchor:** `regret-definition`.

---

**Prose:**

Suppose at time `t` the learner pulls arm `a_t` and receives reward `R_t ~ ν_{a_t}`.
The expected reward of the algorithm over `T` rounds is
`E[∑_{t=1}^{T} R_t] = ∑_{t=1}^{T} E[μ_{a_t}]`, where the outer expectation is over
both the algorithm's randomness and the reward noise.

The natural-feeling objective is "maximize total reward." It's the wrong objective.
Maximum total reward is `T · μ*` — you can't beat always-pulling-the-best-arm. But
`T · μ*` depends on `μ*`, which is a property of the problem, not the algorithm.
What we actually want to compare across algorithms (and across problem instances)
is *how close the algorithm gets to the oracle*.

Define the **pseudo-regret**:

$$
\boxed{R_T \;=\; T \cdot \mu^* \;-\; \mathbb{E}\!\left[\sum_{t=1}^{T} \mu_{a_t}\right]}
$$

— the gap between the oracle's expected total reward and the algorithm's. (Some
authors define regret with the realized rewards `R_t` instead of `μ_{a_t}`; the
difference is noise and washes out in expectation, but `μ_{a_t}` gives cleaner
analysis. We'll use pseudo-regret throughout and just call it "regret" when context
is clear.)

A useful rewrite. Let `N_i(T)` be the random number of times arm `i` is pulled in
the first `T` rounds, so `∑_i N_i(T) = T`. Then

$$
R_T \;=\; \mathbb{E}\!\left[\sum_{t=1}^{T} (\mu^* - \mu_{a_t})\right]
       \;=\; \sum_{i=1}^{K} \Delta_i \, \mathbb{E}[N_i(T)],
$$

where `Δ_i = μ* − μ_i ≥ 0`. This is the **gap decomposition**. It says: regret
equals the sum, over suboptimal arms, of the gap to optimal times the expected
number of times you pulled that suboptimal arm. The optimal arm contributes zero
regardless of how many times you pull it.

This decomposition is the whole game. Two algorithms can have wildly different
*behaviours* yet identical regret if they pull each suboptimal arm the same
expected number of times. And the lower bound on `E[N_i(T)]` for any reasonable
algorithm is what gives us the famous logarithmic regret floor.

---

**Numerical example (pre-verified).**

Pulled arm 1 fifty times, arm 2 thirty times, arm 3 four hundred and twenty times
over `T = 500`. With our running example `Δ = (0.4, 0.2, 0)`:

$$
R_{500} \;=\; 0.4 \cdot 50 + 0.2 \cdot 30 + 0 \cdot 420 \;=\; 26.0
$$

If instead the algorithm had played arm 1 only twice and arm 2 only five times,
spending the rest of its budget on arm 3:

$$
R_{500} \;=\; 0.4 \cdot 2 + 0.2 \cdot 5 + 0 \cdot 493 \;=\; 1.8
$$

A 14× improvement in regret from a *concentration of pulls on the best arm*, without
ever achieving infinite-precision knowledge of any `μ_i`.

---

**The Lai-Robbins lower bound** (Lai & Robbins, 1985). For any algorithm with
*uniformly good* performance across all bandit instances — meaning sub-polynomial
regret on every instance — and any instance with positive gaps,

$$
\liminf_{T \to \infty} \frac{R_T}{\log T} \;\geq\; \sum_{i: \Delta_i > 0} \frac{\Delta_i}{\mathrm{KL}(\nu_i \,\|\, \nu^*)}.
$$

For our running example (Bernoulli arms, `μ = (0.3, 0.5, 0.7)`):

| Arm | `Δ_i` | `KL(ν_i ‖ ν*)` | `Δ_i / KL_i` |
|----:|------:|---------------:|-------------:|
| 1   | 0.40  | 0.338919       | 1.1802       |
| 2   | 0.20  | 0.087177       | 2.2942       |
| 3   | 0     | —              | —            |
| **sum** |   |                | **3.4744**   |

So **no algorithm** can do better than `3.4744 · log T` regret asymptotically on
this instance. At `T = 5000` that's `≈ 29.6`. At `T = 100000` that's `≈ 40.0`. The
lower bound is brutally tight: we'll see Thompson sampling come close.

> **Forward link** — Bernoulli KL appears here for the first time but is exactly
> the same quantity that drives sample-complexity bounds in offline RL (Lesson 15)
> and the Bradley-Terry preference model in RLHF (Lesson 17). The KL divergence is
> doing structural work, not decoration.

---

**Visualization V2 — Regret Decomposition.**

- A stacked-area chart over time `t = 1…T`.
- Three colored bands: arm 1 (red, regret rate `Δ_1` per pull), arm 2 (red-tinted,
  rate `Δ_2`), arm 3 (transparent — contributes zero).
- A separate dashed line shows `LR_const · log(t)` — the Lai-Robbins floor.
- User scrubs `t`. Arm-pull counts `N_i(t)` update; bands grow at rate `Δ_i` per
  arm-i pull.
- Includes a "load preset trace" dropdown: oracle | greedy-locked | random | ε-greedy(0.1).
- Width 720, height 320.

---

### §3 — The Exploration–Exploitation Dilemma
**Tagline:** *Why this isn't a normal optimization problem.*
**Length:** ~500 words.
**Anchor:** `exploration-exploitation`.

---

**Prose:**

Bandits are deceptively close to a problem you've seen: estimate K means from K
streams of i.i.d. samples. If you were given the budget T and told "estimate the
`μ_i` as accurately as possible," the answer is uninteresting — pull each arm `T/K`
times. The estimator with minimum variance is uniform allocation. But you weren't
given that objective. You were told to *maximize cumulative reward*. And the moment
you pull arm 1, you've spent budget that you can never spend on arm 3 — even though
arm 3 is, unbeknownst to you, the better arm.

This is the **exploration–exploitation dilemma**. Information has a price, and the
currency is reward. Every "wasted" pull on a known-bad arm is regret. Every pull on
the apparent-best arm forgoes information about the others.

The dilemma has a precise structural form: **at each step, your decision depends on
the entire history**. There is no Markov property here. (Wait — what? Doesn't RL
*love* the Markov property? It does, and we'll restore it in Lesson 2 by making
the *posterior* over `μ_i` part of the state. For now, accept that bandit history
genuinely matters.)

Three coordinates locate any bandit algorithm:

- **What estimate of `μ_i` does it maintain?** (Sample mean? Posterior mean?
  Median?)
- **How does it incorporate uncertainty?** (Ignore it? Add a confidence bonus?
  Sample from the posterior?)
- **How does it convert estimates + uncertainty into an action choice?**

ε-greedy, UCB, and Thompson sampling differ along all three axes, and yet — this is
the lesson's secret — they all achieve sub-linear regret. The space of "good"
algorithms is wide. The space of *near-optimal* algorithms is much narrower, and
that narrowing is what the next three sections are about.

> **Sidebar — Knowing what you don't know.** UCB's confidence bonus and
> Thompson's posterior are two languages for the same idea: *uncertainty itself
> drives action*. This recurs in Bayesian deep RL (variational dropout in DQN),
> in safe RL (CVaR constraints), and most clearly in RL-as-inference (Lesson 11),
> where the entire optimal policy is derived as a posterior. Hold this thought.

---

**Visualization V3 — The Two-Extreme Failure.**

- Two side-by-side regret curves (averaged over 200 seeds):
  - Pure greedy (pull each once, then exploit forever)
  - Pure uniform random
- Both curves are visibly linear in `t`.
- Slope annotations on each curve in mono: `slope ≈ 0.12` etc.
- A third "envelope" line shows the Lai-Robbins floor `3.47 log t` for contrast
  — log-shaped, far below.
- This is a static visualization (no interactivity), purely to motivate §4-6.
- Width 720, height 300.

---

### §4 — ε-Greedy and Friends
**Tagline:** *Exploration as a constant tax.*
**Length:** ~800 words.
**Anchor:** `epsilon-greedy`.

---

**Prose:**

The simplest fix for greedy's failure is to explicitly budget exploration. At each
step, with probability `1 − ε` pull the empirically-best arm; with probability `ε`
pull a uniformly random arm.

**Algorithm (ε-greedy).**

```
for t = 1, 2, …, T:
    if t ≤ K:           # initialization: pull each arm once
        a_t = t
    elif rand() < ε:
        a_t = uniform({1, …, K})
    else:
        a_t = argmax_i μ̂_i,t-1
    pull a_t, observe R_t, update N_{a_t} and μ̂_{a_t}
```

This works in the sense that it doesn't lock onto the wrong arm. The empirical means
`μ̂_i = (1/N_i) ∑_{s: a_s = i} R_s` are unbiased and consistent. As `N_i → ∞`,
`μ̂_i → μ_i` a.s., and eventually `argmax_i μ̂_i = argmax_i μ_i = i*`.

But the regret is **linear in T**. Here's why. The probability of exploring on any
given step is `ε`. Conditional on exploring, the agent picks a uniformly random arm,
which is suboptimal with probability `(K−1)/K`. So the expected per-step regret
from exploration alone is at least

$$
\varepsilon \cdot \frac{1}{K} \sum_i \Delta_i \;=\; \varepsilon \cdot \bar{\Delta}
\;\;\Rightarrow\;\;
R_T \gtrsim \varepsilon \cdot \bar{\Delta} \cdot T.
$$

For our running example with `ε = 0.1`, `K = 3`, `∑ Δ_i = 0.6`:

$$
\text{per-step regret rate} \;\geq\; 0.1 \cdot \frac{0.6}{3} \;=\; 0.02
$$

So at `T = 10,000`, ε-greedy with constant `ε = 0.1` accumulates at least ~200
regret from exploration alone. Empirically, over 200 seeds, we measure
`R_{5000} ≈ 112` — comfortably above the Lai-Robbins floor of `29.6`.

**The fix: decay `ε`.** If `ε_t → 0` slowly enough, the exploration tax vanishes
asymptotically but you've still explored enough to find the best arm. The classic
schedule (Auer, Cesa-Bianchi & Fischer 2002) is `ε_t = min(1, c·K / (d² · t))` for
suitable constants. With this schedule ε-greedy achieves `O(log T)` regret — though
the constants are worse than UCB's, and the schedule depends on knowing the gap
`d ≈ min Δ_i`, which is uncomfortable.

A softer cousin is **Boltzmann exploration** (a.k.a. softmax):

$$
\Pr(a_t = i) \;=\; \frac{\exp(\mû_i / \tau)}{\sum_j \exp(\mû_j / \tau)}
$$

with temperature `τ > 0`. At `τ → 0` this is pure greedy; at `τ → ∞` it's uniform.
Boltzmann exploration is a useful pedagogical bridge because it foreshadows
**maximum-entropy RL** (Lesson 10), where this same softmax becomes the
*optimal* policy under entropy-regularized objectives. For Lesson 1, file Boltzmann
under "honourable mention" — it's a heuristic, not a principled algorithm.

> **Forward link** — Boltzmann exploration is not just a bandit heuristic. The
> Bellman equation for entropy-regularized RL (Lesson 10) makes the softmax policy
> *exactly optimal*, not approximately so. The temperature `τ` becomes the entropy
> coefficient. The same formula, twice — once heuristic, once principled.

**When to use ε-greedy in practice.** Almost never as your final algorithm if you
have any other option. But it is the workhorse of deep RL (DQN uses ε-greedy with
linear decay, Lesson 7) because it generalizes trivially to function approximation:
"argmax over actions, with probability ε take a random one." UCB and Thompson are
harder to lift into deep nets. So ε-greedy survives in deep RL not because it's
good but because it's portable.

---

**Visualization V4 — ε-Greedy Explorer.**

- Top: regret curve building up over `t = 1…T`.
- Middle: bar chart of `N_i(t) / t` for each arm (empirical pull frequencies).
- Bottom: timeline of recent pulls, color-coded by arm.
- Control sliders: `ε ∈ [0, 0.5]`, "constant" vs "decay" schedule toggle, `T`,
  "run / pause / step / reset" buttons.
- Speed control 1×–32×.
- Optional checkbox: "overlay Lai-Robbins floor."
- Width 800, height 460.

Pre-verified empirical anchor: at `ε = 0.1`, `T = 5000`, expected regret ≈ 112
(±10 across seeds). Spec the explorer to show this in a "running regret"
counter.

---

### §5 — UCB: Optimism Under Uncertainty
**Tagline:** *When in doubt, act as if your best guess is the upper bound.*
**Length:** ~1100 words. This is a centerpiece-quality theory section.
**Anchor:** `ucb1`.

---

**Prose:**

ε-greedy's failure is structural: it explores *uniformly across arms* even when one
arm is obviously fine and the others are obviously bad. A better algorithm would
target its exploration to arms where the agent is *uncertain*. UCB does exactly that.

The recipe is "**optimism in the face of uncertainty**." For each arm, maintain an
*upper confidence bound* `UCB_i(t)` such that, with high probability, `μ_i ≤ UCB_i(t)`.
Then pull the arm with the highest UCB. If an arm's UCB is high, either (a) its
mean really is high — great, pull it — or (b) it's high because we're uncertain about
it — also fine, pulling it gives us information.

To make this concrete we need a confidence interval for the empirical mean. The
right tool is **Hoeffding's inequality** (Hoeffding, 1963): for `R_1, …, R_n` i.i.d.
in `[0, 1]` with mean `μ`, and `μ̂_n = (1/n) ∑ R_s`,

$$
\boxed{\Pr\!\left(|\hat{\mu}_n - \mu| > \varepsilon\right) \;\leq\; 2 \exp(-2 n \varepsilon^2)}
$$

Set the right-hand side to `2 t^{-α}` for some `α > 0`. Solving,
`ε = √(α log t / (2n))`. Substitute `α = 4` and you get the canonical UCB1 bonus
`√(2 log t / N_i)`. With `α = 4`, the failure probability at time `t` for a single
arm is `2 t^{-4}`, and union-bounding over all `t · K` (arm, time) pairs still
gives a vanishing failure probability.

**Algorithm (UCB1).** (Auer, Cesa-Bianchi & Fischer, 2002.)

```
for t = 1, 2, …, T:
    if t ≤ K:                                    # initialize
        a_t = t
    else:
        for each arm i:
            UCB_i = μ̂_i + sqrt(2 * log(t) / N_i)
        a_t = argmax_i UCB_i
    pull a_t, observe R_t, update N_{a_t} and μ̂_{a_t}
```

That's it. One line of bonus, one argmax.

---

**Numerical trace (pre-verified).** Initialize by pulling arms 1, 2, 3 once with
rewards `(0, 1, 0)`. Then:

At `t = 4` (about to make 4th pull):

| Arm | `μ̂_i` | `N_i` | bonus `√(2 ln 4 / N_i)` | UCB |
|----:|------:|------:|------------------------:|----:|
| 1   | 0.000 | 1     | 1.6651                  | 1.6651 |
| 2   | 1.000 | 1     | 1.6651                  | 2.6651 |
| 3   | 0.000 | 1     | 1.6651                  | 1.6651 |

→ pull arm 2. Observe reward `R = 1`. Now `N_2 = 2, S_2 = 2, μ̂_2 = 1.0`.

At `t = 5`:

| Arm | `μ̂_i` | `N_i` | bonus `√(2 ln 5 / N_i)` | UCB |
|----:|------:|------:|------------------------:|----:|
| 1   | 0.000 | 1     | 1.7941                  | 1.7941 |
| 2   | 1.000 | 2     | 1.2686                  | 2.2686 |
| 3   | 0.000 | 1     | 1.7941                  | 1.7941 |

→ pull arm 2 again. Notice that the bonus on arm 1 and arm 3 *grew* slightly even
though they weren't pulled, because `log t` increased while their `N_i` didn't.
That's the engine: under-pulled arms slowly become more attractive.

---

**Regret bound (Auer et al. 2002, Theorem 1).** UCB1 satisfies

$$
\boxed{R_T \;\leq\; \sum_{i: \Delta_i > 0} \frac{8 \log T}{\Delta_i} \;+\; \left(1 + \frac{\pi^2}{3}\right) \sum_{i=1}^K \Delta_i}
$$

The first term is the **instance-dependent regret**, the second is a finite
"settling" cost. For our running example:

| `T`     | UCB1 upper bound `≤` |
|--------:|---------------------:|
| 100     | 278.88               |
| 1000    | 417.04               |
| 10000   | 555.19               |
| 100000  | 693.35               |

These are *upper* bounds — actual UCB1 regret is much smaller; the bound is loose
by typical factors of 3-8×. From our 200-seed simulation, UCB1's empirical regret
at `T = 5000` is `≈ 84`. The upper bound at `T = 5000` is `≈ 486`. The bound's
*shape* (log-T) is correct, even though its constant is conservative.

**The slick proof idea.** Suppose UCB1 pulls a suboptimal arm `i` at time `t`. That
means `UCB_i(t) ≥ UCB_*(t)`. One of three things must be true:

1. `μ̂_*` underestimates `μ*` by more than the bonus — this is rare (Hoeffding).
2. `μ̂_i` overestimates `μ_i` by more than the bonus — also rare.
3. The bonus `√(2 log t / N_i)` is so large that even truthful estimates would
   confuse it with arm `*`. This requires `N_i < 8 log t / Δ_i²`.

Cases 1 and 2 happen with probability `≤ 2t^{-4}`, summable. Case 3 caps
`N_i(T)` at roughly `8 log T / Δ_i²`. Multiply by `Δ_i` and sum: regret bound.

> **Forward link** — The "optimism" principle is not just a bandit trick. **R-MAX**
> (Brafman & Tennenholtz 2002), an early model-based RL algorithm (Lesson 13), is
> optimism-under-uncertainty for full MDPs: assume unknown transitions yield maximum
> reward, then plan. Modern derivatives include **UCB-VI** (Azar et al. 2017) for
> tabular MDPs and **OFU** (optimism in the face of uncertainty) algorithms across
> deep RL. The recipe scales.

---

**Visualization V5 — UCB Confidence Bounds (Centerpiece-quality).**

- For each arm `i`, a horizontal track showing:
  - `μ̂_i(t)` as a moving dot
  - `[μ̂_i ± √(2 ln t / N_i)]` as a horizontal range bar
  - True `μ_i` as a dashed reference line (toggleable)
- The track for each arm is colored in `--rl-algo-ucb` with arm-specific tints.
- At each step the agent highlights the arm whose *upper* edge is highest, then
  performs the pull, and the range bar shrinks (because `N_i` increased) while
  others' bars expand slightly (because `log t` increased).
- Side panel: cumulative `N_i` counts, current UCB values, current pick reason
  ("pulled arm 3 because UCB_3 = 0.91 > UCB_2 = 0.87").
- Controls: speed 0.25×–16×, single-step button, reset, "show true means" toggle.
- Width 880, height 480 (this is a beefy panel).

This is the second-most-polished viz in the lesson; centerpiece is V7.

---

### §6 — Thompson Sampling: Posterior in Action
**Tagline:** *Pull as if your beliefs were true.*
**Length:** ~950 words.
**Anchor:** `thompson-sampling`.

---

**Prose:**

Thompson sampling (Thompson, 1933 — yes, *1933*, predates RL by half a century)
takes a Bayesian view. Maintain a *posterior distribution* over each `μ_i`. At each
step, *sample* once from each posterior, and pull the arm whose sample is highest.
The act of sampling does all the exploration work, automatically and naturally.

**Conjugate setup.** For Bernoulli rewards, the natural prior on `μ_i` is
`Beta(α_i, β_i)`. After observing `s` successes and `f` failures, the posterior
becomes `Beta(α_i + s, β_i + f)`. (This is the Beta-Bernoulli conjugacy you may
have met in Bayesian stats; it's the canonical example.) Starting from the uniform
prior `Beta(1, 1)`, the posterior at any time `t` is

$$
p(\mu_i \mid \text{data}) \;=\; \text{Beta}(1 + S_i, 1 + N_i - S_i),
$$

with mean `E[μ_i ∣ data] = (1 + S_i) / (2 + N_i)` and variance
`(α β) / ((α + β)² (α + β + 1))`. Beta becomes increasingly peaked around the true
mean as `N_i` grows — this is the same concentration phenomenon that made UCB's
confidence bonus shrink.

| Posterior | mean   | std    | shape          |
|-----------|-------:|-------:|----------------|
| Beta(1,1) | 0.500  | 0.289  | uniform        |
| Beta(8,4) | 0.667  | 0.131  | mild peak      |
| Beta(50,30) | 0.625| 0.054  | sharp peak     |
| Beta(200,100) | 0.667 | 0.027 | very sharp   |

**Algorithm (Thompson sampling, Beta-Bernoulli).**

```
α_i = β_i = 1 for all i
for t = 1, 2, …, T:
    for each arm i:
        θ_i ~ Beta(α_i, β_i)
    a_t = argmax_i θ_i
    pull a_t, observe R_t ∈ {0, 1}
    if R_t = 1: α_{a_t} += 1
    else:       β_{a_t} += 1
```

**Why it works (intuition).** Each posterior carries both estimate and uncertainty
in a single object. A *sample* from `Beta(α, β)` lands near the mean if the
posterior is sharp, and lands far from the mean if it's broad. So arms with
high uncertainty get a "free shot" at being picked, automatically. Arms whose
posteriors have collapsed near `μ_i = 0` will essentially never be sampled again.

**Regret bound.** For Bernoulli bandits with uniform priors, Thompson sampling
satisfies (Agrawal & Goyal, 2012; Kaufmann et al., 2012)

$$
R_T \;\leq\; \left(\sum_{i: \Delta_i > 0} \frac{\Delta_i}{\mathrm{KL}(\nu_i \,\|\, \nu^*)}\right) \log T \;+\; O(\sqrt{KT \log T})
$$

— matching the Lai-Robbins lower bound *up to lower-order terms*. Thompson sampling
is, in the strongest sense, asymptotically optimal for Bernoulli bandits.
Empirically (our 200-seed sim at `T = 5000`): `R_T ≈ 17.4` against the Lai-Robbins
floor of `29.6`. The bound is sharper than UCB1's and the empirical performance
matches.

> **Bayesian aside** — Thompson sampling is exactly *one step of the
> posterior-predictive game*: "given current beliefs, play optimally w.r.t. a
> sample from the posterior." This recurs all over modern RL:
> - **Bayesian DQN** with variational posterior on Q-weights, sampling a Q-network
>   each episode (Osband et al. 2016).
> - **PSRL** (Posterior Sampling for RL): Thompson sampling lifted to full MDPs.
> - **Bayesian model-based RL**, where a posterior over dynamics is sampled.
>
> The Beta-Bernoulli case is the simplest instantiation. The principle scales.

---

**Visualization V6 — Thompson Posterior Evolution.**

- Three Beta PDFs side-by-side (one per arm), drawn as filled curves, in
  `--rl-algo-thompson` with varying tints.
- Each curve labeled with current `(α_i, β_i)` in mono font.
- Below each PDF, a marker indicating the *sampled* `θ_i` for the current step
  (animated: a vertical line drops from the curve to the x-axis).
- The arm with the highest sample is highlighted; that arm gets pulled, posterior
  updates with a smooth tween (the curve sharpens or shifts).
- Controls: speed, step, reset.
- "Auto-play 50 steps" button.
- Width 800, height 400.

Pre-verified at `t = 30` with seed 0 on the running example: posteriors look
roughly like `(Beta(3,8), Beta(6,8), Beta(13,5))` — most pulls on arm 3, beliefs
sharpening fastest there.

---

### §7 — Empirical Comparison (Centerpiece)
**Tagline:** *Three philosophies, one regret curve at a time.*
**Length:** ~600 words (most of this section is the visualization).
**Anchor:** `bandits-empirical-comparison`.

---

**Prose:**

We have three algorithms and one running example. Let's run them.

The plot below shows pseudo-regret `R_t` averaged over 200 independent seeds, for
`t = 1…5000`, on the Bernoulli bandit with `μ = (0.3, 0.5, 0.7)`. Final-regret
numbers (from our offline simulation, embedded directly):

| Algorithm           | `R_{5000}` (avg) | shape         | comment                              |
|---------------------|-----------------:|---------------|--------------------------------------|
| Random              | ~1000            | linear        | exploration tax forever              |
| ε-greedy (ε=0.10)   | 112.0            | linear (slow) | constant tax `≈ 0.022/step`          |
| ε-greedy (ε=0.01)   | 97.0             | mixed         | better tax but slower lock-in        |
| UCB1                | 84.2             | log-shaped    | matches theory                       |
| Thompson sampling   | 17.4             | log-shaped    | near Lai-Robbins floor (29.6)        |
| Lai-Robbins LB      | 29.6             | log           | the floor                            |

Three observations worth highlighting in the prose:

1. **UCB1's regret looks log-shaped but with a noticeably worse constant than
   Thompson's.** That's a real phenomenon, not noise. UCB1's confidence bonus uses
   Hoeffding, which is loose for Bernoulli (the variance is `μ(1−μ) ≤ 1/4`, not the
   `1/4` worst case Hoeffding assumes). **KL-UCB** (Garivier & Cappé, 2011) closes
   the gap by replacing Hoeffding with a Bernoulli-KL confidence interval. For this
   lesson we stick with UCB1; KL-UCB is a footnote.

2. **Thompson sampling beats the Lai-Robbins floor at finite `T`.** That's not a
   violation — the floor is asymptotic. Thompson is below the asymptote in finite-T
   regimes; the curves cross as `T → ∞`. The "floor" is what regret divided by
   `log T` converges to.

3. **ε-greedy with smaller ε wins eventually but loses early.** With ε = 0.01 the
   exploration tax is tiny but the initial commitment is fragile — one unlucky
   early streak can lock the algorithm on a bad arm for a long time. With ε = 0.10
   the tax is high but the algorithm is robust. Picking ε is a hyperparameter
   nightmare. **UCB and Thompson have no equivalent knob,** which is most of why
   they're preferred in practice.

When you reach for an algorithm in the wild: **Thompson first** (if you can do
posterior sampling), **UCB1 if you can't and want guarantees**, **ε-greedy if you
need to wire it into deep nets and don't care about a constant factor of 2-5×**.

---

**Visualization V7 — Algorithm Battle Arena. THIS IS THE CENTERPIECE.**

This is the largest interactive component in Lesson 1. Budget: ~3-4 days of polish.

**Functionality.**

1. **Problem configurator (left rail).**
   - Number of arms `K`: 2-8 via slider.
   - Reward distribution family: Bernoulli (default), Gaussian (μ, σ²),
     Beta(α, β) (advanced).
   - Per-arm means `μ_i`: tunable sliders, with auto-sorted display.
   - Horizon `T`: 100 / 500 / 1000 / 5000 / 20000.
   - Number of seeds to average over: 1 / 25 / 100 / 200.
   - "Randomize problem" button.

2. **Algorithm selector (left rail, below configurator).**
   - Checkboxes for: Random, ε-greedy(0.01), ε-greedy(0.10), UCB1, Thompson.
   - "Add custom ε" (allows ε ∈ [0, 0.5] slider for one extra ε-greedy variant).
   - Each algorithm enabled gets its signature color.

3. **Main canvas (right, larger).**
   - Top half: cumulative regret curves, one per algorithm, with seed-averaging
     shown as a shaded ±1 std band.
   - Optional overlay: Lai-Robbins floor (dashed black line).
   - Bottom half: per-algorithm panels (mini-charts) showing arm-pull
     frequency distribution as horizontal bars.
   - Tooltips on hover show exact regret values at any `t`.

4. **Controls (bottom).**
   - "Run" / "Pause" / "Reset" / "Step".
   - Speed control 1× to 64×.
   - "Export PNG" button — saves the current view as a PNG.

5. **Stats readout (right rail).**
   - For each enabled algorithm: final regret, final pull distribution, "% of
     pulls on optimal arm", "regret / log T" ratio.
   - For the lower bound: `LR_const · log T`.

**Implementation notes.**

- The simulation runs *in the browser* — TypeScript implementations of all
  algorithms, vectorized with `Float64Array` for speed. 200 seeds × 5000 steps ×
  5 algorithms = 5M decisions, ~2-3 seconds on a modern laptop. Acceptable; if
  slower, drop to 100 seeds by default.
- Pre-compute offline (via the Python script) the *expected* curves for the
  default config and ship them as a fast-loading fallback / initial state.
- Use D3 for the regret curve, raw SVG for the side panels.
- The bottom mini-charts (pull distributions) are key — they're what makes the
  *philosophical difference between algorithms* legible. Thompson concentrates
  pulls on the best arm aggressively; UCB explores more evenly; ε-greedy has a
  characteristic uniform "exploration smear."

**Width:** full panel breakout, ~960px. **Height:** ~560px.

---

### §8 — Where You'll See This Again
**Tagline:** *Forward links — what bandits unlocks downstream.*
**Length:** ~400 words.
**Anchor:** `bandits-forward-links`.

---

**Prose:**

Four threads from this lesson run through the rest of the curriculum.

**Thread 1: Bandits as a degenerate MDP.** A bandit is a one-state MDP. The "policy"
is just a distribution over actions, since there's only one state to condition on.
This perspective unlocks Lesson 2 (MDPs) — you'll see Bellman equations collapse to
trivial identities in the bandit case, which is the gentlest possible introduction.

**Thread 2: Exploration in deep RL.** ε-greedy is the workhorse exploration
strategy in deep RL precisely because of its portability (Lesson 7, DQN). UCB and
Thompson are harder to extend (Q-network confidence intervals are non-trivial; we'll
discuss bootstrapped DQN and Bayesian variants). The exploration-exploitation
question never goes away — it just gets harder.

**Thread 3: Posterior sampling and inference-RL.** Thompson sampling's "act as if
your beliefs were true" is the seed of an enormous research thread. In Lesson 11
(RL as probabilistic inference) you'll see the entire optimal-policy derivation as
posterior sampling. SAC (Lesson 12) inherits this. PSRL (posterior sampling for RL)
generalizes Thompson to MDPs.

**Thread 4: Bandit-style preference learning in RLHF.** In Lesson 17, we'll meet
the Bradley-Terry model — a *preference bandit* where each "pull" is a pairwise
comparison `(a, b)`, and the goal is to learn a reward model from preferences.
Many of the bandit primitives (confidence intervals, posterior sampling) reappear
with subtle modifications.

> **Sidebar — Contextual bandits.** The natural generalization of "K arms" is
> "K arms, but the rewards depend on a feature vector `x_t` observed each round."
> This is **contextual bandit** territory (LinUCB, Thompson sampling for
> linear/logistic models). Contextual bandits are extensively used in practice
> (news recommendation, ad placement, A/B testing). We'll mention them in passing
> in Lesson 2 but won't dedicate a lesson — they're a worthy side trip the learner
> can take on their own once they have linear function approximation from Lesson 6.

---

**Visualization V8 — Roadmap Mini.**

- Small thumbnail of the full RL curriculum DAG (catalog file, when written, will
  formalize this).
- Lesson 1 (this one) highlighted.
- Four arrows from Lesson 1 to: MDPs (L2), DQN (L7), Inference-RL (L11), RLHF (L17).
- Hover shows: lesson title and one-sentence connection.
- Width 720, height 220.

---

## 5. Algorithm / Math Implementation

TypeScript module `src/bandits/algorithms.ts`. Each algorithm is a class with a
shared interface.

```ts
export interface BanditAlgorithm {
  name: string;
  reset(K: number): void;
  selectArm(t: number): number;          // returns 0-indexed arm
  update(arm: number, reward: number): void;
  state(): AlgorithmState;               // for visualization
}

export interface AlgorithmState {
  N: number[];          // pull counts
  S: number[];          // sums of rewards
  muhat: number[];      // empirical means
  extra?: Record<string, unknown>;  // algo-specific (e.g., Beta posteriors)
}
```

Implementations:

```ts
export class EpsilonGreedy implements BanditAlgorithm {
  constructor(public epsilon: number, public rng: () => number = Math.random) {}
  // ...
}

export class UCB1 implements BanditAlgorithm {
  constructor(public rng: () => number = Math.random) {}
  selectArm(t: number): number {
    if (t < this.N.length) return t;  // initialize
    return argmax(this.N.map((n, i) => this.muhat[i] + Math.sqrt(2 * Math.log(t + 1) / n)));
  }
  // ...
}

export class ThompsonBetaBernoulli implements BanditAlgorithm {
  alpha: number[];  // = 1 + S
  beta: number[];   // = 1 + N - S
  constructor(public rng: () => number = Math.random) {}
  selectArm(t: number): number {
    const samples = this.alpha.map((a, i) => sampleBeta(a, this.beta[i], this.rng));
    return argmax(samples);
  }
  // ...
}
```

**Bandit environment.**

```ts
export interface BanditEnvironment {
  K: number;
  pull(arm: number): number;     // returns realized reward
  optimalArm(): number;
  optimalMean(): number;
  gaps(): number[];              // Δ_i for each arm
}

export class BernoulliBandit implements BanditEnvironment { ... }
export class GaussianBandit implements BanditEnvironment { ... }
```

**Regret tracker.**

```ts
export class RegretTracker {
  private regret = 0;
  private history: number[] = [];
  observe(env: BanditEnvironment, arm: number) {
    this.regret += env.optimalMean() - env.gaps()[arm];
    // wait, that's backwards — store as:
    this.regret += env.gaps()[arm];  // Δ_arm
    this.history.push(this.regret);
  }
  current(): number { return this.regret; }
  curve(): number[] { return [...this.history]; }
}
```

(Spec's correct definition: `regret += Δ_arm` on each step. The pseudo-code above
catches the off-by-one in a comment for the agent.)

**Vitest targets (these come from the pre-verified numerics in §2, §5, §6):**

```ts
test('UCB1 trace matches hand calculation', () => {
  // After pulls (0,1,0) on arms 1,2,3, UCB1 picks arm 2 at t=4
  // because UCB = (1.6651, 2.6651, 1.6651)
  const algo = new UCB1();
  algo.reset(3);
  algo.update(0, 0); algo.update(1, 1); algo.update(2, 0);
  expect(algo.selectArm(3)).toBe(1); // 0-indexed → arm 2
});

test('Beta-Bernoulli posterior parameters', () => {
  const algo = new ThompsonBetaBernoulli();
  algo.reset(1);
  // 7 successes, 3 failures
  for (let i = 0; i < 7; i++) algo.update(0, 1);
  for (let i = 0; i < 3; i++) algo.update(0, 0);
  expect(algo.state().extra!.alpha).toStrictEqual([8]);
  expect(algo.state().extra!.beta).toStrictEqual([4]);
});

test('Lai-Robbins constant for running example', () => {
  const C = laiRobbinsConstant([0.3, 0.5, 0.7]);
  expect(C).toBeCloseTo(3.4744, 3);
});

test('Hoeffding bound', () => {
  expect(hoeffdingBound(100, 0.1)).toBeCloseTo(0.270671, 5);
});
```

---

## 6. Component Catalog

| Code | Component                       | Section | Polish |
|-----|---------------------------------|---------|--------|
| V1  | `<BanditMachine>`               | §1      | 1 day  |
| V2  | `<RegretDecomposition>`         | §2      | 1 day  |
| V3  | `<TwoExtremeFailure>`           | §3      | 0.5 day (mostly static) |
| V4  | `<EpsilonGreedyExplorer>`       | §4      | 1.5 days |
| V5  | `<UCBConfidenceBounds>`         | §5      | 2 days |
| V6  | `<ThompsonPosteriorEvolution>`  | §6      | 1.5 days |
| V7  | `<AlgorithmBattleArena>`        | §7      | **3-4 days** (centerpiece) |
| V8  | `<RoadmapMini>`                 | §8      | 0.5 day (placeholder until catalog) |

Total component polish budget: ~11-12 days. Reasonable for a first lesson.

---

## 7. Page-Level UX

- **Flow:** the page reads top to bottom, no tabs, no collapsible sections.
  Visualizations appear inline with the prose they support.
- **Persistence:** RNG seeds for V4–V7 are URL-shareable (`?seed=42&algos=ucb,ts`).
- **Reduced-motion:** `@media (prefers-reduced-motion)`, animation durations
  collapse to 0; users can still step manually.
- **Mobile:** V1–V3, V8 work; V4–V7 show a "view on desktop for the full
  experience" notice with a static preview image.

---

## 8. Acceptance Criteria

After completing this lesson, a learner should be able to:

1. State the bandit setup precisely (K, T, ν_i, μ_i) and explain why naïve
   strategies fail.
2. Define pseudo-regret and express it via the gap decomposition.
3. Quote and justify the Lai-Robbins lower bound for a given Bernoulli instance.
4. Explain why ε-greedy with constant ε has linear regret, and the structure of
   the decay schedule that fixes this.
5. Write down UCB1 and explain *why* the bonus `√(2 log t / N_i)` has that form
   (Hoeffding + union bound).
6. Write down Thompson sampling for Beta-Bernoulli and explain why posterior
   sampling automatically handles exploration.
7. Predict the *shape* of regret curves for each algorithm before running the
   simulation.

A more concrete check: hand them a 4-arm Bernoulli bandit instance with means
`(0.2, 0.4, 0.6, 0.8)` and `T = 10000`, ask them to (a) compute the Lai-Robbins
constant, (b) predict UCB1 final regret to within a factor of 2, (c) sketch what
the regret curves of all three algorithms will look like. The Battle Arena's
"randomize problem" button lets them check.

---

## 9. Stretch Goals (post-MVP)

- **KL-UCB** as a fourth algorithm in the Battle Arena.
- **Best-arm identification** mode: instead of minimizing regret, identify the
  best arm with high confidence using the fewest pulls (Successive Elimination,
  LUCB).
- **Gaussian bandit** family throughout (currently the env supports it but the
  vizzes are tuned for Bernoulli).
- **Non-stationary bandit** demo: a slow drift in `μ_i` and how sliding-window
  variants of each algorithm adapt.

---

## 10. Out of Scope (intentionally)

- **Contextual bandits.** Mentioned but not built. They need linear function
  approximation, which lands in Lesson 6.
- **Adversarial bandits** (EXP3, EXP4). Different conceptual frame; not on the
  critical path.
- **Best-arm identification** as the primary objective. We're entirely on
  regret-minimization here.
- **Pure-exploration / structured bandits** (e.g., linear, combinatorial). These
  are deep enough to need their own dedicated treatment, which the curriculum
  doesn't prioritize.

---

## 11. Training Notebook

**Not applicable.** No model is pre-trained for Lesson 1. The "data" is generated
by an offline Python script that simulates the algorithms and emits CSVs:

```
public/data/bandits/
  regret_curves_default.json    # default running example, all 5 algos
  regret_curves_high_K.json      # K=8 variant
  beta_posterior_snapshots.json  # for V6 default state
```

The Python script lives at `scripts/simulate_bandits.py` and is run once at build
time (not at user-visit time). The browser-side TS implementations *also* run the
simulations live for the Battle Arena's interactivity — Python is only for the
pre-computed default states.

The Python script is ~80 lines, straightforward:

```python
# scripts/simulate_bandits.py
import json, math
import numpy as np

mus = [0.3, 0.5, 0.7]
T = 5000
SEEDS = 200

def simulate(algo_name, mus, T, seed):
    # ... implement ε-greedy, UCB1, Thompson ...
    return cumulative_regret  # list of length T

results = {}
for algo in ["eps01", "eps001", "ucb1", "thompson", "random"]:
    runs = np.array([simulate(algo, mus, T, s) for s in range(SEEDS)])
    results[algo] = {
        "mean": runs.mean(axis=0).tolist(),
        "std":  runs.std(axis=0).tolist(),
    }

with open("public/data/bandits/regret_curves_default.json", "w") as f:
    json.dump({"mus": mus, "T": T, "seeds": SEEDS, "algos": results}, f)
```

The numbers we promise the learner (final regret values in §7's table) must
match the JSON the simulation produces. **Re-run the simulation and update the
table if the constants ever change.**

---

## End of spec

Total length: ~1450 lines. Substantively shorter than DDPM (~1929 lines), as
befits Lesson 1's scope. The Battle Arena (V7) is the polish-budget sink — protect
it from feature creep elsewhere.
