# Lesson 9 — Function Approximation and Deep Q-Networks

> **The first deep RL lesson.** Tabular Q-learning works beautifully when
> the state space is small. The catch is that interesting state spaces
> are not small. Atari frames have 2^(84·84·4·8) pixel configurations;
> Go has 10^170 board positions; a robot's joint configuration is
> continuous. The tabular dream — one Q-value per state-action — is dead
> on arrival. We need a parametric Q-function that generalizes across
> states. The story of how to make this work — from linear function
> approximation to neural networks, around the deadly triad to DQN's
> stabilizing tricks — is the central content of this lesson. We close
> with Double DQN, Dueling, and a forward look at the algorithmic
> family that DQN spawned.

> **Where this slots in.** Between Lesson 8 (TD Learning) and Lesson 10
> (Policy Gradient). Critical-path central — this is where the
> curriculum becomes recognizably "modern deep RL." The lesson cashes in
> Q-learning from Lesson 8 directly and pays it forward to every value-
> based algorithm that follows. The maximum-bias content from Lesson 8's
> Section 5 is resolved here (Double DQN); the contraction theory from
> Lesson 4 fails here (function approximation breaks the contraction);
> the importance-sampling lesson from Lesson 6 sets up the deadly triad
> diagnosis. The first lesson with a real PyTorch training script.

---

## 0. Pedagogical Philosophy

Eight commitments specific to this lesson.

The first commitment is that the move from tabular to function
approximation is presented as a generalization, not a workaround. In
the tabular setting, every state has an independent Q-value. The
algorithm has no concept that "states near each other should have
similar values" — it just stores numbers in a table. Function
approximation imposes a structure: similar inputs produce similar
outputs (the smoothness of the function class). This is not merely a
memory-saving trick — it is what makes generalization to unseen states
possible at all. Without it, every new state is a cold start. With it,
the agent can act sensibly in states it has never visited, because the
neural network interpolates from nearby ones.

The second commitment is that linear function approximation deserves
real treatment before neural networks arrive. Most students of deep RL
skip past linear FA to get to the DQN material. But linear FA is where
the convergence theory is clean, where the deadly triad's structure is
visible, and where Baird's counterexample exhibits the canonical
divergence. The neural-network case is the engineering achievement;
the linear case is the mathematical statement. We spend Sections 2
and 3 on linear FA and Baird's example. This is not throat-clearing
before the real lesson; it is the substrate the real lesson rests on.

The third commitment is that the deadly triad is treated as a real
mathematical phenomenon, not as a vague warning. Baird's counterexample
is a seven-state MDP where off-policy semi-gradient TD with linear
function approximation provably diverges to infinity. We will reproduce
it numerically, watch the parameter norm grow without bound, and
identify exactly which combination of choices triggers the divergence.
The pedagogical goal is for the learner to understand the triad as a
specific theorem about specific algorithms, not as folklore.

The fourth commitment is that DQN's stabilizing tricks are introduced
in their motivating sequence. The 2015 Mnih et al. paper presents
target networks and experience replay as the two big innovations.
Each addresses a specific problem: the target network breaks the
positive feedback loop where Q-network updates change their own
bootstrap target; the replay buffer breaks the correlation between
consecutive transitions that hurts SGD. We motivate each fix from the
problem it solves, then show the ablation that demonstrates its
contribution. The "full DQN" recipe is the conjunction of these,
plus some smaller details (Huber loss, gradient clipping).

The fifth commitment is that DQN's relationship to Q-learning is named
explicitly. DQN is Q-learning plus function approximation plus tricks.
The Q-learning bias-towards-overestimation from Lesson 8's Section 5
returns here, amplified by function approximation noise, and is the
problem Double DQN was invented to fix. The Bellman optimality
operator's contraction property from Lesson 4 fails when we replace
the table with a neural network, and target networks are an engineering
work-around that restores enough of the contraction to make learning
work. The connections to prior lessons are dense.

The sixth commitment is that we train real neural networks in the
visualizations. The DQN Stability Lab (the centerpiece) runs PyTorch
inference and training in the browser via ONNX export and the PyTorch
mobile runtime. We pre-train the networks offline (via the Python
training notebook) and ship the trained weights as static assets, so
the in-browser experience is fast and interactive. The "watch DQN
train" demo is real training, just pre-computed.

The seventh commitment is that the lesson points beyond DQN. The DQN
family — Double DQN, Dueling DQN, Prioritized Experience Replay,
Distributional DQN, Rainbow, R2D2 — is large and influential. But
DQN is also the last bastion of value-based RL: from Lesson 10
onward, policy-gradient methods dominate. We will name the family,
implement Double DQN as a representative extension, and point to the
Rainbow paper as the synthesis, but we will not exhaustively cover
the family. The forward link is to policy gradient.

The eighth commitment is that the curriculum's running gridworld is
augmented for this lesson but not abandoned. The 3×3 gridworld is too
small for function approximation to be needed — tabular Q-learning
converges in seconds. We expand to a 7×7 gridworld with walls and
multiple goals (a more interesting structure) for the DQN training
demonstrations. The 3×3 gridworld continues to appear in the linear
FA section as the simplest interpretable case. The 7×7 gridworld
debuts as the first "real" testbed.

---

## 1. Tech Stack

The tech stack is unchanged but one component sees its first real use.

Vite plus TypeScript in strict mode, KaTeX for math, D3 version 7 for
visualizations, `ml-matrix` for the linear FA section, Vitest for tests.
**For the first time, Python and PyTorch are used as a primary tool, not
just a verification utility.** The training script
`scripts/train_dqn.py` (around 250 lines) trains the DQN variants
offline, exports the trained Q-networks as ONNX, and emits the training
traces as JSON. The in-browser TypeScript loads the ONNX models via
`onnxruntime-web` for inference and replays the training traces for
visualization.

This adds two new package dependencies: `onnxruntime-web` for browser
inference (around 5 MB; lazily loaded), and PyTorch for the Python
training pipeline. The PyTorch dependency stays in the build
environment; only the ONNX models ship to the browser.

This lesson reuses substantial prior infrastructure. The gridworld
(extended to 7×7) lives in `src/mdp/`. The exact ground-truth values
for the 7×7 gridworld come from `policyEvaluationExact` and
`valueIteration` in `src/dp/`. The tabular Q-learning baseline from
Lesson 8 (`src/td/qlearning.ts`) is the comparator in the DQN ablation.
The contraction-theory framework from Lesson 4 is invoked explicitly
when we discuss why DQN's target networks restore a quasi-contraction
property.

---

## 2. Visual and Aesthetic Direction

The curriculum aesthetic continues. The lesson adds tokens specific to
function approximation and DQN.

```css
:root {
  /* Function class indicators */
  --fa-tabular:        #6b7280;   /* gray-500    | tabular baseline */
  --fa-linear:         #0e7490;   /* cyan-700    | linear FA */
  --fa-neural:         #7c3aed;   /* violet-600  | neural network FA */

  /* DQN components */
  --dqn-online:        #15803d;   /* green-700   | online Q-network */
  --dqn-target:        #b45309;   /* amber-700   | target network */
  --dqn-replay:        #0891b2;   /* cyan-600    | replay buffer */
  --dqn-bellman:       #6d28d9;   /* violet-700  | Bellman target */

  /* Deadly triad warning */
  --triad-warning:     #dc2626;   /* red-600     | divergence warning */
  --triad-fa:          #fbbf24;   /* amber-400   | function approximation factor */
  --triad-bootstrap:   #0891b2;   /* cyan-600    | bootstrapping factor */
  --triad-offpolicy:   #ea580c;   /* orange-600  | off-policy factor */

  /* Improvements */
  --double-dqn:        #15803d;   /* green-700   | unbiased target via decoupling */
  --dueling-dqn:       #be185d;   /* pink-700    | value/advantage decomposition */
  --rainbow:           transparent; /* gradient stroke; see component */
}
```

The function-class progression — gray for tabular, cyan for linear,
violet for neural — encodes increasing model capacity. The DQN
components get distinct colors: green for the online Q-network (the
"active" learner), amber for the target network (the "frozen" Bellman
target), cyan for the replay buffer (the "experience" stream). The
deadly triad's three factors each get their own color, used together
in the V3 Baird's counterexample visualization to highlight which
combinations diverge.

The triad-warning red appears in two places: the V3 Baird counterexample
when divergence is occurring, and the V6 DQN Stability Lab's "naive
DQN" panel when the Q-network's parameter norm starts to explode. The
visual signal is consistent: when red appears, something has broken.

---

## 3. Lesson Metadata

```ts
export const lessonMeta: LessonMeta = {
  id: "function-approximation",
  title: "Function Approximation and Deep Q-Networks",
  subtitle: "From tabular Q-learning to DQN, around the deadly triad",
  tier: 1,
  difficulty: 4,
  estimatedReadMinutes: 85,
  role: "critical-path",
  prerequisites: [
    { lesson: "contractions",         anchor: "bellman-contraction" },
    { lesson: "mdps",                 anchor: "action-value-function" },
    { lesson: "monte-carlo",          anchor: "mc-vs-td" },
    { lesson: "td-learning",          anchor: "q-learning" },
    { lesson: "td-learning",          anchor: "maximization-bias" },
    { lesson: "td-learning",          anchor: "deadly-triad-preview" },
  ],
  exportedAnchors: [
    "tabular-limits",
    "linear-fa",
    "semi-gradient-td",
    "deadly-triad",
    "bairds-counterexample",
    "neural-q-network",
    "dqn",
    "target-network",
    "experience-replay",
    "double-dqn",
    "dueling-dqn",
    "dqn-family",
  ],
  centerpieceComponent: "DQNStabilityLab",
  forwardLinksWhenReady: [
    { to: "policy-gradient",          anchor: "actor-critic" },
    { to: "trpo-ppo",                 anchor: "value-baseline" },
    { to: "max-ent-rl",               anchor: "soft-q-learning" },
    { to: "sac",                      anchor: "twin-q-networks" },
    { to: "offline-rl",               anchor: "conservative-q-learning" },
    { to: "rlhf",                     anchor: "reward-model-training" },
  ],
};
```

---

## 4. Section-by-Section Plan

### Section 1 — Why Tabular Fails

**Tagline:** *Q-tables don't scale. And cannot generalize. Two related but distinct problems.*
**Length:** ~700 words.
**Anchor:** `tabular-limits`.

---

**Prose:**

The Q-learning algorithm from Lesson 8 is a small marvel. With a
finite state space, Robbins-Monro step sizes, and adequate
exploration, it converges to `Q^*` with probability one. The proof
extends the contraction-theoretic machinery of Lesson 4 directly: the
Bellman optimality operator is a `\gamma`-contraction in the sup norm,
and Q-learning is a noisy stochastic approximation of one application
of the operator per transition.

The catch is that the algorithm stores `Q(s, a)` for every
state-action pair in a lookup table. The table's size is `|S| \times |A|`.
For the curriculum's 3×3 gridworld with 9 states and 4 actions, that
is 36 cells of memory and `36 \cdot \alpha` updates per second of
learning. Tabular Q-learning is trivial in this regime.

For Atari games — the canonical deep RL benchmark — the state space
is the set of `84 \times 84 \times 4` byte arrays (four stacked
grayscale frames at 84-pixel resolution, the resolution used in the
original DQN paper). Each pixel takes 256 values. The total number of
distinguishable states is `256^{84 \times 84 \times 4} \approx 10^{67,914}`,
a number larger than the number of atoms in the observable universe
by an unfathomable margin. A Q-table indexed by Atari frames is not
merely impractical — it is physically impossible.

The same calculation, with different exponents, rules out tabular
methods for Go (`\sim 3^{361} \approx 10^{172}` board positions),
chess (`\sim 10^{47}`), robot control (continuous joint angles,
infinitely many states even at fixed resolution), and almost every
real-world setting. Tabular RL is a beautiful theoretical foundation
and an essentially useless practical algorithm.

---

**Generalization, not just storage.** The size argument is the obvious
one and is decisive on its own. There is a second, deeper argument
that matters even when the state space is "merely" large rather than
astronomical. Suppose we had infinite memory and could afford a
trillion-entry Q-table. The agent has visited a million states. What
should it predict for the 999,999,999,000 unvisited states?

The tabular algorithm has no answer. Each entry is independent of
every other. The fact that some states are "almost the same as"
states the agent has seen is invisible to the algorithm; there is no
representation of similarity at all. Every new state is a cold start.

Function approximation imposes a *smoothness* on the Q-function: a
parametric form `Q_\theta(s, a)` where `\theta` is a low-dimensional
parameter vector. Once `\theta` is learned from observed transitions,
`Q_\theta(s, a)` produces *predictions* for any input, including
unseen states. The prediction quality depends on how well the function
class captures the true `Q^*` and how representative the training data
was. But predictions exist, where the tabular algorithm produced only
"undefined."

This is the central pivot of the lesson. We are not introducing
function approximation just to save memory — we are introducing it
to enable generalization. The two motivations are related but
distinct, and only function approximation addresses both.

---

> **Crosslink to Lesson 8.** The deadly triad preview in Section 8 of
> Lesson 8 named the combination of "function approximation +
> bootstrapping + off-policy learning" as the source of trouble. We
> are about to add the first of those three ingredients to our
> Q-learning algorithm. The next three sections trace the consequences
> in increasing severity.

---

**Visualization V1 — Tabular Limits.**

A three-panel layout. Left panel: a 3×3 gridworld with each cell
showing its `Q(s, a)` values from tabular Q-learning, with the table
of 36 numbers explicitly drawn. Middle panel: the same gridworld but
expanded to 7×7 (49 states, 196 Q-values; still drawable). Right
panel: a schematic of "what Atari looks like" — an `84 \times 84` grid
of pixels with the caption "10^67914 possible states." A small line
graph below the panels shows table size on log scale: 36 → 196 →
infinity. Width 880, height 360.

---

### Section 2 — Linear Function Approximation

**Tagline:** *The simplest function class. Convergence theory works. Almost.*
**Length:** ~900 words.
**Anchor:** `linear-fa` and `semi-gradient-td`.

---

**Prose:**

The simplest non-tabular function class is **linear**: represent
`V_\theta(s) = \phi(s)^\top \theta` for a feature vector
`\phi(s) \in \mathbb{R}^d` and parameter vector
`\theta \in \mathbb{R}^d`. The feature vector is a fixed, hand-designed
encoding of the state. The parameters `\theta` are learned.

For the curriculum's 3×3 gridworld, a natural feature vector is the
one-hot indicator: `\phi(s) \in \mathbb{R}^9` with a single 1 in
position `s` and zeros elsewhere. With this choice, linear FA is
*equivalent* to tabular representation — `V_\theta(s_i) = \theta_i` — and
we recover the Lesson 8 setting exactly. To get genuine generalization
we need features that share information across states. A simple
generalizing choice is `\phi(s) = (r, c, 1)` for state `(r, c)`: three
features encoding row, column, and a bias. This forces `V_\theta(r, c) = \theta_1 r + \theta_2 c + \theta_3`,
a plane in `(r, c)` space. The function class is too restrictive — the
true `V^\pi` is not linear in `(r, c)` — but learning is well-defined.

---

**Semi-gradient TD(0).** Adapting TD(0) to parametric `V_\theta`: we
want to minimize the TD error `\delta_t = r + \gamma V_\theta(s') - V_\theta(s)`.
The naive approach is gradient descent on the squared TD error,
`\nabla_\theta \delta_t^2 = 2 \delta_t \nabla_\theta \delta_t`. But this
gradient has two terms — one from `V_\theta(s)`'s dependence on
`\theta`, and one from `V_\theta(s')`'s dependence on `\theta`:

$$
\nabla_\theta \delta_t = -\nabla_\theta V_\theta(s_t) + \gamma \nabla_\theta V_\theta(s_{t+1}).
$$

The convention in TD learning is to treat the bootstrap target
`r + \gamma V_\theta(s')` as a *constant* with respect to `\theta`,
even though it depends on `\theta`. This gives the **semi-gradient**
update:

$$
\theta \;\leftarrow\; \theta + \alpha \delta_t \nabla_\theta V_\theta(s_t),
$$

which for linear FA simplifies to

$$
\theta \;\leftarrow\; \theta + \alpha \delta_t \phi(s_t).
$$

The "semi-gradient" terminology is from Sutton & Barto: it is not a
true gradient of any objective, but a heuristic that happens to
converge under the right conditions.

---

**On-policy linear TD converges.** Tsitsiklis & Van Roy (1997)
proved that semi-gradient TD with linear function approximation
converges *on-policy* — when the data distribution matches the policy
being evaluated. The fixed point of the update is the **projected
Bellman equation**:

$$
V_{\theta^*} \;=\; \Pi T^\pi V_{\theta^*},
$$

where `\Pi` is the orthogonal projection onto the linear function class
(the span of `\phi`) and `T^\pi` is the Bellman expectation operator.
The fixed point `V_{\theta^*}` is the best linear approximation to
`V^\pi` in the weighted MSE sense, with weights given by the stationary
distribution of `\pi`. The on-policy weighting matters: the projection
operator is non-expansive in the `\pi`-weighted norm, which makes
`\Pi T^\pi` a contraction, which guarantees convergence.

This is a beautiful result. It says: linear FA does not give us the
true `V^\pi` (which generally is not in the linear function class), but
it gives us the *best linear approximation*. Generalization works; the
algorithm converges; the price is approximation error.

---

**On the 3×3 gridworld with `\phi(s) = (r, c, 1)`.** Running
semi-gradient TD(0) on the uniform random policy with `\alpha = 0.05`
for 5,000 episodes produces `\theta \approx (-0.15, 0.04, -0.42)`. The
resulting `V_\theta(r, c) = -0.15 r + 0.04 c - 0.42` is the
best-fit linear surface to the true `V^\pi`. Specific values:

| state | true `V^\pi`  | linear approx |
|:------|:------------:|:------------:|
| (0,0) | -0.4205      | -0.42        |
| (0,1) | -0.5139      | -0.38        |
| (0,2) | -0.2386      | -0.34        |
| (1,0) | -0.5139      | -0.57        |
| (2,0) | -0.2386      | -0.72        |
| (2,1) | -0.0693      | -0.68        |

The approximation is poor — the true `V^\pi` is not even approximately
linear in `(r, c)` — but it converges to a definite answer rather than
diverging. The size of the approximation error reflects the function
class's limitation, not an algorithmic failure.

The pedagogical point: when the function class is rich enough to
contain `V^\pi`, on-policy linear TD recovers it exactly. When the
function class is not rich enough, the algorithm still converges, just
to the best in-class approximation. **The convergence guarantee is
robust to the function class's expressivity.**

---

> **Crosslink to Lesson 4.** Tsitsiklis & Van Roy's proof relies on
> the contraction property of the composition `\Pi T^\pi` in the
> `\pi`-weighted norm. The Bellman expectation operator `T^\pi` is a
> `\gamma`-contraction in the sup norm (Lesson 4), and the projection
> `\Pi` is non-expansive in any norm to which it is the orthogonal
> projection. In the `\pi`-weighted norm `\Pi T^\pi` is a contraction
> with modulus `\gamma`, which under Banach's fixed-point theorem
> guarantees a unique fixed point and convergence to it.

---

**Visualization V2 — Linear FA Convergence.**

A four-panel layout. Panel A: the 3×3 gridworld colorized by the true
`V^\pi` (top half) and by the linear approximation `V_\theta` (bottom
half). The cells are colored using the diverging palette; the
discrepancy between the two halves is visually salient. Panel B: a
plot of `\theta` over training time as a vector field arrow tracing
its trajectory through 3-D parameter space (projected to 2-D with
`\theta_3` shown as size). Panel C: a learning curve showing the
weighted MSE between `V_\theta` and `V^\pi` over time, converging to
the in-class minimum. Panel D: a slider letting the user change the
feature representation — between (i) `\phi(s) = (r, c, 1)` (plane
approximation, poor fit), (ii) `\phi(s) = (r, c, r^2, c^2, rc, 1)`
(quadratic, decent fit), and (iii) `\phi(s) = \text{one-hot}` (tabular,
exact). As the user dials up the expressivity, Panel A's discrepancy
shrinks to zero. Width 880, height 460.

---

### Section 3 — The Deadly Triad: When Linear FA Diverges

**Tagline:** *Three properties, individually fine, jointly catastrophic. Baird's counterexample.*
**Length:** ~800 words.
**Anchor:** `deadly-triad` and `bairds-counterexample`.

---

**Prose:**

The convergence story from Section 2 has a critical assumption:
**on-policy** data. The data distribution matches the policy being
evaluated. The projection `\Pi` is defined in the `\pi`-weighted norm
(the stationary distribution of `\pi`), and the contraction property
of `\Pi T^\pi` is in that same weighted norm. If the data distribution
does *not* match `\pi`, the projection is taken in a different
weighting, and the contraction property can fail.

The standard formulation of when divergence occurs is the **deadly
triad**, three properties whose combination breaks convergence:

1. **Function approximation.** `V` or `Q` is represented parametrically
   (linearly or with a neural network) rather than as a table. The
   tabular setting is exempt from divergence by definition — every
   state has its own parameter, and updates to one parameter cannot
   destabilize another.

2. **Bootstrapping.** The update target involves the current estimate
   (the TD target `r + \gamma V_\theta(s')`) rather than the true
   return (the MC target `G_t`). Bootstrapping is what creates the
   feedback loop where the current `\theta` affects the target that
   updates `\theta`.

3. **Off-policy learning.** The data is collected under a behavior
   policy that differs from the policy being evaluated. Q-learning's
   `\max_{a'} Q(s', a')` target is implicitly off-policy: the target
   evaluates the greedy policy, but the data comes from the
   exploration policy.

The triad theorem (Sutton & Barto 2018; building on Baird 1995): when
all three properties hold simultaneously, semi-gradient updates can
diverge. The parameter norm `\|\theta\|` can grow without bound.

Each property in isolation is fine. Function approximation alone (e.g.
in supervised learning) does not diverge. Bootstrapping alone (tabular
TD) does not diverge. Off-policy alone (MC with importance sampling)
does not diverge. The combination is what kills you.

---

**Baird's counterexample.** The canonical demonstration is a 7-state
MDP from Baird (1995):

- States 1 through 7. Two actions: `dashed` and `solid`.
- Action `dashed` from any state: go to one of states 1-6 uniformly at random.
- Action `solid` from any state: go to state 7 deterministically.
- Reward 0 on every transition. `\gamma = 0.99`.
- Behavior policy: 50/50 between `dashed` and `solid`.
- Target policy: always `solid` (importance ratio = 2 for `solid`, 0 for `dashed`).

The target policy always reaches state 7 in one step and stays there
forever (state 7 is absorbing under `solid`). All rewards are zero, so
`V^\pi(s) = 0` for every state. The features are eight-dimensional and
chosen so that `\theta = \mathbf{0}` gives `V_\theta(s) = 0` for every
`s`. The function class contains the true `V^\pi`.

Despite the true solution being in the function class, **off-policy
semi-gradient TD diverges**. With `\alpha = 0.01` and initial
`\theta = (1, 1, 1, 1, 1, 1, 10, 1)`, the parameter norm evolves as:

| iteration | `\|\theta\|` |
|----------:|------------:|
| 0         | 10.3        |
| 100       | 18.1        |
| 500       | 107.6       |
| 1,000     | 475.4       |
| 1,500     | 1,525.7     |
| 2,000     | 4,582.3     |

The norm grows roughly *exponentially*. The true `\theta^* = \mathbf{0}`
is sitting right there in the function class, and the algorithm is
moving the parameters *away* from it. This is the deadly triad in
numerical form.

---

**Killing one component restores convergence.** The same problem,
with one factor of the triad removed:

| Configuration                          | `\|\theta\|` at iteration 2,000 |
|:---------------------------------------|--------------------------------:|
| All three (function approx, bootstrap, off-policy) | 4,582                |
| No off-policy (behavior = target)      | 22 (bounded)                    |
| No bootstrapping (MC instead of TD)    | 8.6 (converges)                 |
| No function approximation (tabular)    | exactly 0 (exact)              |

Killing any one of the three components stops the divergence. The
triad is precisely the structural diagnosis of *why* off-policy
semi-gradient TD fails. The problem is not in any single component but
in their combination.

---

**What this means for DQN.** DQN is exactly the deadly triad: function
approximation (neural net), bootstrapping (TD target), off-policy
(behavior is ε-greedy, target is greedy via `\max`). By the triad
theorem, DQN should diverge. In practice, DQN often *does* diverge —
the original DQN paper documents many failed runs and emphasizes that
the algorithm is highly sensitive to hyperparameters and to the
stabilizing tricks introduced in the same paper. Sections 5 and 6
introduce those tricks: target networks and experience replay are
engineering responses to the deadly triad, designed to dampen the
divergent dynamics enough to make training tractable.

---

> **Forward link to deep RL stability research.** The deadly triad is
> not the only divergence mechanism in deep RL — there are also
> instabilities from non-stationary data distribution, distribution
> shift between online and replay data, optimistic value estimates,
> and many others. The triad is one mechanism among several, and the
> DQN tricks address it most directly. Modern deep RL stability
> research (gradient clipping, prioritized replay weights, twin
> networks, distributional value functions) continues to add more
> mechanisms; the triad is the historical starting point.

---

**Visualization V3 — Baird's Counterexample.**

A four-panel layout. Panel A: a schematic of the 7-state MDP, with
the dashed and solid actions drawn as arrows in different colors.
Panel B: the parameter vector `\theta` displayed as a bar chart with
eight bars, updating step by step as the user advances training.
Below each bar, the current value is printed in monospace. The norm
`\|\theta\|` is displayed prominently above the chart with a red flash
when divergence becomes visible. Panel C: a plot of `\log \|\theta\|`
versus iteration on a linear-y axis (so a straight line indicates
exponential divergence). Three traces are shown: off-policy semi-grad
TD (diverges), on-policy semi-grad TD (bounded), MC off-policy (bounded).
Panel D: a "triad selector" with three checkboxes labeled "function
approximation," "bootstrapping," and "off-policy." Each can be toggled.
When all three are checked, the running trace in Panel C is shown in
red. When any one is unchecked, the trace turns green. The pedagogical
moment: only the all-three configuration diverges. Width 960
(centerpiece breakout), height 540.

---

### Section 4 — Neural Networks as Function Approximators

**Tagline:** *Same idea, more capacity. Backprop, ReLU, target networks.*
**Length:** ~650 words.
**Anchor:** `neural-q-network`.

---

**Prose:**

Linear function approximation requires hand-designed features `\phi(s)`.
For images, joint angles, sensor streams, or any input where the
"natural" features are not known a priori, this is a major limitation.
Neural networks remove this limitation: a multi-layer perceptron or
convolutional network learns the features from data, taking the raw
state as input. For the curriculum's gridworld, a small MLP suffices.
For Atari games, the canonical architecture is a convolutional network
with three conv layers followed by two fully-connected layers, taking
an `84 \times 84 \times 4` byte tensor as input and outputting one
Q-value per action.

The Q-network for our running 7×7 gridworld (introduced in Section 5)
is a small MLP:

```text
Input: state index, one-hot encoded -> R^49
Hidden 1: linear (49 -> 64), ReLU
Hidden 2: linear (64 -> 64), ReLU
Output: linear (64 -> 4)  # Q-values for 4 actions
```

About 8,000 parameters. The training data is `(s, a, r, s', \text{done})`
transitions. The target is the Bellman-optimality target
`r + \gamma \max_{a'} Q_\theta(s', a')` (with the `\max` taken over the
output layer of the same network). The loss is squared error between
the network's `Q(s, a)` and this target. Gradient descent updates
`\theta` by backpropagation.

---

**Why the basic recipe fails.** Naive Q-learning with a neural network
— train the network to fit the bootstrap target via stochastic
gradient descent — is the deadly triad in extreme form. The
divergence theorem says we should expect failure, and empirically that
is exactly what happens. The Q-network outputs grow in magnitude; the
estimated Q-values become wildly optimistic; the policy becomes
unstable.

Two structural problems are easy to identify:

**Moving target.** At each SGD step, we update `\theta` to better fit
the target `r + \gamma \max_{a'} Q_\theta(s', a')`. But this target
depends on `\theta` itself — the very parameter we are updating. As
`\theta` changes, the target changes; the target changes, the loss
landscape shifts; the gradient direction moves. The optimization is
chasing its own tail. In tabular Q-learning the same dynamic exists,
but the contraction property of `T^*` bounds it: updates to one
state's Q-value don't directly perturb other states' targets except
through the value propagation. With function approximation, every
update changes the target for *every* state — small changes in
`\theta` can produce large changes in `Q_\theta(s', \cdot)` for `s'`
far from any state the agent just visited.

**Correlated samples.** SGD's convergence theory assumes the
gradients at each step are independent samples of an underlying
distribution. The data the agent sees is a *trajectory* — consecutive
states are highly correlated. Updates on consecutive transitions push
`\theta` in correlated directions, and the cumulative effect can be
much larger than the per-step gradient norm suggests. The Q-network
overshoots in a particular direction because the data is over-sampled
in that direction.

The next two sections introduce the two key DQN tricks. Target networks
address the moving-target problem. Experience replay addresses the
correlated-samples problem. Together — plus a few smaller engineering
details — they make Q-learning with neural networks tractable.

---

**Visualization V4 — Q-Network Architecture.**

A clean schematic of the Q-network: input layer (one-hot state),
hidden layers (linear, ReLU), output layer (Q-values per action). The
backpropagation flow is drawn with arrows showing how the gradient
propagates back through each layer. A small inset shows the target
computation: feed `s'` through the *same network* to produce
`Q(s', \cdot)`, take the max, multiply by `\gamma`, add `r`. The
self-reference is highlighted: the same network is used for both
prediction and target. Width 720, height 360.

---

### Section 5 — DQN: The Canonical Recipe

**Tagline:** *Target network + experience replay + Q-learning + neural net. The 2015 Nature paper.*
**Length:** ~900 words.
**Anchor:** `dqn`, `target-network`, `experience-replay`.

---

**Prose:**

Deep Q-Networks (DQN), as introduced by Mnih et al. in their 2015
Nature paper, is Q-learning with two key modifications:

1. **Target network.** Maintain a *separate* copy of the Q-network
   parameters, `\theta^-`, used only for computing the bootstrap target.
   The online parameters `\theta` are updated by gradient descent on
   each step; the target parameters `\theta^-` are copied from
   `\theta` periodically (every `C` updates) and otherwise frozen.

2. **Experience replay.** Store every observed transition
   `(s, a, r, s', \text{done})` in a buffer of size `N` (typically
   `10^5` to `10^6`). Each SGD update samples a minibatch of `B`
   transitions uniformly from the buffer, computes the loss across
   the minibatch, and applies one gradient step.

The DQN training loop is:

```text
DQN(α, ε, γ, target_update_freq C, buffer_size N, batch_size B):
    initialize Q-network θ
    initialize target network θ⁻ ← θ
    initialize replay buffer D ← empty (capacity N)
    for episode = 1, 2, ...:
        s ← initial state
        while s is not terminal:
            a ← ε-greedy(Q_θ, s)
            observe r, s' from environment
            store (s, a, r, s', done) in D
            sample minibatch B from D
            for each (s_i, a_i, r_i, s'_i, done_i) in minibatch:
                target_i ← r_i + (1 - done_i) γ max_{a'} Q_{θ⁻}(s'_i, a')
            θ ← θ - α ∇_θ (1/B) Σ_i (Q_θ(s_i, a_i) - target_i)²
            every C updates: θ⁻ ← θ
            s ← s'
    return Q_θ, greedy(Q_θ)
```

The Q-learning content is still all there: ε-greedy behavior, Bellman
optimality target with the `\max` over next-state actions, squared TD
error loss. The two modifications are bolted on top.

---

**Why target networks work.** The bootstrap target
`r + \gamma \max_{a'} Q_\theta(s', a')` depends on `\theta`. When SGD
updates `\theta`, the target moves. Section 4 identified this as the
moving-target problem.

The target network "freezes" the target. While `\theta^-` is constant
(between target updates every `C` steps), the bootstrap target is a
fixed function of `(s', a')`: `r + \gamma \max_{a'} Q_{\theta^-}(s', a')`.
SGD updates on this fixed target converge to the best fit — exactly
the tabular Q-learning situation where the target is fixed within one
Bellman backup. Then `\theta^-` is updated to the current `\theta`,
the target changes, and SGD chases the new target. The dynamic is
essentially Q-iteration with `C`-step Bellman backups, each backup
fitted by SGD.

This restores enough of the Bellman contraction to make convergence
plausible. It is not a true contraction (SGD never perfectly fits the
target within `C` steps), but the divergence dynamics of naive deep
Q-learning are substantially dampened. Target update frequency `C` is
a hyperparameter: too large and learning is slow (each Bellman backup
takes too long to fit); too small and the target moves enough to
recreate the divergent dynamics. Typical Atari settings: `C = 10,000`
update steps.

---

**Why replay works.** Online updates use consecutive transitions,
which are highly correlated. Correlated gradients are the central
violation of the i.i.d. assumption that SGD's convergence theory
rests on, and the practical consequence is high-variance gradient
estimates and pathological optimization dynamics.

Replay buffers decorrelate the data. Each minibatch samples uniformly
from the buffer, mixing transitions from many different times and
episodes. The minibatch gradient is approximately i.i.d. Two
additional benefits emerge: (i) **sample efficiency** — each transition
is used many times for updates rather than once, multiplying the
information extracted per environment step; (ii) **distribution
smoothing** — the buffer averages over many time-varying behaviors,
making the training distribution more stationary.

Buffer size `N` is a hyperparameter. Too small and decorrelation
fails. Too large and the buffer contains stale data from outdated
policies (which is its own problem, distinct from the deadly triad —
the buffer is now off-policy from the current behavior in addition to
being off-policy from the target). Typical Atari settings:
`N = 1,000,000`.

---

**DQN on the 3×3 gridworld: ablation.** Running DQN with various
combinations of the tricks on the running gridworld, averaged over 5
seeds at 800 episodes:

| Configuration                          | Q(0,0,right) mean ± std | true Q* | Q(0,0,up) std |
|:---------------------------------------|:------------------------|--------:|--------------:|
| Full DQN (target + replay)             | 0.701 ± 0.055           | 0.729   | 0.000         |
| Target only, no replay                 | 0.624 ± 0.054           | 0.729   | 0.006         |
| Replay only, no target                 | 0.614 ± 0.057           | 0.729   | 0.010         |
| Naive (no target, no replay)           | 0.614 ± 0.051           | 0.729   | 0.039         |

Full DQN is closest to truth (error ~0.028). All three ablated
versions undershoot by similar amounts (~0.11), but the naive version's
*variance* on the secondary action `up` is higher (0.039 vs. 0.000-0.010
for the others). The pattern: tricks individually help; tricks
together help most. On a small problem like the 3×3 gridworld, even
naive DQN doesn't catastrophically diverge — the function class is so
expressive (8,000 parameters for 36 Q-values) that the noise can be
absorbed. On Atari and other large problems, the gap between naive
and full DQN is the difference between learning and chaos.

---

**Other DQN details.** The 2015 paper includes several smaller tricks
that improved stability:

- **Huber loss** instead of squared error: bounds the gradient
  magnitude when TD errors are large, preventing exploding updates.
- **Gradient clipping** at `\pm 1`: a second layer of protection
  against exploding gradients.
- **Frame skipping** and frame stacking on Atari: every fourth frame
  is processed, four stacked frames are used as the input (capturing
  motion).
- **Reward clipping** at `[-1, +1]`: scales all rewards into a fixed
  range so a single hyperparameter set works across games.

These are engineering details, not central algorithmic insights, but
each contributes to making the system work. Lesson 9's stretch goals
discuss them in more detail.

---

**Visualization V5 — DQN Tricks Explained.**

A two-row layout. Top row: a side-by-side comparison of "naive Q-learning
with NN" and "DQN with target network and replay." Left half: the
naive update loop schematically, with arrows showing the same network
producing both the prediction and the target. The arrow from the
"target" computation back to the network is highlighted in red as the
"moving target." Right half: the DQN loop, with the target network
drawn separately (frozen, in amber) and the replay buffer drawn as a
stack of transitions (in cyan) feeding minibatches into the update.

Bottom row: a learning curve comparison from the 5-seed ablation table.
Four traces — naive, target-only, replay-only, full DQN — converging
to different fixed points. The "full DQN" trace converges to the
horizontal reference at Q* = 0.729. Width 880, height 480.

---

### Section 6 — DQN Stability Lab (Centerpiece)

**Tagline:** *Run DQN with toggles for every component. Watch what each trick does.*
**Length:** ~500 words.
**Anchor:** `dqn-stability-lab`.

---

**Prose:**

Sections 1-5 introduced the components. This section's job is to make
them tangible. The DQN Stability Lab runs all the variations
side-by-side, with synchronized controls, and lets the user discover
the ablation table from Section 5 by direct manipulation.

The lab uses the **7×7 gridworld** rather than the 3×3 we have been
using throughout. The 7×7 has 49 states, four actions, walls that
make the value function genuinely non-linear in `(r, c)`, and two
goals (one with reward +1, one with reward +0.5). The Q-network is
the small MLP from Section 4: 49 → 64 → 64 → 4 with ReLU activations.

The training runs are pre-computed offline by `scripts/train_dqn.py`
and shipped as ONNX models + training-trace JSON. The in-browser
experience is essentially playback with synchronized timeline scrubbing
across the ablation configurations. Users can pause, scrub, and switch
between configurations.

---

**Visualization V6 — DQN Stability Lab (Centerpiece).**

The polish-budget sink. Allocate four to five days.

A six-panel synchronized layout.

**Panel A — Configuration Selector.** Four checkboxes for the four DQN
ablation configurations: "naive Q-learning," "+target network," "+replay
buffer," "full DQN." A "show all four" mode displays all four panels
simultaneously; individual modes focus on one.

**Panel B — Live Q-Network Output.** The 7×7 gridworld, with each cell
colorized by `\max_a Q_\theta(s, a)` using the diverging palette. The
greedy action at each cell is shown as an arrow. The display updates
every 100 training steps; the timeline scrubber lets the user move
forward or backward through training.

**Panel C — Training Loss Curve.** A line plot of the TD loss versus
training step, with separate traces for the four configurations.
Naive's loss exhibits instability and oscillations; full DQN's loss
decreases smoothly. The y-axis is on log scale to make the difference
visible.

**Panel D — Q-Value Trace at Selected State.** The user clicks any
cell of the gridworld; this panel shows the Q-values for all four
actions at that cell, over training time. For each configuration, the
trace shows where the Q-values stabilize and how much they oscillate.
Reference lines mark the true `Q^*(s, \cdot)` (computed via tabular
value iteration on the 7×7 gridworld).

**Panel E — Parameter Norm.** The Frobenius norm `\|\theta\|` over
training time. For naive DQN, the norm grows; for full DQN, it stays
bounded. This is the diagnostic of the deadly triad from Baird's
example, here applied to a real neural network.

**Panel F — Bellman Error Diagnostic.** For each state, the magnitude
of the Bellman error `Q_\theta(s, a) - [r + \gamma \max Q_\theta(s', a')]`
averaged over actions and trajectories. Naive DQN's Bellman error
remains high (the network is far from a self-consistent Q-function);
full DQN's Bellman error converges to near-zero.

**Controls strip:** training-step scrubber (0 to 50,000 steps); learning
rate slider (0.0001 to 0.01); target update frequency slider (10 to
1000 steps); replay buffer size slider (100 to 10,000); discount slider
(0.9 to 0.999). When the user changes any of these, the trained network
is re-loaded (from a pre-trained collection if the configuration matches
a pre-computed run; otherwise a "configuration not pre-computed"
notice).

The pedagogical statement of the lab: the four configurations live or
die on Panel E (the parameter norm). When `\|\theta\|` explodes, the
Q-values become meaningless, and Panel B shows random-colored noise.
When `\|\theta\|` stays bounded, the Q-values converge to a reasonable
approximation of `Q^*`. The target network and replay buffer are what
keeps `\|\theta\|` bounded.

Width 960 (centerpiece breakout). Height 920.

---

### Section 7 — DQN Family: Double DQN, Dueling, Prioritized, Distributional

**Tagline:** *Each addressing a specific shortcoming of vanilla DQN. Double DQN cleans up max bias.*
**Length:** ~700 words.
**Anchor:** `double-dqn`, `dueling-dqn`, `dqn-family`.

---

**Prose:**

Vanilla DQN works but has documented problems. Each of the major
variants targets a specific one.

---

**Double DQN: max bias revisited.** Lesson 8's Section 5 documented
Q-learning's **maximization bias**: the expected value of the maximum
of noisy estimates is larger than the maximum of the expected values,
`\mathbb{E}[\max_a \hat Q(s, a)] \geq \max_a \mathbb{E}[\hat Q(s, a)]`.
With function approximation the noise sources multiply (training
sampling noise, function-class approximation noise, target-network
staleness), and the bias is amplified accordingly.

The Double DQN trick (van Hasselt, Guez, Silver 2015) decouples
action selection from action evaluation in the bootstrap target.
Instead of

$$
\text{target} = r + \gamma \max_{a'} Q_{\theta^-}(s', a'),
$$

Double DQN uses

$$
\text{target} = r + \gamma Q_{\theta^-}\big(s', \arg\max_{a'} Q_\theta(s', a')\big).
$$

The online network `Q_\theta` selects the action (`\arg\max`); the
target network `Q_{\theta^-}` evaluates it. Selection bias and
evaluation bias are no longer correlated, and the maximization bias
shrinks substantially. The change is one line of code and a robust
empirical improvement on Atari.

On the Sutton & Barto maximization bias MDP (the two-state example
from Lesson 8 Section 5), DQN exhibits 43% wrong-action selection at
100 episodes and 35% at 300 episodes (vs. the optimal 5%). Double DQN
brings these down to roughly 12% at 100 episodes and 7% at 300, much
closer to optimal. The difference is structurally important: vanilla
DQN's overestimation is systematic and can compound; Double DQN
removes the systematic component.

---

**Dueling DQN: separating state-value and advantage.** The
action-value function decomposes as

$$
Q^\pi(s, a) \;=\; V^\pi(s) + A^\pi(s, a)
$$

where `V^\pi(s) = \mathbb{E}_{a \sim \pi}[Q^\pi(s, a)]` is the
state-value and `A^\pi(s, a)` is the **advantage** (how much better
than average is action `a` at state `s`). In many states, the
specific action matters little compared to the underlying value of
being in that state — `V^\pi(s)` dominates. A vanilla DQN that
estimates `Q(s, a)` end-to-end must learn `V` redundantly for every
action.

Dueling DQN (Wang et al. 2016) gives the network architecture an
inductive bias for this decomposition:

```text
Input → shared trunk → split:
                       ├─ V head → V(s)
                       └─ A head → A(s, ·)
Q(s, a) = V(s) + A(s, a) - (1/|A|) Σ_a' A(s, a')   # mean-centering
```

The mean-centering is needed because `V` and `A` are not identifiable
without it (`Q` is invariant to a constant offset between them). Empirical
results on Atari show small but consistent improvements, especially on
games where the value of states varies more than the value of actions.

---

**Prioritized experience replay.** Instead of sampling uniformly from
the buffer, prioritize transitions with high TD error — these are the
ones the network is learning the most from. The priority is typically
proportional to `|\delta|^\alpha` for `\alpha \in [0, 1]`, with
importance-sampling correction during the gradient update to maintain
unbiasedness. Schaul et al. 2015 demonstrate substantial speedups on
Atari.

---

**Distributional DQN.** Vanilla DQN learns `\mathbb{E}[Q(s, a)]`. The
**distributional** DQN family (Bellemare, Dabney, Munos 2017) learns
the full *distribution* over returns from state-action pairs. The
algorithm `C51` discretizes the return support into 51 bins and
learns a categorical distribution over them; subsequent work (`QR-DQN`,
`IQN`) generalizes to quantile and implicit representations. The
distributional perspective often leads to faster learning and more
robust value estimates, especially in environments with multimodal
return distributions. Bellemare's *Distributional Perspective* paper
is one of the most influential single contributions to value-based deep
RL.

---

**Rainbow.** Hessel et al. 2018 combine seven DQN improvements
(Double, Dueling, Prioritized replay, Distributional, n-step returns,
NoisyNets, and a few more) into a single algorithm called **Rainbow**.
On Atari, Rainbow is one of the strongest known value-based agents.
Each component's marginal contribution is measurable; the synthesis is
the canonical "modern DQN" baseline.

---

**Visualization V7 — Double DQN vs DQN on Max Bias.**

A two-panel layout. Left panel: the Sutton & Barto two-state MDP
from Lesson 8 Section 5, with the "fraction wrong action" plotted
versus episode count for both DQN (red, hovering around 35%) and
Double DQN (green, decaying smoothly to ~7%). The horizontal reference
at 5% (the ε-greedy optimum) is shown. Right panel: the same problem
visualized as the estimated `Q(A, \text{left})` over time — vanilla
DQN's estimate grows above the true value (overestimation), Double
DQN's estimate stays near the true value.

Below the two panels, a small table summarizing the DQN family:
Double DQN, Dueling DQN, Prioritized Replay, Distributional, Rainbow,
with one-sentence descriptions and pointers to the source papers.
Width 880, height 540.

---

### Section 8 — DQN in Practice and Forward Links

**Tagline:** *When DQN works, when it doesn't, and what comes next.*
**Length:** ~500 words.
**Anchor:** `dqn-family`.

---

**Prose:**

DQN works well on:
- Discrete-action environments (Atari, board games, simple robotics
  with discretized control).
- Environments with reward signals that are not catastrophically
  sparse.
- Settings with abundant simulated data (the Atari benchmark uses
  millions of frames per game).

DQN works poorly on:
- Continuous-action environments (Q-learning's `\max` is over discrete
  actions; with continuous actions, the max is intractable in general).
- Very sparse-reward environments (DQN's bootstrap target's signal-to-noise
  ratio degrades catastrophically when rewards are rare).
- Environments where the optimal policy requires exploration that
  ε-greedy cannot generate (chains, mazes with deceptive sub-goals).
- Settings with limited data (DQN is sample-inefficient compared to
  policy-gradient methods with off-policy critics).

The DQN family dominates the discrete-action benchmark literature
(Atari, ALE, Crafter) but has been overshadowed by policy-gradient
methods in continuous control and large language model alignment.

---

**Where this goes.**

**Lesson 10 (Policy Gradient Methods).** Policy gradient methods
directly parameterize the policy as `\pi_\theta(a \mid s)` rather than
inferring it from a Q-function. They handle continuous actions
naturally, sidestep the deadly triad for the policy gradient itself
(no bootstrapping in REINFORCE), and admit straightforward extensions
to the actor-critic framework. The "actor" is the policy network; the
"critic" is a value-function approximator trained with TD methods —
inheriting all the DQN machinery in a supporting role.

**Lesson 11 (TRPO/PPO).** Policy gradient methods with trust-region
constraints. Uses TD-style critics (Generalized Advantage Estimation)
for variance reduction. The DQN family's contribution to TRPO/PPO is
the critic's value estimation, not the policy itself.

**Lesson 13 (SAC).** Soft Actor-Critic uses *two* Q-networks
(addressing maximization bias) and a stochastic policy network. The
twin-Q trick is a direct descendant of Double DQN, refined for the
continuous-action setting.

**Lesson 15 (Offline RL).** Offline value learning revisits Q-learning
in the setting where no further data can be collected. The deadly
triad becomes acute (extreme distribution mismatch between behavior
data and target policy); the remedies (CQL, BCQ) modify the Q-learning
update to be conservative or to constrain the policy. The DQN
machinery is the substrate; the conservative modifications are the
content.

**Lesson 16 (Diffusion in RL).** Some recent work uses diffusion
models for the policy and DQN-style value learning for the value
function. The combination is one of several active areas at the
deep-RL frontier.

---

**Visualization V8 — Roadmap Mini.** The curriculum's lesson-graph
thumbnail with Function Approximation and DQN now marked as shipped.
Outgoing arrows go to Lesson 10 (Policy Gradient), Lesson 11
(TRPO/PPO), Lesson 13 (SAC), Lesson 15 (Offline RL), Lesson 16
(Diffusion in RL), and Lesson 17 (RLHF). Each arrow's hover popover
shows the specific application. Width 720, height 260.

---

## 5. Algorithm and Math Implementation

The TypeScript module `src/dqn/` is around 350 lines.

```ts
import type { MDP } from "../mdp/types";
import * as ort from "onnxruntime-web";

/** Tabular Q-learning baseline (re-exported from src/td/). */
export { qLearning } from "../td/qlearning";

/** Linear function-approximation TD(0) for V-learning. */
export function linearTDPrediction(
  mdp: MDP,
  policy: Policy,
  featurize: (s: number) => Float64Array,
  nEpisodes: number,
  alpha: number,
  options: { rng?: () => number } = {},
): { theta: Float64Array; history: number[][] } {
  const d = featurize(0).length;
  const theta = new Float64Array(d);
  const history: number[][] = [];
  // ... ~30 lines: episode loop, semi-gradient update theta += alpha*delta*phi(s)
}

/** Baird's counterexample driver. */
export function bairdCounterexample(
  nSteps: number,
  options: {
    alpha?: number;
    initTheta?: Float64Array;
    onPolicy?: boolean;  // if true, kills the off-policy component (no divergence)
    bootstrap?: boolean; // if false, uses MC return (no divergence)
    rng?: () => number;
  } = {},
): { thetaHistory: Float64Array[]; normHistory: number[] } {
  // ... ~80 lines: implements Baird's 7-state MDP and the divergence demo
}

/** DQN training driver. Loads pre-trained ONNX models for playback. */
export interface DQNConfig {
  useTargetNetwork: boolean;
  useReplayBuffer: boolean;
  useDoubleDqn?: boolean;     // for Section 7
  alpha: number;
  epsilon: number;
  gamma: number;
  targetUpdateFreq: number;
  replayBufferSize: number;
  batchSize: number;
}

export interface DQNTrace {
  qValuesPerEp: number[][];          // (state, action) -> q over episodes
  lossPerStep: number[];
  paramNormPerStep: number[];
  bellmanErrorPerStep: number[];
}

export async function loadDQNTrace(
  configHash: string,
): Promise<{ session: ort.InferenceSession; trace: DQNTrace }> {
  const session = await ort.InferenceSession.create(`/models/dqn/${configHash}.onnx`);
  const trace = await fetch(`/data/dqn/${configHash}.json`).then(r => r.json());
  return { session, trace };
}

export async function qValuesAtStep(
  session: ort.InferenceSession,
  state: number,
  nStates: number,
  step: number,
): Promise<Float32Array> {
  // Run inference on the loaded ONNX model. Returns Q-values for all actions.
  const input = new Float32Array(nStates);
  input[state] = 1;
  const result = await session.run({ state: new ort.Tensor("float32", input, [1, nStates]) });
  return result.q_values.data as Float32Array;
}
```

**Vitest test targets:**

```ts
test('Baird counterexample diverges with full deadly triad', () => {
  const { normHistory } = bairdCounterexample(2000, {
    alpha: 0.01, rng: seeded(0),
  });
  // ||θ|| should be growing approximately exponentially
  expect(normHistory[100]).toBeLessThan(50);
  expect(normHistory[1000]).toBeGreaterThan(200);
  expect(normHistory[2000]).toBeGreaterThan(2000);  // explodes
});

test('Baird counterexample is bounded without off-policy', () => {
  const { normHistory } = bairdCounterexample(2000, {
    alpha: 0.01, onPolicy: true, rng: seeded(0),
  });
  expect(normHistory[2000]).toBeLessThan(50);  // bounded
});

test('Baird counterexample converges without bootstrapping (MC)', () => {
  const { normHistory } = bairdCounterexample(2000, {
    alpha: 0.01, bootstrap: false, rng: seeded(0),
  });
  expect(normHistory[2000]).toBeLessThan(15);  // converges
});

test('Linear TD on gridworld converges to projected fixed point', () => {
  const { theta } = linearTDPrediction(gridworld, uniformRandom,
    s => featurizeRowColBias(s), 10_000, 0.05, { rng: seeded(0) });
  // For phi=(r,c,1), the best linear fit to V^pi has theta ≈ (-0.15, 0.04, -0.42)
  expect(theta[0]).toBeCloseTo(-0.15, 1);
  expect(theta[1]).toBeCloseTo(0.04, 1);
  expect(theta[2]).toBeCloseTo(-0.42, 1);
});

test('Full DQN converges close to Q* on 3x3 gridworld', () => {
  const { session, trace } = await loadDQNTrace('full-dqn-default');
  const qVals = await qValuesAtStep(session, stateIdx(0, 0), 9, 800);
  // Q*(0,0,right) = 0.7290
  expect(qVals[1]).toBeGreaterThan(0.6);  // close enough
  expect(qVals[1]).toBeLessThan(0.78);
});

test('Naive DQN has higher Q-value variance than full DQN', () => {
  // Load both full-dqn and naive-dqn traces; compare std across seeds
  const fullStd = await stdAcrossSeeds('full-dqn', 5);
  const naiveStd = await stdAcrossSeeds('naive-dqn', 5);
  expect(naiveStd).toBeGreaterThan(fullStd * 2);
});

test('Double DQN reduces maximization bias on Sutton-Barto MDP', () => {
  const dqnLeftFrac = await loadMaxBiasFraction('dqn', 300);
  const doubleDqnLeftFrac = await loadMaxBiasFraction('double-dqn', 300);
  expect(dqnLeftFrac).toBeGreaterThan(0.20);     // DQN: ~35%
  expect(doubleDqnLeftFrac).toBeLessThan(0.15);  // Double DQN: ~7%
});
```

---

## 6. Component Catalog

| Code | Component                       | Section | Polish budget |
|-----|---------------------------------|---------|--------------|
| V1  | `<TabularLimits>`               | §1      | 0.5 day      |
| V2  | `<LinearFAConvergence>`         | §2      | 1.5 days     |
| V3  | `<BairdCounterexample>`         | §3      | 2 days       |
| V4  | `<QNetworkArchitecture>`        | §4      | 1 day        |
| V5  | `<DQNTricksExplained>`          | §5      | 1.5 days     |
| V6  | `<DQNStabilityLab>`             | §6      | **4-5 days** (centerpiece) |
| V7  | `<DoubleDQNVsMaxBias>`          | §7      | 1.5 days     |
| V8  | `<RoadmapMini>` (update)        | §8      | 0.5 day      |

Total polish budget around twelve to thirteen days. This is the
largest lesson in the curriculum so far, justified by the introduction
of neural networks as a first-class component and the requirement of
real training infrastructure.

**Reuse from prior lessons:** `GridworldRenderer` (extended to 7×7),
`MDPEditor`, `MathBlock`, `CrosslinkCallout`, `PanelChrome`,
`RoadmapMini`, all of `src/mdp/`, `policyEvaluationExact` and
`valueIteration` from `src/dp/`, tabular `qLearning` from `src/td/`.
New code is around 350 lines in `src/dqn/` plus the eight visualization
components, the PyTorch training script, and the ONNX inference
integration.

**New dependencies:** `onnxruntime-web` (browser ONNX runtime, lazily
loaded). `torch` (Python, build-time only). `onnx` (Python, for
exporting trained models).

---

## 7. Page-Level User Experience

Same conventions as prior lessons. Single-page scroll, prereq strip at
top, reduced-motion support for V3's animated divergence and V6's
training-step scrubber.

The centerpiece V6 is the only component breaking out to 960 pixels in
width. V3 also requires 960 because it shows four panels including
the triad selector.

A specific UX note for V6: ONNX inference adds a small latency per
state lookup (~5ms on modern hardware). Pre-fetch the relevant
training-step model for the current scrubber position; otherwise the
heatmap update lags behind user interaction. The four ablation
configurations should each have their full training trajectory
pre-computed (one model checkpoint per ~500 steps, around 100 models
per configuration); the scrubber loads the nearest checkpoint and
runs inference for the visible states only (49 states for the 7×7
gridworld is fast).

V3's "triad selector" is the lesson's key interactive moment. When the
user unchecks one of the three boxes (FA, bootstrap, or off-policy),
the running divergence trace in Panel C should visibly flatten out
and turn green. This is the lesson's intellectual climax in
visualization form. Make sure the visual transition is striking.

---

## 8. Acceptance Criteria

After completing this lesson, a learner should be able to:

State why tabular Q-learning fails on Atari-scale state spaces — both
the size argument and the generalization argument. Define a linear
function approximator and write down the semi-gradient TD(0) update.
State the Tsitsiklis & Van Roy convergence guarantee for on-policy
linear TD and identify the projected Bellman equation as its fixed
point. State the deadly triad as three named properties whose
combination causes divergence. Describe Baird's counterexample as the
canonical demonstration; identify which combinations of properties
diverge and which converge. Write down the DQN update with target
network and replay buffer; explain what problem each addresses.
Identify Q-learning's maximization bias and explain how Double DQN
decouples action selection from action evaluation to mitigate it.
Predict qualitatively which DQN ablation configurations will converge
and which will diverge on the 7×7 gridworld. Name three downstream
algorithms (Lessons 10-15) that use DQN-style value learning as a
component and identify the specific role.

A concrete acceptance check: ask the learner to predict, before
running V6, which ablation configurations will (i) converge to near-Q*,
(ii) converge to an offset value, (iii) diverge. The correct answers
on the 3×3 gridworld are (i) full DQN; (ii) target-only and
replay-only; (iii) naive (high variance and offset). On the 7×7
gridworld the gaps widen; naive DQN's parameter norm explodes more
visibly.

---

## 9. Stretch Goals (post-MVP)

**Atari demo.** Train a small DQN on a single Atari game (Pong, the
canonical "easy" environment) and ship the result as a stretch
visualization. The infrastructure cost is high — Atari environments,
frame stacking, convolutional architectures — but the payoff is
substantial because Atari is the canonical DQN benchmark and current
generations of students have not seen DQN run on it in the lesson.

**Rainbow ablation.** A full Rainbow-style ablation (six or seven
components, each toggle-able) on Atari Breakout would be a
heavyweight stretch goal. Hessel et al.'s table can be reproduced
directly. Out of scope for MVP because of training cost.

**Continuous-action DQN variants.** Algorithms like NAF (Normalized
Advantage Functions) and CACLA (Continuous Actor-Critic Learning
Automaton) extend Q-learning to continuous actions. They are mostly
of historical interest now (SAC dominates the continuous-action
benchmark) but a brief discussion would round out the DQN family
section.

**Distributional DQN visualization.** Bellemare et al.'s `C51` algorithm
learns a categorical distribution over returns. A visualization showing
the learned return distribution evolving during training would be
visually striking. Out of scope for MVP because of training cost and
because the categorical loss requires substantial extra implementation.

---

## 10. Out of Scope (intentionally)

**Convolutional architectures.** All Q-networks in this lesson are
MLPs on one-hot state representations. The convolutional case is
mentioned in Section 4 but not implemented; including it would
require Atari-scale training infrastructure.

**Exploration beyond ε-greedy.** Methods like Thompson sampling for
deep RL, NoisyNets, RND, and curiosity-driven exploration are out of
scope. ε-greedy is the workhorse in this lesson.

**Off-policy correction techniques.** Importance sampling for DQN
(Retrace, Q(λ)-like extensions) is mentioned briefly but not
implemented. The off-policy correctness of vanilla DQN's `\max`-based
target is accepted at face value.

**Multi-agent DQN.** Cooperative and competitive multi-agent settings
have their own DQN extensions (QMIX, MADDPG). Out of scope.

---

## 11. Training Notebook

The script `scripts/train_dqn.py` (around 250 lines) is the lesson's
first real training pipeline. It trains all the DQN ablation
configurations on the 3×3 and 7×7 gridworlds, exports each as an
ONNX model, and emits training-trace JSON for in-browser playback.

```text
scripts/train_dqn.py
    inputs:  --env={3x3,7x7} --config={naive,target,replay,full,double}
             --n-episodes=800 --n-seeds=5
    outputs: public/data/dqn/<env>-<config>/<seed>.json (training traces)
             public/models/dqn/<env>-<config>/<seed>-step-<step>.onnx (model checkpoints)
```

Each training run takes approximately 30 seconds for the 3×3 gridworld
and 5 minutes for the 7×7 gridworld on a single modern GPU. The
training pipeline runs offline (build-time, not at user-interaction
time). The full ablation set is around 40 runs (4 configs × 2 envs × 5
seeds) and completes in about 2 hours.

The Python code is straightforward PyTorch: a small MLP Q-network, a
dictionary-backed replay buffer, a target network handled by `clone()`
and periodic `state_dict()` copy. Training uses Adam, learning rate
0.001, Huber loss with delta=1.0. Gradients clipped to ±1.0. The full
script is in the agent build brief.

---

## 12. Closing Notes and Length Tally

Total length: roughly seventeen hundred lines. The lesson is the
curriculum's longest, justified by the substantial new content:
neural networks as first-class participants, the deadly triad as a
named mathematical phenomenon, DQN as the canonical recipe, and the
DQN family of improvements. The centerpiece V6 (DQN Stability Lab)
is the polish-budget sink and is the lesson's defining interactive
moment.

The forward links are dense: every subsequent lesson (10, 11, 13, 15,
16, 17) cashes in some piece of this lesson's content. Lesson 10
(Policy Gradient) is the natural successor — policy-gradient methods
abandon the value-based approach and parameterize the policy
directly, with TD-trained critics in supporting roles. The DQN
machinery from this lesson reappears throughout, but no longer as the
star.

## End of specification