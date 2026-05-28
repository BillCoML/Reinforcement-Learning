# Lesson 8 — Temporal-Difference Learning

> **The central classical RL lesson.** Monte Carlo waited until the end of
> an episode to compute the return. Temporal-Difference learning makes the
> single move that defines modern RL: don't wait. Estimate `V^\pi(s)`
> from `r + \gamma \hat V^\pi(s')` — substitute your *current estimate*
> for the rest of the return. This single change — bootstrapping — buys
> incremental updates, online learning, low per-update variance, and
> applicability to non-terminating environments. The trade-off is bias:
> while your estimate is wrong, the bootstrap is wrong. The bias vanishes
> as the estimate improves, and the trade is almost always worth it.
>
> The lesson covers TD(0) for prediction, the Robbins-Monro framework
> that gives TD's convergence guarantees, SARSA and Q-learning as the
> on-policy and off-policy TD control algorithms, n-step TD as the
> family interpolating between TD(0) and MC, TD(λ) with eligibility
> traces, and a careful look at the convergence theory tying back to
> contraction mappings from Lesson 4. The lesson is the curriculum's
> longest because it is the gateway to everything that follows: deep RL
> (Lesson 9 builds on Q-learning), policy gradient (Lesson 10 uses TD
> for the value baseline), and almost every modern algorithm.

> **Where this slots in.** Between Lesson 7 (Monte Carlo) and Lesson 9
> (Function Approximation + DQN). The role is critical-path central:
> every subsequent lesson assumes you can derive a TD update from first
> principles, distinguish on-policy and off-policy TD targets, and read
> a learning curve in terms of bias-variance trade-offs. The lesson
> opens as a direct continuation of Lesson 7's Section 6 (the
> MC → TD comparison) and closes with forward pointers into deep RL,
> policy gradients, and the entire offline-RL track.

---

## 0. Pedagogical Philosophy

Eight commitments specific to this lesson.

The first commitment is that the **bootstrap** is the move that
deserves attention. TD methods are presented in many textbooks as "an
alternative to MC" without surfacing the structural change that makes
them different. The structural change is bootstrapping: substituting a
current estimate `\hat V^\pi(s')` for the unknown future return. Every
other property of TD — its low variance, its online updates, its
applicability to continuing tasks — flows from that single move. We
introduce TD by walking the learner through exactly this substitution,
side by side with the MC target, so the move is visible.

The second commitment is that we treat **Robbins-Monro stochastic
approximation** as the foundation, not as a footnote. The master plan
originally had Robbins-Monro as a separate prereq lesson; that lesson
got folded inline into this one. The treatment must therefore be
careful: TD(0) is an instance of Robbins-Monro applied to the Bellman
operator. The convergence proof is "Robbins-Monro plus the Bellman
operator is a contraction" — both ingredients are now available to
the learner (from this lesson's inline development and from Lesson 4's
contraction theory). We make the connection explicit in Section 7.

The third commitment is that **SARSA and Q-learning are two faces of
one trade-off, not two unrelated algorithms**. They differ in exactly
one line: SARSA's target uses the action the policy will actually
take next; Q-learning's target uses the action the optimal policy
would take. That one-line difference is the entire on-policy vs
off-policy distinction in TD control. We surface it directly,
including the precise consequence — SARSA converges to `Q^\pi_{\epsilon\text{-soft}}`,
Q-learning converges to `Q^*` — and we connect both convergence
targets to the corresponding MC control results from Lesson 7.

The fourth commitment is that **Q-learning's IS-bypass is celebrated
explicitly**. Lesson 6 spent considerable effort developing importance
sampling for off-policy correction, and Lesson 7's Section 5 cashed it
in for off-policy MC at the cost of devastating variance. Q-learning's
trick — re-deriving the off-policy target as a max over actions
rather than as an IS-corrected return — sidesteps the entire IS
apparatus. The variance benefit is enormous. We spend a full callout
on this in Section 4 and connect back to the Lesson 6/7 variance
discussion.

The fifth commitment is that we are **honest about the
bias-variance trade-off**. The textbook claim that "TD has lower
variance than MC" is horizon-dependent. On short-horizon problems
(like the running gridworld with mean episode length 8.7), MC variance
is small enough that TD's bootstrap bias is the dominant error source.
On long-horizon problems with high-variance per-step rewards, TD wins
decisively. The lesson reports both regimes empirically and explains
why the conventional wisdom is true on the problems where TD's
advantage actually matters.

The sixth commitment is that **n-step TD and TD(λ) get the treatment
they deserve**. These are not afterthoughts — they are the
parameterized families that link TD(0) to MC and that underlie modern
algorithms like GAE (Lesson 10's policy gradient variance reduction)
and Retrace (offline RL, Lesson 15). We develop n-step TD with the
forward view explicitly, develop TD(λ) with both the forward and
backward views, and show eligibility traces as the practical
implementation of the backward view.

The seventh commitment is that we **explicitly name maximization bias**
as Q-learning's main weakness. Q-learning's `\max_{a'} Q(s', a')`
target uses noisy estimates inside a maximum, and the expected max of
noisy estimates is greater than the max of expected estimates. This
systematic over-estimation is observable, is well-documented, and
motivates Double Q-learning. We demonstrate it on the canonical
Sutton-Barto example and teaser Double Q-learning as a fix; the full
treatment is deferred to Lesson 9 (where it becomes Double DQN).

The eighth commitment is that the **running gridworld continues**.
Same MDP, same start state, same goal, same pit, same γ = 0.9. Every
algorithm in this lesson is run on the same gridworld at least once,
and every numerical anchor — V^π(0,0) = -0.4205, V*(0,0) = 0.7290,
Q^π_ε-soft(0,0,right) = 0.6307 — appears in this lesson exactly where
the learner expects it from Lessons 5 and 7. The accumulating
familiarity is worth more than novelty.

---

## 1. Tech Stack

The tech stack is unchanged from prior lessons. Vite plus TypeScript
in strict mode, KaTeX for math, D3 version 7 for visualizations,
`ml-matrix` lightly used, Vitest for tests. No new dependencies.

This lesson reuses substantial prior infrastructure. The gridworld
and trajectory sampling come from `src/mdp/` (Lesson 3). The
ground-truth value and Q functions come from `policyEvaluationExact`
and the Bellman backup utilities in `src/dp/` (Lesson 5). The MC
implementations in `src/monte-carlo/` (Lesson 7) are used in
comparison visualizations.

New code in `src/td/` is roughly 350 lines — the largest math module
in the curriculum so far. It implements TD(0) prediction, SARSA,
Q-learning, n-step TD, TD(λ) with eligibility traces, and a
canonical maximization-bias demo MDP. Each is independently testable
against the ground-truth values from `src/dp/`.

A small Python script, `scripts/td_traces.py` at around 130 lines,
pre-computes the convergence statistics tables for all six
algorithms across multiple step-size schedules and step counts. The
outputs land as JSON in `public/data/td/` for instant initial render
of the centerpiece. The in-browser TypeScript reproduces these
statistics live when the user interacts.

---

## 2. Visual and Aesthetic Direction

The curriculum aesthetic continues unchanged. The lesson adds a tight
set of tokens specific to the TD theme.

```css
:root {
  /* TD algorithm signatures */
  --td-td0:         #0e7490;   /* cyan-700  | TD(0) prediction */
  --td-sarsa:       #15803d;   /* green-700 | SARSA: on-policy (rhymes with on-policy in L7) */
  --td-qlearning:   #ea580c;   /* orange-600| Q-learning: off-policy (rhymes with off-policy in L6, L7) */
  --td-nstep:       #b45309;   /* amber-700 | n-step TD */
  --td-lambda:      #6d28d9;   /* violet-700| TD(λ) / eligibility traces */

  /* Comparator overlays */
  --td-mc:          #475569;   /* slate-600 | MC reference (the "baseline" comparator) */
  --td-truth:       #1c1e22;   /* black     | true V or Q */

  /* TD-specific UI elements */
  --td-bootstrap:   #0e7490;   /* cyan, same as TD(0) | the "your-current-estimate" bootstrap target */
  --td-target:      #15803d;   /* green     | the "true return" target (used in MC) */
  --td-error:       #dc2626;   /* red-600   | TD error δ_t */
  --td-trace:       #f59e0b;   /* amber-500 | eligibility trace e_t(s) heatmap */
  --td-greedy:      #15803d;   /* green     | greedy action choice */
  --td-explore:     #b45309;   /* amber     | exploratory action choice */
}
```

The color choices follow the curriculum's established conventions.
SARSA-as-green and Q-learning-as-orange replays the on-policy /
off-policy split established in Lessons 6 and 7 (where green was
on-policy / no-IS, orange was off-policy / IS-corrected). TD(λ) and
its eligibility trace stay in violet because λ is the principled
parameter that does the smoothing, and violet has been the
principled-parameter color throughout the curriculum (Thompson
sampling, weighted IS, GLIE schedules, running estimates).

The red `--td-error` color is reserved for the TD error δ_t = r +
γV(s') − V(s), which gets highlighted in the V1 and V3 animations
whenever a non-trivial update happens. The TD error is the lesson's
central computational object, and giving it a dedicated color keeps
it visible across multiple visualizations.

The `--td-trace` amber is used in V6 (the eligibility trace
visualizer) to render the trace `e_t(s)` as a heatmap that decays
spatially across the gridworld with each step.

---

## 3. Lesson Metadata

```ts
export const lessonMeta: LessonMeta = {
  id: "td-learning",
  title: "Temporal-Difference Learning",
  subtitle: "Bootstrap from your current estimate, update online",
  tier: 1,
  difficulty: 4,
  estimatedReadMinutes: 75,
  role: "critical-path",
  prerequisites: [
    { lesson: "mdps",                 anchor: "action-value-function" },
    { lesson: "contractions",         anchor: "banach-fixed-point" },
    { lesson: "dynamic-programming",  anchor: "iterative-policy-evaluation" },
    { lesson: "monte-carlo",          anchor: "mc-policy-evaluation" },
    { lesson: "monte-carlo",          anchor: "mc-control-eps-greedy" },
    { lesson: "monte-carlo",          anchor: "mc-vs-td" },
  ],
  exportedAnchors: [
    "bootstrap",
    "td-zero-prediction",
    "td-error",
    "robbins-monro",
    "sarsa",
    "q-learning",
    "off-policy-td",
    "maximization-bias",
    "n-step-td",
    "td-lambda",
    "eligibility-traces",
    "td-convergence",
  ],
  centerpieceComponent: "TDAlgorithmLab",
  forwardLinksWhenReady: [
    { to: "function-approximation",  anchor: "deep-q-networks" },
    { to: "function-approximation",  anchor: "double-dqn" },
    { to: "policy-gradient",         anchor: "gae" },
    { to: "trpo-ppo",                anchor: "advantage-baselines" },
    { to: "max-ent-rl",              anchor: "soft-bellman" },
    { to: "offline-rl",              anchor: "retrace" },
  ],
};
```

---

## 4. Section-by-Section Plan

### Section 1 — From Monte Carlo to TD: The Bootstrap Move

**Tagline:** *Don't wait for the end of the episode. Plug in your current estimate.*
**Length:** ~700 words.
**Anchor:** `bootstrap`.

---

**Prose:**

Lesson 7 left us with the Monte Carlo policy-evaluation algorithm:
sample an episode under `\pi`, compute the return `G_t` at every step,
and use `G_t` as a sample of `V^\pi(s_t)`. The algorithm's defining
property is that it waits for the episode to end before it can
compute any target. The full return is needed; partial returns are
not used.

There is a different choice. Suppose we want to estimate `V^\pi(s_t)`
at the moment we take a single step — observe reward `r_{t+1}` and
next state `s_{t+1}` — and we are not willing to wait for the rest of
the episode. We could use the **Bellman expectation equation** as a
guide. The equation says

$$
V^\pi(s) \;=\; \mathbb{E}_\pi \!\left[ R_{t+1} + \gamma V^\pi(S_{t+1}) \,\bigg|\, S_t = s \right].
$$

The right-hand side splits the unknown `V^\pi(s)` into two pieces:
the immediate reward `R_{t+1}`, which we just observed; and the value
of the next state `V^\pi(S_{t+1})`, which we do not know. But we have
a **current estimate** `\hat V^\pi(s_{t+1})`. If we substitute that
estimate in for the true `V^\pi(s_{t+1})`, we get a usable target:

$$
\text{TD target:}\quad r_{t+1} + \gamma \hat V^\pi(s_{t+1}).
$$

This is the **bootstrap** move. We are using our own current estimate
of `\hat V^\pi(s')` as a stand-in for the unknown true `V^\pi(s')`,
which lets us turn a one-step observation into an update target. The
update rule is then

$$
\hat V^\pi(s_t) \;\leftarrow\; \hat V^\pi(s_t) + \alpha \left[ \underbrace{r_{t+1} + \gamma \hat V^\pi(s_{t+1})}_{\text{TD target}} - \hat V^\pi(s_t) \right].
$$

The quantity in brackets is the **TD error**, traditionally written

$$
\delta_t \;:=\; r_{t+1} + \gamma \hat V^\pi(s_{t+1}) - \hat V^\pi(s_t).
$$

It measures how surprising the one-step observation was relative to
the current estimate. The update nudges `\hat V^\pi(s_t)` in the
direction that would reduce that surprise.

---

**The trade-off, viewed structurally.** MC's target `G_t` is a sample
of `V^\pi(s_t)`: it is unbiased, but its variance comes from the
entire trajectory's randomness (many rewards, many transitions, all
summed). TD's target `r_{t+1} + \gamma \hat V^\pi(s_{t+1})` uses only
one reward and one bootstrap, so its variance is much smaller per
step; but the bootstrap is biased — `\hat V^\pi` is not yet correct,
and that incorrectness propagates into every TD target.

Two large structural consequences flow from this. First, TD updates
can happen **online**: the moment we see `(s, a, r, s')`, we can
update. We do not need to wait for the episode to end. Second, TD
applies to **continuing tasks** with no terminal state at all. Pure
MC fundamentally cannot: there is no episode end at which to compute
a return. TD's bootstrap lets us learn value functions for systems
that go forever.

The variance reduction is real but **horizon-dependent**. On
long-horizon problems with many per-step rewards, the MC variance
grows with horizon and TD's per-step variance does not — TD wins
decisively. On short-horizon problems, the MC variance is small to
begin with, and TD's bootstrap bias can be the dominant error
source. Lesson 7's gridworld is in the second regime; we will see
the empirical consequence in Section 2.

---

> **Forward link.** The bootstrap move generalizes immediately to
> function approximation (Lesson 9): replace `\hat V^\pi(s')` with
> `V_{\theta^-}(s')`, where `\theta^-` is a periodically-updated
> target network. The same bootstrap, now with the target network
> providing the "current estimate" that gets plugged in. The
> bias-variance trade-off survives intact; the "deadly triad"
> (function approximation + bootstrapping + off-policy) is the
> stability cost.

---

**Visualization V1 — The Bootstrap Move.**

A single, focused animation. Left panel: the MC target,
shown as a full trajectory rolling out from `s_t` to the terminal
state, with the return `G_t = r_{t+1} + \gamma r_{t+2} + \cdots` being
assembled step by step as the trajectory finishes. Right panel: the
TD target, shown as the same trajectory but with the algorithm
"deciding" at step `t+1` to substitute `\hat V^\pi(s_{t+1})` for the
future and producing the target immediately, while the trajectory
continues without affecting the update for `s_t`.

The animation runs in sync on both sides. At every step, the left
side accumulates discounted rewards into `G_t`. The right side
freezes `\hat V^\pi(s_{t+1})` as the bootstrap target the moment it
sees `s_{t+1}`. The visual statement is "MC needs to finish; TD
doesn't." Width 880, height 380.

---

### Section 2 — TD(0) Prediction and Robbins-Monro

**Tagline:** *The simplest bootstrap, and the stochastic-approximation theory that makes it converge.*
**Length:** ~900 words.
**Anchor:** `td-zero-prediction` and `robbins-monro`.

---

**Prose:**

The simplest TD algorithm is **TD(0)**, also called one-step TD: at
every step, apply the TD update with α as a step size and the
one-step bootstrap target.

```text
TD(0) prediction:
    V(s) ← 0 for all s
    for episode = 1, ..., N:
        observe s
        until terminal:
            take action a ~ π(·|s)
            observe r, s'
            V(s) ← V(s) + α [r + γ V(s') - V(s)]
            s ← s'
```

The algorithm is online: each `(s, a, r, s')` triggers exactly one
update. There is no episode buffer, no batch processing, no episode
boundary required for correctness.

---

**Convergence: TD(0) is a Robbins-Monro stochastic approximation.**
The Robbins-Monro framework treats stochastic algorithms that
update an estimate `\theta_n` toward a noisy observation of a target:

$$
\theta_{n+1} \;=\; \theta_n + \alpha_n (Y_n - \theta_n),
$$

where `Y_n` is an unbiased noisy observation of some unknown quantity
`\theta^*`. Under the **Robbins-Monro conditions** on the step sizes,

$$
\sum_{n=1}^\infty \alpha_n \;=\; \infty, \qquad \sum_{n=1}^\infty \alpha_n^2 \;<\; \infty,
$$

the estimate `\theta_n` converges almost surely to `\theta^*`. The
classical example is the running mean, where `\alpha_n = 1/n`
satisfies both conditions and `Y_n` is a fresh sample.

TD(0) has the same structure. The update `V(s) \leftarrow V(s) + \alpha [r + \gamma V(s') - V(s)]`
moves `V(s)` toward `r + \gamma V(s')`, which is a noisy observation
of `(\mathcal T^\pi V)(s)` — the Bellman operator applied to the
current estimate. Two complications relative to the simple running
mean: the target `\mathcal T^\pi V` depends on `V`, so the target
moves as the estimate moves; and the noise on the target is not
independent across updates (it depends on the sampled trajectory).
Both complications are addressed by the convergence theory.

**Tsitsiklis (1994)** proved that TD(0) converges to `V^\pi` almost
surely under three conditions: every state is visited infinitely
often, the Robbins-Monro conditions on `\alpha_n(s)` hold for every
state, and the rewards have bounded variance. The proof uses two
ingredients we have already developed. First, the Bellman operator
`\mathcal T^\pi` is a `\gamma`-contraction in the sup norm
(Lesson 4); its unique fixed point is `V^\pi`. Second, the
stochastic approximation lemma extends from "noisy observations of a
fixed target" to "noisy observations of a moving target driven by a
contraction." The combination guarantees convergence.

> **Pedagogical aside on Robbins-Monro choices.** The condition
> `\sum \alpha_n = \infty` ensures the estimator can travel an
> unbounded distance; `\sum \alpha_n^2 < \infty` ensures the noise
> accumulates only finitely. A constant `\alpha` satisfies the
> first but violates the second — and indeed constant-α TD(0) does
> not converge to `V^\pi`; it oscillates around `V^\pi` with
> variance proportional to `\alpha`. This is correct, important, and
> often overlooked.

---

**Empirical: TD(0) on the gridworld.** On the running 3×3 gridworld
with uniform random policy (true `V^\pi(0,0) = -0.4205`), TD(0) with
constant `α = 0.1` produces:

| `N` (episodes) | mean of estimator | std of estimator | RMSE  |
|---------------:|------------------:|-----------------:|------:|
| 100            | -0.4087           | 0.0481           | 0.0496|
| 500            | -0.4382           | 0.0539           | 0.0566|
| 2,000          | -0.4450           | 0.0296           | 0.0385|
| 5,000          | -0.4255           | 0.0450           | 0.0453|

The estimator's standard deviation does not shrink monotonically with
`N`: it bottoms out around `\sqrt{\alpha / 2} \cdot \sigma \approx 0.04`
and oscillates around `V^\pi` with that asymptotic spread. This is
the constant-α behavior the Robbins-Monro framework predicts.

**Step-size sensitivity (N=5000, 20 trials):**

| `α`   | mean of estimator | std of estimator | RMSE   |
|------:|------------------:|-----------------:|-------:|
| 0.01  | -0.4266           | 0.0156           | 0.0168 |
| 0.05  | -0.4294           | 0.0312           | 0.0325 |
| 0.10  | -0.4255           | 0.0450           | 0.0453 |
| 0.20  | -0.4338           | 0.0742           | 0.0754 |

The standard deviation grows roughly linearly in `α`, as predicted.
Smaller `α` gives lower asymptotic variance but slower convergence;
larger `α` gives faster convergence but larger asymptotic variance.
The trade-off has no free lunch with constant `α`. A decaying
schedule satisfying the Robbins-Monro conditions (e.g., `α_n = 1/n`
or `α_n = c / (c + n)`) avoids it: the estimator both reaches `V^\pi`
and stays there.

---

**A surprising comparison: TD(0) vs MC on this gridworld.** From
Lesson 7, MC at N=2000 has std ≈ 0.011 (50 trials). TD(0) at the
same N with `α = 0.1` has std ≈ 0.030. **MC has lower variance than
TD(0) here.** This contradicts the textbook claim that "TD has lower
variance than MC." The textbook claim is true on long-horizon problems
where MC's per-trajectory variance grows large. On short-horizon
problems (this gridworld has mean episode length 8.7), MC's per-trajectory
variance is small, and TD's constant-α oscillation dominates the
comparison.

The honest summary: TD's variance advantage over MC scales with the
horizon and the per-step reward variance. On short, low-noise
problems, the two are comparable or MC wins. On long, noisy
problems, TD wins decisively. Modern deep RL operates almost entirely
in the second regime, which is why TD dominates practice.

---

**Visualization V2 — TD(0) Prediction Explorer.**

A four-panel layout. Panel A: the 3×3 gridworld with cells colored
by the current `\hat V^\pi` estimate (diverging palette from Lesson 3),
updating cell by cell as each TD update fires. Panel B: a per-state
convergence trace showing `\hat V^\pi(0, 0)` vs episode count, with
horizontal reference at `-0.4205` and shaded asymptotic band of
width `\sqrt{\alpha / 2} \cdot \sigma`. Panel C: a single TD update
animation, showing `s`, `r`, `s'`, the bootstrap target `r + \gamma V(s')`
in cyan, the TD error `δ` in red, and the new `V(s)` after the
update. Panel D: a step-size sensitivity plot, showing the mean and
std of `\hat V^\pi(0, 0)` at N=5000 over a range of `α` values. The
plot reproduces the table above visually.

Controls: `α` slider (linear 0 to 0.5), Robbins-Monro toggle (constant
α vs `α_n = 1/n(s)` vs `α_n = 1/\sqrt{n(s)}`), policy picker
(uniform random vs ε-greedy(Q*) vs all-down vs all-right), reset.
Width 920, height 540.

---

### Section 3 — SARSA: On-Policy TD Control

**Tagline:** *Bootstrap with the action the policy will actually take next.*
**Length:** ~800 words.
**Anchor:** `sarsa`.

---

**Prose:**

We move from prediction (estimate `V^\pi` for a fixed `\pi`) to
control (find a good `\pi`). The MC control story from Lesson 7
generalized to two flavors: Exploring Starts (converges to `Q^*`,
assumes unrealistic initial-state distribution) and ε-greedy
(converges to `Q^\pi_{\epsilon\text{-soft}}`, not `Q^*` unless ε is
annealed via GLIE). TD control has the same structure with one extra
twist: we get an additional, **off-policy** algorithm (Q-learning,
Section 4) that no MC method easily replicates.

**SARSA** is the on-policy TD control algorithm. Its name comes from
the quintuple `(S, A, R, S', A')` of data used in each update: the
agent observes state `s`, takes action `a`, observes reward `r` and
next state `s'`, then samples the **next action** `a'` from its
current ε-greedy policy. The update rule is

$$
Q(s, a) \;\leftarrow\; Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right].
$$

The bootstrap target uses `Q(s', a')` — the value of the action the
policy will actually take next. This makes SARSA **on-policy**: the
target reflects the policy's actual behavior.

```text
SARSA:
    Q(s, a) ← 0 for all s, a
    for episode = 1, ..., N:
        observe s
        a ~ ε-greedy(Q, s)
        until terminal:
            take action a; observe r, s'
            a' ~ ε-greedy(Q, s')
            Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]
            s, a ← s', a'
```

The greedy policy implied by `Q` at the end of training is the
algorithm's output.

---

**SARSA converges to `Q^\pi_{\epsilon\text{-soft}}`, not `Q^*`.**
Because SARSA bootstraps from the actually-sampled next action `a'`,
which is ε-greedy, its target reflects the exploration noise that ε
introduces. The fixed point of the SARSA update is the action-value
function of the ε-soft optimal policy — the same target MC ε-greedy
converged to in Lesson 7. With ε = 0.1 on the running gridworld:

$$
Q^\pi_{\epsilon\text{-soft}}(0, 0, \text{right}) \;=\; 0.6307,
$$

which is strictly less than `Q^*(0, 0, \text{right}) = 0.7290`. The
gap is the cost of exploration.

Empirically with `ε = 0.1` and `α = 0.1` (50,000 episodes, 10 seeds):

| Action | SARSA estimate | Q^π_{ε-soft} target | Q* (for reference) |
|:-------|---------------:|--------------------:|-------------------:|
| up     | ~0.46          | 0.5646              | 0.6561             |
| right  | ~0.56          | 0.6307              | 0.7290             |
| down   | ~0.50          | 0.6307              | 0.7290             |
| left   | ~0.46          | 0.5646              | 0.6561             |

The SARSA estimates are noticeably below `Q^π_{ε-soft}` because
constant-α SARSA does not fully converge to its asymptotic mean. With
GLIE step-size scheduling, the estimates approach the middle column.
With ε annealed to zero (GLIE on ε), the algorithm converges to `Q^*`
(the right column).

---

**GLIE for SARSA.** To recover `Q^*` from SARSA, anneal ε to zero
slowly enough that every state-action pair is still visited infinitely
often. The standard choice is `ε_n = 1/n^c` for some `c \in (0, 1)`.
Under GLIE, SARSA converges to `Q^*` and the deterministic optimal
policy. This is the direct TD analog of the MC GLIE result from
Lesson 7.

> **Connection to Lesson 7.** The Q^π_{ε-soft} convergence target of
> SARSA is identical to the Q^π_{ε-soft} convergence target of MC
> ε-greedy control. They are two algorithms for the same fixed point.
> The numerical anchor `Q^π_{ε-soft}(0, 0, \text{right}) = 0.6307`
> appears in both lessons.

---

**A practical advantage of SARSA: behavior matches what it learns.**
Because SARSA's target reflects the policy's actual ε-greedy
behavior, the values SARSA learns are accurate descriptions of the
ε-greedy agent's actual performance, *including* its exploration
mistakes. Q-learning (next section) learns Q*, which describes the
performance of a different policy — the optimal one — that the agent
is not actually following.

For deployment scenarios where you will continue to use exploration
at inference time (most real systems), SARSA's Q values are more
useful than Q-learning's. This becomes consequential in safety-critical
settings (cliff-walking is the canonical example) where the
"optimistic" Q-learning estimates can lead to behavior that crashes
during training.

---

**Visualization V3 — TD Algorithm Lab (Centerpiece — described below).**

The unified comparator that shows TD(0), SARSA, Q-learning, and n-step
TD running on the same gridworld simultaneously. SARSA's piece of
the lab is fully described in the V3 specification below.

---

### Section 4 — Q-Learning: Off-Policy TD Control

**Tagline:** *Bootstrap with the action the optimal policy would take. Sidestep importance sampling entirely.*
**Length:** ~900 words.
**Anchor:** `q-learning` and `off-policy-td`.

---

**Prose:**

**Q-learning** changes exactly one line of SARSA. Instead of bootstrapping
with `Q(s', a')` — where `a'` is the action the *behavior* policy will
take — Q-learning bootstraps with `\max_{a'} Q(s', a')`, the value of
the action the *optimal* policy would take. The update rule:

$$
Q(s, a) \;\leftarrow\; Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right].
$$

The behavior policy that generates the data can be anything with
sufficient exploration (typically ε-greedy with respect to `Q`).
The bootstrap target uses the max over actions, which is the **target
policy** — the implicit optimal policy that `Q` is trying to learn.

```text
Q-learning:
    Q(s, a) ← 0 for all s, a
    for episode = 1, ..., N:
        observe s
        until terminal:
            a ~ ε-greedy(Q, s)
            take action a; observe r, s'
            Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]
            s ← s'
```

The greedy policy with respect to `Q` is the output.

---

**Q-learning converges to `Q^*`, regardless of behavior policy
(within mild conditions).** This is one of the most consequential
results in RL. As long as the behavior policy visits every (s, a)
pair infinitely often, and the Robbins-Monro step-size conditions
hold, Q-learning's `Q` converges almost surely to `Q^*`. Unlike SARSA
— which converges to `Q^\pi_{behavior}` and is therefore "biased
toward the behavior policy" — Q-learning is unaffected by *which*
exploration scheme the agent uses, as long as that scheme has
sufficient coverage.

Empirically with `ε = 0.1` and `α = 0.1` (10,000 episodes, 20 seeds):

| Action | Q-learning estimate | Q* target | Q^π_{ε-soft} (for reference) |
|:-------|--------------------:|----------:|-----------------------------:|
| up     | **0.6561**          | 0.6561    | 0.5646                        |
| right  | **0.7290**          | 0.7290    | 0.6307                        |
| down   | 0.5905              | 0.7290    | 0.6307                        |
| left   | **0.6561**          | 0.6561    | 0.5646                        |

Q(0,0,right) = 0.7290 to four decimal places — Q-learning has
recovered Q* exactly along the greedy path. The off-path action
"down" is underestimated because the agent rarely chooses it (it is
not greedy) and so its Q-value receives few updates. This is fine
for the algorithm's purpose: actions along the greedy path are
correctly valued.

---

**The off-policy / IS-bypass moment.** Lesson 6 developed the
trajectory importance sampling apparatus for off-policy correction:
to estimate `V^{\pi_t}` using trajectories sampled under `\pi_b`,
multiply each return by the trajectory weight `\rho_{0:T-1}`. Lesson
7 cashed this apparatus in for off-policy MC, with the
consequence that variance grows exponentially in the horizon. At
N=10,000 only 39 of 10,000 trajectories contributed to the estimate;
99.6% of the sampling budget was wasted.

Q-learning **does not use importance sampling at all**. The
off-policy correction comes from a different source: the `\max`
operator. The Bellman optimality equation `Q^*(s, a) = \mathbb{E}[R + \gamma \max_{a'} Q^*(s', a')]`
involves a max over actions, not an expectation over the policy. Once
you have `Q^*`, the optimal policy is `\arg\max_a Q^*(s, a)` — but
you don't need the optimal policy's action distribution to *learn*
`Q^*`; you only need to observe enough `(s, a, r, s')` transitions
to bootstrap.

The variance benefit is enormous. Q-learning's per-update variance is
governed by the one-step reward variance and the local Q estimates,
not by the horizon and not by the divergence between behavior and
target policies. Compared to off-policy MC with weights of 256 on
0.4% of trajectories, Q-learning's update variance is comparable to
on-policy TD(0). The IS apparatus from Lesson 6 is bypassed
completely for this special case.

> **The structural insight.** Q-learning works as an off-policy
> algorithm because the Bellman *optimality* equation can be
> expressed without reference to any specific policy: it is a
> property of the MDP itself, encoded as a max over actions. The
> Bellman *expectation* equation, by contrast, is policy-specific —
> it reduces to the policy's action distribution — and any
> off-policy variant of expectation-based bootstrapping (i.e.
> Expected SARSA off-policy) needs to handle the policy mismatch
> somehow. The two classical handling methods are importance sampling
> (Lesson 6's machinery) and the operator-based bypass (Q-learning's
> max). The operator-based bypass works specifically and only for
> the optimal-value problem.

---

**Maximization bias.** Q-learning has a structural weakness:
`\max_{a'} Q(s', a')` uses noisy estimates inside a maximum, and the
expected value of `max of noisy estimates` is greater than the max of
expected values. This systematic over-estimation is called
**maximization bias** and is observable in textbook MDPs.

Consider the canonical Sutton-Barto figure 6.5 setup. State A is the
start; the agent has two actions: "right" goes to terminal with
reward 0; "left" goes to state B. From B, there are 10 actions, each
leading to terminal with reward `\mathcal N(-0.1, 1)`. The optimal
action from A is **right** (expected reward 0 > -0.1).

Q-learning's behavior on this MDP:

| Episodes | Fraction of episodes choosing 'left' from A | Optimal fraction |
|---------:|--------------------------------------------:|------------------|
| 100      | 0.432                                       | ~0.05 (ε/2)      |
| 300      | 0.354                                       | ~0.05            |

Q-learning systematically over-estimates `\max_{a'} Q(B, a')`
because each noisy `Q(B, a')` estimate has its own positive deviation
sometimes, and the max selects whichever happens to be highest. The
algorithm therefore over-values "left from A" and chooses it far more
often than optimal.

**Double Q-learning** (Hasselt 2010) fixes maximization bias by
maintaining two independent `Q` estimates and using one to *select*
the maximizing action and the other to *evaluate* it. The cross-evaluation
breaks the systematic over-estimation. Double Q-learning is the basis
for Double DQN in Lesson 9.

---

**Visualization V4 — SARSA vs Q-Learning Side-by-Side.**

A two-panel comparator. Left: SARSA. Right: Q-learning. Both running
on the same gridworld with the same exploration schedule, started
from the same seed for direct comparison. Each panel shows the
current `Q(s, \cdot)` for a selected state as a small bar chart, the
greedy policy as arrows on the gridworld, and the learning curve
`\hat V^{\pi_{\text{greedy}}}(0, 0)` vs episode count with reference
lines at `V^*(0, 0) = 0.7290` and `V^π_{ε-soft}(0, 0) = 0.6274`.

The pedagogical moment: SARSA's curve climbs toward the lower
reference; Q-learning's climbs toward the higher one. The gap
**remains** as episodes accumulate (it does not shrink with N). A
toggle adds a third panel with the maximization-bias demo: the
counter showing the fraction of "left from A" choices, with a
reference line at the optimal 5% and Q-learning's actual curve
hovering around 35-45%. Width 920, height 460.

---

### Section 5 — n-step TD: Interpolating Between TD(0) and MC

**Tagline:** *Look ahead n steps before bootstrapping. A continuous family between TD(0) and MC.*
**Length:** ~700 words.
**Anchor:** `n-step-td`.

---

**Prose:**

TD(0) bootstraps after one step. MC waits for the entire episode.
Between these extremes is a continuous family of algorithms that
look ahead `n` steps before bootstrapping. The **n-step return** is

$$
G_{t}^{(n)} \;:=\; r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^{n-1} r_{t+n} + \gamma^n V(s_{t+n}).
$$

For `n = 1` this is the TD(0) target `r_{t+1} + \gamma V(s_{t+1})`.
As `n \to T - t` (length of remaining episode), the bootstrap term
vanishes and `G_t^{(n)}` becomes the MC return `G_t`. For intermediate
`n`, the algorithm mixes observed rewards (no bias from those) with a
bootstrap term (bias from that).

The **n-step TD update**:

$$
V(s_t) \;\leftarrow\; V(s_t) + \alpha [G_t^{(n)} - V(s_t)].
$$

Implementation requires a small buffer: the algorithm must wait `n`
steps after a state is visited before it can compute the update for
that state. Once the episode ends, all remaining updates are MC
(no bootstrap, since the future is fully known).

---

**The bias-variance trade-off across `n`.** Small `n` has high bias
(strongly relies on the bootstrap, which is wrong) and low variance
(few rewards in the sum). Large `n` has low bias (uses many observed
rewards) and high variance (those rewards are all sampled). The
optimal `n` depends on the problem.

The conventional textbook intuition is that the trade-off is
**U-shaped**: there is an interior optimum where bias and variance
are jointly minimized, neither at `n = 1` nor at `n = T`. This is
true on problems with sufficiently long horizons and noisy rewards.
It is **not** universally true: on short-horizon, low-noise problems,
the variance from accumulating rewards is small and bias dominates,
making smaller `n` better.

Empirical on the running gridworld (N=2000 episodes, α=0.1, 20
trials):

| `n` | mean of estimator | std of estimator | RMSE   |
|----:|------------------:|-----------------:|-------:|
| 1   | -0.4450           | 0.0296           | 0.0385 |
| 2   | -0.4634           | 0.0715           | 0.0833 |
| 4   | -0.4664           | 0.1235           | 0.1317 |
| 8   | -0.5003           | 0.1561           | 0.1753 |
| 16  | -0.4933           | 0.2145           | 0.2265 |
| 100 | -0.4918           | 0.2232           | 0.2343 |
| (MC | -0.4216           | 0.0107           | 0.0107)|

The RMSE grows monotonically with `n` on this gridworld. The
horizon is too short for the textbook U-shape to emerge. Note also
that MC (separate row, computed with the proper running-mean
estimator from Lesson 7) has the lowest RMSE of all — for short
episodes, MC's per-trajectory variance is small enough that the
constant-α TD methods cannot match it.

The lesson reports both regimes honestly. On long-horizon problems
(typically `T > 50` with noisy rewards), n-step TD's U-shape becomes
visible, with optimal `n` typically between 4 and 16. This is the
regime modern deep RL operates in, which is why n-step variants are
widely deployed.

---

> **Forward link to GAE (Lesson 10).** The Generalized Advantage
> Estimator (GAE) is a weighted exponential average of n-step
> advantages, controlled by a parameter `\lambda \in [0, 1]`. With
> `\lambda = 0` it reduces to a one-step advantage (TD(0)-style);
> with `\lambda = 1` it reduces to the MC advantage. The exponential
> weighting is the n-step generalization that policy gradient
> methods actually use, and it descends directly from the TD(λ)
> apparatus we develop in Section 6.

---

**Visualization V5 — n-step Backup Diagrams and Interpolation Curve.**

A two-panel layout. Left: a stack of backup diagrams for n = 1, 2,
4, 8, MC, showing visually that each diagram adds another observed
reward and pushes the bootstrap further into the future. The TD(0)
diagram has one observed reward and one bootstrap; the MC diagram
has all observed rewards and no bootstrap; intermediate diagrams
interpolate.

Right: the interpolation curve — mean RMSE vs `n` on the gridworld,
with error bars from 20 trials. The curve shows the monotone growth
seen in the table. A toggle adds a second curve from a longer-horizon
synthetic MDP (e.g., a 20-step chain with noisy rewards) where the
U-shape is visible. The contrast is the pedagogical moment: the
optimal `n` is problem-dependent, not algorithm-intrinsic.
Width 880, height 480.

---

### Section 6 — TD(λ) and Eligibility Traces

**Tagline:** *A geometric mixture of all n-step returns. Implemented efficiently with eligibility traces.*
**Length:** ~800 words.
**Anchor:** `td-lambda` and `eligibility-traces`.

---

**Prose:**

n-step TD with a fixed `n` solves the bias-variance trade-off by
picking a particular point on the trade-off curve. **TD(λ)** takes a
different approach: instead of picking one `n`, take a weighted
geometric average of all n-step returns. The **λ-return** is

$$
G_t^\lambda \;:=\; (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} G_t^{(n)},
$$

with `\lambda \in [0, 1]`. The weights `(1 - \lambda) \lambda^{n-1}`
form a geometric distribution over `n`. With `\lambda = 0`, all weight
is on `n = 1` and `G_t^\lambda = G_t^{(1)}` (TD(0)). With
`\lambda = 1`, all weight goes to `n = \infty` and `G_t^\lambda = G_t`
(MC). For intermediate `\lambda`, the λ-return mixes near and far
bootstraps, with farther bootstraps weighted geometrically less.

The TD(λ) **forward view** update is

$$
V(s_t) \;\leftarrow\; V(s_t) + \alpha [G_t^\lambda - V(s_t)].
$$

This is conceptually clean but computationally awkward: `G_t^\lambda`
requires looking forward arbitrarily far in time, which conflicts
with TD's online property.

---

**The backward view: eligibility traces.** TD(λ) can be implemented
**online** using **eligibility traces**. The trace `e_t(s)` measures
how recently and how often each state has been visited:

$$
e_t(s) \;:=\; \begin{cases} \gamma \lambda e_{t-1}(s) + 1 & \text{if } s = s_t \\ \gamma \lambda e_{t-1}(s) & \text{otherwise} \end{cases}
$$

The trace decays multiplicatively by `\gamma \lambda` at every step
and bumps by 1 every time its state is visited. The TD(λ) update is
then

$$
V(s) \;\leftarrow\; V(s) + \alpha \delta_t e_t(s) \quad \forall s,
$$

where `\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)` is the one-step
TD error. Every state updates by an amount proportional to its
current trace. The trace concentrates the update on recently-visited
states and gradually distributes it backward through the trajectory.

**Equivalence of forward and backward views.** It is a foundational
theorem of RL (Sutton 1988, Sutton and Barto 2018 ch. 12) that the
backward-view TD(λ) and the forward-view TD(λ) produce the same
expected updates, summed over the episode. The forward view is the
*theoretical* characterization; the backward view is the *practical*
implementation.

---

**TD(λ) as a unifying framework.** The λ parameter unifies TD(0) and
MC into a single algorithm with a single dial. λ = 0 → TD(0); λ = 1
→ MC; λ = 0.5 → a mixture that is often the practical sweet spot.
The trace-based implementation gives O(|S|) work per step
(updating every state's trace and value once), which is the same
order as TD(0) — there is no extra cost to using λ > 0.

The λ-return generalizes immediately to action-values, giving
SARSA(λ), and to off-policy settings with importance corrections,
giving the family of off-policy TD(λ) algorithms (Retrace, V-trace,
the algorithms underlying IMPALA). These appear in Lessons 9 and 15.

> **A subtlety about off-policy TD(λ).** The backward-view trace
> works elegantly for on-policy methods. For off-policy methods,
> the trace must be modified to account for the importance weights
> at each step, and the variance of the trace can blow up if those
> weights are unbounded. Truncation schemes (Retrace, V-trace) cap
> the per-step weights to prevent variance explosion. The general
> pattern is that TD(λ)'s elegance survives but requires care in
> the off-policy regime — exactly the regime where Lessons 6 and 7
> warned us about importance sampling's variance.

---

**Visualization V6 — TD(λ) Eligibility Trace Visualizer.**

A two-panel animation. Left: the agent traversing the gridworld
under a fixed policy, with each cell's color encoding its current
eligibility trace `e_t(s)` (amber heatmap, fading multiplicatively by
`\gamma \lambda` per step). Right: the running V-value updates,
with each cell flashing in red when its TD error update fires, and
the heatmap of the diverging V palette underneath showing the
accumulated `\hat V`.

Controls: `λ` slider (0 to 1, with reference markers at 0, 0.5, 1),
`α` slider, policy picker, speed slider. The pedagogical moment is
the slider: at λ = 0 only the current state lights up in the trace
(TD(0) behavior); at λ = 1 the trace persists for the entire
trajectory (essentially MC); at λ = 0.5 a smooth tail decays through
the recent cells (the eligibility-trace sweet spot).

Width 880, height 460.

---

### Section 7 — Convergence Theory: Cashing in Contraction Mappings

**Tagline:** *Why TD(0) converges: the Bellman operator is a contraction, and stochastic approximation handles the noise.*
**Length:** ~600 words.
**Anchor:** `td-convergence`.

---

**Prose:**

The convergence proofs for TD methods sit at the intersection of two
threads we have already developed.

**Thread 1: the Bellman operator is a γ-contraction.** From Lesson 4
we know that the Bellman expectation operator `\mathcal T^\pi V` and
the Bellman optimality operator `\mathcal T V` are γ-contractions in
the sup norm `\|\cdot\|_\infty`. The Banach fixed-point theorem
guarantees they have unique fixed points, `V^\pi` and `V^*`
respectively. The DP algorithms from Lesson 5 (value iteration,
policy iteration) work because they iterate these contractions
exactly and converge to the fixed points.

**Thread 2: stochastic approximation handles noise.** TD methods do
not iterate the Bellman operator exactly. They iterate noisy versions
of it: each update is `V(s) \leftarrow V(s) + \alpha (Y - V(s))`,
where `Y` is a sampled approximation of `(\mathcal T^\pi V)(s)`.
Under the Robbins-Monro conditions on `\alpha`, this stochastic
iteration converges to the same fixed point as the deterministic
iteration would. This is the **Robbins-Monro stochastic approximation
lemma** (Section 2 introduced it for the prediction case).

**The combined result (Tsitsiklis 1994).** TD(0) converges almost
surely to `V^\pi` if every state is visited infinitely often, the
Robbins-Monro conditions hold per-state, and rewards have bounded
variance. The proof combines the contraction argument (the target
`\mathcal T^\pi V` doesn't shift faster than the estimator can chase
it) with the stochastic approximation argument (noise is averaged
out by the decaying step sizes). The combination is precisely the
machinery developed in Lessons 4 and Section 2 of this lesson.

**Q-learning convergence (Watkins and Dayan 1992).** The same
machinery works for Q-learning, with one extra detail: the bootstrap
target `\max_{a'} Q(s', a')` is the Bellman *optimality* operator
applied to `Q`, and that operator is also a γ-contraction. Q-learning
converges almost surely to `Q^*` under the analogous conditions: every
(s, a) pair visited infinitely often, Robbins-Monro on `\alpha`,
bounded reward variance.

**Where convergence breaks.** The clean convergence story falls
apart in two important settings. First, **function approximation**
(Lesson 9): if `V` is parameterized by a neural network and updated
by gradient descent, the projected Bellman operator may not be a
contraction in the relevant norm, and "TD divergence" can occur.
Second, **off-policy with importance corrections** (Retrace, V-trace,
some safe-RL methods): if importance weights are unbounded, the
stochastic approximation noise no longer averages out cleanly, and
the convergence rate can degrade arbitrarily. The "deadly triad" of
function approximation + bootstrapping + off-policy is precisely the
intersection of these two issues, and it is the central technical
challenge of deep RL.

> **The clean tabular case is a reference point, not a deployment
> reality.** Real systems use function approximation, real data is
> often off-policy, and the convergence proofs do not apply directly.
> The deep-RL literature has spent thirty years patching this:
> target networks (Mnih et al. 2015), conservative weight constraints
> (Munos et al. 2016), Polyak averaging (Lillicrap et al. 2015).
> Each is a tactic that recovers some of the tabular convergence
> guarantees in approximate form. The tabular case is what we know
> for sure; everything else is "approximately, with these caveats."

---

**Visualization V7 — Maximization Bias Demo.**

A focused single-panel animation of the Sutton-Barto figure 6.5 MDP.
The agent's choices at state A (left vs right) are tallied
episode-by-episode and plotted as a running fraction over time. A
reference line at the optimal 5% (ε/2). Q-learning's curve hovers
between 30% and 50%. A toggle adds a Double Q-learning curve that
collapses back toward 5% within a few hundred episodes. The
pedagogical moment is the gap, which is the visible signature of
maximization bias and motivates the Double DQN treatment in
Lesson 9.

Width 720, height 360.

---

### Section 8 — Where You'll See This Again

**Tagline:** *Four downstream lessons build directly on TD.*
**Length:** ~400 words.
**Anchor:** `td-forward-links`.

---

**Prose:**

Four lessons cash in this one directly.

**Lesson 9 (Function Approximation and DQN)** is the most direct
descendant. DQN is Q-learning with two modifications: the Q function
is parameterized by a neural network, and a target network provides
a delayed bootstrap to stabilize the dynamics. The convergence-theory
section of Lesson 9 picks up exactly where Section 7 of this lesson
left off: the projected Bellman operator's contraction properties,
the deadly triad, and the engineering tactics (target networks,
experience replay, prioritized replay) that recover stability.
Double DQN, dueling DQN, and rainbow DQN are all direct extensions.

**Lesson 10 (Policy Gradient)** uses TD machinery for the value
baseline that reduces gradient variance. The advantage function
`A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)` is estimated by some flavor of
TD, and the n-step / GAE machinery from Sections 5 and 6 of this
lesson is exactly what GAE generalizes. Policy gradient's "critic"
in actor-critic methods is a TD value function.

**Lesson 11 (TRPO and PPO)** continues the actor-critic line and uses
GAE explicitly. The bias-variance tuning via λ that we developed
here for TD(λ) is the same dial that PPO exposes via its GAE-λ
parameter.

**Lesson 15 (Offline RL)** revisits the off-policy TD setting with
emphasis on data distribution shift. Retrace, V-trace, and the
conservative-Q-learning family are all extensions of the off-policy
TD apparatus introduced here. The maximization-bias concern from
Section 4 becomes critical in offline RL, where the agent cannot
collect new data to correct over-optimistic estimates.

---

**Visualization V8 — Roadmap Mini.** The curriculum's lesson-graph
thumbnail with TD Learning now marked as shipped. Outgoing arrows to
Lessons 9, 10, 11, 13 (SAC, which uses soft TD updates), and 15.
Each arrow's hover popover shows the specific application within
that downstream lesson. Width 720, height 240.

---

## 5. Algorithm and Math Implementation

The TypeScript module `src/td/` is the largest math module in the
curriculum so far, around 350 lines total. Six core functions plus
helpers.

```ts
import type { MDP, Policy, Trajectory } from "../mdp/types";

/** TD(0) prediction. */
export function tdZero(
  mdp: MDP,
  policy: Policy,
  nEpisodes: number,
  alpha: number | ((episode: number, state: number) => number),
  options: { rng?: () => number; maxSteps?: number } = {},
): { V: Float64Array; history: Float64Array } {
  const V = new Float64Array(mdp.nStates);
  const history = new Float64Array(nEpisodes);  // V(s_0) after each episode
  const visitCount = new Int32Array(mdp.nStates);
  const rng = options.rng ?? Math.random;
  const maxSteps = options.maxSteps ?? 200;

  for (let ep = 0; ep < nEpisodes; ep++) {
    let s = mdp.startState;
    for (let step = 0; step < maxSteps; step++) {
      if (mdp.isTerminal(s)) break;
      const a = policy.sampleAction(s, rng);
      const { nextState, reward, done } = mdp.step(s, a);
      visitCount[s]++;
      const aEff = typeof alpha === 'number' ? alpha : alpha(ep, s);
      const target = done ? reward : reward + mdp.gamma * V[nextState];
      V[s] += aEff * (target - V[s]);
      if (done) break;
      s = nextState;
    }
    history[ep] = V[mdp.startState];
  }
  return { V, history };
}

/** SARSA control. On-policy: target uses next action from same ε-greedy policy. */
export function sarsa(
  mdp: MDP,
  nEpisodes: number,
  epsilon: number | ((episode: number) => number),
  alpha: number,
  options: { rng?: () => number; maxSteps?: number } = {},
): { Q: Float64Array; greedyPolicy: Int32Array; history: Float64Array } {
  // Q is flat: index = s * nActions + a
  const Q = new Float64Array(mdp.nStates * mdp.nActions);
  const history = new Float64Array(nEpisodes);
  const rng = options.rng ?? Math.random;
  const maxSteps = options.maxSteps ?? 200;

  const epsAt = typeof epsilon === 'number' ? () => epsilon : epsilon;
  const epsGreedy = (s: number, eps: number) => {
    if (rng() < eps) return Math.floor(rng() * mdp.nActions);
    return argmaxQ(Q, s, mdp.nActions);
  };

  for (let ep = 0; ep < nEpisodes; ep++) {
    const eps = epsAt(ep);
    let s = mdp.startState;
    let a = epsGreedy(s, eps);
    for (let step = 0; step < maxSteps; step++) {
      if (mdp.isTerminal(s)) break;
      const { nextState, reward, done } = mdp.step(s, a);
      const target = done ? reward
                          : reward + mdp.gamma * Q[nextState * mdp.nActions + epsGreedy(nextState, eps)];
      Q[s * mdp.nActions + a] += alpha * (target - Q[s * mdp.nActions + a]);
      if (done) break;
      const aNext = epsGreedy(nextState, eps);
      s = nextState;
      a = aNext;
    }
    history[ep] = computeGreedyValue(Q, mdp.startState, mdp);
  }
  const greedyPolicy = new Int32Array(mdp.nStates);
  for (let s = 0; s < mdp.nStates; s++)
    greedyPolicy[s] = argmaxQ(Q, s, mdp.nActions);
  return { Q, greedyPolicy, history };
}

/** Q-learning. Off-policy: target uses max over actions. */
export function qLearning(
  mdp: MDP,
  nEpisodes: number,
  epsilon: number | ((episode: number) => number),
  alpha: number,
  options: { rng?: () => number; maxSteps?: number } = {},
): { Q: Float64Array; greedyPolicy: Int32Array; history: Float64Array } {
  const Q = new Float64Array(mdp.nStates * mdp.nActions);
  const history = new Float64Array(nEpisodes);
  const rng = options.rng ?? Math.random;
  const maxSteps = options.maxSteps ?? 200;

  const epsAt = typeof epsilon === 'number' ? () => epsilon : epsilon;
  const epsGreedy = (s: number, eps: number) => {
    if (rng() < eps) return Math.floor(rng() * mdp.nActions);
    return argmaxQ(Q, s, mdp.nActions);
  };

  for (let ep = 0; ep < nEpisodes; ep++) {
    const eps = epsAt(ep);
    let s = mdp.startState;
    for (let step = 0; step < maxSteps; step++) {
      if (mdp.isTerminal(s)) break;
      const a = epsGreedy(s, eps);
      const { nextState, reward, done } = mdp.step(s, a);
      const target = done ? reward
                          : reward + mdp.gamma * maxQ(Q, nextState, mdp.nActions);
      Q[s * mdp.nActions + a] += alpha * (target - Q[s * mdp.nActions + a]);
      if (done) break;
      s = nextState;
    }
    history[ep] = computeGreedyValue(Q, mdp.startState, mdp);
  }
  const greedyPolicy = new Int32Array(mdp.nStates);
  for (let s = 0; s < mdp.nStates; s++)
    greedyPolicy[s] = argmaxQ(Q, s, mdp.nActions);
  return { Q, greedyPolicy, history };
}

/** n-step TD prediction. */
export function nStepTD(
  mdp: MDP, policy: Policy, n: number,
  nEpisodes: number, alpha: number,
  options: { rng?: () => number; maxSteps?: number } = {},
): { V: Float64Array; history: Float64Array };

/** TD(λ) prediction with backward-view eligibility traces. */
export function tdLambda(
  mdp: MDP, policy: Policy, lambda: number,
  nEpisodes: number, alpha: number,
  options: { rng?: () => number; maxSteps?: number } = {},
): { V: Float64Array; history: Float64Array; finalTraces: Float64Array };

/** Helper utilities */
function argmaxQ(Q: Float64Array, s: number, nA: number): number { /* ... */ }
function maxQ(Q: Float64Array, s: number, nA: number): number { /* ... */ }
function computeGreedyValue(Q: Float64Array, s: number, mdp: MDP): number { /* ... */ }
```

**Vitest test targets** (from pre-verified Python):

```ts
test('TD(0) at α=0.1, N=2000 produces V(0,0) within 0.05 of -0.4205', () => {
  // 20-trial replication, mean within 0.05 of -0.4205
  const mdp = makeGridworld();
  const policy = uniformRandomPolicy(mdp);
  const trials = Array.from({ length: 20 }, (_, t) =>
    tdZero(mdp, policy, 2000, 0.1, { rng: seeded(t) }).V[stateIdx(0, 0)]
  );
  const mean = trials.reduce((a, b) => a + b) / 20;
  expect(mean).toBeCloseTo(-0.4205, 1);
});

test('TD(0) std grows linearly with α (constant-α oscillation)', () => {
  const std01 = measureTDStd(0.01, 5000, 20);
  const std10 = measureTDStd(0.10, 5000, 20);
  // std should grow roughly proportional to sqrt(α)
  expect(std10 / std01).toBeGreaterThan(2);
  expect(std10 / std01).toBeLessThan(5);
});

test('SARSA at ε=0.1, α=0.1 converges toward Q^π_ε-soft (NOT Q*)', () => {
  // At N=10000, Q(0,0,right) should be in [0.5, 0.65], NOT near 0.7290
  const { Q } = sarsa(mdp, 10000, 0.1, 0.1, { rng: seeded(0) });
  const qRight = Q[stateIdx(0, 0) * 4 + RIGHT];
  expect(qRight).toBeGreaterThan(0.45);
  expect(qRight).toBeLessThan(0.65);
  expect(qRight).not.toBeCloseTo(0.7290, 1);
});

test('Q-learning at ε=0.1, α=0.1 converges exactly to Q* along greedy path', () => {
  const { Q } = qLearning(mdp, 10000, 0.1, 0.1, { rng: seeded(0) });
  expect(Q[stateIdx(0, 0) * 4 + RIGHT]).toBeCloseTo(0.7290, 2);
  expect(Q[stateIdx(0, 0) * 4 + UP]).toBeCloseTo(0.6561, 2);
});

test('Q-learning policy on gridworld matches optimal', () => {
  const { greedyPolicy } = qLearning(mdp, 20000, 0.1, 0.1, { rng: seeded(0) });
  // At (0,0), greedy should be RIGHT (action 1)
  expect(greedyPolicy[stateIdx(0, 0)]).toBe(RIGHT);
});

test('n-step TD with n=1 equals TD(0) given same RNG', () => {
  const { V: v1 } = tdZero(mdp, policy, 1000, 0.1, { rng: seeded(42) });
  const { V: vn } = nStepTD(mdp, policy, 1, 1000, 0.1, { rng: seeded(42) });
  for (let s = 0; s < mdp.nStates; s++) {
    expect(vn[s]).toBeCloseTo(v1[s], 6);
  }
});

test('TD(λ) with λ=0 equals TD(0) given same RNG', () => {
  const { V: v1 } = tdZero(mdp, policy, 1000, 0.1, { rng: seeded(42) });
  const { V: vL } = tdLambda(mdp, policy, 0, 1000, 0.1, { rng: seeded(42) });
  for (let s = 0; s < mdp.nStates; s++) {
    expect(vL[s]).toBeCloseTo(v1[s], 4);  // tolerance for trace numerics
  }
});

test('Maximization bias: Q-learning chooses left from A >25% of episodes', () => {
  const mdp = makeMaxBiasMDP();
  const fractions = Array.from({ length: 20 }, (_, t) => {
    const log = qLearningWithChoiceLog(mdp, 300, 0.1, 0.1, { rng: seeded(t) });
    return log.leftFraction;
  });
  const meanFrac = fractions.reduce((a, b) => a + b) / 20;
  expect(meanFrac).toBeGreaterThan(0.25);  // should be ~0.05 if no bias
});
```

---

## 6. Component Catalog

| Code | Component                       | Section | Polish budget |
|-----|---------------------------------|---------|--------------|
| V1  | `<BootstrapMove>`               | §1      | 1 day        |
| V2  | `<TDZeroExplorer>`              | §2      | 1.5 days     |
| V3  | `<TDAlgorithmLab>`              | §3-5    | **4-5 days** (centerpiece) |
| V4  | `<SARSAvsQLearning>`            | §4      | 1.5 days     |
| V5  | `<NStepBackupDiagrams>`         | §5      | 1 day        |
| V6  | `<TDLambdaTraceVisualizer>`     | §6      | 1.5 days     |
| V7  | `<MaximizationBiasDemo>`        | §7      | 1 day        |
| V8  | `<RoadmapMini>` (update)        | §8      | 0.5 day      |

Total polish budget around twelve days, in line with the master plan's
estimate of "around fifteen hundred lines, eight sections, eight
visualizations." The centerpiece V3 (`TDAlgorithmLab`) is the
polish-budget sink at 4-5 days; it must support four algorithms
(TD(0), SARSA, Q-learning, n-step TD) running synchronously on the
same gridworld with shared seed control.

**Reuse from prior lessons:** all of `src/mdp/`, `policyEvaluationExact`
and `bellmanBackup` from `src/dp/`, the MC implementations from
`src/monte-carlo/` for the MC reference comparator in V1, V2, V3.
`GridworldRenderer`, `MathBlock`, `CrosslinkCallout`, `PanelChrome`,
`RoadmapMini`. New code is ~350 lines in `src/td/` plus the
visualizations.

---

## 7. Page-Level User Experience

Same conventions as prior lessons. Single-page scroll, prereq strip at
top, reduced-motion support is important for V1 (the bootstrap-move
animation), V3 (the centerpiece's synchronous updates), and V6 (the
eligibility trace decay animation).

A specific UX note for V3: the lab supports four algorithms running on
the same gridworld synchronously. The user should be able to start
all four with the same seed and watch them diverge. The
"convergence target" annotation is critical: SARSA, Q-learning, MC
ε-greedy, and MC ES all converge to different fixed points, and the
lab should annotate each algorithm's curve with its target value
(0.6307 for SARSA, 0.7290 for Q-learning, etc.) so the learner can
see the targets are different even when the trajectories cross.

The centerpiece breaks out to 1000 pixels wide (the widest in the
curriculum so far), reflecting four parallel algorithm panels.

---

## 8. Acceptance Criteria

After completing this lesson, a learner should be able to do the
following.

State the bootstrap move and explain how it differs from MC. Write
the TD(0) update rule. Define the TD error δ. State the Robbins-Monro
conditions on the step size and explain why constant-α TD does not
converge to V^π. State the SARSA update rule and explain why SARSA
converges to Q^π_ε-soft, not Q*. State the Q-learning update rule
and explain why Q-learning converges to Q* regardless of behavior
policy (within mild coverage conditions). Explain the IS-bypass
intuition: why Q-learning sidesteps the importance sampling apparatus
of Lesson 6. Describe maximization bias and the structural reason
behind it.

Write the n-step return G_t^{(n)} and explain how it interpolates
between TD(0) (n=1) and MC (n=∞). Write the λ-return G_t^λ as a
geometric average of n-step returns. State the eligibility-trace
update rule and explain how it implements the backward view of
TD(λ).

State Tsitsiklis (1994)'s TD(0) convergence theorem and identify the
two ingredients: contraction (from Lesson 4) and Robbins-Monro (from
Section 2). State the analogous Q-learning convergence theorem. Name
the "deadly triad" and the settings where convergence guarantees
break.

A concrete acceptance check: hand the learner the cliff-walking MDP
(Sutton-Barto figure 6.4) and ask them to predict the qualitative
behavior of SARSA and Q-learning. SARSA learns a "safer" path that
avoids the cliff edge (because it accounts for ε-greedy exploration
mistakes); Q-learning learns the cliff-edge optimal path (because it
ignores the agent's actual behavior). The lab in V4 can demonstrate
this contrast directly.

---

## 9. Stretch Goals (post-MVP)

**Cliff-walking comparison.** The cliff-walking MDP from Sutton-Barto
figure 6.4 makes the SARSA vs Q-learning distinction vivid. Adding it
as an alternative MDP in V4 (a toggle between gridworld and
cliff-walking) would let the learner see the policy difference
directly, not just the Q-value difference.

**Double Q-learning.** Section 4 teases Double Q-learning as a fix
for maximization bias and defers full treatment to Lesson 9. A
stretch version of V7 could include Double Q-learning as a side-by-side
curve, showing the bias collapse within a few hundred episodes.

**Expected SARSA.** Between SARSA (uses one sampled next action) and
Q-learning (uses max over next actions) sits Expected SARSA, which
uses the expected next-action Q-value under the policy. It has lower
variance than SARSA and is on-policy. Worth a callout in Section 3 or
a stretch sub-section.

**Off-policy TD with importance corrections.** The full Retrace /
V-trace family is left for Lesson 15. A teaser sub-section could
show how IS weights enter TD updates and where the variance issues
arise.

**SARSA(λ) and Q(λ).** Eligibility traces generalize cleanly to
SARSA and Q-learning. A stretch sub-section in Section 6 could
develop SARSA(λ) and contrast it with Watkins's Q(λ) (which has
to handle the trace cutoff at non-greedy actions).

---

## 10. Out of Scope (intentionally)

**Function approximation.** All algorithms in this lesson are
tabular. Function-approximation TD (linear-feature TD, neural TD,
DQN) is the entirety of Lesson 9 and is not previewed here beyond
forward links.

**Continuous-time TD.** All algorithms are discrete-time. The
continuous-time analog (TD with stochastic differential equations,
Doya 2000) is out of scope.

**Off-policy n-step TD and TD(λ) with full IS corrections.** Mentioned
briefly in Section 6 but the full treatment (Retrace's truncation
scheme, V-trace's clipping) is deferred to Lesson 15.

**Average-reward TD.** All discussion is discounted-reward (γ < 1).
The average-reward formulation has its own TD machinery and is out
of scope.

**Adaptive step-size methods (RMSProp, Adam) for TD.** These are
properties of the optimizer rather than the TD algorithm itself, and
are covered in Lesson 9 as part of the deep-RL story.

---

## 11. Training Notebook

Not applicable. No models are trained for this lesson. The script
`scripts/td_traces.py` (around 130 lines) pre-computes the
convergence statistics tables, the SARSA / Q-learning learning
curves, the n-step interpolation curves, and the maximization-bias
demo, all into JSON files under `public/data/td/` for instant
initial render. The in-browser TypeScript reproduces these statistics
live when the user interacts.

---

## 12. Closing Notes and Length Tally

Total length: roughly fifteen hundred lines. The longest lesson in
the curriculum so far. The length is justified: this is the central
classical RL lesson, and it must develop six algorithm families
(TD(0), SARSA, Q-learning, n-step TD, TD(λ), maximization-bias
demo), fold in the Robbins-Monro stochastic approximation framework
inline, cash in the contraction-mapping theory from Lesson 4, and
prime the deep-RL transition that occupies Lesson 9. Cutting any
section would degrade the lesson's role as critical path.

The centerpiece V3 (`TDAlgorithmLab`) is the lesson's polish-budget
sink: a four-algorithm comparator with synchronized rollouts under a
shared seed. Its key pedagogical feature is exposing the
convergence-target differences directly — SARSA toward 0.6274,
Q-learning toward 0.7290, both reproducible from the same data
stream — so that the on-policy / off-policy distinction is a visible
fact about the algorithms rather than an abstract textbook claim.

The forward links are saturated: Lessons 9 (most direct cash-in), 10,
11, 13, and 15 are all cited explicitly in Section 8 with specific
applications named. Lesson 9 in particular will spend its opening as
a continuation of this lesson's Section 7 (the deadly triad
discussion), making TD Learning and Function Approximation a tightly
coupled pair.

## End of specification