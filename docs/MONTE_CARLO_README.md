# Lesson 7 — Monte Carlo Methods

> **The first model-free lesson.** We finally drop the assumption that the
> transition kernel is known. All we have is the ability to roll out
> trajectories under some policy, and we want to extract from those rollouts
> an estimate of the value function. Monte Carlo methods are the simplest
> way to do this: sample trajectories, compute returns, average. The lesson
> covers first-visit and every-visit MC for policy evaluation, MC control
> via generalized policy iteration (both Exploring Starts and ε-greedy
> flavors), and off-policy MC using the importance sampling apparatus from
> Lesson 6. The recurring contrast is with Temporal-Difference learning,
> which arrives in Lesson 8 and trades MC's zero bias for substantial
> variance reduction.

> **Where this slots in.** Between Lesson 6 (Importance Sampling) and
> Lesson 8 (Temporal-Difference Learning). The role is critical-path
> bridge: this is where the curriculum's first model-free algorithm
> arrives, and every subsequent lesson assumes you know what "estimate a
> value function from sampled trajectories" means. The lesson cashes in
> three pieces of upstream content at once — the MDP machinery from
> Lesson 3, the policy-evaluation problem statement and ground-truth
> values from Lesson 5, and the trajectory IS apparatus from Lesson 6 —
> and primes the contrast with TD learning that will dominate Lesson 8.

---

## 0. Pedagogical Philosophy

Six commitments specific to this lesson.

The first commitment is that MC is the first algorithm the learner has
seen that does not require knowing the model. Lessons 3 through 5 all
assumed full access to the transition probabilities and reward function.
Dynamic programming computed values by direct backups using `P` and `R`;
even the Bellman operators are defined as integrals over `P`. Monte Carlo
is the first time the learner sees an algorithm that uses only what the
agent could actually observe in the wild: a sequence of states, actions,
and rewards. We should mark this transition explicitly. The lesson opens
by stating what changed, what got harder, and why a perfectly natural
algorithm (sample, compute, average) has the surprising property of
working at all.

The second commitment is that we treat first-visit and every-visit MC as
two natural answers to the same question, not as competitors. Both
converge to the true value function under the same conditions. The
difference shows up only in finite-sample bias and variance, and in
practice the bias of every-visit MC vanishes quickly enough that it is
rarely the deciding factor. We will show this empirically rather than
arguing it abstractly.

The third commitment is that MC control gets the full GPI treatment. The
master plan calls for "MC control via generalized policy iteration with
ε-greedy improvement," and we extend that to also cover Exploring Starts
as the cleaner theoretical case. The reason for covering both is that
Exploring Starts converges to the *true* Q\* but requires an unrealistic
assumption (uniform initial state-action distribution), while ε-greedy
control converges only to the Q-function of the ε-soft optimal policy
(not the same as Q\* in general). The gap between these two outcomes —
and the GLIE schedule that closes it — is one of the lesson's intellectual
moments.

The fourth commitment is that off-policy MC is presented as a direct
application of Lesson 6, not as a rederivation. We have just spent a
lesson on the trajectory IS apparatus. The off-policy MC section's job
is to cash that apparatus in, name the algorithm, point out the parts
that get worse (variance still exponential in horizon, ESS still tiny
for divergent policies), and connect to the IS lesson's gridworld
numerics. The same trajectories, the same weights, the same 256, the
same convergence at N=10000. The learner should feel a direct payoff
from Lesson 6.

The fifth commitment is that the lesson sets up the TD comparison
explicitly without doing TD's job. Section 6 names MC's variance problem,
names TD's bootstrapping move, and gestures at the bias-variance trade-off
between them. It is deliberately short and ends with a "see Lesson 8"
pointer. The temptation to introduce TD here should be resisted. TD is
hard enough to deserve its own lesson; the contrast is what we want now.

The sixth commitment is that the running gridworld continues. Same MDP,
same start state, same goal, same pit, same γ = 0.9. Every numerical
table in this lesson connects to the corresponding table in Lessons 5
and 6. V^π(0,0) under uniform random has been -0.4205 since Lesson 5
and will be -0.4205 here. V\*(0,0) has been 0.7290 since Lesson 3 and
will be 0.7290 here. The accumulating familiarity with this MDP is
worth more than novelty.

---

## 1. Tech Stack

The tech stack is unchanged from prior lessons. Vite plus TypeScript in
strict mode, KaTeX for math, D3 version 7 for visualizations,
`ml-matrix` lightly used, Vitest for tests. No new dependencies.

This lesson reuses substantial prior infrastructure. The gridworld and
trajectory sampling come from `src/mdp/` (Lesson 3). The ground-truth
value function reference comes from `policyEvaluationExact` in
`src/dp/` (Lesson 5). The importance sampling estimators come from
`src/importance-sampling/` (Lesson 6). New code in `src/monte-carlo/`
is roughly 150 lines: first-visit and every-visit policy evaluation,
MC control with ε-greedy, MC control with Exploring Starts, and
off-policy MC for both ordinary and weighted IS variants.

A small Python script, `scripts/mc_traces.py` at around eighty lines,
pre-computes the convergence statistics tables (first-visit vs
every-visit at various N, on-policy vs off-policy at various N, MC
control learning curves with ε-greedy vs Exploring Starts). The
outputs land as JSON in `public/data/mc/` for instant initial render.
The in-browser TypeScript reproduces these statistics live when the
user interacts.

---

## 2. Visual and Aesthetic Direction

The curriculum aesthetic continues unchanged. The lesson adds a small
set of tokens specific to the Monte Carlo theme.

```css
:root {
  /* Visit-counting modes */
  --mc-first-visit:   #0e7490;   /* cyan-700   | first-visit MC */
  --mc-every-visit:   #0891b2;   /* cyan-600   | every-visit MC; slightly brighter */

  /* Sampling modes */
  --mc-on-policy:     #15803d;   /* green-700  | on-policy: behavior == target */
  --mc-off-policy:    #ea580c;   /* orange-600 | off-policy: behavior != target */

  /* Estimator overlays */
  --mc-truth:         #1c1e22;   /* black ref  | true value function */
  --mc-running-mean:  #6d28d9;   /* violet-700 | running estimate */
  --mc-band:          rgba(109, 40, 217, 0.18);  /* violet at 18% | confidence band */

  /* Control-phase markers */
  --mc-explore:       #b45309;   /* amber-700  | ε-greedy exploration (rhymes with Bandits ε-greedy) */
  --mc-greedy:        #15803d;   /* green-700  | greedy choice */
  --mc-glie:          #6d28d9;   /* violet-700 | GLIE schedule indicator */
}
```

The visit-counting modes get a tight cyan pair: same hue, slightly
different brightness. The visual statement is "these are siblings, not
opponents." The on-policy/off-policy pair takes the green/orange split
that has already been established in the IS lesson — green for "no
correction needed" and orange for "needs the IS apparatus." This is
the same green-for-baseline and orange-for-proposal convention from
Lesson 6, applied to a different axis.

The violet for running mean and the violet-tinted confidence band rhyme
with weighted IS from Lesson 6 (violet was the principled-but-biased
estimator) and Thompson sampling from Lesson 1 (violet for principled
Bayesian estimation under uncertainty). All three uses share the
intuition "violet means a running estimate of something uncertain that
is converging to truth."

The control-phase amber/green pair rhymes directly with the Bandits
lesson: ε-greedy was amber in Bandits and is amber here. The "GLIE
schedule" indicator is violet because GLIE is the principled schedule
that gives convergence guarantees, paralleling the violet-for-principled
pattern.

---

## 3. Lesson Metadata

```ts
export const lessonMeta: LessonMeta = {
  id: "monte-carlo",
  title: "Monte Carlo Methods",
  subtitle: "Learning a value function from sampled trajectories",
  tier: 1,
  difficulty: 3,
  estimatedReadMinutes: 55,
  role: "critical-path",
  prerequisites: [
    { lesson: "mdps",                 anchor: "state-value-function" },
    { lesson: "mdps",                 anchor: "action-value-function" },
    { lesson: "dynamic-programming",  anchor: "iterative-policy-evaluation" },
    { lesson: "dynamic-programming",  anchor: "policy-improvement" },
    { lesson: "importance-sampling",  anchor: "trajectory-is" },
    { lesson: "importance-sampling",  anchor: "weighted-is-estimator" },
  ],
  exportedAnchors: [
    "model-free-setting",
    "mc-return-estimate",
    "first-visit-mc",
    "every-visit-mc",
    "mc-policy-evaluation",
    "mc-control-exploring-starts",
    "mc-control-eps-greedy",
    "glie-schedule",
    "off-policy-mc",
    "mc-vs-td",
  ],
  centerpieceComponent: "MCEstimatorLab",
  forwardLinksWhenReady: [
    { to: "td-learning",      anchor: "td-zero-prediction" },
    { to: "td-learning",      anchor: "sarsa" },
    { to: "td-learning",      anchor: "q-learning" },
    { to: "td-learning",      anchor: "n-step-td" },
    { to: "function-approximation", anchor: "deep-q-networks" },
    { to: "offline-rl",       anchor: "offline-mc-evaluation" },
  ],
};
```

---

## 4. Section-by-Section Plan

### Section 1 — The Model-Free Setting

**Tagline:** *No transition probabilities. Just trajectories and returns.*
**Length:** ~600 words.
**Anchor:** `model-free-setting`.

---

**Prose:**

The dynamic programming methods of Lesson 5 all share an assumption: the
agent has full access to the MDP's transition kernel `P(s' \mid s, a)`
and reward function `R(s, a)`. Policy evaluation computes `V^\pi` by
solving a linear system involving `P_\pi` and `R_\pi`. Value iteration
backs up `\max_a \sum_{s'} P(s' \mid s, a) [R(s, a, s') + \gamma V(s')]`
at every step. Even the asynchronous and Gauss-Seidel variants assume
that for any `(s, a)` we can enumerate the next-state distribution.

In most real settings this assumption is not satisfied. We do not have a
closed-form transition kernel for a robot, for a video game, for a
language model. What we have is the ability to **interact** with the
environment: pick an action, observe a reward, observe a next state.
The agent collects sequences of the form

$$
\tau \;=\; (s_0, a_0, r_1, s_1, a_1, r_2, s_2, \ldots, s_T)
$$

where the rewards and next states are drawn from the environment's
underlying (but inaccessible) dynamics. The challenge is to extract from
these sequences enough information to do the work that DP previously
did with `P` and `R` directly.

This is the **model-free** setting. The agent does not know the model,
does not learn the model, and uses only sampled experience. The flavor
of methods that work here is fundamentally statistical: estimate
expectations by averages, accept the variance that comes with finite
samples, and design algorithms whose convergence relies on the law of
large numbers rather than on a contraction in a known operator.

Two large families of model-free methods will occupy the curriculum
from this lesson forward. **Monte Carlo methods** wait until the end
of an episode and use the empirical return as a sample of `V^\pi(s)`.
**Temporal-Difference methods** (Lesson 8) use a bootstrap: estimate
`V^\pi(s)` from `r + \gamma \hat V^\pi(s')`, plugging the agent's
current estimate into the right-hand side. MC has zero bias and high
variance. TD has bias from bootstrapping and substantially lower
variance. This lesson is about the MC family. Lesson 8 is about TD.

---

**The return as a noisy observation of the value.** The state value
function `V^\pi(s)` is defined as the expected discounted return when
starting in state `s` and following policy `\pi`:

$$
V^\pi(s) \;:=\; \mathbb{E}_\pi\!\left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} \,\bigg|\, S_0 = s \right].
$$

For an episodic task, the sum terminates at the end of the episode.
Crucially, if we *sample* a trajectory `\tau` starting from `s` under
policy `\pi`, the random variable

$$
G_0(\tau) \;:=\; \sum_{t=0}^{T-1} \gamma^t R_{t+1}
$$

is an **unbiased estimate of `V^\pi(s)`**. Take many independent
trajectories, average their returns, and by the law of large numbers
the average converges to `V^\pi(s)`. That single observation is the
entire idea of Monte Carlo policy evaluation.

The complication is that we usually want `V^\pi(s)` for *every* state
`s`, not just one. We could run an independent set of trajectories from
each starting state, but that wastes data: each trajectory visits many
states, and the return from each visited state is also a sample of that
state's value. Section 2 takes up the question of how to use a single
trajectory to update value estimates for all states it visits.

---

> **Forward-link callout.** *Function approximation and the deadly
> triad.* In Lesson 9 we will face a setting where we cannot maintain a
> separate `V(s)` for every state — there are too many states — and
> must instead parameterize `V` with a neural network. The Monte Carlo
> estimator generalizes cleanly to that setting: the return is still a
> noisy observation of `V^\pi(s)` and can be used as a regression
> target. The TD estimator, in contrast, runs into the deadly triad
> when combined with bootstrapping and off-policy learning. One reason
> MC remains useful despite its variance is that it sidesteps the
> deadly triad entirely.

---

**Visualization V1 — Model-Free vs Model-Based Side-by-Side.**

A two-column comparison. Left column: a small diagram of the DP backup,
showing the MDP's transition kernel `P` and reward function `R` as
explicit inputs to the Bellman expectation backup. Right column: a
small diagram of the MC update, showing only a trajectory `\tau =
(s_0, a_0, r_1, s_1, \ldots)` as input, with the computed return `G_0`
as the value estimate. The two diagrams share the same gridworld
illustration in the background but differ in what they consume.

The point of the visualization is not to introduce new content but to
mark visually the change in what the algorithm has access to. After
five lessons of seeing `P` and `R` explicitly, the learner needs a
clear moment of "we no longer use those." Width 720, height 280.

---

### Section 2 — First-Visit and Every-Visit Monte Carlo

**Tagline:** *Two natural conventions for using a trajectory's returns. Both work.*
**Length:** ~700 words.
**Anchor:** `first-visit-mc` and `every-visit-mc`.

---

**Prose:**

Consider a single trajectory `\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \ldots, s_T)`.
For any time step `t`, the return from `t` onward is

$$
G_t \;:=\; \sum_{k=0}^{T - t - 1} \gamma^k R_{t + k + 1}.
$$

`G_t` is computed recursively from the rewards by the backward sweep
`G_T = 0`, `G_t = R_{t+1} + \gamma G_{t+1}`. We compute all the `G_t`
in a single pass through the trajectory in reverse.

Now consider a state `s` that the trajectory visits. The trajectory
visits `s` at one or more time steps; call these times
`t_1 < t_2 < \cdots < t_k`. At each visit, the return `G_{t_i}` is a
sample of `V^\pi(s)`. We must decide what to do with these `k`
samples. Two natural choices:

**First-visit Monte Carlo.** Use only the return from the *first* time
the trajectory visited `s`. Discard subsequent visits within the same
trajectory. Over many trajectories, the first-visit returns are
independent (across trajectories) and identically distributed (each is
a sample of `V^\pi(s)` under `\pi`), and their average is an unbiased,
consistent estimator of `V^\pi(s)`.

**Every-visit Monte Carlo.** Use the return from *every* time the
trajectory visited `s`. Within a single trajectory the multiple
returns from `s` are not independent, but their average across many
trajectories still converges to `V^\pi(s)`. Every-visit MC is biased
at any finite `N` (the within-trajectory correlation shows up as
non-zero bias) but the bias is `O(1/N)` and vanishes asymptotically.

---

**On-policy first-visit MC convergence.** On the running gridworld
with uniform random policy `\pi` (and discount γ = 0.9), the true
value function from `(0, 0)` is `V^\pi(0, 0) = -0.4205` (computed
exactly via `policyEvaluationExact` in Lesson 5). The standard
deviation of a single episode's return from `(0, 0)` is
approximately `0.41`. By the central limit theorem, the standard
deviation of the first-visit MC estimator after `N` episodes is
approximately `0.41 / \sqrt N`.

Empirically across 50 independent trials, first-visit MC produces:

| `N` (episodes) | mean of estimator | std of estimator | RMSE from truth |
|---------------:|------------------:|-----------------:|----------------:|
| 100            | -0.4236           | 0.0438           | 0.0439          |
| 1,000          | -0.4235           | 0.0133           | 0.0137          |
| 10,000         | -0.4206           | 0.0036           | 0.0036          |

The empirical standard deviations match the theoretical `0.41 / \sqrt N`
prediction: 0.041, 0.013, 0.0041. The MC estimator converges at the
expected `O(1 / \sqrt N)` rate, with the constant of proportionality
determined by the variance of a single episode's return.

---

**First-visit vs every-visit empirical comparison.** On the same
gridworld, with uniform random policy and `N = 1000` episodes,
averaged over 50 trials:

| Method        | mean of estimator | std of estimator |
|:--------------|------------------:|-----------------:|
| First-visit   | -0.4235           | 0.0133           |
| Every-visit   | -0.4214           | 0.0131           |

The two estimators agree to two decimal places. Every-visit's bias of
about `0.002` at `N = 1000` is real but small; it vanishes as `N`
grows. The variance of the two estimators is essentially identical for
this problem. For longer-horizon, more loop-prone MDPs the difference
becomes more visible, but the qualitative picture — both work, with
small finite-sample differences — holds generally.

---

**Implementation: incremental updates.** Maintaining a list of returns
for each state and recomputing the mean at every episode is wasteful.
The standard incremental formulation maintains a running mean and a
visit count:

$$
N(s) \leftarrow N(s) + 1, \qquad \hat V^\pi(s) \leftarrow \hat V^\pi(s) + \frac{1}{N(s)} \left( G_t - \hat V^\pi(s) \right).
$$

This is the classical "running average" update. The factor `1/N(s)` is
a step size that decays as more samples accumulate. We could
generalize to a constant step size `\alpha`,

$$
\hat V^\pi(s) \leftarrow \hat V^\pi(s) + \alpha \left( G_t - \hat V^\pi(s) \right),
$$

which gives an exponential moving average rather than a true average.
This becomes important for non-stationary settings (when `\pi` is
changing during learning, as in MC control). The same step-size
machinery will reappear in Lesson 8 as the foundation for TD's
learning rate.

---

**Visualization V2 — First-Visit vs Every-Visit Walkthrough.**

A single trajectory rendered on the gridworld at a controlled pace,
roughly 600ms per step. As the agent moves cell to cell, each state's
visit count increments. Below the trajectory animation, two side-by-side
running tables — one for first-visit MC, one for every-visit MC —
showing the current value estimate `\hat V` per state. Each cell of the
gridworld is colorized by its current `\hat V` value using the
diverging palette from Lesson 3. The contrast between first-visit and
every-visit becomes visible on trajectories that loop through the
same state multiple times. A reset button resamples the trajectory.

Beneath both tables, a running plot of `\hat V(0, 0)` over the past
500 trajectories for both methods, both converging toward the
horizontal reference at -0.4205. Width 800, height 460.

---

### Section 3 — MC Policy Evaluation (Centerpiece)

**Tagline:** *Sample trajectories, accumulate returns, watch the value function fill in.*
**Length:** ~900 words.
**Anchor:** `mc-policy-evaluation`.

---

**Prose:**

Given a fixed policy `\pi`, the MC policy evaluation problem is to
estimate `V^\pi(s)` for every state `s` using sampled trajectories.
The algorithm is short: sample `N` trajectories under `\pi`, compute
the per-step returns, and for each state visited update its running
estimate via the first-visit (or every-visit) update from Section 2.

```text
MC_PolicyEvaluation(π, N, first_visit=True):
    V(s) ← 0 for all s
    N(s) ← 0 for all s
    for episode = 1, ..., N:
        sample trajectory τ ~ π
        compute returns G_0, G_1, ..., G_{T-1}
        visited ← ∅
        for t = 0, ..., T - 1:
            s ← s_t
            if first_visit and s ∈ visited:
                continue
            visited ← visited ∪ {s}
            N(s) ← N(s) + 1
            V(s) ← V(s) + (1/N(s)) · (G_t - V(s))
    return V
```

The algorithm converges to `V^\pi` under mild conditions: every state
must be visited infinitely often as `N \to \infty` (a coverage
condition on `\pi` and the environment), and the rewards must have
finite variance. Under these conditions, `\hat V^\pi(s) \xrightarrow{a.s.} V^\pi(s)` for every state `s`.

---

**Convergence rate.** As established in Section 2, the per-state
convergence rate is `O(1 / \sqrt{N(s)})`, where `N(s)` is the number
of episodes that visited `s`. For states the policy visits frequently,
`N(s)` grows roughly linearly in `N` and the estimator converges at
`O(1 / \sqrt N)`. For states the policy visits rarely, `N(s)` may
grow much more slowly, and the per-state estimator converges
correspondingly more slowly. This non-uniform convergence rate is one
of MC's structural features. It is also why off-policy MC will become
so painful in Section 5: the target policy may want to evaluate states
that the behavior policy visits rarely or never.

---

**The full gridworld value function under uniform random.** From
Lesson 5 the exact `V^\pi(s)` under uniform random policy is:

|      | col 0   | col 1   | col 2   |
|-----:|:-------:|:-------:|:-------:|
| row 0 | -0.4205 | -0.5139 | -0.2386 |
| row 1 | -0.5139 |  0.0000 | -0.0693 |
| row 2 | -0.2386 | -0.0693 |  0.0000 |

(Where (1, 1) is the pit with terminal reward -1 and (2, 2) is the
goal with terminal reward +1; both terminal states have value 0 by
convention.)

Running first-visit MC for `N = 10000` episodes starting from `(0, 0)`
produces estimates within 0.01 of every cell. Running for `N = 100`
episodes produces estimates with per-cell standard deviations of
roughly 0.04 to 0.10, depending on how frequently each cell is
visited. The hardest cells to estimate are the ones the policy visits
least often, which on this MDP are the corner cells far from the
start.

---

**Centerpiece visualization V3 — MC Estimator Lab.**

The polish-budget sink of the lesson; allocate three to four days.

A six-panel synchronized layout. The lab supports both on-policy and
off-policy modes via a top-level toggle, and exposes the qualitative
difference between them as its central pedagogical statement.

**Panel A — Policy Picker.** Two side-by-side gridworld renderings.
In on-policy mode, only one is shown: the policy being evaluated.
In off-policy mode, both are shown: behavior `\pi_b` on the left
and target `\pi_t` on the right. Dropdown for each policy:
uniform random, deterministic optimal, ε-soft optimal with adjustable ε,
ε-soft uniform-ish, all-down, all-right. The matching-trajectory rate
(probability that a behavior trajectory has non-zero target weight) is
displayed prominently in monospace.

**Panel B — Live Estimate Heatmap.** The 3×3 gridworld, with each
cell's color driven by the current `\hat V^\pi` estimate, using the
diverging palette from Lesson 3. Cells transition smoothly as
estimates update episode-by-episode. A toggle switches between first-
visit and every-visit MC. In off-policy mode, a sub-toggle switches
between ordinary IS and weighted IS estimators. The ground-truth
`V^\pi` (computed via `policyEvaluationExact` for verification) is
shown as a smaller heatmap below for direct comparison.

**Panel C — Trajectory Counter and Visit Counts.** A live tally of
trajectories sampled so far, and per-state visit counts. The visit
counts let the learner see the non-uniform per-state convergence
directly: states visited frequently have tighter estimates than
states visited rarely.

**Panel D — Per-State Convergence Trace.** A single state's value
estimate over time, with the user able to click any cell of the
gridworld to inspect that cell's trace. The trace shows the running
mean `\hat V^\pi(s)` versus the trajectory count `n`, with a
horizontal reference line at the true `V^\pi(s)`. In off-policy mode
the trace shows both ordinary IS and weighted IS in cyan and violet
respectively, both converging toward the reference.

**Panel E — Estimator Distribution Across Trials.** A live boxplot
showing the distribution of `\hat V^\pi(s_0)` across 50 independent
trials at the current `N`. In on-policy mode, a single box. In
off-policy mode, two boxes side by side for ordinary and weighted IS.
The pre-computed JSON for the default configurations populates this
panel instantly; live configurations recompute in 1–2 seconds.

**Panel F — Sample Trajectory.** One trajectory rendered as the
agent moves cell to cell, with the running return `G_0`, the running
importance weight `\rho_{0:t}` (in off-policy mode only), and any
relevant visit-count updates shown. After the trajectory ends, a
"sample again" button generates a new one.

**Controls strip:** policy selection (driving Panel A); estimator type
(first-visit / every-visit) and IS type (ordinary / weighted) in
off-policy mode; `N` slider on log scale ranging through 100, 1000,
10000, 50000; γ slider; speed slider for Panel F's trajectory
animation; and a reset button.

The central pedagogical moment of the lab is the transition the
learner can drive themselves: start in on-policy mode, watch the
gridworld colorize smoothly as episodes accumulate, observe the
estimator distributions in Panel E being tight. Then flip to
off-policy mode (default: uniform behavior, deterministic optimal
target). The visit counts on most cells go to zero or near-zero. The
estimator distributions in Panel E for ordinary IS get wide. The
weighted IS estimator collapses to a single point. This is the same
emotional climax as Lesson 6's centerpiece, here in the MC framing.

Width 960 pixels (centerpiece breakout). Height 820 pixels.

---

### Section 4 — MC Control via Generalized Policy Iteration

**Tagline:** *Estimate Q, improve the policy, repeat. Exploring Starts and ε-greedy.*
**Length:** ~800 words.
**Anchor:** `mc-control-exploring-starts` and `mc-control-eps-greedy`.

---

**Prose:**

Policy evaluation by itself is useful, but the larger goal is
**control**: find a good (ideally optimal) policy. The dynamic
programming track in Lesson 5 reached this goal via policy iteration,
alternating between policy evaluation and policy improvement. The
generalized policy iteration (GPI) picture from Lesson 5 emphasized
that this alternation does not need to be exact at either step: we
can do a *partial* policy evaluation followed by a *greedy* policy
improvement and still converge.

MC control instantiates GPI with the policy evaluation step replaced
by Monte Carlo sampling. The two algorithms differ in how the
evaluation step is structured and what policy improvement step is
applied.

---

**MC Control with Exploring Starts.** The simplest version of MC
control makes a strong assumption: every (state, action) pair is
chosen as the initial state-action of some episode with positive
probability. This is the **exploring starts** assumption. Under it,
the algorithm is

```text
MC_ES():
    Q(s, a) ← 0 for all s, a
    π(s) ← arbitrary deterministic policy
    for episode = 1, ..., N:
        s_0, a_0 ← uniformly random from valid (s, a)
        roll out one episode from (s_0, a_0) following π
        compute returns G_0, G_1, ..., G_{T-1}
        for each first visit to (s, a) in episode:
            update Q(s, a) by running-mean rule
            π(s) ← argmax_a Q(s, a)
    return π, Q
```

Under the exploring-starts assumption, MC ES converges to the optimal
deterministic policy `\pi^*` and its action-value function `Q^*`.
Empirically on the running gridworld with 50,000 episodes, MC ES
recovers `Q^*` to within 0.01 of every entry and produces the correct
optimal policy: right at (0,0), right at (0,1), down at (0,2), and so on.

---

**The exploring-starts problem.** In practice we cannot choose initial
states arbitrarily. The agent starts where the environment starts.
The exploring-starts assumption is mathematically convenient but
operationally unrealistic. We need a way to ensure that every (state,
action) gets visited often enough without controlling the initial
condition.

The standard workaround is to make the policy itself **soft**:
always assign positive probability to every action, so that every
action eventually gets explored at every state. The simplest soft
policy is **ε-greedy**: with probability `1 - \epsilon`, pick the
greedy action `\arg\max_a Q(s, a)`; with probability `\epsilon`, pick
uniformly among all actions.

---

**MC Control with ε-greedy.** The same algorithm as MC ES, except
that the policy improvement step picks `\epsilon`-greedy with respect
to the current `Q`, and the initial state is the environment's true
start state:

```text
MC_eps_greedy():
    Q(s, a) ← 0 for all s, a
    for episode = 1, ..., N:
        roll out one episode from s_0 = env_start following ε-greedy(Q)
        compute returns G_0, G_1, ..., G_{T-1}
        for each first visit to (s, a) in episode:
            update Q(s, a) by running-mean rule
    return ε-greedy(Q), Q
```

This is the workhorse MC control algorithm. It converges, but **not
to `Q^*`**. With a fixed `\epsilon`, ε-greedy MC converges to the
Q-function of the ε-soft optimal policy — the best policy *among
ε-soft policies*, which is necessarily worse than the true optimal
deterministic policy whenever `\epsilon > 0`.

Empirically on the running gridworld with `\epsilon = 0.1` and
20,000 episodes, the estimated `Q(0, 0)` is

|             | up     | right  | down   | left   |
|:------------|:------:|:------:|:------:|:------:|
| `Q^*`       | 0.6561 | 0.7290 | 0.7290 | 0.6561 |
| `Q^{ε=0.1}` | 0.5646 | 0.6307 | 0.6307 | 0.5646 |
| MC ε-greedy | ~0.49  | ~0.62  | ~0.53  | ~0.51  |

The MC ε-greedy estimates converge toward the ε-soft optimal
Q-function (middle row), *not* toward `Q^*` (top row). The middle
row is what the algorithm is theoretically targeting; the bottom row
is the empirical estimate from MC at finite N. They are not the same
as `Q^*`.

---

**The GLIE schedule and recovery of `Q^*`.** To recover the true
optimal policy from MC control with soft exploration, we need
**Greedy in the Limit with Infinite Exploration (GLIE)**: a schedule
in which `\epsilon_n \to 0` as `n \to \infty`, but slowly enough that
every (state, action) is still visited infinitely often. A common
choice is `\epsilon_n = 1 / n^c` for some `c \in (0, 1)`. Under
GLIE, MC control converges to `Q^*` and the deterministic optimal
policy.

The same GLIE condition reappears in Lesson 8 (for SARSA and Q-learning
control), Lesson 9 (DQN's `\epsilon`-annealing), and in nearly every
policy-gradient method's exploration schedule. The pattern of
"explore enough, exploit when ready" is fundamental.

> **Forward link to bandits.** GLIE is the multi-state analog of the
> decaying-`\epsilon` schedule from the Bandits lesson (Lesson 1). In
> bandits the decay was justified by the regret bound; here it is
> justified by the convergence-to-optimal argument. The intuition is
> the same: enough exploration to know the world, enough exploitation
> to act on that knowledge.

---

**Visualization V4 — MC Control Learning Curve.**

A two-panel layout showing both MC ES and MC ε-greedy training on
the gridworld in parallel. The top panel shows the learned greedy
policy as action arrows on the gridworld, updating every 100 episodes.
The bottom panel shows learning curves: the value of `\hat V^\pi(0, 0)`
under the algorithm's current greedy policy, plotted versus episode
count, with reference lines at `V^*(0, 0) = 0.7290` and
`V^{\epsilon}_*(0, 0) = 0.6274`. MC ES's curve climbs toward the higher
reference; MC ε-greedy's curve climbs toward the lower one. The gap
between them is the cost of ε-greedy exploration. A toggle adds a
third curve: MC with GLIE schedule `\epsilon_n = 1/\sqrt{n}`, which
also climbs to the higher reference but more slowly than MC ES.
Width 880, height 460.

---

### Section 5 — Off-Policy Monte Carlo via Importance Sampling

**Tagline:** *Estimate one policy's value using another policy's trajectories. Variance returns to haunt us.*
**Length:** ~700 words.
**Anchor:** `off-policy-mc`.

---

**Prose:**

The MC methods of Sections 2 through 4 are **on-policy**: the policy
being evaluated (or improved) is the same as the policy that generates
the trajectories. In the off-policy setting we have trajectories
sampled under a **behavior policy** `\pi_b` and we want to estimate
`V^{\pi_t}` for a different **target policy** `\pi_t`. Lesson 6 built
the trajectory importance sampling apparatus that handles this
problem. We now apply it directly.

---

**The off-policy MC estimator.** Recall from Lesson 6 that for a
trajectory `\tau = (s_0, a_0, r_1, s_1, \ldots, s_T)` sampled under
`\pi_b`, the trajectory importance ratio is

$$
\rho_{0:T-1} \;=\; \prod_{t=0}^{T-1} \frac{\pi_t(a_t \mid s_t)}{\pi_b(a_t \mid s_t)}
$$

and the identity is `V^{\pi_t}(s_0) = \mathbb{E}_{\pi_b}[\rho_{0:T-1} \cdot G_0]`.
The ordinary IS estimator after `N` trajectories is

$$
\hat V^{\pi_t}_{\text{ord}}(s_0) \;=\; \frac{1}{N} \sum_{i=1}^N \rho_{0:T-1}^{(i)} \cdot G_0^{(i)}
$$

and the weighted IS estimator is

$$
\hat V^{\pi_t}_{\text{wt}}(s_0) \;=\; \frac{\sum_{i=1}^N \rho_{0:T-1}^{(i)} \cdot G_0^{(i)}}{\sum_{i=1}^N \rho_{0:T-1}^{(i)}}.
$$

Per-state generalizations (using `\rho_{t:T-1}` rather than `\rho_{0:T-1}`
for the return from time `t`) handle the multi-state case. These are
the trajectory-level estimators from Lesson 6, applied with `G_0`
instead of a generic function `f`.

---

**The gridworld numerics, revisited.** From Lesson 6: behavior =
uniform random, target = deterministic optimal, `V^{\pi_t}(0, 0) = 0.729`.
The probability a uniform-random trajectory matches the deterministic
optimal sequence step-for-step is `(1/4)^4 = 1/256 \approx 0.0039`.
Matching trajectories have weight exactly `4^4 = 256`.

The off-policy MC estimator's empirical behavior across 50 trials:

| `N`    | Non-zero / N | Ordinary IS (mean ± SD) | Weighted IS (mean ± SD) |
|-------:|:------------:|:------------------------|:------------------------|
| 100    | 0.6 / 100    | 0.78 ± 1.19             | 0.25 ± 0.35             |
| 1,000  | 3.8 / 1000   | 0.77 ± 0.34             | 0.71 ± 0.10             |
| 10,000 | 39 / 10000   | 0.75 ± 0.12             | **0.7290 ± 0.0000**     |

The last row matches Lesson 6's centerpiece numerics exactly. At
`N = 10000`, weighted IS converges to exactly `0.7290` with zero
sample variance across all 50 trials, because all matching trajectories
under the deterministic target are identical (same path, same return,
same weight). At small `N`, weighted IS has significant bias (it is
0.25 at N=100, far from the truth of 0.729) because the few matching
trajectories dominate a denominator that has not yet stabilized.

---

**Effective sample size as a diagnostic.** From Lesson 6, the
effective sample size of the IS estimator is

$$
N_{\text{eff}} \;=\; \frac{(\sum_i \rho_i)^2}{\sum_i \rho_i^2}.
$$

For the uniform-to-optimal gridworld at `N = 10000`, every non-zero
weight equals 256 (all matching trajectories are identical), so
`N_{\text{eff}} = 39`, exactly the number of non-zero trajectories.
Of our 10,000 sampled trajectories, only 39 are effectively
contributing to the estimate. Ninety-nine point six percent of the
sampling budget is wasted. This is the off-policy MC variance
problem in numerical form.

---

**Off-policy MC control.** In principle one can do MC control entirely
off-policy: sample under any fixed behavior policy with sufficient
coverage, and use weighted IS to estimate `Q(s, a)` for any target
policy of interest. In practice this is rarely a winning strategy. The
trajectory weight variance compounds; the ESS is tiny; the algorithm
needs absurd amounts of data. Off-policy methods come into their own
only when we move from MC to TD (Lesson 8), where one-step backups
replace trajectory weights with single per-step ratios. The variance
goes from "exponential in horizon" to "bounded by the ratio range,"
which is a transformation of the entire problem.

> **Forward link to Q-learning.** Q-learning sidesteps the off-policy
> ratio entirely by re-deriving the off-policy target as a max over
> actions: instead of `\rho \cdot G`, it uses `r + \gamma \max_{a'} Q(s', a')`,
> which has no IS factor at all. This is one of the most consequential
> tricks in RL and is the central content of Lesson 8.

---

**Visualization V5 — Off-Policy MC vs On-Policy MC.**

A four-panel comparison. Top row: on-policy MC (behavior = target =
uniform random) estimating `V^\pi(0, 0) = -0.4205`. Bottom row:
off-policy MC (behavior = uniform random, target = deterministic
optimal) estimating `V^{\pi_t}(0, 0) = 0.7290`. Each row has a
trajectory-count panel and a convergence-trace panel. The visual
contrast is sharp: the on-policy estimator converges smoothly at the
`1/\sqrt N` rate; the off-policy weighted IS estimator either
"jumps" to 0.729 the first time a matching trajectory is found, or
stays stuck at zero for hundreds of trajectories. Width 880, height 460.

---

### Section 6 — Variance, Bias, and the TD Comparison

**Tagline:** *MC has zero bias and huge variance. TD trades some bias for less variance. Lesson 8 picks up here.*
**Length:** ~500 words.
**Anchor:** `mc-vs-td`.

---

**Prose:**

MC methods have a clean theoretical profile. The MC estimator is
**unbiased**: each return `G_t` is exactly a sample of `V^\pi(s_t)`,
and averaging unbiased samples gives an unbiased estimator. The MC
estimator's **variance**, however, is the entire return's variance.
For a horizon-`T` episode, this is the variance of a sum of `T`
discounted rewards, which generally grows with `T` (more rewards
contribute, each with its own randomness). For long-horizon problems,
MC variance is substantial and growing.

There is a structurally different approach that trades some of MC's
zero bias for substantially lower variance. It is **bootstrapping**:
estimate `V^\pi(s)` not by waiting for the full return, but by
plugging in your current estimate `\hat V^\pi(s')` for the next state
and using `r + \gamma \hat V^\pi(s')` as the target. The result is a
biased estimator — the current `\hat V` is not yet correct — but the
variance is much lower because only one reward and one bootstrap
contribute to the target rather than a full return.

This is the **TD(0)** update, and it is the central content of Lesson 8.
The bias from bootstrapping vanishes as `\hat V \to V^\pi`, and under
suitable step-size schedules (the Robbins-Monro conditions) TD(0)
converges. The variance reduction is often dramatic: ten to a hundred
times smaller than MC on long-horizon problems. The trade-off has been
well studied, and TD's variance advantage typically dominates in
practice.

---

**A picture of the trade-off.** A common visualization is the
**bias-variance plane**, with bias on one axis and variance on the
other:

- MC sits at zero bias, high variance.
- TD(0) sits at moderate bias, low variance.
- `n`-step TD interpolates between them; as `n \to T`, n-step TD
  becomes MC; as `n \to 1`, it becomes TD(0).

The `n`-step TD method (Lesson 8) is one of the most elegant
parameterized families in classical RL: a single dial that smoothly
moves between the two extremes, and the optimal setting depends on
problem-specific properties.

---

**What MC keeps doing well.** Despite TD's variance advantage, MC
remains useful for several reasons. MC is unbiased, which matters when
we need theoretical guarantees about the quality of an estimate. MC
sidesteps bootstrapping, which means it avoids the deadly triad
(Lesson 9) when combined with function approximation. MC is simple to
implement and reason about: no learning rates, no target networks, no
TD error backups, just trajectory averages. And MC is the natural fit
for the offline-RL setting (Lesson 15) where we have a fixed dataset
and want to estimate the value of a policy without the
distributional issues that bootstrapping introduces.

In modern deep RL, pure MC is rare but the MC return is often used
as a high-`n` end of n-step TD, and as a target for the value function
in policy-gradient methods (Lesson 10 uses it directly as the
unbiased gradient target).

---

**Visualization V6 — Bias-Variance Plane Preview.**

A schematic of the bias-variance plane with MC and TD(0) marked as
two points. A curve labeled "n-step TD" interpolates between them. A
small inset shows three sample learning curves on the gridworld: MC
(noisy but unbiased mean), TD(0) (smoother but biased low), and
n=4-step TD (between). The inset has a "forward to Lesson 8" pointer
at the bottom. Width 720, height 360.

---

### Section 7 — Where You'll See This Again

**Tagline:** *Three downstream lessons will build directly on MC.*
**Length:** ~400 words.
**Anchor:** `mc-forward-links`.

---

**Prose:**

Three subsequent lessons depend directly on this one.

**Lesson 8 (Temporal-Difference Learning)** is the immediate next
step. TD methods are presented as a "what if we bootstrap?" variant
of MC. The opening of Lesson 8 explicitly compares the MC update
(`G_t` as the target) with the TD(0) update
(`r + \gamma \hat V^\pi(s')` as the target), and develops n-step TD as
the family of methods between them. The first half of Lesson 8 is
essentially "MC, but with bootstrapping," and the conventions of MC
policy evaluation and control transfer with minor modifications. The
forward-link is the strongest in the curriculum.

**Lesson 9 (Function Approximation and Deep Q-Networks)** uses MC
returns as targets in some of its training regimes. The "deadly triad"
discussion explicitly contrasts MC (which sidesteps the triad because
it does not bootstrap) with TD (which falls into it). The MC return
appears as a baseline target whose stability is the standard against
which the TD methods' instability is judged.

**Lesson 10 (Policy Gradient Methods)** uses the MC return as the
unbiased estimate of `Q^\pi(s, a)` in the simplest REINFORCE
gradient. The policy gradient theorem requires *some* estimate of the
return; the MC return is the simplest unbiased one. Variance reduction
techniques in Lesson 10 (baselines, advantage functions, GAE) are
all aimed at reducing the variance of this MC estimate while keeping
it unbiased or low-bias.

**Lesson 15 (Offline RL)** uses off-policy MC as one baseline method.
The variance problem we documented in Section 5 of this lesson is the
central obstacle of offline RL, and many of the algorithms in Lesson 15
(BCQ, CQL, behavior cloning, density-ratio estimation) are working
around exactly the issue we surfaced here. The "39 effective samples
out of 10,000" gridworld diagnostic from Section 5 is the canonical
illustration of why pure off-policy MC fails in the offline setting.

---

**Visualization V7 — Roadmap Mini.** The curriculum's lesson-graph
thumbnail with Monte Carlo now marked as shipped. Outgoing arrows go
to Lesson 8 (TD Learning), Lesson 9 (Function Approximation and DQN),
Lesson 10 (Policy Gradient), and Lesson 15 (Offline RL). Each arrow's
hover popover shows the specific application within that downstream
lesson. Width 720, height 240.

---

## 5. Algorithm and Math Implementation

The TypeScript module `src/monte-carlo/` is small and self-contained,
around 200 lines total.

```ts
import type { MDP, Policy, Trajectory } from "../mdp/types";
import { sampleTrajectory } from "../mdp/sampling";

/** Compute returns G_0, G_1, ..., G_{T-1} via backward sweep. */
export function computeReturns(rewards: number[], gamma: number): number[] {
  const T = rewards.length;
  const Gs = new Array(T).fill(0);
  let running = 0;
  for (let t = T - 1; t >= 0; t--) {
    running = rewards[t] + gamma * running;
    Gs[t] = running;
  }
  return Gs;
}

/** First-visit Monte Carlo policy evaluation. */
export function mcPolicyEvaluation(
  mdp: MDP,
  policy: Policy,
  nEpisodes: number,
  options: { firstVisit?: boolean; rng?: () => number } = {},
): { V: Float64Array; visits: Int32Array } {
  const firstVisit = options.firstVisit ?? true;
  const V = new Float64Array(mdp.nStates);
  const visits = new Int32Array(mdp.nStates);

  for (let ep = 0; ep < nEpisodes; ep++) {
    const traj = sampleTrajectory(mdp, policy, options.rng);
    const Gs = computeReturns(traj.rewards, mdp.gamma);
    const seen = new Set<number>();
    for (let t = 0; t < traj.states.length; t++) {
      const s = traj.states[t];
      if (firstVisit && seen.has(s)) continue;
      seen.add(s);
      visits[s] += 1;
      // Incremental running-mean update
      V[s] += (Gs[t] - V[s]) / visits[s];
    }
  }
  return { V, visits };
}

/** Off-policy MC policy evaluation via importance sampling. */
export function mcOffPolicyEvaluation(
  mdp: MDP,
  target: Policy,
  behavior: Policy,
  nEpisodes: number,
  options: { weighted?: boolean; rng?: () => number } = {},
): { V: Float64Array; ess: Float64Array } {
  const weighted = options.weighted ?? true;
  const num = new Float64Array(mdp.nStates);
  const den = new Float64Array(mdp.nStates);  // weighted: sum of weights
  const counts = new Int32Array(mdp.nStates);  // ordinary: count
  const sumW = new Float64Array(mdp.nStates);
  const sumW2 = new Float64Array(mdp.nStates);

  for (let ep = 0; ep < nEpisodes; ep++) {
    const traj = sampleTrajectory(mdp, behavior, options.rng);
    const Gs = computeReturns(traj.rewards, mdp.gamma);
    // Compute rho_{t:T-1} for each t in a backward sweep
    const T = traj.states.length;
    const rhoSuffix = new Float64Array(T + 1);
    rhoSuffix[T] = 1.0;
    for (let k = T - 1; k >= 0; k--) {
      const s = traj.states[k], a = traj.actions[k];
      const pt = target.probability(s, a);
      const pb = behavior.probability(s, a);
      rhoSuffix[k] = pb === 0 ? 0 : rhoSuffix[k + 1] * (pt / pb);
    }
    const seen = new Set<number>();
    for (let t = 0; t < T; t++) {
      const s = traj.states[t];
      if (seen.has(s)) continue;
      seen.add(s);
      const w = rhoSuffix[t];
      if (weighted) {
        num[s] += w * Gs[t];
        den[s] += w;
      } else {
        num[s] += w * Gs[t];
        counts[s] += 1;
      }
      sumW[s] += w;
      sumW2[s] += w * w;
    }
  }
  const V = new Float64Array(mdp.nStates);
  for (let s = 0; s < mdp.nStates; s++) {
    if (weighted) {
      V[s] = den[s] === 0 ? 0 : num[s] / den[s];
    } else {
      V[s] = counts[s] === 0 ? 0 : num[s] / counts[s];
    }
  }
  const ess = new Float64Array(mdp.nStates);
  for (let s = 0; s < mdp.nStates; s++) {
    ess[s] = sumW2[s] === 0 ? 0 : (sumW[s] * sumW[s]) / sumW2[s];
  }
  return { V, ess };
}

/** MC Control with Exploring Starts. Converges to Q*. */
export function mcControlExploringStarts(
  mdp: MDP,
  nEpisodes: number,
  options: { rng?: () => number } = {},
): { Q: Float64Array; policy: Int32Array } {
  // Returns Q as flat (state, action) array of length nStates * nActions
  // and the greedy policy as Int32Array of length nStates.
  // Implementation omitted for brevity; see source.
  // ... (~30 lines)
}

/** MC Control with ε-greedy. Converges to Q^π for the ε-soft optimal policy. */
export function mcControlEpsGreedy(
  mdp: MDP,
  nEpisodes: number,
  epsilon: number | ((episode: number) => number),  // GLIE schedule supported
  options: { rng?: () => number } = {},
): { Q: Float64Array; policy: Int32Array } {
  // Same shape. Supports both fixed ε and GLIE-style schedule.
  // ... (~35 lines)
}
```

**Vitest test targets** (from pre-verified Python):

```ts
test('First-visit MC of uniform random gridworld converges to -0.4205', () => {
  const mdp = makeGridworld();
  const policy = uniformRandomPolicy(mdp);
  const { V } = mcPolicyEvaluation(mdp, policy, 50_000, { rng: seeded(0) });
  expect(V[stateIdx(0, 0)]).toBeCloseTo(-0.4205, 1);  // within 0.05
});

test('First-visit MC convergence rate at N=100/1000/10000 ≈ 0.04/0.013/0.004', () => {
  // 50-trial replication
  const stds = [100, 1000, 10000].map(N => measureStd(N, 50));
  expect(stds[0]).toBeCloseTo(0.044, 1);
  expect(stds[1]).toBeCloseTo(0.013, 2);
  expect(stds[2]).toBeCloseTo(0.0036, 3);
});

test('Off-policy weighted MC converges to exactly 0.7290 at N=10000', () => {
  // Uniform behavior, deterministic optimal target
  // All matching trajectories are identical with weight 256, so weighted IS
  // becomes exact once any match is observed.
  for (let trial = 0; trial < 10; trial++) {
    const { V } = mcOffPolicyEvaluation(mdp, det_opt, uniform, 10_000, {
      weighted: true, rng: seeded(trial),
    });
    expect(V[stateIdx(0, 0)]).toBeCloseTo(0.7290, 4);
  }
});

test('MC Exploring Starts converges to Q* on gridworld', () => {
  const { Q, policy } = mcControlExploringStarts(mdp, 50_000, { rng: seeded(0) });
  // Q*(0,0,right) = 0.7290, policy(0,0) = RIGHT (action 1)
  expect(Q[stateIdx(0, 0) * 4 + 1]).toBeCloseTo(0.7290, 1);  // within 0.05
  expect(policy[stateIdx(0, 0)]).toBe(1);
});

test('MC ε-greedy with ε=0.1 converges to Q^π_ε-soft, NOT Q*', () => {
  const { Q } = mcControlEpsGreedy(mdp, 50_000, 0.1, { rng: seeded(0) });
  // Q_ε-soft(0,0,right) = 0.6307, NOT Q*(0,0,right) = 0.7290
  expect(Q[stateIdx(0, 0) * 4 + 1]).toBeCloseTo(0.6307, 1);
  expect(Q[stateIdx(0, 0) * 4 + 1]).not.toBeCloseTo(0.7290, 1);
});

test('MC ε-greedy with GLIE schedule converges to Q*', () => {
  const glie = (ep: number) => 1 / Math.sqrt(ep + 1);
  const { Q } = mcControlEpsGreedy(mdp, 100_000, glie, { rng: seeded(0) });
  expect(Q[stateIdx(0, 0) * 4 + 1]).toBeCloseTo(0.7290, 1);
});

test('ESS at N=10000 ≈ 39 (matches non-zero count for det-opt target)', () => {
  const { ess } = mcOffPolicyEvaluation(mdp, det_opt, uniform, 10_000, {
    rng: seeded(42),
  });
  expect(ess[stateIdx(0, 0)]).toBeGreaterThan(20);
  expect(ess[stateIdx(0, 0)]).toBeLessThan(80);
});
```

---

## 6. Component Catalog

| Code | Component                       | Section | Polish budget |
|-----|---------------------------------|---------|--------------|
| V1  | `<ModelFreeVsModelBased>`       | §1      | 0.5 day      |
| V2  | `<FirstVsEveryVisitWalkthrough>`| §2      | 1 day        |
| V3  | `<MCEstimatorLab>`              | §3      | **3-4 days** (centerpiece) |
| V4  | `<MCControlLearningCurve>`      | §4      | 1.5 days     |
| V5  | `<OffPolicyVsOnPolicy>`         | §5      | 1 day        |
| V6  | `<BiasVariancePlanePreview>`    | §6      | 0.5 day      |
| V7  | `<RoadmapMini>` (update)        | §7      | 0.5 day      |

Total polish budget around eight days, in line with the master plan's
estimate of "around eleven hundred lines, seven sections, seven
visualizations."

**Reuse from prior lessons:** `GridworldRenderer`, `MDPEditor`,
`MathBlock`, `CrosslinkCallout`, `PanelChrome`, `RoadmapMini`, all
of `src/mdp/`, `policyEvaluationExact` from `src/dp/`, all of
`src/importance-sampling/` from Lesson 6. New code is around 200
lines in `src/monte-carlo/` plus the visualizations.

---

## 7. Page-Level User Experience

Same conventions as prior lessons. Single-page scroll, prereq strip at
top, reduced-motion support is important for V2's trajectory animation
and V3's heatmap transitions. The centerpiece V3 is the only component
that breaks out to 960 pixels in width.

A specific UX note for V3: the on-policy/off-policy toggle should be
prominent and labeled. Many learners will not initially understand
that flipping the toggle changes the entire pedagogical content of the
lab. The default is on-policy (uniform random evaluating itself, true
value -0.4205, smooth convergence). Flipping to off-policy with the
defaults (uniform behavior, deterministic optimal target) triggers
the variance-blow-up demonstration. A small inline tooltip near the
toggle should hint that "off-policy mode reveals what off-policy
estimation costs."

---

## 8. Acceptance Criteria

After completing this lesson, a learner should be able to do the
following.

State the model-free setting and explain why dynamic programming does
not apply. State the basic MC return estimate and explain why it is
unbiased. Distinguish first-visit and every-visit MC and explain why
both converge. Write down the incremental running-mean update for MC
policy evaluation. State the MC Exploring Starts algorithm and the
condition under which it converges to `Q^*`. Explain why MC ε-greedy
with fixed ε converges to the ε-soft optimal Q-function, not to `Q^*`.
State the GLIE condition and explain how it recovers `Q^*`. Apply the
trajectory importance sampling apparatus to write down both ordinary
and weighted off-policy MC estimators. Explain the variance blow-up
of off-policy MC and identify the effective-sample-size diagnostic.
Articulate the bias-variance trade-off between MC and TD without yet
having seen TD in detail.

A concrete acceptance check: hand the learner a 4×4 gridworld with a
new pit placement and a uniform-random policy. Ask them to predict
qualitatively what `\hat V^\pi(s)` will look like after 100, 1000,
and 10000 MC episodes, and where the per-state convergence will be
fastest and slowest. V3 lets them check their predictions on the
running 3×3 example.

---

## 9. Stretch Goals (post-MVP)

**Weighted vs ordinary IS bias decomposition.** Section 5 could be
extended with an explicit decomposition of weighted IS bias into
denominator-variance and numerator-denominator covariance terms. The
finite-sample bias of weighted IS is `O(1/N)` with specific constants;
quoting them would tie the lesson more tightly to the statistics
literature.

**MC control on a larger MDP.** The 3×3 gridworld is small enough that
MC ES converges in 50,000 episodes. A 5×5 variant with a longer
horizon would expose convergence behavior at scales closer to
real-world problems and would set up Lesson 9's deep-RL transition
more naturally.

**Per-decision off-policy MC.** Lesson 6 introduced per-decision IS;
its MC version (per-decision off-policy MC) is a small variation on
the trajectory off-policy estimator and is the basis for V-trace.
Could appear as a stretch section or a fourth estimator in V3.

**Eligibility-trace-like generalizations.** The "n-step return"
preview in Section 6 could be expanded into a full mini-section on
the geometric weighting of n-step returns and the λ-return as a
weighted sum. This would establish the bridge to TD(λ) and eligibility
traces in Lesson 8 more carefully.

---

## 10. Out of Scope (intentionally)

**Off-policy MC control.** Off-policy versions of MC control (using
weighted IS to estimate Q for a target policy that differs from
behavior) are mentioned in Section 5 but not developed. The variance
makes them impractical for finite-data control, and TD-based off-policy
control (Q-learning, Lesson 8) is strictly better. Coverage of these
methods would add length without pedagogical payoff at this point.

**Bias-corrected MC estimators.** Various jackknife and bootstrap
corrections to MC estimators exist but are rarely used in RL. Out of
scope.

**Continuous-time MC.** All discussion is discrete-time episodic. Out
of scope.

**Variance of every-visit MC formally.** Section 2 notes empirically
that every-visit MC's bias is `O(1/N)` and its variance is essentially
the same as first-visit. A formal characterization (Singh and Sutton
1996) exists but is more technical than the lesson needs.

---

## 11. Training Notebook

Not applicable. No models are trained for this lesson. The script
`scripts/mc_traces.py` (around eighty lines) pre-computes the
convergence statistics tables, the off-policy gridworld numbers, and
the MC control learning curves, all into JSON files under
`public/data/mc/` for instant initial render. The in-browser
TypeScript implementation reproduces these statistics live when the
user interacts.

---

## 12. Closing Notes and Length Tally

Total length: roughly eleven hundred lines. Slightly longer than
Lesson 6 (which was 900 lines), substantially shorter than Lesson 3
(MDPs, 1500 lines). The expanded length compared to Lesson 6 is
justified by three things: there are two distinct flavors of MC
control (Exploring Starts and ε-greedy) requiring careful coverage,
the GLIE/ε-soft distinction is non-trivial and merits explicit
empirical demonstration, and the lesson is the curriculum's first
algorithmic departure from the model-based DP framework which
deserves a clear opening section.

The centerpiece V3 (`MCEstimatorLab`) is the lesson's polish-budget
sink: a six-panel synchronized lab with both on-policy and off-policy
modes, designed to expose the variance contrast between them as a
single interaction. The off-policy mode reuses Lesson 6's gridworld
configuration (uniform behavior, deterministic optimal target,
matching rate 1/256, weight 256) so that the numerical anchors
recognized from Lesson 6 reappear in the new framing.

The forward links are saturated: Lessons 8 (the immediate cash-in),
9, 10, and 15 all cited explicitly in Section 7 with specific
applications named. Lesson 8 in particular will spend its opening
section as a near-direct continuation of this lesson's Section 6,
making Monte Carlo and TD a tightly coupled pair.

## End of specification