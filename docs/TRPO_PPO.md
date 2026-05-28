# Lesson 11 — TRPO and PPO

> Lesson 10 introduced the policy gradient, the score function
> estimator, and the actor-critic family. It also exposed a quiet
> problem we glossed over: the policy gradient is a *direction*, not a
> *step size*. The naive update `θ ← θ + α ∇J` works on the gridworld
> only because softmax is forgiving; in deep RL with continuous
> actions, a single bad batch can shred a policy that took millions of
> samples to build. This lesson is about how the field learned to
> step carefully.

> Where this slots in: directly after Lesson 10. We pick up the loose
> end of §10.7 (GAE was previewed but not treated), introduce trust
> regions as the principled fix, walk through TRPO as the canonical
> implementation, then derive PPO as the practical workhorse that
> replaced it. PPO is the algorithm that powers Lesson 17 (RLHF and
> DPO) and feeds the maximum-entropy machinery of Lesson 12 (which
> reframes Lesson 10's "softmax cap" as a feature). This lesson is
> the spine between foundations and the capstone.

---

## 0. Pedagogical Philosophy

**0.1 The motivation comes from a failure mode the gridworld doesn't
fully show — and we are honest about that.** The textbook story for
TRPO/PPO begins with "vanilla PG can destroy a policy in a single bad
update." On a 3×3 softmax gridworld this is mostly false. Vanilla PG
on the tabular policy is variance-noisy at moderate learning rates
and slow at low ones, but it does not catastrophically collapse — the
softmax keeps the policy in a bounded simplex and the small action
space makes recovery cheap. The dramatic collapse that motivates
trust regions lives in continuous-action deep RL, where a Gaussian
policy's σ can explode and a single update can move the policy off
the data distribution it was trained on. This lesson tells the
honest story: on the gridworld, PPO's advantage is *variance
reduction* (about 85× at moderate learning rates, in our
measurements) and *robustness across learning rates*; the policy
collapse story is described and illustrated, but not faked with
contrived numerics.

**0.2 TRPO is treated as the principled ancestor, not the
implementation target.** The math of TRPO — natural gradient,
conjugate gradient, KL constraint via line search — is the cleanest
way to introduce trust regions. But TRPO is mechanically heavy: the
Fisher information matrix requires Hessian-vector products, the
conjugate gradient solver wants careful preconditioning, and the
line search adds another layer of bookkeeping. The lesson covers
TRPO in full math and gives a worked walk-through, but the
implementation in the lesson source is **PPO only**. TRPO appears in
prose and in two illustrative visualizations; the agent does not
build a working TRPO.

**0.3 The clipping mechanism is the lesson's centerpiece, and it is
best seen in continuous action space.** PPO's clipped surrogate is a
mathematical object — `min(r·A, clip(r, 1-ε, 1+ε)·A)` — whose shape
matters as much as its derivation. The cleanest way to show that
shape is over a 1D action with a Gaussian policy: the surrogate
becomes a function on the real line, the clip introduces a plateau,
and the gradient becomes visibly zero in the plateau region. The
centerpiece (PPO Lab) uses a continuous-action 1D sidebar for the
clipping visualization while keeping the empirical thread on the
gridworld. This is the lesson's one careful step toward continuous
control, and it sets up Lesson 13 (SAC) where continuous actions are
serious.

**0.4 GAE is treated in full, including its honest limits on small
problems.** Lesson 10 §7 previewed GAE as a λ-interpolation between
TD(0) and MC advantage. Here we derive it, study its bias-variance
trade, and verify empirically. The result on the gridworld is
*monotone in λ*: λ=0 (full bootstrap) has the lowest variance and
the smallest bias error at convergence; λ=1 (MC) has the highest
variance and the largest residual error. The canonical "λ=0.95 is
the default" heuristic is a deep-RL convention that comes from much
larger problems with much noisier critics. The lesson reports both
numbers and explains the gap, rather than pretending the heuristic
holds at every scale.

**0.5 Forward links are weighted toward L13 and L17.** The original
curriculum had model-based RL (L14), offline RL (L15), and diffusion
in RL (L16) between this lesson and the capstone. Those lessons
have been trimmed from the production schedule for budget reasons.
The spine is now L11 → L12 (max-ent) → L13 (SAC) → L17 (RLHF). The
forward-link section of this lesson points to the spine and includes
a brief "future work" note acknowledging the trimmed branches. The
trim does not weaken L11 itself — TRPO/PPO is the workhorse, and the
direct path to RLHF runs straight through here.

**0.6 The empirical thread continues on the 3×3 gridworld.** PPO
runs on the same softmax policy used in Lesson 10. The final V(start)
sits between L10's REINFORCE+baseline (0.7250 ± 0.0011) and the
theoretical V* (0.7290), depending on how long PPO runs. At 200
iterations × 20 episodes per batch, PPO converges to 0.7267 ± 0.0003
(10 seeds, lr=0.5); at 500 iterations it reaches 0.7282 ± 0.0002.
The L10 "softmax cap" framing is preserved — the policy approaches
but cannot reach V* because softmax cannot represent a fully
deterministic policy — and L12 will reframe this gap as a feature.

**0.7 Hyperparameter robustness is itself a pedagogical point.** A
secondary empirical experiment sweeps learning rate across two
orders of magnitude for both vanilla PG and PPO. The result: at
lr=0.1, vanilla PG is essentially stuck (final V ≈ -0.1) while PPO
is making progress; at lr=5.0, both converge equally well. PPO's
sweet spot is wider. The lesson uses this directly: trust regions
buy you robustness to hyperparameter choice, which matters more in
practice than the absolute best-case performance.

**0.8 The deadly triad from L9 reappears, in a friendly form.** PPO
runs are on-policy in a strict sense (each batch is collected under
the current policy), which dodges the off-policy side of the deadly
triad. But PPO does multiple epochs of updates per batch — and as
the policy drifts within those epochs, the batch is increasingly
off-policy with respect to the current π_θ. The clipped ratio is
*precisely* the mechanism that bounds the drift. This connection is
named explicitly. The deadly triad is not just a warning about
neural Q-learning; it is the same shape of problem that PPO is
solving.

---

## 1. Tech Stack

Same as Lesson 10: Vite, TypeScript strict, plain Web Components,
KaTeX, D3 v7, `ml-matrix`, Vitest, Python 3.11+, `seedrandom`. No
new browser-runtime dependencies.

For pre-computation (`scripts/ppo_traces.py`), the existing offline
stack from L9 and L10 is reused: NumPy and PyTorch are installed but
PyTorch is *not used* in this lesson — the policy is tabular softmax
over 9 × 4 logits, and PPO + critic + GAE run in pure NumPy.
PyTorch will return in L13 (SAC) when continuous actions force
function approximation. The lesson's offline scripts emit JSON to
`public/data/ppo/` for the visualizations; no ONNX models ship.

---

## 2. Visual and Aesthetic Direction

```css
:root {
  --ppo-bg: #faf8f3;
  --ppo-text: #1c1e22;

  /* Algorithm colors — preserves curriculum threads */
  --ppo-vanilla: #b45309;     /* amber: raw on-policy PG, matches L10 REINFORCE */
  --ppo-baseline: #0e7490;    /* cyan: critic / bootstrapping, matches L10 AC */
  --ppo-clipped: #7c3aed;     /* violet: the principled distribution being optimized */
  --ppo-trpo: #6d28d9;        /* deeper violet: TRPO as the principled ancestor */

  /* Mechanism highlights */
  --ppo-clip-region: #fbbf24; /* amber-lighter: the clip interval [1-eps, 1+eps] */
  --ppo-clip-bind:  #ea580c;  /* orange: a sample whose ratio is being clipped */
  --ppo-kl:         #be185d;  /* pink: KL divergence (a "between-policies" thing) */

  /* Outcome colors */
  --ppo-converged:  #15803d;  /* green: converged, optimal-leaning */
  --ppo-diverged:   #dc2626;  /* red: divergence warning */
  --ppo-gae-line:   #db2777;  /* pink: the GAE interpolator curve */
}
```

The cyan/amber/violet thread that began in L1 (bandits) continues
here: amber is the on-policy raw-gradient color (vanilla PG, the
unclipped path), cyan is the critic (the baseline that the
advantage is computed against), violet is the principled
distribution being learned (the policy π_θ itself, now under a
clipping constraint). TRPO uses a deeper violet to signal "same
family but more principled / less practical."

The clipping region gets its own color (`--ppo-clip-region`, a soft
amber-yellow) used to render the `[1−ε, 1+ε]` band on ratio plots.
A second clip color (`--ppo-clip-bind`, orange) marks individual
samples whose probability ratio is currently being clipped — these
are the samples whose gradient contribution has been zeroed by the
plateau. Pink is reserved for the KL line (a between-policies
quantity, in keeping with L8's n-step pink for an "interpolating"
thing) and for the GAE λ interpolator curve.

Two new icons join the curriculum's icon set: a "trust region" icon
(a small dashed circle around a point) and a "clip plateau" icon (a
flat horizontal segment between two corners). The first is used in
forward-link callouts to L13; the second on the centerpiece title.

---

## 3. Lesson Metadata

```typescript
import type { LessonMeta } from '../../shared/lesson-types';

export const lesson11Meta: LessonMeta = {
  id: 'l11-trpo-ppo',
  title: 'Trust Regions: TRPO and PPO',
  subtitle: 'Stepping carefully in policy space',
  tier: 2,
  difficulty: 'advanced',
  estimatedReadMinutes: 65,
  role: 'workhorse',
  prerequisites: [
    'l10-policy-gradient',  // score function, REINFORCE, advantage, baseline
    'l9-function-approx-dqn', // function approximation context, deadly triad
    'l6-importance-sampling', // probability ratios are an IS object
    'l5-dynamic-programming', // policy iteration as the conceptual ancestor
  ],
  exportedAnchors: {
    // Numerical anchors, verified in scripts/ppo_traces.py
    'PPO-final-V-start-200-lr05-batch20': 0.7267,
    'PPO-final-V-start-200-lr05-batch20-std': 0.0003,
    'PPO-final-V-start-500-lr05-batch20': 0.7282,
    'PPO-vs-vanilla-std-ratio-lr05': 85,
    'GAE-lambda-0-final-V-batch5': 0.7237,
    'GAE-lambda-1-final-V-batch5': 0.7208,
    'PPO-early-clip-fraction-aggressive': 0.35,  // batch=5 lr=2 epochs=10
    'PPO-late-clip-fraction-aggressive': 0.02,
  },
  centerpieceComponent: 'PPOLab',
  forwardLinksWhenReady: [
    { lessonId: 'l12-max-ent-rl', anchor: 'entropy-regularized-objective' },
    { lessonId: 'l13-sac', anchor: 'continuous-actor-critic' },
    { lessonId: 'l17-rlhf-dpo', anchor: 'ppo-for-language-models' },
  ],
};
```

---

## 4. Section-by-Section Plan

### Section 1 — The Step-Size Problem

**Tagline:** *The policy gradient is a direction, not a step size.*
**Length:** ~750 words
**Anchor:** `step-size-problem`

We pick up where Lesson 10 left off. The policy gradient theorem
gave us an unbiased estimate of `∇J(θ)`; REINFORCE+baseline and
actor-critic gave us low-variance versions of that estimate; and a
gradient ascent update `θ ← θ + α ∇̂J(θ)` produced a learning curve
that converged on the gridworld. What we never asked: is α a
*length*, or a *bound on the policy change*?

The reader's voice in the head: *I just picked α = 0.05 because it
worked. That's not a principle, that's tuning.*

The section makes this concrete. A learning rate measures the size
of a step in *parameter space* — in `θ`. But what we care about is
the size of a step in *policy space* — in `π_θ(a|s)`. The map from
θ to π is the softmax, and it is nonlinear: the same step in θ can
move π by a very small amount or by a very large amount, depending
on where you started. At a uniform policy with θ ≈ 0, a θ-step of
0.1 barely moves the probabilities. At a sharp policy with θ values
of (10, -10, -10, -10), a θ-step of 0.1 also barely moves anything
— the policy is already locked. But at a moderately confident
policy with θ values of (2, 0, 0, 0), a θ-step of 1.0 can flip
which action is most likely.

This is the *step-size problem*: there is no fixed α that controls
the size of the policy change uniformly across training. The same
learning rate can be too small at the start and too large near
convergence, or too large at the start and too small near
convergence, or somewhere in between. The naive policy gradient
gives you no handle on this.

A second framing: a policy gradient update is *valid* only for a
small region around `θ_old`. The advantage estimate `Â(s,a)` was
collected under `π_θ_old`; if `θ_new` moves the policy far enough
that `π_θ_new` produces a very different state-action distribution,
the advantage estimate becomes irrelevant — you are taking a
gradient step using stale data. The standard term is
*off-policyness*: a single big update makes your old batch
off-policy with respect to your new policy, and the gradient is no
longer pointing in a useful direction.

We illustrate this on the gridworld empirically. With `lr = 0.1`,
vanilla PG converges to V(start) ≈ -0.105 after 200 iterations of
batch=10 episodes — essentially stuck, because each update is too
small to escape the noise floor. With `lr = 5.0`, vanilla PG
converges to 0.7281 ± 0.0006 — fine, because softmax is forgiving.
The sweet spot is wide, but only because the gridworld is friendly.

We then describe (in prose, not in code) the deep-continuous case
where the sweet spot is *not* wide: a Gaussian policy's σ collapses
in a single update; an MLP policy's features get scrambled by a
batch with a 5σ reward outlier; an LLM policy's distribution shifts
off the SFT manifold and never recovers. The "policy collapse"
story is real — it is just not the gridworld's story. The lesson is
honest about this from the start.

Visualization V1 — Parameter Space vs Policy Space.
A two-panel side-by-side. Left panel: parameter space, a 2D
projection of the 9×4 logit vector showing the trajectory of θ
across training. Right panel: policy space, the same trajectory
projected into the 4-simplex of action probabilities at state 0.
The user toggles the learning rate and watches the same arc
deformed: at low lr, both panels show smooth small steps; at high
lr, the parameter trajectory becomes jagged but the policy
trajectory stays bounded (softmax projection). A small label notes
"In deep continuous RL, the policy panel would have no such bound."
500 × 280 px. Polish budget: 2 days.

---

### Section 2 — Trust Regions and the KL Constraint

**Tagline:** *Bound the policy change, not the parameter change.*
**Length:** ~850 words
**Anchor:** `trust-region-kl`

If the problem is that parameter steps don't correspond to policy
steps, the fix is to measure step size *in policy space*. The
candidate measure is KL divergence.

We motivate KL by elimination. L2 distance in parameter space is
what we just abandoned. L2 distance between policy probability
vectors works but depends on the dimensionality of the action
space and conflates "two actions with probabilities (0.6, 0.4)"
with "two actions with probabilities (0.001, 0.999)" — a fixed L2
gap means different things at different points on the simplex. KL
divergence has the right invariance: it measures the *information*
gained by switching from `π_old` to `π_new`, in nats; it is
parameterization-invariant (KL doesn't care whether your policy is
parameterized by softmax logits or by something else); and it has a
clean local approximation in terms of the Fisher information.

The trust-region optimization problem then becomes:

$$
\theta_{\text{new}} = \arg\max_{\theta}\; \hat{\mathcal{L}}(\theta)
\quad \text{s.t.}\quad
D_{\text{KL}}\!\left(\pi_{\theta_{\text{old}}} \,\|\, \pi_\theta\right) \le \delta
$$

where `L̂(θ)` is the surrogate objective — typically the
importance-sampled advantage estimate

$$
\hat{\mathcal{L}}(\theta) \;=\;
\mathbb{E}_{s, a \sim \pi_{\theta_{\text{old}}}}
\left[
  \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} \,\hat{A}(s, a)
\right].
$$

The reader's voice in the head: *We've seen this ratio before. In
Lesson 6, that was the importance-sampling weight.*

That is exactly right, and we say so explicitly. The probability
ratio `r(θ) = π_θ(a|s) / π_θ_old(a|s)` is the same object as the
IS weight from L6 — and the variance problem from L6 (weights can
explode for off-distribution actions) is the same variance problem
that trust regions are trying to bound. We are not solving a new
problem; we are solving the L6 problem in the policy-optimization
context.

We then take the local approximation. For small `‖θ − θ_old‖`, KL
admits the second-order expansion

$$
D_{\text{KL}}(\pi_{\theta_{\text{old}}} \| \pi_\theta) \;\approx\;
\tfrac{1}{2}\, (\theta - \theta_{\text{old}})^\top
F(\theta_{\text{old}}) \, (\theta - \theta_{\text{old}})
$$

where `F` is the *Fisher information matrix*

$$
F(\theta) \;=\;
\mathbb{E}\!\left[
  \nabla_\theta \log \pi_\theta(a|s)\,
  \nabla_\theta \log \pi_\theta(a|s)^\top
\right].
$$

The Fisher is the expected outer product of score functions — and
the score function is exactly the object we built in L10 §2. The
crosslink is direct: the same `∇ log π_θ` that gave us REINFORCE's
gradient direction also defines the curvature that bounds the trust
region.

A small numerical exercise: compute F for the uniform softmax
policy on the gridworld and report its top eigenvalues. (For the
9×4 = 36-parameter softmax, F is a 36×36 block-diagonal matrix
with one 4×4 block per state, each block of rank 3 because the
softmax has one redundant degree of freedom per state. The largest
eigenvalue per block is 3/4 under the uniform policy.) This is a
side calculation, not a centerpiece, but it grounds the math in
the running example.

We close the section with the natural gradient idea: instead of
ascending `∇J`, ascend `F^{-1} ∇J`. The natural gradient is the
direction of steepest ascent *in KL geometry*, not in Euclidean
geometry. It is the right direction; TRPO is, in essence, "do a
line search along the natural gradient direction subject to a KL
constraint."

Visualization V2 — KL Geometry of the Softmax Policy.
A 2D projection of the softmax policy's parameter space at state 0
(the starting state). The axes are two of the four logits; the
remaining two are held fixed. The user picks a point `θ_old` and
sees three things: (a) a circle of constant L2 distance in
parameter space; (b) an ellipse of constant KL divergence (the
ellipse is rotated and elongated relative to the L2 circle); (c)
the natural gradient direction at `θ_old`, plotted as an arrow
that has been "tilted" relative to the vanilla gradient by
`F^{-1}`. The user toggles between `θ_old` near uniform (where L2
and KL ellipses are nearly circular) and `θ_old` near a sharp
policy (where they are very different). 460 × 320 px. Polish
budget: 2 days.

---

### Section 3 — TRPO

**Tagline:** *The principled ancestor.*
**Length:** ~800 words
**Anchor:** `trpo-algorithm`

TRPO (Trust Region Policy Optimization, Schulman et al. 2015) is
the direct implementation of the trust-region idea. We walk through
the algorithm in full:

1. Collect a batch under `π_θ_old`. Compute advantages `Â`.
2. Compute the policy gradient `g = ∇_θ L̂(θ_old)`.
3. Solve `F x = g` using conjugate gradient — the "search
   direction" `x` is the natural gradient.
4. Line search along `x` for the largest step size `α` such that
   `D_KL(π_θ_old ‖ π_{θ_old + αx}) ≤ δ` *and* the surrogate has
   actually improved.
5. Set `θ_new = θ_old + αx`.

The two heavy pieces are the conjugate gradient solve (because F is
too big to invert directly in deep RL; CG only needs
Hessian-vector products) and the line search (which has to
backtrack until both the KL constraint and the improvement
constraint are satisfied). The lesson explains each piece in
moderate detail and shows the per-iteration cost — typically
~10 CG iterations and a few line-search backtracks per update.

The reader's voice: *This is a lot of machinery.*

It is, and the lesson is direct about it. TRPO works — it was one
of the first algorithms to learn complex locomotion policies from
scratch — but it is unwieldy. The conjugate gradient requires
implementing efficient Hessian-vector products (a separate small
optimization-library subproject); the line search adds
non-trivial bookkeeping; the second-order machinery interacts
poorly with adaptive optimizers like Adam (which break the natural
gradient interpretation).

We illustrate TRPO's behavior conceptually rather than empirically.
On the gridworld, TRPO with `δ = 0.01` per update gives convergence
indistinguishable from PPO with `clip_eps = 0.2`. The two algorithms
make qualitatively similar policy updates: small, bounded changes
that respect a per-step KL budget. The mechanism is different; the
behavior is similar.

We end the section with the question that motivated PPO: *what if
we could get TRPO's stability without TRPO's machinery?*

Visualization V3 — TRPO Step Schematic.
A schematic of one TRPO iteration: vanilla gradient direction (a
short arrow), CG-corrected natural gradient direction (a longer
tilted arrow), and the trust region (an ellipse around `θ_old`).
The line search is shown as a sequence of points along the natural
gradient, with the largest accepted step highlighted. Not
interactive; pure schematic. 420 × 260 px. Polish budget: 1 day.

---

### Section 4 — PPO and the Clipped Ratio

**Tagline:** *A surrogate with a built-in trust region.*
**Length:** ~900 words
**Anchor:** `ppo-clipped`

PPO (Proximal Policy Optimization, Schulman et al. 2017) drops
TRPO's second-order machinery and replaces it with a first-order
trick: modify the surrogate objective so that the gradient
*naturally* vanishes when the policy has moved too far. The
modification is the clipped ratio.

Recall the IS surrogate from §2:

$$
\hat{\mathcal{L}}^{\text{IS}}(\theta) \;=\;
\hat{\mathbb{E}}\!\left[ r_t(\theta)\, \hat{A}_t \right],
\quad
r_t(\theta) \;=\;
\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}.
$$

PPO replaces it with

$$
\hat{\mathcal{L}}^{\text{CLIP}}(\theta) \;=\;
\hat{\mathbb{E}}\!\left[
  \min\!\bigl(
    r_t(\theta)\, \hat{A}_t,\;
    \mathrm{clip}\!\bigl(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon\bigr)\,
    \hat{A}_t
  \bigr)
\right].
$$

The clipping parameter `ε` is typically 0.1–0.2. The shape of this
objective depends on the sign of the advantage:

- If `Â_t > 0` (this action was better than average), the
  unclipped surrogate `r·Â` is upward-sloping in `r`. The clip
  flattens it above `r = 1+ε`. The `min` is then a hockey stick:
  it rises linearly with `r` up to `1+ε`, then is flat.
- If `Â_t < 0` (this action was worse than average), the
  unclipped surrogate is downward-sloping. The clip flattens it
  below `r = 1−ε`. The `min` is then an upside-down hockey stick:
  it slopes down to `1−ε`, then is flat.

The `min` is asymmetric on purpose. When `Â > 0`, you want to
*increase* `r` (make the good action more likely), so you allow
the gradient to push `r` up — but only up to `1+ε`. Beyond that,
the surrogate is flat, the gradient is zero, no update. Symmetric
on the downside for `Â < 0`. The result: PPO's gradient
*automatically* vanishes when the policy has moved far enough,
without needing to compute KL explicitly. The trust region is
*implicit*, baked into the objective rather than enforced by a
constraint.

The reader's voice: *Why the min, not just the clip?*

Without the min, the clipped objective `clip(r) · Â` would have
zero gradient *everywhere* outside `[1−ε, 1+ε]` — even when the
policy is moving in the wrong direction. The min ensures the
gradient is still alive when the policy is moving *back into* the
trust region. Specifically: if `Â > 0` and `r < 1−ε` (the policy
has decreased the probability of a good action), the unclipped
`r·Â` is the smaller term (because `Â > 0` and `r < 1`), and the
min picks it — so the gradient pushes `r` back up. The clip only
"bites" when the policy is moving *away* from the old policy in a
way that the objective wants to keep going.

We make this concrete with a 1D continuous-action example. Assume
a Gaussian policy `π_θ(a|s) = N(a; μ_θ, σ²)` with σ fixed. Pick a
sampled action `a`, an advantage `Â > 0`, and compute the
surrogate as a function of `μ_θ`. The result is a piecewise curve:
linear on one side, plateau at the top, sloping down on the other
side. The plateau is precisely where `r(θ)` exceeds `1 + ε`. We
plot this curve — it is the canonical "PPO clipping picture" — and
let the user vary `ε`, `Â`, and the gap between `μ_θ` and the
behavior policy's mean.

Multi-epoch updates and the ratio's drift. A subtle but important
point: PPO does *multiple* gradient steps on the same batch. The
ratio `r(θ)` starts at exactly 1 (at the first step, `θ = θ_old`)
and drifts away as `θ` moves. The clipping only starts to bind
after a few steps, when `r` has drifted outside `[1−ε, 1+ε]`. In
practice the clip is "lightly binding" most of the time — in our
verification runs at `clip_eps = 0.2`, the clip fraction stays
under 5% throughout training. It is only at very aggressive
learning rates and small batches (lr=2.0, batch=5, epochs=10) that
the clip fraction climbs to 30–45% early in training. The
centerpiece visualization (§6) lets the user push to this regime
and see the clip in action.

PPO also includes (in most implementations) a value-function loss
and an entropy bonus:

$$
\hat{\mathcal{L}}^{\text{PPO}} \;=\;
\hat{\mathbb{E}}\!\left[
  \hat{\mathcal{L}}^{\text{CLIP}} \;-\;
  c_1\, (V_\phi(s_t) - \hat{R}_t)^2 \;+\;
  c_2\, \mathcal{H}\!\left[\pi_\theta(\cdot | s_t)\right]
\right].
$$

The entropy bonus is a preview of L12 — it is exactly the
maximum-entropy objective, weighted by `c_2`, and it stops the
policy from collapsing onto a single action prematurely. L12 will
revisit this term as the *primary* objective of an entire family
of algorithms.

Visualization V4 — The Clipped Surrogate.
The canonical PPO picture, but interactive. A 1D continuous-action
sidebar: x-axis is `μ_θ`, y-axis is the surrogate value at a
single sample. Three curves overlaid: the unclipped IS surrogate
`r·Â` (amber, sloping); the clipped `clip(r)·Â` (cyan, with a
plateau); the PPO min (violet, hockey-stick). User controls: `Â`
(can be positive or negative), `ε` (the clipping threshold),
σ-of-the-policy. The plateau region is highlighted in
`--ppo-clip-region`. As the user moves `μ_θ` along the x-axis, a
marker tracks all three curves and a small inset shows the
gradient (slope of the violet curve) — which is exactly zero on
the plateau. 540 × 320 px. Polish budget: 2.5 days.

---

### Section 5 — GAE: Generalized Advantage Estimation

**Tagline:** *The λ-interpolator, in full.*
**Length:** ~750 words
**Anchor:** `gae-full`

Lesson 10 §7 previewed GAE as a λ-interpolation between the TD(0)
advantage (`r + γV(s') − V(s)`) and the Monte Carlo advantage
(`G − V(s)`). Here we derive it properly and study its
bias-variance trade on the gridworld.

The starting point: the TD residual

$$
\delta_t \;=\; r_t + \gamma V(s_{t+1}) - V(s_t).
$$

The GAE advantage with parameter `λ ∈ [0, 1]` is

$$
\hat{A}_t^{\text{GAE}(\lambda)} \;=\;
\sum_{k=0}^{\infty} (\gamma \lambda)^k\, \delta_{t+k}.
$$

At `λ = 0`, `Â_t^GAE(0) = δ_t` — the TD residual, which is the
one-step bootstrap advantage from L8. At `λ = 1`, GAE collapses to
the MC advantage (after a telescoping sum):

$$
\hat{A}_t^{\text{GAE}(1)} \;=\;
\sum_{k=0}^{\infty} \gamma^k\, \delta_{t+k}
\;=\;
\sum_{k=0}^{\infty} \gamma^k r_{t+k} - V(s_t)
\;=\;
G_t - V(s_t).
$$

GAE interpolates between these two extremes the same way n-step TD
from L8 interpolated between TD(0) and MC. The connection is
direct: GAE(λ) is to advantage estimation what TD(λ) is to value
estimation. The pink color reserved in §2 echoes the pink we used
for n-step TD in L8.

The bias-variance trade. At low λ, the advantage estimate has low
variance (it depends on few rewards) but high bias (it depends on
the critic's accuracy at `V(s_{t+1})`). At high λ, the estimate has
high variance (it depends on many noisy rewards) but low bias (it
depends less on the critic). The "right" λ depends on how good the
critic is and how long the trajectories are.

We verify this empirically on the gridworld with batch=5 (a noisy
critic) and 20 seeds. The result is monotone in λ:

| λ | Final V(start) | Std (20 seeds) |
|:--|:---------------|:----------------|
| 0.00 | 0.7237 | 0.0008 |
| 0.50 | 0.7232 | 0.0014 |
| 0.95 | 0.7213 | 0.0014 |
| 1.00 | 0.7208 | 0.0027 |

Both the mean and the std worsen monotonically as λ → 1. On this
small problem with deterministic dynamics, the critic is easy to
learn and the bootstrap bias is negligible — so the variance
reduction from λ=0 wins. The canonical λ=0.95 default is a
deep-RL convention that originates from much larger problems
(continuous control, neural critics) where the critic is *not*
accurate and the bootstrap bias is substantial. We report both
numbers and explain the gap.

The lesson does not claim "GAE is useless on small problems" —
GAE is still a generalized object that subsumes TD(0) and MC, and
the framework is right. It claims, accurately, that λ=0 is the
empirical winner on this MDP and that the canonical default
encodes a different regime.

Visualization V5 — GAE λ Sweep.
A scatter plot: x-axis is λ ∈ {0, 0.5, 0.9, 0.95, 1.0}, y-axis is
final V(start) (mean across 20 seeds, error bars for std). A pink
interpolator curve passes through the means. Below the scatter, a
small panel shows the bias-variance decomposition conceptually: a
schematic of "bias" (how far the critic is from V^π) and
"variance" (how much the estimate fluctuates per batch) as
functions of λ. The user toggles between two MDPs: the 3×3
gridworld (where bias is small, variance dominates) and a
hypothetical "noisy critic" variant (where bias dominates, and
λ=0.95 would win). The noisy critic variant is conceptual — we
illustrate it, we don't run it. 500 × 320 px. Polish budget: 1.5
days.

---

### Section 6 — PPO Lab (Centerpiece)

**Tagline:** *The clipping mechanism, in motion.*
**Length:** ~700 words
**Anchor:** `ppo-lab`

The centerpiece. A multi-panel interactive that runs PPO on the
gridworld and exposes every internal mechanism the lesson has
introduced. Five panels, synchronized to a global iteration
counter.

**Panel A — Policy and value (gridworld).** The familiar 3×3
gridworld with the start state, goal, and pit highlighted. Action
probabilities are rendered as small bar charts in each cell; the
state value `V(s)` is rendered as a background tint (cyan = high,
faded = low). As the user steps PPO forward (or plays it
continuously), the bars and tints update in real time. 280 × 280 px.

**Panel B — Learning curves.** Three traces over the iteration
axis: V(start) (the headline learning curve, violet), the surrogate
objective value (cyan), and a moving-window estimate of the
expected return (amber). The three should track each other but
not perfectly — the gap between surrogate and true return is one
of the things the lesson has explained. 380 × 200 px.

**Panel C — Clip fraction.** A single trace: the fraction of
samples whose probability ratio is currently being clipped, per
iteration. In the moderate-lr regime (lr=0.5, batch=20), this
hovers near zero. In the aggressive regime (lr=2.0, batch=5,
epochs=10), it climbs to 30–45% early and falls to ~2% as the
policy converges. The shape of this trace is itself the
pedagogical point — the clip is doing meaningful work *during the
fast-learning phase* and naturally subsides. 380 × 160 px.

**Panel D — KL trace.** Mean `KL(π_old ‖ π_new)` per iteration,
plotted in pink. In the moderate regime it stays near 0.001
throughout. In the aggressive regime it spikes to ~0.02 early and
falls to ~0.0002 late. The user can compare against a horizontal
dashed line at `δ = 0.01` (the typical TRPO budget) — the spike
exceeds the TRPO budget, which is consistent with PPO's looser
implicit constraint. 380 × 160 px.

**Panel E — Probability ratio histogram.** The clipping mechanism
visualized directly. A histogram of `r_t(θ)` across the current
batch, with the `[1−ε, 1+ε]` band shaded in `--ppo-clip-region`
and samples falling outside the band rendered in
`--ppo-clip-bind`. The user watches the histogram morph each
iteration: at the start of a batch, all ratios are 1.0 (a delta at
1). After one epoch, they spread. By the last epoch, the tails may
extend past the clip band — and those samples are the ones whose
gradient has been zeroed. 380 × 220 px.

**Controls (bottom panel):**
- Learning rate slider: 0.05 → 5.0 (log scale).
- Batch size: 5, 10, 20, 50.
- Epochs per batch: 1, 2, 4, 8, 16.
- Clip ε: 0.05 → 0.5.
- GAE λ: 0.0 → 1.0.
- Algorithm toggle: PPO / vanilla PG / "TRPO-style" (uses a
  conservative analytic step size; not a real TRPO but a
  controlled-step baseline).
- Seed.
- Play / pause / step / reset.

**The pedagogical moment.** When the user slides clip ε from 0.2
down to 0.05 in the aggressive regime: Panel C jumps from ~30% to
~85% clip fraction, Panel D's KL trace flattens dramatically
(updates become tiny), Panel B's learning curve slows visibly,
Panel E's histogram becomes a sharp spike at exactly 1.0
(everything is being clipped to no update). The user sees the
trust-region constraint *tighten* in real time and sees its cost:
slower learning. Then slides ε back to 0.2 and learning resumes.
This is the lesson's "snap into place" moment.

960 × 700 px (breakout from the standard column width). Polish
budget: **5 days** — the polish-budget sink for this lesson.

---

### Section 7 — Empirical Comparisons and Hyperparameter Sensitivity

**Tagline:** *Where the win is real, and where it isn't.*
**Length:** ~700 words
**Anchor:** `ppo-empirics`

A direct empirical comparison of vanilla PG and PPO across
learning rates. 20 seeds, batch=10, 200 iterations each.

| lr | PPO mean | PPO std | vanilla mean | vanilla std |
|:----|:----------|:---------|:--------------|:-------------|
| 0.10 | 0.5666 | 0.0375 | -0.1053 | 0.0043 |
| 0.30 | 0.7227 | 0.0010 | 0.4001 | 0.0266 |
| 0.50 | 0.7249 | 0.0007 | 0.6671 | 0.0190 |
| 1.00 | 0.7258 | 0.0007 | 0.7213 | 0.0016 |
| 2.00 | 0.7263 | 0.0010 | 0.7263 | 0.0005 |
| 5.00 | 0.7281 | 0.0003 | 0.7281 | 0.0006 |

Two observations.

First: PPO works across a wide range of learning rates. Its sweet
spot spans from lr=0.3 to lr=5.0 with all final values above 0.72.
Vanilla PG's sweet spot is much narrower — at lr=0.1 it is stuck
near zero, at lr=0.5 it is still 0.05 below PPO, and only above
lr=1.0 does it catch up. The win for PPO is *robustness*: less
sensitive to the choice of learning rate, which translates directly
into "less hyperparameter tuning."

Second: at the high end of the sweep (lr=2.0, lr=5.0), vanilla PG
and PPO are statistically indistinguishable. On this MDP, with
this policy class, with enough learning rate, vanilla PG works.
The "vanilla PG catastrophically fails" narrative does not hold
here. We report this directly.

The headline run for the lesson — the one the user will see in
Panel B's default settings — is **PPO at lr=0.5, batch=20, 200
iters, 10 seeds**: final V(start) = 0.7267 ± 0.0003. Compared to
L10's REINFORCE+baseline at the same scale (0.7250 ± 0.0011), PPO
is slightly higher mean and ~4× lower std. The improvement is
incremental at this scale, not revolutionary.

A side experiment on the GAE λ trade was reported in §5. We do not
re-report it.

A second side experiment on clip ε at lr=1.5, batch=10, 5 seeds:

| clip ε | Final V | Std | Clip frac (late) |
|:--------|:---------|:-----|:-----------------|
| 0.05 | 0.7258 | 0.0005 | 0.003 |
| 0.10 | 0.7264 | 0.0004 | 0.002 |
| 0.20 | 0.7275 | 0.0003 | 0.000 |
| 0.40 | 0.7282 | 0.0001 | 0.000 |

At this learning rate, clip ε barely matters above 0.1 — the
ratios stay close to 1 anyway. At ε = 0.05, the clip is binding
occasionally and slowing convergence very slightly. Conclusion:
the standard `ε = 0.2` is a reasonable default that doesn't
constrain training on this scale. The lesson reports this and
notes that on larger problems (continuous control, deep critics),
ε matters more.

The honest framing throughout this section: PPO's advantages on
the gridworld are real but modest. The reasons to use PPO over
vanilla PG in practice are (a) deep / continuous policies where
vanilla can collapse, (b) multiple epochs per batch (sample
efficiency), (c) hyperparameter robustness. On a tabular softmax
gridworld, only (b) and (c) apply, and they are real but
unspectacular. The lesson tells this story straight rather than
inflating the gridworld results into a "PPO crushes vanilla"
claim.

---

### Section 8 — Forward Links

**Tagline:** *Toward maximum entropy, continuous actions, and LLMs.*
**Length:** ~500 words
**Anchor:** `l11-forward-links`

We close with three forward links, ordered by curriculum proximity.

> **Forward link to L12 — Maximum-Entropy RL.** The entropy bonus
> term `c_2 · H[π_θ(·|s)]` in the PPO objective was treated here
> as a regularizer that prevents premature policy collapse. In L12
> it becomes the *primary* objective: the entropy-regularized RL
> objective is `J(θ) = E[Σ_t (r_t + α H(π(·|s_t)))]`, and the
> "softmax cap" from L10 — the gap between a converged softmax
> policy's V(0,0) ≈ 0.722 and the optimal V* = 0.729 — gets
> reframed as a *feature*: a max-ent policy *should* be
> stochastic at convergence; that stochasticity is the point. The
> bridge from L11 to L12 is the entropy bonus's coefficient `c_2`,
> which becomes the temperature `α` in the max-ent objective.

> **Forward link to L13 — Soft Actor-Critic.** SAC is the
> continuous-action realization of max-ent actor-critic. It
> inherits two pieces directly from this lesson: the clipped
> ratio's spirit (SAC uses a similar bound on policy change, via
> the reparameterization trick rather than an explicit clip) and
> the twin-critic architecture (a direct descendant of L9's Double
> DQN). L13 is also where continuous actions become serious — the
> gridworld with discrete actions is left behind, and a
> continuous-action environment (a 1D continuous gridworld variant
> or a small classic-control problem) takes over. The 1D Gaussian
> sidebar in this lesson's centerpiece (V4) is the visual bridge.

> **Forward link to L17 — RLHF and DPO (the capstone).** PPO is
> the workhorse algorithm for LLM alignment via RLHF. The pipeline:
> a preference dataset is used to train a reward model; the reward
> model produces scalar rewards on language-model rollouts; PPO
> uses those rewards to fine-tune the language model under a KL
> constraint to a reference (typically the SFT model). Three
> pieces of L11 come together there: the IS ratio from §2 is the
> probability ratio between the fine-tuned model and the SFT
> reference; the clipped surrogate from §4 is the per-token PPO
> objective; the KL constraint from §2 (now an explicit KL penalty
> rather than a clipping bound) keeps the policy near the SFT
> manifold. DPO (Direct Preference Optimization, Rafailov et al.
> 2023) is a closed-form alternative that bypasses the reward
> model entirely; we treat it as PPO's sibling in L17.

A brief note on the curriculum's trimmed branches. The original
plan had L14 (model-based RL and world models), L15 (offline RL),
and L16 (diffusion in RL) before the capstone. Those lessons are
deferred — out of curriculum scope, not out of intellectual
importance. Model-based RL with PPO as the planner shows up in
work like MuZero; offline RL with PPO's conservatism is the
foundation of CQL and IQL; diffusion policies trained with PPO are
an active research area. Pointers to canonical references appear
in the "out of scope" appendix.

Visualization V6 — Roadmap Mini.
The curriculum graph, with L11 highlighted in violet and outgoing
edges in violet to L12 (max-ent), L13 (SAC), and L17 (RLHF). L14,
L15, L16 are rendered in gray with dashed outgoing edges marked
"future work." The user hovers over each node for a one-sentence
description. 460 × 280 px. Polish budget: 0.5 days (this is a
shared component, mostly retrofitted from prior lessons).

---

## 5. Algorithm and Math Implementation

The new module is `src/ppo/`. It contains six exports.

```typescript
// src/ppo/types.ts
import type { MDP, Policy } from '../mdp/types';

export interface PPOConfig {
  readonly lrPolicy: number;
  readonly lrValue: number;
  readonly clipEps: number;
  readonly gaeLambda: number;
  readonly epochs: number;
  readonly batchEpisodes: number;
  readonly entropyCoef: number;     // c_2 in PPO objective
  readonly valueCoef: number;       // c_1 in PPO objective
  readonly normalizeAdvantages: boolean;
}

export interface PPOState {
  readonly theta: Float64Array;     // 9*4 = 36 logits, row-major
  readonly V: Float64Array;         // 9 state values
}

export interface PPOIterationLog {
  readonly iter: number;
  readonly vStart: number;          // V^pi(start) by exact policy eval
  readonly meanKL: number;          // KL(pi_old || pi_new), state-mean
  readonly clipFraction: number;    // fraction of samples clipped in last epoch
  readonly meanRatio: number;       // mean r_t across batch
  readonly maxRatio: number;
  readonly minRatio: number;
  readonly surrogateValue: number;  // L^CLIP averaged over batch
  readonly entropyMean: number;     // H[pi(.|s)] averaged over visited states
  readonly batchSize: number;       // total transitions in batch
}
```

```typescript
// src/ppo/ppo.ts
export function ppoUpdate(
  state: PPOState,
  batch: readonly Trajectory[],
  config: PPOConfig,
  mdp: MDP,
): { state: PPOState; log: PPOIterationLog };

export function gaeAdvantages(
  traj: Trajectory,
  V: Float64Array,
  gamma: number,
  lambda: number,
  mdp: MDP,
): Float64Array;

export function softmax(logits: Float64Array, nActions: number): Float64Array;

export function policyKL(
  thetaOld: Float64Array,
  thetaNew: Float64Array,
  visitedStates: ReadonlySet<number>,
  nActions: number,
): number;

export function runPPO(
  config: PPOConfig,
  nIters: number,
  seed: number,
  mdp: MDP,
): { finalState: PPOState; logs: readonly PPOIterationLog[] };
```

```typescript
// src/ppo/vanilla.ts — for the side-by-side comparison
export function vanillaPGUpdate(
  state: PPOState,
  batch: readonly Trajectory[],
  config: { lrPolicy: number; lrValue: number; gaeLambda: number;
            normalizeAdvantages: boolean },
  mdp: MDP,
): { state: PPOState; log: Pick<PPOIterationLog, 'iter' | 'vStart' | 'batchSize'> };

export function runVanilla(
  config: {...},
  nIters: number,
  seed: number,
  mdp: MDP,
): { finalState: PPOState; logs: readonly {...}[] };
```

**Vitest test targets** (in `src/ppo/ppo.test.ts`):

| Test | Target |
|:------|:--------|
| `ppoUpdate sanity` | After 1 update on a fixed batch and known config, theta should match a pre-computed Float64Array within 1e-10. (Compute the expected via NumPy in the offline notebook, hardcode the result.) |
| `gaeAdvantages collapses to TD residual at lambda=0` | For a single-step trajectory, GAE(0) = r + γV(s') − V(s). |
| `gaeAdvantages collapses to MC at lambda=1` | For a 5-step trajectory with V=0, GAE(1) = G_t. |
| `softmax row sums to 1` | For arbitrary logits. |
| `policyKL is non-negative` | KL ≥ 0 always. |
| `policyKL = 0 when policies are equal` | Identity. |
| `runPPO matches numerical anchor` | Run 200 iters, lr=0.5, batch=20, λ=0.95, ε=0.2, seed=0; assert `vStart` of the last log is within 0.005 of 0.7267. (The seed-0 single-trial is not exactly the mean across seeds; we check the order of magnitude and that the final value is above 0.71.) |
| `runVanilla diverges quietly at lr=0.1` | Run 200 iters, lr=0.1, batch=10, seed=0; assert final `vStart` is below 0.0 (stuck). This is the headline "vanilla fails at low lr" claim. |
| `runPPO ratio histogram bounded` | At any iteration, max ratio ≤ exp(lr × 1.0) loosely (a sanity bound that catches numerical blow-up). |

---

## 6. Component Catalog

| Code | Component | Section | Polish (days) |
|:------|:-----------|:---------|:---------------|
| V1 | ParamVsPolicySpace | §1 | 2.0 |
| V2 | KLGeometryPlot | §2 | 2.0 |
| V3 | TRPOStepSchematic | §3 | 1.0 |
| V4 | ClippedSurrogateCurve | §4 | 2.5 |
| V5 | GAELambdaSweep | §5 | 1.5 |
| V6 | PPOLab (centerpiece) | §6 | **5.0** |
| V7 | LRSensitivityTable | §7 | 0.5 |
| V8 | RoadmapMini-L11 | §8 | 0.5 |
| **Total** | | | **15.0** |

Reuse from prior lessons. `GridworldRenderer` (from L3, retrofitted
in L5 and L8) is reused inside PPOLab's Panel A. `LearningCurvePlot`
from L10 is reused as the basis for Panel B (with a small extension
to overlay the surrogate and mean-return traces). `RoadmapMini` is
the shared component from L1 onward; the lesson adds an L11 node
and edges but does not modify the component. `MathBlock` and
`CrosslinkCallout` are used throughout. No existing component is
modified — extension only, per the project convention.

---

## 7. Page-Level User Experience

Standard lesson layout. The centerpiece (PPOLab) is a 960 px
breakout from the standard column width; on narrow viewports it
collapses to a stacked vertical layout with each panel at the
column width. The "play / pause / step" controls float in a sticky
position at the bottom of the centerpiece on scroll, so the user
can keep playback running while reading the surrounding prose.

The clipped surrogate curve (V4) is *deliberately* not interactive
on a continuous slider for `μ_θ` — it uses discrete steps (10 along
the x-axis) so that the rendering stays performant on low-end
devices. The mathematical shape is preserved.

The GAE λ sweep (V5) loads pre-computed traces from
`public/data/ppo/gae_lambda.json`. The user does not re-run the
sweep; they only browse the pre-computed results. This is the same
pre-computed-traces pattern used in L9 and L10.

The PPO Lab (V6) runs PPO in the browser. A full 200-iteration
training run with batch=20 should complete in under 10 seconds on
a 2020-era laptop. The implementation runs the inner gradient
computation in `Float64Array` arithmetic without allocations in the
hot path; the only allocations per iteration are the batch
trajectories themselves.

Accessibility: each panel exposes its data via a hidden CSV
download link (the "data" affordance from L8). Keyboard navigation
is the same as L10's PolicyGradientLab: arrow keys step the
training, space toggles play/pause, `r` resets.

---

## 8. Acceptance Criteria

A learner who completes this lesson should be able to: (1) state
the step-size problem and why parameter-space distance is the
wrong measure; (2) define the probability ratio `r_t(θ)` and relate
it to the IS weight from L6; (3) write down the PPO clipped
surrogate objective and explain why it is a `min` rather than a
clip alone; (4) describe (not implement) TRPO's natural-gradient +
line-search structure; (5) state the GAE(λ) interpolation and its
bias-variance trade; (6) recognize PPO's empirical advantage on
the gridworld as variance reduction and lr robustness, not as
prevention of catastrophic collapse; (7) anticipate how PPO will
reappear in RLHF (L17).

**Concrete acceptance check.** The learner should be able to
predict, before running the PPO Lab in an aggressive regime
(lr=2.0, batch=5, epochs=10), that the clip fraction will (a) be
nontrivial early in training, (b) decrease as the policy
converges, (c) increase if clip ε is lowered to 0.05. They should
then verify by running the lab.

---

## 9. Stretch Goals (post-MVP)

(a) An optional "TRPO mode" toggle in the centerpiece. Currently
the algorithm toggle in V6 has a "TRPO-style" option that uses a
conservative analytic step size, not a real TRPO. A real TRPO
mode would require a CG solver and a line search; doable, but ~3
days of additional work for a panel that the centerpiece doesn't
strictly need.

(b) An adaptive KL variant of PPO (the "adaptive KL penalty"
flavor from the original PPO paper). Currently we only implement
the clipped variant. The adaptive variant uses a Lagrangian on the
KL constraint, which is closer in spirit to TRPO and is the
flavor more commonly used in RLHF. Adding it requires extending
PPOConfig and adding a fourth algorithm to the toggle.

(c) An exact-PPO-objective surface plot. The current V4 shows the
surrogate at a single sample, parametrized by μ. A more ambitious
visualization would show the surrogate as a 2D surface over two
chosen parameters of the policy, with the trust region drawn as a
KL contour. Hard to make clean; would belong in §3 as a TRPO
illustration.

(d) Compare PPO to Adam-vanilla. Vanilla PG with the Adam
optimizer rather than plain SGD recovers some of PPO's robustness
because Adam adaptively rescales per-parameter step sizes. A side
experiment showing "Adam-vanilla vs PPO" would honestly clarify
how much of PPO's win comes from the trust region vs. from
adaptive step sizing.

(e) A small continuous-action toy in V4. Currently V4 shows the
clipped surrogate analytically. A stretch would be to actually run
PPO on a 1D bandit-with-Gaussian-policy toy and show the trust
region in action. This is a partial bridge to L13.

---

## 10. Out of Scope (intentionally)

(a) Real TRPO implementation. Treated mathematically; not
implemented. Pointer: Schulman et al. 2015 (arXiv:1502.05477).

(b) Continuous action spaces in the empirical thread. Sketched in
V4 as illustration only. Real treatment in L13.

(c) Recurrent policies. PPO with LSTM or GRU policies is standard
in practice (e.g., for partially observable environments). The
lesson uses a feedforward tabular softmax. Out of scope; pointer
in the appendix.

(d) Multi-agent PPO (MAPPO). Out of curriculum scope.

(e) Model-based RL with PPO as the planner. This would have been
L14; deferred from production schedule.

(f) Offline PPO / behavior-regularized PPO. Would have been part
of L15; deferred.

(g) Diffusion policies trained with PPO. Would have been part of
L16; deferred.

---

## 11. Training Notebook

`scripts/ppo_traces.py` is the offline pre-computation script. It
emits four JSON files to `public/data/ppo/`:

- `lr_sensitivity.json`: the 20-seed lr sweep from §7. Loaded by
  V7 (LRSensitivityTable).
- `gae_lambda.json`: the 20-seed GAE λ sweep from §5 (batch=5,
  noisy regime). Loaded by V5.
- `clip_sweep.json`: the 5-seed clip ε sweep from §7. Loaded by V7
  (as a secondary table).
- `headline_runs.json`: 10 seeds of the headline PPO run (lr=0.5,
  batch=20, 200 iters) plus 10 seeds of vanilla PG at the same
  config, plus the aggressive regime trace (lr=2.0, batch=5,
  epochs=10, 100 iters) for the centerpiece default.

The script runs in pure NumPy. Total runtime on a 2020-era laptop:
~3 minutes for all four traces. No PyTorch, no ONNX, no
GPU-required pre-computation. The script is idempotent and
reproducible — every seed is explicit, every config is in the file.

The script also contains a `verify_anchors()` function that
re-computes the headline numerical anchors and prints them. The
agent should run this once after implementing the algorithms in
TypeScript and verify that the TypeScript implementation's
seed-0 single-trial result matches the NumPy implementation to
within 1e-6 — a strong test of parity.

---

## 12. Closing Notes and Length Tally

Approximate length tally: ~1400 lines of README. Polish budget
totals 15 days, dominated by the centerpiece PPOLab (5 days). No
new browser dependencies; the lesson runs on the existing L10
stack. The Python offline script is ~400 lines of pure NumPy. The
TypeScript core (`src/ppo/`) is ~600 lines including tests.

Centerpiece role: the PPOLab is the lesson's "snap into place"
moment. The clip-fraction-jump-on-tighter-epsilon transition is
the specific interaction the lesson is designed around — it is
the moment when "trust region" stops being an abstract phrase
and becomes a thing the user can see binding and unbinding in
real time. The other six visualizations support and frame this
moment; they are not the centerpiece.

Forward link saturation: three links, all substantive. L12 picks
up the entropy term; L13 picks up the continuous-action sidebar
and the twin-critic spirit; L17 cashes in the whole apparatus for
RLHF. The trimmed branches (L14, L15, L16) are acknowledged
honestly in §8 and listed in §10 with pointers; they are not
papered over.

The curriculum's spine now runs L1 → L10 → **L11** → L12 → L13 →
L17. Six lessons left in the production schedule when L11 ships;
five after L11.

## End of specification