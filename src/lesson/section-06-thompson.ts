import type { Section } from "./section";
import { sectionFromHTML } from "./section";
import { sidebar } from "../components/CrosslinkCallout";

export const section06: Section = {
  id: "thompson-sampling",
  title: "Thompson Sampling: Posterior in Action",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§6</span>Thompson Sampling: Posterior in Action</h2>
<p class="tagline">Pull as if your beliefs were true.</p>

<p>Thompson sampling (Thompson, 1933 — yes, <em>1933</em>, predates RL by half a century)
takes a Bayesian view. Maintain a <em>posterior distribution</em> over each $\\mu_i$. At each
step, <em>sample</em> once from each posterior, and pull the arm whose sample is highest.
The act of sampling does all the exploration work, automatically and naturally.</p>

<h3>Conjugate setup</h3>
<p>For Bernoulli rewards, the natural prior on $\\mu_i$ is
$\\text{Beta}(\\alpha_i, \\beta_i)$. After observing $s$ successes and $f$ failures, the posterior
becomes $\\text{Beta}(\\alpha_i + s, \\beta_i + f)$. (This is the Beta-Bernoulli conjugacy you may
have met in Bayesian stats; it's the canonical example.) Starting from the uniform
prior $\\text{Beta}(1, 1)$, the posterior at any time $t$ is</p>

$$p(\\mu_i \\mid \\text{data}) \\;=\\; \\text{Beta}(1 + S_i,\\ 1 + N_i - S_i),$$

<p>with mean $\\mathbb{E}[\\mu_i \\mid \\text{data}] = (1 + S_i) / (2 + N_i)$ and variance
$\\alpha\\beta / ((\\alpha + \\beta)^2 (\\alpha + \\beta + 1))$. Beta becomes increasingly peaked around the true
mean as $N_i$ grows — this is the same concentration phenomenon that made UCB's
confidence bonus shrink.</p>

<table>
  <thead><tr><th>Posterior</th><th>mean</th><th>std</th><th>shape</th></tr></thead>
  <tbody>
    <tr><td>Beta(1,1)</td><td>0.500</td><td>0.289</td><td>uniform</td></tr>
    <tr><td>Beta(8,4)</td><td>0.667</td><td>0.131</td><td>mild peak</td></tr>
    <tr><td>Beta(50,30)</td><td>0.625</td><td>0.054</td><td>sharp peak</td></tr>
    <tr><td>Beta(200,100)</td><td>0.667</td><td>0.027</td><td>very sharp</td></tr>
  </tbody>
</table>

<h3>Algorithm (Thompson sampling, Beta-Bernoulli)</h3>
<pre><code>α_i = β_i = 1 for all i
for t = 1, 2, …, T:
    for each arm i:
        θ_i ~ Beta(α_i, β_i)
    a_t = argmax_i θ_i
    pull a_t, observe R_t ∈ {0, 1}
    if R_t = 1: α_{a_t} += 1
    else:       β_{a_t} += 1</code></pre>

<p><strong>Why it works (intuition).</strong> Each posterior carries both estimate and uncertainty
in a single object. A <em>sample</em> from $\\text{Beta}(\\alpha, \\beta)$ lands near the mean if the
posterior is sharp, and lands far from the mean if it's broad. So arms with
high uncertainty get a "free shot" at being picked, automatically. Arms whose
posteriors have collapsed near $\\mu_i = 0$ will essentially never be sampled again.</p>

<h3>Regret bound</h3>
<p>For Bernoulli bandits with uniform priors, Thompson sampling
satisfies (Agrawal &amp; Goyal, 2012; Kaufmann et al., 2012)</p>

$$R_T \\;\\leq\\; \\left(\\sum_{i: \\Delta_i > 0} \\frac{\\Delta_i}{\\mathrm{KL}(\\nu_i \\,\\|\\, \\nu^*)}\\right) \\log T \\;+\\; O(\\sqrt{KT \\log T})$$

<p>— matching the Lai–Robbins lower bound <em>up to lower-order terms</em>. Thompson sampling
is, in the strongest sense, asymptotically optimal for Bernoulli bandits.
Empirically (our simulation at $T = 5000$): $R_T \\approx 17$ against the Lai–Robbins
floor of $29.6$. The bound is sharper than UCB1's and the empirical performance matches.</p>

${sidebar(
  "Bayesian aside",
  `<p>Thompson sampling is exactly <em>one step of the posterior-predictive game</em>: "given
  current beliefs, play optimally w.r.t. a sample from the posterior." This recurs all
  over modern RL — <strong>Bayesian DQN</strong> with a variational posterior on Q-weights
  (Osband et al. 2016), <strong>PSRL</strong> (Posterior Sampling for RL) lifting Thompson to full
  MDPs, and Bayesian model-based RL sampling a posterior over dynamics. The
  Beta-Bernoulli case is the simplest instantiation. The principle scales.</p>`,
)}

<p>The animation below replays a real Thompson run on the running example. Each step
samples a $\\theta_i$ from every posterior (the dropping markers), pulls the arm with the
highest sample, and tweens its Beta curve to the updated posterior. Around $t = 30$ you'll
see arm 3's belief has sharpened the most and grabs nearly all the pulls — exploration
shutting itself off, automatically.</p>

<thompson-posterior-evolution></thompson-posterior-evolution>`,
    );
  },
};
