import type { Section } from '../section';
import { sectionFromHTML } from '../section';

export const isSection01: Section = {
  id: 'is-identity',
  title: 'The Importance Sampling Identity',
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§1</span>The Importance Sampling Identity</h2>
<p class="tagline">Change of measure with a multiplicative correction.</p>

<p>Suppose we want to compute $\\mathbb{E}_p[f(X)]$, the expectation of some function
$f$ of a random variable distributed according to $p$. The obvious approach is to
sample $X_i \\sim p$, compute each $f(X_i)$, and average. By the law of large numbers,</p>

$$\\frac{1}{N} \\sum_{i=1}^N f(X_i) \\;\\xrightarrow{a.s.}\\; \\mathbb{E}_p[f(X)].$$

<p>Sometimes we cannot sample from $p$. The reasons vary: $p$ is a target policy we
want to evaluate but we only have trajectories from some other behavior policy; $p$ is
computationally expensive to sample from while a related proposal $q$ is cheap; $p$ is
defined only up to a normalization constant (this case will recur in Lesson 12 when we
look at energy-based policies). In each of these cases we have samples from a different
distribution $q$ and want to use them to estimate an expectation under $p$.</p>

<p>The <strong>importance sampling identity</strong> is the elementary algebraic fact that
makes this possible. For any two probability distributions $p$ and $q$ such that
$q(x) > 0$ wherever $p(x) f(x) \\neq 0$,</p>

$$\\boxed{\\mathbb{E}_p[f(X)] \\;=\\; \\int f(x) \\, p(x) \\, dx \\;=\\;
\\int f(x) \\, \\frac{p(x)}{q(x)} \\, q(x) \\, dx \\;=\\;
\\mathbb{E}_q\\!\\left[ f(X) \\, \\frac{p(X)}{q(X)} \\right].}$$

<p>The trick is a multiplication-and-division by $q(x)$. The new integrand is
$f(x) \\cdot w(x)$ evaluated under $q$, where $w(x) := p(x)/q(x)$ is called the
<strong>importance weight</strong>. The identity says we can compute an expectation
under $p$ by sampling under $q$ and reweighting each sample by $w$.</p>

<p>The single requirement is that $q$ "cover" the support of $p$ on the regions where
$f$ is non-zero. Formally: $q(x) > 0$ whenever $p(x) f(x) \\neq 0$. If this fails —
if there are regions where $p$ has mass but $q$ does not — then the IS identity is
technically infinite, because we would be dividing by zero on a non-negligible region.
In reinforcement learning this requirement is called the <strong>coverage
condition</strong>, and it has a clean policy-language translation: a behavior policy
that assigns zero probability to an action that the target policy might take cannot
be used to evaluate that target policy. Coverage is the single most important
assumption in off-policy methods.</p>

<h3>The two canonical estimators</h3>

<p>Given $N$ samples $X_1, \\ldots, X_N \\sim q$, there are two natural ways to turn
the identity into an estimator. The first is the <strong>ordinary importance sampling
estimator</strong>:</p>

$$\\boxed{\\hat\\mu_{\\text{ord}} \\;:=\\; \\frac{1}{N} \\sum_{i=1}^N w(X_i) \\, f(X_i).}$$

<p>This estimator is <strong>unbiased</strong> in a strong sense: by linearity of
expectation under $q$, we have $\\mathbb{E}_q[\\hat\\mu_{\\text{ord}}] =
\\mathbb{E}_p[f(X)]$ exactly, with no error term that vanishes only asymptotically.</p>

<p>The second estimator is the <strong>weighted</strong> or
<strong>self-normalized importance sampling estimator</strong>:</p>

$$\\boxed{\\hat\\mu_{\\text{wt}} \\;:=\\; \\frac{\\sum_{i=1}^N w(X_i) \\, f(X_i)}{\\sum_{i=1}^N w(X_i)}.}$$

<p>The denominator is an estimator of $\\mathbb{E}_q[w(X)] = \\int (p/q)\\,q = \\int p = 1$,
so for large $N$ the denominator is close to 1 and the two estimators agree.
For small or moderate $N$, the denominator can differ noticeably from 1, and
the two estimators behave differently. This difference is what §2 is about.</p>

<div class="component-host">
  <is-identity-demonstrator></is-identity-demonstrator>
</div>
`);
  },
};
