import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const contrSection04: Section = {
  id: "completeness-required",
  title: "When Contractions Fail",
  build() {
    return sectionFromHTML(this.id, `<h2><span class="sec-num">§4</span>When Contractions Fail</h2>
<p class="tagline">Three things can go wrong, and each illuminates a hypothesis.</p>

<p>The theorem has three hypotheses: $T$ is a contraction, $T : X \\to X$ maps $X$ to
itself, and $X$ is complete. Each hypothesis is <em>necessary</em> — violate one and the
conclusion can fail.</p>

<p><strong>Failure 1: $c = 1$ (translation).</strong> Consider $T(x) = x + 1$ on $\\mathbb{R}$.
This is <em>non-expansive</em> — it preserves distances exactly. But the iteration is
$0, 1, 2, 3, \\ldots$, which diverges. No fixed point exists, because $T(x) = x
\\iff x + 1 = x$ has no solution. The strict inequality $c &lt; 1$ prevents this.</p>

<p><strong>Failure 2: Non-completeness (rationals).</strong> Consider $T(x) = (x + 2/x)/2$
on $\\mathbb{Q}_{>0}$. This is Newton's method for finding $\\sqrt{2}$. Iterating
from $x_0 = 1$ produces $3/2, 17/12, 577/408, \\ldots$ — a Cauchy sequence that
converges to $\\sqrt{2}$. But $\\sqrt{2} \\notin \\mathbb{Q}$. The space isn't
complete, so the iteration "escapes" the space.</p>

<p>Switch the same $T$ to $\\mathbb{R}_{>0}$: now completeness holds, the theorem
applies, and the iteration converges to $\\sqrt{2}$. Same map, different space,
different fate.</p>

<p><strong>Failure 3: Multiple fixed points (when $T$ isn't a contraction).</strong>
Consider $T(x) = x^2$ on $[0, 1]$. Two fixed points exist: $0$ and $1$.
Iterates from $x_0 = 0.9$ converge to $0$; iterates from $x_0 = 1.0$ stay at $1$.
The map is <em>not</em> a contraction on $[0,1]$ because $|T'(x)| = 2x$ can be as
large as $2$ at $x = 1$.</p>

<blockquote>
<strong>Important nuance.</strong> In RL we need a <em>global</em> contraction on a full vector
space — $T^\\pi : \\mathbb{R}^K \\to \\mathbb{R}^K$. We can't rely on locality.
The proof in §5 shows that the contraction constant $\\gamma$ is <em>uniform</em>
over the entire space, which is what makes value iteration converge from any
starting $V$.
</blockquote>

<counterexample-gallery></counterexample-gallery>`);
  },
};
