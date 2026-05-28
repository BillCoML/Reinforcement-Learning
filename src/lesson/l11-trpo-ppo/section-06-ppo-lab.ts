import type { Section } from "../section";
import { sectionFromHTML } from "../section";

export const ppoSection06: Section = {
  id: "ppo-lab",
  title: "PPO Lab",
  build() {
    return sectionFromHTML(this.id, `
<h2><span class="sec-num">§6</span>PPO Lab</h2>
<p class="tagline"><em>The clipping mechanism, in motion.</em></p>

<p>The clipped surrogate of §4 is best understood by watching it bind.
The five-panel lab below runs PPO on the 3×3 gridworld and exposes every internal mechanism
introduced so far — policy, learning curve, clip fraction, KL divergence, and the ratio histogram —
synchronized to a single iteration counter.</p>

<p>The lab defaults to an <strong>aggressive regime</strong> (lr=2.0, batch=5, epochs=10)
where the clipping mechanism is visibly active. In this regime:</p>
<ul>
<li><strong>Panel C</strong> (clip fraction) climbs to 30–45% in the first 20 iterations,
then falls to ~2% as the policy converges. The clip is doing meaningful work during the
fast-learning phase and naturally subsides.</li>
<li><strong>Panel D</strong> (KL trace) spikes above $\\delta = 0.01$ (the typical TRPO budget)
early in training, then falls well below it. PPO's implicit trust region is looser than TRPO's
explicit constraint — and on this problem, that's fine.</li>
<li><strong>Panel E</strong> (ratio histogram) shows the probability ratios spreading from a
delta at 1.0 into a distribution whose tails extend past the clip band.</li>
</ul>

<p><strong>The pedagogical moment:</strong> slide the clip $\\varepsilon$ from 0.2 down to 0.05.
Panel C jumps from ~30% to ~85% clip fraction; Panel D's KL trace flattens; Panel B's
learning curve slows visibly; Panel E becomes a sharp spike at exactly 1.0
(everything is being clipped, so no update). This is the trust region <em>tightening</em>
in real time — and its cost: slower learning. Slide $\\varepsilon$ back to 0.2
and learning resumes.</p>

<ppo-lab></ppo-lab>
`);
  },
};
