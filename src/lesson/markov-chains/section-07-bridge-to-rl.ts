import type { Section } from "../section";
import { sectionFromHTML } from "../section";
import { forwardLink } from "../../components/CrosslinkCallout";

export const mcSection07: Section = {
  id: "policy-induced-chain",
  title: "The Bridge to RL",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§7</span>The Bridge to RL</h2>
<p class="tagline">A policy turns an MDP into a Markov chain, and everything we just learned applies.</p>

<p>Here's the one paragraph that justifies this prereq.</p>

<p>In Lesson 2, an <strong>MDP</strong> will be defined as $(\\mathcal{S}, \\mathcal{A}, P, r, \\gamma)$
— states, actions, transition kernel, reward, discount. The transition kernel
is now $P(s' \\mid s, a)$ — it depends on a chosen <em>action</em>. An MDP is not yet
a Markov chain on states; it's a <em>controlled</em> process. But the moment we fix
a <strong>policy</strong> $\\pi(a \\mid s)$ (a rule for choosing actions given states),
something magical happens: the marginal-over-actions transition</p>

$$\\boxed{P^\\pi_{ss'} \\;=\\; \\sum_a \\pi(a \\mid s) \\, P(s' \\mid s, a)}$$

<p>is an ordinary $|S| \\times |S|$ stochastic matrix. The pair $(P^\\pi, \\mu_0)$
<em>is</em> a Markov chain on the state space. The state visitation distribution
under policy $\\pi$ is the stationary distribution $d^\\pi$ of $P^\\pi$.</p>

<p>Every theorem from this lesson applies:</p>

<ul>
  <li><strong>Irreducibility of $P^\\pi$</strong> ⇔ the policy can reach every state from
  every state (with positive probability under randomness in $\\pi$ and $P$).</li>
  <li><strong>Aperiodicity of $P^\\pi$</strong> ⇔ the policy doesn't trap the chain in periodic
  cycles.</li>
  <li><strong>Ergodicity of $P^\\pi$</strong> ⇔ the policy has a unique stationary distribution
  that <em>averages over the agent's behaviour</em>.</li>
</ul>

<p>Three concrete payoffs downstream:</p>

<ol>
  <li><strong>Policy evaluation (Lessons 3, 5).</strong> When TD(0) or Monte Carlo evaluates
  $V^\\pi(s) = \\mathbb{E}_\\pi[\\sum_t \\gamma^t r_t \\mid s_0 = s]$, the <em>distribution
  of states encountered</em> under the policy is the stationary distribution of
  $P^\\pi$. Sample complexity bounds depend on the spectral gap.</li>
  <li><strong>Policy gradient (Lesson 8).</strong> The policy gradient theorem will say
  $$\\nabla_\\theta J(\\theta) = \\mathbb{E}_{s \\sim d^{\\pi_\\theta},\\, a \\sim \\pi_\\theta}[\\nabla_\\theta \\log \\pi_\\theta(a|s) \\cdot Q^\\pi(s,a)].$$
  That $s \\sim d^{\\pi_\\theta}$ is exactly the stationary distribution we
  just defined. Without it, "on-policy" has no meaning.</li>
  <li><strong>Off-policy correction (Lessons 4, 15).</strong> When we use samples from one
  policy to evaluate another, the <em>correction factor</em> is a ratio of
  stationary distributions — importance sampling on chains. This is where
  the entire offline-RL story begins.</li>
</ol>

<p>You now have the substrate. Lesson 2 builds the structure of decision-making
on top of it.</p>

${forwardLink({
  destination: "Lesson 2 — Markov Decision Processes",
  html: `<p>The gridworld below is one policy away from a full MDP. Add a reward signal and a
  discount, ask for the policy that <em>maximizes</em> return, and the Bellman equations
  arrive. Lesson 2 generalizes this preview into a full MDP explorer with values, rewards,
  and Bellman backups — reusing this very transition-graph machinery.</p>`,
})}

<p>A 4×4 gridworld with a chosen policy. The induced $P^\\pi$ is a plain
$16\\times16$ chain; its stationary $d^\\pi$ shades the cells (darker = visited
more), and the agent token walks a sampled trajectory. Switch policies and
watch $d^\\pi$ shift — this is "policy → chain" made visceral.</p>

<policy-induced-chain-preview></policy-induced-chain-preview>`,
    );
  },
};
