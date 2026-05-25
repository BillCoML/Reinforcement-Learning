import type { Section } from "../section";
import { sectionFromHTML } from "../section";
import { sidebar } from "../../components/CrosslinkCallout";

export const mcSection03: Section = {
  id: "communicating-classes",
  title: "Classifying States",
  build() {
    return sectionFromHTML(
      this.id,
      `<h2><span class="sec-num">§3</span>Classifying States</h2>
<p class="tagline">Reachability, recurrence, periodicity — three independent axes.</p>

<p>To predict whether $P^n$ converges (and to what), we need a vocabulary for
the <em>structure</em> of a chain. Three classifications matter for RL:
<strong>reachability</strong> (which states can be reached from which), <strong>recurrence</strong>
(do you keep coming back?), and <strong>periodicity</strong> (do you return on a regular
schedule?).</p>

<p><strong>Reachability and communicating classes.</strong> State $j$ is <em>reachable</em> from
state $i$ if there exists $n \\geq 0$ such that $(P^n)_{ij} > 0$. (We allow
$n=0$, so every state is trivially reachable from itself.) States $i$ and
$j$ <strong>communicate</strong> if each is reachable from the other; this is an
equivalence relation, partitioning $\\mathcal{S}$ into <strong>communicating
classes</strong>.</p>

<p>A chain is <strong>irreducible</strong> if there's exactly one communicating class — every
state can reach every other. The weather chain is irreducible. A chain that
splits into two non-communicating classes is <strong>reducible</strong>.</p>

<p><strong>Recurrence.</strong> A state $i$ is <em>recurrent</em> if, starting from $i$, the chain
returns to $i$ with probability 1. Otherwise it is <em>transient</em>. In a finite
chain, every state is either recurrent or transient, and recurrence is a
<strong>class property</strong> — all states in a communicating class are recurrent
together or transient together. In a finite irreducible chain <em>every state
is recurrent</em> (an easy consequence of finiteness + irreducibility).</p>

<p><strong>Periodicity.</strong> The <em>period</em> of state $i$ is the GCD of all $n \\geq 1$ for
which $(P^n)_{ii} > 0$. If the period is 1, the state is <strong>aperiodic</strong>.
Aperiodicity is a class property; an irreducible chain has a single period.</p>

${sidebar(
  "Quick check",
  `<p>The two-state chain with $P=\\begin{pmatrix}0&1\\\\1&0\\end{pmatrix}$
is irreducible, recurrent, and has period 2 (returns happen only at even $n$).
The weather chain is irreducible, recurrent, and aperiodic — the diagonal
entries $P_{ii} > 0$, so $(P^1)_{ii} > 0$ and $\\gcd\\{1, 2, 3, \\ldots\\} = 1$.</p>`,
)}

<p>A useful shorthand: a finite chain is called <strong>ergodic</strong> if it is
irreducible and aperiodic. (Some authors require also "positive recurrent,"
which is automatic in finite chains.) Ergodic finite chains are precisely
the ones for which $P^n$ converges to a rank-one matrix $\\mathbf{1} \\pi^\\top$.
This is the punchline; we'll prove it in §5.</p>

<h3>Counterexample gallery (pre-verified)</h3>

<p><strong>Reducible 4-state chain.</strong> Consider</p>

$$P = \\begin{pmatrix}
0.7 & 0.3 & 0   & 0   \\\\
0.4 & 0.6 & 0   & 0   \\\\
0.1 & 0   & 0.3 & 0.6 \\\\
0   & 0.2 & 0.5 & 0.3
\\end{pmatrix}.$$

<p>States $\\{0, 1\\}$ communicate (forming a recurrent class). States $\\{2, 3\\}$
also communicate with each other but they "leak" into $\\{0, 1\\}$ via
$P_{20} = 0.1$ and $P_{31} = 0.2$ — and crucially nothing in $\\{0,1\\}$
returns to $\\{2,3\\}$. So $\\{0,1\\}$ is recurrent, $\\{2,3\\}$ is <strong>transient</strong>.</p>

<p>$P^{50}$ starting from state 2 (transient):
$(0.571, \\; 0.428, \\; 0.000, \\; 0.000)$</p>

<p>Nearly all mass has leaked into the absorbing class. $P^{50}$ starting from
state 0 (in the absorbing class):
$(0.571, \\; 0.429, \\; 0, \\; 0)$</p>

<p>Same limiting masses on the recurrent states, but exactly zero on transient
ones from the very first step (since the recurrent class is closed).</p>

${sidebar(
  "Sidebar",
  `<p>This is the structure of every <strong>episodic MDP</strong>: terminal
states form an absorbing class. The agent's job is to navigate the
transient set toward favorable absorbing states. Lesson 2 will rest on
exactly this picture.</p>`,
)}

<p>Edit the matrix or load a preset; the graph recolors nodes by communicating
class and the summary reports recurrence and period as you go.</p>

<state-classification-inspector></state-classification-inspector>`,
    );
  },
};
