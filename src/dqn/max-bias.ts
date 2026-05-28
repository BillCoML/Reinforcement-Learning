/**
 * Sutton-Barto maximization-bias MDP (Figure 6.7 / Example 6.7 in SB2).
 * Two states: A and B. Terminal state T.
 *
 * From A: action "right" → T (reward 0), action "left" → B (reward 0).
 * From B: N_B actions, each → T with reward ~ N(−0.1, 1).
 *
 * Optimal: always choose "right" from A (expected return 0 > −0.1).
 * DQN (standard Q-learning) shows systematic overestimation of Q(A, left)
 * because max over noisy Q(B, ·) is always optimistic.
 * Double DQN decouples action selection from evaluation, reducing bias.
 */
import { mulberry32 } from "../importance-sampling/gaussian";

export type RNG = () => number;

const N_B_ACTIONS = 10; // number of actions from state B
const A_STATE = 0;
const B_STATE = 1;
const TERMINAL = 2;
const MEAN_B_REWARD = -0.1;
const STD_B_REWARD = 1.0;

/** Box-Muller normal sample. */
function normalSample(rng: RNG): number {
  const u1 = rng();
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1 + 1e-15)) * Math.cos(2 * Math.PI * u2);
}

export interface MaxBiasResult {
  /** Per-episode fraction of left-from-A choices across all episodes, averaged over seeds. */
  leftFractionPerEpisode: number[];
  /** Per-episode estimated Q(A, left), averaged over seeds. */
  qALeftPerEpisode: number[];
}

export interface MaxBiasOptions {
  nEpisodes?: number;
  nSeeds?: number;
  alpha?: number;
  epsilon?: number;
  gamma?: number;
  useDoubleDqn?: boolean;
  masterSeed?: number;
}

/**
 * Simulate DQN (standard Q-learning) or Double DQN on the max-bias MDP.
 * Returns per-episode statistics averaged across seeds.
 */
export function runMaxBias(options: MaxBiasOptions = {}): MaxBiasResult {
  const {
    nEpisodes = 300,
    nSeeds = 1000,
    alpha = 0.1,
    epsilon = 0.1,
    gamma = 1.0,
    useDoubleDqn = false,
    masterSeed = 42,
  } = options;

  // Total Q-value slots: A has 2 actions (left=0, right=1), B has N_B_ACTIONS
  // For Double DQN, we use two separate Q-tables
  const nAActions = 2; // from A: left(0), right(1)
  const sumLeftFraction = new Float64Array(nEpisodes);
  const sumQALeft = new Float64Array(nEpisodes);

  const seedRng = mulberry32(masterSeed);

  for (let seed = 0; seed < nSeeds; seed++) {
    const rng = mulberry32((seedRng() * 2 ** 32) >>> 0);

    // Q1[s][a]: primary Q-table
    const Q1_A = new Float64Array(nAActions); // A has 2 actions
    const Q1_B = new Float64Array(N_B_ACTIONS);
    // Q2[s][a]: secondary Q-table for Double DQN
    const Q2_B = new Float64Array(N_B_ACTIONS);

    let leftChoices = 0;
    let totalAVisits = 0;

    for (let ep = 0; ep < nEpisodes; ep++) {
      let s = A_STATE;

      for (let step = 0; step < 50; step++) {
        if (s === TERMINAL) break;

        // ε-greedy action selection (always using Q1 for selection)
        let a: number;
        if (s === A_STATE) {
          if (rng() < epsilon) {
            a = Math.floor(rng() * nAActions);
          } else {
            a = Q1_A[0] >= Q1_A[1] ? 0 : 1;
          }

          totalAVisits++;
          if (a === 0) leftChoices++; // left

          let sp: number;
          let r: number;
          if (a === 0) {
            sp = B_STATE;
            r = 0;
          } else {
            sp = TERMINAL;
            r = 0;
          }

          if (sp === B_STATE) {
            // Choose action from B (ε-greedy)
            let bAction: number;
            if (rng() < epsilon) {
              bAction = Math.floor(rng() * N_B_ACTIONS);
            } else {
              bAction = argmax(Q1_B);
            }

            // Reward from B
            const rb = MEAN_B_REWARD + STD_B_REWARD * normalSample(rng);

            // Update Q(B, bAction)
            if (useDoubleDqn) {
              // Each step, randomly assign update to Q1 or Q2
              if (rng() < 0.5) {
                Q1_B[bAction] += alpha * (rb - Q1_B[bAction]);
              } else {
                Q2_B[bAction] += alpha * (rb - Q2_B[bAction]);
              }
            } else {
              Q1_B[bAction] += alpha * (rb - Q1_B[bAction]);
            }

            // Update Q(A, left=0): target is max Q(B, ·)
            let nextVal: number;
            if (useDoubleDqn) {
              // Double DQN: select action from Q1, evaluate with Q2
              const bestBAction = argmax(Q1_B);
              nextVal = Q2_B[bestBAction];
            } else {
              nextVal = Math.max(...Array.from(Q1_B));
            }

            if (useDoubleDqn) {
              Q1_A[0] += alpha * (r + gamma * nextVal - Q1_A[0]);
            } else {
              Q1_A[0] += alpha * (r + gamma * nextVal - Q1_A[0]);
            }

            s = TERMINAL;
          } else {
            // right → terminal: Q(A, right) update
            if (useDoubleDqn) {
              Q1_A[1] += alpha * (r - Q1_A[1]);
            } else {
              Q1_A[1] += alpha * (r - Q1_A[1]);
            }
            s = TERMINAL;
          }
        } else {
          s = TERMINAL;
        }
      }

      // Record per-episode stats
      const frac = totalAVisits > 0 ? leftChoices / totalAVisits : 0.5;
      sumLeftFraction[ep] += frac;
      sumQALeft[ep] += Q1_A[0];
    }
  }

  return {
    leftFractionPerEpisode: Array.from(sumLeftFraction).map((v) => v / nSeeds),
    qALeftPerEpisode: Array.from(sumQALeft).map((v) => v / nSeeds),
  };
}

function argmax(arr: Float64Array): number {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

/**
 * Get the fraction of left-from-A choices at a specific episode number,
 * averaged over a per-episode rolling window.
 */
export function leftFractionAtEpisode(
  result: MaxBiasResult,
  episode: number,
  windowSize = 10,
): number {
  const start = Math.max(0, episode - windowSize);
  const end = Math.min(result.leftFractionPerEpisode.length, episode + 1);
  const slice = result.leftFractionPerEpisode.slice(start, end);
  return slice.reduce((a, b) => a + b, 0) / slice.length;
}

export { mulberry32 };
