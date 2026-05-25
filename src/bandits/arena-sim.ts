/**
 * Battle-Arena simulation engine. Runs many seeds of several algorithms over a
 * shared bandit instance, vectorized with Float64Array state and zero
 * object-allocation in the hot loop. The simulation advances a *time frontier*
 * (all seeds stepped together up to tCur) so the regret curve can be animated
 * left-to-right while computing, and a frame can stop mid-way to stay <16ms.
 */
import { mulberry32, sampleBeta, sampleNormal, type RNG } from "./stats";

export type RewardFamily = "bernoulli" | "gaussian" | "beta";
export type AlgoKind = "random" | "epsilon" | "ucb1" | "thompson";

export interface AlgoSpec {
  key: string; // unique id (e.g. "eps01")
  kind: AlgoKind;
  label: string;
  epsilon?: number;
}

export interface ArenaConfig {
  K: number;
  means: number[];
  family: RewardFamily;
  gaussianSigma: number;
  betaConcentration: number;
  T: number;
  seeds: number;
  baseSeed: number;
  algos: AlgoSpec[];
}

interface AlgoState {
  spec: AlgoSpec;
  N: Float64Array; // S*K pull counts
  sum: Float64Array; // S*K reward sums (for μ̂)
  succ: Float64Array; // S*K successes (Thompson α-1)
  regret: Float64Array; // S current cumulative regret
  meanSum: Float64Array; // T running Σ regret over seeds
  sqSum: Float64Array; // T running Σ regret²
  rngs: RNG[]; // per-seed stream
}

export interface AlgoResult {
  key: string;
  mean: number[]; // length tCur
  std: number[];
  pullFrac: number[]; // length K, fraction of pulls per arm (up to tCur)
  finalRegret: number;
  pctOptimal: number;
  regretOverLogT: number;
}

export class ArenaSim {
  readonly cfg: ArenaConfig;
  readonly gaps: number[];
  readonly optArm: number;
  readonly muStar: number;
  private states: AlgoState[] = [];
  private tCur = 0; // frontier (number of completed time steps)

  constructor(cfg: ArenaConfig) {
    this.cfg = cfg;
    this.muStar = Math.max(...cfg.means);
    this.optArm = cfg.means.indexOf(this.muStar);
    this.gaps = cfg.means.map((m) => this.muStar - m);

    const { seeds: S, K, T } = cfg;
    cfg.algos.forEach((spec, ai) => {
      const rngs: RNG[] = [];
      for (let s = 0; s < S; s++) {
        rngs.push(mulberry32((cfg.baseSeed * 1_000_003 + ai * 97_001 + s * 31 + 1) >>> 0));
      }
      this.states.push({
        spec,
        N: new Float64Array(S * K),
        sum: new Float64Array(S * K),
        succ: new Float64Array(S * K),
        regret: new Float64Array(S),
        meanSum: new Float64Array(T),
        sqSum: new Float64Array(T),
        rngs,
      });
    });
  }

  get frontier(): number {
    return this.tCur;
  }
  get done(): boolean {
    return this.tCur >= this.cfg.T;
  }

  /** One reward draw for seed `s`, arm `a`, clipped to [0,1] for the bounded-reward algorithms. */
  private pull(rng: RNG, a: number): number {
    const mu = this.cfg.means[a];
    switch (this.cfg.family) {
      case "bernoulli":
        return rng() < mu ? 1 : 0;
      case "gaussian": {
        const r = mu + this.cfg.gaussianSigma * sampleNormal(rng);
        return r < 0 ? 0 : r > 1 ? 1 : r;
      }
      case "beta": {
        const c = this.cfg.betaConcentration;
        return sampleBeta(Math.max(0.05, mu * c), Math.max(0.05, (1 - mu) * c), rng);
      }
    }
  }

  /** Advance the time frontier to at most `targetT`, but stop once `budgetMs` elapses. */
  advance(targetT: number, budgetMs = 12): void {
    const start = performance.now();
    const { K, seeds: S } = this.cfg;
    const gaps = this.gaps;
    const tEnd = Math.min(targetT, this.cfg.T);

    while (this.tCur < tEnd) {
      const t = this.tCur;
      for (const st of this.states) {
        const { N, sum, succ, regret, meanSum, sqSum, rngs, spec } = st;
        const kind = spec.kind;
        const eps = spec.epsilon ?? 0;
        let mSum = 0;
        let sSum = 0;
        for (let s = 0; s < S; s++) {
          const base = s * K;
          const rng = rngs[s];
          let arm: number;

          if (kind === "random") {
            arm = (rng() * K) | 0;
          } else if (t < K) {
            arm = t; // shared init: pull each arm once
          } else if (kind === "epsilon") {
            if (rng() < eps) {
              arm = (rng() * K) | 0;
            } else {
              // argmax μ̂
              let best = 0;
              let bestV = sum[base] / N[base];
              for (let i = 1; i < K; i++) {
                const v = sum[base + i] / N[base + i];
                if (v > bestV) {
                  bestV = v;
                  best = i;
                }
              }
              arm = best;
            }
          } else if (kind === "ucb1") {
            const logT = Math.log(t + 1);
            let best = 0;
            let bestV = sum[base] / N[base] + Math.sqrt((2 * logT) / N[base]);
            for (let i = 1; i < K; i++) {
              const ni = N[base + i];
              const v = sum[base + i] / ni + Math.sqrt((2 * logT) / ni);
              if (v > bestV) {
                bestV = v;
                best = i;
              }
            }
            arm = best;
          } else {
            // thompson: sample Beta(1+succ, 1+N-succ), argmax
            let best = 0;
            let bestV = sampleBeta(1 + succ[base], 1 + N[base] - succ[base], rng);
            for (let i = 1; i < K; i++) {
              const v = sampleBeta(1 + succ[base + i], 1 + N[base + i] - succ[base + i], rng);
              if (v > bestV) {
                bestV = v;
                best = i;
              }
            }
            arm = best;
          }

          const reward = this.pull(rng, arm);
          const idx = base + arm;
          N[idx] += 1;
          sum[idx] += reward;
          if (reward >= 0.5) succ[idx] += 1;

          regret[s] += gaps[arm];
          mSum += regret[s];
          sSum += regret[s] * regret[s];
        }
        meanSum[t] = mSum;
        sqSum[t] = sSum;
      }
      this.tCur++;

      // check time budget every 16 steps
      if ((this.tCur & 15) === 0 && performance.now() - start > budgetMs) break;
    }
  }

  /** Snapshot results up to the current frontier for rendering. */
  results(): AlgoResult[] {
    const { seeds: S, K } = this.cfg;
    const tc = this.tCur;
    return this.states.map((st) => {
      const mean = new Array(tc);
      const std = new Array(tc);
      for (let t = 0; t < tc; t++) {
        const m = st.meanSum[t] / S;
        const v = Math.max(0, st.sqSum[t] / S - m * m);
        mean[t] = m;
        std[t] = Math.sqrt(v);
      }
      // pull distribution from N (summed over seeds)
      const pull = new Array(K).fill(0);
      let total = 0;
      for (let s = 0; s < S; s++) {
        for (let i = 0; i < K; i++) {
          pull[i] += st.N[s * K + i];
          total += st.N[s * K + i];
        }
      }
      const pullFrac = pull.map((p) => (total ? p / total : 0));
      const finalRegret = tc ? mean[tc - 1] : 0;
      return {
        key: st.spec.key,
        mean,
        std,
        pullFrac,
        finalRegret,
        pctOptimal: pullFrac[this.optArm] * 100,
        regretOverLogT: tc > 1 ? finalRegret / Math.log(tc) : 0,
      };
    });
  }
}
