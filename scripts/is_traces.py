#!/usr/bin/env python3
"""
Pre-computes IS lesson statistics for instant initial render.
Outputs three JSON files into public/data/is/.
Matches the TypeScript estimators exactly: perDecisionIS uses ρ_{0:t}.
"""

import json, math, random, os
from pathlib import Path

OUT = Path(__file__).parent.parent / "public" / "data" / "is"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Gaussian helpers
# ---------------------------------------------------------------------------

def normal_pdf(x, mu, sigma):
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2 * math.pi))

def box_muller(rng):
    """Generate two N(0,1) samples."""
    u1 = rng.random()
    u2 = rng.random()
    if u1 == 0:
        u1 = 1e-10
    r = math.sqrt(-2 * math.log(u1))
    theta = 2 * math.pi * u2
    return r * math.cos(theta), r * math.sin(theta)

def sample_normal(n, mu, sigma, rng):
    out = []
    while len(out) < n:
        a, b = box_muller(rng)
        out.append(mu + sigma * a)
        if len(out) < n:
            out.append(mu + sigma * b)
    return out[:n]

def ordinary_is(samples, f, w):
    if not samples:
        return 0.0
    return sum(w(x) * f(x) for x in samples) / len(samples)

def weighted_is(samples, f, w):
    num = den = 0.0
    for x in samples:
        wi = w(x)
        num += wi * f(x)
        den += wi
    return 0.0 if den == 0 else num / den

def eff_sample_size(weights):
    s = sum(weights)
    s2 = sum(wi**2 for wi in weights)
    return 0.0 if s2 == 0 else s*s / s2

# ---------------------------------------------------------------------------
# Gridworld
# ---------------------------------------------------------------------------
# 3×3 grid: idx(r,c) = 3r+c; pit=4 (1,1); goal=8 (2,2)
# Actions: UP=0, RIGHT=1, DOWN=2, LEFT=3
GRID = 3
PIT, GOAL = 4, 8
TERMINALS = {PIT, GOAL}
DELTAS = [(-1,0),(0,1),(1,0),(0,-1)]

def idx(r, c): return r * GRID + c
def rc(s): return s // GRID, s % GRID

def move(r, c, dr, dc):
    nr, nc = r+dr, c+dc
    if nr < 0 or nr >= GRID or nc < 0 or nc >= GRID:
        return r, c
    return nr, nc

# Optimal path from (0,0): RIGHT RIGHT DOWN DOWN → states 0,1,2,5,goal
OPT_ACTIONS = {0: 1, 1: 1, 2: 2, 5: 2}  # state → optimal action

def pi_target(s, a):
    return 1.0 if OPT_ACTIONS.get(s) == a else 0.0

def pi_behavior(s, a):
    return 0.25  # uniform over 4 actions

def reward(s, a):
    r, c = rc(s)
    dr, dc = DELTAS[a]
    nr, nc = move(r, c, dr, dc)
    sp = idx(nr, nc)
    return 1.0 if sp == GOAL else (-1.0 if sp == PIT else 0.0)

def rollout(s0, pi, gamma, max_steps, rng, dense_step_reward=0.0):
    """Sample a trajectory. Returns list of (s, a, r)."""
    steps = []
    s = s0
    for _ in range(max_steps):
        if s in TERMINALS:
            break
        probs = pi(s)
        u = rng.random()
        acc = 0.0
        a = 3
        for i, p in enumerate(probs):
            acc += p
            if u <= acc:
                a = i
                break
        r_sa = reward(s, a)
        r_sa += dense_step_reward  # dense variant: add small step cost
        r, c = rc(s)
        dr, dc = DELTAS[a]
        nr, nc = move(r, c, dr, dc)
        sp = idx(nr, nc)
        steps.append((s, a, r_sa))
        s = sp
    return steps

def uniform_policy(s):
    return [0.25, 0.25, 0.25, 0.25]

def traj_is_weight(steps):
    rho = 1.0
    for s, a, _ in steps:
        pb = pi_behavior(s, a)
        if pb == 0:
            return 0.0
        rho *= pi_target(s, a) / pb
        if rho == 0:
            return 0.0
    return rho

def traj_return(steps, gamma):
    return sum(gamma**t * r for t, (_, _, r) in enumerate(steps))

def per_decision_contribution(steps, gamma):
    """Uses ρ_{0:t} weighting (update ratio before adding reward)."""
    total = 0.0
    rho = 1.0
    for t, (s, a, r) in enumerate(steps):
        pb = pi_behavior(s, a)
        if pb == 0:
            return total
        rho *= pi_target(s, a) / pb
        if rho == 0:
            return total
        total += gamma**t * r * rho
    return total

# ---------------------------------------------------------------------------
# 1. Gaussian variance sweep
# ---------------------------------------------------------------------------

def gaussian_sweep(n_trials=50, N=1000, seed=42):
    sigma_qs = [0.3, 0.4, 0.5, 0.6, 0.707, 0.8, 1.0, 1.5, 2.0, 3.0]
    results = []
    rng = random.Random(seed)
    p = lambda x: normal_pdf(x, 0, 1)
    f = lambda x: x * x
    true_val = 1.0

    for sq in sigma_qs:
        q = lambda x, s=sq: normal_pdf(x, 0, s)
        w = lambda x, s=sq: normal_pdf(x, 0, 1) / normal_pdf(x, 0, s)
        ord_ests, wt_ests, ess_vals = [], [], []
        for _ in range(n_trials):
            samples = sample_normal(N, 0, sq, rng)
            weights = [w(x) for x in samples]
            ord_ests.append(ordinary_is(samples, f, w))
            wt_ests.append(weighted_is(samples, f, w))
            ess_vals.append(eff_sample_size(weights) / N)

        def stats(vals):
            m = sum(vals) / len(vals)
            sd = math.sqrt(sum((v-m)**2 for v in vals) / len(vals))
            return round(m, 6), round(sd, 6)

        om, osd = stats(ord_ests)
        wm, wsd = stats(wt_ests)
        em, esd = stats(ess_vals)
        results.append({
            "sigmaQ": round(sq, 4),
            "finiteVariance": sq > 1/math.sqrt(2),
            "ordinaryMean": om, "ordinarySD": osd,
            "weightedMean": wm, "weightedSD": wsd,
            "essRatioMean": em, "essRatioSD": esd,
            "trueVal": true_val,
        })
    return results

# ---------------------------------------------------------------------------
# 2. Gridworld IS statistics
# ---------------------------------------------------------------------------

def gridworld_stats(n_trials=50, Ns=(100, 1000, 10000), gamma=0.9, seed=123):
    rng = random.Random(seed)
    true_val = gamma**3  # 0.729

    table = []
    for N in Ns:
        ord_ests, wt_ests, nz_counts = [], [], []
        # Also collect boxplot arrays
        for _ in range(n_trials):
            weights, g0s = [], []
            nz = 0
            for _ in range(N):
                steps = rollout(0, uniform_policy, gamma, 20, rng)
                w = traj_is_weight(steps)
                g = traj_return(steps, gamma)
                weights.append(w)
                g0s.append(g)
                if w > 0:
                    nz += 1

            total_w = sum(w * g for w, g in zip(weights, g0s))
            ord_est = total_w / N
            sum_w = sum(weights)
            wt_est = (sum(w * g for w, g in zip(weights, g0s)) / sum_w) if sum_w > 0 else 0.0
            ord_ests.append(round(ord_est, 6))
            wt_ests.append(round(wt_est, 6))
            nz_counts.append(nz)

        def stats(vals):
            m = sum(vals) / len(vals)
            sd = math.sqrt(sum((v-m)**2 for v in vals) / len(vals))
            return round(m, 6), round(sd, 6)

        om, osd = stats(ord_ests)
        wm, wsd = stats(wt_ests)
        nzm = round(sum(nz_counts) / len(nz_counts), 2)

        table.append({
            "N": N,
            "nonZeroMean": nzm,
            "ordinaryMean": om, "ordinarySD": osd,
            "weightedMean": wm, "weightedSD": wsd,
            "boxplotOrdinary": ord_ests,
            "boxplotWeighted": wt_ests,
            "trueVal": round(true_val, 6),
        })

    return {"gamma": gamma, "table": table}

# ---------------------------------------------------------------------------
# 3. Per-decision comparison
# ---------------------------------------------------------------------------

def dense_true_value(gamma, step_r=-0.01, path_len=4):
    """True V^π_t(s0) for deterministic optimal 4-step path with dense step reward."""
    # All T=4 non-terminal transitions earn step_r, terminal earns (1 + step_r)
    return sum(gamma**t * step_r for t in range(path_len - 1)) + gamma**(path_len-1) * (1 + step_r)

def per_decision_comparison(n_trials=50, Ns=(1000, 10000), gamma=0.9, seed=456):
    rng = random.Random(seed)

    results = []
    for variant in ("sparse", "dense"):
        step_r = 0.0 if variant == "sparse" else -0.01
        true_val = gamma**3 if step_r == 0 else dense_true_value(gamma, step_r)
        variant_results = []
        for N in Ns:
            ord_ests, pd_ests = [], []
            for _ in range(n_trials):
                ord_sum = 0.0
                pd_sum = 0.0
                for _ in range(N):
                    steps = rollout(0, uniform_policy, gamma, 20, rng,
                                    dense_step_reward=step_r)
                    w = traj_is_weight(steps)
                    g = traj_return(steps, gamma)
                    ord_sum += w * g
                    pd_sum += per_decision_contribution(steps, gamma)
                ord_ests.append(round(ord_sum / N, 6))
                pd_ests.append(round(pd_sum / N, 6))

            def sd(vals):
                m = sum(vals) / len(vals)
                return round(math.sqrt(sum((v-m)**2 for v in vals) / len(vals)), 6)

            variant_results.append({
                "N": N,
                "ordinarySD": sd(ord_ests),
                "perDecisionSD": sd(pd_ests),
                "ordinaryMean": round(sum(ord_ests)/len(ord_ests), 6),
                "perDecisionMean": round(sum(pd_ests)/len(pd_ests), 6),
            })
        results.append({
            "variant": variant,
            "stepReward": step_r,
            "trueVal": round(true_val, 6),
            "data": variant_results,
        })
        true_val = 0  # reset (unused after loop)
    return results

# ---------------------------------------------------------------------------
# Run and write
# ---------------------------------------------------------------------------

print("Computing Gaussian variance sweep...")
sweep = gaussian_sweep()
(OUT / "gaussian_variance_sweep.json").write_text(
    json.dumps(sweep, indent=2))
print(f"  Written {len(sweep)} sigma_q entries")

print("Computing gridworld IS statistics...")
gw = gridworld_stats()
(OUT / "gridworld_is_stats.json").write_text(
    json.dumps(gw, indent=2))
for row in gw["table"]:
    print(f"  N={row['N']:6d}: nonZero={row['nonZeroMean']:.1f}, "
          f"ord={row['ordinaryMean']:.4f}±{row['ordinarySD']:.4f}, "
          f"wt={row['weightedMean']:.4f}±{row['weightedSD']:.4f}")

print("Computing per-decision comparison...")
pd_cmp = per_decision_comparison()
(OUT / "per_decision_comparison.json").write_text(
    json.dumps(pd_cmp, indent=2))
for v in pd_cmp:
    print(f"  {v['variant']}: ", end="")
    for d in v['data']:
        print(f"N={d['N']} ord_sd={d['ordinarySD']:.4f} pd_sd={d['perDecisionSD']:.4f}  ", end="")
    print()

print("Done. JSON files written to", OUT)
