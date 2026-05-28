#!/usr/bin/env python3
"""
Pre-compute Policy Gradient convergence traces for Lesson 10.
Outputs three JSON files to public/data/pg/:
  pg_training_trace.json    — single REINFORCE run (for V4 scrubber)
  pg_variance_comparison.json — gradient norm histograms + convergence bands (V5)
  pg_advantage_rmse.json    — n-step advantage RMSE at various n (V7)
"""

import json, math
from pathlib import Path

OUT = Path(__file__).parent.parent / "public" / "data" / "pg"
OUT.mkdir(parents=True, exist_ok=True)

# ── Mulberry32 PRNG (matches TypeScript implementation) ─────────────────────

def mulberry32(seed: int):
    s = seed & 0xFFFFFFFF
    def rng():
        nonlocal s
        s = (s + 0x6d2b79f5) & 0xFFFFFFFF
        t = ((s ^ (s >> 15)) * (1 | s)) & 0xFFFFFFFF
        t = (t + ((t ^ (t >> 7)) * (61 | t))) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return rng

# ── 3×3 gridworld (deterministic, γ=0.9) ────────────────────────────────────

GRID = 3
GAMMA = 0.9
PIT   = 4   # (1,1)
GOAL  = 8   # (2,2)
TERMINALS = {PIT, GOAL}
START = 0
NS, NA = GRID * GRID, 4
DELTAS = [(-1,0),(0,1),(1,0),(0,-1)]  # Up Right Down Left

def _idx(r, c): return r * GRID + c
def _rc(s): return divmod(s, GRID)

def _move(r, c, dr, dc):
    nr, nc = r + dr, c + dc
    if 0 <= nr < GRID and 0 <= nc < GRID:
        return _idx(nr, nc)
    return _idx(r, c)

# P[s][a][sp] = transition probability, R[s][a] = expected reward
P = [[[0.0]*NS for _ in range(NA)] for _ in range(NS)]
R = [[0.0]*NA for _ in range(NS)]

enter_r = lambda sp: 1.0 if sp == GOAL else (-1.0 if sp == PIT else 0.0)
for s in range(NS):
    r0, c0 = _rc(s)
    for a in range(NA):
        if s in TERMINALS:
            P[s][a][s] = 1.0
        else:
            dr, dc = DELTAS[a]
            sp = _move(r0, c0, dr, dc)
            P[s][a][sp] = 1.0
            R[s][a] = enter_r(sp)

def sample_step(s, a, rng):
    u = rng()
    acc = 0.0
    for sp, p in enumerate(P[s][a]):
        acc += p
        if u <= acc:
            return sp, R[s][a], (sp in TERMINALS)
    return NS - 1, R[s][a], False

# ── Exact V^π via linear system: (I - γ P^π) V = R^π ───────────────────────

def policy_eval_exact(pi):
    """pi[s] = list of nA action probabilities. Returns V[s] list."""
    # Build R_pi and P_pi
    R_pi = [sum(pi[s][a] * R[s][a] for a in range(NA)) for s in range(NS)]
    P_pi = [[sum(pi[s][a] * P[s][a][sp] for a in range(NA)) for sp in range(NS)]
            for s in range(NS)]
    # Solve (I - γ P^π) V = R_pi by Gauss-Seidel iteration
    V = [0.0] * NS
    for _ in range(10000):
        V_new = [R_pi[s] + GAMMA * sum(P_pi[s][sp] * V[sp] for sp in range(NS))
                 for s in range(NS)]
        if max(abs(V_new[s] - V[s]) for s in range(NS)) < 1e-12:
            V = V_new
            break
        V = V_new
    return V

# ── Softmax policy ───────────────────────────────────────────────────────────

def softmax(theta_s):
    m = max(theta_s)
    exps = [math.exp(x - m) for x in theta_s]
    s = sum(exps)
    return [x / s for x in exps]

class SoftmaxPolicy:
    def __init__(self):
        self.theta = [[0.0]*NA for _ in range(NS)]

    def probs(self, s):
        return softmax(self.theta[s])

    def sample(self, s, rng):
        p = self.probs(s)
        u = rng()
        acc = 0.0
        for a, pa in enumerate(p):
            acc += pa
            if u <= acc:
                return a
        return NA - 1

    def score(self, s, a):
        p = self.probs(s)
        return [( 1.0 if ap == a else 0.0) - p[ap] for ap in range(NA)]

    def to_pi(self):
        return [self.probs(s) for s in range(NS)]

def compute_returns(rewards):
    T = len(rewards)
    Gs = [0.0] * T
    g = 0.0
    for t in range(T - 1, -1, -1):
        g = rewards[t] + GAMMA * g
        Gs[t] = g
    return Gs

# ── Estimate V(s=0) via 20 MC rollouts with a separate eval RNG ─────────────

def estimate_v(policy, ep_idx):
    rng = mulberry32(0xDEAD0000 + (ep_idx & 0xFFFF))
    total = 0.0
    for _ in range(20):
        s, G, t = 0, 0.0, 0
        while s not in TERMINALS and t < 200:
            a = policy.sample(s, rng)
            sp, r, _ = sample_step(s, a, rng)
            G += (GAMMA ** t) * r
            s, t = sp, t + 1
        total += G
    return total / 20

# ── REINFORCE ────────────────────────────────────────────────────────────────

def reinforce(n_episodes, alpha, rng, use_baseline=False, alpha_critic=0.1,
              max_steps=500, record_every=1):
    policy = SoftmaxPolicy()
    V_critic = [0.0] * NS  # TD critic for baseline
    history, grad_norms, ep_returns = [], [], []
    recorded = []

    for ep in range(n_episodes):
        states, actions, rewards = [], [], []
        s = START
        for _ in range(max_steps):
            if s in TERMINALS:
                break
            a = policy.sample(s, rng)
            sp, r, done = sample_step(s, a, rng)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            if use_baseline and alpha_critic > 0:
                v_next = 0.0 if done else V_critic[sp]
                V_critic[s] += alpha_critic * (r + GAMMA * v_next - V_critic[s])
            s = sp
            if done:
                break

        T = len(states)
        if T == 0:
            history.append(estimate_v(policy, ep))
            grad_norms.append(0.0)
            ep_returns.append(0.0)
            if ep % record_every == 0:
                recorded.append({
                    "ep": ep, "v": history[-1], "gradNorm": 0.0,
                    "probs_s0": policy.probs(0)
                })
            continue

        Gs = compute_returns(rewards)
        ep_returns.append(Gs[0])

        # Running-mean fallback (not used when alpha_critic > 0)
        running_mean = sum(ep_returns) / len(ep_returns)

        grad_norm_sq = 0.0
        for t in range(T):
            st, at = states[t], actions[t]
            b = 0.0
            if use_baseline:
                b = V_critic[st] if alpha_critic > 0 else running_mean
            advantage = Gs[t] - b
            gamma_t = GAMMA ** t
            sc = policy.score(st, at)
            for ap in range(NA):
                delta = alpha * gamma_t * advantage * sc[ap]
                policy.theta[st][ap] += delta
                grad_norm_sq += delta * delta

        gn = math.sqrt(grad_norm_sq)
        history.append(estimate_v(policy, ep))
        grad_norms.append(gn)

        if ep % record_every == 0:
            recorded.append({
                "ep": ep, "v": history[-1], "gradNorm": gn,
                "probs_s0": policy.probs(0)
            })

    return policy, history, grad_norms, ep_returns, recorded

# ── Actor-Critic ─────────────────────────────────────────────────────────────

def actor_critic(n_episodes, alpha_actor, alpha_critic, rng, max_steps=500):
    policy = SoftmaxPolicy()
    V_critic = [0.0] * NS
    history, grad_norms, ep_returns = [], [], []

    def ev(ep):
        rng2 = mulberry32(0xCAFE0000 + (ep & 0xFFFF))
        total = 0.0
        for _ in range(20):
            s, G, t = 0, 0.0, 0
            while s not in TERMINALS and t < 200:
                a = policy.sample(s, rng2)
                sp, r, _ = sample_step(s, a, rng2)
                G += (GAMMA ** t) * r
                s, t = sp, t + 1
            total += G
        return total / 20

    for ep in range(n_episodes):
        s = START
        ep_return, grad_norm_sq, t = 0.0, 0.0, 0
        while s not in TERMINALS and t < max_steps:
            a = policy.sample(s, rng)
            sp, r, done = sample_step(s, a, rng)
            v_next = 0.0 if done else V_critic[sp]
            delta = r + GAMMA * v_next - V_critic[s]
            V_critic[s] += alpha_critic * delta
            sc = policy.score(s, a)
            for ap in range(NA):
                g = alpha_actor * delta * sc[ap]
                policy.theta[s][ap] += g
                grad_norm_sq += g * g
            ep_return += (GAMMA ** t) * r
            s, t = sp, t + 1
            if done:
                break
        history.append(ev(ep))
        grad_norms.append(math.sqrt(grad_norm_sq))
        ep_returns.append(ep_return)

    return policy, history, grad_norms, ep_returns

# ════════════════════════════════════════════════════════════════════════════
# 1. Training trace (for V4 scrubber)
# ════════════════════════════════════════════════════════════════════════════
print("Computing training trace...")

N_TRACE = 1500
ALPHA_TRACE = 0.1
pol, hist, gnorms, epreturns, recorded = reinforce(
    N_TRACE, ALPHA_TRACE, mulberry32(0), record_every=5
)

trace_data = {
    "nEpisodes": N_TRACE,
    "alpha": ALPHA_TRACE,
    "gamma": GAMMA,
    "recordEvery": 5,
    "history": [round(x, 5) for x in hist],
    "gradNorms": [round(x, 5) for x in gnorms],
    "episodeReturns": [round(x, 4) for x in epreturns],
    "snapshots": [
        {
            "ep": r["ep"],
            "v": round(r["v"], 5),
            "gradNorm": round(r["gradNorm"], 5),
            "probsS0": [round(p, 4) for p in r["probs_s0"]],
        }
        for r in recorded
    ],
    "finalExactV": round(policy_eval_exact(pol.to_pi())[START], 5),
}

(OUT / "pg_training_trace.json").write_text(json.dumps(trace_data, indent=2))
print(f"  Done. Final exact V(0,0) = {trace_data['finalExactV']}")

# ════════════════════════════════════════════════════════════════════════════
# 2. Variance comparison (for V5)
# ════════════════════════════════════════════════════════════════════════════
print("Computing variance comparison...")

# 2a. Gradient norm histograms at the uniform policy (fixed, not updated)
N_HIST = 300
ALPHA_HIST = 0.1

def grad_norms_at_uniform(n_episodes, rng, use_baseline, alpha_critic=0.1):
    """Compute gradient norms at the uniform policy (theta stays zero)."""
    policy = SoftmaxPolicy()  # uniform (theta=0)

    # Build TD critic by separate pre-training if using baseline
    V_pre = [0.0] * NS
    if use_baseline and alpha_critic > 0:
        rng_pre = mulberry32(0x5EED)
        for _ in range(2000):
            s = START
            for __ in range(200):
                if s in TERMINALS: break
                a = policy.sample(s, rng_pre)
                sp, r, done = sample_step(s, a, rng_pre)
                v_next = 0.0 if done else V_pre[sp]
                V_pre[s] += alpha_critic * (r + GAMMA * v_next - V_pre[s])
                s = sp
                if done: break

    norms = []
    for _ in range(n_episodes):
        states, actions, rewards = [], [], []
        s = START
        for __ in range(200):
            if s in TERMINALS: break
            a = policy.sample(s, rng)
            sp, r, done = sample_step(s, a, rng)
            states.append(s); actions.append(a); rewards.append(r)
            s = sp
            if done: break

        T = len(states)
        if T == 0:
            norms.append(0.0)
            continue

        Gs = compute_returns(rewards)
        norm_sq = 0.0
        for t in range(T):
            st, at = states[t], actions[t]
            b = V_pre[st] if use_baseline else 0.0
            adv = Gs[t] - b
            gamma_t = GAMMA ** t
            sc = policy.score(st, at)
            for ap in range(NA):
                g = ALPHA_HIST * gamma_t * adv * sc[ap]
                norm_sq += g * g
        norms.append(math.sqrt(norm_sq))
    return norms

rng_hist = mulberry32(0xABCD)
vanilla_norms = grad_norms_at_uniform(N_HIST, rng_hist, use_baseline=False)
baseline_norms = grad_norms_at_uniform(N_HIST, rng_hist, use_baseline=True)

# 2b. Convergence bands: 10 seeds of vanilla vs baseline REINFORCE
N_BAND = 600
ALPHA_BAND = 0.1
N_SEEDS = 10

vanilla_curves, baseline_curves = [], []
for seed in range(N_SEEDS):
    _, h_van, _, _, _ = reinforce(N_BAND, ALPHA_BAND, mulberry32(seed), use_baseline=False)
    vanilla_curves.append([round(v, 5) for v in h_van])
    _, h_base, _, _, _ = reinforce(N_BAND, ALPHA_BAND, mulberry32(seed), use_baseline=True)
    baseline_curves.append([round(v, 5) for v in h_base])

variance_data = {
    "histNEpisodes": N_HIST,
    "alphaHist": ALPHA_HIST,
    "vanillaGradNorms": [round(x, 6) for x in vanilla_norms],
    "baselineGradNorms": [round(x, 6) for x in baseline_norms],
    "bandNEpisodes": N_BAND,
    "alphaband": ALPHA_BAND,
    "nSeeds": N_SEEDS,
    "vanillaCurves": vanilla_curves,
    "baselineCurves": baseline_curves,
}

(OUT / "pg_variance_comparison.json").write_text(json.dumps(variance_data, indent=2))
van_std = math.sqrt(sum((x-sum(vanilla_norms)/N_HIST)**2 for x in vanilla_norms)/N_HIST)
base_std = math.sqrt(sum((x-sum(baseline_norms)/N_HIST)**2 for x in baseline_norms)/N_HIST)
print(f"  Gradient norm std: vanilla={van_std:.4f}, baseline={base_std:.4f}, ratio={van_std/max(base_std,1e-9):.2f}x")

# ════════════════════════════════════════════════════════════════════════════
# 3. Advantage RMSE (for V7)
# ════════════════════════════════════════════════════════════════════════════
print("Computing advantage RMSE...")

# Train a near-optimal policy first
pol_opt, _, _, _, _ = reinforce(2000, 0.1, mulberry32(42), use_baseline=True, alpha_critic=0.1)
pi_opt = pol_opt.to_pi()

# Compute exact V^π and Q^π(0, RIGHT) via policy evaluation
V_true = policy_eval_exact(pi_opt)

# Q^π(s, a) = R(s,a) + γ Σ P(sp|s,a) V^π(sp)
Q_right = R[0][1] + GAMMA * sum(P[0][1][sp] * V_true[sp] for sp in range(NS))
true_adv = Q_right - V_true[0]
print(f"  True A^π(0, RIGHT): {true_adv:.4f}  (V^π(0)={V_true[0]:.4f}, Q^π(0,R)={Q_right:.4f})")

# Build TD critic for near-optimal policy
V_critic_opt = [0.0] * NS
rng_pre = mulberry32(0x7EED)
for _ in range(5000):
    s = START
    for __ in range(500):
        if s in TERMINALS: break
        a = pol_opt.sample(s, rng_pre)
        sp, r, done = sample_step(s, a, rng_pre)
        v_next = 0.0 if done else V_critic_opt[sp]
        V_critic_opt[s] += 0.1 * (r + GAMMA * v_next - V_critic_opt[s])
        s = sp
        if done: break

def n_step_advantage(rewards, state_seq, t, n):
    T = len(rewards)
    G = 0.0
    for k in range(n):
        if t + k >= T: break
        G += (GAMMA ** k) * rewards[t + k]
    boot_idx = t + n
    if boot_idx < len(state_seq) and state_seq[boot_idx] not in TERMINALS:
        G += (GAMMA ** n) * V_critic_opt[state_seq[boot_idx]]
    return G - V_critic_opt[state_seq[t]]

N_ADVEPISODES = 500
rng_adv = mulberry32(0x9EED)
N_VALUES = [1, 2, 3, 5, 8, 13, 21, 50, 200]

adv_results = []
for n in N_VALUES:
    estimates = []
    rng_adv2 = mulberry32(0x9EED + n)
    for _ in range(N_ADVEPISODES):
        # Force first action = RIGHT from s=0
        sp_first, r_first, done_first = sample_step(0, 1, rng_adv2)
        rewards = [r_first]
        state_seq = [0, sp_first]
        cur = sp_first
        for __ in range(n - 1):
            if cur in TERMINALS or done_first: break
            a = pol_opt.sample(cur, rng_adv2)
            sp, r, done = sample_step(cur, a, rng_adv2)
            rewards.append(r)
            state_seq.append(sp)
            cur = sp
            if done: break
        adv = n_step_advantage(rewards, state_seq, 0, n)
        estimates.append(adv)

    mean_est = sum(estimates) / len(estimates)
    bias = mean_est - true_adv
    variance = sum((e - mean_est)**2 for e in estimates) / len(estimates)
    rmse = math.sqrt(bias**2 + variance)
    adv_results.append({
        "n": n,
        "rmse": round(rmse, 6),
        "biasSq": round(bias**2, 6),
        "variance": round(variance, 6),
        "meanEstimate": round(mean_est, 5),
    })
    print(f"  n={n:3d}: rmse={rmse:.4f}  bias={bias:.4f}  var={variance:.4f}")

advantage_data = {
    "trueAdvantage": round(true_adv, 5),
    "trueV0": round(V_true[0], 5),
    "trueQ0Right": round(Q_right, 5),
    "nEpisodesPerN": N_ADVEPISODES,
    "results": adv_results,
}

(OUT / "pg_advantage_rmse.json").write_text(json.dumps(advantage_data, indent=2))
print("Done. All PG trace files written to public/data/pg/")
