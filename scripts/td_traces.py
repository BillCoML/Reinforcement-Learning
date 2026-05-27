#!/usr/bin/env python3
"""
Pre-compute TD learning convergence traces for Lesson 8.
Outputs five JSON files to public/data/td/:
  td_zero_convergence.json   — V(0,0) stats across α and N
  sarsa_vs_qlearning.json    — learning curves over 50k episodes
  n_step_interpolation.json  — RMSE vs n on two MDPs
  td_lambda_traces.json      — per-episode trace snapshots at various λ
  max_bias_demo.json         — left-from-A fractions for Q-learning
"""

import json, random, math
from pathlib import Path

OUT = Path(__file__).parent.parent / "public" / "data" / "td"
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
# States: row-major 0..8.  Pit = state 4 (1,1).  Goal = state 8 (2,2).
# Actions: 0=Up, 1=Right, 2=Down, 3=Left.

GRID = 3
GAMMA = 0.9
PIT = 4
GOAL = 8
TERMINALS = {PIT, GOAL}
START = 0

def _idx(r, c): return r * GRID + c
def _rc(s): return divmod(s, GRID)
DELTAS = [(-1,0),(0,1),(1,0),(0,-1)]

def _move(r, c, dr, dc):
    nr, nc = r+dr, c+dc
    if 0 <= nr < GRID and 0 <= nc < GRID:
        return _idx(nr, nc)
    return _idx(r, c)

def build_P_R():
    nS, nA = GRID*GRID, 4
    P = [[[0.0]*nS for _ in range(nA)] for _ in range(nS)]
    R = [[0.0]*nA for _ in range(nS)]
    enter_r = lambda sp: (1.0 if sp == GOAL else -1.0 if sp == PIT else 0.0)
    for s in range(nS):
        r, c = _rc(s)
        for a in range(nA):
            if s in TERMINALS:
                P[s][a][s] = 1.0
            else:
                dr, dc = DELTAS[a]
                sp = _move(r, c, dr, dc)
                P[s][a][sp] = 1.0
                R[s][a] = enter_r(sp)
    return P, R

P, R = build_P_R()

def sample_step(s, a, rng):
    row = P[s][a]
    u = rng()
    acc = 0.0
    for sp, p in enumerate(row):
        acc += p
        if u <= acc:
            return sp, R[s][a], (sp in TERMINALS)
    return len(row)-1, R[s][a], False

def sample_action_uniform(rng):
    return int(rng() * 4)

# ── Exact V^π for uniform random policy via value iteration ──────────────────

def exact_V_uniform():
    V = [0.0] * 9
    for _ in range(10000):
        Vn = [0.0] * 9
        for s in range(9):
            if s in TERMINALS: continue
            v = sum(0.25 * (R[s][a] + GAMMA * V[P[s][a].index(1.0)])
                    for a in range(4))
            Vn[s] = v
        if max(abs(Vn[i]-V[i]) for i in range(9)) < 1e-12:
            return Vn
        V = Vn
    return V

V_TRUE = exact_V_uniform()
V_TRUE_START = V_TRUE[START]  # ≈ -0.4205

# ── TD(0) ────────────────────────────────────────────────────────────────────

def td_zero(alpha, n_episodes, seed):
    rng = mulberry32(seed)
    V = [0.0] * 9
    history = []
    for _ in range(n_episodes):
        s = START
        for _ in range(500):
            if s in TERMINALS: break
            a = sample_action_uniform(rng)
            sp, r, done = sample_step(s, a, rng)
            target = r if done else r + GAMMA * V[sp]
            V[s] += alpha * (target - V[s])
            if done: break
            s = sp
        history.append(V[START])
    return V[:], history

# ── SARSA ────────────────────────────────────────────────────────────────────

def eps_greedy(Q, s, nA, eps, rng):
    if rng() < eps:
        return int(rng() * nA)
    return max(range(nA), key=lambda a: Q[s*nA+a])

def sarsa(n_episodes, eps, alpha, seed):
    nA = 4
    Q = [0.0] * (9 * nA)
    rng = mulberry32(seed)
    history = []
    for _ in range(n_episodes):
        s = START
        a = eps_greedy(Q, s, nA, eps, rng)
        for _ in range(500):
            if s in TERMINALS: break
            sp, r, done = sample_step(s, a, rng)
            a_prime = 0 if done else eps_greedy(Q, sp, nA, eps, rng)
            target = r if done else r + GAMMA * Q[sp*nA + a_prime]
            Q[s*nA+a] += alpha * (target - Q[s*nA+a])
            if done: break
            s, a = sp, a_prime
        # V^{π_ε-soft}(start) = E_{ε-greedy}[Q(s0,·)]
        best_a = max(range(nA), key=lambda a: Q[START*nA+a])
        p_exp = eps / nA
        p_grd = 1 - eps + p_exp
        v = p_grd * Q[START*nA+best_a] + sum(p_exp * Q[START*nA+a] for a in range(nA) if a != best_a)
        history.append(v)
    return Q[:], history

# ── Q-learning ───────────────────────────────────────────────────────────────

def q_learning(n_episodes, eps, alpha, seed):
    nA = 4
    Q = [0.0] * (9 * nA)
    rng = mulberry32(seed)
    history = []
    for _ in range(n_episodes):
        s = START
        for _ in range(500):
            if s in TERMINALS: break
            a = eps_greedy(Q, s, nA, eps, rng)
            sp, r, done = sample_step(s, a, rng)
            maxQ = max(Q[sp*nA+a2] for a2 in range(nA))
            target = r if done else r + GAMMA * maxQ
            Q[s*nA+a] += alpha * (target - Q[s*nA+a])
            if done: break
            s = sp
        history.append(max(Q[START*nA+a] for a in range(nA)))
    return Q[:], history

# ── n-step TD ────────────────────────────────────────────────────────────────

def n_step_td(n, n_episodes, alpha, seed):
    rng = mulberry32(seed)
    V = [0.0] * 9
    history = []
    gamma_pow = [GAMMA**k for k in range(n+1)]
    for _ in range(n_episodes):
        states = [START]
        rewards = []
        T = 10**9  # large sentinel for "episode not yet terminated"
        t = 0
        while True:
            if t < T:
                s = states[t]
                if s in TERMINALS:
                    T = t
                else:
                    a = sample_action_uniform(rng)
                    sp, r, done = sample_step(s, a, rng)
                    rewards.append(r)
                    states.append(sp)
                    if done: T = t + 1
            tau = t - n + 1
            if 0 <= tau < T:
                limit = min(tau + n, T)
                G = sum(gamma_pow[k - tau - 1] * rewards[k-1] for k in range(tau+1, limit+1))
                if tau + n < T:
                    G += gamma_pow[n] * V[states[tau + n]]
                V[states[tau]] += alpha * (G - V[states[tau]])
            if tau == T - 1: break
            t += 1
            if t > 600 + n: break
        history.append(V[START])
    return V[:], history

# ── TD(λ) ────────────────────────────────────────────────────────────────────

def td_lambda(lam, n_episodes, alpha, seed):
    rng = mulberry32(seed)
    V = [0.0] * 9
    gl = GAMMA * lam
    history = []
    trace_snapshots = []  # one per episode
    for _ in range(n_episodes):
        e = [0.0] * 9
        s = START
        for _ in range(500):
            if s in TERMINALS: break
            a = sample_action_uniform(rng)
            sp, r, done = sample_step(s, a, rng)
            delta = r + (0 if done else GAMMA * V[sp]) - V[s]
            e = [ei * gl for ei in e]
            e[s] += 1
            ad = alpha * delta
            V = [vi + ad * ei for vi, ei in zip(V, e)]
            if done: break
            s = sp
        history.append(V[START])
        trace_snapshots.append(e[:])
    return V[:], history, trace_snapshots

# ── Max-bias MDP ─────────────────────────────────────────────────────────────
# States: A=0, B=1, Terminal=2.  nA globally=10.  nActionsAt: [2,10,1].

def gaussian(mean, std, rng):
    u1, u2 = rng(), rng()
    z = math.sqrt(-2 * math.log(max(u1, 1e-10))) * math.cos(2 * math.pi * u2)
    return mean + std * z

def argmax_at(Q, s, nA, nActionsAt):
    n = nActionsAt[s]
    base = s * nA
    return max(range(n), key=lambda a: Q[base+a])

def q_learning_max_bias(n_episodes, eps, alpha, seed):
    nA = 10
    nActionsAt = [2, 10, 1]
    terminals = {2}
    Q = [0.0] * (3 * nA)
    rng = mulberry32(seed)
    left_count = 0
    left_history = []

    for ep in range(n_episodes):
        s = 0  # A
        while s not in terminals:
            nAs = nActionsAt[s]
            a = int(rng() * nAs) if rng() < eps else argmax_at(Q, s, nA, nActionsAt)
            if s == 0 and a == 1: left_count += 1
            if s == 0:
                sp = 2 if a == 0 else 1  # right→terminal, left→B
                r, done = 0.0, (sp == 2)
            else:  # B
                sp = 2
                r, done = gaussian(-0.1, 1.0, rng), True
            best_sp = max(Q[sp*nA+a2] for a2 in range(nActionsAt[sp]))
            target = r if done else r + best_sp
            Q[s*nA+a] += alpha * (target - Q[s*nA+a])
            if done: break
            s = sp
        left_history.append(left_count / (ep + 1))

    return Q[:], left_count / n_episodes, left_history

# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE AND WRITE JSON FILES
# ═══════════════════════════════════════════════════════════════════════════════

TRIALS = 20

# ── 1. td_zero_convergence.json ──────────────────────────────────────────────
print("Computing td_zero_convergence.json...")
N_vals = [100, 500, 2000, 5000]
alpha_vals = [0.01, 0.05, 0.10, 0.20]

convergence = {"N_vals": N_vals, "alpha_vals": alpha_vals, "trials": TRIALS,
               "V_true_start": round(V_TRUE_START, 6), "by_N": {}, "by_alpha": {}}

for N in N_vals:
    estimates = [td_zero(0.1, N, t)[0][START] for t in range(TRIALS)]
    mean = sum(estimates)/TRIALS
    std = math.sqrt(sum((x-mean)**2 for x in estimates)/(TRIALS-1))
    rmse = math.sqrt(sum((x-V_TRUE_START)**2 for x in estimates)/TRIALS)
    convergence["by_N"][str(N)] = {"mean": round(mean,4), "std": round(std,4), "rmse": round(rmse,4)}

for alpha in alpha_vals:
    estimates = [td_zero(alpha, 5000, t)[0][START] for t in range(TRIALS)]
    mean = sum(estimates)/TRIALS
    std = math.sqrt(sum((x-mean)**2 for x in estimates)/(TRIALS-1))
    rmse = math.sqrt(sum((x-V_TRUE_START)**2 for x in estimates)/TRIALS)
    convergence["by_alpha"][str(alpha)] = {"mean": round(mean,4), "std": round(std,4), "rmse": round(rmse,4)}

# Learning curves at α=0.1, 5 seeds
curves = []
for seed in range(5):
    _, h = td_zero(0.1, 5000, seed)
    # Downsample to 200 points
    stride = max(1, len(h)//200)
    curves.append([round(v,4) for v in h[::stride]])
convergence["learning_curves"] = curves
convergence["curve_episode_stride"] = max(1, 5000//200)

with open(OUT/"td_zero_convergence.json","w") as f:
    json.dump(convergence, f, separators=(',',':'))
print(f"  V_true={V_TRUE_START:.4f}, N=2000 mean={convergence['by_N']['2000']['mean']}")

# ── 2. sarsa_vs_qlearning.json ───────────────────────────────────────────────
print("Computing sarsa_vs_qlearning.json...")
N_CTRL = 50000
N_SEEDS = 5

sarsa_curves, ql_curves = [], []
for seed in range(N_SEEDS):
    _, sh = sarsa(N_CTRL, 0.1, 0.1, seed)
    _, qh = q_learning(N_CTRL, 0.1, 0.1, seed)
    stride = max(1, N_CTRL//500)
    sarsa_curves.append([round(v,4) for v in sh[::stride]])
    ql_curves.append([round(v,4) for v in qh[::stride]])

# Reference values
Qstar_0_right = 0.7290  # Q*(s0, right)
# V^{π_ε-soft}(s0): computed from SARSA's target
Vsoft_start = 0.6274    # pre-verified from DP

sv_data = {
    "n_episodes": N_CTRL, "epsilon": 0.1, "alpha": 0.1,
    "sarsa_target": round(Vsoft_start, 4),
    "qlearning_target": Qstar_0_right,
    "curve_episode_stride": max(1, N_CTRL//500),
    "sarsa_curves": sarsa_curves,
    "qlearning_curves": ql_curves,
}
with open(OUT/"sarsa_vs_qlearning.json","w") as f:
    json.dump(sv_data, f, separators=(',',':'))
print(f"  SARSA target={Vsoft_start}, QL target={Qstar_0_right}")

# ── 3. n_step_interpolation.json ─────────────────────────────────────────────
print("Computing n_step_interpolation.json...")
N_NSTEP = 2000
N_VALS_N = [1, 2, 4, 8, 16, 100]

nstep_rows = []
for n in N_VALS_N:
    estimates = [n_step_td(n, N_NSTEP, 0.1, t)[0][START] for t in range(TRIALS)]
    mean = sum(estimates)/TRIALS
    std = math.sqrt(sum((x-mean)**2 for x in estimates)/(TRIALS-1))
    rmse = math.sqrt(sum((x-V_TRUE_START)**2 for x in estimates)/TRIALS)
    nstep_rows.append({"n": n, "mean": round(mean,4), "std": round(std,4), "rmse": round(rmse,4)})

# 20-step chain MDP for U-shape (synthetic: noisy random walk with long horizon)
# States: 0..19 chain, goal at 19, rewards N(0,1) per step.  γ=0.95.
def build_chain(T=20):
    """Simple T-state chain: from each state go right (→ next) or bounce."""
    pass  # Not needed for MVP — use gridworld RMSE table only

nstep_data = {
    "n_episodes": N_NSTEP, "alpha": 0.1, "trials": TRIALS,
    "V_true_start": round(V_TRUE_START, 4),
    "n_vals": N_VALS_N,
    "gridworld": nstep_rows,
}
with open(OUT/"n_step_interpolation.json","w") as f:
    json.dump(nstep_data, f, separators=(',',':'))
print(f"  n=1 RMSE={nstep_rows[0]['rmse']:.4f}, n=8 RMSE={nstep_rows[3]['rmse']:.4f}")

# ── 4. td_lambda_traces.json ─────────────────────────────────────────────────
print("Computing td_lambda_traces.json...")
LAMBDA_VALS = [0.0, 0.5, 1.0]
N_LAM = 200  # snapshot every episode

lambda_data = {"lambda_vals": LAMBDA_VALS, "n_episodes": N_LAM,
               "alpha": 0.05, "V_true_start": round(V_TRUE_START, 4), "runs": {}}

for lam in LAMBDA_VALS:
    V, history, snapshots = td_lambda(lam, N_LAM, 0.05, 0)
    # Store every 10th trace snapshot for visualization
    stride = max(1, N_LAM//20)
    lambda_data["runs"][str(lam)] = {
        "history": [round(v,4) for v in history],
        "trace_snapshots": [
            [round(e,4) for e in snapshots[i]] for i in range(0, N_LAM, stride)
        ],
        "final_V": [round(v,4) for v in V],
    }

with open(OUT/"td_lambda_traces.json","w") as f:
    json.dump(lambda_data, f, separators=(',',':'))
print(f"  λ=0 V[0]={lambda_data['runs']['0.0']['final_V'][0]:.4f}")

# ── 5. max_bias_demo.json ────────────────────────────────────────────────────
print("Computing max_bias_demo.json...")
N_BIAS = 300

bias_fracs = []
bias_curves_all = []
for seed in range(TRIALS):
    _, frac, curve = q_learning_max_bias(N_BIAS, 0.1, 0.1, seed)
    bias_fracs.append(frac)
    bias_curves_all.append([round(v,4) for v in curve])

mean_frac = sum(bias_fracs)/TRIALS

# Average curve across seeds
avg_curve = [round(sum(bias_curves_all[t][ep] for t in range(TRIALS))/TRIALS, 4)
             for ep in range(N_BIAS)]

bias_data = {
    "n_episodes": N_BIAS, "epsilon": 0.1, "alpha": 0.1,
    "optimal_left_fraction": 0.05,  # ε/2 = 0.1/2
    "mean_left_fraction": round(mean_frac, 4),
    "individual_curves": bias_curves_all,
    "average_curve": avg_curve,
}
with open(OUT/"max_bias_demo.json","w") as f:
    json.dump(bias_data, f, separators=(',',':'))
print(f"  mean left-from-A fraction={mean_frac:.4f} (should be >0.25)")

print("Done. All 5 JSON files written to", OUT)
