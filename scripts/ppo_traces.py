#!/usr/bin/env python3
"""
Pre-compute PPO convergence traces for Lesson 11 (TRPO and PPO).
Outputs four JSON files to public/data/ppo/:
  headline_runs.json   — 10-seed PPO + vanilla PG + aggressive regime traces
  gae_lambda.json      — 20-seed GAE λ sweep (batch=5, noisy regime)
  lr_sensitivity.json  — 20-seed lr sweep for PPO and vanilla PG
  clip_sweep.json      — 5-seed clip ε sweep at lr=1.5, batch=10

Usage:
  python3 scripts/ppo_traces.py          # generate all JSON files
  python3 scripts/ppo_traces.py --verify # re-compute and print headline anchors
"""

import json, math, sys
from pathlib import Path

OUT = Path(__file__).parent.parent / "public" / "data" / "ppo"
OUT.mkdir(parents=True, exist_ok=True)

# ── Mulberry32 PRNG (matches TypeScript) ─────────────────────────────────────

def mulberry32(seed: int):
    s = seed & 0xFFFFFFFF
    def rng():
        nonlocal s
        s = (s + 0x6d2b79f5) & 0xFFFFFFFF
        t = ((s ^ (s >> 15)) * (1 | s)) & 0xFFFFFFFF
        t = (t + ((t ^ (t >> 7)) * (61 | t))) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return rng

# ── 3×3 gridworld (deterministic, γ=0.9) ─────────────────────────────────────

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
    for sp, prob in enumerate(P[s][a]):
        acc += prob
        if u <= acc:
            return sp, R[s][a], (sp in TERMINALS)
    return NS - 1, R[s][a], False

# ── Exact V^π via iterative policy evaluation ─────────────────────────────────

def policy_eval_exact(pi):
    """pi[s][a] = probability. Returns V[s] list."""
    R_pi = [sum(pi[s][a] * R[s][a] for a in range(NA)) for s in range(NS)]
    P_pi = [[sum(pi[s][a] * P[s][a][sp] for a in range(NA)) for sp in range(NS)]
            for s in range(NS)]
    V = [0.0] * NS
    for _ in range(10000):
        V_new = [R_pi[s] + GAMMA * sum(P_pi[s][sp] * V[sp] for sp in range(NS))
                 for s in range(NS)]
        if max(abs(V_new[s] - V[s]) for s in range(NS)) < 1e-12:
            V = V_new
            break
        V = V_new
    return V

# ── Softmax helpers ───────────────────────────────────────────────────────────

def softmax(theta_s):
    m = max(theta_s)
    exps = [math.exp(x - m) for x in theta_s]
    s = sum(exps)
    return [x / s for x in exps]

def sample_action(probs, rng):
    u = rng()
    acc = 0.0
    for a, p in enumerate(probs):
        acc += p
        if u <= acc:
            return a
    return NA - 1

# ── Rollout ───────────────────────────────────────────────────────────────────

def rollout(theta, rng, max_steps=500):
    states, actions, rewards = [], [], []
    s = START
    steps = 0
    while s not in TERMINALS and steps < max_steps:
        probs = softmax(theta[s])
        a = sample_action(probs, rng)
        sp, r, done = sample_step(s, a, rng)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s = sp
        steps += 1
        if s in TERMINALS:
            states.append(s)
            break
    return states, actions, rewards

# ── GAE ───────────────────────────────────────────────────────────────────────

def gae_advantages(states, actions, rewards, V, gamma, lam):
    T = len(rewards)
    adv = [0.0] * T
    last_gae = 0.0
    for t in range(T - 1, -1, -1):
        s = states[t]
        s_next = states[t + 1] if t + 1 < len(states) else -1
        v_next = V[s_next] if s_next >= 0 and s_next not in TERMINALS else 0.0
        delta = rewards[t] + gamma * v_next - V[s]
        last_gae = delta + gamma * lam * last_gae
        adv[t] = last_gae
    return adv

# ── PPO update ────────────────────────────────────────────────────────────────

def ppo_update(theta, V, batch, lr_policy, lr_value, clip_eps, gae_lambda,
               epochs, normalize_adv=True, entropy_coef=0.0, value_coef=0.5):
    theta = [row[:] for row in theta]
    V = V[:]
    theta_old = [row[:] for row in theta]

    all_states, all_actions, all_adv, all_returns = [], [], [], []
    visited = set()

    for states, actions, rewards in batch:
        adv = gae_advantages(states, actions, rewards, V, GAMMA, gae_lambda)
        T = len(rewards)
        G = 0.0
        returns = [0.0] * T
        for t in range(T - 1, -1, -1):
            G = rewards[t] + GAMMA * G
            returns[t] = G
        for t in range(T):
            all_states.append(states[t])
            all_actions.append(actions[t])
            all_adv.append(adv[t])
            all_returns.append(returns[t])
            visited.add(states[t])

    N = len(all_states)
    if N == 0:
        return theta, V, {}, 0.0

    if normalize_adv and N > 1:
        # Clamp before normalization to prevent float overflow at high lr.
        all_adv = [max(-1e6, min(1e6, x)) for x in all_adv]
        mean = sum(all_adv) / N
        variance = sum((x - mean)**2 for x in all_adv) / N
        std = math.sqrt(variance) + 1e-8
        all_adv = [(x - mean) / std for x in all_adv]

    last_clip_count = 0
    last_surrogate = 0.0
    last_ratios = [1.0] * N

    for epoch in range(epochs):
        clip_count = 0
        surrogate = 0.0
        ratios = [0.0] * N
        grad = [[0.0]*NA for _ in range(NS)]

        for i in range(N):
            s = all_states[i]
            a = all_actions[i]
            A_hat = all_adv[i]

            p_new = softmax(theta[s])
            p_old = softmax(theta_old[s])

            ratio = p_new[a] / (p_old[a] + 1e-12)
            ratios[i] = ratio

            r_clip = max(1 - clip_eps, min(1 + clip_eps, ratio))
            obj1 = ratio * A_hat
            obj2 = r_clip * A_hat
            min_obj = min(obj1, obj2)
            surrogate += min_obj

            is_clipped = (A_hat > 0 and ratio > 1 + clip_eps) or \
                         (A_hat < 0 and ratio < 1 - clip_eps)
            if is_clipped:
                clip_count += 1
                continue

            for ap in range(NA):
                grad[s][ap] += A_hat * ((1.0 if ap == a else 0.0) - p_new[ap])

        for s in range(NS):
            for ap in range(NA):
                theta[s][ap] += lr_policy * grad[s][ap] / N

        last_clip_count = clip_count
        last_surrogate = surrogate / N
        last_ratios = ratios

    # Critic update.
    for i in range(N):
        s = all_states[i]
        V[s] += lr_value * (all_returns[i] - V[s])
    for s in TERMINALS:
        V[s] = 0.0

    # Metrics.
    kl = 0.0
    for s in visited:
        p_old = softmax(theta_old[s])
        p_new = softmax(theta[s])
        for a in range(NA):
            if p_old[a] > 1e-12:
                kl += p_old[a] * math.log(p_old[a] / (p_new[a] + 1e-12))
    mean_kl = kl / len(visited) if visited else 0.0
    clip_frac = last_clip_count / N if N > 0 else 0.0

    pi = [[softmax(theta[s])[a] for a in range(NA)] for s in range(NS)]
    V_exact = policy_eval_exact(pi)
    v_start = V_exact[START]

    log = {
        "vStart": v_start,
        "meanKL": mean_kl,
        "clipFraction": clip_frac,
        "meanRatio": sum(last_ratios) / len(last_ratios) if last_ratios else 1.0,
        "maxRatio": max(last_ratios) if last_ratios else 1.0,
        "minRatio": min(last_ratios) if last_ratios else 1.0,
        "surrogateValue": last_surrogate,
        "batchSize": N,
    }
    return theta, V, log, clip_frac

# ── Vanilla PG update ─────────────────────────────────────────────────────────

def vanilla_pg_update(theta, V, batch, lr_policy, lr_value, gae_lambda,
                      normalize_adv=True):
    theta = [row[:] for row in theta]
    V = V[:]
    all_states, all_actions, all_adv, all_returns = [], [], [], []

    for states, actions, rewards in batch:
        adv = gae_advantages(states, actions, rewards, V, GAMMA, gae_lambda)
        T = len(rewards)
        G = 0.0
        returns = [0.0] * T
        for t in range(T - 1, -1, -1):
            G = rewards[t] + GAMMA * G
            returns[t] = G
        for t in range(T):
            all_states.append(states[t])
            all_actions.append(actions[t])
            all_adv.append(adv[t])
            all_returns.append(returns[t])

    N = len(all_states)
    if N == 0:
        return theta, V, {"vStart": -999, "batchSize": 0}

    if normalize_adv and N > 1:
        all_adv = [max(-1e6, min(1e6, x)) for x in all_adv]
        mean = sum(all_adv) / N
        variance = sum((x - mean)**2 for x in all_adv) / N
        std = math.sqrt(variance) + 1e-8
        all_adv = [(x - mean) / std for x in all_adv]

    grad = [[0.0]*NA for _ in range(NS)]
    for i in range(N):
        s = all_states[i]
        a = all_actions[i]
        A_hat = all_adv[i]
        p = softmax(theta[s])
        for ap in range(NA):
            grad[s][ap] += A_hat * ((1.0 if ap == a else 0.0) - p[ap])

    for s in range(NS):
        for ap in range(NA):
            theta[s][ap] += lr_policy * grad[s][ap] / N

    for i in range(N):
        V[all_states[i]] += lr_value * (all_returns[i] - V[all_states[i]])
    for s in TERMINALS:
        V[s] = 0.0

    pi = [[softmax(theta[s])[a] for a in range(NA)] for s in range(NS)]
    V_exact = policy_eval_exact(pi)
    return theta, V, {"vStart": V_exact[START], "batchSize": N}

# ── Run helpers ───────────────────────────────────────────────────────────────

def run_ppo(seed, n_iters, lr_policy, lr_value, clip_eps, gae_lambda,
            epochs, batch_episodes, normalize_adv=True, entropy_coef=0.0):
    rng = mulberry32(seed)
    theta = [[0.0]*NA for _ in range(NS)]
    V = [0.0] * NS
    logs = []
    for it in range(n_iters):
        batch = [rollout(theta, rng) for _ in range(batch_episodes)]
        theta, V, log, _ = ppo_update(
            theta, V, batch, lr_policy, lr_value, clip_eps, gae_lambda,
            epochs, normalize_adv, entropy_coef
        )
        if log:
            log["iter"] = it
            logs.append(log)
    return theta, V, logs

def run_vanilla(seed, n_iters, lr_policy, lr_value, gae_lambda,
                batch_episodes, normalize_adv=True):
    rng = mulberry32(seed)
    theta = [[0.0]*NA for _ in range(NS)]
    V = [0.0] * NS
    logs = []
    for it in range(n_iters):
        batch = [rollout(theta, rng) for _ in range(batch_episodes)]
        theta, V, log = vanilla_pg_update(
            theta, V, batch, lr_policy, lr_value, gae_lambda, normalize_adv
        )
        log["iter"] = it
        logs.append(log)
    return theta, V, logs

# ── headline_runs.json ────────────────────────────────────────────────────────

def gen_headline_runs():
    print("Generating headline_runs.json...")
    N_SEEDS = 10

    # PPO headline: lr=0.5, batch=20, 200 iters, 10 seeds
    ppo_runs = []
    for seed in range(N_SEEDS):
        _, _, logs = run_ppo(seed, 200, 0.5, 0.5, 0.2, 0.95, 4, 20)
        ppo_runs.append([l["vStart"] for l in logs])
    ppo_mean = [sum(ppo_runs[s][i] for s in range(N_SEEDS)) / N_SEEDS for i in range(200)]

    # Vanilla headline: same config
    van_runs = []
    for seed in range(N_SEEDS):
        _, _, logs = run_vanilla(seed, 200, 0.5, 0.5, 0.95, 20)
        van_runs.append([l["vStart"] for l in logs])
    van_mean = [sum(van_runs[s][i] for s in range(N_SEEDS)) / N_SEEDS for i in range(200)]

    # Aggressive trace: lr=2.0, batch=5, epochs=10, 5 seeds (for centerpiece)
    agg_runs = []
    for seed in range(5):
        _, _, logs = run_ppo(seed, 100, 2.0, 1.0, 0.2, 0.95, 10, 5)
        agg_logs = []
        for l in logs:
            agg_logs.append({
                "iter": l["iter"],
                "vStart": l["vStart"],
                "clipFraction": l["clipFraction"],
                "meanKL": l["meanKL"],
            })
        agg_runs.append(agg_logs)

    out = {
        "ppo_headline": {
            "config": {"lr": 0.5, "batch": 20, "epochs": 4, "clip_eps": 0.2, "gae_lambda": 0.95, "n_iters": 200, "n_seeds": N_SEEDS},
            "per_seed": ppo_runs,
            "mean": ppo_mean,
            "final_mean": ppo_mean[-1],
            "final_std": math.sqrt(sum((ppo_runs[s][-1] - ppo_mean[-1])**2 for s in range(N_SEEDS)) / N_SEEDS),
        },
        "vanilla_headline": {
            "config": {"lr": 0.5, "batch": 20, "epochs": 1, "gae_lambda": 0.95, "n_iters": 200, "n_seeds": N_SEEDS},
            "per_seed": van_runs,
            "mean": van_mean,
            "final_mean": van_mean[-1],
            "final_std": math.sqrt(sum((van_runs[s][-1] - van_mean[-1])**2 for s in range(N_SEEDS)) / N_SEEDS),
        },
        "aggressive_trace": {
            "config": {"lr": 2.0, "lr_value": 1.0, "batch": 5, "epochs": 10, "clip_eps": 0.2, "gae_lambda": 0.95, "n_iters": 100, "n_seeds": 5},
            "per_seed": agg_runs,
        },
    }

    with open(OUT / "headline_runs.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  PPO final: {out['ppo_headline']['final_mean']:.4f} ± {out['ppo_headline']['final_std']:.4f}")
    print(f"  Vanilla final: {out['vanilla_headline']['final_mean']:.4f} ± {out['vanilla_headline']['final_std']:.4f}")
    return out

# ── gae_lambda.json ───────────────────────────────────────────────────────────

def gen_gae_lambda():
    print("Generating gae_lambda.json...")
    N_SEEDS = 20
    LAMBDAS = [0.0, 0.5, 0.9, 0.95, 1.0]

    results = {}
    for lam in LAMBDAS:
        final_vs = []
        for seed in range(N_SEEDS):
            _, _, logs = run_ppo(seed, 200, 0.5, 0.5, 0.2, lam, 4, 5)
            final_vs.append(logs[-1]["vStart"])
        mean = sum(final_vs) / N_SEEDS
        std = math.sqrt(sum((v - mean)**2 for v in final_vs) / N_SEEDS)
        results[str(lam)] = {
            "lambda": lam,
            "per_seed_final": final_vs,
            "mean": mean,
            "std": std,
        }
        print(f"  λ={lam:.2f}: {mean:.4f} ± {std:.4f}")

    out = {"config": {"lr": 0.5, "batch": 5, "epochs": 4, "clip_eps": 0.2, "n_iters": 200, "n_seeds": N_SEEDS}, "results": results}
    with open(OUT / "gae_lambda.json", "w") as f:
        json.dump(out, f, indent=2)
    return out

# ── lr_sensitivity.json ───────────────────────────────────────────────────────

def gen_lr_sensitivity():
    print("Generating lr_sensitivity.json...")
    N_SEEDS = 20
    LRS = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]

    ppo_results = {}
    vanilla_results = {}

    for lr in LRS:
        lr_val = min(lr * 0.5, 0.5)  # cap critic lr to prevent divergence
        # PPO
        ppo_finals = []
        for seed in range(N_SEEDS):
            _, _, logs = run_ppo(seed, 200, lr, lr_val, 0.2, 0.95, 4, 10)
            ppo_finals.append(logs[-1]["vStart"])
        ppo_mean = sum(ppo_finals) / N_SEEDS
        ppo_std = math.sqrt(sum((v - ppo_mean)**2 for v in ppo_finals) / N_SEEDS)
        ppo_results[str(lr)] = {"lr": lr, "mean": ppo_mean, "std": ppo_std, "per_seed": ppo_finals}

        # Vanilla
        van_finals = []
        for seed in range(N_SEEDS):
            _, _, logs = run_vanilla(seed, 200, lr, lr_val, 0.95, 10)
            van_finals.append(logs[-1]["vStart"])
        van_mean = sum(van_finals) / N_SEEDS
        van_std = math.sqrt(sum((v - van_mean)**2 for v in van_finals) / N_SEEDS)
        vanilla_results[str(lr)] = {"lr": lr, "mean": van_mean, "std": van_std, "per_seed": van_finals}

        print(f"  lr={lr}: PPO {ppo_mean:.4f}±{ppo_std:.4f}  vanilla {van_mean:.4f}±{van_std:.4f}")

    out = {
        "config": {"batch": 10, "epochs": 4, "clip_eps": 0.2, "gae_lambda": 0.95, "n_iters": 200, "n_seeds": N_SEEDS},
        "ppo": ppo_results,
        "vanilla": vanilla_results,
    }
    with open(OUT / "lr_sensitivity.json", "w") as f:
        json.dump(out, f, indent=2)
    return out

# ── clip_sweep.json ───────────────────────────────────────────────────────────

def gen_clip_sweep():
    print("Generating clip_sweep.json...")
    N_SEEDS = 5
    EPS_VALUES = [0.05, 0.10, 0.20, 0.40]

    results = {}
    for eps in EPS_VALUES:
        finals = []
        late_clips = []
        for seed in range(N_SEEDS):
            rng = mulberry32(seed)
            theta = [[0.0]*NA for _ in range(NS)]
            V = [0.0] * NS
            last_clip = 0.0
            for it in range(200):
                batch = [rollout(theta, rng) for _ in range(10)]
                theta, V, log, clip_frac = ppo_update(
                    theta, V, batch, 1.5, 0.75, eps, 0.95, 4
                )
                if it >= 180:
                    last_clip = clip_frac
            pi = [[softmax(theta[s])[a] for a in range(NA)] for s in range(NS)]
            V_exact = policy_eval_exact(pi)
            finals.append(V_exact[START])
            late_clips.append(last_clip)

        mean = sum(finals) / N_SEEDS
        std = math.sqrt(sum((v - mean)**2 for v in finals) / N_SEEDS)
        late_mean = sum(late_clips) / N_SEEDS
        results[str(eps)] = {"eps": eps, "mean": mean, "std": std, "late_clip_frac": late_mean, "per_seed": finals}
        print(f"  ε={eps}: {mean:.4f} ± {std:.4f}, late clip: {late_mean:.4f}")

    out = {
        "config": {"lr": 1.5, "batch": 10, "epochs": 4, "gae_lambda": 0.95, "n_iters": 200, "n_seeds": N_SEEDS},
        "results": results,
    }
    with open(OUT / "clip_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    return out

# ── verify_anchors ────────────────────────────────────────────────────────────

def verify_anchors():
    print("=" * 60)
    print("PPO Lesson 11 — Numerical Anchors Verification")
    print("=" * 60)

    # V*(0,0) = 0.7290
    uniform_pi = [[1.0/NA]*NA for _ in range(NS)]
    # Deterministic optimal policy (right/down to goal)
    opt_pi = [[0.0]*NA for _ in range(NS)]
    for s in range(NS):
        if s in TERMINALS:
            opt_pi[s][0] = 1.0
        else:
            r, c = _rc(s)
            # Greedy path: go right if c < 2, else go down
            if c < 2:
                opt_pi[s][1] = 1.0  # RIGHT
            elif r < 2:
                opt_pi[s][2] = 1.0  # DOWN
            else:
                opt_pi[s][0] = 1.0
    V_star = policy_eval_exact(opt_pi)
    V_uniform = policy_eval_exact(uniform_pi)
    print(f"V*(0,0) = {V_star[START]:.4f}    (target: 0.7290)")
    print(f"V^pi_uniform(0,0) = {V_uniform[START]:.4f}  (target: -0.4205)")

    # PPO headline
    N_SEEDS = 10
    ppo_finals = []
    for seed in range(N_SEEDS):
        _, _, logs = run_ppo(seed, 200, 0.5, 0.5, 0.2, 0.95, 4, 20)
        ppo_finals.append(logs[-1]["vStart"])
    ppo_mean = sum(ppo_finals) / N_SEEDS
    ppo_std = math.sqrt(sum((v - ppo_mean)**2 for v in ppo_finals) / N_SEEDS)
    print(f"PPO headline (lr=0.5 batch=20 200 iters 10 seeds): {ppo_mean:.4f} ± {ppo_std:.4f}  (target: 0.7267 ± 0.0003)")

    # Vanilla headline
    van_finals = []
    for seed in range(N_SEEDS):
        _, _, logs = run_vanilla(seed, 200, 0.5, 0.5, 0.95, 20)
        van_finals.append(logs[-1]["vStart"])
    van_mean = sum(van_finals) / N_SEEDS
    van_std = math.sqrt(sum((v - van_mean)**2 for v in van_finals) / N_SEEDS)
    print(f"Vanilla headline (same config): {van_mean:.4f} ± {van_std:.4f}  (target: ~0.6504 ± 0.0218)")

    # PPO long-run (500 iters)
    ppo_long = []
    for seed in range(5):
        _, _, logs = run_ppo(seed, 500, 0.5, 0.5, 0.2, 0.95, 4, 20)
        ppo_long.append(logs[-1]["vStart"])
    long_mean = sum(ppo_long) / len(ppo_long)
    long_std = math.sqrt(sum((v - long_mean)**2 for v in ppo_long) / len(ppo_long))
    print(f"PPO long-run (500 iters, 5 seeds): {long_mean:.4f} ± {long_std:.4f}  (target: 0.7282 ± 0.0002)")

    # Aggressive regime
    rng = mulberry32(0)
    theta = [[0.0]*NA for _ in range(NS)]
    V = [0.0] * NS
    early_clips = []
    late_clips = []
    for it in range(100):
        batch = [rollout(theta, rng) for _ in range(5)]
        theta, V, log, clip_frac = ppo_update(theta, V, batch, 2.0, 1.0, 0.2, 0.95, 10)
        if it < 5:
            early_clips.append(clip_frac)
        if it >= 90:
            late_clips.append(clip_frac)
    early_mean = sum(early_clips) / len(early_clips) if early_clips else 0
    late_mean = sum(late_clips) / len(late_clips) if late_clips else 0
    print(f"Aggressive regime (lr=2.0 batch=5 epochs=10) early clip frac: ~{early_mean:.2f}  (target: ~0.35)")
    print(f"Aggressive regime late clip frac: ~{late_mean:.2f}  (target: ~0.02)")

    # GAE λ sweep
    for lam in [0.0, 1.0]:
        finals = []
        for seed in range(20):
            _, _, logs = run_ppo(seed, 200, 0.5, 0.5, 0.2, lam, 4, 5)
            finals.append(logs[-1]["vStart"])
        m = sum(finals) / len(finals)
        s = math.sqrt(sum((v - m)**2 for v in finals) / len(finals))
        target = "0.7237 ± 0.0008" if lam == 0.0 else "0.7208 ± 0.0027"
        print(f"GAE λ={lam:.1f} batch=5: {m:.4f} ± {s:.4f}  (target: {target})")

    # LR sensitivity vanilla at lr=0.1
    van_lr01 = []
    for seed in range(20):
        _, _, logs = run_vanilla(seed, 200, 0.1, 0.1, 0.95, 10)
        van_lr01.append(logs[-1]["vStart"])
    print(f"LR sensitivity at lr=0.1 vanilla: {sum(van_lr01)/len(van_lr01):.4f}  (target: -0.1053)")
    print("=" * 60)

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify_anchors()
    else:
        gen_headline_runs()
        gen_gae_lambda()
        gen_lr_sensitivity()
        gen_clip_sweep()
        print("Done. JSON files written to public/data/ppo/")
