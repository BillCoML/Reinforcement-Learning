"""
Pre-computation script for Lesson 12: Maximum-Entropy RL.
Writes three JSON files to public/data/maxent/:
  - alpha_sweep.json       : soft VI results at ~61 alpha values
  - convergence_traces.json: per-iteration data at 5 alpha values
  - maxent_rollouts.json   : Monte Carlo diagnostics at each alpha

Run: python scripts/maxent_traces.py
     python scripts/maxent_traces.py --verify   (prints headline anchors)
"""
import argparse
import json
import math
import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# 3×3 Gridworld (same dynamics as src/mdp/gridworld.ts)
# State indexing: row-major, idx(r,c) = 3r+c
# Pit at (1,1) = state 4; Goal at (2,2) = state 8
# Actions: 0=Up, 1=Right, 2=Down, 3=Left
# ---------------------------------------------------------------------------
GRID = 3
N_S = 9
N_A = 4
GAMMA = 0.9
PIT = 4    # state index of (1,1)
GOAL = 8   # state index of (2,2)
START = 0  # state index of (0,0)
TERMINALS = {PIT, GOAL}

DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]   # Up, Right, Down, Left

def _move(r, c, dr, dc):
    nr, nc = r + dr, c + dc
    if nr < 0 or nr >= GRID or nc < 0 or nc >= GRID:
        return r * GRID + c
    return nr * GRID + nc

def build_mdp():
    """Build transition matrix P[s,a,s'] and reward r[s,a]."""
    P = np.zeros((N_S, N_A, N_S))
    r = np.zeros((N_S, N_A))
    for s in range(N_S):
        row, col = divmod(s, GRID)
        for a in range(N_A):
            if s in TERMINALS:
                P[s, a, s] = 1.0  # absorbing
            else:
                dr, dc = DELTAS[a]
                sp = _move(row, col, dr, dc)
                P[s, a, sp] = 1.0
                r[s, a] = (1.0 if sp == GOAL else -1.0 if sp == PIT else 0.0)
    return P, r

P, R = build_mdp()

# ---------------------------------------------------------------------------
# Soft Value Iteration
# ---------------------------------------------------------------------------
def logsumexp_alpha(q_row, alpha):
    """alpha * log( sum exp(q_row / alpha) ), numerically stable."""
    m = np.max(q_row)
    return alpha * np.log(np.sum(np.exp((q_row - m) / alpha))) + m

def soft_vi(alpha, tol=1e-10, max_iter=10000, return_traces=False):
    """
    Soft value iteration.
    Returns (Q, V_soft, pi) where:
      Q      : (N_S, N_A) soft Q-values
      V_soft : (N_S,) soft Bellman fixed-point (includes entropy bonus)
      pi     : (N_S, N_A) Boltzmann policy
    """
    V = np.zeros(N_S)
    traces_V = [V.copy()] if return_traces else None
    traces_pi = None

    for it in range(max_iter):
        Q = R + GAMMA * np.einsum('san,n->sa', P, V)  # (N_S, N_A)
        Vnew = np.zeros(N_S)
        for s in range(N_S):
            if s in TERMINALS:
                continue
            Vnew[s] = logsumexp_alpha(Q[s], alpha)
        delta = np.max(np.abs(Vnew - V))
        V = Vnew
        if return_traces:
            traces_V.append(V.copy())
        if delta < tol:
            break

    # Final Q from converged V
    Q = R + GAMMA * np.einsum('san,n->sa', P, V)
    for s in TERMINALS:
        Q[s] = 0.0

    # Boltzmann policy
    pi = np.zeros((N_S, N_A))
    for s in range(N_S):
        if s in TERMINALS:
            pi[s] = 1.0 / N_A
        else:
            q = Q[s] - np.max(Q[s])
            exp_q = np.exp(q / alpha)
            pi[s] = exp_q / np.sum(exp_q)

    if return_traces:
        # Compute pi traces
        traces_pi = []
        for Vt in traces_V:
            Qt = R + GAMMA * np.einsum('san,n->sa', P, Vt)
            pi_t = np.zeros((N_S, N_A))
            for s in range(N_S):
                if s in TERMINALS:
                    pi_t[s] = 1.0 / N_A
                else:
                    q = Qt[s] - np.max(Qt[s])
                    eq = np.exp(q / alpha)
                    pi_t[s] = eq / np.sum(eq)
            traces_pi.append(pi_t)
        return Q, V, pi, traces_V, traces_pi

    return Q, V, pi

# ---------------------------------------------------------------------------
# Policy Evaluation (standard, no entropy)
# ---------------------------------------------------------------------------
def policy_eval_exact(pi):
    """Solve (I - gamma * P^pi) V = R^pi for V^pi (no entropy term)."""
    P_pi = np.einsum('sa,san->sn', pi, P)  # (N_S, N_S)
    R_pi = np.einsum('sa,sa->s', pi, R)    # (N_S,)
    A = np.eye(N_S) - GAMMA * P_pi
    V = np.linalg.solve(A, R_pi)
    for s in TERMINALS:
        V[s] = 0.0
    return V

# ---------------------------------------------------------------------------
# Soft Policy Evaluation (with entropy bonus) → V_soft^pi
# ---------------------------------------------------------------------------
def soft_policy_eval(pi, alpha):
    """Solve (I - gamma * P^pi) V = R_soft^pi, where R_soft includes alpha*H(pi)."""
    H_pi = -np.sum(np.where(pi > 0, pi * np.log(pi), 0.0), axis=1)  # (N_S,)
    P_pi = np.einsum('sa,san->sn', pi, P)
    R_pi = np.einsum('sa,sa->s', pi, R)
    R_soft = R_pi + alpha * H_pi
    for s in TERMINALS:
        R_soft[s] = 0.0
    A = np.eye(N_S) - GAMMA * P_pi
    V_soft = np.linalg.solve(A, R_soft)
    for s in TERMINALS:
        V_soft[s] = 0.0
    return V_soft

# ---------------------------------------------------------------------------
# Hard Value Iteration (reference, alpha -> 0)
# ---------------------------------------------------------------------------
def hard_vi(tol=1e-12, max_iter=10000):
    V = np.zeros(N_S)
    for _ in range(max_iter):
        Q = R + GAMMA * np.einsum('san,n->sa', P, V)
        Vnew = np.max(Q, axis=1)
        for s in TERMINALS:
            Vnew[s] = 0.0
        if np.max(np.abs(Vnew - V)) < tol:
            break
        V = Vnew
    return V

# ---------------------------------------------------------------------------
# Monte Carlo diagnostic
# ---------------------------------------------------------------------------
def mc_diagnostic(pi, n_rollouts=5000, max_steps=500, seed=42):
    rng = np.random.default_rng(seed)
    goal_count = pit_count = timeout_count = 0
    terminated_steps = 0
    terminated_count = 0
    hist = np.zeros(50)
    bin_width = max_steps / 50

    for _ in range(n_rollouts):
        s = START
        steps = 0
        terminated = False
        while steps < max_steps:
            if s in TERMINALS:
                terminated = True
                break
            a = rng.choice(N_A, p=pi[s])
            sp = rng.choice(N_S, p=P[s, a])
            s = sp
            steps += 1

        if terminated:
            if s == GOAL:
                goal_count += 1
            elif s == PIT:
                pit_count += 1
            else:
                timeout_count += 1
            terminated_steps += steps
            terminated_count += 1
        else:
            timeout_count += 1

        bin_idx = min(int(steps / bin_width), 49)
        hist[bin_idx] += 1

    hist /= n_rollouts
    mean_steps = (terminated_steps / terminated_count) if terminated_count > 0 else max_steps
    return {
        'goalReachProb': goal_count / n_rollouts,
        'pitReachProb': pit_count / n_rollouts,
        'timeoutProb': timeout_count / n_rollouts,
        'meanStepsToTerminal': mean_steps,
        'lengthHistogram': hist.tolist(),
    }

# ---------------------------------------------------------------------------
# Alpha grid
# ---------------------------------------------------------------------------
def build_alpha_grid():
    """~60 log-spaced alphas plus the must-have specific values."""
    log_grid = np.logspace(np.log10(0.001), np.log10(1.0), 50).tolist()
    must_have = [0.001, 0.005, 0.01, 0.0198, 0.02, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
    combined = sorted(set([round(a, 6) for a in log_grid + must_have]))
    return combined

# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------
def generate_all(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    alphas = build_alpha_grid()

    print(f"Running soft VI at {len(alphas)} alpha values...")

    # --- alpha_sweep.json ---
    sweep = []
    for alpha in alphas:
        Q, V_soft, pi = soft_vi(alpha)
        V_pi = policy_eval_exact(pi)
        # Mean entropy (non-terminal states)
        H = -np.sum(np.where(pi > 0, pi * np.log(pi), 0.0), axis=1)
        non_terminal_mask = np.array([s not in TERMINALS for s in range(N_S)], dtype=bool)
        mean_H = float(np.mean(H[non_terminal_mask]))
        # KL to uniform: H_uniform - H = log(4) - H
        H_uniform = math.log(N_A)
        kl_to_uniform = float(np.mean((H_uniform - H[non_terminal_mask])))
        sweep.append({
            'alpha': alpha,
            'Q': Q.tolist(),
            'V_soft': V_soft.tolist(),
            'pi': pi.tolist(),
            'V_pi': V_pi.tolist(),
            'meanEntropy': mean_H,
            'klToUniform': kl_to_uniform,
        })
        print(f"  alpha={alpha:.4f}  V_soft[0]={V_soft[0]:.4f}  V_pi[0]={V_pi[0]:.4f}")

    path = os.path.join(out_dir, 'alpha_sweep.json')
    with open(path, 'w') as f:
        json.dump(sweep, f, separators=(',', ':'))
    print(f"Wrote {path} ({os.path.getsize(path)//1024} KB)")

    # --- convergence_traces.json ---
    trace_alphas = [0.01, 0.05, 0.1, 0.5, 1.0]
    traces = {}
    for alpha in trace_alphas:
        _, _, _, traces_V, traces_pi = soft_vi(alpha, return_traces=True)
        # Keep at most 40 iterations
        step = max(1, len(traces_V) // 40)
        V_arr = [v.tolist() for v in traces_V[::step]][:40]
        pi_arr = [p[START].tolist() for p in traces_pi[::step]][:40]
        traces[str(alpha)] = {'V': V_arr, 'pi_start': pi_arr}
    path = os.path.join(out_dir, 'convergence_traces.json')
    with open(path, 'w') as f:
        json.dump(traces, f, separators=(',', ':'))
    print(f"Wrote {path} ({os.path.getsize(path)//1024} KB)")

    # --- maxent_rollouts.json ---
    print("Running Monte Carlo diagnostics (5000 rollouts × 61 alphas) ...")
    rollouts = []
    for entry in sweep:
        alpha = entry['alpha']
        pi = np.array(entry['pi'])
        diag = mc_diagnostic(pi, n_rollouts=5000, max_steps=500, seed=42)
        diag['alpha'] = alpha
        rollouts.append(diag)
        print(f"  alpha={alpha:.4f}  goal={diag['goalReachProb']:.3f}  "
              f"timeout={diag['timeoutProb']:.3f}  steps={diag['meanStepsToTerminal']:.1f}")
    path = os.path.join(out_dir, 'maxent_rollouts.json')
    with open(path, 'w') as f:
        json.dump(rollouts, f, separators=(',', ':'))
    print(f"Wrote {path} ({os.path.getsize(path)//1024} KB)")

    return sweep, rollouts

# ---------------------------------------------------------------------------
# Verify anchors
# ---------------------------------------------------------------------------
def verify_anchors():
    print("\n=== Verifying headline anchors ===\n")
    V_hard = hard_vi()
    print(f"V*(0,0) hard VI                = {V_hard[START]:.4f}  [expected 0.7290]")

    Q, V_s, pi = soft_vi(1e-4)
    print(f"V_soft(0,0) at alpha=1e-4      = {V_s[START]:.4f}  [expected ≈ 0.7291]")

    Q, V_s, pi = soft_vi(0.02)
    V_p = policy_eval_exact(pi)
    print(f"V_soft(0,0) at alpha=0.02      = {V_s[START]:.4f}  [expected 0.7392]")
    print(f"V^pi(0,0)   at alpha=0.02      = {V_p[START]:.4f}  [expected 0.7217 — L10 cap]")

    for alpha, exp in [(0.05, 0.5928), (0.10, 0.0711), (0.20, 0.0006)]:
        _, _, pi = soft_vi(alpha)
        V_p = policy_eval_exact(pi)
        print(f"V^pi(0,0)   at alpha={alpha}      = {V_p[START]:.4f}  [expected {exp}]")

    for alpha, exp_g, exp_s in [(0.10, 0.999, 78.1), (0.20, 0.051, 252.8), (0.50, 0.001, None)]:
        _, _, pi = soft_vi(alpha)
        diag = mc_diagnostic(pi, n_rollouts=5000, max_steps=500, seed=42)
        s_str = f"  mean_steps={diag['meanStepsToTerminal']:.1f}" if exp_s else ""
        print(f"Goal reach prob at alpha={alpha}  = {diag['goalReachProb']:.3f}  [expected {exp_g}]{s_str}")

    print("\nAll anchors printed. Compare against expected values above.")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true', help='Print headline anchors only')
    parser.add_argument('--out', default='public/data/maxent', help='Output directory')
    args = parser.parse_args()

    if args.verify:
        verify_anchors()
    else:
        generate_all(args.out)
        print("\nDone. Run with --verify to print headline anchors.")
