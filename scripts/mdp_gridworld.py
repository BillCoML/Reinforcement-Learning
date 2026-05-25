#!/usr/bin/env python3
"""Offline value generator for RL Lesson 2 (Markov Decision Processes).

Pre-computes the 3×3 gridworld's V^π (uniform), V*, Q*, and a 40-step Bellman
backup trace, for instant initial render. The browser recomputes these live via
src/mdp/*.ts; the JSON is a static fallback. Numbers here must match the prose
tables and the in-browser values to 4 decimals.

Run once during development:  python3 scripts/mdp_gridworld.py
Not part of the Vite build.
"""

import json
import os

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "public", "data", "mdp")

N, NA = 9, 4                       # 3×3 grid, actions Up/Right/Down/Left
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DELTAS = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}
PIT, GOAL, GAMMA = (1, 1), (2, 2), 0.9


def idx(r, c):
    return r * 3 + c


def build():
    """Deterministic gridworld → (P[s,a,s'], r[s,a], terminals[s])."""
    pit, goal = idx(*PIT), idx(*GOAL)
    terminals = np.array([s in (pit, goal) for s in range(N)])
    enter = lambda sp: 1.0 if sp == goal else -1.0 if sp == pit else 0.0
    P = np.zeros((N, NA, N))
    r = np.zeros((N, NA))
    for s in range(N):
        row, col = divmod(s, 3)
        for a in range(NA):
            if terminals[s]:
                P[s, a, s] = 1.0
                continue
            dr, dc = DELTAS[a]
            nr, nc = row + dr, col + dc
            if not (0 <= nr < 3 and 0 <= nc < 3):
                nr, nc = row, col       # walls bounce
            sp = idx(nr, nc)
            P[s, a, sp] += 1.0
            r[s, a] = sum(P[s, a, k] * enter(k) for k in range(N))
    return P, r, terminals


def policy_eval_exact(P, r, terminals, pi):
    Ppi = np.einsum("sa,sap->sp", pi, P)
    Rpi = np.einsum("sa,sa->s", pi, r)
    V = np.linalg.solve(np.eye(N) - GAMMA * Ppi, Rpi)
    V[terminals] = 0.0
    return V


def q_from_v(P, r, terminals, V):
    Q = r + GAMMA * np.einsum("sap,p->sa", P, V)
    Q[terminals] = 0.0
    return Q


def opt_backup(P, r, terminals, V):
    Q = r + GAMMA * np.einsum("sap,p->sa", P, V)
    Vn = Q.max(axis=1)
    Vn[terminals] = 0.0
    return Vn


def exp_backup(P, r, terminals, pi, V):
    Q = r + GAMMA * np.einsum("sap,p->sa", P, V)
    Vn = np.einsum("sa,sa->s", pi, Q)
    Vn[terminals] = 0.0
    return Vn


def grid(V):
    return [[round(float(V[idx(rr, cc)]), 6) for cc in range(3)] for rr in range(3)]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    P, r, terminals = build()
    uniform = np.full((N, NA), 1.0 / NA)

    V_pi = policy_eval_exact(P, r, terminals, uniform)
    V_star = np.zeros(N)
    for _ in range(200):
        V_star = opt_backup(P, r, terminals, V_star)
    Q_star = q_from_v(P, r, terminals, V_star)
    A_star = Q_star - V_star[:, None]
    optimal_actions = [
        [a for a in range(NA) if Q_star[s, a] >= Q_star[s].max() - 1e-9]
        if not terminals[s] else []
        for s in range(N)
    ]

    # 40-step traces from V_0 = 0, both operators.
    def trace(backup):
        V, out = np.zeros(N), []
        out.append({"k": 0, "V": [round(float(x), 6) for x in V]})
        for k in range(1, 41):
            V = backup(V)
            out.append({"k": k, "V": [round(float(x), 6) for x in V]})
        return out

    files = {
        "gridworld_v_pi_uniform.json": {
            "gamma": GAMMA, "policy": "uniform",
            "V": [round(float(x), 6) for x in V_pi], "grid": grid(V_pi),
        },
        "gridworld_v_star.json": {
            "gamma": GAMMA,
            "V": [round(float(x), 6) for x in V_star], "grid": grid(V_star),
            "optimalActions": optimal_actions,
        },
        "gridworld_q_star.json": {
            "gamma": GAMMA,
            "Q": [[round(float(x), 6) for x in row] for row in Q_star],
            "A": [[round(float(x), 6) for x in row] for row in A_star],
        },
        "gridworld_backup_trace.json": {
            "gamma": GAMMA,
            "expectation": trace(lambda V: exp_backup(P, r, terminals, uniform, V)),
            "optimality": trace(lambda V: opt_backup(P, r, terminals, V)),
        },
    }

    for name, data in files.items():
        with open(os.path.join(OUT_DIR, name), "w") as f:
            json.dump(data, f, separators=(",", ":"))

    print("V^pi(0,0) =", round(float(V_pi[idx(0, 0)]), 4), "(want -0.4205)")
    print("V*(0,0)   =", round(float(V_star[idx(0, 0)]), 4), "(want 0.729)")
    print("V*(2,1)   =", round(float(V_star[idx(2, 1)]), 4), "(want 1.0)")
    print("Q*((1,0),R) =", round(float(Q_star[idx(1, 0), RIGHT]), 4), "(want -1.0)")
    print("A*((1,0),R) =", round(float(A_star[idx(1, 0), RIGHT]), 4), "(want -1.81)")
    print("(0,0) optimal actions:", optimal_actions[idx(0, 0)], "(want [1,2] = R,D)")
    print(f"\nWrote {len(files)} files to {os.path.relpath(OUT_DIR)}")


if __name__ == "__main__":
    main()
