#!/usr/bin/env python3
"""Offline bandit simulation for RL Lesson 1.

Produces the pre-computed default datasets the lesson ships as fast-loading
initial state / static fallback:

    public/data/bandits/regret_curves_default.json
    public/data/bandits/beta_posterior_snapshots.json

Run once during development (and re-run whenever the algorithms or
hyperparameters change). The numbers promised in the prose (§7 table) must
match what this script emits. Not part of the Vite build.

Usage:  python3 scripts/simulate_bandits.py
"""

import json
import math
import os

import numpy as np

# ---- Running example (locked by the spec) -------------------------------
MUS = [0.3, 0.5, 0.7]
T = 5000
# The offline reference curves average over more seeds than the Battle Arena's
# live default (200). ε-greedy(0.01) is fat-tailed (std ≈ 160 at T=5000), so a
# 200-seed mean is noisy (stderr ≈ 11); 1000 seeds pins the mean (stderr ≈ 4.6)
# and smooths the shipped curve. All other algorithms are already stable at 200.
SEEDS = 1000

ALGOS = ["random", "eps01", "eps001", "ucb1", "thompson"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "public", "data", "bandits")


def simulate(algo: str, mus, T: int, rng: np.random.Generator):
    """Return the cumulative pseudo-regret curve (length T) for one seed."""
    K = len(mus)
    N = np.zeros(K)
    S = np.zeros(K)
    muhat = np.zeros(K)
    alpha = np.ones(K)  # Thompson Beta posterior
    beta = np.ones(K)
    mu_star = max(mus)
    gaps = np.array([mu_star - m for m in mus])

    eps = 0.1 if algo == "eps01" else 0.01
    curve = np.empty(T)
    regret = 0.0

    for t in range(T):
        if algo == "random":
            a = int(rng.integers(K))
        elif algo in ("eps01", "eps001"):
            if t < K:
                a = t  # initialize: pull each arm once
            elif rng.random() < eps:
                a = int(rng.integers(K))
            else:
                a = int(np.argmax(muhat))
        elif algo == "ucb1":
            if t < K:
                a = t
            else:
                bonus = np.sqrt(2.0 * math.log(t + 1) / N)
                a = int(np.argmax(muhat + bonus))
        elif algo == "thompson":
            theta = rng.beta(alpha, beta)
            a = int(np.argmax(theta))
        else:
            raise ValueError(f"unknown algo {algo}")

        r = 1.0 if rng.random() < mus[a] else 0.0
        N[a] += 1
        S[a] += r
        muhat[a] = S[a] / N[a]
        if algo == "thompson":
            if r >= 0.5:
                alpha[a] += 1
            else:
                beta[a] += 1

        regret += gaps[a]
        curve[t] = regret

    return curve


def regret_curves(mus, T: int, seeds: int):
    results = {}
    for algo in ALGOS:
        runs = np.empty((seeds, T))
        for s in range(seeds):
            rng = np.random.default_rng(s)  # fixed seed sequence, reproducible
            runs[s] = simulate(algo, mus, T, rng)
        mean = runs.mean(axis=0)
        std = runs.std(axis=0)
        results[algo] = {
            "mean": np.round(mean, 2).tolist(),
            "std": np.round(std, 2).tolist(),
            "final": round(float(mean[-1]), 2),
        }
    return results


def lai_robbins_constant(mus):
    mu_star = max(mus)

    def kl(p, q):
        return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

    c = 0.0
    for m in mus:
        d = mu_star - m
        if d > 0:
            c += d / kl(m, mu_star)
    return c


def beta_snapshots(mus, n_steps: int = 120, seed: int = 0):
    """Deterministic Thompson trace for V6's canonical default animation."""
    K = len(mus)
    rng = np.random.default_rng(seed)
    alpha = np.ones(K)
    beta = np.ones(K)
    trace = []
    for t in range(n_steps):
        theta = rng.beta(alpha, beta)
        a = int(np.argmax(theta))
        r = 1.0 if rng.random() < mus[a] else 0.0
        if r >= 0.5:
            alpha[a] += 1
        else:
            beta[a] += 1
        trace.append(
            {
                "t": t,
                "arm": a,
                "reward": int(r),
                "samples": [round(float(x), 4) for x in theta],
                "alpha": [int(x) for x in alpha],
                "beta": [int(x) for x in beta],
            }
        )
    snapshot_at = [5, 10, 20, 30, 50, 100]
    snapshots = {str(s): trace[s - 1] for s in snapshot_at if s - 1 < len(trace)}
    return {"trace": trace, "snapshots": snapshots}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    lr_const = lai_robbins_constant(MUS)

    curves = regret_curves(MUS, T, SEEDS)
    default = {
        "mus": MUS,
        "T": T,
        "seeds": SEEDS,
        "laiRobbinsConstant": round(lr_const, 4),
        "algos": curves,
    }
    path1 = os.path.join(OUT_DIR, "regret_curves_default.json")
    with open(path1, "w") as f:
        json.dump(default, f, separators=(",", ":"))

    snaps = beta_snapshots(MUS)
    snaps["mus"] = MUS
    snaps["seed"] = 0
    path2 = os.path.join(OUT_DIR, "beta_posterior_snapshots.json")
    with open(path2, "w") as f:
        json.dump(snaps, f, separators=(",", ":"))

    # ---- Report ----
    print(f"Lai-Robbins constant: {lr_const:.4f}  (floor at T={T}: {lr_const*math.log(T):.1f})")
    print("Final regret R_5000 (avg over 200 seeds):")
    for algo in ALGOS:
        print(f"  {algo:10s} {curves[algo]['final']:8.2f}")
    print(f"\nWrote {path1}  ({os.path.getsize(path1)/1024:.1f} KB)")
    print(f"Wrote {path2}  ({os.path.getsize(path2)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
