#!/usr/bin/env python3
"""Offline example-chain generator for RL Prereq A (Markov Chains).

Produces the pre-baked chain examples the lesson ships for instant initial
render. The browser recomputes these live via src/markov/chain.ts; the JSON is
just a fast static fallback. Numbers here must match the prose tables and the
in-browser values to 6 decimals.

Run once during development:  python3 scripts/markov_examples.py
Not part of the Vite build.
"""

import json
import math
import os

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "public", "data", "markov")

# Filename -> transition matrix (locked by the spec).
CHAINS = {
    "weather": [[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]],
    "periodic2": [[0.0, 1.0], [1.0, 0.0]],
    "reducible": [[0.7, 0.3, 0, 0], [0.4, 0.6, 0, 0], [0.1, 0, 0.3, 0.6], [0, 0.2, 0.5, 0.3]],
    "slow_mixing": [[0.9, 0.1, 0, 0], [0.1, 0.8, 0.1, 0], [0, 0.1, 0.8, 0.1], [0, 0, 0.1, 0.9]],
    "birth_death": [[0.5, 0.5, 0], [0.3, 0.4, 0.3], [0, 0.6, 0.4]],
}


def stationary(P):
    """Left eigenvector of P for eigenvalue 1, normalized to the simplex."""
    vals, vecs = np.linalg.eig(P.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    v = np.real(vecs[:, idx])
    v = v / v.sum()
    v = np.clip(v, 0, None)
    return v / v.sum()


def total_variation(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def analyze(P):
    P = np.array(P, dtype=float)
    K = P.shape[0]
    pi = stationary(P)
    vals = np.linalg.eigvals(P)
    mags = np.sort(np.abs(vals))[::-1]
    lambda_star = float(mags[1]) if len(mags) > 1 else 0.0
    gap = 1.0 - lambda_star
    eps = 0.01
    mix = math.log(1 / eps) / gap if gap > 1e-12 else None
    # TV-to-stationarity curve from a delta on state 0.
    mu = np.zeros(K)
    mu[0] = 1.0
    tv_curve = []
    for n in range(0, 21):
        tv_curve.append(round(total_variation(mu @ np.linalg.matrix_power(P, n), pi), 8))
    return {
        "P": P.tolist(),
        "K": K,
        "pi": [round(float(x), 6) for x in pi],
        "eigenvalues": [{"re": round(float(z.real), 6), "im": round(float(z.imag), 6),
                         "abs": round(float(abs(z)), 6)} for z in vals],
        "lambdaStar": round(lambda_star, 6),
        "spectralGap": round(gap, 6),
        "mixingTimeBound": (round(mix, 6) if mix is not None else None),
        "tvFromState0": tv_curve,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, P in CHAINS.items():
        data = analyze(P)
        path = os.path.join(OUT_DIR, f"{name}.json")
        with open(path, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        ls = data["lambdaStar"]
        print(f"{name:12s} pi={data['pi']}  lambda*={ls}  mix={data['mixingTimeBound']}")
    print(f"\nWrote {len(CHAINS)} files to {os.path.relpath(OUT_DIR)}")


if __name__ == "__main__":
    main()
