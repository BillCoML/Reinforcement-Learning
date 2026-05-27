#!/usr/bin/env python3
"""
Pre-computes MC lesson statistics for instant initial render.
Outputs four JSON files into public/data/mc/.
Mirrors the TypeScript monte-carlo/ module exactly.
"""

import json, math, random
from pathlib import Path

OUT = Path(__file__).parent.parent / "public" / "data" / "mc"
OUT.mkdir(parents=True, exist_ok=True)

# ── Gridworld ─────────────────────────────────────────────────────────────────
GRID, NA, NS = 3, 4, 9
PIT, GOAL = 4, 8
TERMINALS = {PIT, GOAL}
DELTAS = [(-1,0),(0,1),(1,0),(0,-1)]  # UP RIGHT DOWN LEFT
GAMMA, MAX_STEPS = 0.9, 200

def idx(r, c): return r * GRID + c
def rc(s): return s // GRID, s % GRID
def move(r, c, dr, dc):
    nr, nc = r+dr, c+dc
    return (nr, nc) if 0<=nr<GRID and 0<=nc<GRID else (r, c)

def next_s(s, a):
    r, c = rc(s); dr, dc = DELTAS[a]; nr, nc = move(r,c,dr,dc)
    return idx(nr, nc)

def r_sa(s, a):
    sp = next_s(s, a)
    return 1.0 if sp==GOAL else (-1.0 if sp==PIT else 0.0)

OPT = {0:1,1:1,2:2,3:2,5:2,6:1,7:1}  # optimal actions (value-iteration, first-found)

def uniform_pi(s): return [0.25]*4

def det_opt_pi(s):
    row = [0.0]*4; row[OPT.get(s,0)] = 1.0; return row

def rollout(s0, pi_fn, rng):
    steps, s = [], s0
    for _ in range(MAX_STEPS):
        if s in TERMINALS: break
        probs = pi_fn(s); u = rng.random(); a, acc = 0, 0.0
        for i, p in enumerate(probs):
            acc += p
            if u <= acc: a = i; break
        sp = next_s(s, a); r = r_sa(s, a)
        steps.append((s, a, r, sp)); s = sp
    return steps

def returns(rewards, gamma):
    T, G, Gs = len(rewards), 0.0, [0.0]*len(rewards)
    for t in range(T-1,-1,-1): G = rewards[t]+gamma*G; Gs[t] = G
    return Gs

# ── MC on-policy eval ─────────────────────────────────────────────────────────
def mc_eval(pi_fn, n_ep, first_visit=True, s0=0, rng=None):
    rng = rng or random.Random(0)
    V, visits = [0.0]*NS, [0]*NS
    for _ in range(n_ep):
        steps = rollout(s0, pi_fn, rng)
        if not steps: continue
        Gs = returns([r for _,_,r,_ in steps], GAMMA)
        seen = set()
        for t, (s,_,_,_) in enumerate(steps):
            if first_visit and s in seen: continue
            seen.add(s); visits[s] += 1
            V[s] += (Gs[t]-V[s])/visits[s]
    return V, visits

# ── MC off-policy eval ────────────────────────────────────────────────────────
def mc_off_policy(target_fn, behavior_fn, n_ep, weighted=True, s0=0, rng=None):
    rng = rng or random.Random(0)
    num, den, cnt, sumW, sumW2 = [0.0]*NS,[0.0]*NS,[0]*NS,[0.0]*NS,[0.0]*NS
    nonzero = 0
    for _ in range(n_ep):
        steps = rollout(s0, behavior_fn, rng)
        T = len(steps)
        if T == 0: cnt[s0]+=1; continue
        Gs = returns([r for _,_,r,_ in steps], GAMMA)
        # backward suffix weights
        rho = [0.0]*(T+1); rho[T] = 1.0
        for k in range(T-1,-1,-1):
            s,a,_,_ = steps[k]
            pb = behavior_fn(s)[a]; pt = target_fn(s)[a]
            rho[k] = 0.0 if pb==0 else rho[k+1]*(pt/pb)
        if rho[0] > 0: nonzero += 1
        seen = set()
        for t,(s,_,_,_) in enumerate(steps):
            if s in seen: continue
            seen.add(s); w = rho[t]
            num[s] += w*Gs[t]
            if weighted: den[s] += w
            else: cnt[s] += 1
            sumW[s] += w; sumW2[s] += w*w
    V = [num[s]/den[s] if (weighted and den[s]) else (num[s]/cnt[s] if (not weighted and cnt[s]) else 0.0) for s in range(NS)]
    ess = [(sumW[s]**2)/sumW2[s] if sumW2[s] else 0.0 for s in range(NS)]
    return V, ess, nonzero

# ── MC control ────────────────────────────────────────────────────────────────
def mc_control_es(n_ep, rng=None):
    rng = rng or random.Random(0)
    Q = [[0.0]*NA for _ in range(NS)]; cnt = [[0]*NA for _ in range(NS)]
    greedy = [0]*NS; non_term = [s for s in range(NS) if s not in TERMINALS]
    for _ in range(n_ep):
        s0 = non_term[int(rng.random()*len(non_term))]; a0 = int(rng.random()*NA)
        sp0 = next_s(s0,a0); r0 = r_sa(s0,a0)
        def gpi(s, _g=greedy): row=[0.0]*NA; row[_g[s]]=1.0; return row
        rest = rollout(sp0, gpi, rng)
        full = [(s0,a0,r0,sp0)]+rest
        Gs = returns([r for _,_,r,_ in full], GAMMA)
        seen = set()
        for t,(s,a,_,_) in enumerate(full):
            if s in TERMINALS: continue
            key=(s,a)
            if key in seen: continue
            seen.add(key); cnt[s][a]+=1; Q[s][a]+=(Gs[t]-Q[s][a])/cnt[s][a]
            greedy[s] = max(range(NA), key=lambda aa,ss=s: Q[ss][aa])
    return Q, greedy

def mc_control_eps(eps_fn, n_ep, s0=0, rng=None):
    rng = rng or random.Random(0)
    Q = [[0.0]*NA for _ in range(NS)]; cnt = [[0]*NA for _ in range(NS)]
    greedy = [0]*NS
    for ep in range(n_ep):
        eps = eps_fn(ep)
        def eppi(s, _g=greedy, _e=eps): row=[_e/NA]*NA; row[_g[s]]+=1-_e; return row
        steps = rollout(s0, eppi, rng)
        if not steps: continue
        Gs = returns([r for _,_,r,_ in steps], GAMMA)
        seen = set()
        for t,(s,a,_,_) in enumerate(steps):
            if s in TERMINALS: continue
            key=(s,a)
            if key in seen: continue
            seen.add(key); cnt[s][a]+=1; Q[s][a]+=(Gs[t]-Q[s][a])/cnt[s][a]
            greedy[s] = max(range(NA), key=lambda aa,ss=s: Q[ss][aa])
    return Q, greedy

def stats(vals):
    m = sum(vals)/len(vals)
    sd = math.sqrt(sum((v-m)**2 for v in vals)/(len(vals)-1)) if len(vals)>1 else 0.0
    return round(m,6), round(sd,6)

SEED_BASE = 1_000_003

# ── 1. On-policy convergence ──────────────────────────────────────────────────
rows = []
for N in [100, 1000, 10000]:
    ests = []
    for trial in range(50):
        rng = random.Random((42*SEED_BASE+trial*97001) % (2**31))
        V, _ = mc_eval(uniform_pi, N, rng=rng)
        ests.append(V[0])
    m, sd = stats(ests)
    rows.append({"N": N, "mean": m, "std": sd, "rmse": round(math.sqrt((m+0.4205)**2+sd**2),6)})
    print(f"  on-policy N={N}: mean={m:.4f} std={sd:.4f}")

with open(OUT/"on_policy_convergence.json","w") as f:
    json.dump({"trueValue": -0.4205, "rows": rows}, f, indent=2)

# ── 2. FV vs EV comparison ────────────────────────────────────────────────────
fv_ests, ev_ests = [], []
for trial in range(50):
    rng_fv = random.Random((7*SEED_BASE+trial*97001) % (2**31))
    rng_ev = random.Random((13*SEED_BASE+trial*97001) % (2**31))
    V_fv, _ = mc_eval(uniform_pi, 1000, first_visit=True, rng=rng_fv)
    V_ev, _ = mc_eval(uniform_pi, 1000, first_visit=False, rng=rng_ev)
    fv_ests.append(V_fv[0]); ev_ests.append(V_ev[0])

fv_m, fv_sd = stats(fv_ests); ev_m, ev_sd = stats(ev_ests)
print(f"  fv_vs_ev N=1000: fv={fv_m:.4f}±{fv_sd:.4f}  ev={ev_m:.4f}±{ev_sd:.4f}")
with open(OUT/"fv_vs_ev_comparison.json","w") as f:
    json.dump({"N": 1000, "trueValue": -0.4205,
               "firstVisit": {"mean": fv_m, "std": fv_sd},
               "everyVisit": {"mean": ev_m, "std": ev_sd}}, f, indent=2)

# ── 3. Off-policy gridworld ───────────────────────────────────────────────────
off_rows = []
for N in [100, 1000, 10000]:
    ord_ests, wt_ests, nz_counts = [], [], []
    for trial in range(50):
        rng = random.Random((99*SEED_BASE+trial*97001) % (2**31))
        V_ord, _, nz_ord = mc_off_policy(det_opt_pi, uniform_pi, N, weighted=False, rng=rng)
        rng2 = random.Random((101*SEED_BASE+trial*97001) % (2**31))
        V_wt, ess, nz_wt = mc_off_policy(det_opt_pi, uniform_pi, N, weighted=True, rng=rng2)
        ord_ests.append(V_ord[0]); wt_ests.append(V_wt[0]); nz_counts.append(nz_wt)
    om, osd = stats(ord_ests); wm, wsd = stats(wt_ests)
    avg_nz = round(sum(nz_counts)/len(nz_counts), 1)
    off_rows.append({"N": N, "avgNonZero": avg_nz,
                     "ordinaryMean": om, "ordinarySD": osd,
                     "weightedMean": wm, "weightedSD": wsd})
    print(f"  off-policy N={N}: ord={om:.4f}±{osd:.4f}  wt={wm:.4f}±{wsd:.4f}  nz≈{avg_nz}")

with open(OUT/"off_policy_gridworld.json","w") as f:
    json.dump({"trueValue": 0.7290, "rows": off_rows}, f, indent=2)

# ── 4. Control learning curves ────────────────────────────────────────────────
def run_es_curve(seed, n_ep=50000, interval=100):
    rng = random.Random(seed)
    Q = [[0.0]*NA for _ in range(NS)]; cnt = [[0]*NA for _ in range(NS)]
    greedy = [0]*NS; non_term = [s for s in range(NS) if s not in TERMINALS]
    curve = []
    for ep in range(n_ep):
        if ep % interval == 0:
            curve.append(round(max(Q[0]), 6))
        s0_ep = non_term[int(rng.random()*len(non_term))]; a0 = int(rng.random()*NA)
        sp0 = next_s(s0_ep,a0); r0 = r_sa(s0_ep,a0)
        def gpi(s,_g=greedy): row=[0.0]*NA; row[_g[s]]=1.0; return row
        rest = rollout(sp0, gpi, rng)
        full = [(s0_ep,a0,r0,sp0)]+rest
        Gs = returns([r for _,_,r,_ in full], GAMMA)
        seen = set()
        for t,(s,a,_,_) in enumerate(full):
            if s in TERMINALS: continue
            key=(s,a)
            if key in seen: continue
            seen.add(key); cnt[s][a]+=1; Q[s][a]+=(Gs[t]-Q[s][a])/cnt[s][a]
            greedy[s] = max(range(NA), key=lambda aa,ss=s: Q[ss][aa])
    curve.append(round(max(Q[0]), 6))
    return curve

def run_eps_curve(eps_fn, seed, n_ep=50000, interval=100):
    rng = random.Random(seed)
    Q = [[0.0]*NA for _ in range(NS)]; cnt = [[0]*NA for _ in range(NS)]
    greedy = [0]*NS; curve = []
    for ep in range(n_ep):
        if ep % interval == 0:
            curve.append(round(max(Q[0]), 6))
        eps = eps_fn(ep)
        def eppi(s,_g=greedy,_e=eps): row=[_e/NA]*NA; row[_g[s]]+=1-_e; return row
        steps = rollout(0, eppi, rng)
        if not steps: continue
        Gs = returns([r for _,_,r,_ in steps], GAMMA)
        seen = set()
        for t,(s,a,_,_) in enumerate(steps):
            if s in TERMINALS: continue
            key=(s,a)
            if key in seen: continue
            seen.add(key); cnt[s][a]+=1; Q[s][a]+=(Gs[t]-Q[s][a])/cnt[s][a]
            greedy[s] = max(range(NA), key=lambda aa,ss=s: Q[ss][aa])
    curve.append(round(max(Q[0]), 6))
    return curve

print("  computing control curves (5 seeds × 3 algorithms)...")
es_curves, eps01_curves, glie_curves = [], [], []
for seed in range(5):
    s = (seed+1)*SEED_BASE % (2**31)
    es_curves.append(run_es_curve(s))
    eps01_curves.append(run_eps_curve(lambda ep: 0.1, s+7))
    glie_curves.append(run_eps_curve(lambda ep: 1/math.sqrt(ep+1), s+13))
    print(f"    seed {seed}: es_final={es_curves[-1][-1]:.4f}  eps01={eps01_curves[-1][-1]:.4f}  glie={glie_curves[-1][-1]:.4f}")

with open(OUT/"control_learning_curves.json","w") as f:
    json.dump({
        "episodeInterval": 100,
        "nEpisodes": 50000,
        "refOptimal": 0.7290,
        "refEpsSoft": 0.6274,
        "mcES": es_curves,
        "mcEps01": eps01_curves,
        "mcGlie": glie_curves,
    }, f, indent=2)

print("Done. JSON files written to", OUT)
