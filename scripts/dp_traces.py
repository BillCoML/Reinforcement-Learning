"""Pre-compute PI/VI traces and sweep-order comparisons for Lesson 3 (DP).

Outputs:
  public/data/dp/pi_trace.json        — 3-iteration PI trace
  public/data/dp/vi_trace.json        — 5-iteration VI trace
  public/data/dp/sweep_comparison.json — iter counts for all sweep orders
"""
import json, os, math

GAMMA = 0.9
GRID = 3
N_S = GRID * GRID  # 9 states
N_A = 4            # Up, Right, Down, Left
PIT = 4            # idx(1,1)
GOAL = 8           # idx(2,2)
DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

def idx(r, c): return r * GRID + c
def rc(s): return divmod(s, GRID)

def build_gridworld(slippery=False, gamma=GAMMA):
    terminals = [s == PIT or s == GOAL for s in range(N_S)]
    enter_reward = lambda sp: 1 if sp == GOAL else (-1 if sp == PIT else 0)

    P = [[[0.0]*N_S for _ in range(N_A)] for _ in range(N_S)]
    r = [[0.0]*N_A for _ in range(N_S)]

    for s in range(N_S):
        row, col = rc(s)
        for a in range(N_A):
            if terminals[s]:
                P[s][a][s] = 1.0
                r[s][a] = 0.0
                continue
            dr, dc = DELTAS[a]
            nr, nc = row + dr, col + dc
            if not (0 <= nr < GRID and 0 <= nc < GRID):
                nr, nc = row, col

            if slippery:
                perps = [1, 3] if a in (0, 2) else [0, 2]
                for aa, prob in [(a, 0.8)] + [(p, 0.1) for p in perps]:
                    ddr, ddc = DELTAS[aa]
                    nnr, nnc = row + ddr, col + ddc
                    if not (0 <= nnr < GRID and 0 <= nnc < GRID):
                        nnr, nnc = row, col
                    P[s][a][idx(nnr, nnc)] += prob
            else:
                P[s][a][idx(nr, nc)] = 1.0

            r[s][a] = sum(P[s][a][sp] * enter_reward(sp) for sp in range(N_S))

    return dict(nS=N_S, nA=N_A, gamma=gamma, P=P, r=r, terminals=terminals)

def uniform_policy(nS, nA):
    return [[1/nA]*nA for _ in range(nS)]

def bellman_expectation(mdp, policy, V):
    Vn = [0.0]*mdp['nS']
    for s in range(mdp['nS']):
        if mdp['terminals'][s]: continue
        v = 0.0
        for a in range(mdp['nA']):
            qsa = mdp['r'][s][a]
            for sp in range(mdp['nS']):
                qsa += mdp['gamma'] * mdp['P'][s][a][sp] * V[sp]
            v += policy[s][a] * qsa
        Vn[s] = v
    return Vn

def bellman_optimality(mdp, V):
    Vn = [0.0]*mdp['nS']
    for s in range(mdp['nS']):
        if mdp['terminals'][s]: continue
        best = -math.inf
        for a in range(mdp['nA']):
            q = mdp['r'][s][a] + mdp['gamma'] * sum(mdp['P'][s][a][sp]*V[sp] for sp in range(mdp['nS']))
            best = max(best, q)
        Vn[s] = best
    return Vn

def q_from_v(mdp, V):
    Q = [[0.0]*mdp['nA'] for _ in range(mdp['nS'])]
    for s in range(mdp['nS']):
        if mdp['terminals'][s]: continue
        for a in range(mdp['nA']):
            Q[s][a] = mdp['r'][s][a] + mdp['gamma'] * sum(mdp['P'][s][a][sp]*V[sp] for sp in range(mdp['nS']))
    return Q

def policy_improvement(mdp, V):
    Q = q_from_v(mdp, V)
    pi = [[0.0]*mdp['nA'] for _ in range(mdp['nS'])]
    for s in range(mdp['nS']):
        if mdp['terminals'][s]:
            pi[s][0] = 1.0
            continue
        best_a, best_q = 0, -math.inf
        for a in range(mdp['nA']):
            if Q[s][a] > best_q:
                best_q, best_a = Q[s][a], a
        pi[s][best_a] = 1.0
    return pi

def policy_evaluation_exact(mdp, policy):
    # (I - gamma P^pi) V = R^pi, solved via Gauss elimination
    import numpy as np
    nS = mdp['nS']
    Ppi = [[sum(policy[s][a] * mdp['P'][s][a][sp] for a in range(mdp['nA'])) for sp in range(nS)] for s in range(nS)]
    Rpi = [sum(policy[s][a] * mdp['r'][s][a] for a in range(mdp['nA'])) for s in range(nS)]
    A = np.eye(nS) - mdp['gamma'] * np.array(Ppi)
    V = np.linalg.solve(A, Rpi).tolist()
    for s in range(nS):
        if mdp['terminals'][s]: V[s] = 0.0
    return V

def sup_dist(V1, V2):
    return max(abs(a - b) for a, b in zip(V1, V2))

def policies_equal(pi1, pi2):
    for s in range(len(pi1)):
        for a in range(len(pi1[s])):
            if abs(pi1[s][a] - pi2[s][a]) > 1e-12: return False
    return True

# ── Policy Iteration ─────────────────────────────────────────────────────────
def policy_iteration(mdp):
    policy = uniform_policy(mdp['nS'], mdp['nA'])
    history = []
    for k in range(100):
        V = policy_evaluation_exact(mdp, policy)
        history.append({'V': [round(v, 6) for v in V], 'policy': [list(row) for row in policy]})
        new_policy = policy_improvement(mdp, V)
        if policies_equal(policy, new_policy):
            return {'iterations': k+1, 'history': history, 'V': [round(v, 6) for v in V]}
        policy = new_policy
    raise RuntimeError('PI did not converge')

# ── Value Iteration ──────────────────────────────────────────────────────────
def value_iteration(mdp, epsilon=1e-8):
    V = [0.0]*mdp['nS']
    trace = [[round(v, 6) for v in V]]
    stop_thr = epsilon * (1 - mdp['gamma']) / mdp['gamma']
    for k in range(10000):
        Vn = bellman_optimality(mdp, V)
        trace.append([round(v, 6) for v in Vn])
        if sup_dist(Vn, V) < stop_thr:
            return {'iterations': k+1, 'trace': trace, 'V': [round(v, 6) for v in Vn]}
        V = Vn
    raise RuntimeError('VI did not converge')

# ── Async VI ─────────────────────────────────────────────────────────────────
def async_vi(mdp, sweep_order, epsilon=1e-8):
    V = [0.0]*mdp['nS']
    stop_thr = epsilon * (1 - mdp['gamma']) / mdp['gamma']
    for k in range(10000):
        max_delta = 0.0
        for s in sweep_order:
            if mdp['terminals'][s]: continue
            old = V[s]
            best = max(mdp['r'][s][a] + mdp['gamma']*sum(mdp['P'][s][a][sp]*V[sp] for sp in range(mdp['nS']))
                       for a in range(mdp['nA']))
            V[s] = best
            max_delta = max(max_delta, abs(V[s] - old))
        if max_delta < stop_thr:
            return k + 1
    raise RuntimeError('Async VI did not converge')

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs('../public/data/dp', exist_ok=True)
    mdp = build_gridworld()

    # PI trace
    pi = policy_iteration(mdp)
    print(f"PI: {pi['iterations']} iterations, V(0,0) sequence: " +
          str([h['V'][idx(0,0)] for h in pi['history']]))
    with open('../public/data/dp/pi_trace.json', 'w') as f:
        json.dump(pi, f, indent=2)

    # VI trace
    vi = value_iteration(mdp)
    print(f"VI: {vi['iterations']} iterations")
    with open('../public/data/dp/vi_trace.json', 'w') as f:
        json.dump(vi, f, indent=2)

    # Sweep comparison
    forward = [0,1,2,3,4,5,6,7,8]
    reverse = [8,7,6,5,4,3,2,1,0]
    smart   = sorted(range(9), key=lambda s: abs(rc(s)[0]-2)+abs(rc(s)[1]-2))
    jacobi_iters = vi['iterations']  # synchronous = jacobi
    fwd_iters = async_vi(mdp, forward)
    rev_iters = async_vi(mdp, reverse)
    sma_iters = async_vi(mdp, smart)
    print(f"Sweep comparison: Jacobi={jacobi_iters}, Forward={fwd_iters}, Reverse={rev_iters}, Smart={sma_iters}")
    sweep = {
        'jacobi':  {'iterations': jacobi_iters, 'order': forward},
        'forward': {'iterations': fwd_iters, 'order': forward},
        'reverse': {'iterations': rev_iters, 'order': reverse},
        'smart':   {'iterations': sma_iters, 'order': smart},
    }
    with open('../public/data/dp/sweep_comparison.json', 'w') as f:
        json.dump(sweep, f, indent=2)

    print("Done. Files written to public/data/dp/")

if __name__ == '__main__':
    main()
