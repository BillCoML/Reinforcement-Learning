#!/usr/bin/env python3
"""
DQN training pipeline for Lesson 9: Function Approximation & DQN.

Trains four ablation configs × 5 seeds on a 7×7 gridworld:
  - full:    DQN (target network + replay buffer)
  - target:  target network only, no replay
  - replay:  replay buffer only, no target network
  - naive:   standard Q-learning on neural net (no tricks)

Per seed, exports:
  public/data/dqn/<config>/<env>/seed<N>.json   — DQNTrace (loss, param norm, Q-traces, Q-grids)
  public/models/dqn/<config>/<env>/seed<N>/step<K>.onnx  — ONNX checkpoints

Usage:
  pip install torch onnx numpy
  python scripts/train_dqn.py [--configs all] [--seeds 5] [--episodes 800]
"""

import json
import math
import random
import argparse
from pathlib import Path
from collections import deque
from copy import deepcopy
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ── Output paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_OUT = ROOT / "public" / "data" / "dqn"
MODEL_OUT = ROOT / "public" / "models" / "dqn"

# ── 7×7 Gridworld ─────────────────────────────────────────────────────────────
GRID = 7
N_STATES = GRID * GRID  # 49
N_ACTIONS = 4            # up, right, down, left
GAMMA = 0.95

WALL_CELLS = {GRID * r + 3 for r in range(1, 6)}   # col 3, rows 1-5
GOAL_MAIN = GRID * 6 + 6   # (6,6) reward +1.0
GOAL_ALT  = GRID * 0 + 6   # (0,6) reward +0.5
TERMINALS = {GOAL_MAIN, GOAL_ALT}

DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left

def idx(r, c):
    return r * GRID + c

def rc(s):
    return divmod(s, GRID)

def step_env(s, a):
    """Take action a from state s; return (next_state, reward, done)."""
    if s in TERMINALS:
        return s, 0.0, True
    r, c = rc(s)
    dr, dc = DELTAS[a]
    nr, nc = r + dr, c + dc
    # Clip to grid bounds
    nr = max(0, min(GRID - 1, nr))
    nc = max(0, min(GRID - 1, nc))
    ns = idx(nr, nc)
    # Walls: bounce back
    if ns in WALL_CELLS:
        ns = s
    reward = 0.0
    done = False
    if ns == GOAL_MAIN:
        reward, done = 1.0, True
    elif ns == GOAL_ALT:
        reward, done = 0.5, True
    return ns, reward, done

def non_terminal_states():
    return [s for s in range(N_STATES) if s not in TERMINALS and s not in WALL_CELLS]

# ── Q-Network ─────────────────────────────────────────────────────────────────
class QNet(nn.Module):
    def __init__(self, n_states=N_STATES, n_actions=N_ACTIONS, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)

def one_hot(s, n=N_STATES):
    v = torch.zeros(n)
    v[s] = 1.0
    return v

# ── Replay Buffer ─────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buf.append((s, a, r, ns, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return list(s), list(a), list(r), list(ns), list(d)

    def __len__(self):
        return len(self.buf)

# ── Training ──────────────────────────────────────────────────────────────────
CHECKPOINT_INTERVAL = 100  # save ONNX + Q-grid every N episodes
TRACE_STATES = [idx(0, 0), idx(3, 0), idx(6, 0), idx(0, 3), idx(3, 6)]  # representative states

def param_norm(model):
    total = 0.0
    for p in model.parameters():
        total += p.data.norm(2).item() ** 2
    return math.sqrt(total)

def compute_q_grid(model):
    """Return flattened Q-grid [N_STATES × N_ACTIONS] row-major."""
    model.eval()
    with torch.no_grad():
        inputs = torch.eye(N_STATES)
        q = model(inputs).numpy()  # [49, 4]
    model.train()
    return q.flatten().tolist()

def compute_bellman_error(model, target_net_or_none):
    """Mean |Q(s,a) - (r + γ max_a' Q_target(s',a'))| across all non-terminal states."""
    tgt = target_net_or_none if target_net_or_none is not None else model
    model.eval()
    tgt.eval()
    errors = []
    with torch.no_grad():
        for s in non_terminal_states():
            s_oh = one_hot(s).unsqueeze(0)
            q_s = model(s_oh).squeeze(0)
            for a in range(N_ACTIONS):
                ns, r, done = step_env(s, a)
                if done:
                    target = r
                else:
                    ns_oh = one_hot(ns).unsqueeze(0)
                    target = r + GAMMA * tgt(ns_oh).max().item()
                errors.append(abs(q_s[a].item() - target))
    model.train()
    tgt.train()
    return float(np.mean(errors))

def train_config(config: str, seed: int, n_episodes: int) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    use_target = config in ("full", "target")
    use_replay = config in ("full", "replay")

    online = QNet()
    target_net = deepcopy(online) if use_target else None
    optimizer = optim.Adam(online.parameters(), lr=1e-3)
    replay = ReplayBuffer(10000) if use_replay else None

    target_update_freq = 10   # episodes between target net syncs
    batch_size = 64
    min_replay = 256
    epsilon = 0.15

    loss_per_step = []
    param_norm_per_step = []
    bellman_per_step = []
    q_traces = [[] for _ in TRACE_STATES]
    checkpoint_steps = []
    q_grids = []

    step = 0
    nts = non_terminal_states()

    for ep in range(n_episodes):
        s = random.choice(nts)
        done = False
        ep_losses = []

        while not done:
            # ε-greedy
            if random.random() < epsilon:
                a = random.randrange(N_ACTIONS)
            else:
                with torch.no_grad():
                    a = online(one_hot(s).unsqueeze(0)).argmax().item()

            ns, r, done = step_env(s, a)

            # Store transition
            if use_replay:
                replay.push(s, a, r, ns, done)
                if len(replay) < min_replay:
                    s = ns
                    step += 1
                    continue
                ss, aa, rr, nss, dd = replay.sample(batch_size)
            else:
                ss, aa, rr, nss, dd = [s], [a], [r], [ns], [done]

            # Build batch tensors
            s_batch = torch.stack([one_hot(si) for si in ss])
            ns_batch = torch.stack([one_hot(si) for si in nss])
            r_tensor = torch.tensor(rr, dtype=torch.float32)
            done_tensor = torch.tensor(dd, dtype=torch.float32)

            q_vals = online(s_batch)
            q_sa = q_vals.gather(1, torch.tensor(aa).unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                tgt_src = target_net if use_target else online
                q_next = tgt_src(ns_batch).max(1).values
                targets = r_tensor + GAMMA * q_next * (1 - done_tensor)

            loss = nn.functional.mse_loss(q_sa, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_losses.append(loss.item())
            loss_per_step.append(loss.item())
            param_norm_per_step.append(param_norm(online))
            bellman_per_step.append(compute_bellman_error(online, target_net))
            step += 1
            s = ns

        # Record Q-traces for representative states
        with torch.no_grad():
            for i, ts in enumerate(TRACE_STATES):
                q_ts = online(one_hot(ts).unsqueeze(0)).squeeze(0)
                q_traces[i].append(float(q_ts.max().item()))

        # Target net sync
        if use_target and ep % target_update_freq == 0:
            target_net.load_state_dict(online.state_dict())

        # Checkpoint
        if ep % CHECKPOINT_INTERVAL == 0 or ep == n_episodes - 1:
            checkpoint_steps.append(ep)
            q_grids.append(compute_q_grid(online))

    return {
        "config": config,
        "env": "gridworld7x7",
        "seed": seed,
        "nEpisodes": n_episodes,
        "lossPerStep": loss_per_step,
        "paramNormPerStep": param_norm_per_step,
        "bellmanErrorPerStep": bellman_per_step,
        "qTracesPerState": q_traces,
        "checkpointSteps": checkpoint_steps,
        "qGridPerCheckpoint": q_grids,
        "_checkpointModels": None,  # filled separately via ONNX export
    }, online

def export_onnx(model, config, env, seed, step):
    """Export QNet to ONNX (opset 17)."""
    import onnx  # noqa: F401 — verify onnx installed
    out_dir = MODEL_OUT / config / env / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"step{step}.onnx"
    dummy = torch.zeros(1, N_STATES)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        opset_version=17,
        do_constant_folding=False,
        input_names=["state"],
        output_names=["q_values"],
        dynamic_axes={"state": {0: "batch"}, "q_values": {0: "batch"}},
    )
    return path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="all", help="Comma-separated configs or 'all'")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--no-onnx", action="store_true", help="Skip ONNX export (faster, no torch.onnx needed)")
    args = parser.parse_args()

    if not HAS_TORCH:
        print("ERROR: PyTorch not installed. Run: pip install torch onnx numpy")
        return 1

    all_configs = ["full", "target", "replay", "naive"]
    configs = all_configs if args.configs == "all" else args.configs.split(",")

    print(f"Training {len(configs)} configs × {args.seeds} seeds × {args.episodes} episodes")
    print(f"Configs: {configs}")

    for config in configs:
        for seed in range(1, args.seeds + 1):
            print(f"  {config}/seed{seed}...", end="", flush=True)
            trace, model = train_config(config, seed, args.episodes)

            # Save JSON trace
            out_dir = DATA_OUT / config / "gridworld7x7"
            out_dir.mkdir(parents=True, exist_ok=True)
            trace_path = out_dir / f"seed{seed}.json"
            json_trace = {k: v for k, v in trace.items() if k != "_checkpointModels"}
            with open(trace_path, "w") as f:
                json.dump(json_trace, f, separators=(",", ":"))

            # Export ONNX checkpoints
            if not args.no_onnx:
                for step in trace["checkpointSteps"]:
                    export_onnx(model, config, "gridworld7x7", seed, step)

            print(f" done → {trace_path}")

    print("All done.")
    return 0

if __name__ == "__main__":
    exit(main())
