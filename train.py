"""
Train Q-Learning and SARSA traffic-signal agents, evaluate against baselines,
and save all figures/results to ./results.

Usage:  python train.py            (full run, ~1-2 min on a laptop)
        python train.py --quick    (fewer episodes / seeds for a fast demo)
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from traffic_env import TrafficIntersectionEnv
from agents import (QLearningAgent, SarsaAgent, FixedTimeController,
                    LongestQueueController, RandomController)

OUT = "results"
HP = dict(alpha=0.1, gamma=0.95, eps_start=1.0, eps_end=0.02, eps_decay=0.97)


def train(agent_cls, episodes, seed):
    env = TrafficIntersectionEnv(seed=seed)
    agent = agent_cls(env.n_states, env.N_ACTIONS, seed=seed, **HP)
    returns, waits = [], []
    for ep in range(episodes):
        s = env.reset()
        a = agent.act(s)
        done, G = False, 0
        while not done:
            s2, r, done = env.step(a)
            a2 = agent.act(s2)
            agent.update(s, a, r / 10.0, s2, a2, done)   # reward scaled for stability
            s, a, G = s2, a2, G + r
        agent.decay()
        returns.append(G)
        waits.append(env.metrics()["avg_wait_s"])
    return agent, np.array(returns), np.array(waits)


def evaluate(policy, n_eps, seed, rl=False, profile="peak", trace=False):
    res, q_trace = [], None
    for i in range(n_eps):
        env = TrafficIntersectionEnv(seed=10_000 + seed * 100 + i, demand_profile=profile)
        s, done, tr = env.reset(), False, []
        while not done:
            a = policy.act(s, greedy=True) if rl else policy.act_env(env)
            s, _, done = env.step(a)
            tr.append(env.queues.sum())
        res.append(env.metrics())
        if trace and i == 0:
            q_trace = np.array(tr)
    agg = {k: float(np.mean([m[k] for m in res])) for k in res[0]}
    agg_sd = {k + "_sd": float(np.std([m[k] for m in res])) for k in res[0]}
    return {**agg, **agg_sd}, q_trace


def smooth(x, w=10):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main(quick=False):
    os.makedirs(OUT, exist_ok=True)
    episodes = 60 if quick else 200
    seeds = [0, 1] if quick else [0, 1, 2, 3, 4]
    eval_eps = 5 if quick else 20

    curves = {"Q-Learning": [], "SARSA": []}
    wait_curves = {"Q-Learning": [], "SARSA": []}
    agents = {}
    for name, cls in [("Q-Learning", QLearningAgent), ("SARSA", SarsaAgent)]:
        for sd in seeds:
            ag, R, W = train(cls, episodes, sd)
            curves[name].append(R); wait_curves[name].append(W)
            if sd == seeds[0]:
                agents[name] = ag
        print(f"trained {name}")

    # ---- evaluation (unseen seeds), including a hyper-parameter-free uniform-demand test
    ctrls = {"Random": (RandomController(), False),
             "Fixed-Time": (FixedTimeController(cycle=6), False),
             "Longest-Queue": (LongestQueueController(), False),
             "SARSA": (agents["SARSA"], True),
             "Q-Learning": (agents["Q-Learning"], True)}
    results, traces, uniform = {}, {}, {}
    for name, (pol, rl) in ctrls.items():
        results[name], traces[name] = evaluate(pol, eval_eps, 7, rl=rl, trace=True)
        uniform[name], _ = evaluate(pol, eval_eps, 8, rl=rl, profile="uniform")
        print(f"{name:14s} wait={results[name]['avg_wait_s']:.1f}s "
              f"queue={results[name]['avg_queue']:.2f}")

    # ---- Fig 1: learning curves (mean +/- std over seeds)
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    for name, c in [("Q-Learning", "#1f77b4"), ("SARSA", "#d62728")]:
        R = np.array(curves[name]); W = np.array(wait_curves[name])
        for k, (arr, a_) in enumerate([(R, ax[0]), (W, ax[1])]):
            m = smooth(arr.mean(0)); sdv = smooth(arr.std(0))
            x = np.arange(len(m)) + 10
            a_.plot(x, m, color=c, label=name)
            a_.fill_between(x, m - sdv, m + sdv, color=c, alpha=0.18)
    ft = results["Fixed-Time"]["avg_wait_s"]
    ax[1].axhline(ft, ls="--", color="gray", label="Fixed-Time baseline")
    ax[0].set(xlabel="Episode", ylabel="Episode return (sum of rewards)",
              title="(a) Training reward")
    ax[1].set(xlabel="Episode", ylabel="Avg. waiting time per vehicle (s)",
              title="(b) Waiting time during training")
    ax[1].set_ylim(0, min(np.array(wait_curves["SARSA"]).mean(0)[:10].mean(), 120))
    for a_ in ax: a_.grid(alpha=0.3); a_.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_training.png", dpi=200); plt.close(fig)

    # ---- Fig 2: bar comparison + queue trace
    names = list(ctrls)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    cols = ["#9e9e9e", "#ff9f40", "#59a14f", "#d62728", "#1f77b4"]
    vals = [results[n]["avg_wait_s"] for n in names]
    errs = [results[n]["avg_wait_s_sd"] for n in names]
    ax[0].bar(names, vals, yerr=errs, color=cols, capsize=3)
    for i, v in enumerate(vals):
        ax[0].text(i, v + errs[i] + 0.4, f"{v:.1f}", ha="center", fontsize=8)
    ax[0].set(ylabel="Avg. waiting time per vehicle (s)", title="(a) Controller comparison")
    ax[0].tick_params(axis="x", rotation=20, labelsize=8)
    for n, c in zip(names[1:], cols[1:]):
        ax[1].plot(smooth(traces[n], 15), label=n, color=c, lw=1.2)
    ax[1].set(xlabel="Time step (5 s each)", ylabel="Vehicles queued",
              title="(b) Queue length over a peak-hour episode")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    for a_ in ax: a_.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_comparison.png", dpi=200); plt.close(fig)

    # ---- Fig 3: learned policy heat-map (Q-Learning, phase NS-green, time-in-phase >= 10)
    Q = agents["Q-Learning"].Q
    pol = np.zeros((6, 6))
    for qn in range(6):
        for qe in range(6):
            s = ((qn * 6 + qe) * 2 + 0) * 4 + 2
            pol[qn, qe] = Q[s, 1] - Q[s, 0]
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    im = ax.imshow(pol, cmap="coolwarm", origin="lower")
    ax.set(xlabel="East-West queue level", ylabel="North-South queue level",
           title="Q(switch) - Q(keep), NS green, 4-7 steps")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_policy.png", dpi=200); plt.close(fig)

    json.dump({"peak": results, "uniform": uniform, "hyperparameters": HP,
               "episodes": episodes, "seeds": seeds, "eval_episodes": eval_eps},
              open(f"{OUT}/results.json", "w"), indent=2)
    np.save(f"{OUT}/q_table_qlearning.npy", agents["Q-Learning"].Q)
    print("saved to", OUT)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    main(p.parse_args().quick)
