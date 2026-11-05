"""Demonstration: trained Q-Learning agent vs fixed-time controller on the same traffic.
Prints the signal and queues for a short window, then the full-hour metrics.
Run `python train.py` first (it saves results/q_table_qlearning.npy)."""
import numpy as np
from traffic_env import TrafficIntersectionEnv
from agents import QLearningAgent, FixedTimeController

env0 = TrafficIntersectionEnv()
agent = QLearningAgent(env0.n_states, env0.N_ACTIONS)
agent.Q = np.load("results/q_table_qlearning.npy")
fixed = FixedTimeController(cycle=6)


def run(name, use_agent, show=(280, 310)):
    env = TrafficIntersectionEnv(seed=2026)
    s, done = env.reset(), False
    print(f"\n=== {name} ===  (steps {show[0]}-{show[1]}, 5 s each)")
    while not done:
        a = agent.act(s, greedy=True) if use_agent else fixed.act_env(env)
        s, _, done = env.step(a)
        if show[0] <= env.t < show[1]:
            light = "NS-GREEN" if env.phase == 0 else "EW-GREEN"
            if env.yellow: light = "YELLOW  "
            q = env.queues
            print(f"t={env.t:3d} {light}  N:{'#'*q[0]:<12}S:{'#'*q[1]:<12}"
                  f"E:{'#'*q[2]:<10}W:{'#'*q[3]:<10}")
    m = env.metrics()
    print(f"-> avg wait {m['avg_wait_s']:.1f} s | avg queue {m['avg_queue']:.2f} | "
          f"served {m['throughput']} | switches {m['switches']}")
    return m


f = run("Fixed-Time controller", False)
q = run("Q-Learning agent", True)
print(f"\nQ-Learning reduces waiting time by "
      f"{100*(f['avg_wait_s']-q['avg_wait_s'])/f['avg_wait_s']:.1f}% on this episode.")
