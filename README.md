# Adaptive Traffic Signal Control using Q-Learning and SARSA

Mini Project – Reinforcement Learning (Final Year B.Tech CSE)
Author: Daksh Gupta

An RL agent learns when to switch the green light at a four-way intersection so that vehicles wait as little as possible. Q-Learning and SARSA are implemented from scratch and compared against Random, Fixed-Time and Longest-Queue controllers.

## Results (20 unseen peak-hour episodes)

| Controller    | Avg. wait (s) | Avg. queue | Switches/h |
|---------------|---------------|------------|------------|
| Random        | 43.2          | 13.9       | 240        |
| Fixed-Time    | 16.8          | 5.4        | 119        |
| Longest-Queue | 12.1          | 3.9        | 280        |
| SARSA         | 9.9           | 3.2        | 229        |
| **Q-Learning**| **9.7**       | **3.1**    | 221        |

Q-Learning cuts waiting time by ~42% vs fixed-time and ~20% vs the greedy actuated controller.

## Files

| File | Purpose |
|------|---------|
| `traffic_env.py` | Intersection simulator (MDP: states, actions, reward, peak-hour demand) |
| `agents.py` | Q-Learning, SARSA, and baseline controllers |
| `train.py` | Trains agents (5 seeds × 200 episodes), evaluates, saves plots + `results.json` |
| `make_diagram.py` | Draws the architecture diagram |
| `build_report.py` | Generates `Report.pdf` from the results |
| `demo.py` | Live text demo: watch the trained agent control the signal |
| `results/` | Figures, `results.json`, saved Q-table |

## How to run

```bash
pip install numpy matplotlib reportlab
python train.py          # full experiment, ~1 min  (use --quick for ~15 s)
python demo.py           # demonstration of the trained agent vs fixed-time
python make_diagram.py   # optional: redraw architecture figure
python build_report.py   # optional: rebuild Report.pdf
```

## MDP design

- **State (288 states):** NS queue level (6) × EW queue level (6) × current phase (2) × time-in-phase level (4)
- **Actions:** 0 = keep phase, 1 = switch phase (allowed only after 10 s minimum green; switch costs one 5 s yellow step)
- **Reward:** −(total vehicles queued) each step
- **Hyperparameters:** α = 0.1, γ = 0.95, ε decays 1.0 → 0.02 (×0.97 per episode)

## Viva quick notes

- **Q-Learning vs SARSA:** Q-Learning is off-policy (target uses max over next actions); SARSA is on-policy (target uses the next action actually taken). They converge to similar policies here because the min-green rule removes risky exploratory actions.
- **Why RL beats Longest-Queue:** the greedy rule switches too often and wastes a yellow interval each time; the agent learned to hold green until the opposing queue justifies the lost time.
- **Why throughput is equal:** demand is below capacity, so every controller eventually serves all cars; RL serves them *sooner*.
- **Waiting time metric:** Little's law — mean delay = (time-integral of queue) / vehicles served.
# TRAFFIC-SIGNAL-PROJECT
