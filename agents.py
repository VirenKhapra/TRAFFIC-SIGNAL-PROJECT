"""Tabular RL agents (Q-Learning, SARSA) and rule-based baseline controllers."""
import numpy as np


class TabularAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95,
                 eps_start=1.0, eps_end=0.02, eps_decay=0.97, seed=0):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha, self.gamma = alpha, gamma
        self.eps, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def act(self, s, greedy=False):
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(self.n_actions))
        q = self.Q[s]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    def decay(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)


class QLearningAgent(TabularAgent):
    """Off-policy TD control: target uses max_a' Q(s', a')."""
    def update(self, s, a, r, s2, a2, done):
        target = r if done else r + self.gamma * self.Q[s2].max()
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])


class SarsaAgent(TabularAgent):
    """On-policy TD control: target uses Q(s', a') of the action actually taken."""
    def update(self, s, a, r, s2, a2, done):
        target = r if done else r + self.gamma * self.Q[s2, a2]
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])


# ---------------------------------------------------------------- baselines
class FixedTimeController:
    """Classic pre-timed signal: switch every `cycle` steps regardless of traffic."""
    def __init__(self, cycle=6):
        self.cycle = cycle

    def act_env(self, env):
        return 1 if env.time_in_phase >= self.cycle else 0


class LongestQueueController:
    """Actuated (greedy) controller: give green to the direction with the longer queue."""
    def act_env(self, env):
        ns = env.queues[0] + env.queues[1]
        ew = env.queues[2] + env.queues[3]
        want = 0 if ns >= ew else 1
        return 1 if want != env.phase else 0


class RandomController:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act_env(self, env):
        return int(self.rng.integers(2))
