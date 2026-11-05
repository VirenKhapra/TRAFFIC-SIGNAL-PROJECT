"""
Single four-way intersection traffic-signal environment (discrete-time queue model).

Lanes: N, S, E, W. Two signal phases:
  phase 0 -> North-South green, East-West red
  phase 1 -> East-West green, North-South red

Each time step (= 5 s of real time):
  1. Vehicles arrive on each lane (Bernoulli/Poisson arrivals with lane-specific rates).
  2. If the agent switches phase, a 1-step yellow/all-red interval follows (no discharge).
  3. Lanes with green discharge up to `sat_flow` vehicles.

Reward = -(total vehicles queued) after the step, i.e. the agent minimises congestion.
"""
import numpy as np


class TrafficIntersectionEnv:
    N_ACTIONS = 2  # 0 = keep current phase, 1 = switch phase

    def __init__(self, arrival_rates=(0.45, 0.45, 0.25, 0.25), sat_flow=2,
                 max_queue=40, episode_len=720, min_green=2, demand_profile="peak",
                 seed=None):
        self.base_rates = np.array(arrival_rates, dtype=float)
        self.sat_flow = sat_flow
        self.max_queue = max_queue
        self.episode_len = episode_len          # 720 steps x 5 s = 1 hour
        self.min_green = min_green
        self.demand_profile = demand_profile
        self.rng = np.random.default_rng(seed)
        # Queue-length discretisation bins used for the tabular state
        self.q_bins = np.array([0, 1, 3, 5, 8, 13])    # -> 6 levels per approach (vehicles per direction)
        self.t_bins = np.array([0, 2, 4, 8])           # time-in-phase -> 4 levels
        self.n_states = 6 * 6 * 2 * 4
        self.reset()

    # ---------------------------------------------------------------- helpers
    def _rates(self):
        """Time-varying demand: a morning-peak shape where NS demand swells mid-episode
        and EW demand rises in the last third (tidal flow)."""
        r = self.base_rates.copy()
        if self.demand_profile == "peak":
            frac = self.t / self.episode_len
            ns_mult = 1.0 + 0.6 * np.exp(-((frac - 0.4) ** 2) / 0.02)
            ew_mult = 1.0 + 1.0 * np.exp(-((frac - 0.8) ** 2) / 0.01)
            r[:2] *= ns_mult
            r[2:] *= ew_mult
        return np.clip(r, 0, 0.95)

    def _bin(self, x, bins):
        return int(np.searchsorted(bins, x, side="right") - 1)

    def encode(self):
        ns = self.queues[0] + self.queues[1]
        ew = self.queues[2] + self.queues[3]
        qn = self._bin(ns, self.q_bins)
        qe = self._bin(ew, self.q_bins)
        tb = self._bin(self.time_in_phase, self.t_bins)
        return ((qn * 6 + qe) * 2 + self.phase) * 4 + tb

    # ---------------------------------------------------------------- API
    def reset(self):
        self.queues = np.zeros(4, dtype=int)
        self.phase = 0
        self.time_in_phase = 0
        self.yellow = 0
        self.t = 0
        self.served = 0
        self.arrived = 0
        self.queue_integral = 0
        self.switches = 0
        return self.encode()

    def step(self, action):
        # enforce minimum green time (safety constraint)
        if action == 1 and self.time_in_phase >= self.min_green and self.yellow == 0:
            self.phase = 1 - self.phase
            self.time_in_phase = 0
            self.yellow = 1
            self.switches += 1

        # arrivals
        arrivals = self.rng.poisson(self._rates())
        self.queues = np.minimum(self.queues + arrivals, self.max_queue)
        self.arrived += int(arrivals.sum())

        # discharge
        if self.yellow > 0:
            self.yellow -= 1
        else:
            green = [0, 1] if self.phase == 0 else [2, 3]
            for lane in green:
                out = min(self.queues[lane], self.sat_flow)
                self.queues[lane] -= out
                self.served += out

        self.time_in_phase += 1
        self.t += 1
        total_q = int(self.queues.sum())
        self.queue_integral += total_q
        reward = -total_q
        done = self.t >= self.episode_len
        return self.encode(), reward, done

    def metrics(self, step_seconds=5):
        avg_queue = self.queue_integral / self.t
        # Little's law: mean delay = mean queue / throughput rate
        avg_wait = (self.queue_integral * step_seconds) / max(self.served, 1)
        return {"avg_queue": avg_queue, "avg_wait_s": avg_wait,
                "throughput": self.served, "switches": self.switches}
