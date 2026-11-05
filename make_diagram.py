"""Draws the agent-environment workflow diagram used in the report (results/fig_architecture.png)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, ax = plt.subplots(figsize=(9, 2.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 36); ax.axis("off")

def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fc, ec="#333", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)

def arrow(x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#333"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.3, label, ha="center", fontsize=7.5, style="italic")

box(1, 10, 15, 10, "Poisson vehicle\narrivals\n(peak-hour demand)", "#fde7c8")
box(22, 6, 22, 18, "Intersection simulator\n4 lane queues, 2 phases\nyellow interval,\nmin-green constraint\nsaturation flow = 2 veh/step", "#d9e8f5")
box(52, 18, 19, 9, "State encoder\n(NS level, EW level,\nphase, time-in-phase)", "#e3f1dc")
box(52, 3, 19, 9, "Reward\nr = -(total queue)", "#f6dada")
box(78, 6, 21, 18, "RL agent\nQ-table 288 x 2\nepsilon-greedy\nQ-Learning / SARSA\nTD update", "#ece0f5")

arrow(16, 15, 22, 15, "arrivals")
arrow(44, 20, 52, 22.5)
arrow(44, 10, 52, 7.5)
arrow(71, 22.5, 78, 19, "s'")
arrow(71, 7.5, 78, 11, "r")
ax.annotate("", xy=(33, 24.4), xytext=(88.5, 24.4),
            arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#333",
                            connectionstyle="arc3,rad=0.18"))
ax.text(61, 31.6, "action a: keep phase / switch phase", ha="center", fontsize=7.5, style="italic")
fig.tight_layout()
fig.savefig("results/fig_architecture.png", dpi=220, bbox_inches="tight")
