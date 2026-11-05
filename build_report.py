"""Builds the 4-page project report (Report.pdf) from results/results.json and figures."""
import json
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table,
                                TableStyle, KeepTogether, PageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping

# Liberation Serif is metric-compatible with Times New Roman
FD = "/usr/share/fonts/truetype/liberation/"
for name, f in [("TNR", "LiberationSerif-Regular.ttf"), ("TNR-B", "LiberationSerif-Bold.ttf"),
                ("TNR-I", "LiberationSerif-Italic.ttf"), ("TNR-BI", "LiberationSerif-BoldItalic.ttf")]:
    pdfmetrics.registerFont(TTFont(name, FD + f))
addMapping("TNR", 0, 0, "TNR"); addMapping("TNR", 1, 0, "TNR-B")
addMapping("TNR", 0, 1, "TNR-I"); addMapping("TNR", 1, 1, "TNR-BI")

D = json.load(open("results/results.json"))
P, U = D["peak"], D["uniform"]

body = ParagraphStyle("body", fontName="TNR", fontSize=12, leading=18, alignment=TA_JUSTIFY,
                      spaceAfter=4)
h1 = ParagraphStyle("h1", fontName="TNR-B", fontSize=14, leading=18, spaceBefore=6, spaceAfter=2, keepWithNext=1)
title = ParagraphStyle("t", fontName="TNR-B", fontSize=16, leading=21, alignment=TA_CENTER)
meta = ParagraphStyle("m", parent=body, alignment=TA_CENTER, spaceAfter=0, leading=16)
cap = ParagraphStyle("cap", fontName="TNR", fontSize=10, leading=13, alignment=TA_CENTER,
                     spaceAfter=4)
ref = ParagraphStyle("ref", fontName="TNR", fontSize=11, leading=15, leftIndent=22,
                     firstLineIndent=-22, alignment=TA_JUSTIFY, spaceAfter=2)
cell = ParagraphStyle("cell", fontName="TNR", fontSize=10, leading=12)

W = A4[0] - 2 * inch


def para(t): return Paragraph(t, body)
def fig(path, h, caption):
    return KeepTogether([Image(path, width=W, height=h), Paragraph(caption, cap)])


def table(rows, widths, caption):
    t = Table([[Paragraph(str(c), cell) for c in r] for r in rows], colWidths=widths)
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "TNR"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return KeepTogether([Paragraph(caption, cap), t, Spacer(1, 4)])


def imp(a, b): return 100 * (P[b]["avg_wait_s"] - P[a]["avg_wait_s"]) / P[b]["avg_wait_s"]

ql, sa, ft, lq, rnd = (P[k] for k in ["Q-Learning", "SARSA", "Fixed-Time", "Longest-Queue", "Random"])

s = []
s += [Paragraph("Adaptive Traffic Signal Control using Tabular Reinforcement Learning: "
                "A Comparative Study of Q-Learning and SARSA", title), Spacer(1, 4)]
s += [Paragraph("<b>Student:</b> Daksh Gupta &nbsp;&nbsp;|&nbsp;&nbsp; <b>Roll No.:</b> ____________", meta),
      Paragraph("<b>Course:</b> Reinforcement Learning, Final Year B.Tech (CSE), "
                "Manipal University Jaipur", meta),
      Paragraph("<b>Faculty:</b> ____________ &nbsp;&nbsp;|&nbsp;&nbsp; <b>Category:</b> Mini Project", meta),
      Spacer(1, 6)]

s += [Paragraph("Abstract", h1), para(
    "Fixed-time traffic signals ignore real-time demand, which causes avoidable queues and delay "
    "at urban intersections, especially during peak hours. This project formulates signal control "
    "at an isolated four-way intersection as a Markov Decision Process and trains two tabular "
    "temporal-difference agents, Q-Learning (off-policy) and SARSA (on-policy), to decide at each "
    "5-second step whether to keep or switch the current green phase. A custom Python simulator "
    "models Poisson vehicle arrivals with a time-varying peak-hour profile, saturation-flow "
    "discharge, a yellow interval and a minimum-green safety constraint. The reward is the "
    "negative total queue length. Agents were trained for 200 one-hour episodes over five random "
    f"seeds and evaluated on unseen traffic against random, fixed-time and greedy longest-queue "
    f"controllers. Q-Learning reduced average waiting time per vehicle to {ql['avg_wait_s']:.1f} s, "
    f"a {imp('Q-Learning','Fixed-Time'):.0f}% reduction over the fixed-time controller "
    f"({ft['avg_wait_s']:.1f} s) and {imp('Q-Learning','Longest-Queue'):.0f}% over the greedy "
    f"actuated controller ({lq['avg_wait_s']:.1f} s), with {lq['switches']-ql['switches']:.0f} fewer "
    "phase changes per hour than the greedy controller. SARSA performed comparably. The results show "
    "that even compact tabular RL can learn demand-responsive signal timing.")]

s += [Paragraph("1. Introduction", h1), para(
    "<b>Background and motivation.</b> Traffic congestion wastes fuel, increases emissions and "
    "lengthens commutes in Indian cities. Most signals still run pre-timed plans derived from "
    "historical counts [7], which cannot react when demand shifts within an hour. Reinforcement "
    "learning (RL) is attractive because a controller can learn a timing policy from interaction "
    "with traffic, without a hand-built model of the flow [4], [9]."), para(
    "<b>Problem statement.</b> Given the current queues and signal state at an intersection, choose "
    "when to switch the green phase so that the total vehicle delay is minimised, while respecting "
    "a minimum green time and the lost time of each yellow interval."), para(
    "<b>Objectives.</b> (i) Model the intersection as an MDP and build a reproducible simulator; "
    "(ii) implement Q-Learning and SARSA from scratch; (iii) compare them with three baselines on "
    "waiting time, queue length, throughput and switching frequency; (iv) analyse convergence and "
    "limitations.")]

s += [Paragraph("2. Methodology", h1), para(
    "<b>Environment.</b> One four-way intersection with lanes N, S, E and W and two phases "
    "(phase 0: North-South green; phase 1: East-West green). Each step represents 5 s, so an "
    "episode of 720 steps is one hour. Arrivals per lane are Poisson with base rates of 0.45 "
    "(N/S) and 0.25 (E/W) vehicles/step, multiplied by Gaussian peaks so that North-South demand "
    "rises by 60% around 40% of the hour and East-West demand doubles near the end (tidal flow). "
    "A green lane discharges up to 2 vehicles per step. Switching phase triggers one all-red/yellow "
    "step with no discharge, and a switch is only allowed after 2 steps (10 s) of green."), para(
    "<b>State space.</b> The state is the tuple (NS-queue level, EW-queue level, phase, "
    "time-in-phase level). Directional queues are binned at {0, 1-2, 3-4, 5-7, 8-12, 13+} vehicles "
    "and time-in-phase at {0-1, 2-3, 4-7, 8+} steps, giving 6 x 6 x 2 x 4 = 288 discrete states. "
    "<b>Action space:</b> {0 = keep phase, 1 = switch phase}. "
    "<b>Reward:</b> r<sub>t</sub> = -(sum of queued vehicles after step t), scaled by 1/10 during "
    "learning. By Little's law, minimising the time-integral of queue length minimises total delay."),
    para(
    "<b>Algorithms.</b> Both agents keep a Q-table and act epsilon-greedily. Q-Learning [2] uses the "
    "off-policy target r + gamma max<sub>a'</sub> Q(s',a'), while SARSA [3] uses the on-policy "
    "target r + gamma Q(s',a') with a' the action actually taken next. The update is "
    "Q(s,a) &larr; Q(s,a) + alpha[target - Q(s,a)] [1]. Ties are broken randomly.")]

hp = D["hyperparameters"]
s += [table([["Hyperparameter", "Value", "Hyperparameter", "Value"],
             ["Learning rate alpha", hp["alpha"], "Discount gamma", hp["gamma"]],
             ["Initial epsilon", hp["eps_start"], "Final epsilon", hp["eps_end"]],
             ["Epsilon decay / episode", hp["eps_decay"], "Training episodes", D["episodes"]],
             ["Training seeds", len(D["seeds"]), "Evaluation episodes", D["eval_episodes"]]],
            [1.75 * inch, 0.9 * inch, 1.75 * inch, 0.9 * inch],
            "Table 1. Hyperparameters used for both agents.")]

s += [para(
    "<b>Training strategy and baselines.</b> Each agent is trained for 200 episodes on five seeds; "
    "learning curves report the mean and standard deviation across seeds. The greedy policy is then "
    "evaluated on 20 unseen episodes. Baselines: <i>Random</i> (switch with probability 0.5), "
    "<i>Fixed-Time</i> (switch every 6 steps, i.e. 30 s green, a Webster-style plan [7]) and "
    "<i>Longest-Queue</i> (an actuated rule that always serves the longer direction, related to "
    "max-pressure control [8]). Metrics are average waiting time per vehicle (queue-time integral "
    "divided by vehicles served), average queue length, throughput and number of phase switches."),
    para(
    "<b>Software framework and architecture.</b> Python 3 with NumPy for the simulator and agents "
    "and Matplotlib for plots; no external RL library is used, so every update is transparent. "
    "Fig. 1 shows the agent-environment loop.")]
s += [fig("results/fig_architecture.png", 1.95 * inch,
          "Fig. 1. System architecture and agent-environment interaction loop.")]

s += [Paragraph("3. Experimental Results", h1), para(
    "<b>Training behaviour.</b> Fig. 2 shows that both agents improve rapidly during the first "
    "50 episodes as epsilon decays and then stabilise. Waiting time during training falls below the "
    "fixed-time baseline after about 25 episodes and settles near 10 s. The narrow shaded bands "
    "show that convergence is consistent across seeds.")]
s += [fig("results/fig_training.png", 2.25 * inch,
          "Fig. 2. Learning curves (mean +/- std over 5 seeds, 10-episode moving average): "
          "(a) episode return, (b) average waiting time per vehicle.")]

rows = [["Controller", "Avg. wait (s) peak", "Avg. queue (veh)", "Throughput (veh/h)",
         "Switches /h", "Avg. wait (s) uniform"]]
for n in ["Random", "Fixed-Time", "Longest-Queue", "SARSA", "Q-Learning"]:
    rows.append([f"<b>{n}</b>" if n == "Q-Learning" else n,
                 f"{P[n]['avg_wait_s']:.2f} &plusmn; {P[n]['avg_wait_s_sd']:.2f}",
                 f"{P[n]['avg_queue']:.2f}", f"{P[n]['throughput']:.0f}",
                 f"{P[n]['switches']:.0f}", f"{U[n]['avg_wait_s']:.2f}"])
s += [table(rows, [1.15 * inch, 1.2 * inch, 0.9 * inch, 0.95 * inch, 0.75 * inch, 1.05 * inch],
            "Table 2. Evaluation on 20 unseen episodes (mean &plusmn; std). "
            "Uniform = constant demand without peaks (generalisation test).")]

s += [fig("results/fig_comparison.png", 2.25 * inch,
          "Fig. 3. (a) Average waiting time per controller under peak-hour demand; "
          "(b) total queue during one evaluation episode (15-step moving average).")]

s += [para(
    f"<b>Performance analysis.</b> Table 2 and Fig. 3 show that both RL agents outperform all "
    f"baselines. Q-Learning achieved {ql['avg_wait_s']:.2f} s average waiting time, "
    f"{imp('Q-Learning','Fixed-Time'):.1f}% lower than Fixed-Time and "
    f"{imp('Q-Learning','Longest-Queue'):.1f}% lower than Longest-Queue; SARSA reached "
    f"{sa['avg_wait_s']:.2f} s. Throughput is nearly identical for all controllers because demand "
    "is below capacity, so the gain comes from serving vehicles sooner rather than serving more. "
    "Fig. 3(b) explains the gap: the fixed-time plan cannot react to the North-South surge around "
    "step 300 and its queue climbs above 20 vehicles, while the RL agents keep it near 5. The "
    "greedy controller reacts but switches too often "
    f"({lq['switches']:.0f} vs {ql['switches']:.0f} switches/h), wasting a yellow interval each "
    "time; the RL agents learned to hold green slightly longer until the opposing queue justifies "
    "the lost time. On constant (uniform) demand, never seen in training, the agents still beat "
    "every baseline, indicating that the policy depends on queue state rather than memorising the "
    "peak profile. Q-Learning and SARSA converge to similar policies here because the "
    "minimum-green rule removes the risky exploratory actions where SARSA's on-policy caution "
    "usually matters; Q-Learning showed slightly lower variance across episodes.")]

s += [Paragraph("4. Conclusion", h1), para(
    "Tabular Q-Learning and SARSA learned adaptive signal timing that cut average waiting time by "
    "about 42% compared with a fixed-time plan and about 20% compared with a greedy actuated rule, "
    "using a 288-state table that trains in about a minute on a laptop."), para(
    "<b>Limitations.</b> The simulator is a single isolated intersection with two phases, no "
    "turning movements, pedestrians or vehicle dynamics. The hand-crafted queue bins limit "
    "resolution, and tabular methods do not scale to large networks or continuous sensor inputs. "
    "Results are from simulation and are not yet validated on real traffic data."), para(
    "<b>Future scope.</b> (i) Replace the Q-table with a Deep Q-Network [5] or Double/Dueling DQN to "
    "use raw lane occupancy; (ii) move to the SUMO microscopic simulator with real Jaipur junction "
    "counts; (iii) multi-agent coordination across corridors for green waves [6], [9]; "
    "(iv) add pedestrian phases and emergency-vehicle priority to the reward.")]

s += [Paragraph("References", h1)]
refs = [
    'R. S. Sutton and A. G. Barto, <i>Reinforcement Learning: An Introduction</i>, 2nd ed. Cambridge, MA, USA: MIT Press, 2018.',
    'C. J. C. H. Watkins and P. Dayan, "Q-learning," <i>Machine Learning</i>, vol. 8, no. 3-4, pp. 279-292, 1992.',
    'G. A. Rummery and M. Niranjan, "On-line Q-learning using connectionist systems," Cambridge Univ. Eng. Dept., Tech. Rep. CUED/F-INFENG/TR 166, 1994.',
    'B. Abdulhai, R. Pringle, and G. J. Karakoulas, "Reinforcement learning for true adaptive traffic signal control," <i>J. Transp. Eng.</i>, vol. 129, no. 3, pp. 278-285, 2003.',
    'V. Mnih <i>et al.</i>, "Human-level control through deep reinforcement learning," <i>Nature</i>, vol. 518, pp. 529-533, 2015.',
    'H. Wei, G. Zheng, H. Yao, and Z. Li, "IntelliLight: A reinforcement learning approach for intelligent traffic light control," in <i>Proc. ACM SIGKDD</i>, 2018, pp. 2496-2505.',
    'F. V. Webster, "Traffic signal settings," Road Research Laboratory, London, U.K., Tech. Paper 39, 1958.',
    'P. Varaiya, "Max pressure control of a network of signalized intersections," <i>Transp. Res. Part C</i>, vol. 36, pp. 177-195, 2013.',
    'H. Wei, G. Zheng, V. Gayah, and Z. Li, "A survey on traffic signal control methods," arXiv:1904.08117, 2019.',
]
s += [Paragraph(f"[{i}] {r}", ref) for i, r in enumerate(refs, 1)]

doc = SimpleDocTemplate("Report.pdf", pagesize=A4, leftMargin=inch, rightMargin=inch,
                        topMargin=inch, bottomMargin=inch,
                        title="Adaptive Traffic Signal Control using Tabular RL", author="Daksh Gupta")
doc.build(s)
