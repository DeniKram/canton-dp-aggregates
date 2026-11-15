"""Draw the article's figures from the experiment CSVs in code/results.

    python make_figures.py        # writes figures/fig1..fig5
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Polygon

ROOT = Path(__file__).parent
RES, OUT = ROOT.parent / "results", ROOT / "figures"
BLUE, GREY, INK, QUIET = "#2a78d6", "#bdbcb0", "#1f1f1f", "#666666"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def fig1_frontier() -> None:
    df = pd.read_csv(RES / "e2_epsilon_sweep.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for unit, colour, label in (("trade", GREY, "Hide one trade (Δ = 1M)"), ("dealer_day", BLUE, "Hide a dealer-day (Δ = 12M)")):
        g = df[df.unit == unit].sort_values("epsilon")
        ax.plot(g.median_rel_error * 100, g.victim_recovered * 100, "-o", color=colour, lw=2.5 if unit == "trade" else 2, ms=4, label=label)
        one = g[g.epsilon == 1.0].iloc[0]
        ax.plot(one.median_rel_error * 100, one.victim_recovered * 100, "o", color=colour, ms=10, mec=INK)
        ax.annotate(f"ε = 1: {one.median_rel_error*100:.1f}% error, {one.victim_recovered*100:.0f}% recovered",
                    (one.median_rel_error * 100, one.victim_recovered * 100), xytext=(10, 8), textcoords="offset points",
                    fontsize=9, fontweight="bold")
    ax.axhline(100, ls="--", color=QUIET, lw=1)
    ax.text(450, 96, "Exact totals: 0% error, 100% recovered", ha="right", va="top", color=QUIET, fontsize=9)
    ax.set(xscale="log", xlim=(0.25, 600), ylim=(0, 105), xlabel="Error of the published daily total (log scale)",
           ylabel="Victim days recovered within ±10%, %")
    ax.set_title("One curve while noise dominates; clipping puts a ~4% floor under trade-level protection",
                 loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, loc="center right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_frontier.png", dpi=160)
    plt.close(fig)


def fig2_market_size() -> None:
    df = pd.read_csv(RES / "e4_dealers.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    colours = [BLUE if d == 20 else GREY for d in df.dealers]
    bars = ax.bar([f"{d} dealers" for d in df.dealers], df.median_rel_error * 100, color=colours, width=0.55)
    for b, v in zip(bars, df.median_rel_error * 100):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title("Dealer-level privacy gets affordable as the market grows: 42% error with 3 dealers, 6% with 20\n"
                 "Median error of the published daily total, hiding a whole dealer-day, ε = 1",
                 loc="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_market_size.png", dpi=160)
    plt.close(fig)


def fig3_cap_table() -> None:
    df = pd.read_csv(RES / "e6_cap_table.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    styles = {"increments": (BLUE, "-", "Noise on each day's new shares"), "tree": (INK, "-", "Binary-tree mechanism"),
              "naive-split": (GREY, "--", "Noise split evenly over 64 days"),
              "naive-per-day": (GREY, ":", "Per-day noise, budget not counted")}
    for mech, (c, ls, label) in styles.items():
        g = df[df.mechanism == mech].sort_values("eps_actual")
        ax.plot(g.eps_actual, g.median_rel_error * 100, ls, marker="o", color=c, lw=2.5 if mech == "increments" else 1.8, label=label)
    i5 = df[(df.mechanism == "increments") & (df.eps_claimed == 5)].iloc[0]
    t5 = df[(df.mechanism == "tree") & (df.eps_claimed == 5)].iloc[0]
    s5 = df[(df.mechanism == "naive-split") & (df.eps_claimed == 5)].iloc[0]
    p5 = df[(df.mechanism == "naive-per-day") & (df.eps_claimed == 5)].iloc[0]
    ax.annotate(f"{t5.median_rel_error*100:.1f}%", (5, t5.median_rel_error * 100), xytext=(8, 4), textcoords="offset points", fontweight="bold")
    ax.annotate(f"{i5.median_rel_error*100:.1f}%", (5, i5.median_rel_error * 100), xytext=(-30, -16), textcoords="offset points",
                fontweight="bold", color=BLUE)
    ax.annotate(f"{s5.median_rel_error*100:.0f}%", (5, s5.median_rel_error * 100), xytext=(8, 6), textcoords="offset points", fontweight="bold")
    ax.annotate(f"claimed ε = 5, real {p5.eps_actual:.0f}:\naveraging recovers {p5.recovered_strong*100:.0f}% of grants",
                (p5.eps_actual, p5.median_rel_error * 100), xytext=(-170, -40), textcoords="offset points", fontsize=9,
                arrowprops={"arrowstyle": "-", "color": QUIET})
    ax.set(xscale="log", yscale="log", xlim=(0.8, 900), ylim=(0.8, 100), xlabel="Total privacy budget actually spent (log)",
           ylabel="Error of published total vs true, % (log)")
    ax.set_title(f"Same honest budget (ε = 5), 64 days: {i5.median_rel_error*100:.1f}% error with noise on daily increments,\n"
                 f"{t5.median_rel_error*100:.1f}% with the tree, {s5.median_rel_error*100:.0f}% with an even split",
                 loc="left", fontsize=10.5, fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_cap_table.png", dpi=160)
    plt.close(fig)


def fig5_daml_design() -> None:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.set(xlim=(0, 760), ylim=(372, 0))
    ax.axis("off")

    def box(x, y, w, h, lines, accent=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8",
                                    fc=BLUE if accent else "none", alpha=0.15 if accent else 1, ec=BLUE if accent else GREY, lw=2 if accent else 1.2))
        if accent:
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8", fc="none", ec=BLUE, lw=2))
        for i, (t, bold) in enumerate(lines):
            ax.text(x + 14, y + 24 + 17 * i, t, fontsize=9.5 if bold else 8.5, fontweight="bold" if bold else "normal",
                    color=INK if bold else QUIET, va="baseline")

    def arrow(x1, y1, x2, y2, dashed=False):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops={"arrowstyle": "-" if dashed else "->", "color": GREY,
                                                        "ls": "--" if dashed else "-", "lw": 1.2})

    ax.text(24, 28, "Only a release within the on-ledger privacy budget becomes public", fontsize=11, fontweight="bold")
    for i, n in enumerate("ABC"):
        box(24, 108 + 44 * i, 140, 36, [(f"Dealer {n}", True)])
        arrow(164, 126 + 44 * i, 200, 126 + 44 * i)
    box(200, 118, 216, 100, [("AggregationRound", True), ("holds Submission contracts,", False),
                             ("each seen only by its dealer", False), ("and the aggregator", False)])
    box(400, 52, 200, 36, [("PrivacyBudget: Σε ≤ cap", True)])
    ax.add_patch(Polygon([(440, 168), (500, 128), (560, 168), (500, 208)], closed=True, fc="none", ec=GREY, lw=1.2))
    ax.text(500, 172, "Within\nbudget?", ha="center", va="center", fontsize=9)
    box(590, 126, 146, 84, [("Published total", True), ("value + noise", False), ("and ε spent, public", False)], accent=True)
    box(420, 236, 160, 36, [("Release refused", True)])
    box(200, 250, 216, 68, [("Auditor", True), ("sees N submissions and the", False), ("noise commitment, not values", False)])
    arrow(416, 168, 440, 168)
    arrow(500, 88, 500, 128)
    arrow(560, 168, 590, 168)
    arrow(500, 208, 500, 236)
    arrow(308, 218, 308, 250, dashed=True)
    ax.text(427, 160, "ε", fontsize=8.5, color=QUIET, ha="center")
    ax.text(508, 112, "checked", fontsize=8.5, color=QUIET)
    ax.text(574, 160, "yes", fontsize=8.5, color=QUIET, ha="center")
    ax.text(508, 226, "no", fontsize=8.5, color=QUIET)
    ax.text(24, 352, "Noise is drawn off-ledger (Daml is deterministic) and committed before the total is revealed.",
            fontsize=8.5, color=QUIET)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_daml_design.png", dpi=160)
    plt.close(fig)


def fig4_horizon() -> None:
    df = pd.read_csv(RES / "e9_horizon.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for col, c, ls, label in (("increments_abs_error", BLUE, "-", "Noise on each day's new shares"),
                              ("tree_abs_error", INK, "-", "Binary-tree mechanism"),
                              ("naive-split_abs_error", GREY, "--", "Noise split evenly over all days")):
        ax.plot(df.days, df[col] / 1e6, ls, marker="o", color=c, lw=2.5 if c == BLUE else 1.8, label=label)
    ax.plot(df.days, df.clipped_away_shares / 1e6, ":", color=QUIET, lw=1.8, label="Shares cut by the 200k clip (never recovered)")
    ax.set(xscale="log", yscale="log", xlabel="Days in the stream (log)", ylabel="Million shares (log)")
    ax.set_xticks(df.days, [str(d) for d in df.days])
    ax.minorticks_off()
    short, long_ = df[df.days == 64].iloc[0], df[df.days == 1024].iloc[0]
    ax.set_title(f"Short streams favour simple increment noise ({short.increments_abs_error/1e3:.0f}k vs {short.tree_abs_error/1e3:.0f}k shares at 64 days);\n"
                 f"past ~250 days clipping, not noise, drives the error ({long_.clipped_away_share*100:.0f}% of issued shares are cut)",
                 loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_horizon.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig1_frontier()
    fig2_market_size()
    fig3_cap_table()
    fig4_horizon()
    fig5_daml_design()
    print("figures written to", OUT)
