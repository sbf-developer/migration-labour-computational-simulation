"""Publication-quality figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"

MECH_LABELS = {
    0: "M0: Labour supply only",
    1: "M1: + Local demand",
    2: "M2: + Firm entry/exit",
    3: "M3: + Capital adjustment",
    4: "M4: + Task specialisation",
    5: "M5: + Migrant entrepreneurship",
    6: "M6: + Innovation",
}

MECH_SHORT = {
    0: "M0",
    1: "M1",
    2: "M2",
    3: "M3",
    4: "M4",
    5: "M5",
    6: "M6",
}


def _style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def fig_schematic() -> None:
    _style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        (1, 4.2, "Immigration\nshock"),
        (3.5, 4.2, "Labour supply\n& matching"),
        (6, 4.2, "Production\n(CES tasks)"),
        (8.5, 4.2, "Wages &\nemployment"),
        (3.5, 2.2, "Household\nconsumption"),
        (6, 2.2, "Firm entry/\nexit & capital"),
        (8.5, 2.2, "Innovation &\nspecialisation"),
    ]
    for x, y, txt in boxes:
        ax.add_patch(plt.Rectangle((x - 0.9, y - 0.55), 1.8, 1.1, fill=False, lw=1.5))
        ax.text(x, y, txt, ha="center", va="center", fontsize=9)

    arrows = [
        ((1.9, 4.2), (2.6, 4.2)),
        ((4.4, 4.2), (5.1, 4.2)),
        ((6.9, 4.2), (7.6, 4.2)),
        ((3.5, 3.65), (3.5, 2.75)),
        ((6, 3.65), (6, 2.75)),
        ((4.4, 2.2), (5.1, 2.2)),
        ((6.9, 2.2), (7.6, 2.2)),
        ((8.5, 2.75), (8.5, 3.65)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.2))

    ax.set_title("Model architecture: nested general-equilibrium feedbacks", fontsize=11, pad=12)
    fig.savefig(FIGURES_DIR / "fig1_schematic.pdf")
    fig.savefig(FIGURES_DIR / "fig1_schematic.png")
    plt.close(fig)


def fig_mechanism_ladder(ladder: pd.DataFrame) -> None:
    _style()
    main = ladder[ladder["treatment"] == "shock_5pct_balanced"].copy()

    agg = main.groupby("mechanism_level").agg(
        short=("native_wage_change_short", "mean"),
        long=("native_wage_change_long", "mean"),
        short_se=("native_wage_change_short", "sem"),
        long_se=("native_wage_change_long", "sem"),
        emp_short=("native_employment_short", "mean"),
        emp_short_se=("native_employment_short", "sem"),
        unemp_short=("unemployment_short", "mean"),
        unemp_short_se=("unemployment_short", "sem"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = agg["mechanism_level"]

    ax = axes[0]
    ax.errorbar(x, agg["short"], yerr=1.96 * agg["short_se"], fmt="o-", capsize=3, label="Short run (0--2 yr)")
    ax.errorbar(x, agg["long"], yerr=1.96 * agg["long_se"], fmt="s--", capsize=3, label="Long run (8--12 yr)")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xticks(range(7))
    ax.set_xticklabels([MECH_SHORT[i] for i in range(7)])
    ax.set_xlabel("Mechanism level")
    ax.set_ylabel("Native wage-index change (\\%)")
    ax.legend(frameon=True, fontsize=8)
    ax.set_title("(a) Wage index")

    ax = axes[1]
    ax.errorbar(
        x, agg["emp_short"], yerr=1.96 * agg["emp_short_se"],
        fmt="o-", capsize=3, color="#2980b9", label="Native employment",
    )
    ax.errorbar(
        x, agg["unemp_short"], yerr=1.96 * agg["unemp_short_se"],
        fmt="s--", capsize=3, color="#c0392b", label="Unemployment (all workers)",
    )
    ax.set_xticks(range(7))
    ax.set_xticklabels([MECH_SHORT[i] for i in range(7)])
    ax.set_xlabel("Mechanism level")
    ax.set_ylabel("Rate (short run, 0--2 yr)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, fontsize=8)
    ax.set_title("(b) Employment and unemployment")

    fig.suptitle("Mechanism ladder: wages vs. labour-market congestion", y=1.03)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_mechanism_ladder.pdf")
    fig.savefig(FIGURES_DIR / "fig2_mechanism_ladder.png")
    plt.close(fig)


def fig_decomposition(decomp: pd.DataFrame) -> None:
    _style()
    main = decomp[decomp["treatment"] == "shock_5pct_balanced"].copy()
    main["label"] = main["to_level"].map(lambda k: MECH_SHORT.get(k, str(k)))

    agg = main.groupby("to_level").agg(
        delta=("delta_wage_change_short", "mean"),
        se=("delta_wage_change_short", "sem"),
    ).reset_index()
    agg["label"] = agg["to_level"].map(MECH_SHORT)

    colors = ["#c0392b" if v < 0 else "#27ae60" for v in agg["delta"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(agg["label"], agg["delta"], yerr=1.96 * agg["se"], color=colors, alpha=0.85, capsize=3)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Mechanism added at this step")
    ax.set_ylabel("Incremental native wage effect (pp)")
    ax.set_title("Mechanism decomposition: $\\Delta w^{M_k} - \\Delta w^{M_{k-1}}$")
    fig.savefig(FIGURES_DIR / "fig3_decomposition.pdf")
    fig.savefig(FIGURES_DIR / "fig3_decomposition.png")
    plt.close(fig)


def fig_time_paths(ts: pd.DataFrame) -> None:
    _style()
    shock_year = 10
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    wage_col = "native_wage_index_change_pct" if "native_wage_index_change_pct" in ts.columns else "native_wage_change_pct"
    panels = [
        (wage_col, "Native wage-index change (\\%)", axes[0, 0]),
        ("native_employment", "Native employment rate", axes[0, 1]),
        ("unemployment_rate", "Unemployment rate (all workers)", axes[1, 0]),
        ("output_per_capita", "Output per capita", axes[1, 1]),
    ]
    for col, title, ax in panels:
        sub = ts.groupby("year")[col].agg(["mean", "sem"]).reset_index()
        sub = sub.dropna(subset=["mean"])
        ax.plot(sub["year"], sub["mean"], lw=2)
        ax.fill_between(
            sub["year"],
            sub["mean"] - 1.96 * sub["sem"],
            sub["mean"] + 1.96 * sub["sem"],
            alpha=0.2,
        )
        ax.axvline(shock_year, color="crimson", ls="--", lw=1, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Year")

    fig.suptitle("Dynamic adjustment after 5\\% balanced immigration shock (M6)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_time_paths.pdf")
    fig.savefig(FIGURES_DIR / "fig4_time_paths.png")
    plt.close(fig)


def fig_m0_m6_comparison(ladder: pd.DataFrame) -> None:
    """Side-by-side M0 vs M6: wages, native employment, unemployment."""
    _style()
    main = ladder[
        (ladder["treatment"] == "shock_5pct_balanced") & (ladder["mechanism_level"].isin([0, 6]))
    ]
    metrics = [
        ("native_wage_change_short", "Wage-index change (\\%)"),
        ("native_employment_short", "Native employment rate"),
        ("unemployment_short", "Unemployment rate"),
    ]
    labels = ["M0\n(labour only)", "M6\n(full model)"]
    x = np.arange(2)
    width = 0.22
    colors = ["#3498db", "#27ae60", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (col, ylab) in enumerate(metrics):
        means = [main[main["mechanism_level"] == lvl][col].mean() for lvl in (0, 6)]
        ses = [main[main["mechanism_level"] == lvl][col].sem() for lvl in (0, 6)]
        offset = (i - 1) * width
        ax.bar(x + offset, means, width, yerr=[1.96 * s for s in ses], capsize=3, label=ylab, color=colors[i], alpha=0.85)

    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Short-run outcome (0--2 yr after shock)")
    ax.legend(frameon=True, fontsize=8, loc="upper left")
    ax.set_title("M0 vs M6: wages rise under full GE; M0 shows severe congestion")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig7_m0_m6_comparison.pdf")
    fig.savefig(FIGURES_DIR / "fig7_m0_m6_comparison.png")
    plt.close(fig)


def fig_treatment_heterogeneity(treatments: pd.DataFrame) -> None:
    _style()
    agg = treatments.groupby("treatment").agg(
        short=("native_wage_change_short", "mean"),
        short_se=("native_wage_change_short", "sem"),
        unemp=("unemployment_short", "mean"),
        unemp_se=("unemployment_short", "sem"),
    ).reset_index()
    label_map = {k: v.label.replace("\\%", "%") for k, v in __import__("config").TREATMENTS.items()}
    agg["label"] = agg["treatment"].map(label_map)
    agg = agg.sort_values("short")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    y_pos = np.arange(len(agg))

    ax = axes[0]
    ax.barh(y_pos, agg["short"], xerr=1.96 * agg["short_se"], color="#2980b9", alpha=0.85, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(agg["label"])
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Wage-index change (\\%)")
    ax.set_title("(a) Wages (all scenarios positive, narrow range)")

    ax = axes[1]
    ax.barh(y_pos, agg["unemp"], xerr=1.96 * agg["unemp_se"], color="#c0392b", alpha=0.85, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.set_xlabel("Unemployment rate")
    ax.set_title("(b) Unemployment (limited scenario variation)")

    fig.suptitle("Treatment heterogeneity under M6", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_treatments.pdf")
    fig.savefig(FIGURES_DIR / "fig5_treatments.png")
    plt.close(fig)


def fig_skill_heterogeneity(ladder: pd.DataFrame) -> None:
    _style()
    main = ladder[(ladder["treatment"] == "shock_5pct_balanced") & (ladder["mechanism_level"] == 6)]
    skills = ["Low", "Medium", "High"]
    cols = ["native_wage_skill_0_short", "native_wage_skill_1_short", "native_wage_skill_2_short"]
    means = [main[c].mean() for c in cols]
    ses = [main[c].sem() for c in cols]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(skills, means, yerr=[1.96 * s for s in ses], capsize=4, color=["#e74c3c", "#f39c12", "#2ecc71"])
    ax.set_ylabel("Native wage index (level, short run)")
    ax.set_title("Skill gradient: levels, not changes (M6, 5\\% shock)")
    fig.savefig(FIGURES_DIR / "fig6_skill_heterogeneity.pdf")
    fig.savefig(FIGURES_DIR / "fig6_skill_heterogeneity.png")
    plt.close(fig)


def generate_all_figures(
    ladder: pd.DataFrame,
    decomp: pd.DataFrame,
    treatments: pd.DataFrame,
    ts: pd.DataFrame,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_schematic()
    fig_mechanism_ladder(ladder)
    fig_decomposition(decomp)
    fig_time_paths(ts)
    fig_treatment_heterogeneity(treatments)
    fig_skill_heterogeneity(ladder)
    fig_m0_m6_comparison(ladder)
