"""Experiment orchestration and mechanism decomposition."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import numpy as np
import pandas as pd

from config import (
    MechanismConfig,
    SimulationConfig,
    TREATMENTS,
    TreatmentSpec,
)
from model import LabourMarketModel


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
PAPER_DIR = ROOT / "paper"


def _run_single(
    mechanism_level: int,
    treatment_name: str,
    seed: int,
) -> dict:
    treatment = TREATMENTS[treatment_name]
    cfg = SimulationConfig(
        rng_seed=seed,
        shock_size=treatment.shock_size,
        shock_skill=treatment.shock_skill,
        economic_state=treatment.economic_state,
        substitution_tightness=treatment.substitution_tightness,
        adjustment_speed="fixed" if mechanism_level <= 1 else treatment.adjustment_speed,
        migrant_remittance=treatment.migrant_remittance,
        migrant_entrepreneurship_multiplier=treatment.migrant_entrepreneurship_multiplier,
    )
    mech = MechanismConfig.from_level(mechanism_level)
    model = LabourMarketModel(config=cfg, mechanisms=mech, rng=np.random.default_rng(seed))
    history = model.run(treatment if treatment.shock_size > 0 else None)
    df = pd.DataFrame(history)
    shock_year = cfg.shock_year

    def horizon_mean(col: str, start: int, end: int) -> float:
        if col not in df.columns:
            return np.nan
        sub = df[(df["year"] >= start) & (df["year"] <= end)]
        return float(sub[col].mean()) if len(sub) else np.nan

    out = {
        "mechanism_level": mechanism_level,
        "treatment": treatment_name,
        "seed": seed,
        "native_wage_short": horizon_mean("native_wage", shock_year, shock_year + 2),
        "native_wage_medium": horizon_mean("native_wage", shock_year + 3, shock_year + 7),
        "native_wage_long": horizon_mean("native_wage", shock_year + 8, shock_year + 12),
        "native_employment_short": horizon_mean("native_employment", shock_year, shock_year + 2),
        "native_employment_long": horizon_mean("native_employment", shock_year + 8, shock_year + 12),
        "unemployment_short": horizon_mean("unemployment_rate", shock_year, shock_year + 2),
        "output_long": horizon_mean("output_per_capita", shock_year + 8, shock_year + 12),
        "vacancy_short": horizon_mean("vacancy_rate", shock_year, shock_year + 2),
        "firm_count_long": horizon_mean("firm_count", shock_year + 8, shock_year + 12),
        "native_wage_skill_0_short": horizon_mean("native_wage_skill_0", shock_year, shock_year + 2),
        "native_wage_skill_1_short": horizon_mean("native_wage_skill_1", shock_year, shock_year + 2),
        "native_wage_skill_2_short": horizon_mean("native_wage_skill_2", shock_year, shock_year + 2),
    }
    if "native_wage_index_change_pct" in df.columns:
        out["native_wage_change_short"] = horizon_mean("native_wage_index_change_pct", shock_year + 1, shock_year + 2)
        out["native_wage_change_long"] = horizon_mean("native_wage_index_change_pct", shock_year + 8, shock_year + 12)
    elif "native_wage_change_pct" in df.columns:
        out["native_wage_change_short"] = horizon_mean("native_wage_change_pct", shock_year + 1, shock_year + 2)
        out["native_wage_change_long"] = horizon_mean("native_wage_change_pct", shock_year + 8, shock_year + 12)
    return out


def run_mechanism_ladder(
    treatment_name: str = "shock_5pct_balanced",
    n_replicates: int = 120,
    parallel: bool = False,
) -> pd.DataFrame:
    """Run M0--M6 for one treatment."""
    tasks = [
        (level, treatment_name, seed)
        for level in range(7)
        for seed in range(n_replicates)
    ]
    rows: list[dict] = []
    if parallel:
        with ProcessPoolExecutor(max_workers=min(8, len(tasks))) as ex:
            futures = [ex.submit(_run_single, *t) for t in tasks]
            for fut in as_completed(futures):
                rows.append(fut.result())
    else:
        for t in tasks:
            rows.append(_run_single(*t))
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / f"mechanism_ladder_{treatment_name}.csv", index=False)
    return df


def run_treatment_grid(
    mechanism_level: int = 6,
    n_replicates: int = 120,
) -> pd.DataFrame:
    rows = []
    for name in TREATMENTS:
        if name == "baseline":
            continue
        for seed in range(n_replicates):
            rows.append(_run_single(mechanism_level, name, seed))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "treatment_grid_M6.csv", index=False)
    return df


def run_time_series(
    mechanism_level: int = 6,
    treatment_name: str = "shock_5pct_balanced",
    n_replicates: int = 80,
) -> pd.DataFrame:
    """Collect year-by-year paths for plotting."""
    all_rows = []
    for seed in range(n_replicates):
        treatment = TREATMENTS[treatment_name]
        cfg = SimulationConfig(rng_seed=seed + 1000)
        mech = MechanismConfig.from_level(mechanism_level)
        model = LabourMarketModel(config=cfg, mechanisms=mech, rng=np.random.default_rng(seed + 1000))
        history = model.run(treatment)
        for row in history:
            row = dict(row)
            row["seed"] = seed
            row["mechanism_level"] = mechanism_level
            row["treatment"] = treatment_name
            all_rows.append(row)
    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / f"time_series_{treatment_name}_M{mechanism_level}.csv", index=False)
    return df


def mechanism_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Compute incremental wage effects Δw^Mk - Δw^{M(k-1)}."""
    agg = (
        df.groupby(["mechanism_level", "treatment"])
        .agg(
            native_wage_short=("native_wage_short", "mean"),
            native_wage_long=("native_wage_long", "mean"),
            native_wage_change_short=("native_wage_change_short", "mean"),
            native_wage_change_long=("native_wage_change_long", "mean"),
            native_employment_short=("native_employment_short", "mean"),
            output_long=("output_long", "mean"),
        )
        .reset_index()
    )
    decomp_rows = []
    for treatment, sub in agg.groupby("treatment"):
        sub = sub.sort_values("mechanism_level")
        for i in range(1, len(sub)):
            cur = sub.iloc[i]
            prev = sub.iloc[i - 1]
            decomp_rows.append(
                {
                    "treatment": treatment,
                    "from_level": int(prev["mechanism_level"]),
                    "to_level": int(cur["mechanism_level"]),
                    "delta_wage_short": cur["native_wage_short"] - prev["native_wage_short"],
                    "delta_wage_long": cur["native_wage_long"] - prev["native_wage_long"],
                    "delta_wage_change_short": cur["native_wage_change_short"] - prev["native_wage_change_short"],
                    "delta_employment_short": cur["native_employment_short"] - prev["native_employment_short"],
                    "delta_output_long": cur["output_long"] - prev["output_long"],
                }
            )
    decomp = pd.DataFrame(decomp_rows)
    decomp.to_csv(RESULTS_DIR / "mechanism_decomposition.csv", index=False)
    return decomp


def write_latex_stats(df: pd.DataFrame, decomp: pd.DataFrame) -> None:
    """Generate paper/generated_stats.tex with key numbers."""
    main = df[df["treatment"] == "shock_5pct_balanced"]
    m0_short = main[main["mechanism_level"] == 0]["native_wage_change_short"].mean()
    m6_short = main[main["mechanism_level"] == 6]["native_wage_change_short"].mean()
    m0_long = main[main["mechanism_level"] == 0]["native_wage_change_long"].mean()
    m6_long = main[main["mechanism_level"] == 6]["native_wage_change_long"].mean()
    m6_low = main[main["mechanism_level"] == 6]["native_wage_skill_0_short"].mean()
    m6_high = main[main["mechanism_level"] == 6]["native_wage_skill_2_short"].mean()
    m0_emp = main[main["mechanism_level"] == 0]["native_employment_short"].mean()
    m6_emp = main[main["mechanism_level"] == 6]["native_employment_short"].mean()
    m0_unemp = main[main["mechanism_level"] == 0]["unemployment_short"].mean()
    m6_unemp = main[main["mechanism_level"] == 6]["unemployment_short"].mean()
    m0_emp_se = main[main["mechanism_level"] == 0]["native_employment_short"].sem()
    m6_emp_se = main[main["mechanism_level"] == 6]["native_employment_short"].sem()
    m0_unemp_se = main[main["mechanism_level"] == 0]["unemployment_short"].sem()
    m6_unemp_se = main[main["mechanism_level"] == 6]["unemployment_short"].sem()
    m0_short_se = main[main["mechanism_level"] == 0]["native_wage_change_short"].sem()

    decomp_main = decomp[decomp["treatment"] == "shock_5pct_balanced"]
    demand_contrib = decomp_main[decomp_main["to_level"] == 1]["delta_wage_change_short"].mean()
    entry_contrib = decomp_main[decomp_main["to_level"] == 2]["delta_wage_change_short"].mean()
    innov_contrib = decomp_main[decomp_main["to_level"] == 6]["delta_wage_change_short"].mean()

    n_rep = main["seed"].nunique()
    lines = [
        "% Auto-generated by experiment.py — do not edit by hand",
        f"\\newcommand{{\\NReplicates}}{{{n_rep}}}",
        f"\\newcommand{{\\MZeroShortWage}}{{{m0_short:.2f}}}",
        f"\\newcommand{{\\MSixShortWage}}{{{m6_short:.2f}}}",
        f"\\newcommand{{\\MZeroLongWage}}{{{m0_long:.2f}}}",
        f"\\newcommand{{\\MSixLongWage}}{{{m6_long:.2f}}}",
        f"\\newcommand{{\\LowSkillShortWage}}{{{m6_low:.3f}}}",
        f"\\newcommand{{\\HighSkillShortWage}}{{{m6_high:.3f}}}",
        f"\\newcommand{{\\DemandContrib}}{{{demand_contrib:.2f}}}",
        f"\\newcommand{{\\EntryContrib}}{{{entry_contrib:.2f}}}",
        f"\\newcommand{{\\InnovContrib}}{{{innov_contrib:.2f}}}",
        f"\\newcommand{{\\MZeroEmpShort}}{{{100 * m0_emp:.1f}}}",
        f"\\newcommand{{\\MSixEmpShort}}{{{100 * m6_emp:.1f}}}",
        f"\\newcommand{{\\MZeroUnempShort}}{{{100 * m0_unemp:.1f}}}",
        f"\\newcommand{{\\MSixUnempShort}}{{{100 * m6_unemp:.1f}}}",
        f"\\newcommand{{\\MZeroShortWageSE}}{{{1.96 * m0_short_se:.2f}}}",
    ]
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    (PAPER_DIR / "generated_stats.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Mechanism ladder table for paper
    agg = (
        main.groupby("mechanism_level")
        .agg(
            short=("native_wage_change_short", "mean"),
            short_se=("native_wage_change_short", "sem"),
            long=("native_wage_change_long", "mean"),
            long_se=("native_wage_change_long", "sem"),
        )
        .reset_index()
    )
    table_lines = [
        "% Auto-generated results table",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Native wage-index change (\\%) by mechanism level: balanced 5\\% shock}",
        "  \\label{tab:results}",
        "  \\begin{tabular}{lcc}",
        "    \\toprule",
        "    Model & Short run (0--2 yr) & Long run (8--12 yr) \\\\",
        "    \\midrule",
    ]
    labels = ["M0", "M1", "M2", "M3", "M4", "M5", "M6"]
    for _, row in agg.iterrows():
        lvl = int(row["mechanism_level"])
        table_lines.append(
            f"    {labels[lvl]} & {row['short']:.2f} ({1.96*row['short_se']:.2f}) & "
            f"{row['long']:.2f} ({1.96*row['long_se']:.2f}) \\\\"
        )
    table_lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "  \\begin{minipage}{0.92\\textwidth}",
            "    \\small\\emph{Note:} Entries are mean native skill-wage-index changes (\\%) relative to the pre-shock baseline, with 95\\% replicate standard errors in parentheses. $N=\\NReplicates{}$ replicates per cell.",
            "  \\end{minipage}",
            "\\end{table}",
        ]
    )
    (PAPER_DIR / "generated_results_table.tex").write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    emp_agg = (
        main.groupby("mechanism_level")
        .agg(
            emp_short=("native_employment_short", "mean"),
            emp_short_se=("native_employment_short", "sem"),
            unemp_short=("unemployment_short", "mean"),
            unemp_short_se=("unemployment_short", "sem"),
            emp_long=("native_employment_long", "mean"),
            emp_long_se=("native_employment_long", "sem"),
        )
        .reset_index()
    )
    emp_lines = [
        "% Auto-generated employment table",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Native employment and unemployment after a balanced 5\\% shock}",
        "  \\label{tab:employment}",
        "  \\small",
        "  \\begin{tabular}{@{}lcccc@{}}",
        "    \\toprule",
        "    Model & Native emp.\\ (short) & Unemp.\\ (short) & Native emp.\\ (long) & Comment \\\\",
        "    \\midrule",
    ]
    comments = {
        0: "Severe congestion; fixed jobs",
        1: "Demand restores matching",
        6: "Wages up; employment drifts down over time",
    }
    for lvl in (0, 1, 6):
        row = emp_agg[emp_agg["mechanism_level"] == lvl].iloc[0]
        emp_lines.append(
            f"    M{lvl} & {100 * row['emp_short']:.1f}\\% ({100 * 1.96 * row['emp_short_se']:.1f}) & "
            f"{100 * row['unemp_short']:.1f}\\% ({100 * 1.96 * row['unemp_short_se']:.1f}) & "
            f"{100 * row['emp_long']:.1f}\\% ({100 * 1.96 * row['emp_long_se']:.1f}) & "
            f"{comments[lvl]} \\\\"
        )
    emp_lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "  \\begin{minipage}{0.92\\textwidth}",
            "    \\small\\emph{Note:} Short run = years 0--2 after shock; long run = years 8--12. Unemployment is economy-wide. 95\\% replicate standard errors in parentheses.",
            "  \\end{minipage}",
            "\\end{table}",
        ]
    )
    (PAPER_DIR / "generated_employment_table.tex").write_text("\n".join(emp_lines) + "\n", encoding="utf-8")

    summary = {
        "n_replicates": int(n_rep),
        "m0_short_wage_change_pct": float(m0_short),
        "m6_short_wage_change_pct": float(m6_short),
        "m0_long_wage_change_pct": float(m0_long),
        "m6_long_wage_change_pct": float(m6_long),
        "m0_native_employment_short": float(m0_emp),
        "m6_native_employment_short": float(m6_emp),
        "m0_unemployment_short": float(m0_unemp),
        "m6_unemployment_short": float(m6_unemp),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_all(n_replicates: int = 120) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ladder = run_mechanism_ladder("shock_5pct_balanced", n_replicates=n_replicates, parallel=False)
    treatments = run_treatment_grid(mechanism_level=6, n_replicates=min(n_replicates, 40))
    ts = run_time_series(mechanism_level=6, n_replicates=min(n_replicates, 50))

    decomp = mechanism_decomposition(ladder)
    write_latex_stats(ladder, decomp)

    from viz import generate_all_figures

    generate_all_figures(ladder, decomp, treatments, ts)

    print(f"Completed {len(ladder)} mechanism-ladder runs and {len(treatments)} treatment runs.")
