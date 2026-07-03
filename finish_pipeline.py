"""Fast finish: stats, tables, figures from existing ladder; minimal extra runs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd

from experiment import (
    FIGURES_DIR,
    RESULTS_DIR,
    _run_single,
    mechanism_decomposition,
    run_time_series,
    write_latex_stats,
)
from config import TREATMENTS
from viz import generate_all_figures


def run_treatment_grid_fast(mechanism_level: int = 6, n_replicates: int = 15) -> pd.DataFrame:
    rows = []
    for name in TREATMENTS:
        if name == "baseline":
            continue
        for seed in range(n_replicates):
            rows.append(_run_single(mechanism_level, name, seed + 200))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "treatment_grid_M6.csv", index=False)
    return df


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ladder = pd.read_csv(RESULTS_DIR / "mechanism_ladder_shock_5pct_balanced.csv")
    print(f"Loaded ladder: {len(ladder)} rows")

    print("Treatment grid (15 replicates x 7 treatments)...")
    treatments = run_treatment_grid_fast(n_replicates=15)

    print("Time series (20 replicates)...")
    ts = run_time_series(mechanism_level=6, n_replicates=20)

    decomp = mechanism_decomposition(ladder)
    write_latex_stats(ladder, decomp)
    generate_all_figures(ladder, decomp, treatments, ts)
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
