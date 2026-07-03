"""Generate paper outputs from ladder CSV + minimal supplementary runs."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd

from config import TREATMENTS
from experiment import (
    FIGURES_DIR,
    RESULTS_DIR,
    _run_single,
    mechanism_decomposition,
    write_latex_stats,
)
from viz import (
    fig_decomposition,
    fig_mechanism_ladder,
    fig_m0_m6_comparison,
    fig_schematic,
    fig_skill_heterogeneity,
    fig_time_paths,
    fig_treatment_heterogeneity,
)


def main() -> None:
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ladder = pd.read_csv(RESULTS_DIR / "mechanism_ladder_shock_5pct_balanced.csv")
    decomp = mechanism_decomposition(ladder)
    write_latex_stats(ladder, decomp)
    print("Stats and results table written.", flush=True)

    fig_schematic()
    fig_mechanism_ladder(ladder)
    fig_decomposition(decomp)
    fig_skill_heterogeneity(ladder)
    fig_m0_m6_comparison(ladder)
    print("Core figures written.", flush=True)

    # Minimal treatment grid (10 replicates)
    rows = []
    for i, name in enumerate(TREATMENTS):
        if name == "baseline":
            continue
        print(f"Treatment {name}...", flush=True)
        for seed in range(8):
            rows.append(_run_single(6, name, seed + 300))
    treatments = pd.DataFrame(rows)
    treatments.to_csv(RESULTS_DIR / "treatment_grid_M6.csv", index=False)
    fig_treatment_heterogeneity(treatments)
    print("Treatment figure written.", flush=True)

    # Minimal time series (12 replicates)
    from config import MechanismConfig, SimulationConfig
    from model import LabourMarketModel
    import numpy as np

    all_rows = []
    for seed in range(8):
        print(f"Time series seed {seed}...", flush=True)
        cfg = SimulationConfig(rng_seed=seed + 500)
        mech = MechanismConfig.from_level(6)
        model = LabourMarketModel(cfg, mech, rng=np.random.default_rng(seed + 500))
        for row in model.run(TREATMENTS["shock_5pct_balanced"]):
            row = dict(row)
            row["seed"] = seed
            all_rows.append(row)
    ts = pd.DataFrame(all_rows)
    ts.to_csv(RESULTS_DIR / "time_series_shock_5pct_balanced_M6.csv", index=False)
    fig_time_paths(ts)
    print(f"Done in {time.time()-t0:.0f}s.", flush=True)


if __name__ == "__main__":
    main()
