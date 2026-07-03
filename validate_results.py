"""Pre-push validation: check stats, files, and text-number consistency."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "paper"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def _read_macro(name: str) -> float:
    text = (PAPER / "generated_stats.tex").read_text(encoding="utf-8")
    m = re.search(rf"\\newcommand{{\\{name}}}{{([^}}]+)}}", text)
    if not m:
        raise ValueError(f"Missing macro {name}")
    return float(m.group(1))


def main() -> int:
    errors: list[str] = []

    required = [
        PAPER / "main.tex",
        PAPER / "main.pdf",
        PAPER / "references.bib",
        PAPER / "generated_stats.tex",
        PAPER / "generated_results_table.tex",
        RESULTS / "mechanism_ladder_shock_5pct_balanced.csv",
        RESULTS / "mechanism_decomposition.csv",
        RESULTS / "summary.json",
    ]
    for f in required:
        if not f.exists():
            errors.append(f"Missing required file: {f.relative_to(ROOT)}")

    for i in range(1, 7):
        if not (FIGURES / f"fig{i}_schematic.pdf").exists() and i == 1:
            pass
        pdf = list(FIGURES.glob(f"fig{i}_*.pdf"))
        if not pdf:
            errors.append(f"Missing figure pdf for fig{i}")

    if (RESULTS / "mechanism_ladder_shock_5pct_balanced.csv").exists():
        df = pd.read_csv(RESULTS / "mechanism_ladder_shock_5pct_balanced.csv")
        main = df[df.treatment == "shock_5pct_balanced"]
        n = main.seed.nunique()
        if n < 30:
            errors.append(f"Mechanism ladder has only {n} replicates (recommend >= 30)")

        if (PAPER / "generated_stats.tex").exists():
            m0 = main[main.mechanism_level == 0].native_wage_change_short.mean()
            m6 = main[main.mechanism_level == 6].native_wage_change_short.mean()
            if abs(_read_macro("MZeroShortWage") - m0) > 0.02:
                errors.append("MZeroShortWage macro does not match CSV")
            if abs(_read_macro("MSixShortWage") - m6) > 0.02:
                errors.append("MSixShortWage macro does not match CSV")
            if int(_read_macro("NReplicates")) != n:
                errors.append("NReplicates macro does not match CSV")

    summary = json.loads((RESULTS / "summary.json").read_text())
    if summary.get("n_replicates", 0) != 50:
        errors.append("summary.json n_replicates unexpected")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(" -", e)
        return 1

    print("Validation passed.")
    print(f"  Replicates: {int(_read_macro('NReplicates'))}")
    print(f"  M0 short: {_read_macro('MZeroShortWage'):.2f} pp")
    print(f"  M6 short: {_read_macro('MSixShortWage'):.2f} pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
