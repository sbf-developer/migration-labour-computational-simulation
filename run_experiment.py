"""Entry point: run experiment and generate all outputs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from experiment import run_all

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run immigration labour-market ABM experiments")
    parser.add_argument("--replicates", type=int, default=120, help="Stochastic replicates per cell")
    args = parser.parse_args()
    run_all(n_replicates=args.replicates)
