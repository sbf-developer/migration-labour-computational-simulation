# From Labour-Supply Shock to Economic Adjustment

Agent-based immigration labour-market simulation with nested mechanism decomposition (M0–M6), stylised calibration, and academic paper.

**Repository:** [github.com/sbf-developer/migration-labour-computational-simulation](https://github.com/sbf-developer/migration-labour-computational-simulation)

## Quick start

```bash
pip install -r requirements.txt
python run_experiment.py --replicates 50
python generate_paper_outputs.py
python validate_results.py
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Output: `paper/main.pdf`, figures in `figures/`, results in `results/`.

## Structure

| Path | Purpose |
|------|---------|
| `src/model.py` | Agent-based labour market (workers, firms, M0–M6 mechanisms) |
| `src/experiment.py` | Experiment orchestration and LaTeX stats |
| `src/viz.py` | Publication figures |
| `data/calibration_targets.json` | Stylised calibration moments |
| `paper/main.tex` | Manuscript (Scott Brodie Forsyth) |

## Citation

Forsyth, S. B. (2026). *From Labour-Supply Shock to Economic Adjustment: A Stylised Agent-Based Model of Immigration, Wages, and Firm Dynamics.*
