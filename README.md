# Associative Sensitization in Stentor

Behavioral data and analysis for associative sensitization in *Stentor coeruleus*. Cells are presented with repeated pairs of a weak mechanical tap (CS) and a strong tap (US) and each response is scored from video as a binary contraction. The data covers the primary weak-strong pairing paradigm across multiple ISI/ITI timing conditions, along with weak-only, strong-only, weak-weak, and habituation controls.

**Preprint**: [doi:10.64898/2026.02.25.708045](https://doi.org/10.64898/2026.02.25.708045)

## Data

Two CSV files in `data/`, both recording single-cell contraction responses (one row per cell per stimulus). See [`data/data_dictionary.md`](data/data_dictionary.md) for column definitions, condition key, and per-file details.

| File | Experiment |
|---|---|
| `ws_single.csv` | Weak-strong pairings and timing conditions |
| `control.csv` | Arousal and sensitization controls |

## Repository structure

```
data/           Raw experimental data and data dictionary
analysis/       Per-figure analysis scripts and shared utilities (Julia)
figures/        Output PNGs produced by the analysis scripts
```

## Analysis scripts

Each script in `analysis/` includes `common.jl` (shared data loading, models, and helpers) and produces one figure in `figures/`.

| Script | Paper figure |
|---|---|
| `fig2_main_effects.jl` | Fig 2 — Main effects and controls |
| `fig4_temporal.jl` | Fig 4 — Temporal parameter effects |
| `fig5_single_cell.jl` | Fig 5 — Single-cell characterization |
| `fig6_model.jl` | Fig 6 — Computational model |
| `figS3_smoothing.jl` | Fig S3 — Smoothing robustness |
| `figS4_temporal_breakdown.jl` | Fig S4 — Temporal parameter breakdown |

## Setup

1. Install Julia.
2. Start the Julia REPL.
3. Activate the project with `] activate .`
4. Instantiate the project with `] instantiate`.
5. Run a figure script: `cd("analysis"); include("fig2_main_effects.jl")`
