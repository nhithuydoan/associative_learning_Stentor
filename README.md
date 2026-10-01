# Associative Learning in Stentor

Behavioral data and analysis for associative sensitization in *Stentor coeruleus*. 

**Preprint**: [doi:10.64898/2026.02.25.708045](https://doi.org/10.64898/2026.02.25.708045)

## Data

CSV files in `data/`, each recording single-cell contraction responses (one row per cell per stimulus). See [`data/data_dictionary.md`](data/data_dictionary.md) for details. Folders include both original submission and follow-up data. 

## Setup

1. Install Julia.
2. Start the Julia REPL.
3. Activate the project with `] activate .`
4. Instantiate the project with `] instantiate`.
5. Run a figure script: `cd("analysis"); include("fig2_main_effects.jl")`
6. Run followup: `cd("analysis/followup"); include("fig3_common_test.jl")`
