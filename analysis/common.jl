using CSV, Statistics, DataFrames, NaNStatistics, Random,
     GLMakie, CairoMakie, QuadGK, HypothesisTests, StatsBase,
     MixedModels, GLM, ColorSchemes, Turing, LinearAlgebra, LsqFit

import DataFrames: transform!

include(joinpath(@__DIR__, "..", "BayesianTTest.jl"))
using .BayesianTTest

const DATA_DIR    = joinpath(@__DIR__, "..", "data")
const FIGURES_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURES_DIR)

# trial window for smoothing individual-cell responses. I also included the analysis with different rolling windows.
const ROLLING_WINDOW = 8

# different thresholds for quantifying learners
const PEAK_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# minimum rolling average to classify a cell as a learner
const LEARNER_THRESHOLD = 0.60

# NUTS posterior samples for quadratic regression
const N_BAYES_SAMPLES   = 20_000

const LINE_STYLES       = [:solid, :dash, :dot]

my_theme = Theme(
    fontsize = 20,
    Axis = (xlabelsize = 20, ylabelsize = 20, xticklabelsize = 20,
            yticklabelsize = 20, titlegap = 20, xgridvisible = false,
            ygridvisible = false, topspinevisible = false,
            rightspinevisible = false),
    Legend = (labelsize = 20, framevisible = false),
    Label  = (fontsize = 24, font = :bold))
set_theme!(my_theme)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Load weak-strong data
data = CSV.read(joinpath(DATA_DIR, "ws_single.csv"), DataFrame)

# Each cell has a unique id
transform!(data,
    [:run, :cell_number] => ByRow((run, cell) -> "$(run)_$(cell)") => :cell_id)

# Informativeness = (ITI + ISI) / ISI
transform!(data,
    [:isi, :iti] => ByRow((isi, iti) -> (iti + isi) / isi) => :informativeness)

control_data = CSV.read(joinpath(DATA_DIR, "control.csv"), DataFrame)
control_data.condition_pattern = [split(cond, "_")[1] for cond in control_data.condition]

# Label each stimulus as strong or weak based on the pattern string
function is_strong_stimulus(pattern, stimulus_num)
    if pattern == "s"
        return true
    elseif stimulus_num <= length(pattern)
        return pattern[stimulus_num] == 's'
    else
        return false
    end
end

control_data.is_strong = [is_strong_stimulus(row.condition_pattern, row.stimulus)
                          for row in eachrow(control_data)]

# Add a unique id for each cell
transform!(control_data,
    [:condition, :run, :cell_number] =>
    ByRow((cond, run, cell) -> "$(cond[1:5])_$(run)_$(cell)") => :cell_id)

# ---------------------------------------------------------------------------
# Rolling rate helpers
# ---------------------------------------------------------------------------

"""
    get_centered_rolling_rate!(df::DataFrame, k::Int) -> DataFrame

Calculate rolling mean of each cell over a window of k trials. This window extends k÷2 trials both before and after each trial

Args
    df: dataframe with cell_id and contract columns.
    k: window size.

Returns
    Df with a new column centered_rolling_rate_k.
"""
function get_centered_rolling_rate!(df::DataFrame, k::Int)
    col  = Symbol("centered_rolling_rate_$(k)")
    half = div(k, 2)
    transform!(groupby(df, [:condition, :cell_id])) do group
        n = nrow(group)
        # use nanmean to handle missing values in columns
        group[!, col] = [nanmean(group.contract[max(1, i - half):min(n, i + half)])
                         for i in 1:n]
        return group
    end
    return df
end

"""
    get_rolling_rate!(df::DataFrame, k::Int) -> DataFrame

Calculate rolling mean of each cell over a window of k trials.

Args
    df: dataframe with cell_id and contract columns.
    k: window size.

Returns
    Df with a new column backward_rolling_rate_k.
"""
function get_rolling_rate!(df::DataFrame, k::Int)
    col = Symbol("rolling_rate_$(k)")
    transform!(groupby(df, [:condition, :cell_id])) do group
        n = nrow(group)
        # use nanmean to handle missing values in columns
        group[!, col] = [nanmean(group.contract[max(1, i - k + 1):i]) for i in 1:n]
        return group
    end
    return df
end

# ---------------------------------------------------------------------------
# Mathematical models
# ---------------------------------------------------------------------------

"""
    model_strong_response(n, s0, αs) -> Float64

Model cells' responses to strong stimuli using an exponential curve
    rₛ(n) = s0 · exp(−αs · n)

Args
    n: trial number
    s0: initial strong response
    αs: strong habituation rate constant

Returns
    Predicted response probability at trial n.
"""
model_strong_response(n, s0, αs) = s0 * exp(-αs * n)

"""
    model_weak_response(n, s0, w0, αw, αs, αc) -> Float64

Model cells' responses to weak tap as a result of associative drive and habituation process

Args
    n:  trial number
    s0: initial strong-tap response magnitude
    w0: initial weak-tap response magnitude
    αw: weak habituation rate constant
    αs: strong habituation rate constant
    αc: associative coupling strength
"""
function model_weak_response(n, s0, w0, αw, αs, αc)
    if αw ≈ αs
        # CS-US share similar strength
        return exp(-αs * n) * (s0 * αc * n + w0)
    else
        # Weak-strong pairing
        assoc_term = (s0 * αc) / (αw - αs) * (exp(-αs * n) - exp(-αw * n))
        hab_term   = w0 * exp(-αw * n)
        return assoc_term + hab_term
    end
end

# ---------------------------------------------------------------------------
# Curve fitting
# ---------------------------------------------------------------------------

"""
    fit_strong_habituation(trial_numbers, response_data) -> NamedTuple

Fit population data to get strong initial response and habituation rate

Args
    trial_numbers: number of trials (1-60)
    response_data: Mean contraction response

Returns
    A NamedTuple with fields s0 (initial response), αs (habituation rate),
    fit, and predictions (model-predicted values at each trial).
"""
function fit_strong_habituation(trial_numbers, response_data)
    @. model(n, p) = p[1] * exp(-p[2] * n)

    # Initial guesses
    p0  = [maximum(response_data), 0.1]

    # Start at 0 so trial_number .- 1
    fit = curve_fit(model, trial_numbers .- 1, response_data, p0)

    s0, αs = fit.param
    return (s0 = s0, αs = αs, fit = fit,
            predictions = model_strong_response.(trial_numbers .- 1, s0, αs))
end

"""
    fit_weak_habituation(trial_numbers, response_data, s0_init, αs_init) -> NamedTuple

Fit population data to get

Args
    trial_numbers: Vector of stimulus numbers (1-indexed).
    response_data: Vector of population-mean contraction proportions.
    s0_init: Initial value for s0, typically from fit_strong_habituation.
    αs_init: Initial value for αs, typically from fit_strong_habituation.

Returns
    A NamedTuple with fields s0, w0, αw, αs, αc, δ (fitted parameter values),
    fit (raw LsqFit result), and predictions (model-predicted values at each trial).
"""
function fit_weak_habituation(trial_numbers, response_data, s0_init, αs_init)
    @. model(n, p) = ((p[1] * p[5]) / (p[3] + p[5]) *
        (exp(-p[4] * n) - exp(-(p[3] + p[4] + p[5]) * n)) +
         p[2] * exp(-(p[3] + p[4] + p[5]) * n))

    # p = [s0, w0, δ, αs, αc]  with αw = αs + δ + αc
    # Initial guesses
    p0    = [s0_init, response_data[1], 0.01,   αs_init, 0.5  ]
    # Lower bounds
    lower = [0.7,     0.01,             0.0001, 0.0001,  0.05 ]
    # Upper bounds
    upper = [1.0,     Inf,              Inf,    Inf,     Inf  ]

    # Fit models
    fit   = curve_fit(model, trial_numbers .- 1, response_data, p0; lower, upper)

    s0_f, w0, δ, αs_f, αc = fit.param
    αw = αs_f + δ + αc
    return (w0 = w0, αw = αw, δ = δ, αc = αc, s0 = s0_f, αs = αs_f, fit = fit,
            predictions = model_weak_response.(trial_numbers .- 1, s0_f, w0, αw, αs_f, αc))
end

# ---------------------------------------------------------------------------
# Bayesian quadratic regression
# ---------------------------------------------------------------------------

"""
    linear_regression(x, y)

Turing probabilistic model for Bayesian quadratic regression of learning curves

Args
    x: stimulus number
    y: population-mean contraction proportions.

Returns
    A Turing model object. Sample with NUTS via sample_quad.
"""
@model function linear_regression(x, y)
    #normalize stimulus number to [0,1]
    x_norm = x ./ maximum(x)
    σ  ~ truncated(Normal(0, 0.2); lower = 0)

    # β0 is centered near the expected baseline response
    b0 ~ Normal(0.4, 0.3)

    # flat priors on the β1 and β2
    b1 ~ Normal(0, 1)
    b2 ~ Normal(0, 1)

    mu = b0 .+ x_norm .* b1 .+ x_norm .^ 2 .* b2
    return y ~ MvNormal(mu, σ^2 * I)
end

"""
    sample_quad(condition::String, max_stim::Int, population_df::DataFrame) -> Dict

Run Bayesian quadratic regression for each condition and returne the
posterior distribution of the learning curve peak location.

Args
    condition: name of condition
    max_stim: number of stimulus to fit (10)
    population_df: population data

Returns
    A Dict with keys b0, b1, b2 (posterior sample vectors), chain (full Turing chain), and p (posterior probability that −b1/(2b2) ∈ (0,1) and b2 < 0
"""
function sample_quad(condition::String, max_stim::Int, population_df::DataFrame)
    # Filter condition
    dt    = filter(r -> r.condition == condition && r.stimulus in 1:max_stim, population_df)
    mdl   = linear_regression(dt.stimulus, dt.prop)

    chain = sample(mdl, NUTS(), N_BAYES_SAMPLES)
    b0 = vec(Array(chain[:b0]))
    b1 = vec(Array(chain[:b1]))
    b2 = vec(Array(chain[:b2]))
    peak = -b1 ./ (2 .* b2)
    p_nonmono = mean((0 .< peak .< 1) .& (b2 .< 0))
    return Dict(:b0 => b0, :b1 => b1, :b2 => b2, :chain => chain, :p => p_nonmono)
end

# ---------------------------------------------------------------------------
# Summary / aggregation
# ---------------------------------------------------------------------------

"""
    summarize_by_var(df, y_col, x_vars) -> Vector{DataFrame}

Summarize data with means and CI by looping through interested variables

Args
    df: dataframe
    y_col:  name of the outcome column (e.g. :trial_to_acquisition)
    x_vars: vector of grouping variable symbols (e.g. [:isi, :iti, :informativeness])

Returns
    A dataframe has columns x_value (the ISI/ITI/informativeness value for that group), mean_val, n_cells (number of cells), and standard error (se).
"""
function summarize_by_var(df::DataFrame, y_col::Symbol, x_vars::Vector{Symbol})
    [combine(groupby(df, var)) do group
        vals = collect(skipmissing(group[!, y_col]))
        n  = length(vals)
        m  = n > 0 ? mean(vals) : missing
        se = n > 1 ? 1.96 * std(vals) / sqrt(n) : missing
        (x_value = first(group[!, var]), mean_val = m, n_cells = n, se = se)
    end for var in x_vars]
end

"""
    summarize_learner_prop(df, x_vars) -> Vector{DataFrame}

Summarize proportion of learner cells and CI by looping through interested variables.

Args
    df: dataframe with a learner Bool column
    x_vars: vector of grouping variable symbols (e.g. [:isi, :iti, :informativeness])

Returns
    A dataframe with columns x_value (the ISI/ITI/informativeness value for that group), prop_learners, n_cells (total cells), and standard error (se).
"""
function summarize_learner_prop(df::DataFrame, x_vars::Vector{Symbol})
    [combine(groupby(df, var)) do group
        n  = nrow(group)
        p  = mean(group.learner)
        se = 1.96 * sqrt(p * (1 - p) / n)
        (x_value = first(group[!, var]), prop_learners = p, n_cells = n, se = se)
    end for var in x_vars]
end

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

"""
    decompose_weak_response(n, s0, w0, αw, αs, αc) -> Tuple{Float64, Float64}

Decompose the total weak-tap response into its two constituent processes. This function is used for calculate_process_dominance().

Args
    n, s0, w0, αw, αs, αc: Same as model_weak_response.

Returns
    A tuple (term1, term2) where term1 is the associative coupling and term2 is the habituation contribution
"""
function decompose_weak_response(n, s0, w0, αw, αs, αc)
    # Break the formula into associative drive and habituation
    if αw ≈ αs
        term1 = exp(-αs * n) * s0 * αc * n
        term2 = exp(-αs * n) * w0
    else
        term1 = (s0 * αc) / (αw - αs) * (exp(-αs * n) - exp(-αw * n))
        term2 = w0 * exp(-αw * n)
    end
    return term1, term2
end

"""
    calculate_process_dominance(n, s0, w0, αw, αs, αc) -> Float64

Calculate what whether assciative or habituation dominates at that time (for plotting purposes)

Args
    n, s0, w0, αw, αs, αc: Same as model_weak_response.

Returns
    +1 (associative coupling dominates), 0 (equal strength), or −1 (habituation dominates)
"""
function calculate_process_dominance(n, s0, w0, αw, αs, αc)
    term1, term2 = decompose_weak_response(n, s0, w0, αw, αs, αc)
    total = term1 + term2
    total ≈ 0 && return 0.0
    raw = (term1 - term2) / total
    return (2 / π) * atan(2 * raw) #Apply arctan scaling for smoother transitions
end

# ---------------------------------------------------------------------------
# Shared computed data
# ---------------------------------------------------------------------------

get_rolling_rate!(data, ROLLING_WINDOW)

population = combine(groupby(data, [:condition, :stimulus])) do group
    n = length(unique(group.cell_id))
    c = nansum(group.contract)
    (contracted = c, num_cells = n, prop = c / n, is_strong = false)
end

control_population = combine(groupby(control_data, [:condition, :stimulus])) do group
    n = length(unique(group.cell_id))
    c = nansum(group.contract)
    (contracted = c, num_cells = n, prop = c / n,
     condition_pattern = first(group.condition_pattern),
     is_strong = first(group.is_strong))
end

data_population = filter(r -> r.condition in
    ["ww_ISI1_ITI45", "ws_ISI1_ITI45", "w_ISI1_ITI59", "hab_ws_ISI1_ITI59"], population)

all_control = vcat(control_population, data_population, cols = :union)

individual_runs = combine(groupby(data, [:condition, :stimulus, :run])) do group
    n = length(unique(group.cell_id))
    c = nansum(group.contract)
    (contracted = c, num_cells = n, prop = c / n, run = first(group.run))
end

control_individual_runs = combine(groupby(control_data, [:condition, :stimulus, :run])) do group
    n = length(unique(group.cell_id))
    c = nansum(group.contract)
    (contracted = c, num_cells = n, prop = c / n, run = first(group.run))
end

all_individual_runs = vcat(individual_runs, control_individual_runs, cols = :union)

# Model fits
strong_pop = filter(r -> r.condition == "s_ISI1_ITI45", control_population)
strong_fit = fit_strong_habituation(strong_pop.stimulus, strong_pop.prop)

fit_conditions = [("ws_ISI1_ITI45", "#D4A5A5", "Weak-strong"),
                  ("hab_ws_ISI1_ITI59", :lightskyblue2, "Hab ws")]

fitted_params = Dict(
    cond => fit_weak_habituation(
        filter(r -> r.condition == cond, population).stimulus,
        filter(r -> r.condition == cond, population).prop,
        strong_fit.s0, strong_fit.αs)
    for (cond, _, _) in fit_conditions)

# Single-cell data
cell_data = combine(groupby(data, [:condition, :cell_id])) do group
    responded_to_first = any(group.stimulus .== 1 .&& group.contract .== 1)
    above   = group.rolling_rate_8 .>= LEARNER_THRESHOLD
    crossed = any(above)

    if crossed
        trial_acquired = group.stimulus[findfirst(above)]
        trial_lost     = group.stimulus[findlast(above)]
        duration       = trial_lost - trial_acquired
    else
        trial_acquired, trial_lost, duration = missing, missing, missing
    end

    max_rate   = maximum(group.rolling_rate_8)
    trial_peak = group.stimulus[argmax(group.rolling_rate_8)]
    total_resp = sum(group.contract)
    isi, iti   = group.isi[1], group.iti[1]

    return (
        trial_to_acquisition     = crossed ? log2(trial_acquired) : missing,
        trial_at_loss            = crossed ? log2(trial_lost) : missing,
        duration_above_threshold = duration,
        trial_at_peak            = log2(trial_peak),
        peak_local_rate          = max_rate,
        informativeness          = log2((iti + isi) / isi),
        iti = log2(iti), isi = log2(isi),
        responsiveness           = total_resp,
        first_stim_response      = responded_to_first,
        threshold_crossed        = crossed)
end

# Keep only weak-strong conditions for single-cell analyses
cell_data = filter(r -> startswith(r.condition, "ws"), cell_data)

# A learner: initially non-responsive, capable of contracting, crosses threshold
cell_data.learner = (.!cell_data.first_stim_response .&&
                      cell_data.responsiveness .> 0 .&&
                      cell_data.threshold_crossed)

learners_group    = filter(r -> r.learner, cell_data)
learners_complete = filter(
    r -> r.learner && !ismissing(r.trial_to_acquisition) &&
         !ismissing(r.duration_above_threshold), cell_data)

println("common.jl loaded: $(nrow(data)) ws rows, $(nrow(control_data)) control rows")
