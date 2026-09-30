using CSV, Statistics, DataFrames, NaNStatistics, Random,
     GLMakie, CairoMakie, QuadGK, HypothesisTests, StatsBase,
     MixedModels, GLM, ColorSchemes, Turing, LinearAlgebra, LsqFit

import DataFrames: transform!

include(joinpath(@__DIR__, "..", "BayesianTTest.jl"))
using .BayesianTTest

const DATA_DIR    = joinpath(@__DIR__, "..", "data")
const FIGURES_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURES_DIR)

const ROLLING_WINDOW    = 8
const PEAK_THRESHOLDS   = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
const LEARNER_THRESHOLD = 0.60
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

data = CSV.read(joinpath(DATA_DIR, "ws_single.csv"), DataFrame)
transform!(data,
    [:run, :cell_number] => ByRow((run, cell) -> "$(run)_$(cell)") => :cell_id)
transform!(data,
    [:isi, :iti] => ByRow((isi, iti) -> (iti + isi) / isi) => :informativeness)

control_data = CSV.read(joinpath(DATA_DIR, "control.csv"), DataFrame)
control_data.condition_pattern = [split(cond, "_")[1] for cond in control_data.condition]

function is_strong_stimulus(pattern, stimulus_num)
    pattern == "s" && return true
    stimulus_num <= length(pattern) && return pattern[stimulus_num] == 's'
    return false
end

control_data.is_strong = [is_strong_stimulus(row.condition_pattern, row.stimulus)
                          for row in eachrow(control_data)]
transform!(control_data,
    [:condition, :run, :cell_number] =>
    ByRow((cond, run, cell) -> "$(cond[1:5])_$(run)_$(cell)") => :cell_id)

# ---------------------------------------------------------------------------
# Rolling rate helpers
# ---------------------------------------------------------------------------

function get_centered_rolling_rate!(df::DataFrame, k::Int)
    col  = Symbol("centered_rolling_rate_$(k)")
    half = div(k, 2)
    transform!(groupby(df, [:condition, :cell_id])) do group
        n = nrow(group)
        group[!, col] = [nanmean(group.contract[max(1, i - half):min(n, i + half)])
                         for i in 1:n]
        return group
    end
    return df
end

function get_rolling_rate!(df::DataFrame, k::Int)
    col = Symbol("rolling_rate_$(k)")
    transform!(groupby(df, [:condition, :cell_id])) do group
        n = nrow(group)
        group[!, col] = [nanmean(group.contract[max(1, i - k + 1):i]) for i in 1:n]
        return group
    end
    return df
end

# ---------------------------------------------------------------------------
# Mathematical models
# ---------------------------------------------------------------------------

model_strong_response(n, s0, αs) = s0 * exp(-αs * n)

function model_weak_response(n, s0, w0, αw, αs, αc)
    if αw ≈ αs
        return exp(-αs * n) * (s0 * αc * n + w0)
    else
        assoc_term = (s0 * αc) / (αw - αs) * (exp(-αs * n) - exp(-αw * n))
        hab_term   = w0 * exp(-αw * n)
        return assoc_term + hab_term
    end
end

# ---------------------------------------------------------------------------
# Curve fitting
# ---------------------------------------------------------------------------

function fit_strong_habituation(trial_numbers, response_data)
    @. model(n, p) = p[1] * exp(-p[2] * n)
    p0  = [maximum(response_data), 0.1]
    fit = curve_fit(model, trial_numbers .- 1, response_data, p0)
    s0, αs = fit.param
    return (s0 = s0, αs = αs, fit = fit,
            predictions = model_strong_response.(trial_numbers .- 1, s0, αs))
end

function fit_weak_habituation(trial_numbers, response_data, s0_init, αs_init)
    @. model(n, p) = ((p[1] * p[5]) / (p[3] + p[5]) *
        (exp(-p[4] * n) - exp(-(p[3] + p[4] + p[5]) * n)) +
         p[2] * exp(-(p[3] + p[4] + p[5]) * n))

    p0    = [s0_init, response_data[1], 0.01,   αs_init, 0.5  ]
    lower = [0.7,     0.01,             0.0001, 0.0001,  0.05 ]
    upper = [1.0,     Inf,              Inf,    Inf,     Inf  ]

    fit   = curve_fit(model, trial_numbers .- 1, response_data, p0; lower, upper)
    s0_f, w0, δ, αs_f, αc = fit.param
    αw = αs_f + δ + αc
    return (w0 = w0, αw = αw, δ = δ, αc = αc, s0 = s0_f, αs = αs_f, fit = fit,
            predictions = model_weak_response.(trial_numbers .- 1, s0_f, w0, αw, αs_f, αc))
end

# ---------------------------------------------------------------------------
# Bayesian quadratic regression
# ---------------------------------------------------------------------------

@model function linear_regression(x, y)
    x_norm = x ./ maximum(x)
    σ  ~ truncated(Normal(0, 0.2); lower = 0)
    b0 ~ Normal(0.4, 0.3)
    b1 ~ Normal(0, 1)
    b2 ~ Normal(0, 1)
    mu = b0 .+ x_norm .* b1 .+ x_norm .^ 2 .* b2
    return y ~ MvNormal(mu, σ^2 * I)
end

function sample_quad(condition::String, max_stim::Int, population_df::DataFrame)
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

function summarize_by_var(df::DataFrame, y_col::Symbol, x_vars::Vector{Symbol})
    [combine(groupby(df, var)) do group
        vals = collect(skipmissing(group[!, y_col]))
        n  = length(vals)
        m  = n > 0 ? mean(vals) : missing
        se = n > 1 ? 1.96 * std(vals) / sqrt(n) : missing
        (x_value = first(group[!, var]), mean_val = m, n_cells = n, se = se)
    end for var in x_vars]
end

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

function decompose_weak_response(n, s0, w0, αw, αs, αc)
    if αw ≈ αs
        term1 = exp(-αs * n) * s0 * αc * n
        term2 = exp(-αs * n) * w0
    else
        term1 = (s0 * αc) / (αw - αs) * (exp(-αs * n) - exp(-αw * n))
        term2 = w0 * exp(-αw * n)
    end
    return term1, term2
end

function calculate_process_dominance(n, s0, w0, αw, αs, αc)
    term1, term2 = decompose_weak_response(n, s0, w0, αw, αs, αc)
    total = term1 + term2
    total ≈ 0 && return 0.0
    raw = (term1 - term2) / total
    return (2 / π) * atan(2 * raw)
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

cell_data = filter(r -> startswith(r.condition, "ws"), cell_data)
cell_data.learner = (.!cell_data.first_stim_response .&&
                      cell_data.responsiveness .> 0 .&&
                      cell_data.threshold_crossed)

learners_group    = filter(r -> r.learner, cell_data)
learners_complete = filter(
    r -> r.learner && !ismissing(r.trial_to_acquisition) &&
         !ismissing(r.duration_above_threshold), cell_data)

println("common.jl loaded: $(nrow(data)) ws rows, $(nrow(control_data)) control rows")
