include("common.jl")

col_titles  = ["ISI", "ITI", "Informativeness"]
col_xlabels = ["ISI (log₂ s)", "ITI (log₂ s)", "Informativeness (log₂)"]
row_ylabels = ["Proportion of learners", "Log₂ Trials to acquisition",
               "Peak local rate", "Duration above threshold"]
y_limits    = [(0.0, 0.35), (0.0, 5.0), (0.5, 1.1), (0.0, 20.0)]
x_vars      = [:isi, :iti, :informativeness]

# Fit all models
models_learner = [
    glm(@formula(learner ~ isi),              cell_data, Bernoulli(), LogitLink()),
    glm(@formula(learner ~ iti),              cell_data, Bernoulli(), LogitLink()),
    glm(@formula(learner ~ informativeness),  cell_data, Bernoulli(), LogitLink())]

acq_data = dropmissing(learners_group, :trial_to_acquisition)
models_acq = [
    lm(@formula(trial_to_acquisition ~ isi),              acq_data),
    lm(@formula(trial_to_acquisition ~ iti),              acq_data),
    lm(@formula(trial_to_acquisition ~ informativeness),  acq_data)]

models_peak = [
    glm(@formula(peak_local_rate ~ isi),              learners_group, Gamma(), LogLink()),
    glm(@formula(peak_local_rate ~ iti),              learners_group, Gamma(), LogLink()),
    glm(@formula(peak_local_rate ~ informativeness),  learners_group, Gamma(), LogLink())]

dur_data = dropmissing(learners_group, :duration_above_threshold)
models_dur = [
    glm(@formula(duration_above_threshold ~ isi),              dur_data, Poisson(), LogLink()),
    glm(@formula(duration_above_threshold ~ iti),              dur_data, Poisson(), LogLink()),
    glm(@formula(duration_above_threshold ~ informativeness),  dur_data, Poisson(), LogLink())]

# Put them all in a dataset
all_models   = [models_learner, models_acq, models_peak, models_dur]
row_datasets = [cell_data, acq_data, learners_group, dur_data]

# Use Wald test to draw line if coef. is significant
is_sig(model) = abs(coef(model)[2] / stderror(model)[2]) > 1.96

function pred_y(model, x_range, row)
    β0, β1 = coef(model)[1], coef(model)[2]
    if row == 1
        return 1 ./ (1 .+ exp.(-(β0 .+ β1 .* x_range)))
    elseif row == 2
        return β0 .+ β1 .* x_range
    else
        return exp.(β0 .+ β1 .* x_range)
    end
end

figS4 = Figure(size = (1100, 1300))

for row in 1:4, col in 1:3
    var     = x_vars[col]
    dataset = row_datasets[row]

    summary = combine(groupby(dataset, var)) do group
        x_val = first(group[!, var])

        if row == 1 # Plot prop of learners
            n  = nrow(group)
            p  = mean(group.learner)
            se = 1.96 * sqrt(p * (1 - p) / n)
            (x_value = x_val, y_value = p, se = se)
        elseif row == 2 # Plot trials to acquisition
            vals = collect(skipmissing(group.trial_to_acquisition))
            n = length(vals)
            m  = n > 0 ? mean(vals) : missing
            se = n > 1 ? 1.96 * std(vals) / sqrt(n) : missing
            (x_value = x_val, y_value = m, se = se)
        elseif row == 3 # Plot peak local rate
            vals = group.peak_local_rate
            n  = length(vals)
            m  = mean(vals)
            se = 1.96 * std(vals) / sqrt(n)
            (x_value = x_val, y_value = m, se = se)
        else
            vals = collect(skipmissing(group.duration_above_threshold))
            n = length(vals)
            m  = n > 0 ? mean(vals) : missing
            se = n > 1 ? 1.96 * std(vals) / sqrt(n) : missing
            (x_value = x_val, y_value = m, se = se)
        end
    end

    valid = .!ismissing.(summary.y_value)

    ax = Axis(figS4[row, col],
              xlabel = col_xlabels[col],
              ylabel = col == 1 ? row_ylabels[row] : "",
              title  = row == 1 ? col_titles[col] : "")

    errorbars!(ax,
        summary.x_value[valid],
        collect(Float64, summary.y_value[valid]),
        collect(Float64, summary.se[valid]);
        whiskerwidth = 10, color = :steelblue)

    scatter!(ax,
        summary.x_value[valid],
        collect(Float64, summary.y_value[valid]);
        markersize = 15, color = :steelblue)

    model = all_models[row][col]
    if is_sig(model)
        x_data  = dataset[!, var]
        x_range = range(minimum(x_data), maximum(x_data); length = 100)
        lines!(ax, collect(x_range), pred_y(model, x_range, row);
               color = :steelblue, linewidth = 2.5, linestyle = :dash)
    end

    ylims!(ax, y_limits[row])
end

colgap!(figS4.layout, 20)
rowgap!(figS4.layout, 15)

CairoMakie.activate!()
save(joinpath(FIGURES_DIR, "figS4_temporal_breakdown.png"), figS4, px_per_unit = 3)
GLMakie.activate!()
println("Saved figS4_temporal_breakdown.png")
