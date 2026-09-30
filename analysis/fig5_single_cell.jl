include("common.jl")

# ---------------------------------------------------------------------------
# Learner proportions by threshold (ws vs ww)
# ---------------------------------------------------------------------------

learner_props_by_threshold = let
    interest_conditions = ["ws_ISI1_ITI45", "ww_ISI1_ITI45"]
    k_data = filter(r -> r.condition in interest_conditions, data)
    cell_peaks = combine(groupby(k_data, [:condition, :cell_id])) do group
        (max_rate = maximum(group.rolling_rate_8),
         responded_to_first = any(group.stimulus .== 1 .&& group.contract .== 1),
         responsiveness = sum(group.contract))
    end
    nonmono = filter(r -> !r.responded_to_first && r.responsiveness > 0, cell_peaks)
    results = combine(groupby(nonmono, :condition)) do group
        DataFrame(threshold = PEAK_THRESHOLDS, n_cells = nrow(group),
            prop_learners = [sum(group.max_rate .>= t) / nrow(group) for t in PEAK_THRESHOLDS])
    end
    transform!(results,
        [:prop_learners, :n_cells] =>
        ByRow((p, n) -> 1.96 * sqrt(p * (1 - p) / n)) => :se)
end

# ---------------------------------------------------------------------------
# GLM models
# ---------------------------------------------------------------------------

model_iti_learner  = glm(@formula(learner ~ iti), cell_data, Bernoulli(), LogitLink())
model_isi_peak     = glm(@formula(peak_local_rate ~ isi), learners_group, Gamma(), LogLink())
model_info_peak    = glm(@formula(peak_local_rate ~ informativeness), learners_group, Gamma(), LogLink())
model_duration_acq = fit(MixedModel,
    @formula(duration_above_threshold ~ exp2(trial_to_acquisition) + (1 | cell_id)),
    learners_complete, Poisson())

# ---------------------------------------------------------------------------
# Summary statistics for Panels D-G
# ---------------------------------------------------------------------------

x_vars     = [:isi, :iti, :informativeness]
var_names  = ["ISI", "ITI", "Informativeness"]
var_colors = [:purple, :darkgoldenrod2, :teal]

learner_summaries     = summarize_learner_prop(cell_data, x_vars)
acquisition_summaries = summarize_by_var(learners_group, :trial_to_acquisition, x_vars)
peak_summaries        = summarize_by_var(learners_group, :peak_local_rate, x_vars)
duration_summaries    = summarize_by_var(learners_group, :duration_above_threshold, x_vars)

# ---------------------------------------------------------------------------
# Figure 5
# ---------------------------------------------------------------------------

fig5 = Figure(size = (1300, 1200))

#### Panel A: Example learner vs non-learner traces
Random.seed!(100)
learner_row    = shuffle(filter(r ->  r.learner, cell_data))[1, :]
nonlearner_row = shuffle(filter(r -> !r.learner, cell_data))[1, :]
learner_trace    = filter(r -> r.cell_id == learner_row.cell_id    && r.condition == learner_row.condition,    data)
nonlearner_trace = filter(r -> r.cell_id == nonlearner_row.cell_id && r.condition == nonlearner_row.condition, data)

Label(fig5[1, 1:2, TopLeft()], "A", padding = (0, 20, 20, 0), halign = :right)
ax4A = Axis(fig5[1, 1:2], xlabel = "Stimulus Number", ylabel = "Rolling response", title = "Example responses")
lines!(ax4A, learner_trace.stimulus,    learner_trace.rolling_rate_8,    color = "#C11C84", linewidth = 3, label = "Learner")
lines!(ax4A, nonlearner_trace.stimulus, nonlearner_trace.rolling_rate_8, color = "gray",    linewidth = 3, label = "Non-learner")
scatter!(ax4A, learner_trace.stimulus,    learner_trace.contract,    color = "#C11C84", markersize = 12, label = "Binary data")
scatter!(ax4A, nonlearner_trace.stimulus, nonlearner_trace.contract, color = "gray",    markersize = 8)
axislegend(ax4A, position = :rt, framevisible = false)
ylims!(ax4A, 0, 1.2)

#### Panel B: Proportion of learners across thresholds
Label(fig5[1, 3:4, TopLeft()], "B", padding = (0, 20, 20, 0), halign = :right)
ax4B = Axis(fig5[1, 3:4],
    xlabel = "Peak threshold (rolling window = $ROLLING_WINDOW)",
    ylabel = "Proportion of learners", title = "Nonmonotonic cells")
cond_colors = ["black", "gray"]
for (i, cond) in enumerate(["ws_ISI1_ITI45", "ww_ISI1_ITI45"])
    cd = sort(filter(r -> r.condition == cond, learner_props_by_threshold), :threshold)
    lines!(ax4B,     cd.threshold, cd.prop_learners, linewidth = 3, color = cond_colors[i], linestyle = LINE_STYLES[i])
    errorbars!(ax4B, cd.threshold, cd.prop_learners, cd.se, color = cond_colors[i], linewidth = 2, whiskerwidth = 2)
    scatter!(ax4B,   cd.threshold, cd.prop_learners, markersize = 12, color = cond_colors[i],
             label = "$(split(cond, "_")[1]) (n=$(cd.n_cells[1]))")
end
ylims!(ax4B, 0, 1)
ax4B.yticks = 0:0.1:1
ax4B.xticks = PEAK_THRESHOLDS
vspan!(ax4B, LEARNER_THRESHOLD - 0.02, LEARNER_THRESHOLD + 0.02, color = ("gray", 0.1))
axislegend(ax4B, position = :rt, framevisible = false)

#### Panel C: Trials to acquisition vs duration
Label(fig5[1, 5:6, TopLeft()], "C", padding = (0, 20, 20, 0), halign = :right)
ax4C = Axis(fig5[1, 5:6], xlabel = "Trials to acquisition", ylabel = "Duration of learning", title = "Learning speed vs duration")
trial_nums = 2 .^ learners_complete.trial_to_acquisition
scatter!(ax4C, trial_nums, learners_complete.duration_above_threshold, markersize = 12, color = :gray, alpha = 0.6)
b0_c, b1_c = coef(model_duration_acq)
trial_range = range(minimum(trial_nums), maximum(trial_nums), length = 100)
lines!(ax4C, trial_range, exp.(b0_c .+ b1_c .* trial_range), color = :gray, linewidth = 2.5, linestyle = :dash)

#### Panels D-G: Prop of learners, learning speed, consistency, duration
panel_defs = [
    ("D", "Proportion of learners",       "Proportion of learners",       (0, 0.3),  learner_summaries),
    ("E", "Learning speed",               "Log2 Trials to acquisition",   (1, 4),    acquisition_summaries),
    ("F", "Consistency of responses",      "Peak",                         (0.6, 1.0), peak_summaries),
    ("G", "Duration of learned responses", "Duration of learned responses", (0, 15),   duration_summaries)]

panel_positions = [(2, 1:3), (2, 4:6), (3, 1:3), (3, 4:6)]

for ((panel_label, title, ylabel, ylim, summaries), (row, cols)) in zip(panel_defs, panel_positions)
    Label(fig5[row, cols, TopLeft()], panel_label, padding = (0, 20, 20, 0), halign = :right)
    ax = Axis(fig5[row, cols], xlabel = "Values", ylabel = ylabel, title = title)

    for (var_idx, summary) in enumerate(summaries)
        if panel_label == "D"
            valid = trues(nrow(summary))
            errorbars!(ax, summary.x_value[valid], summary.prop_learners[valid], summary.se[valid],
                       whiskerwidth = 10, color = var_colors[var_idx])
            scatter!(ax, summary.x_value[valid], summary.prop_learners[valid],
                     markersize = 15, color = var_colors[var_idx], label = var_names[var_idx])
            if var_idx == 2
                iti_range = range(minimum(cell_data.iti), maximum(cell_data.iti), length = 100)
                b0_d, b1_d = coef(model_iti_learner)
                lines!(ax, iti_range, 1 ./ (1 .+ exp.(-(b0_d .+ b1_d .* iti_range))),
                       color = var_colors[var_idx], linewidth = 2.5, linestyle = :dash)
            end
        else
            valid = .!ismissing.(summary.mean_val)
            errorbars!(ax, summary.x_value[valid], summary.mean_val[valid], summary.se[valid],
                       whiskerwidth = 10, color = var_colors[var_idx])
            scatter!(ax, summary.x_value[valid], summary.mean_val[valid],
                     markersize = 15, color = var_colors[var_idx], label = var_names[var_idx])
            if panel_label == "F"
                if var_idx == 1
                    isi_range = range(minimum(learners_group.isi), maximum(learners_group.isi), length = 100)
                    b0_f, b1_f = coef(model_isi_peak)
                    lines!(ax, isi_range, exp.(b0_f .+ b1_f .* isi_range),
                           color = var_colors[var_idx], linewidth = 2.5, linestyle = :dash)
                elseif var_idx == 3
                    info_range = range(minimum(learners_group.informativeness), maximum(learners_group.informativeness), length = 100)
                    b0_f, b1_f = coef(model_info_peak)
                    lines!(ax, info_range, exp.(b0_f .+ b1_f .* info_range),
                           color = var_colors[var_idx], linewidth = 2.5, linestyle = :dash)
                end
            end
        end
    end

    ylims!(ax, ylim)
    axislegend(ax, position = :rt, labelsize = 14, framevisible = false)
end

CairoMakie.activate!()
save(joinpath(FIGURES_DIR, "fig5_single_cell.png"), fig5, px_per_unit = 3)
GLMakie.activate!()
println("Saved fig5_single_cell.png")
