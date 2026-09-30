include("common.jl")

using CategoricalArrays

kernels         = [3, 5, 8, 12]
peak_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
colors          = ["black", "gray"]
linestyles      = [:solid, :dash, :dot]

interest_conditions = ["ws_ISI1_ITI45", "ww_ISI1_ITI45"]
filtered_data       = filter(r -> r.condition in interest_conditions, data)

# ============================================================================
# Figure 1: Learner proportions across rolling window sizes (ws vs ww)
# ============================================================================

compare_rolling_window = DataFrame()
compare                = DataFrame()
kernel_results         = Dict()

fig_all_kernels = Figure(size = (1200, 800))

for (idx, k) in enumerate(kernels)
    row = ceil(Int, idx / 2)
    col = ((idx - 1) % 2) + 1

    k_data = deepcopy(filtered_data)
    get_rolling_rate!(k_data, k)
    column_name = Symbol("rolling_rate_$(k)")

    cell_data_threshold = combine(groupby(k_data, [:condition, :cell_id])) do group
        responded_to_first = any(group.stimulus .== 1 .&& group.contract .== 1)
        max_rate = maximum(group[!, column_name])
        idx_of_max = findfirst(x -> x == max_rate, group[!, column_name])
        total_responsiveness = sum(group.contract)

        if idx_of_max === nothing || max_rate === nothing
            idx_of_max = 1
        end
        trial_at_peak = group.stimulus[idx_of_max]

        (max_rate = max_rate,
         responded_to_first = responded_to_first,
         trial_at_peak = trial_at_peak,
         responsiveness = total_responsiveness)
    end

    non_monotonic = filter(r -> r.responded_to_first == false && r.responsiveness > 0,
                           cell_data_threshold)

    non_monotonic.window  = fill(k, nrow(non_monotonic))
    non_monotonic.learner = non_monotonic.max_rate .>= LEARNER_THRESHOLD
    append!(compare_rolling_window, non_monotonic)

    if k == 8
        for t in peak_thresholds
            temp = copy(non_monotonic)
            temp.threshold .= t
            temp.learner = temp.max_rate .>= t
            append!(compare, temp)
        end
    end

    results = combine(groupby(non_monotonic, :condition)) do group
        DataFrame(threshold     = peak_thresholds,
                  n_cells       = nrow(group),
                  n_learners    = [sum(group.max_rate .>= t) for t in peak_thresholds],
                  prop_learners = [sum(group.max_rate .>= t) / nrow(group) for t in peak_thresholds])
    end

    transform!(results,
        [:prop_learners, :n_cells] =>
        ByRow((p, n) -> 1.96 * sqrt(p * (1 - p) / n)) => :se)

    kernel_results[k] = results

    ax = Axis(fig_all_kernels[row, col],
              xlabel = "Peak threshold",
              ylabel = col == 1 ? "Proportion of learners" : "",
              title  = "Rolling window = $k", titlesize = 22)

    for (i, condition) in enumerate(interest_conditions)
        condition_data = filter(r -> r.condition == condition, results)
        sort!(condition_data, :threshold)
        lines!(ax, condition_data.threshold, condition_data.prop_learners,
               linewidth = 3, color = colors[i], linestyle = linestyles[i])
        errorbars!(ax, condition_data.threshold, condition_data.prop_learners,
                   condition_data.se, color = colors[i], linewidth = 2, whiskerwidth = 2)
        scatter!(ax, condition_data.threshold, condition_data.prop_learners,
                 markersize = 15, color = colors[i],
                 label = "$condition (n=$(condition_data.n_cells[1]))")
    end

    if idx == 1
        axislegend(ax, position = :rt, labelsize = 16, framevisible = false)
    end

    ylims!(ax, 0, 1)
    ax.yticks = 0:0.1:1
    ax.xticks = peak_thresholds
end

Label(fig_all_kernels[0, :],
      "Proportion of learners in nonmonotonic cells",
      fontsize = 26, font = :bold)

# condition × threshold mixed model
compare.condition = categorical(compare.condition)
compare.cell_id   = categorical(compare.cell_id)
m_cond_thresh = fit(MixedModel,
    @formula(learner ~ condition * threshold + (1 | cell_id)), compare, Bernoulli())
println("Condition × threshold mixed model:")
println(m_cond_thresh)

# ============================================================================
# Figure 2: Backward vs centered rolling comparison
# ============================================================================

interest_condition = "ws_ISI1_ITI45"
rolling_data       = filter(r -> r.condition == interest_condition, data)

kernel_results_compare = Dict()
fig_rolling_compare    = Figure(size = (1200, 700))

for (idx, k) in enumerate(kernels)
    row = ceil(Int, idx / 2)
    col = ((idx - 1) % 2) + 1

    backward_data = deepcopy(rolling_data)
    get_rolling_rate!(backward_data, k)
    backward_data.method = fill("Backward", nrow(backward_data))

    centered_data = deepcopy(rolling_data)
    get_centered_rolling_rate!(centered_data, k)
    centered_data.method = fill("Centered", nrow(centered_data))

    results_combined = DataFrame()

    for (method_name, method_data, col_name) in [
        ("Backward", backward_data, Symbol("rolling_rate_$(k)")),
        ("Centered", centered_data, Symbol("centered_rolling_rate_$(k)"))]

        cell_data_threshold = combine(groupby(method_data, [:condition, :cell_id])) do group
            responded_to_first = any(group.stimulus .== 1 .&& group.contract .== 1)
            max_rate = maximum(group[!, col_name])
            idx_of_max = findfirst(x -> x == max_rate, group[!, col_name])
            total_responsiveness = sum(group.contract)

            if idx_of_max === nothing || max_rate === nothing
                idx_of_max = 1
            end
            trial_at_peak = group.stimulus[idx_of_max]

            (max_rate = max_rate,
             responded_to_first = responded_to_first,
             trial_at_peak = trial_at_peak,
             responsiveness = total_responsiveness,
             method = method_name)
        end

        non_monotonic = filter(r -> r.responded_to_first == false && r.responsiveness > 0,
                               cell_data_threshold)

        results = combine(groupby(non_monotonic, [:condition, :method])) do group
            DataFrame(threshold     = peak_thresholds,
                      n_cells       = nrow(group),
                      n_learners    = [sum(group.max_rate .>= t) for t in peak_thresholds],
                      prop_learners = [sum(group.max_rate .>= t) / nrow(group) for t in peak_thresholds])
        end

        transform!(results,
            [:prop_learners, :n_cells] =>
            ByRow((p, n) -> 1.96 * sqrt(p * (1 - p) / n)) => :se)

        append!(results_combined, results)
    end

    kernel_results_compare[k] = results_combined

    ax = Axis(fig_rolling_compare[row, col],
              xlabel = "Peak threshold",
              ylabel = col == 1 ? "Proportion of learners" : "",
              title  = "k = $k")

    methods = ["Backward", "Centered"]
    for (i, method) in enumerate(methods)
        method_data = filter(r -> r.method == method, results_combined)
        sort!(method_data, :threshold)

        lines!(ax, method_data.threshold, method_data.prop_learners,
               linewidth = 3, color = colors[i], linestyle = linestyles[i])
        errorbars!(ax, method_data.threshold, method_data.prop_learners,
                   method_data.se, color = colors[i], linewidth = 2, whiskerwidth = 2)
        scatter!(ax, method_data.threshold, method_data.prop_learners,
                 markersize = 15, color = colors[i], label = "$method")
    end

    if idx == 1
        axislegend(ax, position = :rt, framevisible = false)
    end

    ylims!(ax, 0, 1)
    ax.yticks = 0:0.25:1
    ax.xticks = peak_thresholds
end

# k=5 vs k=8 combined panel
ax5 = Axis(fig_rolling_compare[1:2, 3],
           xlabel = "Peak threshold",
           ylabel = "Proportion of learners",
           title  = "Backward rolling",
           titlegap = 20, xgridvisible = false, ygridvisible = false,
           topspinevisible = false, rightspinevisible = false)
colrs = [:magenta4, :teal]
for (i, k) in enumerate([5, 8])
    method_data = filter(r -> r.method == "Backward", kernel_results_compare[k])
    sort!(method_data, :threshold)
    lines!(ax5, method_data.threshold, method_data.prop_learners,
           linewidth = 3, color = colrs[i])
    errorbars!(ax5, method_data.threshold, method_data.prop_learners,
               method_data.se, color = colrs[i], linewidth = 2, whiskerwidth = 2)
    scatter!(ax5, method_data.threshold, method_data.prop_learners,
             markersize = 15, color = colrs[i], label = "k=$k")
end

axislegend(ax5, position = :rt, framevisible = false)
ylims!(ax5, 0, 1)
ax5.yticks = 0:0.1:1
ax5.xticks = peak_thresholds

Label(fig_rolling_compare[0, :],
      "Effect of window size and averaging method on learner detection",
      fontsize = 26, font = :bold, padding = (0, 0, 0, 20))

# k=5 vs k=8 mixed model
rolling_compare_df = filter(
    r -> r.window in [5, 8] && r.condition == "ws_ISI1_ITI45",
    compare_rolling_window)
model_k5_vs_k8 = fit(MixedModel,
    @formula(learner ~ window + (1 | cell_id)), rolling_compare_df, Bernoulli())
println("\nk=5 vs k=8 mixed model:")
println(model_k5_vs_k8)

# Save both figures
CairoMakie.activate!()
save(joinpath(FIGURES_DIR, "figS3_kernel_compare.png"), fig_all_kernels, px_per_unit = 3)
save(joinpath(FIGURES_DIR, "figS3_rolling_compare.png"), fig_rolling_compare, px_per_unit = 3)
GLMakie.activate!()
println("Saved figS3_kernel_compare.png and figS3_rolling_compare.png")
