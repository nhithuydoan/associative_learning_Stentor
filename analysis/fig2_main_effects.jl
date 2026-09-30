include("common.jl")

# ---------------------------------------------------------------------------
# Bayesian quadratic regression (also used by fig4)
# ---------------------------------------------------------------------------

isi_iti_conditions = ["ws_ISI1_ITI45",   "ww_ISI1_ITI45",   "hab_ws_ISI1_ITI59",
    "ws_ISI10_ITI45",  "ws_ISI20_ITI45",  "ws_ISI1_ITI59",
    "ws_ISI10_ITI59",  "ws_ISI10_ITI590"]

bayes_results = Dict(cond => sample_quad(cond, 10, population) for cond in isi_iti_conditions)

# ---------------------------------------------------------------------------
# Mixed-effects quadratic models
# ---------------------------------------------------------------------------

quadratic_conditions = ["ws_ISI1_ITI45",   "ww_ISI1_ITI45",   "hab_ws_ISI1_ITI59",
    "ws_ISI10_ITI45",  "ws_ISI20_ITI45",  "ws_ISI1_ITI59",
    "ws_ISI10_ITI59",  "ws_ISI10_ITI590"]

function fit_quadratic_mixed(condition::String)
    dt = filter(r -> r.condition == condition && r.stimulus in 1:10, all_individual_runs)
    dt.stimulus_sq = dt.stimulus .^ 2
    m  = fit(MixedModel,
             @formula(contracted ~ stimulus + stimulus_sq + (1 | run)), dt)
    ct = coeftable(m)
    (model          = m,
     beta_stimulus  = ct.cols[1][2],  p_stimulus  = ct.cols[4][2],
     beta_stimulus_sq = ct.cols[1][3], p_stimulus_sq = ct.cols[4][3])
end

quadratic_models = Dict(cond => fit_quadratic_mixed(cond) for cond in quadratic_conditions)

# ---------------------------------------------------------------------------
# Control t-tests
# ---------------------------------------------------------------------------

# wswww: compare stim 1 (before strong) vs stim 3 (after strong)
stim1_wswww = filter(r -> r.condition == "wswww_ISI1_ITI45" && r.stimulus == 1, all_individual_runs)
stim3_wswww = filter(r -> r.condition == "wswww_ISI1_ITI45" && r.stimulus == 3, all_individual_runs)
sort!(stim1_wswww, :run); sort!(stim3_wswww, :run)
ttest_arousal = OneSampleTTest(stim3_wswww.prop .- stim1_wswww.prop)

# wsssw: compare stim 1 (baseline) vs stim 5 (first weak after strong)
stim1_wsssw = filter(r -> r.condition == "wsssw_ISI1_ITI45" && r.stimulus == 1, all_individual_runs)
stim5_wsssw = filter(r -> r.condition == "wsssw_ISI1_ITI45" && r.stimulus == 5, all_individual_runs)
sort!(stim1_wsssw, :run); sort!(stim5_wsssw, :run)
ttest_sensitization = OneSampleTTest(stim5_wsssw.prop .- stim1_wsssw.prop)

# ---------------------------------------------------------------------------
# Figure 2: Main effect and controls (7 panels)
# ---------------------------------------------------------------------------

fig2_conditions = [
    "s_ISI1_ITI45",     "ws_ISI1_ITI45",    "ww_ISI1_ITI45",
    "wswww_ISI1_ITI45", "wsssw_ISI1_ITI45",
    "w_ISI1_ITI59",     "hab_ws_ISI1_ITI59"]

fig2 = Figure(size = (1200, 1400))
panel_grids = [
    fig2[1, 1:3], fig2[1, 4:6],
    fig2[2, 1:2], fig2[2, 3:4], fig2[2, 5:6],
    fig2[3, 1:3], fig2[3, 4:6]]
panel_labels = ["A", "B", "C", "D", "E", "F", "G"]

for (condition, grid_pos, label) in zip(fig2_conditions, panel_grids, panel_labels)
    cond_pop = filter(r -> r.condition == condition, all_control)
    # Calculate 95% CI across cells for this condition
    transform!(cond_pop,
        [:prop, :num_cells] =>
        ByRow((p, n) -> 1.96 * sqrt(p * (1 - p) / n)) => :sem)
    sort!(cond_pop, :stimulus)

    # Add condition name
    m = match(r"(\w+)_ISI(\d+)_ITI(\d+)", condition)
    pattern, isi, iti = m.captures
    title_str = pattern in ["ws", "ww", "hab_ws"] ?
        "$pattern, ISI = $isi s, ITI = $iti s" : "$pattern, ITI = $iti s"
    n_cells = cond_pop[1, :num_cells]

    Label(grid_pos[1, 1, TopLeft()], label, padding = (0, 20, 20, 0), halign = :right)

    ax = Axis(grid_pos[1, 1],
        xlabel    = "Stimulus number",
        ylabel    = label in ["A", "C", "F"] ? "Proportion of responses" : "",
        title     = "$title_str\n(n = $n_cells)",
        titlesize = label == "B" ? 24 : 20,
    )

    # Add individual experimental runs as gray lines
    cond_runs = filter(r -> r.condition == condition, all_individual_runs)
    for run_id in unique(cond_runs.run)
        run_data = sort(filter(r -> r.run == run_id, cond_runs), :stimulus)
        nrow(run_data) > 1 &&
            lines!(ax, run_data.stimulus, run_data.prop,
                   color = (:gray, 0.4), linewidth = 1)
    end

    # Mean ± CI
    if "is_strong" in names(cond_pop)
        strong_pts = filter(r ->  r.is_strong, cond_pop)
        weak_pts   = filter(r -> !r.is_strong, cond_pop)

        if nrow(strong_pts) > 0
            band!(ax,    strong_pts.stimulus,
                         strong_pts.prop .- strong_pts.sem,
                         strong_pts.prop .+ strong_pts.sem,
                         color = (:darkgoldenrod1, 0.3))
            scatter!(ax, strong_pts.stimulus, strong_pts.prop,
                     color = :darkgoldenrod1, markersize = 12)
        end
        if nrow(weak_pts) > 0
            band!(ax, weak_pts.stimulus, weak_pts.prop .- weak_pts.sem, weak_pts.prop .+ weak_pts.sem, color = (:navy, 0.3))
            scatter!(ax, weak_pts.stimulus, weak_pts.prop,
                     color = :navy, markersize = 12)
        end
    else
        band!(ax, cond_pop.stimulus, cond_pop.prop .- cond_pop.sem, cond_pop.prop .+ cond_pop.sem, color = (:navy, 0.3))
        scatter!(ax, cond_pop.stimulus, cond_pop.prop,
                 color = :navy, markersize = 12)
    end

    xlims!(ax, 1, 60)
    ylims!(ax, 0, 1)

    # Legend placed on Panel B only
    if label == "B"
        lines!(ax, [NaN], [NaN], color = (:gray, 0.4), linewidth = 1,      label = "Individual runs")
        scatter!(ax, [NaN], [NaN], color = :darkgoldenrod1, markersize = 8,  label = "Strong stimulus")
        scatter!(ax, [NaN], [NaN], color = :navy, markersize = 8, label = "Weak stimulus")
        axislegend(ax, position = :rt, labelsize = 20, framevisible = false)
    end
end

for row in 1:3
    rowsize!(fig2.layout, row, Relative(1 / 3))
end

# ---------------------------------------------------------------------------
# Print statistics
# ---------------------------------------------------------------------------

println("\n--- Statistical tests ---")
println("Arousal control (wswww): ", ttest_arousal)
println("Sensitization control (wsssw): ", ttest_sensitization)
for cond in quadratic_conditions
    qm = quadratic_models[cond]
    println("$cond: β_stim=$(round(qm.beta_stimulus, digits=3)), p=$(round(qm.p_stimulus, digits=4)), β_stim²=$(round(qm.beta_stimulus_sq, digits=3)), p=$(round(qm.p_stimulus_sq, digits=4))")
end

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

CairoMakie.activate!()
save(joinpath(FIGURES_DIR, "fig2_main_effects.png"), fig2, px_per_unit = 3)
GLMakie.activate!()
println("Saved fig2_main_effects.png")
