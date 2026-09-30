include("common.jl")

# ---------------------------------------------------------------------------
# Bayesian quadratic regression for all ISI/ITI conditions
# ---------------------------------------------------------------------------

isi_iti_conditions = ["ws_ISI1_ITI45",   "ww_ISI1_ITI45",   "hab_ws_ISI1_ITI59",
                      "ws_ISI10_ITI45",  "ws_ISI20_ITI45",  "ws_ISI1_ITI59",
                      "ws_ISI10_ITI59",  "ws_ISI10_ITI590"]

bayes_results = Dict(cond => sample_quad(cond, 10, population)
                     for cond in isi_iti_conditions)

# ---------------------------------------------------------------------------
# Figure 4: Temporal parameters modulate associative learning strength
# ---------------------------------------------------------------------------

ws_conditions = filter(c -> startswith(c, "ws"), unique(data.condition))
sort!(ws_conditions,
      by = x -> (parse(Int, split(x, "ITI")[2]),
                 parse(Int, replace(split(x, "_")[2], "ISI" => ""))))

fig4 = Figure(size = (1200, 1100))

for (i, condition) in enumerate(ws_conditions)
    row = (i - 1) ÷ 3 + 1
    col = (i - 1) % 3 + 1

    cond_pop  = filter(r -> r.condition == condition, population)
    cond_runs = filter(r -> r.condition == condition, individual_runs)
    transform!(cond_pop,
        [:prop, :num_cells] =>
        ByRow((p, n) -> 1.96 * sqrt(p * (1 - p) / n)) => :sem)
    sort!(cond_pop, :stimulus)

    m = match(r"(\w+)_ISI(\d+)_ITI(\d+)", condition)
    _, isi, iti = m.captures
    n_cells = cond_pop.num_cells[1]

    Label(fig4[row, col, TopLeft()], ["A","B","C","D","E","F"][i],
          padding = (0, 20, 20, 0), halign = :right)

    ax = Axis(fig4[row, col],
        xlabel = row == 2 ? "Stimulus number" : "",
        ylabel = "Proportion of responses",
        title  = "ISI = $(isi)s, ITI = $(iti)s\n(n=$n_cells)")

    for run_id in unique(cond_runs.run)
        run_data = sort(filter(r -> r.run == run_id, cond_runs), :stimulus)
        nrow(run_data) > 1 &&
            lines!(ax, run_data.stimulus, run_data.prop,
                   color = (:gray, 0.4), linewidth = 1)
    end

    band!(ax, cond_pop.stimulus,
          cond_pop.prop .- cond_pop.sem,
          cond_pop.prop .+ cond_pop.sem,
          color = (:navy, 0.3))
    scatter!(ax, cond_pop.stimulus, cond_pop.prop,
             color = :navy, markersize = 8)
    xlims!(ax, 1, 60)
    ylims!(ax, 0, 1)

    if i == 1
        lines!(ax, [NaN], [NaN], color = (:gray, 0.4), linewidth = 1,
               label = "Individual runs")
        scatter!(ax, [NaN], [NaN], color = :navy, markersize = 8,
                 label = "Response to CS")
        axislegend(ax, position = :rt, labelsize = 12, framevisible = false)
    end
end

# Panel G: Posterior distribution of peak location
Label(fig4[3, 1, TopLeft()], "G", padding = (0, 20, 20, 0), halign = :right)

ax4G = Axis(fig4[3, 1:3],
    xlabel = "Peak location (normalized stimulus)",
    ylabel = "Density",
    title  = "Posterior distribution of peak location")

posterior_colors = [:brown4, :navy, :green, :goldenrod, :purple, :gray, :brown, :black]

for (i, condition) in enumerate(isi_iti_conditions)
    res   = bayes_results[condition]
    peaks = -res[:b1] ./ (2 .* res[:b2])
    valid = peaks[(peaks .> -1) .& (peaks .< 3)]
    m = match(r"(\w+)_ISI(\d+)_ITI(\d+)", condition)
    cond_name, isi, iti = m.captures
    density!(ax4G, valid,
        color = (posterior_colors[i], 0.2),
        strokecolor = posterior_colors[i],
        strokewidth = 2,
        label = "$cond_name, ISI=$(isi), ITI=$(iti)")
end

vlines!(ax4G, [1.0], color = :black, linestyle = :dash, linewidth = 2,
        label = "Last stimulus")
xlims!(ax4G, -0.5, 2.5)
axislegend(ax4G, position = :rt, labelsize = 10, framevisible = false)

CairoMakie.activate!()
save(joinpath(FIGURES_DIR, "fig4_temporal.png"), fig4, px_per_unit = 3)
GLMakie.activate!()
println("Saved fig4_temporal.png")
