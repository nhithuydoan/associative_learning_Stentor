include("common.jl")

fig6   = Figure(size = (1300, 1200))
ws_fit = fitted_params["ws_ISI1_ITI45"]

# --- Panel A: Associative drive decomposition ---
Label(fig6[1, 1, TopLeft()], "A", padding = (10, 0, 10, 0), halign = :left)
ax5A = Axis(fig6[1, 1:2], xlabel = "Stimulus number", ylabel = "Proportion of responses")

s0_A, w0_A, aw_A, as_A, ac_A = ws_fit.s0, ws_fit.w0, ws_fit.αw, ws_fit.αs, ws_fit.αc

trials_A = 0:30
ws_pop_A = filter(r -> r.stimulus in trials_A,
                  filter(r -> r.condition == "ws_ISI1_ITI45", population))
response_curve  = [model_weak_response(n, s0_A, w0_A, aw_A, as_A, ac_A) for n in trials_A]
dominance_curve = [calculate_process_dominance(n, s0_A, w0_A, aw_A, as_A, ac_A) for n in trials_A]

for i in 1:(length(trials_A) - 1)
    lines!(ax5A, trials_A[i:i+1], response_curve[i:i+1],
           color = get(ColorSchemes.RdBu_11, (dominance_curve[i] + 1) / 2), linewidth = 5)
end
scatter!(ax5A, ws_pop_A.stimulus, ws_pop_A.prop,
         color = (:gray, 0.3), markersize = 20, label = "Experimental data")

key_idx = [2, argmax(response_curve), 8]
scatter!(ax5A, trials_A[key_idx], response_curve[key_idx],
         color = [get(ColorSchemes.RdBu_11, (dominance_curve[k] + 1) / 2) for k in key_idx],
         strokecolor = :black, strokewidth = 3, markersize = 20)
axislegend(ax5A, position = :rt, labelsize = 20, framevisible = false)
ylims!(ax5A, 0, 0.6)

Label(fig6[1, 3, Top()], "Process", fontsize = 16, padding = (0, 0, 20, 20))
Colorbar(fig6[1, 3], limits = (-1, 1), colormap = Reverse(:RdBu_11),
    ticks = ([-0.5, 0.5], ["Habituation", "Associative coupling"]),
    width = 25, ticklabelsize = 14)

# --- Panel B: Simulation ---
Label(fig6[2, 1, TopLeft()], "B", padding = (10, 0, 10, 0), halign = :left)
ax5B = Axis(fig6[2, 1], xlabel = "Stimulus number", ylabel = "Proportion of responses",
            title = "Simulation")
sim_trials = 0:60

lines!(ax5B, sim_trials, model_weak_response.(sim_trials, 0.8, 0.3, 0.5, 0.1, 0.4),
       label = "Weak-strong (αs < αw)", linewidth = 5, color = "#D4A5A5")
lines!(ax5B, sim_trials, model_weak_response.(sim_trials, 0.3, 0.3, 0.5, 0.5, 0.2),
       label = "Weak-weak (αs = αw)", linewidth = 5, color = :brown4)
lines!(ax5B, sim_trials, model_weak_response.(sim_trials, 0.0, 0.3, 0.5, 0.0, 0.0),
       label = "CS alone (s0=0, αc=0)", linewidth = 5, color = :black)
axislegend(ax5B, position = :rt, labelsize = 20, framevisible = false)
ylims!(ax5B, 0, 1)

# --- Panel C: Effect of αc and αw ---
Label(fig6[2, 2, TopLeft()], "C", padding = (10, 0, 10, 0), halign = :left)
ax5C = Axis(fig6[2, 2], xlabel = "Stimulus number", ylabel = "Proportion of responses",
            title = "Effect of αc and αw")

ac_sweep = [0.0, 0.2, 0.4, 0.6]
aw_sweep = [0.5, 0.8]
ac_palette = cgrad(:viridis, length(ac_sweep), categorical = true)

for (i, ac) in enumerate(ac_sweep), (j, aw) in enumerate(aw_sweep)
    lines!(ax5C, sim_trials,
           model_weak_response.(sim_trials, 0.8, 0.3, aw, 0.05, ac),
           linewidth = 3, color = ac_palette[i], linestyle = LINE_STYLES[j])
end
for (j, aw) in enumerate(aw_sweep)
    lines!(ax5C, [NaN], [NaN], color = :black, linestyle = LINE_STYLES[j],
           linewidth = 3, label = "αw = $aw")
end

axislegend(ax5C, position = :rt, labelsize = 20, framevisible = false)
ylims!(ax5C, 0, 1)

Label(fig6[2, 3, Top()], "αc", fontsize = 16, padding = (0, 0, 5, 0))
Colorbar(fig6[2, 3], limits = (minimum(ac_sweep), maximum(ac_sweep)),
    colormap = :viridis, width = 25, ticklabelsize = 14)

# --- Panel D: Model fits ---
Label(fig6[3, 1, TopLeft()], "D", padding = (10, 0, 10, 0), halign = :left)
ax5D = Axis(fig6[3, 1], xlabel = "Stimulus number", ylabel = "Response", title = "Model fit")
for (cond, color, label) in fit_conditions
    cond_pop = filter(r -> r.condition == cond, population)
    fp = fitted_params[cond]
    scatter!(ax5D, cond_pop.stimulus, cond_pop.prop,
             color = (color, 0.5), markersize = 12, label = "$label data")
    lines!(ax5D, 0:60,
           model_weak_response.(0:60, fp.s0, fp.w0, fp.αw, fp.αs, fp.αc),
           color = color, linewidth = 5, label = "Fit")
end
axislegend(ax5D, position = :rt, labelsize = 20, framevisible = false)

# --- Panel E: Fitted parameters ---
Label(fig6[3, 2, TopLeft()], "E", padding = (10, 0, 10, 0), halign = :left)
ax5E = Axis(fig6[3, 2], xlabel = "Parameter", ylabel = "Value",
            title = "Fitted parameters", titlesize = 22)
param_labels  = ["s0", "w0", "αw", "αs", "αc"]
param_names_e = Dict("ws_ISI1_ITI45" => "Weak-strong",
                     "hab_ws_ISI1_ITI59" => "Hab_ws")

for (cond, color, _) in fit_conditions
    fp   = fitted_params[cond]
    vals = [fp.s0, fp.w0, fp.αw, fp.αs, fp.αc]
    scatter!(ax5E, 1:5, vals, color = color, markersize = 18,
             strokecolor = :black, strokewidth = 1.5, label = param_names_e[cond])
end

ax5E.xticks = (1:5, param_labels)
ylims!(ax5E, 0, 1.2)
axislegend(ax5E, position = :rt, framevisible = false)

CairoMakie.activate!()
save(joinpath(FIGURES_DIR, "fig6_model.png"), fig6, px_per_unit = 3)
GLMakie.activate!()
println("Saved fig6_model.png")
