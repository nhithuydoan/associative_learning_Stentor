include(joinpath(@__DIR__, "..", "common.jl"))
using AnovaGLM

const FOLLOWUP_FIGURES_DIR = joinpath(@__DIR__, "..", "..", "figures", "followup")
mkpath(FOLLOWUP_FIGURES_DIR)

const DATASETS = [
    (file = "dataset_followup_1.csv", label = "Set 1",
     desc = "ISI=1, ITI=59, break₁₂=5400s, break₂₃=300s, 60-stim hab"),
    (file = "dataset_followup_2.csv", label = "Set 2",
     desc = "ISI=1, ITI=59, break₁₂=2400s, break₂₃=600s, 20-stim hab"),
]

function load_followup(filename)
    df = CSV.read(joinpath(DATA_DIR, filename), DataFrame)
    df.type = ifelse.(
        (df.condition .== "control") .&
        (df.phase .== "train") .&
        (df.stimulus .>= 2) .& (df.stimulus .<= 5), "s", "w")
    df.cell_id = string.(df.condition, "_", df.run, "_", df.cell_number)

    run_props = combine(groupby(df, [:condition, :folder, :phase, :stimulus, :type]),
        :contract => nanmean => :prop)
    summary_df = combine(groupby(run_props, [:condition, :phase, :stimulus, :type]),
        :prop => nanmean => :mean_prop,
        :prop => (x -> nanstd(x) / sqrt(length(x))) => :sem)

    return df, run_props, summary_df
end

# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

function run_stats(df, run_props)
    first_at(phase) = filter(r -> r.phase == phase && r.stimulus == 1 &&
                                   !isnan(r.contract), df)

    pvalues = Dict(phase => coeftable(
        fit(MixedModel, @formula(contract ~ condition + (1|folder)),
            first_at(phase), Bernoulli(), LogitLink())).cols[4][2]
        for phase in ["hab", "train", "test"])

    group_effect = filter(r -> r.phase in ["train", "test"] &&
                               r.stimulus == 1 && !isnan(r.contract), df)
    group_effect.cell = string.(group_effect.run, "_", group_effect.cell_id)

    gm = fit(MixedModel, @formula(
        contract ~ condition * phase + (1 | folder) + (1 | cell)),
        group_effect, Bernoulli(), LogitLink())
    p_interaction = let ct = coeftable(gm)
        ct.cols[4][findfirst(n -> occursin("&", n), ct.rownms)]
    end

    return pvalues, p_interaction, gm
end

# ---------------------------------------------------------------------------
# Main figure (panels B–E)
# ---------------------------------------------------------------------------

function plot_followup(df, run_props, summary_df, pvalues, p_interaction)
    paper_theme = Theme(
        Axis = (titlegap = 30, titlesize = 22,
                xgridvisible = false, ygridvisible = false,
                topspinevisible = false, rightspinevisible = false,
                xlabelsize = 20, ylabelsize = 20,
                xticklabelsize = 16, yticklabelsize = 16,
                xlabelpadding = 4, ylabelpadding = 4),
        Legend = (framevisible = false, backgroundcolor = :transparent,
                  padding = (0, 0, 0, 0), labelsize = 16))

    fig = with_theme(paper_theme) do
        fig = Figure(size = (1200, 900))

        navy  = RGBf(0.0, 0.0, 0.502)
        gold  = RGBf(0.855, 0.647, 0.125)
        space = 0.12
        markerof(c) = c == "control" ? :rect : :circle
        ymax  = 0.8
        fmt_p(p) = p < 0.001 ? "p < 0.001" : "p = $(round(p, digits = 3))"

        pos_summary = fig[1, 1:3]
        pos_slope   = fig[2, 1]
        pos_hab     = fig[2, 2]
        pos_train   = fig[2, 3]

        rowsize!(fig.layout, 1, Relative(0.45))
        rowgap!(fig.layout, 40)

        phases       = ["hab", "train", "test"]
        curve_phases = ["hab", "train"]
        panel_pos    = Dict("hab" => pos_hab, "train" => pos_train)
        titles       = Dict("hab" => "Habituation", "train" => "Training")

        ncells(c) = length(unique(zip(df.folder[df.condition .== c],
                                      df.cell_number[df.condition .== c])))
        cond_legend(ax) = axislegend(ax,
            [MarkerElement(marker = :circle, color = navy, markersize = 12),
             MarkerElement(marker = :rect, color = :transparent,
                           strokecolor = navy, strokewidth = 1.2, markersize = 12)],
            ["Experimental (n=$(ncells("experimental")))",
             "Control (n=$(ncells("control")))"];
            position = :rt, framevisible = false, rowgap = 2,
            patchsize = (14, 14), labelsize = 15)

        # response curves per phase
        for ph in curve_phases
            ax = Axis(panel_pos[ph];
                xlabel = "Stimulus",
                ylabel = ph == "hab" ? "Proportion of responses" : "",
                title  = titles[ph],
                xticks = ph == "hab" ? [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60] :
                                       collect(1:5),
                limits = (nothing, (-0.05, ymax)))
            for cond in ["experimental", "control"]
                exp = cond == "experimental"
                sub = sort(filter(r -> r.condition == cond && r.phase == ph, summary_df), :stimulus)
                isempty(sub) && continue
                x    = Float64.(sub.stimulus)
                y    = Float64.(sub.mean_prop)
                e    = 1.96 .* Float64.(sub.sem)
                cols = [t == "s" ? gold : navy for t in sub.type]
                lo, hi = y .- e, y .+ e
                for k in 1:length(x)-1
                    seg_strong = sub.type[k] == "s" && sub.type[k+1] == "s"
                    band!(ax, x[k:k+1], lo[k:k+1], hi[k:k+1];
                        color = ((seg_strong ? gold : navy), 0.12))
                end
                lines!(ax, x, y; color = navy, linestyle = exp ? :solid : :dash)
                scatter!(ax, x, y;
                    marker      = markerof(cond), markersize = 9,
                    color       = exp ? cols : :transparent,
                    strokecolor = cols,
                    strokewidth = exp ? 0 : 1.2)
            end
            if ph == "hab"
                cond_legend(ax)
            else
                axislegend(ax,
                    [MarkerElement(marker = :circle, color = navy, markersize = 12),
                     MarkerElement(marker = :circle, color = gold, markersize = 12)],
                    ["Weak tap", "Strong tap"];
                    position = :rt, framevisible = false, rowgap = 2,
                    patchsize = (14, 14), labelsize = 15)
            end
        end

        # first response at each stage
        ax_summary = Axis(pos_summary;
            ylabel = "Proportion responding",
            title  = "First responses at each stage",
            xticks = ([1, 2, 3], ["Habituation", "Training", "Testing"]),
            limits = (0.4, 3.6, -0.05, ymax))
        first_by_run = filter(r -> r.stimulus == 1, run_props)

        for cond in ["experimental", "control"]
            exp = cond == "experimental"
            for (idx, p) in enumerate(phases)
                d = filter(r -> r.condition == cond && r.phase == p, first_by_run).prop
                base = idx + (exp ? -space : space)
                m    = nanmean(d)
                ci   = 1.96 * nanstd(d) / sqrt(max(count(!isnan, d), 1))
                barplot!(ax_summary, [base], [m];
                    width       = 0.18,
                    color       = exp ? (navy, 0.18) : :transparent,
                    strokecolor = navy, strokewidth = 1.2)
                jitter = base .+ (rand(length(d)) .- 0.5) .* 0.06
                scatter!(ax_summary, jitter, d; marker = markerof(cond),
                    markersize = 9, color = exp ? (navy, 0.6) : :transparent,
                    strokecolor = navy, strokewidth = exp ? 0 : 1.2)
                errorbars!(ax_summary, [base], [m], [ci];
                    color = navy, whiskerwidth = 10, linewidth = 1.5)
            end
        end
        cond_legend(ax_summary)
        for (idx, ph) in enumerate(phases)
            ybar = nanmaximum(filter(r -> r.phase == ph, first_by_run).prop) + 0.06
            lines!(ax_summary, [idx - space, idx + space], [ybar, ybar];
                color = :black, linewidth = 1.5)
            text!(ax_summary, idx, ybar + 0.02; text = fmt_p(pvalues[ph]),
                align = (:center, :bottom), fontsize = 16)
        end

        # first response by run (slope plot)
        ax_ba = Axis(pos_slope;
            ylabel = "Proportion responding",
            title  = "First response by runs",
            xticks = ([1, 2], ["Training", "Testing"]),
            limits = (0.5, 2.5, -0.05, ymax))
        for cond in ["experimental", "control"]
            exp = cond == "experimental"
            off = exp ? -space : space
            pre  = filter(r -> r.condition == cond && r.phase == "train" &&
                               r.stimulus == 1, run_props)
            post = filter(r -> r.condition == cond && r.phase == "test"  &&
                               r.stimulus == 1, run_props)
            for f in intersect(pre.folder, post.folder)
                y1 = pre.prop[findfirst(==(f), pre.folder)]
                y2 = post.prop[findfirst(==(f), post.folder)]
                xs = [1 + off, 2 + off]
                lines!(ax_ba, xs, [y1, y2]; color = (navy, 0.35), linewidth = 1)
                scatter!(ax_ba, xs, [y1, y2];
                    marker = markerof(cond), markersize = 9,
                    color = exp ? (navy, 0.7) : :transparent,
                    strokecolor = navy, strokewidth = exp ? 0 : 1.2)
            end
        end
        cond_legend(ax_ba)
        let d = filter(r -> r.phase in ["train", "test"] && r.stimulus == 1, run_props)
            ybar = nanmaximum(d.prop) - 0.02
            lines!(ax_ba, [1.5 - space, 1.5 + space], [ybar, ybar];
                color = :black, linewidth = 1.5)
            text!(ax_ba, 1.5, ybar + 0.02; text = fmt_p(p_interaction),
                align = (:center, :bottom), fontsize = 16)
        end

        for (lbl, pos) in zip(["B", "C", "D", "E"],
                              [fig[1, 1, TopLeft()], fig[2, 1, TopLeft()],
                               fig[2, 2, TopLeft()], fig[2, 3, TopLeft()]])
            Label(pos, lbl; fontsize = 22, font = :bold,
                  padding = (0, 5, 5, 0), halign = :right)
        end
        fig
    end
    return fig
end

# ---------------------------------------------------------------------------
# Run both sets
# ---------------------------------------------------------------------------

CairoMakie.activate!()

for ds in DATASETS
    println("\n=== $(ds.label): $(ds.desc) ===")
    df, run_props, summary_df = load_followup(ds.file)

    pvalues, p_interaction, gm = run_stats(df, run_props)
    println("Phase p-values (condition effect on first stimulus):")
    for ph in ["hab", "train", "test"]
        println("  $ph: p = $(round(pvalues[ph], digits = 4))")
    end
    println("Condition × phase interaction: p = $(round(p_interaction, sigdigits = 4))")
    println(gm)

    fig = plot_followup(df, run_props, summary_df, pvalues, p_interaction)

    tag = ds.label == "Set 1" ? "set1" : "set2"
    outpath = joinpath(FOLLOWUP_FIGURES_DIR, "fig3_common_test_$(tag).png")
    save(outpath, fig, px_per_unit = 5)
    println("Saved $(basename(outpath))")
end

GLMakie.activate!()
