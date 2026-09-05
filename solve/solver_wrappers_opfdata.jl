module SolverWrappers

using PowerModels
using InfrastructureModels
using Mosek
using MosekTools
using DataFrames
using CSV
using JuMP
using SparseArrays

include(joinpath(@__DIR__, "..", "..", "src_jl", "ChordalStatsLite.jl"))
using .ChordalStatsLite: compute_stats_from_vars
export solve
println("Julia threads = ", Threads.nthreads())

const OPFDATA_OUTPUT_ROOT = normpath(joinpath(@__DIR__, "..", "outputs"))



# ============================================================
# Mosek log capture
# ============================================================
function _run_with_capture(f::Function)
    outpath = tempname()
    errpath = tempname()

    ret = nothing
    caught_err = nothing

    try
        ret = open(outpath, "w") do outio
            open(errpath, "w") do errio
                redirect_stdout(outio) do
                    redirect_stderr(errio) do
                        f()
                    end
                end
            end
        end
    catch err
        caught_err = (err, catch_backtrace())
    end

    out_s = isfile(outpath) ? read(outpath, String) : ""
    err_s = isfile(errpath) ? read(errpath, String) : ""

    # Even if Mosek crashes, print everything captured before the crash
    println(out_s)
    println(err_s)
    flush(stdout)

    try
        rm(outpath; force=true)
        rm(errpath; force=true)
    catch
    end

    # Re-throw the original error after printing the solver log
    if caught_err !== nothing
        err, bt = caught_err
        Base.throw(err, bt)
    end

    return ret, string(out_s, "\n", err_s)
end

function parse_mosek_log_all(logtxt::AbstractString)
    iters = try
        m = match(r"Iterations:\s+(\d+)", logtxt)
        m === nothing ? missing : parse(Int, m.captures[1])
    catch
        missing
    end

    pfeas = try
        m = match(
            r"Primal\s+(?:in)?feasibility\s*[:=]\s*([0-9.eE+\-]+)",
            logtxt,
        )
        m === nothing ? missing : parse(Float64, m.captures[1])
    catch
        missing
    end

    dfeas = try
        m = match(
            r"Dual\s+(?:in)?feasibility\s*[:=]\s*([0-9.eE+\-]+)",
            logtxt,
        )
        m === nothing ? missing : parse(Float64, m.captures[1])
    catch
        missing
    end

    relgap = try
        m = match(
            r"Relative\s+gap\s*[:=]\s*([0-9.eE+\-]+)",
            logtxt,
        )
        m === nothing ? missing : parse(Float64, m.captures[1])
    catch
        missing
    end

    time_sec = try
        m = match(
            r"Optimizer terminated\. Time:\s*([0-9.]+)",
            logtxt,
        )
        m === nothing ? missing : parse(Float64, m.captures[1])
    catch
        missing
    end

    # Fallback: read the last interior-point iteration row.
    if any(ismissing, (iters, pfeas, dfeas, relgap))
        last = nothing

        for ln in eachline(IOBuffer(logtxt))
            m = match(
                r"^\s*(\d+)\s+([0-9.eE+\-]+)\s+([0-9.eE+\-]+)\s+" *
                r"[0-9.eE+\-]+\s+\S+\s+([0-9.eE+\-]+)\s+" *
                r"([0-9.eE+\-]+)\s+[0-9.eE+\-]+\s+([0-9.]+)\s*$",
                ln,
            )
            if m !== nothing
                last = m
            end
        end

        if last !== nothing
            iters = ismissing(iters) ? parse(Int, last.captures[1]) : iters
            pfeas = ismissing(pfeas) ? parse(Float64, last.captures[2]) : pfeas
            dfeas = ismissing(dfeas) ? parse(Float64, last.captures[3]) : dfeas

            if ismissing(relgap)
                pobj = parse(Float64, last.captures[4])
                dobj = parse(Float64, last.captures[5])
                relgap = abs(pobj - dobj) / max(1.0, abs(pobj))
            end

            time_sec = ismissing(time_sec) ?
                       parse(Float64, last.captures[6]) :
                       time_sec
        end
    end

    return iters, pfeas, dfeas, relgap, time_sec
end


_get_any(dict, k, default=missing) =
    haskey(dict, k) ? dict[k] : default

_get_any(dict, ks::Vector, default=missing) = begin
    for k in ks
        if haskey(dict, k)
            return dict[k]
        end
    end
    return default
end


function _getnum(d, keysets...; default=missing)
    for ks in keysets
        v = _get_any(d, ks, nothing)
        if v isa Number
            return Float64(v)
        end
    end
    return default
end


function _coeff_ratio_from_data(data; tol=1e-16)
    coeffs = Float64[]

    for (_k, b) in get(data, "bus", Dict())
        for key in ("pd", "qd", "gs", "bs", "gsh", "bsh")
            v = get(b, key, nothing)
            v isa Number && push!(coeffs, abs(float(v)))
        end
    end

    for (_k, br) in get(data, "branch", Dict())
        for key in ("r", "x", "b", "g", "br_b", "br_r", "br_x")
            v = get(br, key, nothing)
            v isa Number && push!(coeffs, abs(float(v)))
        end
        for key in ("rate_a", "rate_b", "rate_c")
            v = get(br, key, nothing)
            v isa Number && push!(coeffs, abs(float(v)))
        end
    end

    for (_k, g) in get(data, "gen", Dict())
        for key in ("pmax", "pmin", "qmax", "qmin")
            v = get(g, key, nothing)
            v isa Number && push!(coeffs, abs(float(v)))
        end
    end

    coeffs = filter(!iszero, coeffs)

    return isempty(coeffs) ?
           missing :
           maximum(coeffs) / max(minimum(coeffs), tol)
end


function _count_active_limits(result, data; tol=1e-4)
    sol = _get_any(result, ["solution", :solution], nothing)
    sol isa AbstractDict || return missing

    cnt = 0

    sbus = _get_any(sol, ["bus", :bus], Dict())
    dbus = get(data, "bus", Dict())

    for (id, bv) in sbus
        vm = _get_any(bv, ["vm", :vm], nothing)
        vm isa Number || continue

        bd = get(dbus, string(id), get(dbus, id, nothing))
        bd isa AbstractDict || continue

        vmin = _get_any(bd, ["vmin", :vmin], nothing)
        vmax = _get_any(bd, ["vmax", :vmax], nothing)

        (vmin isa Number && abs(vm - vmin) <= tol) && (cnt += 1)
        (vmax isa Number && abs(vm - vmax) <= tol) && (cnt += 1)
    end

    sgen = _get_any(sol, ["gen", :gen], Dict())
    dgen = get(data, "gen", Dict())

    for (id, gv) in sgen
        gd = get(dgen, string(id), get(dgen, id, nothing))
        gd isa AbstractDict || continue

        pg = _get_any(gv, ["pg", :pg], nothing)
        qg = _get_any(gv, ["qg", :qg], nothing)

        if pg isa Number
            pmin = _get_any(gd, ["pmin", :pmin], nothing)
            pmax = _get_any(gd, ["pmax", :pmax], nothing)

            (pmin isa Number && abs(pg - pmin) <= tol) && (cnt += 1)
            (pmax isa Number && abs(pg - pmax) <= tol) && (cnt += 1)
        end

        if qg isa Number
            qmin = _get_any(gd, ["qmin", :qmin], nothing)
            qmax = _get_any(gd, ["qmax", :qmax], nothing)

            (qmin isa Number && abs(qg - qmin) <= tol) && (cnt += 1)
            (qmax isa Number && abs(qg - qmax) <= tol) && (cnt += 1)
        end
    end

    return cnt
end


# ============================================================
# Save chordal graph
# ============================================================
function _save_chordal_graph(
    cadj,
    sigma,
    q,
    lookup_index,
    case_name::AbstractString,
    bus_adjacency_id::AbstractString,
    scene_id::AbstractString,
)
    # Graph depend on the bus adjacency and decomposition strategy,
    base_dir = joinpath(
        OPFDATA_OUTPUT_ROOT,
        "chordal_graph_matrix",
        case_name,
        bus_adjacency_id,
    )

    mkpath(base_dir)

    edge_path = joinpath(base_dir, "$(scene_id)_chordal_edges.csv")
    sigma_path = joinpath(base_dir, "$(scene_id)_sigma.csv")
    q_path = joinpath(base_dir, "$(scene_id)_q.csv")
    lookup_path = joinpath(base_dir, "$(scene_id)_lookup_index.csv")
    if all(isfile, (edge_path, sigma_path, q_path, lookup_path))
        println(
            "⏭️  chordal graph already exists; reuse: ",
            case_name,
            " / ",
            bus_adjacency_id,
            " / ",
            scene_id,
        )
        return base_dir
    end

    I, J, V = findnz(cadj)

    df_edges = DataFrame(
        i = I .- 1,
        j = J .- 1,
        w = V,
    )

    CSV.write(
        edge_path,
        df_edges;
        header=false,
    )

    CSV.write(
        sigma_path,
        DataFrame(sigma=sigma);
        header=false,
    )

    CSV.write(
        q_path,
        DataFrame(q=q);
        header=false,
    )

    # lookup_index: PowerModels bus id -> matrix index (1-based).
    bus_ids = collect(keys(lookup_index))
    sort!(bus_ids)

    mat_ids = [lookup_index[bus] for bus in bus_ids]

    CSV.write(
        lookup_path,
        DataFrame(
            bus=bus_ids,
            matrix_idx=mat_ids,
        );
        header=true,
    )

    return base_dir
end


# ============================================================
# Main solve
# ============================================================

function solve(
    data,
    model,
    clique_merging,
    case_name;
    alpha=3.0,
    id_name=nothing,
    csv_name=nothing,
    file_name="default",
    dataset_type,
    sample_id,
    group_id,
    structure_id,
    bus_adjacency_id,
    outage_type,
    outage_component,
    replicate_id,
)
    # --------------------------------------------------------
    # 1. Initialize PowerModels
    # --------------------------------------------------------
    model_type = getfield(PowerModels, Symbol(model))
    pm = InfrastructureModels.InitializeInfrastructureModel(
        model_type,
        data,
        PowerModels._pm_global_keys,
        PowerModels.pm_it_sym,
    )

    PowerModels.ref_add_core!(pm.ref)
    nw = collect(
        InfrastructureModels.nw_ids(pm, PowerModels.pm_it_sym)
    )[1]
    adj, cadj, lookup_index, sigma, q =
        PowerModels._chordal_extension_original_alpha(
            pm,
            nw,
            clique_merging,
            alpha,
        )

    @assert q == invperm(sigma) "Permutation mismatch: q must equal invperm(sigma)."

    cliques = PowerModels._maximal_cliques(cadj)

    lookup_bus_index = Dict(
        reverse(p) for p in pairs(lookup_index)
    )

    groups = [
        [lookup_bus_index[gi] for gi in g]
        for g in cliques
    ]

    pm.ext[:SDconstraintDecomposition] =
        PowerModels._SDconstraintDecomposition(
            groups,
            lookup_index,
            sigma,
        )

    network_type = "original_0"
    scene_id = "$(network_type)_$(alpha)_$(string(model))"

    # --------------------------------------------------------
    # 2. Save graph artifacts immediately
    # --------------------------------------------------------

    if file_name != "default"
        graph_dir = _save_chordal_graph(
            cadj,
            sigma,
            q,
            lookup_index,
            case_name,
            bus_adjacency_id,
            scene_id,
        )

        println("✅ chordal graph directory: ", graph_dir)
    end

    # --------------------------------------------------------
    # 3. Structural statistics
    # --------------------------------------------------------

    stats = compute_stats_from_vars(
        ;
        cadj=cadj,
        sigma=sigma,
        cliques=cliques,
        cadj0=adj,
    )

    # --------------------------------------------------------
    # 4. Build and solve SDP
    # --------------------------------------------------------

    PowerModels.build_opf(pm)

    opt = optimizer_with_attributes(
        Mosek.Optimizer,
        "MSK_IPAR_NUM_THREADS" => 2,
        "MSK_IPAR_LOG" => 1,
        "MSK_IPAR_LOG_INTPNT" => 1,
        "QUIET" => 0,
    )

    println(
        "🔷 solving OPFData: ",
        case_name,
        " / ",
        file_name,
        " / ",
        model,
        " / merge=",
        clique_merging,
        " / alpha=",
        alpha,
    )

    result, mosek_log = _run_with_capture() do
        optimize_model!(
            pm,
            optimizer=opt,
        )
    end
    println(mosek_log)
    flush(stdout)
    # --------------------------------------------------------
    # 5. Solver diagnostics
    # --------------------------------------------------------

    log_iters, log_pr, log_dr, log_rg, log_time =
        parse_mosek_log_all(mosek_log)

    iterations = log_iters
    primal_res = log_pr
    dual_res = log_dr
    rel_gap = log_rg
    mosektime = log_time

    _get(r, ks, default=missing) = begin
        for k in ks
            if haskey(r, k)
                return r[k]
            end
        end
        return default
    end

    solve_time = _get(
        result,
        ["solve_time", :solve_time],
        NaN,
    )

    term_status = string(
        _get(
            result,
            [
                "termination_status",
                :termination_status,
                "status",
                :status,
            ],
            "",
        ),
    )

    obj_val = _get(
        result,
        [
            "objective",
            :objective,
            "obj_val",
            :obj_val,
        ],
        NaN,
    )

    sol_status = string(
        _get(
            result,
            [
                "solution_status",
                :solution_status,
                "primal_status",
                :primal_status,
            ],
            "",
        ),
    )

    if rel_gap === missing
        obj_lb = _getnum(
            result,
            [
                "objective_lb",
                :objective_lb,
                "best_bound",
                :best_bound,
                "dual_objective",
                :dual_objective,
            ],
        )

        if !(obj_val === missing || obj_lb === missing)
            denom = max(1.0, abs(obj_val))
            rel_gap = abs(obj_val - obj_lb) / denom
        end
    end

    kkt_cond_proxy =
        _coeff_ratio_from_data(data)

    active_limits =
        _count_active_limits(result, data; tol=1e-4)

    # --------------------------------------------------------
    # 6. Keep the same core result columns as the legacy CSV
    # --------------------------------------------------------
    df_core = DataFrame(
        network_type      = [network_type],

        # OPFData experiment metadata.
        dataset_type      = [dataset_type],
        group_id          = [group_id],
        structure_id      = [structure_id],
        bus_adjacency_id  = [bus_adjacency_id],
        outage_type       = [outage_type],
        outage_component  = [outage_component],

        Formulation       = [string(model)],
        Perturbation      = ["OPFData"],
        Case              = [case_name],
        Merge             = [clique_merging],
        A_parameter       = [alpha],
        SolveTime         = [solve_time],
        mosektime         = [mosektime],
        Status            = [term_status],
        objective         = [obj_val],
        SolutionStatus    = [sol_status],

        # ID now comes directly from experiment_metadata["sample_id"].
        ID                = [Int(sample_id)],
        replicate_id      = [Int(replicate_id)],
        load_id           = [id_name],

        Iterations        = [iterations],
        PrimalRes         = [primal_res],
        DualRes           = [dual_res],
        RelGap            = [rel_gap],
        KKTCondProxy      = [kkt_cond_proxy],
        ActiveLimits      = [active_limits],
    )

    df_stats = DataFrame(
        r_max        = [stats.r_max],
        t            = [stats.t],
        r_var        = [stats.r_var],
        sum_r_sq     = [stats.sum_r_sq],
        sum_r_cu     = [stats.sum_r_cu],
        sep_max      = [stats.sep_max],
        sep_mean     = [stats.sep_mean],
        sum_sep_sq   = [stats.sum_sep_sq],
        tree_max_deg = [stats.tree_max_degree],
        tree_h       = [stats.tree_height],
        fillin       = [stats.fillin_ratio],
        coupling     = [stats.coupling_proxy],
    )

    df = hcat(df_core, df_stats)

    # --------------------------------------------------------
    # 7. Save CSV in the new OPFData tree
    # --------------------------------------------------------

    csv_name === nothing &&
        (csv_name = "$(file_name)_results.csv")

    # Solver results are separated by case, dataset type, and OPFData group.
    # The caller supplies a replicate-specific CSV name, e.g.
    # example_0_rep_03_results.csv.
    # Example:
    #   opfdata_pipeline/outputs/case14/fulltop/group_0/
    #   opfdata_pipeline/outputs/case14/nminusone/group_0/
    stats_csv_path = joinpath(
        OPFDATA_OUTPUT_ROOT,
        "clique_stats",
        case_name,
        dataset_type,
        group_id,
        csv_name,
    )

    mkpath(dirname(stats_csv_path))

    if isfile(stats_csv_path)
        CSV.write(
            stats_csv_path,
            df;
            append=true,
            writeheader=false,
        )
    else
        CSV.write(
            stats_csv_path,
            df;
            writeheader=true,
        )
    end

    println("✅ result CSV saved to: ", stats_csv_path)
    return df, solve_time, stats_csv_path
end


end # module SolverWrappers
