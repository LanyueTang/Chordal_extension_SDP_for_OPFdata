#!/usr/bin/env julia

include(joinpath(@__DIR__, "opf_parser.jl"))
include(joinpath(@__DIR__, "solver_wrappers_opfdata.jl"))

using .OPFParser
using .SolverWrappers
using DataFrames
using CSV
using JSON

function run_one_case(
    case_name::String,
    json_path::String,
    fm::String,
    merging::Bool,
    alpha::Float64,
    replicate_id::Int,
)
    println("==============================================")
    println(" OPFData single-case solve")
    println("==============================================")
    println("case      = ", case_name)
    println("json      = ", json_path)
    println("strategy  = ", fm)
    println("merging   = ", merging)
    println("alpha        = ", alpha)
    println("replicate_id = ", replicate_id)
    println("==============================================")

    # --------------------------------------------------------
    # 1. Read experiment metadata written by preprocessing
    # --------------------------------------------------------
    raw_obj = JSON.parsefile(json_path)

    haskey(raw_obj, "experiment_metadata") ||
        error(
            "Missing experiment_metadata in OPFData JSON: $json_path\n" *
            "Run annotate_opfdata_metadata.py before solving."
        )

    meta = raw_obj["experiment_metadata"]

    required_meta = (
        "dataset_type",
        "sample_id",
        "group_id",
        "structure_id",
        "bus_adjacency_id",
        "outage_type",
        "outage_component",
    )

    for key in required_meta
        haskey(meta, key) ||
            error("experiment_metadata is missing required field '$key': $json_path")
    end

    dataset_type = String(meta["dataset_type"])
    sample_id = Int(meta["sample_id"])
    group_id = String(meta["group_id"])
    structure_id = String(meta["structure_id"])
    bus_adjacency_id = String(meta["bus_adjacency_id"])
    outage_type = String(meta["outage_type"])
    outage_component = String(meta["outage_component"])

    println("dataset_type      = ", dataset_type)
    println("sample_id         = ", sample_id)
    println("group_id          = ", group_id)
    println("structure_id      = ", structure_id)
    println("bus_adjacency_id  = ", bus_adjacency_id)
    println("outage_type       = ", outage_type)
    println("outage_component  = ", outage_component)
    println("==============================================")

    # --------------------------------------------------------
    # 2. Parse OPFData grid into PowerModels-compatible data
    # --------------------------------------------------------
    data = OPFParser.parse_opfdata(json_path)
    OPFParser.print_network_summary(data)

    # Keep the same objective scaling convention as the legacy pipeline.
    for (_id, gen) in data["gen"]
        gen["cost"] .= gen["cost"] ./ 1e3
    end

    fname = basename(json_path)
    file_name = splitext(fname)[1]

    # One CSV = one sample × one replicate × 15 strategies.
    rep_token = lpad(string(replicate_id), 2, '0')
    csv_name = "$(file_name)_rep_$(rep_token)_results.csv"

    # --------------------------------------------------------
    # 3. CSV recovery / strategy-level resume
    # --------------------------------------------------------
    # The CSV is the fine-grained recovery record. If this exact strategy
    # already exists for this replicate, do not solve it again.
    output_root = normpath(joinpath(@__DIR__, "..", "outputs"))
    csv_path = joinpath(
        output_root,
        "clique_stats",
        case_name,
        dataset_type,
        group_id,
        csv_name,
    )

    if isfile(csv_path)
        existing_df = CSV.read(csv_path, DataFrame)

        matching_idx = Int[]
        for (i, row) in enumerate(eachrow(existing_df))
            same_fm = string(row.Formulation) == fm
            same_merge = Bool(row.Merge) == merging
            same_alpha = isapprox(
                Float64(row.A_parameter),
                alpha;
                atol=1e-12,
                rtol=0.0,
            )

            same_rep = if :replicate_id in propertynames(existing_df)
                Int(row.replicate_id) == replicate_id
            else
                false
            end

            if same_fm && same_merge && same_alpha && same_rep
                push!(matching_idx, i)
            end
        end

        if length(matching_idx) > 1
            error(
                "Duplicate strategy rows found in $csv_path for " *
                "$fm | Merge=$merging | alpha=$alpha | replicate_id=$replicate_id"
            )
        elseif length(matching_idx) == 1
            row_idx = only(matching_idx)
            recovered = existing_df[row_idx:row_idx, :]
            solve_time = Float64(recovered[1, :SolveTime])

            println()
            println(
                "⏭️  Strategy already exists in CSV; skip solve: ",
                fm,
                " | Merge=",
                merging,
                " | alpha=",
                alpha,
                " | replicate_id=",
                replicate_id,
            )
            println("CSV = ", csv_path)

            return recovered, solve_time, csv_path
        end
    end

    model = fm
    println()
    println("Starting SolverWrappers.solve ...")
    println()

    df, solve_time, csv_path = SolverWrappers.solve(
        data,
        model,
        merging,
        case_name;
        alpha=alpha,
        id_name=fname,
        csv_name=csv_name,
        file_name=file_name,
        dataset_type=dataset_type,
        sample_id=sample_id,
        group_id=group_id,
        structure_id=structure_id,
        bus_adjacency_id=bus_adjacency_id,
        outage_type=outage_type,
        outage_component=outage_component,
        replicate_id=replicate_id,
    )

    println()
    println("==============================================")
    println(" Solve finished")
    println("==============================================")
    println("solve_time = ", solve_time)
    println("csv_path   = ", csv_path)
    println()
    show(df; allcols=true, allrows=true)
    println()
    println("==============================================")

    return df, solve_time, csv_path
end

function __main__(args)
    if length(args) != 6
        error(
            "Usage: julia --project=. run_one_case_opfdata.jl " *
            "<case_name> <json_path> <fm> <merging> <alpha> <replicate_id>"
        )
    end

    case_name = args[1]
    json_path = args[2]
    fm = args[3]
    merging = lowercase(args[4]) == "true"
    alpha = parse(Float64, args[5])
    replicate_id = parse(Int, args[6])

    run_one_case(
        case_name,
        json_path,
        fm,
        merging,
        alpha,
        replicate_id,
    )
end

if !isempty(ARGS)
    __main__(ARGS)
end
