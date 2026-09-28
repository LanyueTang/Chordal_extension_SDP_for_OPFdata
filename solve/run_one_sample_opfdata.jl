#!/usr/bin/env julia

using CSV
using DataFrames
using JSON

const ORIGINAL_STRATEGIES = [
    ("Chordal_MD",  false, 0.0, "original", 0.0),
    ("Chordal_MD",  true,  2.0, "original", 0.0),
    ("Chordal_MD",  true,  3.0, "original", 0.0),
    ("Chordal_MD",  true,  4.0, "original", 0.0),
    ("Chordal_MD",  true,  5.0, "original", 0.0),
    ("Chordal_AMD", false, 0.0, "original", 0.0),
    ("Chordal_AMD", true,  2.0, "original", 0.0),
    ("Chordal_AMD", true,  3.0, "original", 0.0),
    ("Chordal_AMD", true,  4.0, "original", 0.0),
    ("Chordal_AMD", true,  5.0, "original", 0.0),
    ("Chordal_MFI", false, 0.0, "original", 0.0),
    ("Chordal_MFI", true,  2.0, "original", 0.0),
    ("Chordal_MFI", true,  3.0, "original", 0.0),
    ("Chordal_MFI", true,  4.0, "original", 0.0),
    ("Chordal_MFI", true,  5.0, "original", 0.0),
]

function configured_strategies()
    merge_strategy = lowercase(strip(get(ENV, "OPFDATA_MERGE_STRATEGY", "original")))

    merge_strategy in ("original", "linking") ||
        error(
            "OPFDATA_MERGE_STRATEGY must be 'original' or 'linking'; " *
            "got '$merge_strategy'."
        )

    if merge_strategy == "original"
        # Exact legacy experiment set: 15 rows.
        return copy(ORIGINAL_STRATEGIES)
    end

    # Linking-aware experiment set: one no-merge baseline and one
    # linking-aware merge for each chordal-extension heuristic.
    lambda = parse(Float64, get(ENV, "OPFDATA_LINKING_LAMBDA", "1e-7"))
    lambda >= 0 || error("OPFDATA_LINKING_LAMBDA must be >= 0")

    return [
        ("Chordal_MD",  false, 0.0, "linking", lambda),
        ("Chordal_MD",  true,  0.0, "linking", lambda),
        ("Chordal_AMD", false, 0.0, "linking", lambda),
        ("Chordal_AMD", true,  0.0, "linking", lambda),
        ("Chordal_MFI", false, 0.0, "linking", lambda),
        ("Chordal_MFI", true,  0.0, "linking", lambda),
    ]
end

function expected_csv_path(
    case_name::String,
    json_path::String,
    replicate_id::Int,
)
    obj = JSON.parsefile(json_path)

    haskey(obj, "experiment_metadata") ||
        error("Missing experiment_metadata in $json_path")

    meta = obj["experiment_metadata"]

    for key in ("dataset_type", "group_id")
        haskey(meta, key) ||
            error("experiment_metadata missing '$key' in $json_path")
    end

    dataset_type = String(meta["dataset_type"])
    group_id = String(meta["group_id"])

    file_name = splitext(basename(json_path))[1]
    rep_token = lpad(string(replicate_id), 2, '0')
    csv_name = "$(file_name)_rep_$(rep_token)_results.csv"

    output_root = normpath(joinpath(@__DIR__, "..", "outputs"))

    return joinpath(
        output_root,
        "clique_stats",
        case_name,
        dataset_type,
        group_id,
        csv_name,
    )
end


function run_strategy(
    runner::String,
    case_name::String,
    json_path::String,
    fm::String,
    merging::Bool,
    alpha::Float64,
    merge_strategy::String,
    lambda::Float64,
    replicate_id::Int,
    time_limit_sec::Float64,
)
    julia = Base.julia_cmd()

    cmd = `$julia --project=$(Base.active_project()) $runner $case_name $json_path $fm $(string(merging)) $(string(alpha)) $(string(replicate_id)) $(string(time_limit_sec)) $merge_strategy $(string(lambda))`

    println()
    println("============================================================")
    println("Running strategy")
    println("============================================================")
    println("Formulation  = ", fm)
    println("Merge        = ", merging)
    println("alpha          = ", alpha)
    println("merge_strategy = ", merge_strategy)
    println("lambda         = ", lambda)
    println("replicate_id   = ", replicate_id)
    println("time_limit   = ", time_limit_sec, " sec")
    println("============================================================")

    run(cmd)
end


function validate_completed_csv(
    csv_path::String,
    replicate_id::Int,
    strategies=configured_strategies(),
)
    isfile(csv_path) ||
        error("Expected CSV does not exist: $csv_path")

    df = CSV.read(csv_path, DataFrame)

    required_cols = [
        :Formulation,
        :Merge,
        :A_parameter,
        :replicate_id,
    ]

    missing_cols = [c for c in required_cols if !(c in propertynames(df))]
    isempty(missing_cols) ||
        error("CSV missing required columns: $missing_cols")

    # Do not only check nrow == 15. Check exact unique strategy set.
    actual = Set(
        (
            String(row.Formulation),
            Bool(row.Merge),
            Float64(row.A_parameter),
            (:MergeStrategy in propertynames(df) ? String(row.MergeStrategy) : "original"),
            (:Lambda in propertynames(df) ? Float64(row.Lambda) : 0.0),
            Int(row.replicate_id),
        )
        for row in eachrow(df)
    )

    expected = Set(
        (
            fm,
            merging,
            alpha,
            merge_strategy,
            lambda,
            replicate_id,
        )
        for (fm, merging, alpha, merge_strategy, lambda) in strategies
    )

    if actual != expected
        missing = collect(setdiff(expected, actual))
        extra = collect(setdiff(actual, expected))

        error(
            "Incomplete or inconsistent strategy CSV:\n" *
            "CSV: $csv_path\n" *
            "rows=$(nrow(df)), unique=$(length(actual))\n" *
            "missing=$(missing)\n" *
            "extra=$(extra)"
        )
    end

    if nrow(df) != length(strategies)
        error(
            "Strategy set is complete but CSV has $(nrow(df)) rows instead of $(length(strategies)). " *
            "This indicates duplicate rows."
        )
    end

    println()
    println("============================================================")
    println("Sample × replicate completed")
    println("============================================================")
    println("CSV = ", csv_path)
    println("✅ exactly ", length(strategies), " rows")
    println("✅ all configured strategy combinations present")
    println("✅ replicate_id = ", replicate_id)
    println("============================================================")

    return df
end


function run_one_sample(
    case_name::String,
    json_path::String,
    replicate_id::Int,
    time_limit_sec::Float64,
)
    json_path = abspath(json_path)

    isfile(json_path) ||
        error("JSON file does not exist: $json_path")

    runner = joinpath(@__DIR__, "run_one_case_opfdata.jl")

    isfile(runner) ||
        error(
            "Cannot find run_one_case_opfdata.jl beside this file: $runner"
        )

    csv_path = expected_csv_path(
        case_name,
        json_path,
        replicate_id,
    )
    strategies = configured_strategies()

    println("============================================================")
    println("OPFData one sample × one replicate")
    println("============================================================")
    println("case_name    = ", case_name)
    println("json_path    = ", json_path)
    println("replicate_id = ", replicate_id)
    println("time_limit   = ", time_limit_sec, " sec")
    println("result_csv   = ", csv_path)
    println("merge_mode   = ", get(ENV, "OPFDATA_MERGE_STRATEGY", "original"))
    if lowercase(get(ENV, "OPFDATA_MERGE_STRATEGY", "original")) == "linking"
        println("lambda       = ", get(ENV, "OPFDATA_LINKING_LAMBDA", "1e-7"))
    end
    println("============================================================")

    # Fast path: if the CSV is already complete, skip the whole sample.
    if isfile(csv_path)
        try
            validate_completed_csv(csv_path, replicate_id, strategies)
            println("⏭️  Entire sample × replicate already complete; skip all solves.")
            return csv_path
        catch
            println(
                "ℹ️  Existing CSV is incomplete; resume missing strategies."
            )
        end
    end

    for (idx, (fm, merging, alpha, merge_strategy, lambda)) in enumerate(strategies)
        println()
        println("################ Strategy $idx / $(length(strategies)) ################")

        run_strategy(
            runner,
            case_name,
            json_path,
            fm,
            merging,
            alpha,
            merge_strategy,
            lambda,
            replicate_id,
            time_limit_sec,
        )
    end

    validate_completed_csv(csv_path, replicate_id, strategies)

    return csv_path
end


function __main__(args)
    if length(args) != 4
        error(
            "Usage:\n" *
            "  julia --project=. run_one_sample_opfdata.jl " *
            "<case_name> <json_path> <replicate_id> <time_limit_sec>"
        )
    end

    case_name = args[1]
    json_path = args[2]
    replicate_id = parse(Int, args[3])
    time_limit_sec = parse(Float64, args[4])

    time_limit_sec > 0 ||
        error("time_limit_sec must be > 0")

    run_one_sample(
        case_name,
        json_path,
        replicate_id,
        time_limit_sec,
    )
end


__main__(ARGS)
