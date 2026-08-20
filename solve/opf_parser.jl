module OPFParser
using JSON
export parse_opfdata
export validate_opfdata
export print_network_summary

"""
generate the data used by PowerModels from an OPFData file.
OPFData:
    0, 1, ..., N-1
PowerModels:
    1, 2, ..., N
"""
_bus_id(idx) = Int(idx) + 1
function _check_same_length(name::String, arrays...)
    lengths = length.(arrays)
    if length(unique(lengths)) != 1
        error(
            "$name has inconsistent lengths: " *
            join(string.(lengths), ", ")
        )
    end
end


function _require_feature_length(
    feature,
    expected::Int,
    component::String,
    idx::Int,
)
    if length(feature) != expected
        error(
            "$component $idx has $(length(feature)) features, " *
            "expected $expected."
        )
    end
end

"""
Validate the structural consistency of an OPFData example.
"""
function validate_opfdata(obj::Dict)

    haskey(obj, "grid") ||
        error("OPFData JSON does not contain 'grid'.")

    grid = obj["grid"]

    haskey(grid, "nodes") ||
        error("grid does not contain 'nodes'.")

    haskey(grid, "edges") ||
        error("grid does not contain 'edges'.")

    nodes = grid["nodes"]
    edges = grid["edges"]

    for key in ("bus", "generator", "load", "shunt")
        haskey(nodes, key) ||
            error("grid.nodes does not contain '$key'.")
    end

    for key in (
        "ac_line",
        "transformer",
        "generator_link",
        "load_link",
        "shunt_link",
    )
        haskey(edges, key) ||
            error("grid.edges does not contain '$key'.")
    end


    # --------------------------------------------------------
    # Bus features
    # --------------------------------------------------------
    buses = nodes["bus"]
    for (i, feat) in enumerate(buses)
        _require_feature_length(
            feat,
            4,
            "bus",
            i,
        )
    end
    n_bus = length(buses)
    n_bus > 0 ||
        error("OPFData example contains zero buses.")
    # --------------------------------------------------------
    # Generator features + links
    # --------------------------------------------------------

    generators = nodes["generator"]
    for (i, feat) in enumerate(generators)
        _require_feature_length(
            feat,
            11,
            "generator",
            i,
        )
    end

    gen_link = edges["generator_link"]
    _check_same_length(
        "generator_link",
        gen_link["senders"],
        gen_link["receivers"],
    )
    length(gen_link["senders"]) == length(generators) ||
        error(
            "Number of generator links does not match " *
            "number of generators."
        )


    # --------------------------------------------------------
    # Load features + links
    # --------------------------------------------------------

    loads = nodes["load"]

    for (i, feat) in enumerate(loads)
        _require_feature_length(
            feat,
            2,
            "load",
            i,
        )
    end

    load_link = edges["load_link"]
    _check_same_length(
        "load_link",
        load_link["senders"],
        load_link["receivers"],
    )

    length(load_link["senders"]) == length(loads) ||
        error(
            "Number of load links does not match number of loads."
        )


    # --------------------------------------------------------
    # Shunt features + links
    # --------------------------------------------------------

    shunts = nodes["shunt"]
    for (i, feat) in enumerate(shunts)
        _require_feature_length(
            feat,
            2,
            "shunt",
            i,
        )
    end

    shunt_link = edges["shunt_link"]
    _check_same_length(
        "shunt_link",
        shunt_link["senders"],
        shunt_link["receivers"],
    )

    length(shunt_link["senders"]) == length(shunts) ||
        error(
            "Number of shunt links does not match number of shunts."
        )


    # --------------------------------------------------------
    # AC lines
    # --------------------------------------------------------

    ac = edges["ac_line"]
    _check_same_length(
        "ac_line",
        ac["senders"],
        ac["receivers"],
        ac["features"],
    )

    for (i, feat) in enumerate(ac["features"])
        _require_feature_length(
            feat,
            9,
            "ac_line",
            i,
        )
    end


    # --------------------------------------------------------
    # Transformers
    # --------------------------------------------------------

    tr = edges["transformer"]

    _check_same_length(
        "transformer",
        tr["senders"],
        tr["receivers"],
        tr["features"],
    )

    for (i, feat) in enumerate(tr["features"])
        _require_feature_length(
            feat,
            11,
            "transformer",
            i,
        )
    end


    # --------------------------------------------------------
    # Check all bus references
    # --------------------------------------------------------

    function check_bus_index(idx, where)
        idx_i = Int(idx)

        if !(0 <= idx_i < n_bus)
            error(
                "$where references invalid OPFData bus index " *
                "$idx_i; valid range is 0:$(n_bus - 1)."
            )
        end
    end


    for (i, b) in enumerate(gen_link["receivers"])
        check_bus_index(b, "generator_link[$i]")
    end

    for (i, b) in enumerate(load_link["receivers"])
        check_bus_index(b, "load_link[$i]")
    end

    for (i, b) in enumerate(shunt_link["receivers"])
        check_bus_index(b, "shunt_link[$i]")
    end

    for i in eachindex(ac["senders"])
        check_bus_index(ac["senders"][i], "ac_line sender[$i]")
        check_bus_index(ac["receivers"][i], "ac_line receiver[$i]")
    end

    for i in eachindex(tr["senders"])
        check_bus_index(tr["senders"][i], "transformer sender[$i]")
        check_bus_index(tr["receivers"][i], "transformer receiver[$i]")
    end


    # --------------------------------------------------------
    # Reference bus
    # --------------------------------------------------------

    bus_types = Int.(round.(getindex.(buses, 2)))

    count(==(3), bus_types) >= 1 ||
        error("No reference bus (bus_type = 3) found.")


    return true
end


# ============================================================
# Individual component builders
# ============================================================


"""
Build PowerModels-compatible bus dictionaries.
OPFData bus feature order:
    1 base_kv
    2 bus_type
    3 vmin
    4 vmax
"""
function _build_buses(bus_features)
    buses = Dict{String,Any}()
    for (i, feat) in enumerate(bus_features)
        base_kv  = Float64(feat[1])
        bus_type = Int(round(feat[2]))
        vmin     = Float64(feat[3])
        vmax     = Float64(feat[4])
        bus_id = i
        buses[string(bus_id)] = Dict{String,Any}(
            "index"     => bus_id,
            "bus_i"     => bus_id,
            "bus_type"  => bus_type,

            # flat start
            "vm"        => 1.0,
            "va"        => 0.0,

            "base_kv"   => base_kv,
            "vmin"      => vmin,
            "vmax"      => vmax,

            # Not provided by OPFData; neutral defaults.
            "area"      => 1,
            "zone"      => 1,

            "source_id" => Any["bus", bus_id],
        )
    end

    return buses
end


"""
Build PowerModels-compatible generator dictionaries.

OPFData generator feature order:
    1  mbase
    2  pg
    3  pmin
    4  pmax
    5  qg
    6  qmin
    7  qmax
    8  vg
    9  cost_squared
    10 cost_linear
    11 cost_offset
"""
function _build_generators(generator_features, generator_link)

    generators = Dict{String,Any}()
    senders   = generator_link["senders"]
    receivers = generator_link["receivers"]
    gen_to_bus = Dict{Int,Int}()

    for i in eachindex(senders)
        gen_idx_0 = Int(senders[i])
        bus_idx_0 = Int(receivers[i])
        gen_to_bus[gen_idx_0] = _bus_id(bus_idx_0)
    end


    for (row_i, feat) in enumerate(generator_features)
        gen_idx_0 = row_i - 1
        haskey(gen_to_bus, gen_idx_0) ||
            error("Generator $gen_idx_0 has no generator_link.")

        gen_id  = row_i
        gen_bus = gen_to_bus[gen_idx_0]

        mbase       = Float64(feat[1])
        pg          = Float64(feat[2])
        pmin        = Float64(feat[3])
        pmax        = Float64(feat[4])
        qg          = Float64(feat[5])
        qmin        = Float64(feat[6])
        qmax        = Float64(feat[7])
        vg          = Float64(feat[8])
        cost_squared = Float64(feat[9])
        cost_linear  = Float64(feat[10])
        cost_offset  = Float64(feat[11])

        generators[string(gen_id)] = Dict{String,Any}(
            "index"       => gen_id,
            "gen_bus"     => gen_bus,

            "pg"          => pg,
            "qg"          => qg,

            "pmin"        => pmin,
            "pmax"        => pmax,
            "qmin"        => qmin,
            "qmax"        => qmax,

            "vg"          => vg,
            "mbase"       => mbase,

            "gen_status"  => 1,

            # c2 * pg^2 + c1 * pg + c0
            "model"       => 2,
            "ncost"       => 3,
            "cost"        => Float64[
                cost_squared,
                cost_linear,
                cost_offset,
            ],

            "source_id"   => Any["gen", gen_id],
        )
    end

    return generators
end


"""
Build PowerModels-compatible load dictionaries.
OPFData load feature order:
    1 pd
    2 qd
"""
function _build_loads(load_features, load_link)

    loads = Dict{String,Any}()
    senders   = load_link["senders"]
    receivers = load_link["receivers"]
    load_to_bus = Dict{Int,Int}()
    for i in eachindex(senders)
        load_idx_0 = Int(senders[i])
        bus_idx_0  = Int(receivers[i])
        load_to_bus[load_idx_0] = _bus_id(bus_idx_0)
    end

    for (row_i, feat) in enumerate(load_features)
        load_idx_0 = row_i - 1
        haskey(load_to_bus, load_idx_0) ||
            error("Load $load_idx_0 has no load_link.")
        load_id  = row_i
        load_bus = load_to_bus[load_idx_0]
        pd = Float64(feat[1])
        qd = Float64(feat[2])
        loads[string(load_id)] = Dict{String,Any}(
            "index"     => load_id,
            "load_bus"  => load_bus,
            "status"    => 1,
            "pd"        => pd,
            "qd"        => qd,
            "source_id" => Any["load", load_id],
        )
    end

    return loads
end


"""
Build PowerModels-compatible shunt dictionaries.
OPFData shunt feature order:
    1 bs
    2 gs

NOTE:
The OPFData paper specifies the order as [bs, gs].
"""
function _build_shunts(shunt_features, shunt_link)

    shunts = Dict{String,Any}()
    senders   = shunt_link["senders"]
    receivers = shunt_link["receivers"]
    shunt_to_bus = Dict{Int,Int}()
    for i in eachindex(senders)
        shunt_idx_0 = Int(senders[i])
        bus_idx_0   = Int(receivers[i])

        shunt_to_bus[shunt_idx_0] = _bus_id(bus_idx_0)
    end


    for (row_i, feat) in enumerate(shunt_features)

        shunt_idx_0 = row_i - 1

        haskey(shunt_to_bus, shunt_idx_0) ||
            error("Shunt $shunt_idx_0 has no shunt_link.")

        shunt_id  = row_i
        shunt_bus = shunt_to_bus[shunt_idx_0]

        bs = Float64(feat[1])
        gs = Float64(feat[2])

        shunts[string(shunt_id)] = Dict{String,Any}(
            "index"      => shunt_id,
            "shunt_bus"  => shunt_bus,
            "status"     => 1,
            "gs"         => gs,
            "bs"         => bs,
            "source_id"  => Any["shunt", shunt_id],
        )
    end

    return shunts
end


# ============================================================
# Branch builders
# ============================================================


"""
Append OPFData AC lines to data["branch"].

AC line feature order:
    1 angmin
    2 angmax
    3 b_fr
    4 b_to
    5 br_r
    6 br_x
    7 rate_a
    8 rate_b
    9 rate_c
"""
function _append_ac_lines!(
    branches::Dict{String,Any},
    ac_line,
    next_branch_id::Int,
)

    senders   = ac_line["senders"]
    receivers = ac_line["receivers"]
    features  = ac_line["features"]

    branch_id = next_branch_id

    for i in eachindex(senders)

        f_bus = _bus_id(senders[i])
        t_bus = _bus_id(receivers[i])

        feat = features[i]

        angmin = Float64(feat[1])
        angmax = Float64(feat[2])
        b_fr   = Float64(feat[3])
        b_to   = Float64(feat[4])
        br_r   = Float64(feat[5])
        br_x   = Float64(feat[6])
        rate_a = Float64(feat[7])
        rate_b = Float64(feat[8])
        rate_c = Float64(feat[9])

        branches[string(branch_id)] = Dict{String,Any}(
            "index"        => branch_id,

            "f_bus"        => f_bus,
            "t_bus"        => t_bus,

            "br_r"         => br_r,
            "br_x"         => br_x,

            "g_fr"         => 0.0,
            "g_to"         => 0.0,
            "b_fr"         => b_fr,
            "b_to"         => b_to,

            "rate_a"       => rate_a,
            "rate_b"       => rate_b,
            "rate_c"       => rate_c,

            # Ordinary AC line.
            "tap"          => 1.0,
            "shift"        => 0.0,

            "angmin"       => angmin,
            "angmax"       => angmax,

            "br_status"    => 1,

            "transformer"  => false,

            "source_id"    => Any["branch", branch_id],
        )

        branch_id += 1
    end

    return branch_id
end


"""
Append OPFData transformers to data["branch"].

Transformer feature order:
    1  angmin
    2  angmax
    3  br_r
    4  br_x
    5  rate_a
    6  rate_b
    7  rate_c
    8  tap
    9  shift
    10 b_fr
    11 b_to
"""
function _append_transformers!(
    branches::Dict{String,Any},
    transformer,
    next_branch_id::Int,
)

    senders   = transformer["senders"]
    receivers = transformer["receivers"]
    features  = transformer["features"]

    branch_id = next_branch_id

    for i in eachindex(senders)

        f_bus = _bus_id(senders[i])
        t_bus = _bus_id(receivers[i])

        feat = features[i]

        angmin = Float64(feat[1])
        angmax = Float64(feat[2])
        br_r   = Float64(feat[3])
        br_x   = Float64(feat[4])
        rate_a = Float64(feat[5])
        rate_b = Float64(feat[6])
        rate_c = Float64(feat[7])
        tap    = Float64(feat[8])
        shift  = Float64(feat[9])
        b_fr   = Float64(feat[10])
        b_to   = Float64(feat[11])

        branches[string(branch_id)] = Dict{String,Any}(
            "index"        => branch_id,

            "f_bus"        => f_bus,
            "t_bus"        => t_bus,

            "br_r"         => br_r,
            "br_x"         => br_x,

            "g_fr"         => 0.0,
            "g_to"         => 0.0,
            "b_fr"         => b_fr,
            "b_to"         => b_to,

            "rate_a"       => rate_a,
            "rate_b"       => rate_b,
            "rate_c"       => rate_c,

            "tap"          => tap,
            "shift"        => shift,

            "angmin"       => angmin,
            "angmax"       => angmax,

            "br_status"    => 1,

            "transformer"  => true,

            "source_id"    => Any["branch", branch_id],
        )

        branch_id += 1
    end

    return branch_id
end


function _build_branches(edges)

    branches = Dict{String,Any}()

    next_id = 1

    next_id = _append_ac_lines!(
        branches,
        edges["ac_line"],
        next_id,
    )

    _append_transformers!(
        branches,
        edges["transformer"],
        next_id,
    )

    return branches
end


# ============================================================
# Main parser
# ============================================================

"""
parse_opfdata(json_path; validate=true)
Convert one raw OPFData example JSON into a
PowerModels-compatible network dictionary.
"""
function parse_opfdata(
    json_path::AbstractString;
    validate::Bool = true,
)

    isfile(json_path) ||
        error("OPFData file does not exist: $json_path")

    obj = JSON.parsefile(json_path)

    if validate
        validate_opfdata(obj)
    end

    grid  = obj["grid"]
    nodes = grid["nodes"]
    edges = grid["edges"]
    # --------------------------------------------------------
    # baseMVA
    # --------------------------------------------------------

    haskey(grid, "context") ||
        error("grid does not contain context/baseMVA.")

    context = grid["context"]

    baseMVA = try
        Float64(context[1][1][1])
    catch
        error(
            "Could not parse OPFData context. " *
            "Expected context[1][1][1] = baseMVA."
        )
    end


    # --------------------------------------------------------
    # Build PowerModels data dictionary
    # --------------------------------------------------------

    data = Dict{String,Any}()
    data["name"]        = splitext(basename(json_path))[1]
    data["source_type"] = "opfdata"
    data["source_file"] = abspath(json_path)
    data["baseMVA"]     = baseMVA
    data["per_unit"]    = true
    data["bus"] = _build_buses(
        nodes["bus"],
    )
    data["gen"] = _build_generators(
        nodes["generator"],
        edges["generator_link"],
    )

    data["load"] = _build_loads(
        nodes["load"],
        edges["load_link"],
    )

    data["shunt"] = _build_shunts(
        nodes["shunt"],
        edges["shunt_link"],
    )

    data["branch"] = _build_branches(
        edges,
    )
    data["dcline"]  = Dict{String,Any}()
    data["switch"]  = Dict{String,Any}()
    data["storage"] = Dict{String,Any}()


    return data
end



function print_network_summary(data)

    println()
    println("==============================================")
    println(" OPFData → PowerModels network summary")
    println("==============================================")

    println("name       = ", get(data, "name", ""))
    println("baseMVA    = ", get(data, "baseMVA", missing))

    println("buses      = ", length(get(data, "bus", Dict())))
    println("generators = ", length(get(data, "gen", Dict())))
    println("loads      = ", length(get(data, "load", Dict())))
    println("shunts     = ", length(get(data, "shunt", Dict())))
    println("branches   = ", length(get(data, "branch", Dict())))

    n_transformers = count(
        br -> get(br, "transformer", false),
        values(get(data, "branch", Dict())),
    )

    println(
        "AC lines   = ",
        length(get(data, "branch", Dict())) - n_transformers,
    )

    println("transformers = ", n_transformers)

    ref_buses = [
        b["bus_i"]
        for b in values(get(data, "bus", Dict()))
        if get(b, "bus_type", 1) == 3
    ]

    println("reference buses = ", sort(ref_buses))

    println("==============================================")
    println()

    return nothing
end

end # module OPFParser