module ChordalStatsLite

using SparseArrays
using Statistics
using DataFrames
using CSV

export ChordalStats,
       compute_stats_from_vars,
       cliques_from_peo,
       stats_dataframe,
       append_stats_csv


Base.@kwdef struct ChordalStats
    # PSD 块结构
    r_max::Int = 0
    r_var::Float64 = 0.0
    t::Int = 0
    sum_r_sq::Float64 = 0.0
    sum_r_cu::Float64 = 0.0
    # 块间耦合（separators & clique tree）
    sep_max::Int = 0
    sep_mean::Float64 = 0.0
    sum_sep_sq::Float64 = 0.0
    tree_max_degree::Int = 0
    tree_height::Int = 0
    # 弦化填充
    fillin_ratio::Float64 = 1.0
    # 结构性耦合强度代理
    coupling_proxy::Float64 = 0.0
end

# ---------------- 内部工具 ----------------

# 从邻接矩阵生成邻接集合
function _neighbor_sets(cadj::AbstractMatrix{<:Integer})
    n = size(cadj, 1)
    Nbr = [Set{Int}() for _=1:n]
    @inbounds for i in 1:n, j in 1:n
        if i != j && cadj[i,j] != 0
            push!(Nbr[i], j)
        end
    end
    return Nbr
end

"由 PEO 生成候选团，并移除被包含的团（最大团近似）"
function cliques_from_peo(cadj::AbstractMatrix{<:Integer}, sigma::Vector{Int})
    n = size(cadj, 1)
    pos = zeros(Int, n); for (k,v) in enumerate(sigma); pos[v] = k; end
    Nbr = _neighbor_sets(cadj)
    cliques = Vector{Vector{Int}}()
    for v in sigma
        later = [u for u in Nbr[v] if pos[u] > pos[v]]
        clique = sort!(unique!([v; later]))
        push!(cliques, clique)
    end
    # 去重（删除被包含的团）
    sort!(cliques, by = x -> (-length(x), x))
    kept = Vector{Vector{Int}}()
    for C in cliques
        setC = Set(C)
        if all(!(setC ⊆ Set(D)) for D in kept)
            push!(kept, C)
        end
    end
    return kept
end

"构造 clique 图，边权为交集大小"
function _clique_graph(cliques::Vector{Vector{Int}})
    t = length(cliques)
    W = spzeros(Int, t, t)
    sets = map(Set, cliques)
    for i in 1:t-1, j in i+1:t
        s = length(intersect(sets[i], sets[j]))
        if s > 0
            W[i,j] = s
            W[j,i] = s
        end
    end
    return W
end

"最大生成树（按交集大小）——近似 clique tree，返回邻接表"
function _max_spanning_tree(W::SparseMatrixCSC{Int,Int})
    t = size(W,1)
    edges = Tuple{Int,Int,Int}[]
    for j in 1:t
        for idx in W.colptr[j]:(W.colptr[j+1]-1)
            i = W.rowval[idx]; w = W.nzval[idx]
            if i < j && w > 0
                push!(edges, (i,j,w))
            end
        end
    end
    sort!(edges, by = x->-x[3])  # 权重大优先
    parent = collect(1:t)
    function find(x); while parent[x] != x; x = parent[x]; end; return x; end
    function unite(a,b)
        ra, rb = find(a), find(b)
        if ra != rb; parent[rb] = ra; return true; end
        return false
    end
    T = [Int[] for _=1:t]
    cnt = 0
    for (u,v,_) in edges
        if unite(u,v)
            push!(T[u], v); push!(T[v], u)
            cnt += 1
            cnt == t-1 && break
        end
    end
    return T
end

"树高度（近似）与最大度"
function _tree_height_and_maxdeg(T::Vector{Vector{Int}})
    t = length(T)
    t == 0 && return 0, 0
    maxdeg = maximum(length.(T))

    # BFS 两次估直径，再折半作为高度近似
    function bfs(src::Int)
        dist = fill(-1, t); dist[src] = 0
        q = [src]; head = 1
        while head <= length(q)
            u = q[head]; head += 1
            for v in T[u]
                if dist[v] == -1
                    dist[v] = dist[u] + 1
                    push!(q, v)
                end
            end
        end
        far, dmax = 1, 0
        for i in 1:t
            if dist[i] > dmax; far = i; dmax = dist[i]; end
        end
        return far, dmax
    end

    a, _ = bfs(1)
    _, diam = bfs(a)
    height = ceil(Int, diam ÷ 2)
    return height, maxdeg
end

# ---------------- 主入口：----------------

"""
compute_stats_from_vars(; cadj, sigma, cliques=nothing, cadj0=nothing) -> ChordalStats

- cadj::AbstractMatrix{<:Integer}  弦化后的邻接矩阵（必需）
- sigma::Vector{Int}               PEO 消去顺序（必需）
- cliques::Vector{Vector{Int}}?    如果你已有 _maximal_cliques(cadj) 结果就传进来；否则自动由 PEO 推出
- cadj0::AbstractMatrix{<:Integer}? 原始邻接（用于 fill-in 比值；没有就别传）
"""
function compute_stats_from_vars(; cadj::AbstractMatrix{<:Float64},
                                    sigma::Vector{Int},
                                    cliques::Union{Nothing,Vector{Vector{Int}}}=nothing,
                                    cadj0::Union{Nothing,AbstractMatrix{<:Float64}}=nothing)
    cliques === nothing && (cliques = cliques_from_peo(cadj, sigma))

    rs = map(length, cliques)
    rmax = maximum(rs)
    rvar = var(rs)
    t    = length(rs)
    s2   = sum(r->r^2, rs)
    s3   = sum(r->r^3, rs)

    # separators
    W = _clique_graph(cliques)
    seps = Int[]
    for j in 1:size(W,2)
        for idx in W.colptr[j]:(W.colptr[j+1]-1)
            i = W.rowval[idx]
            i < j || continue
            w = W.nzval[idx]
            w > 0 && push!(seps, w)
        end
    end
    sep_max     = isempty(seps) ? 0   : maximum(seps)
    sep_mean    = isempty(seps) ? 0.0 : mean(seps)
    sum_sep_sq  = isempty(seps) ? 0.0 : sum(s->s^2, seps)

    # clique tree 近似
    T = _max_spanning_tree(W)
    tree_h, tree_deg = _tree_height_and_maxdeg(T)

    # fill-in
    fillin = 1.0
    if cadj0 !== nothing
        e0 = count(!iszero, cadj0) - size(cadj0,1)
        e1 = count(!iszero, cadj)  - size(cadj,1)
        fillin = e0 > 0 ? e1 / e0 : 1.0
    end

    coupling_proxy = sum_sep_sq > 0 ? (sum_sep_sq / max(s2, 1.0)) : 0.0

    return ChordalStats(;
        r_max = rmax,
        t = t,
        r_var = rvar,
        sum_r_sq = s2,
        sum_r_cu = s3,
        sep_max = sep_max,
        sep_mean = sep_mean,
        sum_sep_sq = sum_sep_sq,
        tree_max_degree = tree_deg,
        tree_height = tree_h,
        fillin_ratio = fillin,
        coupling_proxy = coupling_proxy
    )
end

# ---------------- 结果输出便捷函数 ----------------

"把 ChordalStats 转成一行 DataFrame；meta 可附加 case_id/solver 等元数据"
function stats_dataframe(stats::ChordalStats; meta::Dict{Symbol,Any}=Dict{Symbol,Any}())
    row = (;  # 关键指标
        r_max        = stats.r_max,
        r_var        = stats.r_var,
        t            = stats.t,
        sum_r_sq     = stats.sum_r_sq,
        sum_r_cu     = stats.sum_r_cu,
        sep_max      = stats.sep_max,
        sep_mean     = stats.sep_mean,
        sum_sep_sq   = stats.sum_sep_sq,
        tree_max_deg = stats.tree_max_degree,
        tree_h       = stats.tree_height,
        fillin       = stats.fillin_ratio,
        coupling     = stats.coupling_proxy,
    )
    # 合并 meta 字段（放在前面更方便筛选）
    meta_pairs = collect(pairs(meta))
    meta_df = isempty(meta_pairs) ? DataFrame() :
              DataFrame(Dict(k=>[v] for (k,v) in meta_pairs))
    stats_df = DataFrame(Dict(k=>[getfield(row, k)] for k in fieldnames(typeof(row))))
    return isempty(meta_df) ? stats_df : hcat(meta_df, stats_df)
end

"把 stats 追加写入 CSV（不存在则创建）；meta 可带 case_id/solver 等"
# path = "/home/goatoine/Documents/Lanyue/data/clique_stats/$case_name"
# pglib_opf_case14_k_0.05_1_perturbation
#Formulation,perturbation,Case,Merge,A_parameter, ID, load_id
function append_stats_csv(path::AbstractString, stats::ChordalStats; meta::Dict{Symbol,Any}=Dict{Symbol,Any}())
    df = stats_dataframe(stats; meta=meta)
    if isfile(path)
        CSV.write(path, df; append=true)
    else
        CSV.write(path, df)
    end
    return nothing
end

end # module
