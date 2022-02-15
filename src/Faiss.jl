"""
    module Faiss

An interface to the Faiss library for similarity-search of vectors (i.e. nearest-neighbour searching).

For basic usage, see [`Index`](@ref), [`add`](@ref) and [`search`](@ref).
"""
module Faiss

using PythonCall
export np, Index, add, search, add_search, local_rank, add_with_ids, remove_with_ids, add_search_with_ids

const faiss = PythonCall.pynew()
const np = PythonCall.pynew()

function __init__()
    PythonCall.pycopy!(faiss, pyimport("faiss"))
    PythonCall.pycopy!(np, pyimport("numpy"))
end

"""
    Index(dim, spec="Flat")

Create a Faiss index of the given dimension.

The `spec` is an index factory string describing the type of index to construct.
"""
struct Index
    py::Py
end

# Advanced API
Index(d::Integer, str::AbstractString="Flat") = Index(faiss.index_factory(convert(Int, d), convert(String, str)))

# Basic API
function Index(feat_dim::Integer, gpus::String)
    # feat数据存储在这里面. 数据量巨大时,容易爆显存
    if gpus == ""
        ngpus = 0
    else
        ngpus = length(split(gpus, ","))
    end
    if ngpus == 0
        cpu_index = faiss.IndexFlatL2(feat_dim)
        index = cpu_index
    elseif ngpus == 1
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0  # int(gpus[0])
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, feat_dim, flat_config)  # use one gpu. 
    else
        cpu_index = faiss.IndexFlatL2(feat_dim)
        index = faiss.index_cpu_to_all_gpus(cpu_index)  # use all gpus
    end
    index = faiss.IndexIDMap2(index)
    Index(index)
end

Base.size(idx::Index) = (size(idx, 1), size(idx, 2))

Base.size(idx::Index, i::Integer) = i == 1 ? pyconvert(Int, idx.py.d) : i == 2 ? pyconvert(Int, idx.py.ntotal) : error()

function Base.show(io::IO, ::MIME"text/plain", idx::Index)
    print(io, typeof(idx), " of ", size(idx, 2), " vectors of dimension ", size(idx, 1))
end


"""
    add(idx::Index, vs::AbstractMatrix)

Add the columns of `vs` to the index.
"""
function add(idx::Index, vs::AbstractMatrix)
    size(vs, 2) == size(idx, 1) || error("expecting $(size(idx, 1)) rows")
    # vs_ = Py(convert(AbstractMatrix{Float32}, vs)').__array__()
    vs_ = convert(AbstractMatrix{Float32}, vs)
    vs_ = np.array(pyrowlist(vs_), dtype=np.float32)
    idx.py.add(vs_)
    return idx
end

"""
    add_with_ids(idx::Index, vs::AbstractMatrix, ids::Array{Int64})

"""
function add_with_ids(idx::Index, vs::AbstractMatrix, ids::Array{Int64})
    size(vs, 2) == size(idx, 1) || error("expecting $(size(idx, 1)) rows")
    size(vs, 1) == size(ids, 1) || error("expecting $(size(vs, 1)) rows")
    # vs_ = Py(convert(AbstractMatrix{Float32}, vs)').__array__()
    vs_ = convert(AbstractMatrix{Float32}, vs)
    vs_ = np.array(pyrowlist(vs_), dtype=np.float32)
    ids_ = np.array(pyrowlist(ids), dtype=np.int64)
    idx.py.add_with_ids(vs_, ids_)
    return idx
end

"""
    remove_with_ids(idx::Index, ids::Array{Int64})

"""
function remove_with_ids(idx::Index, ids::Array{Int64})
    ids_ = np.array(pyrowlist(ids), dtype=np.int64)
    idx.remove_ids(ids_)
end

"""
    search(idx::Index, vs::AbstractMatrix, k::Integer)

Search the index for the `k` nearest neighbours of each column of `vs`.

Return `(D, I)` where `I` is a matrix where each column gives the ids of the `k` nearest
neighbours of the corresponding column of `vs` and `D` is the corresponding matrix of
distances.
"""
function search(idx::Index, vs::AbstractMatrix, k::Integer; metric::AbstractString="cos")
    size(vs, 2) == size(idx, 1) || error("expecting $(size(idx, 1)) rows")
    # vs_ = Py(convert(AbstractMatrix{Float32}, vs)').__array__()
    vs_ = convert(AbstractMatrix{Float32}, vs)
    vs_ = np.array(pyrowlist(vs_), dtype=np.float32)
    k_ = convert(Int, k)

    D_, I_ = idx.py.search(vs_, k_)

    D = pyconvert(Array{Float32, 2}, D_) 
    I = pyconvert(Array{Int64, 2}, I_)
    if metric == "cos"
        D = 1.0 .- D / 2.0   # 转换为cos相似度
    end
    return (D, I)
end


"""
    add_search(idx::Index, vs_query::AbstractMatrix, vs_gallery::AbstractMatrix; 
                k::Integer=100, flag::Bool=true, metric::AbstractString="cos")

Add `vs_gallery` to idx and Search the index for the `k` nearest neighbours of each column of `vs_query`.

Return `(D, I)` where `I` is a matrix where each column gives the ids of the `k` nearest
neighbours of the corresponding column of `vs` and `D` is the corresponding matrix of distances.
"""
function add_search(idx::Index, vs_query::AbstractMatrix, vs_gallery::AbstractMatrix; 
                    k::Integer=100, flag::Bool=true, metric::AbstractString="cos")
    if flag
        add(idx, vs_gallery)
    end
    D, I = search(idx, vs_query, k) 

    return (D, I)
end

function add_search_with_ids(idx::Index, vs_query::AbstractMatrix, vs_gallery::AbstractMatrix, ids::Array{Int64}; 
    k::Integer=100, flag::Bool=true, metric::AbstractString="cos")
    if flag
        add_with_ids(idx, vs_gallery, ids)
    end
    D, I = search(idx, vs_query, k) 
    
    return (D, I)
end

"""
    local_rank(vs_query::AbstractMatrix, vs_gallery::AbstractMatrix; k::Integer=10, 
                metric::AbstractString="cos", gpus::AbstractString="")

Create Index and Add `vs_gallery` to idx and Search the index for the `k` nearest neighbours of each column of `vs_query`.

Return `(D, I)` where `I` is a matrix where each column gives the ids of the `k` nearest
neighbours of the corresponding column of `vs` and `D` is the corresponding matrix of distances.
"""
function local_rank(vs_query::AbstractMatrix, vs_gallery::AbstractMatrix; k::Integer=10, 
                    metric::AbstractString="cos", gpus::AbstractString="")
    feat_dim = size(vs_query, 2)
    idx = Index(feat_dim, gpus)
    D, I = add_search(idx, vs_query, vs_gallery; k=k, metric=metric)
    return (D, I)
end


"""
    downcast(idx::Index)

Return the same index downcasted to its most specific type.
"""
downcast(idx::Index) = Index(faiss.downcast_index(idx.py))

end # module
