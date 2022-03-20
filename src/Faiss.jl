"""
    module Faiss

An interface to the Faiss library for similarity-search of vectors (i.e. nearest-neighbour searching).

For basic usage, see [`Index`](@ref), [`add`](@ref), [`search`](@ref), [`add_with_ids`](@ref), [`local_rank`](@ref).
"""
module Faiss

using PythonCall
export np, Index, add, search, add_with_ids, remove_with_ids, add_search, local_rank, add_search_with_ids

const faiss = PythonCall.pynew()
const np = PythonCall.pynew()

function __init__()
    PythonCall.pycopy!(faiss, pyimport("faiss"))
    PythonCall.pycopy!(np, pyimport("numpy"))
end

struct Index
    py::Py
end

# cpu index
Index(dim::Integer, str::AbstractString="Flat", metric::Integer=1) = 
    Index(faiss.index_factory(convert(Int, dim), convert(String, str), metric))

"""
    Index(dim::Integer; str::AbstractString="Flat", metric::String="L2", gpus::String="")

Create a Faiss index of the given dimension and factory string.
- The `dim` is denote dimension of data to Index.
- The `str` is an index factory string describing the type of index to construct. reference:[index-factory](https://github.com/facebookresearch/faiss/wiki/The-index-factory)
- The `metric` is a metric of distance, have "L2", "IP"
- The `gpus` is a string of setting gpu id. if "" denote use cpu.
"""
function Index(dim::Integer; str::AbstractString="Flat", metric::String="L2", gpus::String="")
    # feat数据存储在这里面. 数据量巨大时,容易爆显存
    ENV["CUDA_VISIBLE_DEVICES"] = gpus
    if metric == "L2"
        metric_flag = faiss.METRIC_L2   # 1
    elseif metric == "IP"
        metric_flag = faiss.METRIC_INNER_PRODUCT  # 2
    end
    metric_flag = pyconvert(Integer, metric_flag)
    str_list = split(str, ",")
    if "IDMap2" in str_list
        str = join(str_list[str_list .!="IDMap2"] , ",")  # py faiss not support IDMap2.
        cpu_index = Index(dim, str, metric_flag)
        cpu_index = faiss.IndexIDMap2(cpu_index.py)
    else
        cpu_index = Index(dim, str, metric_flag)
        cpu_index = cpu_index.py
    end

    # println("faiss:", faiss.__version__, " gpus:", ENV["CUDA_VISIBLE_DEVICES"])
    if gpus == ""
        ngpus = 0
    else
        ngpus = length(split(ENV["CUDA_VISIBLE_DEVICES"], ","))
    end
    
    if ngpus == 0
        index = cpu_index
    elseif ngpus == 1
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # use one gpu. make it into a gpu index
    else
        index = faiss.index_cpu_to_all_gpus(cpu_index)  # use all gpus
    end
    return Index(index)
end

"""
    size(idx::Index)

"""
Base.size(idx::Index) = (size(idx, 1), size(idx, 2))

Base.size(idx::Index, i::Integer) = i == 1 ? pyconvert(Int, idx.py.d) : i == 2 ? pyconvert(Int, idx.py.ntotal) : error()

"""
    show(io::IO, ::MIME"text/plain", idx::Index)

"""
function Base.show(io::IO, ::MIME"text/plain", idx::Index)
    metric_dict = Dict(1=>"METRIC_L2", 2=>"METRIC_INNER_PRODUCT")
    println(io, typeof(idx), " of ", size(idx, 2), " vectors of dimension ", size(idx, 1), 
    ", metric_type:", metric_dict[pyconvert(Int64, idx.py.metric_type)])
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
    return nothing
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
    return nothing
end

"""
    remove_with_ids(idx::Index, ids::Array{Int64})

"""
function remove_with_ids(idx::Index, ids::Array{Int64})
    ids_ = np.array(pyrowlist(ids), dtype=np.int64)
    idx.py.remove_ids(ids_)
end

"""
    search(idx::Index, vs::AbstractMatrix, k::Integer; threshold::Real=0.0)

Search the index for the `k` nearest neighbours of each column of `vs`.

Return `(D, I)` where `I` is a matrix where each column gives the ids of the `k` nearest
neighbours of the corresponding column of `vs` and `D` is the corresponding matrix of
distances.
"""
function search(idx::Index, vs::AbstractMatrix, k::Integer; threshold::Real=0.0)
    size(vs, 2) == size(idx, 1) || error("expecting $(size(idx, 1)) rows")
    # vs_ = Py(convert(AbstractMatrix{Float32}, vs)').__array__()
    vs_ = convert(AbstractMatrix{Float32}, vs)
    vs_ = np.array(pyrowlist(vs_), dtype=np.float32)
    k_ = convert(Int, k)
    th_ = convert(Float32, threshold)

    if pyconvert(Int64, idx.py.ntotal) == 0
        size_1 = size(vs, 1)
        D = zeros(Float32, (size_1, k))
        I = zeros(Int32, (size_1, k))
        if pyconvert(Int64, idx.py.metric_type) == 1
            D = D .+ 2
        end
    else
        D_, I_ = idx.py.search(vs_, k_, threshold=th_)
        D = pyconvert(Array{Float32, 2}, D_) 
        I = pyconvert(Array{Int32, 2}, I_)
    end

    return (D, I)
end


"""
    add_search(idx::Index, vs_query::AbstractMatrix, vs_gallery::AbstractMatrix; 
                k::Integer=100, threshold::Real=0.0, flag::Bool=true)

Add `vs_gallery` to idx and Search the index for the `k` nearest neighbours of each column of `vs_query`.

Return `(D, I)` where `I` is a matrix where each column gives the ids of the `k` nearest
neighbours of the corresponding column of `vs` and `D` is the corresponding matrix of distances.
"""
function add_search(idx::Index, vs_query::AbstractMatrix, vs_gallery::AbstractMatrix; 
                    k::Integer=100, threshold::Real=0.0, flag::Bool=true)
    if flag
        add(idx, vs_gallery)
    end
    D, I = search(idx, vs_query, k; threshold=threshold) 

    return (D, I)
end

"""
    add_search_with_ids(idx::Index, vs_query::AbstractMatrix, vs_gallery::AbstractMatrix, ids::Array{Int64}; 
                        k::Integer=100, threshold::Real=0.0, flag::Bool=true)

Add `vs_gallery` with `ids` to idx and Search the index for the `k` nearest neighbours of each column of `vs_query`.

"""
function add_search_with_ids(idx::Index, vs_query::AbstractMatrix, vs_gallery::AbstractMatrix, ids::Array{Int64}; 
    k::Integer=100, threshold::Real=0.0, flag::Bool=true)
    if flag
        add_with_ids(idx, vs_gallery, ids)
    end
    D, I = search(idx, vs_query, k; threshold=threshold) 
    
    return (D, I)
end

"""
    local_rank(vs_query::AbstractMatrix, vs_gallery::AbstractMatrix; k::Integer=10, 
                str::String="Flat", metric::String="L2", threshold::Real=0.0, gpus::String="")

Create Index and Add `vs_gallery` to idx and Search the index for the `k` nearest neighbours of each column of `vs_query`.

Return `(D, I)` where `I` is a matrix where each column gives the ids of the `k` nearest
neighbours of the corresponding column of `vs` and `D` is the corresponding matrix of distances.
"""
function local_rank(vs_query::AbstractMatrix, vs_gallery::AbstractMatrix; k::Integer=10, 
                    str::String="Flat", metric::String="L2", threshold::Real=0.0, gpus::String="")
    feat_dim = size(vs_query, 2)
    idx = Index(feat_dim; str=str, metric=metric, gpus=gpus)
    D, I = add_search(idx, vs_query, vs_gallery; k=k, threshold=threshold)
    # PythonCall.pydel!(idx.py)
    return (D, I)
end


"""
    downcast(idx::Index)

Return the same index downcasted to its most specific type.
"""
downcast(idx::Index) = Index(faiss.downcast_index(idx.py))

end # module
