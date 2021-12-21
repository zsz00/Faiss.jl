"""
    module Faiss

An interface to the Faiss library for similarity-search of vectors (i.e. nearest-neighbour searching).

For basic usage, see [`Index`](@ref), [`add!`](@ref) and [`search`](@ref).
"""
module Faiss

using PythonCall

const faiss = PythonCall.pynew()

function __init__()
    PythonCall.pycopy!(faiss, pyimport("faiss"))
end

"""
    Index(dim, spec="Flat")

Create a Faiss index of the given dimension.

The `spec` is an index factory string describing the type of index to construct.
"""
struct Index
    py::Py
end

Base.size(x::Index) = (size(x, 1), size(x, 2))

Base.size(x::Index, i::Integer) = i == 1 ? pyconvert(Int, x.py.d) : i == 2 ? pyconvert(Int, x.py.ntotal) : error()

function Base.show(io::IO, ::MIME"text/plain", x::Index)
    print(io, typeof(x), " of ", size(x, 2), " vectors of dimension ", size(x, 1))
end

Index(d::Integer, str::AbstractString="Flat") = Index(faiss.index_factory(convert(Int, d), convert(String, str)))

"""
    add!(idx::Index, vs::AbstractMatrix)

Add the columns of `vs` to the index.
"""
function add!(x::Index, vs::AbstractMatrix)
    size(vs, 1) == size(x, 1) || error("expecting $(size(x, 1)) rows")
    vs_ = Py(convert(AbstractMatrix{Float32}, vs)').__array__()
    x.py.add(vs_)
    return x
end

"""
    search(idx::Index, vs::AbstractMatrix, k::Integer)

Search the index for the `k` nearest neighbours of each column of `vs`.

Return `(D, I)` where `I` is a matrix where each column gives the ids of the `k` nearest
neighbours of the corresponding column of `vs` and `D` is the corresponding matrix of
distances.
"""
function search(x::Index, vs::AbstractMatrix, k::Integer)
    size(vs, 1) == size(x, 1) || error("expecting $(size(x, 1)) rows")
    vs_ = Py(convert(AbstractMatrix{Float32}, vs)').__array__()
    k_ = convert(Int, k)
    D_, I_ = x.py.search(vs_, k_)
    D = PyArray{Float32,2,true,true,Float32}(D_.T)
    I = PyArray{Int64,2,true,true,Int64}(I_.T)
    return (D, I)
end

"""
    downcast(idx::Index)

Return the same index downcasted to its most specific type.
"""
downcast(x::Index) = Index(faiss.downcast_index(x.py))

end # module
