# Faiss.jl

A simple Julia wrapper around the [Faiss](https://github.com/facebookresearch/Faiss) library for similarity search whith [`PythonCall.jl`](https://github.com/cjdoris/PythonCall.jl).

While functional and faster then [`NearestNeighbors.jl`](https://github.com/KristofferC/NearestNeighbors.jl).

Faiss is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning. Faiss is written in C++ with complete wrappers for Python/numpy. Some of the most useful algorithms are implemented on the GPU. It is developed primarily at Facebook AI Research.


## Installation

The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add Faiss
```
if use a already existed python env, you can:
```
julia> ENV["JULIA_PYTHONCALL_EXE"] = "your/path/of/python"
pkg> add Faiss
```

## usage
```julia
using Faiss

feats = rand(10^4, 128)
top_k = 10
feat_dim = size(feats, 2)   # # dimension
idx = Index(feat_dim; str="IDMap2,Flat", metric="L2", gpus="4")  # init Faiss Index
show(idx)   # show idx info

vs_gallery = feats
vs_query = feats[1:100, :]
ids = collect(range(1, size(feats, 1))

# add(idx, vs_gallery)
add_with_ids(idx, vs_gallery, ids)
D, I = search(idx, vs_query, top_k) 
println(typeof(D), size(D))
```

## Documentation

[Faiss wiki](https://github.com/facebookresearch/faiss/wiki)
- **STABLE** &mdash; **most recently tagged version of the documentation.** (under construction)
- [**LATEST**][docs-dev-url] &mdash; *in-development version of the documentation.*
