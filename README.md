# Faiss.jl
 [![][docs-dev-img]][docs-dev-url]
 
A simple Julia wrapper around the [Faiss](https://github.com/facebookresearch/Faiss) library for similarity search with [`PythonCall.jl`](https://github.com/cjdoris/PythonCall.jl).

While functional and faster than [`NearestNeighbors.jl`](https://github.com/KristofferC/NearestNeighbors.jl).

Faiss is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning. Faiss is written in C++ with complete wrappers for Python/numpy. Some of the most useful algorithms are implemented on the GPU. It is developed primarily at Facebook AI Research.


## Installation
The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add Faiss CondaPkg
julia> using CondaPkg     # type ] to enter Pkg REPL mode
pkg> conda status
pkg> conda add -c pytorch
pkg> conda add faiss-gpu cudatoolkit=11.2  # Install a specific version of faiss based on your need.
```

If using an already existing Python env, you can:
```
pkg> add Faiss
julia> ENV["JULIA_PYTHONCALL_EXE"] = "/your/path/of/python"
julia> using Faiss
```

## Usage
```julia
using Faiss

println("faiss:", Faiss.faiss.__version__, ", gpus:", ENV["CUDA_VISIBLE_DEVICES"], 
        ", faiss path:", Faiss.faiss.__path__[0], ", num_gpus:", Faiss.faiss.get_num_gpus())
# Faiss.faiss.  Enter Tab to list faiss api

feats = rand(10^4, 128);
top_k = 10
feat_dim = size(feats, 2)   # dimension
idx = Index(feat_dim; str="IDMap2,Flat", metric="L2", gpus="4")  # init Faiss Index
show(idx)   # show idx info

vs_gallery = feats;
vs_query = feats[1:100, :];
ids = collect(range(1, size(feats, 1)))

# add(idx, vs_gallery)
add_with_ids(idx, vs_gallery, ids)
D, I = search(idx, vs_query, top_k) 
println(typeof(D), size(D))
println(D[1:5, :])
```

## Documentation
- [**LATEST**][docs-dev-url] &mdash; *in-development version of the documentation.* 
- [Faiss wiki](https://github.com/facebookresearch/faiss/wiki)
- [ann-benchmarks](http://ann-benchmarks.com/)

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://zsz00.github.io/Faiss.jl/dev

## Relevant Pkgs
- [NearestNeighbors.jl](https://github.com/KristofferC/NearestNeighbors.jl)
- [SimilaritySearch.jl](https://github.com/sadit/SimilaritySearch.jl)
- [Rayuela.jl](https://github.com/una-dinosauria/Rayuela.jl)
