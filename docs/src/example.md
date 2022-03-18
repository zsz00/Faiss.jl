# Faiss.jl usage examples

## example 1 
```julia
ENV["JULIA_PYTHONCALL_EXE"] = "/home/xx/miniconda3/bin/python"
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

## example 2
#### A simple test comparison
```julia
ENV["JULIA_PYTHONCALL_EXE"] = "/home/xx/miniconda3/bin/python"
using Faiss
using NearestNeighbors

function faiss_test_1()
    feats = rand(10^6, 128);
    top_k = 100
    vs_gallery = feats;
    vs_query = feats[1:10^4, :];

    D, I = local_rank(vs_query, vs_gallery, k=top_k, metric="IP", gpus="")
end

function nn_test()
    data = rand(128, 10^6)
    k = 100
    query = rand(128, 10^4)

    kdtree = KDTree(data)
    idxs, dists = knn(kdtree, query, k, true)
end

@time faiss_test_1()
@time nn_test()

used time in my machine (2022.3):
28.934969 seconds (4.76 M allocations: 1.699 GiB, 0.14% gc time, 8.35% compilation time)
1197.160527 seconds (1.26 M allocations: 2.981 GiB, 0.02% gc time, 0.12% compilation time)
```

## example 3
```julia

ENV["JULIA_PYTHONCALL_EXE"] = "/home/xx/miniconda3/bin/python"
using Faiss
using PythonCall
using ProgressMeter
# np = pyimport("numpy")

function test()
    dir_1 = "/mnt/xx_data/data/longhu_1/sorted_2/"
    feats = np.load(joinpath(dir_1, "feats.npy"))
    println(typeof(feats), feats.shape)
    feats = pyconvert(Array{Float32, 2}, feats)
    # D, I = local_rank(vs_query, vs_gallery, k=10, gpus="")

    feat_dim = size(feats, 2)
    idx = Index(feat_dim; str="IDMap2,Flat", metric="L2", gpus="4")  # IDMap2. L2,IP
    Faiss.show(idx)
    k = 10
    @showprogress for i in range(1, 1000)
        vs_gallery = feats[100*i+1:100*(i+1),:]
        # println(typeof(feats), size(feats))
        vs_query = vs_gallery
        
        # D, I = add_search(idx, vs_query, vs_gallery; k=10, flag=true, metric="cos")
        # D, I = add_search_with_ids(idx, vs_query, vs_gallery; k=10)
        ids = collect(range(100*i+1, 100*(i+1))) .+ 100
        println(typeof(ids), size(ids))
        add_with_ids(idx, vs_gallery, ids)
        D, I = search(idx, vs_query, k) 
        # println(typeof(D), size(D))
        # println(typeof(I), size(I))
    end
end

@time test()

```

