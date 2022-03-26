ENV["JULIA_PYTHONCALL_EXE"] = "/home/zhangyong/miniconda3/bin/python"
using Faiss
using PythonCall
using ProgressMeter
using NearestNeighbors
# np = pyimport("numpy")


function test()
    dir_1 = "/mnt/zy_data/data/longhu_1/sorted_2/"
    feats = np.load(joinpath(dir_1, "feats.npy"))
    println(typeof(feats), feats.shape)
    # feats = PyArray{Float32, 2, true, true}(feats)
    feats = pyconvert(Array{Float32, 2}, feats)  # py to julia Matrix
    # D, I = local_rank(vs_query, vs_gallery, k=10, metric="IP", gpus="")

    feat_dim = size(feats, 2)
    idx = Index(feat_dim; str="IDMap,Flat", metric="IP", gpus="4")  # IDMap2. L2,IP
    Faiss.show(idx)
    k = 10
    @showprogress for i in range(1, 1000)
        vs_gallery = feats[100*i+1:100*(i+1),:]
        # println(typeof(feats), size(feats))
        vs_query = vs_gallery
        
        # D, I = add_search(idx, vs_query, vs_gallery; k=10, flag=true)
        
        ids = collect(range(100*i+1, 100*(i+1))) .+ 100
        D, I = add_search_with_ids(idx, vs_query, vs_gallery, ids; k=10)
        # println(typeof(ids), size(ids))
        # add_with_ids(idx, vs_gallery, ids)
        # D, I = search(idx, vs_query, k) 
        # println(typeof(D), size(D))
        # println(typeof(I), size(I))
        if i == 2
            println(D[1:2, 1:5])
            println(I[1:2, 1:5])
            break
        end
    end
end

function faiss_test_1()
    feats = rand(10^6, 128);
    top_k = 100
    vs_gallery = feats;
    vs_query = feats[1:10^4, :];

    D, I = local_rank(vs_query, vs_gallery, k=top_k, metric="IP", gpus="")

    # println(typeof(D), size(D))
    # println(D[1:5, :])
end

function nn_test()
    data = rand(128, 10^6)
    k = 100
    query = rand(128, 10^4)

    kdtree = KDTree(data)
    idxs, dists = knn(kdtree, query, k, true)
    # println(typeof(idxs), size(idxs))
end


function faiss_test_2()
    feats = rand(10^4, 128);
    top_k = 100
    vs_gallery = feats;
    vs_query = feats[1:10^2, :];

    feat_dim = size(feats, 2)
    idx = Index(feat_dim; str="Flat", metric="IP", gpus="")  # IDMap2. L2,IP  IDMap2,
    Faiss.show(idx)
    k = 10

    add(idx, vs_gallery)
    # D, I = search(idx, vs_query, k) 
    D, I = range_search(idx, vs_query, 20) 
    println(size(D))
    # println(D[1:5, :])
end


# @time test()
# @time faiss_test_1()
# @time nn_test()
@time faiss_test_2()

#=
julia --project=/home/zhangyong/codes/Faiss.jl/Project.toml "/home/zhangyong/codes/Faiss.jl/test/test_1.jl"

cpu: bs=100,k=10
237.4seconds=4min (6.41 M allocations: 800.244 MiB, 0.13% gc time, 1.39% compilation time)
gpu: bs=100,k=10
25.010865 seconds (6.38 M allocations: 797.798 MiB, 1.01% gc time, 14.27% compilation time)


add does not make sense with IndexIDMap, use add_with_ids
IndexIDMap 只支持add_with_ids?

=#

