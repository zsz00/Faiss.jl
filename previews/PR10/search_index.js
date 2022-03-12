var documenterSearchIndex = {"docs":
[{"location":"apis/#API-Reference","page":"API","title":"API Reference","text":"","category":"section"},{"location":"apis/#Faiss-API","page":"API","title":"Faiss API","text":"","category":"section"},{"location":"apis/","page":"API","title":"API","text":"Faiss.Index\nFaiss.add\nFaiss.search\nFaiss.add_with_ids\nFaiss.remove_with_ids\nFaiss.size\nFaiss.show\nFaiss.downcast","category":"page"},{"location":"apis/#Faiss.Index","page":"API","title":"Faiss.Index","text":"Index(dim, spec=\"Flat\")\n\nCreate a Faiss index of the given dimension.\n\nThe spec is an index factory string describing the type of index to construct.\n\n\n\n\n\n","category":"type"},{"location":"apis/#Faiss.add","page":"API","title":"Faiss.add","text":"add(idx::Index, vs::AbstractMatrix)\n\nAdd the columns of vs to the index.\n\n\n\n\n\n","category":"function"},{"location":"apis/#Faiss.search","page":"API","title":"Faiss.search","text":"search(idx::Index, vs::AbstractMatrix, k::Integer)\n\nSearch the index for the k nearest neighbours of each column of vs.\n\nReturn (D, I) where I is a matrix where each column gives the ids of the k nearest neighbours of the corresponding column of vs and D is the corresponding matrix of distances.\n\n\n\n\n\n","category":"function"},{"location":"apis/#Faiss.add_with_ids","page":"API","title":"Faiss.add_with_ids","text":"add_with_ids(idx::Index, vs::AbstractMatrix, ids::Array{Int64})\n\n\n\n\n\n","category":"function"},{"location":"apis/#Faiss.remove_with_ids","page":"API","title":"Faiss.remove_with_ids","text":"remove_with_ids(idx::Index, ids::Array{Int64})\n\n\n\n\n\n","category":"function"},{"location":"apis/#Faiss.downcast","page":"API","title":"Faiss.downcast","text":"downcast(idx::Index)\n\nReturn the same index downcasted to its most specific type.\n\n\n\n\n\n","category":"function"},{"location":"apis/#Extended-Functions","page":"API","title":"Extended Functions","text":"","category":"section"},{"location":"apis/","page":"API","title":"API","text":"Faiss.add_search\nFaiss.add_search_with_ids\nFaiss.local_rank","category":"page"},{"location":"apis/#Faiss.add_search","page":"API","title":"Faiss.add_search","text":"add_search(idx::Index, vs_query::AbstractMatrix, vs_gallery::AbstractMatrix; \n            k::Integer=100, flag::Bool=true, metric::AbstractString=\"cos\")\n\nAdd vs_gallery to idx and Search the index for the k nearest neighbours of each column of vs_query.\n\nReturn (D, I) where I is a matrix where each column gives the ids of the k nearest neighbours of the corresponding column of vs and D is the corresponding matrix of distances.\n\n\n\n\n\n","category":"function"},{"location":"apis/#Faiss.add_search_with_ids","page":"API","title":"Faiss.add_search_with_ids","text":"add_search_with_ids(idx::Index, vs_query::AbstractMatrix, vs_gallery::AbstractMatrix, ids::Array{Int64}; \n                    k::Integer=100, flag::Bool=true, metric::AbstractString=\"cos\")\n\nAdd vs_gallery with ids to idx and Search the index for the k nearest neighbours of each column of vs_query.\n\n\n\n\n\n","category":"function"},{"location":"apis/#Faiss.local_rank","page":"API","title":"Faiss.local_rank","text":"local_rank(vs_query::AbstractMatrix, vs_gallery::AbstractMatrix; k::Integer=10, \n            str::String=\"Flat\", metric::String=\"L2\", gpus::String=\"\")\n\nCreate Index and Add vs_gallery to idx and Search the index for the k nearest neighbours of each column of vs_query.\n\nReturn (D, I) where I is a matrix where each column gives the ids of the k nearest neighbours of the corresponding column of vs and D is the corresponding matrix of distances.\n\n\n\n\n\n","category":"function"},{"location":"#Faiss.jl","page":"Introduction","title":"Faiss.jl","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"A simple Julia wrapper around the Faiss library for similarity search with PythonCall.jl.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"While functional and faster then NearestNeighbors.jl.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"Faiss is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning. Faiss is written in C++ with complete wrappers for Python/numpy. Some of the most useful algorithms are implemented on the GPU. It is developed primarily at Facebook AI Research.","category":"page"},{"location":"#Installation","page":"Introduction","title":"Installation","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"The package can be installed with the Julia package manager. From the Julia REPL, type ] to enter the Pkg REPL mode and run:","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"pkg> add Faiss","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"if use a already existed python env, you can:","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"julia> ENV[\"JULIA_PYTHONCALL_EXE\"] = \"/your/path/of/python\"\npkg> add Faiss","category":"page"},{"location":"#usage","page":"Introduction","title":"usage","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"using Faiss\n\nprintln(\"faiss:\", Faiss.faiss.__version__, \", gpus:\", ENV[\"CUDA_VISIBLE_DEVICES\"], \n        \", faiss path:\", Faiss.faiss.__path__[0], \", num_gpus:\", Faiss.faiss.get_num_gpus())\n# Faiss.faiss.  Enter Tab to list faiss api\n\nfeats = rand(10^4, 128);\ntop_k = 10\nfeat_dim = size(feats, 2)   # dimension\nidx = Index(feat_dim; str=\"IDMap2,Flat\", metric=\"L2\", gpus=\"4\")  # init Faiss Index\nshow(idx)   # show idx info\n\nvs_gallery = feats;\nvs_query = feats[1:100, :];\nids = collect(range(1, size(feats, 1)))\n\n# add(idx, vs_gallery)\nadd_with_ids(idx, vs_gallery, ids)\nD, I = search(idx, vs_query, top_k) \nprintln(typeof(D), size(D))\nprintln(D[1:5, :])","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"Faiss wiki","category":"page"},{"location":"example/#Faiss.jl-usage-examples","page":"Example","title":"Faiss.jl usage examples","text":"","category":"section"},{"location":"example/#example-1","page":"Example","title":"example 1","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"ENV[\"JULIA_PYTHONCALL_EXE\"] = \"/home/zhangyong/miniconda3/bin/python\"\nusing Faiss\n\nprintln(\"faiss:\", Faiss.faiss.__version__, \", gpus:\", ENV[\"CUDA_VISIBLE_DEVICES\"], \n        \", faiss path:\", Faiss.faiss.__path__[0], \", num_gpus:\", Faiss.faiss.get_num_gpus())\n# Faiss.faiss.  Enter Tab to list faiss api\n\nfeats = rand(10^4, 128);\ntop_k = 10\nfeat_dim = size(feats, 2)   # dimension\nidx = Index(feat_dim; str=\"IDMap2,Flat\", metric=\"L2\", gpus=\"4\")  # init Faiss Index\nshow(idx)   # show idx info\n\nvs_gallery = feats;\nvs_query = feats[1:100, :];\nids = collect(range(1, size(feats, 1)))\n\n# add(idx, vs_gallery)\nadd_with_ids(idx, vs_gallery, ids)\nD, I = search(idx, vs_query, top_k) \nprintln(typeof(D), size(D))\nprintln(D[1:5, :])","category":"page"},{"location":"example/#example-2","page":"Example","title":"example 2","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"using Faiss\n\nfeats = rand(10^4, 128);\ntop_k = 10\nvs_gallery = feats;\nvs_query = feats[1:100, :];\nids = collect(range(1, size(feats, 1)))\n\nD, I = local_rank(vs_query, vs_gallery, k=top_k, metric=\"IP\", gpus=\"\")\n\nprintln(typeof(D), size(D))\nprintln(D[1:5, :])\n","category":"page"},{"location":"example/#example-3","page":"Example","title":"example 3","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"\nENV[\"JULIA_PYTHONCALL_EXE\"] = \"/home/zhangyong/miniconda3/bin/python\"\nusing Faiss\nusing PythonCall\nusing ProgressMeter\n# np = pyimport(\"numpy\")\n\n\nfunction test()\n    dir_1 = \"/mnt/zy_data/data/longhu_1/sorted_2/\"\n    feats = np.load(joinpath(dir_1, \"feats.npy\"))\n    println(typeof(feats), feats.shape)\n    feats = pyconvert(Array{Float32, 2}, feats)\n    # D, I = local_rank(vs_query, vs_gallery, k=10, gpus=\"\")\n\n    feat_dim = size(feats, 2)\n    idx = Index(feat_dim; str=\"IDMap2,Flat\", metric=\"L2\", gpus=\"4\")  # IDMap2. L2,IP\n    Faiss.show(idx)\n    k = 10\n    @showprogress for i in range(1, 1000)\n        vs_gallery = feats[100*i+1:100*(i+1),:]\n        # println(typeof(feats), size(feats))\n        vs_query = vs_gallery\n        \n        # D, I = add_search(idx, vs_query, vs_gallery; k=10, flag=true, metric=\"cos\")\n        # D, I = add_search_with_ids(idx, vs_query, vs_gallery; k=10)\n        ids = collect(range(100*i+1, 100*(i+1))) .+ 100\n        println(typeof(ids), size(ids))\n        add_with_ids(idx, vs_gallery, ids)\n        D, I = search(idx, vs_query, k) \n        # println(typeof(D), size(D))\n        # println(typeof(I), size(I))\n    end\nend\n\n@time test()\n","category":"page"}]
}
