ENV["JULIA_PYTHONCALL_EXE"] = "/home/zhangyong/miniconda3/bin/python"
using Pkg, Faiss, SimilaritySearch, LinearAlgebra, JLD2


function create_benchmark(dim, n, m)
    X = rand(Float32, dim, n)
    Q = rand(Float32, dim, m)
    for c in eachcol(X) normalize!(c) end
    for c in eachcol(Q) normalize!(c) end
    X, Q
end

function create_gold(X, Q, k, dist)
    db = MatrixDatabase(X)
    queries = MatrixDatabase(Q)
    ex = ExhaustiveSearch(; db, dist)
    searchbatch(ex, queries, k; parallel=true)
end

function simsearch_test(X, Q, k, dist)
     db = MatrixDatabase(X)
     k = 100
     queries = MatrixDatabase(Q)
     G = SearchGraph(; db, dist)
     index!(G; parallel_block=512)
     searchbatch(G, queries, k; parallel=true)
end

function faiss_index(X, Q, k)
	dim, n = size(X)
	str = "HNSW,Flat"
	# str = "Flat"
	metric = "IP"
	gpus = ""
	idx = Faiss.Index(dim; str, metric, gpus)  # init Faiss Index
	show(idx)   # show idx info
	add(idx, permutedims(X))
	D, I = Faiss.search(idx, permutedims(Q), k)
	I, D = permutedims(I), permutedims(D)
	I .+= 1
	I, D
end

function faiss_test(X, Q, k)
     vs_gallery = permutedims(X)
     vs_query = permutedims(Q)
     D, I = local_rank(vs_query, vs_gallery, k=k, metric="IP", gpus="")
     I, D = permutedims(I), permutedims(D)
	 I .+= 1
	 I, D
end

function main()
	dim = 128
	k = 100
	dbfile = "benchmark-$dim.jld2"
	goldfile = "gold-$dim.jld2"
	faissfile = "faiss-$dim.jld2"
	faissindexfile = "faissindex-$dim.jld2"
	simfile = "sim-$dim.jld2"
	dist = NormalizedCosineDistance()

	X, Q = if isfile(dbfile)
		load(dbfile, "X", "Q")
	else
		X, Q = create_benchmark(dim, 10^6, 10^3)
		jldsave(dbfile, X=X, Q=Q)
		X, Q
	end

	@info "== gold"
	gI, gD, exhaustivetime = if isfile(goldfile)
		load(goldfile, "gI", "gD", "exhaustivetime")
	else
		exhaustivetime = @elapsed gI, gD = create_gold(X, Q, k, dist)
		jldsave(goldfile; gI, gD, exhaustivetime)
		gI, gD, exhaustivetime
	end
	@show exhaustivetime
	@info "== faiss"
	fI, fD, faisstime = if isfile(faissfile)
		load(faissfile, "fI", "fD", "faisstime")
	else
		faisstime = @elapsed fI, fD = faiss_test(X, Q, k)
		jldsave(faissfile; fI, fD, faisstime)
		fI, fD, faisstime
	end
	@show faisstime

	@info "== faiss index"
	fiI, fiD, faissindextime = if isfile(faissindexfile)
		load(faissindexfile, "fiI", "fiD", "faissindextime")
	else
		faissindextime = @elapsed fiI, fiD = faiss_index(X, Q, k)
		jldsave(faissindexfile; fiI, fiD, faissindextime)
		fiI, fiD, faissindextime
	end
	@show faissindextime
	
	@info "== SearchGraph"
	sI, sD, simtime = if isfile(simfile)
		load(simfile, "sI", "sD", "simtime")
	else
		simtime = @elapsed sI, sD = simsearch_test(X, Q, k, dist)
		jldsave(simfile; sI, sD, simtime)
		sI, sD, simtime
	end
	@show simtime
	
	@show faisstime => macrorecall(gI, fI)
	@show faissindextime => macrorecall(gI, fiI)
	@show simtime => macrorecall(gI, sI)

	println(Pkg.status())
end

main()


#=
julia --project=/home/zhangyong/codes/Faiss.jl/Project.toml "/home/zhangyong/codes/Faiss.jl/test/test_2.jl"


=#
