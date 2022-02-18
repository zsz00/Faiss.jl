using Documenter, Faiss

makedocs(;
    modules=[Faiss],
    format=Documenter.HTML(),
    pages=[
        "Introduction" => "index.md",
        "API" => "apis.md",
        "example" => "example.md",
    ],
    repo="https://github.com/zsz00/Faiss.jl/blob/{commit}{path}#L{line}",
    sitename="Faiss.jl",
    authors="",
    assets=String[],
)

deploydocs(;
    repo="github.com/zsz00/Faiss.jl",
)
