using Documenter, Faiss

makedocs(;
    modules=[Faiss],
    format=Documenter.HTML(),
    pages=[
        "Introduction" => "index.md",
        "Example" => "example.md",
        "API" => "apis.md",
    ],
    repo="https://github.com/zsz00/Faiss.jl/blob/{commit}{path}#L{line}",
    sitename="Faiss.jl",
    authors="",
    assets=String[],
)

deploydocs(;
    repo="github.com/zsz00/Faiss.jl.git",
    push_preview = true
)
