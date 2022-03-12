using Documenter, Faiss

makedocs(;
    modules=[Faiss],
    format=Documenter.HTML(edit_link="main"),
    pages=[
        "Introduction" => "index.md",
        "Example" => "example.md",
        "API" => "apis.md",
    ],
    repo="https://github.com/zsz00/Faiss.jl/blob/{commit}{path}#L{line}",
    sitename="Faiss.jl",
    authors="zsz00",
)

deploydocs(;
    repo="github.com/zsz00/Faiss.jl.git",
    devbranch = "main",
    push_preview = true
)
