using Documenter, Faiss

makedocs(;
    modules=[Faiss],
    format=Documenter.HTML(edit_link="main"),
    pages=[
        "Introduction" => "index.md",
        "Example" => "example.md",
        "API" => "apis.md",
    ],
    sitename="Faiss.jl",
    authors="zsz00",
)

deploydocs(;
    repo="github.com/zsz00/Faiss.jl.git",
    devbranch = "main",
    push_preview = true
)
