"""
    ngrams(token, n=3, start_end_token='#')

Return the `n`grams of `token` as a Vector{SubString{String}} including `start_end_token`.
"""
function ngrams(token, n=3, start_end_token='#')
    s = string(start_end_token, token, start_end_token)
    [SubString(s, k, k+n-1) for k in 1:(length(s)+1-n)]
end

struct CueMatrix
    cue::SparseMatrixCSC{Float64,Int}
    f2i::Dict{SubString{String},Int}
    i2f::Dict{Int,SubString{String}}
    gold_inds::Vector{Vector{Int}}
    A::SparseMatrixCSC{Float64,Int}
end

function CueMatrix(tokens, grams=3, start_end_token='#')
    ngms = ngrams.(tokens, grams, start_end_token)       # vector of vectors of ngrams
    features = foldl(union, ngms)                        # vector of unique ngrams
    nfeat = length(features)
    f2i = Dict(v => i for (i, v) in enumerate(features)) # map ngram -> index

    inds = [[f2i[x] for x in y] for y in ngms]           # vector of vectors of indices

    CueMatrix(
        sparse(                                          # sparse cue matrix
            foldl(append!, fill(i, length(v)) for (i,v) in enumerate(inds)),
            foldl(append!, inds, init=Int[]), 1.0, length(tokens), nfeat,
        ),
        f2i,
        Dict(i => v for (i, v) in enumerate(features)), # map index -> ngram
        inds,
        sparse(                                         # adjacency matrix
            foldl(append!, [view(v, 1:(length(v) - 1)) for v in inds], init=Int[]),
            foldl(append!, [view(v, 2:length(v)) for v in inds], init=Int[]),
            1.0, nfeat, nfeat, *,                       # * combiner so nz's are all 1's
        )
    )
end
