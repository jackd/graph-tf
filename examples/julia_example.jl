using NPZ
using Laplacians
using SparseArrays
using Statistics
using LinearAlgebra

function load_data(path::String)
    data = npzread(path, [
        "adj_matrix.shape",
        "adj_matrix.indptr",
        "adj_matrix.indices",
        "adj_matrix.data",
    ])

    m, n = data["adj_matrix.shape"]
    colptr = data["adj_matrix.indptr"] .+ 1  # 1 based
    rowval = data["adj_matrix.indices"] .+ 1  # 1 based
    nzval = data["adj_matrix.data"]
    a = SparseMatrixCSC(m, n, colptr, rowval, nzval)
    return (a + a') / 2
end

function shifted_lap(a::SparseMatrixCSC, alpha)
    return sparse(Diagonal(a * ones(size(a)[1]))) - (1 - alpha) .* a
end

path = "/home/jackd/graph-tf-data/bojchevski/cora-full.npz";
alpha = 0.1

@time a = load_data(path)

@time shifted_la = shifted_lap(a, alpha)
@time sol = approxchol_sddm(shifted_la)

b = randn(size(a, 1));

@time x = sol(b);
print(norm(shifted_la * x - b) / norm(b))

@time a = load_data(path)

@time shifted_la = shifted_lap(a, alpha)
@time sol = approxchol_sddm(shifted_la)

b = randn(size(a, 1));

@time x = sol(b);
print(norm(shifted_la * x - b) / norm(b))
