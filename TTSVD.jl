
using LinearAlgebra
using TensorOperations

function tt_svd(X, tol = 1e-6)
    dims = size(X)
    d = length(dims) 
    r = ones(Int, d + 1)
    G = Vector{Array{Float64}}(undef, d)
    C = reshape(X, (r[1], dims...))

    for j = 1:d-1
        C_unfolded = reshape(C, (r[j] * dims[j], prod(dims[j+1:end])))
        U, S, V = svd(C_unfolded)
        r[j+1] = sum(S .> tol)
        G[j] = reshape(U[:, 1:r[j+1]], (r[j], dims[j], r[j+1]))
        C = reshape(Diagonal(S[1:r[j+1]]) * V[:, 1:r[j+1]]', (r[j+1], dims[j+1:end]...))
    end
    G[d] = reshape(C, (r[d], size(X,d), r[d+1]))
    return G,r
end


function tt_to_tens(tt_cores)
    d = length(tt_cores)
    full_tensor = tt_cores[1]
    new_size = Int64[size(full_tensor, 2)]

    for k in 2:d
        r_prev, n_k, r_k = size(tt_cores[k])
        push!(new_size, n_k)

        full_tensor = reshape(full_tensor, :, size(full_tensor, ndims(full_tensor)))
        core = reshape(tt_cores[k], size(tt_cores[k], 1), :)
        full_tensor = full_tensor * core
        full_tensor = reshape(full_tensor, :, n_k, r_k)
    end

    full_tensor = dropdims(full_tensor, dims=tuple(findall(size(full_tensor) .== 1)...))
    return reshape(full_tensor, Tuple(new_size))
end

n, m, p = 5,6,7
t = zeros(n,m,p)
for i in 1:n
    for j in 1:m
        for k in 1:p
            t[i, j, k] = sin((i + j + k))
        end
    end
end

cores, r = tt_svd(t,1e-5)
println("ranks: ", r)
res = tt_to_tens(cores)

println("CNORM ", maximum(abs.(vec(res) - vec(t))))
