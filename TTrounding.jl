using LinearAlgebra

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

function tt_rounding(cores; eps=1e-10)
    d = length(cores) 
    new_cores = copy(cores) 
    
    for k in 1:d-1
        r_prev, n_k, r_k = size(new_cores[k])
        M = reshape(new_cores[k], (r_prev * n_k, r_k))
        Q, R = qr(M) 
        Q = Matrix(Q)
        new_cores[k] = reshape(Q, (r_prev, n_k, size(Q, 2)))

        if k < d
            r_next = size(new_cores[k+1], 3)
            new_cores[k+1] = reshape(R * reshape(new_cores[k+1], (r_k, :)), (size(R, 1), size(new_cores[k+1], 2), r_next))
        end
    end

    for j in reverse(2:d)
        r_old, n_j, r_j = size(new_cores[j])
        U, S, G = svd(reshape(new_cores[j], size(new_cores[j], 1), :))

        # Find new rank
        new_rank = length(S)
        for i in 2:length(S)
            if S[i] < eps
                new_rank = i - 1
                break
            end 
        end
        # println("new_rank = $new_rank")

        U = U[:, 1:new_rank]
        S = S[1:new_rank]
        G = G'[1:new_rank, :]

        new_cores[j] = reshape(G, new_rank, n_j, r_j)
        sz = size(new_cores[j-1], 2)
        new_core = reshape(new_cores[j-1], :, r_old) * U * Diagonal(S)
        # println("size_new_core: ", size(new_core))
        new_cores[j-1] = reshape(new_core, (:, sz, new_rank))
    end
    return new_cores
end


a = randn(10)
b = randn(10)
c = randn(10)

T = zeros(10, 10, 10)
for i in 1:10, j in 1:10, k in 1:10
    T[i, j, k] = a[i] * b[j] * c[k]
end

G1 = zeros(1, 10, 5)
G2 = zeros(5, 10, 5)
G3 = zeros(5, 10, 1)

G1[1, :, 1] = a
G1[1, :, 2:5] = randn(10, 4) * 1e-5
G2[1, :, 1] = b
G2[2:5, :, 2:5] = randn(4, 10, 4) * 1e-5
G3[1, :, 1] = c
G3[2:5, :, 1] = randn(4, 10) * 1e-5

cores = [G1, G2, G3]
println("CNORM: ", maximum(abs.(tt_to_tens(cores) - T)))

old_ranks = Int64[size(cores[i], 1) for i in 1:length(cores)]
push!(old_ranks, 1)

compressed_cores = tt_rounding(cores, eps=1e-10)


new_ranks = [size(core, 1) for core in compressed_cores]
push!(new_ranks, size(compressed_cores[end], 3))
println("ranks before compress: ", old_ranks)
println("ranks after compress: ", new_ranks)

new_tensor = tt_to_tens(compressed_cores)
println("CNORM res: ", maximum(abs.(new_tensor - T)))
