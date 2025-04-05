using LinearAlgebra

function get_slice(A, direction::Int, index::Int, N)
    slice_indices = Vector{Any}(fill(:, N))
    slice_indices[direction] = index
    return A[Tuple(slice_indices)...]
end

function khatri_rao(A, B)
    m1, n1 = size(A)
    m2, n2 = size(B)
    res = zeros(Float64, m1 * m2, n1)
    for i in 1:n1
        res[:, i] = kron(B[:, i], A[:, i])
    end
    return res
end

function als_tensor(X, R, max_iter=100)
    dims = size(X)
    
    num_factors = length(dims)
    factors = Matrix{Float64}[]
    for i in 1:num_factors
        push!(factors, rand(dims[i], R))
    end

    for iter in 1:max_iter
        for k in 1:num_factors
            for j in 1:dims[k]
                BC_kr = ones(1, R)
                for i in 1:num_factors
                    if i != k
                        BC_kr = khatri_rao(BC_kr, factors[i])
                    end
                end
                X_slice = vec(get_slice(X, k, j, length(dims)))
                
                factors[k][j, :] = (BC_kr' * BC_kr) \ (BC_kr' * X_slice)
            end
        end
    end

    return factors
end

n = 2
m = 3
k = 4
tens = zeros(n,m,k)

for i in 1:n
     for j in 1:m
          for k in 1:k
               tens[i,j,k] = i + j + k
          end
     end
end

rank = 4

factors = als_tensor(tens, rank)

global khatr = ones(1, rank)
for i in 1:length(size(tens))
    global khatr = khatri_rao(khatr, factors[i])
end

t = sum(khatr, dims = 2) 
println("CNORM: ", maximum(abs.(t - reshape(tens,(prod(size(tens)), 1)))))