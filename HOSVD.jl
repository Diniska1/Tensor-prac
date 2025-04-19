using LinearAlgebra


function tensor_unfold(A, k)
    N = length(size(A))
    dims = collect(1:N)
    other_dims = filter(x -> x != k, dims)
    permutation = vcat(k, other_dims) 
    A_permuted = permutedims(A, Tuple(permutation))

    rows = size(A, k)
    cols = prod(size(A)[other_dims]) 
    return reshape(A_permuted, (rows, cols)) 
end


function tensor_multiply(A, U, k)
    N = length(size(A))
    dims = collect(1:N)
    other_dims = filter(x -> x != k, dims) 
    permutation = vcat(k, other_dims)
    A_permuted = permutedims(A, Tuple(permutation))

    rows = size(A, k)
    cols = prod(size(A)[other_dims])
    A_reshaped = reshape(A_permuted, (rows, cols))

    result_reshaped = U * A_reshaped

    output_size = collect(size(A))
    output_size[k] = size(U, 1)
    output_size = Tuple(output_size)

    result_permuted = reshape(result_reshaped, output_size[permutation])
    result = permutedims(result_permuted, Tuple(invperm(permutation)))

    return result
end

function HOSVD(A)
    G = copy(A)
    U_array = Matrix{Float64}[]
    for i in 1:length(size(A))
        U, _, _ = svd(tensor_unfold(G, i))
        push!(U_array, U)

        G = tensor_multiply(G, U', i)
    end

    return G, U_array
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

# A = float(reshape(1:24, (2, 3, 4))) # A[i,j,k]
A = tens

G, U_array = HOSVD(A)

#check
res = copy(G)
for i in reverse(1:length(U_array))
    global res = tensor_multiply(res, U_array[i], i)
end

println("Cnorm: ", maximum(abs.(vec(res) - vec(A))))
