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

function TT_orth_leftTOright!(new_cores)
    d = length(new_cores)
    
    for k in 1:d-1
        r_prev, n_k, r_k = size(new_cores[k])
        G_matrix = reshape(new_cores[k], r_prev * n_k, r_k)

        Q, R = qr(G_matrix)
        Q = Matrix(Q)  
        
        new_cores[k] = reshape(Q, r_prev, n_k, size(Q, 2))
        r_next = size(new_cores[k+1], 3)
        new_cores[k+1] = reshape(R * reshape(new_cores[k+1], size(new_cores[k+1], 1), :), (size(R, 1), size(new_cores[k+1], 2), r_next))
    end
    
    return tt_cores
end

tt = [rand(1, 3, 4), rand(4, 4, 2), rand(2, 5, 1)]
ortho_tt = TT_orth_leftTOright!(copy(tt))

println("CNORM: ", maximum(abs.(tt_to_tens(tt) - tt_to_tens(ortho_tt))))

# CHECK ORTH
is_orthogonal = true
for i in 1:length(ortho_tt)-1
    sz = size(ortho_tt[i])
    G = reshape(ortho_tt[i], sz[1]*sz[2], sz[3])
    if isapprox(G' * G, I(sz[3])) == false
        global is_orthogonal = false
        break
    end
end
println("is ORTH: ", is_orthogonal)


