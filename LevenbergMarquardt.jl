using LinearAlgebra
using Random

"""
SOURCE: 
https://www.researchgate.net/publication/369217167_A_Levenberg-Marquardt_Method_for_Tensor_Approximation
"""

function compute_A_approx(u_vec, I_dims, m)
    A_approx = zeros(Tuple(I_dims))
    offset = 0
    for r in 1:R
        factors = []
        for j in 1:m
            len_j = I_dims[j]
            vec_part = u_vec[offset+1 : offset+len_j]
            push!(factors, vec_part)
            offset += len_j
        end

        rank1_tens = factors[1]
        for v in factors[2:end]
            rank1_tens = kron(rank1_tens, v)
        end
        A_approx += reshape(rank1_tens, Tuple(I_dims))
    end
    return A_approx
end

function f(u_vec, I_dims, m)
    return vec(A - compute_A_approx(u_vec, I_dims, m))
end

function jacobian(u_vec, N0, Nd, dims, I_dims, m; delta=1e-6)
    J = zeros(N0, Nd)
    f0 = f(u_vec, I_dims,m)
    for i in 1:Nd
        du = zeros(Nd)
        du[i] = delta
        J[:, i] = (f(u_vec + du, I_dims,m) - f0) / delta
    end
    return J
end

function lev_mark(A, R; max_iter=100, eps1=1e-10, eps2=1e-10,
                                                 tau=1e-3, eta=2.0, sigma=0.5, beta=0.5)
    dims = size(A)
    m = ndims(A)
    I_dims = collect(dims)
    N0 = prod(dims)
    Nd = sum(I_dims) * R


    Random.seed!(42)
    u = [rand(R, I_dims[j]) for j in 1:m]
    u_vec = vcat([vec(u[j]) for j in 1:m]...)

    k = 0
    mu = tau
    nu = 2.0 
    found = false
    F_prev = Inf

    while !found && k < max_iter
        k += 1

        f_val = f(u_vec, I_dims,m)
        J_val = jacobian(u_vec, N0, Nd, dims, I_dims,m)
        g = J_val' * f_val          # Gradient
        A_mat = J_val' * J_val      # Hessian
        F_u = 0.5 * dot(f_val, f_val) 

        if maximum(abs.(g)) <= eps1
            println("SMALL G")
            found = true
            break
        end

        Iden = Matrix{Float64}(I, Nd, Nd)
        h = - (A_mat + mu * Iden) \ g

        if norm(h) <= eps2 * (norm(u_vec) + eps2)
            println("STOP")
            found = true
            break
        end

        u_new = u_vec + h
        F_new = 0.5 * dot(f(u_new, I_dims, m), f(u_new, I_dims, m))
        L_diff = 0.5 * dot(h, mu * h - g)

        q = 0
        if L_diff != 0
            q = (F_u - F_new) / L_diff
        end

        if q > 0
            mu *= max(1 / 3, 1 - (2 * q - 1)^3)
            nu = 2.0

            armiho_iter = 0
            while true
                alpha = beta^armiho_iter
                u_armiho = u_vec + alpha * h
                val = f(u_armiho, I_dims, m)
                F_armiho = 0.5 * dot(val, val)

                # Armiho condition
                if F_armiho <= F_u + sigma * alpha * dot(h, g)
                    u_new = u_armiho
                    break
                end

                armiho_iter += 1
                if armiho_iter > 100
                    println("BRUH :(")
                    break
                end
            end

        else
            mu *= nu
            nu *= eta
            u_new = u_vec
        end

        u_vec = u_new
        F_prev = F_u
    end

    A_approx = compute_A_approx(u_vec, I_dims, m)
    println("iterations = ", k)
    return A_approx, u_vec
end

A = zeros(5, 5, 5)
for i in 1:5
    for j in 1:5
        for k in 1:5
            A[i, j, k] = sin(i + j + k)
        end
    end
end
R = 3

A_approx, u_vec = lev_mark(A, R)

println("CNORM: ", maximum(abs.(A - A_approx)))
