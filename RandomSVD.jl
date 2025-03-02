using LinearAlgebra

function RSVD(A::Matrix, r)
    m = size(A, 1) # rows
    n = size(A, 2) # cols

    rand_matrix = rand(m, r)
    Q, R = qr(A * rand_matrix)
    Q = Matrix(Q)

    F = svd(Q' * A, full = false)

    return Q * F.U, F.S, F.V
end

r = 2
A = [1 2 3;
    4 5 6;
    7 8 9]

U, S, V = RSVD(A, r)

println("C_NORM of A - U * S * V* is ", maximum(abs.(A - U * Diagonal(S) * V')))


