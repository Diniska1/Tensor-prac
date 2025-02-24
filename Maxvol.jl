using LinearAlgebra

function print_matrix(A)
    for i in 1:(size(A, 1))
        println(A[i, :])
    end
    println()
end

function maxvol(T,r)

    F = qr(T, Val(true)) # QR_Pivoting
    P = F.p

    matrix = T[:, P]

    C = matrix[:, 1:r]
    A = C[1:r, :]
    inv_A = inv(A)
    global D = C * inv_A

    global ind_max_elem = argmax(abs.(D))
    res_columns = collect(1:r)

    global iter = 0
    while ind_max_elem[1] >= r && iter < r * r
        global iter += 1

        # Formulas got from article
        # "How to find a good submatrix"

        row_max_elem = copy(D[ind_max_elem[1], :])
        row_max_elem[ind_max_elem[2]] -= 1
        row_max_elem /= D[ind_max_elem]

        col_max_elem = copy(D[:, ind_max_elem[2]])
        col_max_elem[ind_max_elem[2]] -= 1
        col_max_elem[ind_max_elem[1]] += 1

        global D -= col_max_elem * row_max_elem'

        res_columns[ind_max_elem[2]] = ind_max_elem[1]

        global ind_max_elem = argmax(abs.(D))
    end
    return res_columns, P
end

r = 2
T = [1 2 3;
    4 5 6;
    7 8 7]

res_columns, P = maxvol(T, 2)
println("res_columns:\n", res_columns)
res = (T[:, P])[res_columns, res_columns]
println("res")
print_matrix(res)



