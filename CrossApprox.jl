
function cross(A, r, eps)
    n = size(A, 1) # rows
    m = size(A, 2) # cols

    U = float(zeros(n, r))
    V = float(zeros(r, m))


    columns = collect(1:m)
    i = 1
    max_el = eps + 1

    while i <= r && max_el * sqrt((n - i) * (m - i)) > eps
        ind = rand(1:(m - i))
        col = columns[ind]
        
        row = argmax(abs.(A[:, col] - (U * V)[:, col]))
        col = argmax(abs.(A[row, :] - (U * V)[row, :]))


        col_diff = A[:, col] - (U * V)[:, col]
        row = argmax(abs.(col_diff))
        row_diff = A[row, :] - (U * V)[row, :]
        max_el = col_diff[row]

        U[:, i] = col_diff / sqrt(abs.(max_el))
        V[i, :] = row_diff * sqrt(abs.(max_el)) / max_el

        max_el = abs(max_el)

        for k in 1:(m - i)
            if columns[k] == col
                global ind = col
            end
        end
        deleteat!(columns, ind)
        i += 1
    end

    return U, V, i
end

A = [0 1 2;
    3 4 5;
    6 7 8;] # Matrix rank = 2
r = 2
eps = 0.01

U, V, rank = cross(A, r, eps)

println("====== RES =======")

println("C_NORM of U * V - A is ", maximum(abs.(U * V - A)))


