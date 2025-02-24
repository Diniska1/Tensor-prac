using LinearAlgebra

function get_ek(k, size)
    e = zeros(size)
    e[k] = 1
    return e
end

function print_matrix(A, rows)
    for i in 1:rows
        println(A[i, :])
    end
    println()
end

function cnorm(A)
    rows = size(A, 1)
    cols = size(A, 2)
    m = 0
    for i in 1:rows
        for j in 1:cols
            if m < abs(A[i,j])
                m = abs(A[i,j])
            end
        end
    end
    return m
end

function find_max_col(R, j)
    rows = size(R, 1)
    cols = size(R, 2)
    m = 0
    ind = 0

    for i in j:cols
        x = R[i:rows, i]
        norm = x' * x
        if norm > m
            m = norm
            ind = i
        end
    end
    return ind
end


A = [1 3 2;
    4 6 5;
    7 9 7]
A = [12 -51 4;
     6 167 -68;
     -4 24 -41]
rows = size(A, 1)
cols = size(A, 2)


Q = zeros(rows, rows)
for i in 1:rows
    Q[i, i] = 1.0
end
R = copy(A)
R = float(R)
P = Matrix(I, rows, rows)

for j in 1:rows-1
    ind = find_max_col(R, j)                # find index of column with max norm
    R[:, j], R[:, ind] = R[:, ind], R[:, j] # swap
    P[:, j], P[:, ind] = P[:, ind], P[:, j]

    # HOUSEHOLDER REFLECTION
    x = R[j:end, j]
    a = sqrt(x' * x)
    u = x - a * get_ek(1, rows - j + 1)
    v = u / sqrt(u' * u)

    H = zeros(rows, rows)
    H[1:j-1, 1:j-1] = float(Matrix(I, j-1, j-1))
    E = Matrix(I, rows - j + 1, rows - j + 1)
    H[j:rows, j:rows] = E - 2 * v * v'

    global R = H * R
    global Q = Q * H

end
println("Q")
print_matrix(Q, rows)
println("R")
print_matrix(R, rows)

println("RES")
print_matrix(Q * R, rows)

println("C_NORM(AP - QR) = ", cnorm(A * P - Q * R))

