using AMD
using SparseArrays

A = sparse([
    1.0 1 0 0;
    1   1 1 0;
    0   1 1 1;
    0   0 1 1
])

p = AMD.amd(A)

println(p)