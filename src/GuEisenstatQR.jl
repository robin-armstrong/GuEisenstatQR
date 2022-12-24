"""
This package provides `srrqr`, a strong rank-revealing QR factorization using the algorithm of Gu and Eisenstat (1996).
"""
module GuEisenstatQR

export srrqr, SRRQR

include("returnstructs.jl")
include("updateUtilities.jl")
include("srrqr.jl")

end
