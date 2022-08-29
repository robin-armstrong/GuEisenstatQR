using LinearAlgebra

"""
	updateFactors!(i, j, k, A, B, C, Q, perm, AinvB, gamma, omega)
	
Function to update a pivoted QR factorization in the Gu-Eisenstat algorithm. Permutes column `i`
to position `k + 1` and column `k + j` to position `k`. Then updates the orthogonal factor `Q` and the 
triangular factor `R`, which has the block form `R = [A B; 0 C]`, where `A` is a square upper-triangular
matrix of order `k`. Also updates the column permutation `perm`, the matrix `AinvB = inv(A)*B`,
the vector `gamma` containing the column-norms of `C`, and the vector `omega` containing the inverted
row-norms of `inv(A)`. Modifies its arguments and has no return value.
"""
function updateFactors!(i::Integer, j::Integer, k::Integer,
						Q::Matrix{F}, A::Matrix{F}, B::Matrix{F}, C::Matrix{F}, perm::Vector{Itg},
						AinvB::Matrix{F}, gamma::Vector{F}, omega::Vector{F}) where {Itg <: Integer, F <: AbstractFloat}	
	
	# putting the (k + j)th column in (k + 1)st position
	if(j > 1)
		tmp = B[:, 1]
		B[:, 1] = B[:, j]
		B[:, j] = tmp
		
		tmp = C[:, 1]
		C[:, 1] = C[:, j]
		C[:, j] = tmp
		
		tmp = AinvB[:, 1]
		AinvB[:, 1] = AinvB[:, j]
		AinvB[:, j] = tmp
		
		tmp = gamma[1]
		gamma[1] = gamma[j]
		gamma[j] = tmp
		
		tmp = perm[k + 1]
		perm[k + 1] = perm[k + j]
		perm[k + j] = tmp
	end
	
	# putting the i^th column in k^th position
	if(i < k)
		tmp = A[:, i]
		A[:, i:k - 1] = A[:, i + 1:k]
		A[:, k] = tmp
		
		tmp = perm[i]
		perm[i:k - 1] = perm[i + 1:k]
		perm[k] = tmp
		
		tmp = omega[i]
		omega[i:k - 1] = omega[i + 1:k]
		omega[k] = tmp
		
		tmp = AinvB[i, :]
		AinvB[i:k - 1, :] = AinvB[i + 1:k, :]
		AinvB[k, :] = tmp
		
		# re-triangularizing A using Givens rotations
		for r = i:k - 1
			rho = sqrt(A[r, r]^2 + A[r + 1, r]^2)
			givens = [A[r, r]/rho A[r + 1, r]/rho; -A[r + 1, r]/rho A[r, r]/rho]
			A[r:r + 1, :] = givens*A[r:r + 1, :]
			B[r:r + 1, :] = givens*B[r:r + 1, :]
			Q[:, r:r + 1] = Q[:, r:r + 1]*givens'
		end
	end
end

function ssqrRank(M::Matrix, k::Integer, f::AbstractFloat)

end

function ssqrTol(M::Matrix, tol::AbstractFloat, f::AbstractFloat)

end
