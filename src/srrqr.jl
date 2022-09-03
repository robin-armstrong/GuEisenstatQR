"""
	srrqr(M; f = 2.0, tol = 1e-12)

Compute a strong rank-revealing QR factorization of `M` using the algorithm of Gu
and Eisenstat (1996) with parameter `f`. Return `k` (the numerical rank of `M`),
a column permutation `perm`, an orthogonal matrix `Q`, and `R` such that `M[:, perm]`
is equal to `Q*R`.

The `R` factor has the block form `R = [A B; 0 C]` where `A` is a `k` x `k`
upper-triangular matrix with strictly positive diagonal entries, and where the
column-norms of `C` are bounded by `tol*maxnorm`, with `maxnorm` being the maximum
column-norm of `M`. The entries of `inv(A)*B` are bounded in magnitude by `f`.
"""
function srrqr(M::Matrix{T}; f::Real = 2.0, tol::Real = 1e-12) where {T <: Real}
	if(tol <= 0)
		throw(ArgumentError("the tolerance must be positive"))
	elseif(f < 1)
		throw(ArgumentError("f must be at least 1.0"))
	elseif(min(size(M, 1), size(M, 2)) <= 1)
		throw(ArgumentError("matrix must have at least 2 rows and at least 2 columns"))
	end
	
	M = Matrix{Float64}(M)
	
	# finding the largest column of M
	
	maxnorm = 0.
	jmax = 0
	
	for j = 1:size(M, 2)
		nrm = norm(M[:, j])
		if(nrm > maxnorm)
			maxnorm = nrm
			jmax = j
		end
	end
	
	# normalizing the tolerance to the size of the matrix
	
	delta = tol*maxnorm
	
	# initializing the factorization
	
	perm = Vector{Int64}(range(1, size(M, 2), size(M, 2)))
	perm[1] = jmax
	perm[jmax] = 1
	
	v = zeros(size(M, 1))	# representation of Q as a Householder reflector
	v[1] = sign(M[1, jmax])*norm(M[:, jmax])
	v = v + M[:, jmax]
	v /= sqrt(v'*v)
	
	Q = Matrix{Float64}(I(size(M, 1))) - 2*v*v'
	
	A = Matrix{Float64}(undef, 1, 1)
	A[1, 1] = -sign(M[1, jmax])*maxnorm
	
	QM = M - 2*v*(v'*M)
	
	B = QM[1:1, 2:end]
	C = QM[2:end, 2:end]
	
	if(jmax > 1)
		B[1, jmax - 1] = QM[1, 1]
		C[:, jmax - 1] = QM[2:end, 1]
	end
	
	# initializing other data that the algorithm uses
	
	AinvB = B/maxnorm
	
	gamma = Vector{Float64}(undef, size(C, 2))
	for i = 1:length(gamma)
		gamma[i] = norm(C[:, i])
	end
	
	omega = Vector{Float64}(undef, 1)
	omega[1] = abs(A[1, 1])
	
	firstIteration = true
	k = 1		# initially the factorization is for numerical rank 1
	
	# greedy algorithm to compute the numerical rank and an srrqr
	while(true)
		if(firstIteration)
			firstIteration = false
			
		elseif(k == min(size(M, 1), size(M, 2)) - 1)
			if(abs(C[1, 1]) > delta)
				v = zeros(size(C, 1))	# constructing a Householder reflector
				v[1] = sign(C[1, 1])*norm(C[:, 1])
				v = v + C[:, 1]
				v /= sqrt(v'*v)
				
				C[:, :] = C - 2*v*(v'*C)
				Q[:, k + 1:end] = Q[:, k + 1:end] - 2*(Q[:, k + 1:end]*v)*v'
				C[2:end, 1] = zeros(size(C, 1) - 1)
				
				k += 1
			end
			
			break
			
		else
			# looking for the next column to bring into the basis
			maxgamma = delta
			jmax = 0
			
			for j = 1:length(gamma)
				if(gamma[j] > maxgamma)
					maxgamma = gamma[j]
					jmax = j
				end
			end
			
			if(jmax == 0)
				break		# if we have not found a large column, then the factorization is complete
			else
				# increasing the numerical rank and continuing the factorization
				A, B, C, AinvB = updateRank!(jmax, Q, A, B, C, perm, AinvB, gamma, omega)
				k += 1
			end
		end
		
		# computing a strong rank-revealing QR factorization for the current numerical rank
		while(true)
			foundLargeEntry = false
			ilarge = 0
			jlarge = 0
			
			for i = 1:size(AinvB, 1)
				if(foundLargeEntry)
					break
				end
				
				for j = 1:size(AinvB, 2)
					rho = sqrt(AinvB[i, j]^2 + (gamma[j]/omega[i])^2)
					
					if(rho > f)
						foundLargeEntry = true
						ilarge = i
						jlarge = j
					end
					
					if(foundLargeEntry)
						break
					end
				end
			end
			
			if(!foundLargeEntry)
				break
			end
			
			updateFactors!(ilarge, jlarge, A, B, C, Q, perm, AinvB, gamma, omega)
		end
	end
	
	R = [A B; zeros(size(C, 1), size(A, 2)) C]
	
	# ensuring that the diagonal entries of A are positive
	
	for i = 1:k
		s = sign(R[i, i])
		R[i, 1:end] *= s
		Q[:, i] *= s
	end
	
	return SRRQR(k, perm, Q, R)
end

function srrqrRank(M::Matrix{F}, k::Integer ;
					f::AbstractFloat = 2., tol::AbstractFloat = 1e-12) where {F <: AbstractFloat}
	m, n = size(M)
	
	if((k < 1) || (k > min(m, n)))
		throw(ArgumentError("the target rank must be at least 1 and at most "*string(min(m, n))))
	elseif(f < 1)
		throw(ArgumentError("the third argument must be at least 1.0"))
	end
end
