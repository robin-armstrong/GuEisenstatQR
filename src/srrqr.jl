"""
	srrqr(M; f = 2.0, tol = 1e-12, numrank = nothing)

Compute a strong rank-revealing QR factorization of `M` using the algorithm of Gu
and Eisenstat (1996). In other words, compute a permutation `perm`, an orthogonal
matrix `Q`, and a matrix `R = [A B; 0 C]` where `A` is square and upper-triangular
with strictly positive diagonal entries, such that `M[:, perm]` is equal to `Q*R`.

The factorization satisfies the inequalities
```
\$\\sigma_i(A) \\geq \\frac{ \\sigma_i(M) }{ \\sqrt{1 + f^2 k(n - k)} }\$
```

for ```\$1 \\leq i \\leq k\$```, where `k` is the dimension of `A` and `n` is the number of
columns in `M`, as well as
```
\$\\sigma_j(C) \\leq \\sigma_{k + j}(M) \\sqrt{1 + f^2 k(n - k)}\$
```

for ```\$1 \\leq j \\leq n - k\$```. If a value for `numrank` is provided, then `k` is set
equal to that value. Otherwise, `k` is chosen so that the maximum column norm of `C`
is at most `tol` times the maximum column norm of `M`.
"""
function srrqr(M::Matrix{T}; f::Real = 2.0, tol::Real = 1e-12, numrank::Union{Integer, Nothing} = nothing) where {T <: Real}
	if(f < 1)
		throw(ArgumentError("f must be at least 1.0"))
	elseif(tol <= 0)
		throw(ArgumentError("tol must be positive"))
	elseif(min(size(M, 1), size(M, 2)) <= 1)
		throw(ArgumentError("matrix must have at least 2 rows and at least 2 columns"))
	elseif(numrank != nothing)
		if((numrank < 1) || (numrank > min(size(M, 1), size(M, 2))))
			throw(ArgumentError("numrank must be at least 1 and at most the smaller dimension of the matrix"))
		end
	end
	
	rankspecified = numrank != nothing
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
	
	QtM = M - 2*v*(v'*M)
	
	B = QtM[1:1, 2:end]
	C = QtM[2:end, 2:end]
	
	if(jmax > 1)
		B[1, jmax - 1] = QtM[1, 1]
		C[:, jmax - 1] = QtM[2:end, 1]
	end
	
	# initializing other data that the algorithm uses
	
	AinvB = B/maxnorm
	
	gamma = Vector{Float64}(undef, size(C, 2))
	for i = 1:length(gamma)
		gamma[i] = norm(C[:, i])
	end
	
	omega = Vector{Float64}(undef, 1)
	omega[1] = abs(A[1, 1])
	
	if(rankspecified)
		for i = 2:min(numrank, size(M, 1) - 1, size(M, 2) - 1)
			# finding the largest column of C to bring into the basis
			
			maxnorm = 0.
			jmax = 0
			
			for j = 1:size(C, 2)
				nrm = norm(C[:, j])
				if(nrm > maxnorm)
					maxnorm = nrm
					jmax = j
				end
			end
			
			# pivoting that column into the basis and updating the factorization
			
			A, B, C, AinvB = updateRank!(jmax, Q, A, B, C, perm, AinvB, gamma, omega)
		end
	else
		numrank = 1
	end
	
	if(numrank == min(size(M, 1), size(M, 2)))
		R = [A B; zeros(size(C, 1), size(A, 2)) C]
		return SRRQR(numrank, perm, Q, R)
	end
	
	while(numrank < min(numrank, size(M, 1) - 1, size(M, 2) - 1))
		# making column pivots to compute a strong rank-revealing QR at the current rank
		
		while(true)
			ilarge = 0
			jlarge = 0
			foundLargeEntry = false
			
			for i = 1:size(A, 1)
				!foundLargeEntry || break
				
				for j = 1:size(B, 2)
					rho = AinvB[i, j]^2 + (gamma[j]/omega[i])^2
					
					if(rho > f^2)
						ilarge = i
						jlarge = j
						foundLargeEntry = true
					end
				end
			end
			
			foundLargeEntry || break
			updateFactors!(ilarge, jlarge, A, B, C, Q, perm, AinvB, gamma, omega)
		end
		
		!rankspecified || break
		
		# looking for a large column in C
		
		maxgamma = delta
		jmax = 0
		
		for j = 1:length(gamma)
			if(gamma[j] > maxgamma)
				maxgamma = gamma[j]
				jmax = j
			end
		end
		
		# if we found a large column then we pivot it into the basis, update the rank, and continue
		
		if(jmax == 0)
			break
		else
			A, B, C, AinvB = updateRank!(jmax, Q, A, B, C, perm, AinvB, gamma, omega)
			numrank += 1
		end
	end
	
	# we make a final update to the rank, if necessary
	if(numrank == min(numrank, size(M, 1) - 1, size(M, 2) - 1))
		numrank += (gamma[1] > delta ? 1 : 0)
	end
	
	# ensuring that A has strictly positive diagonal entries
	for i = 1:size(A, 1)
		s = sign(A[i, i])
		A[i, i:end] *= s
		Q[:, i] *= s
	end
	
	# returning the factorization
	R = [A B; zeros(size(C, 1), size(A, 2)) C]
	return SRRQR(numrank, perm, Q, R)
end
