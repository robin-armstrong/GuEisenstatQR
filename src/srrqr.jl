"""
	srrqr(M; f = 2.0, tol = 1e-12, kmax = minimum(size(M)))

Compute a strong rank-revealing QR factorization of `M` using the algorithm of Gu
and Eisenstat (1996). In other words, compute a permutation `perm`, an orthogonal
matrix `Q`, and a matrix `R = [A B; 0 C]` where `A` is square and upper-triangular
with strictly positive diagonal entries, such that `M[:, perm]` is equal to `Q*R`.

The factorization satisfies the singular value inequalities
```
\$\\sigma_i(A) \\geq \\frac{ \\sigma_i(M) }{ \\sqrt{1 + f^2 k(n - k)} }\$
```

for ```\$1 \\leq i \\leq k\$```, where `k` is the dimension of `A` and `n` is the
number of columns in `M`, as well as
```
\$\\sigma_j(C) \\leq \\sigma_{k + j}(M) \\sqrt{1 + f^2 k(n - k)}\$
```

for ```\$1 \\leq j \\leq n - k\$```. In addition, the elements of interpolation 
matrix `inv(A)*B` are bounded in magnitude by `f`.

The algorithm chooses `k` large enough that the maximum squared column norm of `C` 
is at most `tol` times the maximum squared column norm of `M`, subject to the
restriction `k <= kmax`. If `kmax == minimum(size(M))`, then `k` serves as an
estimate for the numerical rank of `M`. Setting `kmax < minimum(size(M))` restricts
the number of column pivots that the algorithm computes, thus shortening the 
computation time.
"""
function srrqr(M::Matrix{T}; f::Real = 2.0, tol::Real = 1e-12, kmax::Integer = minimum(size(M))) where {T <: Real}
	L = minimum(size(M))
	
	if(f < 1)
		throw(ArgumentError("f must be at least 1.0"))
	elseif(tol <= 0)
		throw(ArgumentError("tol must be positive"))
	elseif(L <= 1)
		throw(ArgumentError("matrix must have at least 2 rows and at least 2 columns"))
	elseif((kmax < 1) || (kmax > L))
		throw(ArgumentError("kmax must be at least 1 and at most the smaller dimension of the matrix"))
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
	
	delta = tol*maxnorm^2
	
	# initializing the factorization
	
	perm = Vector{Int64}(range(1, size(M, 2), size(M, 2)))
	perm[1] = jmax
	perm[jmax] = 1
	
	v = zeros(size(M, 1))	# representation of Q as a Householder reflector
	s = M[1, jmax] == 0. ? 1. : sign(M[1, jmax])
	v[1] = s*norm(M[:, jmax])
	v = v + M[:, jmax]
	v /= sqrt(v'*v)
	
	Q = Matrix{Float64}(I(size(M, 1))) - 2*v*v'
	
	A = Matrix{Float64}(undef, 1, 1)
	A[1, 1] = -s*maxnorm
	
	QtM = M - 2*v*(v'*M)
	
	B = QtM[1:1, 2:end]
	C = QtM[2:end, 2:end]
	
	if(jmax > 1)
		B[1, jmax - 1] = QtM[1, 1]
		C[:, jmax - 1] = QtM[2:end, 1]
	end
	
	# initializing other data that the algorithm uses
	
	AinvB = B/A[1, 1]
	
	gamma = Vector{Float64}(undef, size(C, 2))
	for i = 1:length(gamma)
		gamma[i] = norm(C[:, i])^2
	end
	
	omega = Vector{Float64}(undef, 1)
	omega[1] = 1/A[1, 1]^2
	
	# this loop computes the SRRQR
	
	k = 1
	while(k < L)
		# this loop makes column pivots to compute a strong rank-revealing QR for the current rank estimate
		
		while(true)
			ilarge = 0
			jlarge = 0
			foundLargeEntry = false
			rhomax = f^2
			
			for i = 1:size(A, 1)				
				for j = 1:size(B, 2)
					rho = AinvB[i, j]^2 + gamma[j]*omega[i]
					
					if(rho >= rhomax)
						ilarge = i
						jlarge = j
						foundLargeEntry = true
						rhomax = rho
					end
				end
			end
			
			foundLargeEntry || break
			updateFactors!(ilarge, jlarge, Q, A, B, C, perm, AinvB, gamma, omega)
		end
		
		# respecting user bounds on the size of A
		if(k == kmax)
			break
		end
		
		# looking for a large column in C to bring into the basis for a rank update
		
		maxgamma = delta
		jmax = 0
		
		for j = 1:length(gamma)
			if(gamma[j] > maxgamma)
				maxgamma = gamma[j]
				jmax = j
			end
		end
		
		# if we found a large column then we pivot it into the basis, update the rank estimate, and continue
		
		if(jmax == 0)
			break
		else
			A, B, C, AinvB = updateRank!(jmax, Q, A, B, C, perm, AinvB, gamma, omega)
			k += 1
		end
	end
	
	R = [A B; zeros(size(C, 1), size(A, 2)) C]
	
	# ensuring that A has strictly positive diagonal entries
	for i = 1:k
		s = sign(R[i, i])
		R[i, i:end] *= s
		Q[:, i] *= s
	end
	
	return SRRQR(k, perm, Q, R)
end
