using LinearAlgebra

"""
	updateRank!(j, Q, A, B, C, perm, AinvB, gamma, omega)

Function to update the pivoted QR factorization `M[:, perm] = Q*R`, where `R` has the
block form `[A B; 0 C]` and `A` is a square upper-triangular matrix. Requires `gamma`,
the vector of squared column-norms of `C`, and `omega`, the vector of squared row-norms of 
`inv(A)`. Permutes column `k + j - 1` to position `k`, where `k - 1` is the dimension of
`A`, updating `perm` accordingly. Then computes `M[:, perm] = Q_new*[A_new B_new; 0 C_new]`,
where `A_new` is square and upper-triangular with order `k`, as well as `AinvB_new = inv(A_new)*B_new`,
and updates (and resizes) `gamma` and `omega`. Returns `A_new`, `B_new`, `C_new`, and
`AinvB_new`, while `Q`, `perm`, `gamma`, and `omega` are modified in place.
"""
function updateRank!(j::Int64,
			Q::Matrix{Float64}, A::Matrix{Float64}, B::Matrix{Float64}, C::Matrix{Float64}, perm::Vector{Int64},
			AinvB::Matrix{Float64}, gamma::Vector{Float64}, omega::Vector{Float64})
	
	k = size(A, 1) + 1
	
	# putting the (k + j - 1)th column in kth position
	
	if(j > 1)
		tmp = perm[k]
		perm[k] = perm[k + j - 1]
		perm[k + j - 1] = tmp
		
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
	end
	
	# creating zeros in the [2:end, 1] block of C
	
	v = zeros(size(C, 1))	# constructing a Householder reflector
	v[1] = sign(C[1, 1])*norm(C[:, 1])
	v = v + C[:, 1]
	v /= sqrt(v'*v)
	
	C[:, :] = C - 2*v*(v'*C)
	Q[:, k:end] = Q[:, k:end] - 2*(Q[:, k:end]*v)*v'
	C[2:end, 1] = zeros(size(C, 1) - 1)
	
	# computing A_new, B_new, C_new, and AinvB_new
	
	g = C[1, 1]
	b = B[:, 1]
	c = C[1, 2:end]
	u = A \ b
	
	A_new = [A b; zeros(1, size(A, 2)) g]
	B_new = [B[:, 2:end]; c']
	C_new = C[2:end, 2:end]
	AinvB_new = [AinvB[:, 2:end] - u*c'/g; c'/g]
	
	# updating omega
	
	append!(omega, 1/g^2)
	
	for r = 1:k - 1
		omega[r] += u[r]^2/g^2
	end
	
	# updating gamma
	
	for r = 2:length(gamma)
		gamma[r] = gamma[r] - c[r - 1]^2
		
		if(gamma[r] < 0)
			@warn "gamma["*string(r)*"] is negative.\ngamma["*string(r)*"] = "*string(gamma[r])
		end
	end
	
	deleteat!(gamma, 1)
	
	return A_new, B_new, C_new, AinvB_new
end

"""
	updateFactors!(i, j, A, B, C, Q, perm, AinvB, gamma, omega)
	
Function to update the pivoted QR factorization `M[:, perm] = Q*R`, where `R` has the
block form `[A B; 0 C]` and `A` is a square upper-triangular matrix. Permutes column
`i` to position `k + 1` and column `k + j` to position `k`, where `k` is the dimension
of `A`. Then updates the orthogonal factor `Q`, the triangular factor `R`, the column
permutation `perm`, the matrix `AinvB = inv(A)*B`, the vector `gamma` containing the
squared column-norms of `C`, and the vector `omega` containing the squared row-norms of
`inv(A)`. Modifies its arguments and has no return value.
"""
function updateFactors!(i::Int64, j::Int64,
						Q::Matrix{Float64}, A::Matrix{Float64}, B::Matrix{Float64}, C::Matrix{Float64}, perm::Vector{Int64},
						AinvB::Matrix{Float64}, gamma::Vector{Float64}, omega::Vector{Float64})
	
	k = size(A, 1)
	
	# putting the (k + j)th column in (k + 1)st position
	if(j > 1)
		tmp = perm[k + 1]
		perm[k + 1] = perm[k + j]
		perm[k + j] = tmp
		
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
	end
	
	# putting the i^th column in k^th position
	if(i < k)
		tmp = perm[i]
		perm[i:k - 1] = perm[i + 1:k]
		perm[k] = tmp
		
		tmp = A[:, i]
		A[:, i:k - 1] = A[:, i + 1:k]
		A[:, k] = tmp
		
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
			A[r + 1, r] = 0.
			B[r:r + 1, :] = givens*B[r:r + 1, :]
			Q[:, r:r + 1] = Q[:, r:r + 1]*givens'
		end
	end
	
	# introducing zeros in the [2:end, 1] block of C
	
	v = zeros(size(C, 1))	# constructing a Householder reflector
	v[1] = sign(C[1, 1])*norm(C[:, 1])
	v = v + C[:, 1]
	v /= sqrt(v'*v)
	
	C[:, :] = C - 2*v*(v'*C)
	Q[:, k + 1:end] = Q[:, k + 1:end] - 2*(Q[:, k + 1:end]*v)*v'
	
	# swapping columns k and k + 1
	
	tmp = perm[k]
	perm[k] = perm[k + 1]
	perm[k + 1] = tmp
	
	# making final updates to the factorization
	
	b1 = A[1:k - 1, k]
	b2 = B[1:k - 1, 1]
	g = A[k, k]
	mu = B[k, 1]/g
	nu = C[1, 1]/g
	rho = sqrt(mu^2 + nu^2)
	gBar = g*rho
	c1 = B[k, 2:end]
	c2 = C[1, 2:end]
	c1Bar = (mu*c1 + nu*c2)/rho
	c2Bar = (nu*c1 - mu*c2)/rho
	u = A[1:k - 1, 1:k - 1] \ b1
	u1 = AinvB[1:k - 1, 1]
	
	# updating Q
	
	Q[:, k:k + 1] = Q[:, k:k + 1]*[mu/rho nu/rho; nu/rho -mu/rho]
	
	# updating A
	A[1:k - 1, k] = b2
	A[k, k] = gBar
	
	# updating B
	B[1:k - 1, 1] = b1
	B[k, 1] /= rho
	B[k, 2:end] = c1Bar
	
	# updating C
	C[1, 1] /= rho
	C[1, 2:end] = c2Bar
	
	# updating AinvB
	AinvB[1:k - 1, 1] = (nu^2*u - mu*u1)/rho^2
	AinvB[1:k - 1, 2:end] += (nu*u*c2Bar' - u1*c1Bar')/gBar
	AinvB[k, 1] /= rho^2
	AinvB[k, 2:end] = c1Bar'/gBar
	
	# updating gamma
	gamma[1] = C[1, 1]^2
	for r = 2:length(gamma)
		gamma[r] = gamma[r] + c2Bar[r - 1]^2 - c2[r - 1]^2
		
		if(gamma[r] < 0)
			@warn "gamma["*string(r)*"] is negative.\ngamma["*string(r)*"] = "*string(gamma[r])
		end
	end
	
	# updating omega
	omega[k] = 1/gBar^2
	for r = 1:k - 1
		omega[r] += (u1[r] + mu*u[r])^2/gBar^2 - u[r]^2/g^2
		
		if(omega[r] < 0)
			@warn "omega["*string(r)*"] is negative.\nomega["*string(r)*"] = "*string(omega[r])
		end
	end
end
