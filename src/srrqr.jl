function srrqr(M::Matrix{F};
				   f::AbstractFloat = 2., tol::AbstractFloat = 1e-12) where {F <: AbstractFloat}
	if(tol <= 0)
		throw(ArgumentError("the second argument must be positive"))
	elseif(f < 1)
		throw(ArgumentError("the third argument must be at least 1.0"))
	end
	
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
	
	Q = Matrix{F}(I(size(M, 1))) - 2*v*v'
	
	A = Matrix{F}(undef, 1, 1)
	A[1, 1] = -sign(M[1, jmax])*maxnorm
	
	QM = M - 2*v*(v'*M)
	
	B = QM[1:1, 2:end]
	B[1, jmax - 1] = QM[1, 1]
	
	C = QM[2:end, 2:end]
	C[:, jmax - 1] = QM[2:end, 1]
	
	# initializing other data that the altorithm uses
	
	AinvB = B/maxnorm
	
	gamma = Vector{F}(undef, size(C, 2))
	for i = 1:length(gamma)
		gamma[i] = norm(C[:, i])
	end
	
	omega = Vector{Float64}(undef, 1)
	omega[1] = abs(A[1, 1])
	
	# greedy algorithm to compute the srrqr
	
	firstIteration = true
	foundLargeColumn = true
	k = 1
	
	while(true)
		if(firstIteration)
			firstIteration = false
		else
			foundLargeColumn = false
			maxgamma = 0.
			jmax = 0
			
			for j = 1:length(gamma)
				if(gamma[j] > maxgamma)
					foundLargeColumn = true
					maxgamma = gamma[j]
					jmax = j
				end
			end
		end
		
		if(!foundLargeColumn)
			break
		end
		
		# updating the rank of the factorization, using the large column at index jmax as a pivot
		# BUG: this run of updateRank! sometimes breaks, see line 25 of updateUtilities.jl
		A, B, C, AinvB = updateRank!(jmax, Q, A, B, C, perm, AinvB, gamma, omega)
		k += 1
		
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
	
	return Q, [A B; zeros(size(C, 1), size(A, 2)) C], perm, k
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
