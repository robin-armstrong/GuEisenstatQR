using GuEisenstatQR
using LinearAlgebra
using Random
using Test

# function to print debug message for a failed test
function showInfo(msg, testResult)
	if(typeof(testResult) != Test.Pass)
		@info msg
	end
end

@testset "test updateRank!" begin
	m = 150
	n = 150
	k = 75
	epsilon = 1e-12
	
	Q_init = Matrix(qr(randn(m, m)).Q)
	R_init = qr(randn(m, n)).R
	perm_init = randperm(n)
	M = Matrix{Float64}(undef, m, n)
	M[:, perm_init] = Q_init*R_init
	
	A_init = R_init[1:k, 1:k]
	B_init = R_init[1:k, k + 1:n]
	C_init = R_init[k + 1:m, k + 1:n]
	
	Ainv_init = Matrix{Float64}(undef, k, k)
	for i = 1:k
		ei = zeros(k)
		ei[i] = 1.
		Ainv_init[:, i] = A_init \ ei
	end
	
	AinvB_init = Ainv_init*B_init
	
	gamma_init = Vector{Float64}(undef, n - k)
	for r = 1:n - k
		gamma_init[r] = norm(C_init[:, r])^2
	end
	
	omega_init = Vector{Float64}(undef, k)
	for r = 1:k
		omega_init[r] = norm(Ainv_init[r, :])^2
	end
	
	for j in [1, 3, n - k]
		params_str = "parameters are: j = "*string(j)
		
		A = deepcopy(A_init)
		B = deepcopy(B_init)
		C = deepcopy(C_init)
		Q = deepcopy(Q_init)
		perm = deepcopy(perm_init)
		AinvB = deepcopy(AinvB_init)
		gamma = deepcopy(gamma_init)
		omega = deepcopy(omega_init)
		
		A, B, C, AinvB = GuEisenstatQR.updateRank!(j, Q, A, B, C, perm, AinvB, gamma, omega)
		
		showInfo(params_str, @test size(A) == (size(A_init, 1) + 1, size(A_init, 2) + 1))
		showInfo(params_str, @test size(B) == (size(B_init, 1) + 1, size(B_init, 2) - 1))
		showInfo(params_str, @test size(C) == (size(C_init, 1) - 1, size(C_init, 2) - 1))
		
		R = [A B; zeros(size(C, 1), size(A, 2)) C]
		
		showInfo(params_str, @test norm(M[:, perm] - Q*R)/norm(M) < epsilon)
		showInfo(params_str, @test norm(tril(A, -1)) < epsilon)
		showInfo(params_str, @test norm(Q'*Q - I(m))/m < epsilon)
		showInfo(params_str, @test perm[k + j] == perm_init[k + 1])
		showInfo(params_str, @test perm[k + 1] == perm_init[k + j])
		
		Ainv_target = Matrix{Float64}(undef, k + 1, k + 1)
		for i = 1:k + 1
			ei = zeros(k + 1)
			ei[i] = 1.
			Ainv_target[:, i] = A \ ei
		end
		
		AinvB_target = Ainv_target*B
		
		gamma_target = Vector{Float64}(undef, size(C, 2))
		for r = 1:size(C, 2)
			gamma_target[r] = norm(C[:, r])^2
		end

		omega_target = Vector{Float64}(undef, size(A, 1))
		for r = 1:size(A, 1)
			omega_target[r] = norm(Ainv_target[r, :])^2
		end
		
		AinvBerr = norm(AinvB - AinvB_target)/norm(AinvB_target)
		if(AinvBerr > epsilon)
			@warn "relative error in AinvB is "*string(AinvBerr)
		end

		showInfo(params_str, @test norm(gamma - gamma_target)/norm(gamma_target) < epsilon)
		
		omegaerr = norm(omega - omega_target)/norm(omega_target)
		if(omegaerr > epsilon)
			@warn "relative error in omegas is "*string(omegaerr)
		end
	end
end

@testset "test updateFactors!" begin
	m = 150
	n = 150
	k = 75
	epsilon = 1e-12
	
	Q_init = Matrix(qr(randn(m, m)).Q)
	R_init = qr(randn(m, n)).R
	perm_init = randperm(n)
	M = Matrix{Float64}(undef, m, n)
	M[:, perm_init] = Q_init*R_init
	
	A_init = R_init[1:k, 1:k]
	B_init = R_init[1:k, k + 1:n]
	C_init = R_init[k + 1:m, k + 1:n]
	
	Ainv_init = Matrix{Float64}(undef, k, k)
	for i = 1:k
		ei = zeros(k)
		ei[i] = 1.
		Ainv_init[:, i] = A_init \ ei
	end
	
	AinvB_init = Ainv_init*B_init
	
	gamma_init = Vector{Float64}(undef, n - k)
	for r = 1:n - k
		gamma_init[r] = norm(C_init[:, r])^2
	end
	
	omega_init = Vector{Float64}(undef, k)
	for r = 1:k
		omega_init[r] = norm(Ainv_init[r, :])^2
	end
	
	for i in [1, 3, k]
		for j in [1, 3, n - k]
			params_str = "parameters are: i = "*string(i)*", j = "*string(j)
			
			A = deepcopy(A_init)
			B = deepcopy(B_init)
			C = deepcopy(C_init)
			Q = deepcopy(Q_init)
			perm = deepcopy(perm_init)
			AinvB = deepcopy(AinvB_init)
			gamma = deepcopy(gamma_init)
			omega = deepcopy(omega_init)
			
			GuEisenstatQR.updateFactors!(i, j, Q, A, B, C, perm, AinvB, gamma, omega)
			R = [A B; zeros(m - k, k) C]
			
			showInfo(params_str, @test norm(M[:, perm] - Q*R)/norm(M) < epsilon)
			showInfo(params_str, @test norm(tril(A, -1)) < epsilon)
			showInfo(params_str, @test norm(Q'*Q - I(m))/m < epsilon)
			showInfo(params_str, @test perm[k + 1] == perm_init[i])
			showInfo(params_str, @test perm[k] == perm_init[k + j])
			
			Ainv_target = Matrix{Float64}(undef, k, k)
			for i = 1:k
				ei = zeros(k)
				ei[i] = 1.
				Ainv_target[:, i] = A \ ei
			end
			
			AinvB_target = Ainv_target*B
			
			gamma_target = Vector{Float64}(undef, n - k)
			for r = 1:n - k
				gamma_target[r] = norm(C[:, r])^2
			end
	
			omega_target = Vector{Float64}(undef, k)
			for r = 1:k
				omega_target[r] = norm(Ainv_target[r, :])^2
			end
				
			AinvBerr = norm(AinvB - AinvB_target)/norm(AinvB_target)
			if(AinvBerr > epsilon)
				@warn "relative error in AinvB is "*string(AinvBerr)
			end
			
			showInfo(params_str, @test norm(gamma - gamma_target)/norm(gamma_target) < epsilon)
			
			omegaerr = norm(omega - omega_target)/norm(omega_target)
			if(omegaerr > epsilon)
				@warn "relative error in omegas is "*string(omegaerr)
			end
		end
	end
end

@testset "srrqr tests (full rank)" begin
	m = 150
	n = 150
	epsilon = 1e-12
	tolparam = 1e-8
	
	M = randn(m, n)
	
	U, sigmaM, V = svd(M)
	
	maxnorm = 0.
	for i = 1:size(M, 2)
		maxnorm = max(maxnorm, norm(M[:, i])^2)
	end
	
	for fparam in [1.001, 1.1, 1.5, 2]
		params_str = "parameters are: f = "*string(fparam)
		
		k, perm, Q, R = srrqr(M, f = fparam, tol = tolparam)
		
		showInfo(params_str, @test k == min(m, n))
		showInfo(params_str, @test size(Q) == (m, m))
		showInfo(params_str, @test size(R) == (m, n))
		showInfo(params_str, @test length(perm) == n)
		showInfo(params_str, @test norm(M[:, perm] - Q*R)/norm(M) < epsilon)
		showInfo(params_str, @test norm(Q'*Q - I(m))/m < epsilon)
		showInfo(params_str, @test norm(tril(R[1:k, 1:k]), -1)/norm(R) < epsilon)
	end
	
	for fparam in [1.001, 1.1, 1.5, 2]
		params_str = "parameters are: f = "*string(fparam)
		
		kmaxparam = 20
		k, perm, Q, R = srrqr(M, f = fparam, tol = tolparam, kmax = kmaxparam)
		
		showInfo(params_str, @test k == kmaxparam)
		showInfo(params_str, @test size(Q) == (m, m))
		showInfo(params_str, @test size(R) == (m, n))
		showInfo(params_str, @test length(perm) == n)
		showInfo(params_str, @test norm(M[:, perm] - Q*R)/norm(M) < epsilon)
		showInfo(params_str, @test norm(Q'*Q - I(m))/m < epsilon)
		showInfo(params_str, @test norm(tril(R[1:k, 1:k]), -1)/norm(R) < epsilon)
		
		A = R[1:k, 1:k]
		C = R[k + 1:end, k + 1:end]
		
		sigmaA = svd(A).S
		sigmaC = svd(C).S
		q1 = sqrt(1 + fparam*k*(size(M, 2) - k))
		
		for i = 1:k
			params_str_highsigma = params_str*", i = "*string(i)
			showInfo(params_str_highsigma, @test sigmaA[i] >= sigmaM[i]/q1)
		end
		
		for j = 1:min(size(C, 1), size(C, 2))
			params_str_lowsigma = params_str*", j = "*string(j)
			showInfo(params_str_lowsigma, @test sigmaC[j] <= sigmaM[j + k]*q1)
		end
	end
end

@testset "srrqr tests (rank deficient)" begin
	m = 150
	n = 150
	epsilon = 1e-12
	tolparam = 1e-8
	k0 = 75
	
	M = randn(m, n)
	
	U, sigmaM, V = svd(M)
	sigmamin = 1e-8

	for i = k0 + 1:min(m, n)
		sigmaM[i] = sigmamin
	end
	
	M = U*diagm(sigmaM)*V'
	
	maxnorm = 0.
	for i = 1:size(M, 2)
		maxnorm = max(maxnorm, norm(M[:, i])^2)
	end
	
	for fparam in [1.001, 1.1, 1.5, 2]
		params_str = "parameters are: f = "*string(fparam)
		
		k, perm, Q, R = srrqr(M, f = fparam, tol = tolparam)
		
		showInfo(params_str, @test k == k0)
		showInfo(params_str, @test size(Q) == (m, m))
		showInfo(params_str, @test size(R) == (m, n))
		showInfo(params_str, @test length(perm) == n)
		showInfo(params_str, @test norm(M[:, perm] - Q*R)/norm(M) < epsilon)
		showInfo(params_str, @test norm(Q'*Q - I(m))/m < epsilon)
		showInfo(params_str, @test norm(tril(R[1:k, 1:k]), -1)/norm(R) < epsilon)
		
		A = R[1:k, 1:k]
		B = R[1:k, k + 1:end]
		C = R[k + 1:end, k + 1:end]
		
		showInfo(params_str, @test maximum(broadcast(abs, inv(A)*B)) <= fparam)
		
		Cmaxnorm = 0.
		for i = 1:size(C, 2)
			Cmaxnorm = max(Cmaxnorm, norm(C[:, i])^2)
		end
		
		showInfo(params_str, @test Cmaxnorm <= tolparam*maxnorm)
		
		sigmaA = svd(A).S
		sigmaC = svd(C).S
		q1 = sqrt(1 + fparam*k*(size(M, 2) - k))
		
		for i = 1:k
			params_str_highsigma = params_str*", i = "*string(i)
			showInfo(params_str_highsigma, @test sigmaA[i] >= sigmaM[i]/q1)
		end
		
		for j = 1:min(size(C, 1), size(C, 2))
			params_str_lowsigma = params_str*", j = "*string(j)
			showInfo(params_str_lowsigma, @test sigmaC[j] <= sigmaM[j + k]*q1)
		end
	end
end

@testset "srrqr tests (rank deficient, restricted pivoting)" begin
	m = 150
	n = 150
	epsilon = 1e-12
	tolparam = 1e-8
	k0 = 75
	
	M = randn(m, n)
	
	U, sigmaM, V = svd(M)
	sigmamin = 1e-8

	for i = k0 + 1:min(m, n)
		sigmaM[i] = sigmamin
	end
	
	M = U*diagm(sigmaM)*V'
	
	maxnorm = 0.
	for i = 1:size(M, 2)
		maxnorm = max(maxnorm, norm(M[:, i])^2)
	end
	
	for fparam in [1.001, 1.1, 1.5, 2]
		for kmaxparam in [20, 100]
			params_str = "parameters are: f = "*string(fparam)
			
			k, perm, Q, R = srrqr(M, f = fparam, tol = tolparam, kmax = kmaxparam)
			
			showInfo(params_str, @test k == min(k0, kmaxparam))
			showInfo(params_str, @test size(Q) == (m, m))
			showInfo(params_str, @test size(R) == (m, n))
			showInfo(params_str, @test length(perm) == n)
			showInfo(params_str, @test norm(M[:, perm] - Q*R)/norm(M) < epsilon)
			showInfo(params_str, @test norm(Q'*Q - I(m))/m < epsilon)
			showInfo(params_str, @test norm(tril(R[1:k, 1:k]), -1)/norm(R) < epsilon)
			
			A = R[1:k, 1:k]
			B = R[1:k, k + 1:end]
			C = R[k + 1:end, k + 1:end]
			
			showInfo(params_str, @test maximum(broadcast(abs, inv(A)*B)) <= fparam)
			
			sigmaA = svd(A).S
			sigmaC = svd(C).S
			q1 = sqrt(1 + fparam*k*(size(M, 2) - k))
			
			for i = 1:k
				params_str_highsigma = params_str*", i = "*string(i)
				showInfo(params_str_highsigma, @test sigmaA[i] >= sigmaM[i]/q1)
			end
			
			for j = 1:min(size(C, 1), size(C, 2))
				params_str_lowsigma = params_str*", j = "*string(j)
				showInfo(params_str_lowsigma, @test sigmaC[j] <= sigmaM[j + k]*q1)
			end
		end
	end
end
